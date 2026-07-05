"""Audit whether observed MIMIC treatment context improves factual forecasts.

This is not a causal treatment-effect audit.  It compares three factual
forecasting feature sets on the same observed MIMIC-IV transition cohort:

* baseline: current physiology, masks/ages, and non-treatment context;
* candidate: baseline plus observed ``hist_*`` and ``act_*`` treatment context;
* placebo: baseline plus the same number of random-noise columns.

The candidate must beat both baseline and placebo under patient-heldout splits
and careunit/time heldout checks before any target is treated as validated.
Reports are aggregate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_full_variable_coverage_audit import (
    forecast_feature_columns,
    forecast_rows,
    prepare_features,
)
from eicu_nowcasting_audit import DEFAULT_SEEDS
from eicu_sepsis_target_router import _bootstrap_ci, split_subjects
from mimiciv_cross_database_coverage_audit import add_time_holdout_column


DEFAULT_TARGETS = ("glucose", "potassium", "map", "bicarbonate")


def _round(value, digits: int = 6):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def feature_sets(frame: pd.DataFrame, target: str) -> dict[str, list[str]]:
    candidate = forecast_feature_columns(frame, target)
    treatment = [
        column for column in candidate
        if column.startswith("hist_") or column.startswith("act_")
    ]
    baseline = [column for column in candidate if column not in treatment]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "treatment": treatment,
    }


def _numeric_current(rows: pd.DataFrame, target: str) -> np.ndarray:
    return pd.to_numeric(rows[f"{target}_t"], errors="coerce").to_numpy(dtype=np.float64)


def fit_ridge_residual(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target: str,
    features: list[str],
    alpha: float,
    seed: int,
    noise_count: int = 0,
) -> np.ndarray:
    if train.empty or len(train) < max(20, len(features) + int(noise_count) + 2):
        return np.full(len(predict), np.nan, dtype=np.float64)
    x_train, x_predict = prepare_features(train, predict, features)
    if noise_count > 0:
        rng = np.random.default_rng(int(seed))
        x_train = np.c_[x_train, rng.normal(size=(len(train), int(noise_count)))]
        x_predict = np.c_[x_predict, rng.normal(size=(len(predict), int(noise_count)))]
    current = _numeric_current(train, target)
    y = train["truth"].to_numpy(dtype=np.float64) - current
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_predict = np.c_[np.ones(len(x_predict)), x_predict]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return _numeric_current(predict, target) + x_predict @ beta


def clustered_delta(rows: pd.DataFrame, candidate: np.ndarray, comparator: np.ndarray) -> pd.Series:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(comparator)
    if not finite.any():
        return pd.Series(dtype="float64")
    temp = pd.DataFrame({
        "subject_id": rows.loc[finite, "subject_id"].to_numpy(dtype=object),
        "delta": np.abs(candidate[finite] - truth[finite]) - np.abs(comparator[finite] - truth[finite]),
    })
    return temp.groupby("subject_id")["delta"].mean()


def delta_summary(
    rows: pd.DataFrame,
    candidate: np.ndarray,
    comparator: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    deltas = clustered_delta(rows, candidate, comparator)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=bootstrap_samples)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def mae(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(truth)
    if not finite.any():
        return None
    return float(np.abs(prediction[finite] - truth[finite]).mean())


def evaluate_predictions(
    rows: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    placebo: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(rows)),
        "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
        "mae": {
            "baseline_no_treatment_features": _round(mae(baseline, truth)),
            "candidate_with_treatment_features": _round(mae(candidate, truth)),
            "placebo_equal_random_features": _round(mae(placebo, truth)),
        },
        "candidate_vs_baseline": delta_summary(
            rows,
            candidate,
            baseline,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "candidate_vs_placebo": delta_summary(
            rows,
            candidate,
            placebo,
            seed=seed + 137,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def oof_predictions(
    discovery: pd.DataFrame,
    target: str,
    baseline_features: list[str],
    candidate_features: list[str],
    treatment_feature_count: int,
    seed: int,
    alpha: float,
    folds: int,
) -> dict[str, np.ndarray]:
    from eicu_sepsis_target_router import _group_folds

    baseline = np.full(len(discovery), np.nan, dtype=np.float64)
    candidate = np.full(len(discovery), np.nan, dtype=np.float64)
    placebo = np.full(len(discovery), np.nan, dtype=np.float64)
    split = _group_folds(discovery["subject_id"].to_numpy(dtype=object), seed=seed + 17, folds=folds)
    for fold_index, (train_idx, val_idx) in enumerate(split):
        train = discovery.iloc[train_idx]
        val = discovery.iloc[val_idx]
        baseline[val_idx] = fit_ridge_residual(
            train,
            val,
            target,
            baseline_features,
            alpha,
            seed=seed + fold_index,
        )
        candidate[val_idx] = fit_ridge_residual(
            train,
            val,
            target,
            candidate_features,
            alpha,
            seed=seed + 1_000 + fold_index,
        )
        placebo[val_idx] = fit_ridge_residual(
            train,
            val,
            target,
            baseline_features,
            alpha,
            seed=seed + 2_000 + fold_index,
            noise_count=treatment_feature_count,
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "placebo": placebo,
    }


def fit_heldout_predictions(
    discovery: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    baseline_features: list[str],
    candidate_features: list[str],
    treatment_feature_count: int,
    seed: int,
    alpha: float,
) -> dict[str, np.ndarray]:
    return {
        "baseline": fit_ridge_residual(
            discovery,
            heldout,
            target,
            baseline_features,
            alpha,
            seed=seed + 10_000,
        ),
        "candidate": fit_ridge_residual(
            discovery,
            heldout,
            target,
            candidate_features,
            alpha,
            seed=seed + 20_000,
        ),
        "placebo": fit_ridge_residual(
            discovery,
            heldout,
            target,
            baseline_features,
            alpha,
            seed=seed + 30_000,
            noise_count=treatment_feature_count,
        ),
    }


def audit_split(
    rows: pd.DataFrame,
    target: str,
    baseline_features: list[str],
    candidate_features: list[str],
    treatment_feature_count: int,
    discovery_groups: set[object],
    heldout_groups: set[object],
    group_column: str,
    seed: int,
    alpha: float,
    inner_folds: int,
    bootstrap_samples: int,
    min_pairs: int,
    min_subjects: int,
) -> dict[str, object]:
    discovery = rows[rows[group_column].isin(discovery_groups)].copy()
    heldout = rows[rows[group_column].isin(heldout_groups)].copy()
    if (
        len(discovery) < int(min_pairs)
        or discovery["subject_id"].nunique() < int(min_subjects)
        or len(heldout) == 0
        or treatment_feature_count == 0
    ):
        return {
            "status": "insufficient_support",
            "discovery_rows": int(len(discovery)),
            "discovery_subjects": int(discovery["subject_id"].nunique()) if len(discovery) else 0,
            "heldout_rows": int(len(heldout)),
            "treatment_feature_count": int(treatment_feature_count),
        }
    oof = oof_predictions(
        discovery,
        target,
        baseline_features,
        candidate_features,
        treatment_feature_count,
        seed,
        alpha,
        inner_folds,
    )
    discovery_eval = evaluate_predictions(
        discovery,
        oof["baseline"],
        oof["candidate"],
        oof["placebo"],
        seed=seed + 40_000,
        bootstrap_samples=bootstrap_samples,
    )
    selected = bool(
        discovery_eval["candidate_vs_baseline"]["significant"]
        and discovery_eval["candidate_vs_placebo"]["significant"]
    )
    heldout_pred = fit_heldout_predictions(
        discovery,
        heldout,
        target,
        baseline_features,
        candidate_features,
        treatment_feature_count,
        seed,
        alpha,
    )
    heldout_eval = evaluate_predictions(
        heldout,
        heldout_pred["baseline"],
        heldout_pred["candidate"],
        heldout_pred["placebo"],
        seed=seed + 50_000,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "status": "evaluated",
        "selected_candidate": selected,
        "discovery": discovery_eval,
        "heldout": heldout_eval,
    }


def split_passed(report: dict[str, object]) -> bool:
    if report.get("status") != "evaluated":
        return False
    heldout = report.get("heldout") or {}
    return bool(
        report.get("selected_candidate")
        and (heldout.get("candidate_vs_baseline") or {}).get("significant")
        and (heldout.get("candidate_vs_placebo") or {}).get("significant")
    )


def summarize_random(reports: list[dict[str, object]]) -> dict[str, int]:
    evaluated = sum(1 for report in reports if report.get("status") == "evaluated")
    selected = sum(1 for report in reports if report.get("selected_candidate"))
    heldout_base = sum(
        1 for report in reports
        if (report.get("heldout") or {}).get("candidate_vs_baseline", {}).get("significant")
    )
    heldout_placebo = sum(
        1 for report in reports
        if (report.get("heldout") or {}).get("candidate_vs_placebo", {}).get("significant")
    )
    return {
        "evaluated_splits": int(evaluated),
        "selected_candidate_splits": int(selected),
        "heldout_beats_baseline_splits": int(heldout_base),
        "heldout_beats_placebo_splits": int(heldout_placebo),
    }


def external_split(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
    baseline_features: list[str],
    candidate_features: list[str],
    treatment_feature_count: int,
    group_column: str,
    discovery_fraction: float,
    seed: int,
    alpha: float,
    inner_folds: int,
    bootstrap_samples: int,
    min_pairs: int,
    min_subjects: int,
) -> dict[str, object]:
    if group_column not in rows or rows[group_column].nunique(dropna=True) < 2:
        return {"available": False}
    if group_column == "time_holdout_group":
        discovery_groups = {"early"}
        heldout_groups = {"late"}
    else:
        discovery_groups, heldout_groups = split_subjects(
            frame,
            seed,
            discovery_fraction,
            group_column=group_column,
        )
    return {
        "available": True,
        **audit_split(
            rows,
            target,
            baseline_features,
            candidate_features,
            treatment_feature_count,
            discovery_groups,
            heldout_groups,
            group_column,
            seed,
            alpha,
            inner_folds,
            bootstrap_samples,
            min_pairs,
            min_subjects,
        ),
    }


def audit_target(
    frame: pd.DataFrame,
    target: str,
    future_suffix: str,
    seeds: tuple[int, ...],
    discovery_fraction: float,
    alpha: float,
    inner_folds: int,
    bootstrap_samples: int,
    min_pairs: int,
    min_subjects: int,
    max_rows_per_target: int | None,
) -> dict[str, object]:
    rows = forecast_rows(frame, target, future_suffix)
    if max_rows_per_target is not None and len(rows) > int(max_rows_per_target):
        rng = np.random.default_rng(stable_seed("treatment-context", target))
        subjects = rows["subject_id"].drop_duplicates().to_numpy(dtype=object)
        rng.shuffle(subjects)
        keep = []
        total = 0
        counts = rows.groupby("subject_id").size()
        for subject in subjects:
            keep.append(subject)
            total += int(counts.loc[subject])
            if total >= int(max_rows_per_target):
                break
        rows = rows[rows["subject_id"].isin(set(keep))].copy()
    sets = feature_sets(frame, target)
    treatment_feature_count = len(sets["treatment"])
    random_reports = []
    for seed in seeds:
        discovery_groups, heldout_groups = split_subjects(frame, seed, discovery_fraction)
        random_reports.append(audit_split(
            rows,
            target,
            sets["baseline"],
            sets["candidate"],
            treatment_feature_count,
            discovery_groups,
            heldout_groups,
            "subject_id",
            seed,
            alpha,
            inner_folds,
            bootstrap_samples,
            min_pairs,
            min_subjects,
        ))
    careunit = external_split(
        frame,
        rows,
        target,
        sets["baseline"],
        sets["candidate"],
        treatment_feature_count,
        "first_careunit",
        discovery_fraction,
        9001,
        alpha,
        inner_folds,
        bootstrap_samples,
        min_pairs,
        min_subjects,
    )
    time = external_split(
        frame,
        rows,
        target,
        sets["baseline"],
        sets["candidate"],
        treatment_feature_count,
        "time_holdout_group",
        discovery_fraction,
        9901,
        alpha,
        inner_folds,
        bootstrap_samples,
        min_pairs,
        min_subjects,
    )
    summary = summarize_random(random_reports)
    split_count = len(seeds)
    validated = bool(
        summary["evaluated_splits"] == split_count
        and summary["selected_candidate_splits"] == split_count
        and summary["heldout_beats_baseline_splits"] == split_count
        and summary["heldout_beats_placebo_splits"] == split_count
        and split_passed(careunit)
        and split_passed(time)
    )
    return {
        "support": {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
            "baseline_feature_count": int(len(sets["baseline"])),
            "candidate_feature_count": int(len(sets["candidate"])),
            "treatment_feature_count": int(treatment_feature_count),
            "treatment_features": sets["treatment"],
        },
        "random_patient_splits": summary,
        "careunit_holdout": {
            "available": bool(careunit.get("available")),
            "gate_passed": split_passed(careunit),
            "heldout": careunit.get("heldout") if careunit.get("available") else None,
        },
        "time_holdout": {
            "available": bool(time.get("available")),
            "gate_passed": split_passed(time),
            "heldout": time.get("heldout") if time.get("available") else None,
        },
        "validated": validated,
    }


def run_audit(
    cohort: Path,
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    discovery_fraction: float,
    alpha: float,
    inner_folds: int,
    bootstrap_samples: int,
    future_suffix: str,
    min_pairs: int,
    min_subjects: int,
    max_rows_per_target: int | None,
) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    frame = add_time_holdout_column(frame, discovery_fraction)
    target_reports = {}
    for target in targets:
        print(f"Auditing treatment-context target: {target}", flush=True)
        target_reports[target] = audit_target(
            frame,
            target,
            future_suffix,
            seeds,
            discovery_fraction,
            alpha,
            inner_folds,
            bootstrap_samples,
            min_pairs,
            min_subjects,
            max_rows_per_target,
        )
    validated = [target for target, report in target_reports.items() if report["validated"]]
    return {
        "artifact": "MIMIC-IV observed treatment context factual forecast accuracy audit",
        "cohort": str(cohort),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "targets": target_reports,
        "validated_targets": sorted(validated),
        "counts": {
            "targets_evaluated": int(len(target_reports)),
            "targets_validated": int(len(validated)),
        },
        "gate": {
            "baseline": "ridge residual forecast without hist_*/act_* treatment features",
            "candidate": "same forecast plus observed factual hist_*/act_* treatment features",
            "placebo": "baseline plus equal number of random-noise features",
            "patient_heldout_splits": list(seeds),
            "careunit_heldout_required": True,
            "time_heldout_required": True,
            "candidate_must_beat": ["baseline", "placebo"],
        },
        "safety_boundary": {
            "factual_observed_treatment_context_only": True,
            "causal_claim_allowed": False,
            "counterfactual_treatment_effect_allowed": False,
            "clinical_claim_allowed": False,
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
        },
    }


def write_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# MIMIC-IV Observed Treatment Context Accuracy Findings",
        "",
        "This audit asks whether observed treatment context improves factual forecasts. It does not estimate treatment effects.",
        "",
        "## Gate",
        "",
        "- Baseline: ridge forecast without `hist_*` / `act_*` treatment features.",
        "- Candidate: same forecast plus observed factual treatment features.",
        "- Placebo: baseline plus an equal number of random-noise features.",
        "- Pass rule: candidate must beat both baseline and placebo across 7 patient-heldout splits, careunit-heldout, and time-heldout.",
        "",
        "## Result",
        "",
        f"- Cohort rows: `{report['rows']}`",
        f"- Subjects: `{report['subjects']}`",
        f"- Validated targets: `{', '.join(report['validated_targets']) if report['validated_targets'] else 'none'}`",
        "",
        "| Target | Rows | Subjects | Treatment Features | 7-Split Selected | 7-Split vs Baseline | 7-Split vs Placebo | Careunit | Time | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for target, item in report["targets"].items():
        support = item["support"]
        random = item["random_patient_splits"]
        status = "validated" if item["validated"] else "fallback"
        lines.append(
            f"| {target} | {support['rows']} | {support['subjects']} | "
            f"{support['treatment_feature_count']} | "
            f"{random['selected_candidate_splits']}/7 | "
            f"{random['heldout_beats_baseline_splits']}/7 | "
            f"{random['heldout_beats_placebo_splits']}/7 | "
            f"{'pass' if item['careunit_holdout']['gate_passed'] else 'fail'} | "
            f"{'pass' if item['time_holdout']['gate_passed'] else 'fail'} | {status} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Observed treatment context is a factual input. This audit does not allow causal, counterfactual, clinical, or treatment-recommendation claims.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--output", type=Path, default=Path("mimiciv_treatment_context_accuracy_audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("MIMICIV_TREATMENT_CONTEXT_ACCURACY_FINDINGS.md"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--future-suffix", default="tp6")
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--max-rows-per-target", type=int, default=80_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    targets = tuple(target.strip() for target in args.targets.split(",") if target.strip())
    max_rows = None if args.max_rows_per_target <= 0 else int(args.max_rows_per_target)
    report = run_audit(
        args.cohort,
        targets,
        seeds,
        args.discovery_fraction,
        args.ridge_alpha,
        args.inner_folds,
        args.bootstrap_samples,
        args.future_suffix,
        args.min_pairs,
        args.min_subjects,
        max_rows,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown:
        write_markdown(report, args.markdown)
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown) if args.markdown else None,
        "validated_targets": report["validated_targets"],
        "counts": report["counts"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
