"""Audit explicit whole-body latent coupling beyond all-belief baseline.

This is stricter than the all-model belief rerun.  The baseline already includes
all five personalized belief families.  The candidate adds only explicit
cross-system latent coupling features.  The placebo adds the same number of
random columns.

Passing means the whole-body coupling layer adds incremental factual signal
beyond simply concatenating per-system belief states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from eicu_all_model_belief_coupling_audit import (  # noqa: E402
    FULL_COHORTS,
    SEEDS,
    _resolve_path,
    attach_all_beliefs_cached,
    auto_targets,
    target_rows,
)
from eicu_body_system_target_router import _fit_ridge  # noqa: E402
from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _round, _subject_column, split_subjects  # noqa: E402
from whole_body_coupling_latent import whole_body_coupling_latent_features  # noqa: E402


def placebo_columns(frame: pd.DataFrame, count: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    if count <= 0:
        return frame.copy(), []
    columns = [f"placebo_wbc_{index}" for index in range(count)]
    values = np.random.default_rng(seed).normal(size=(len(frame), count))
    placebo = pd.DataFrame(values, index=frame.index, columns=columns, dtype=np.float64)
    return pd.concat([frame, placebo], axis=1), columns


def delta_by_subject(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> pd.Series:
    mask = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(baseline)
    if not mask.any():
        return pd.Series(dtype="float64")
    subject_col = _subject_column(rows)
    temp = rows.loc[mask, [subject_col]].copy()
    temp["delta"] = np.abs(pred[mask] - truth[mask]) - np.abs(baseline[mask] - truth[mask])
    return temp.groupby(subject_col)["delta"].mean()


def delta_summary(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray, seed: int) -> dict[str, object]:
    deltas = delta_by_subject(rows, truth, pred, baseline)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_baseline": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=200)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_baseline": bool(point < 0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def split_target_report(
    frame: pd.DataFrame,
    target: str,
    suffix: str,
    belief_columns: list[str],
    coupling_columns: list[str],
    seed: int,
    group_column: str | None = None,
    discovery_fraction: float = 0.67,
    active_only: bool = True,
) -> dict[str, object]:
    group_column = group_column or _subject_column(frame)
    discovery_groups, heldout_groups = split_subjects(
        frame,
        seed=seed,
        discovery_fraction=discovery_fraction,
        group_column=group_column,
    )
    discovery = frame[frame[group_column].isin(discovery_groups)].copy()
    heldout = frame[frame[group_column].isin(heldout_groups)].copy()
    train = target_rows(discovery, target, suffix)
    test = target_rows(heldout, target, suffix)
    active_column = next((column for column in test.columns if column.endswith("_active_t")), None)
    if active_only and active_column is not None:
        test = test[test[active_column].fillna(False).astype(bool)].copy()
    if train.empty or test.empty:
        return {"status": "insufficient_pairs", "rows": int(len(test)), "subjects": 0}

    excluded = set(belief_columns) | set(coupling_columns)
    ordinary_features = [column for column in _feature_columns(train, target) if column not in excluded]
    usable_beliefs = [
        column for column in belief_columns
        if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    usable_couplings = [
        column for column in coupling_columns
        if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    baseline_features = sorted(set(ordinary_features + usable_beliefs))
    candidate_features = sorted(set(baseline_features + usable_couplings))

    baseline = _fit_ridge(train, test, target, baseline_features, 10.0, suffix)
    candidate = _fit_ridge(train, test, target, candidate_features, 10.0, suffix)
    placebo_train, placebo = placebo_columns(train, len(usable_couplings), seed + 101)
    placebo_test, _ = placebo_columns(test, len(usable_couplings), seed + 202)
    placebo_pred = _fit_ridge(placebo_train, placebo_test, target, sorted(set(baseline_features + placebo)), 10.0, suffix)

    truth = test[f"{target}_{suffix}"].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "group_column": group_column,
        "rows": int(len(test)),
        "subjects": int(test[_subject_column(test)].nunique()) if len(test) else 0,
        "feature_counts": {
            "ordinary": int(len(ordinary_features)),
            "belief": int(len(usable_beliefs)),
            "coupling": int(len(usable_couplings)),
            "baseline": int(len(baseline_features)),
            "candidate": int(len(candidate_features)),
            "placebo": int(len(placebo)),
        },
        "candidate_vs_all_belief_baseline": delta_summary(test, truth, candidate, baseline, seed + 11),
        "candidate_vs_placebo": delta_summary(test, truth, candidate, placebo_pred, seed + 23),
    }


def summarize(target_reports: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [report for report in target_reports if report.get("status") == "evaluated"]
    both = [
        report for report in evaluated
        if report["candidate_vs_all_belief_baseline"]["significant"] and report["candidate_vs_placebo"]["significant"]
    ]
    base_deltas = [
        float(report["candidate_vs_all_belief_baseline"]["point_delta"])
        for report in evaluated
        if report["candidate_vs_all_belief_baseline"]["point_delta"] is not None
    ]
    placebo_deltas = [
        float(report["candidate_vs_placebo"]["point_delta"])
        for report in evaluated
        if report["candidate_vs_placebo"]["point_delta"] is not None
    ]
    return {
        "evaluated_splits": int(len(evaluated)),
        "pass_both_count": int(len(both)),
        "median_delta_vs_all_belief_baseline": _round(float(np.median(base_deltas))) if base_deltas else None,
        "median_delta_vs_placebo": _round(float(np.median(placebo_deltas))) if placebo_deltas else None,
    }


def audit_cohort(spec: dict[str, object], *, cache_dir: Path, rebuild_cache: bool = False) -> dict[str, object]:
    print(f"Loading {spec['name']}", flush=True)
    cohort_path = _resolve_path(str(spec["path"]))
    if not cohort_path.exists():
        return {"name": spec["name"], "status": "missing_cohort", "cohort": str(spec["path"]), "targets": {}}
    frame = pd.read_parquet(cohort_path).reset_index(drop=True)
    frame, belief_columns, cache_info = attach_all_beliefs_cached(
        frame,
        spec=spec,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    coupling = whole_body_coupling_latent_features(frame).reset_index(drop=True)
    coupling_columns = list(coupling.columns)
    frame = pd.concat([frame.reset_index(drop=True), coupling], axis=1)
    print(
        f"Attached beliefs={len(belief_columns)} coupling={len(coupling_columns)} "
        f"rows={len(frame)} cache_rebuilt={cache_info.get('rebuilt')}",
        flush=True,
    )

    requested = spec["targets"]
    targets = auto_targets(frame, str(spec["future_suffix"])) if requested == "auto" else tuple(requested)
    reports = {}
    for target in targets:
        split_reports = [
            split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                coupling_columns,
                seed,
            )
            for seed in SEEDS
        ]
        holdout = None
        if "hospitalid" in frame and frame["hospitalid"].nunique(dropna=True) >= 2:
            holdout = split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                coupling_columns,
                9901,
                group_column="hospitalid",
            )
        elif spec.get("holdout_group_column") in frame and frame[str(spec["holdout_group_column"])].nunique(dropna=True) >= 2:
            holdout = split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                coupling_columns,
                9901,
                group_column=str(spec["holdout_group_column"]),
            )
        reports[str(target)] = {
            "patient_splits": summarize(split_reports),
            "holdout": holdout,
        }
        print(spec["name"], target, reports[str(target)]["patient_splits"], flush=True)

    return {
        "name": spec["name"],
        "cohort": str(spec["path"]),
        "future_suffix": str(spec["future_suffix"]),
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "holdout_group_column": "hospitalid" if "hospitalid" in frame else spec.get("holdout_group_column"),
        "belief_feature_count": int(len(belief_columns)),
        "coupling_feature_count": int(len(coupling_columns)),
        "cache": cache_info,
        "targets": reports,
    }


def is_validated(report: dict[str, object]) -> bool:
    patient = report["patient_splits"]
    holdout = report.get("holdout") or {}
    return bool(
        patient["pass_both_count"] == len(SEEDS)
        and holdout.get("status") == "evaluated"
        and holdout.get("candidate_vs_all_belief_baseline", {}).get("significant")
        and holdout.get("candidate_vs_placebo", {}).get("significant")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="whole_body_coupling_latent_audit.json")
    parser.add_argument("--cohorts", default="cardiovascular_full_6h,sepsis_6h_full,aki_24h_full,aki_48h_full,mimiciv_observation_6h")
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/osler_belief_cache_full"))
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {item.strip() for item in args.cohorts.split(",") if item.strip()}
    specs = [spec for spec in FULL_COHORTS if str(spec["name"]) in requested]
    cohorts = [audit_cohort(spec, cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache) for spec in specs]
    validated = []
    for cohort in cohorts:
        if cohort.get("status") == "missing_cohort":
            continue
        for target, report in cohort["targets"].items():
            if is_validated(report):
                patient = report["patient_splits"]
                validated.append({
                    "cohort": cohort["name"],
                    "target": target,
                    "median_delta_vs_all_belief_baseline": patient["median_delta_vs_all_belief_baseline"],
                    "median_delta_vs_placebo": patient["median_delta_vs_placebo"],
                })

    output = {
        "artifact": "explicit whole-body latent coupling audit",
        "definition": "all-belief baseline vs all-belief plus explicit cross-system latent coupling features vs capacity-matched placebo",
        "validated": validated,
        "cohorts": cohorts,
        "boundary": {
            "factual_observed_inputs_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_authority": False,
            "row_level_outputs_written": False,
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validated": validated, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
