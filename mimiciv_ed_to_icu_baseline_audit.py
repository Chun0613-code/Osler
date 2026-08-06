"""Audit whether ED trajectory features improve early ICU factual prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_full_variable_coverage_audit import (
    delta_summary,
    eligible_forecast_targets,
    forecast_rows,
    hospital_gate_pass,
    interval_summary,
    mae,
)
from eicu_nowcasting_audit import DEFAULT_SEEDS, is_future_column, target_column
from eicu_sepsis_target_router import _group_folds, split_subjects
from mimiciv_cross_database_coverage_audit import add_time_holdout_column, capped_subject_rows


DEFAULT_TARGETS = ("heart_rate", "map", "o2sat", "respiratory_rate", "temperature")


ID_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "ed_stay_id",
    "first_careunit",
    "last_careunit",
    "arrival_transport",
    "disposition",
    "intime",
    "outtime",
    "ed_intime",
    "ed_outtime",
    "t",
    "t_plus",
}


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def base_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {target_column(target), f"{target}_age_hr"}
    columns = []
    for column in frame.columns:
        if column in ID_COLUMNS or column in excluded:
            continue
        if column.startswith("ed_"):
            continue
        if is_future_column(column):
            continue
        if column.endswith("_sources") or column.endswith("_kinds"):
            continue
        if (
            column.endswith("_t")
            or column.endswith("_age_hr")
            or column in {"hours_since_onset"}
        ):
            values = numeric_series(frame, column)
            if values.notna().any():
                columns.append(column)
    return sorted(set(columns))


def ed_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if not column.startswith("ed_"):
            continue
        if column in {"ed_stay_id", "ed_intime", "ed_outtime"}:
            continue
        values = numeric_series(frame, column)
        if values.notna().any():
            columns.append(column)
    return sorted(set(columns))


def prepare_features(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    columns: list[str],
    placebo_ed: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.zeros((len(train), 0), dtype=np.float64), np.zeros((len(predict), 0), dtype=np.float64)
    train_x = train[columns].apply(pd.to_numeric, errors="coerce")
    predict_x = predict[columns].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    predict_x = predict_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    if placebo_ed:
        rng = np.random.default_rng(int(seed))
        for column in columns:
            if column.startswith("ed_"):
                train_x[column] = rng.permutation(train_x[column].to_numpy(dtype=np.float64))
                predict_x[column] = rng.permutation(predict_x[column].to_numpy(dtype=np.float64))
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return (
        ((train_x - mean) / std).to_numpy(dtype=np.float64),
        ((predict_x - mean) / std).to_numpy(dtype=np.float64),
    )


def fit_ridge_delta(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target: str,
    features: list[str],
    alpha: float,
    placebo_ed: bool,
    seed: int,
) -> np.ndarray:
    if train.empty or len(train) < max(20, len(features) + 2):
        return np.full(len(predict), np.nan, dtype=np.float64)
    x_train, x_predict = prepare_features(train, predict, features, placebo_ed=placebo_ed, seed=seed)
    y = train["truth"].to_numpy(dtype=np.float64) - numeric_series(train, target_column(target)).to_numpy(dtype=np.float64)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_predict = np.c_[np.ones(len(x_predict)), x_predict]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return numeric_series(predict, target_column(target)).to_numpy(dtype=np.float64) + x_predict @ beta


def eval_against(rows: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    return {
        "baseline_mae": round(float(mae(baseline, truth)), 6) if mae(baseline, truth) is not None else None,
        "candidate_mae": round(float(mae(candidate, truth)), 6) if mae(candidate, truth) is not None else None,
        "delta": delta_summary(rows, candidate, baseline, seed=seed, samples=samples),
    }


def oof_predictions(
    discovery: pd.DataFrame,
    target: str,
    icu_features: list[str],
    ed_plus_features: list[str],
    seed: int,
    inner_folds: int,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    persistence = np.full(len(discovery), np.nan, dtype=np.float64)
    icu_only = np.full(len(discovery), np.nan, dtype=np.float64)
    ed_plus = np.full(len(discovery), np.nan, dtype=np.float64)
    ed_placebo = np.full(len(discovery), np.nan, dtype=np.float64)
    folds = _group_folds(discovery["subject_id"].to_numpy(dtype=object), seed=seed + 17, folds=inner_folds)
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        train = discovery.iloc[train_idx]
        val = discovery.iloc[val_idx]
        persistence[val_idx] = numeric_series(val, target_column(target)).to_numpy(dtype=np.float64)
        icu_only[val_idx] = fit_ridge_delta(train, val, target, icu_features, ridge_alpha, False, seed + 101 * fold_index)
        ed_plus[val_idx] = fit_ridge_delta(train, val, target, ed_plus_features, ridge_alpha, False, seed + 201 * fold_index)
        ed_placebo[val_idx] = fit_ridge_delta(train, val, target, ed_plus_features, ridge_alpha, True, seed + 301 * fold_index)
    return {
        "persistence": persistence,
        "icu_only": icu_only,
        "ed_plus": ed_plus,
        "ed_placebo": ed_placebo,
    }


def fit_predictions(
    discovery: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    icu_features: list[str],
    ed_plus_features: list[str],
    seed: int,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    return {
        "persistence": numeric_series(heldout, target_column(target)).to_numpy(dtype=np.float64),
        "icu_only": fit_ridge_delta(discovery, heldout, target, icu_features, ridge_alpha, False, seed + 1000),
        "ed_plus": fit_ridge_delta(discovery, heldout, target, ed_plus_features, ridge_alpha, False, seed + 2000),
        "ed_placebo": fit_ridge_delta(discovery, heldout, target, ed_plus_features, ridge_alpha, True, seed + 3000),
    }


def audit_split(
    rows: pd.DataFrame,
    target: str,
    icu_features: list[str],
    ed_plus_features: list[str],
    discovery_groups: set[object],
    heldout_groups: set[object],
    group_column: str,
    seed: int,
    min_pairs: int,
    min_subjects: int,
    inner_folds: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    conformal_level: float,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, object]:
    discovery = rows[rows[group_column].isin(discovery_groups)].copy()
    heldout = rows[rows[group_column].isin(heldout_groups)].copy()
    supported = (
        len(discovery) >= int(min_pairs)
        and discovery["subject_id"].nunique() >= int(min_subjects)
        and len(icu_features) > 0
        and len(ed_plus_features) > len(icu_features)
    )
    if not supported:
        return {
            "status": "insufficient_support",
            "discovery_rows": int(len(discovery)),
            "discovery_subjects": int(discovery["subject_id"].nunique()) if len(discovery) else 0,
            "icu_feature_count": int(len(icu_features)),
            "ed_plus_feature_count": int(len(ed_plus_features)),
        }
    oof = oof_predictions(discovery, target, icu_features, ed_plus_features, seed, inner_folds, ridge_alpha)
    discovery_eval = {
        "vs_persistence": eval_against(discovery, oof["ed_plus"], oof["persistence"], seed + 10_000, bootstrap_samples),
        "vs_icu_only": eval_against(discovery, oof["ed_plus"], oof["icu_only"], seed + 11_000, bootstrap_samples),
        "vs_ed_placebo": eval_against(discovery, oof["ed_plus"], oof["ed_placebo"], seed + 12_000, bootstrap_samples),
    }
    selected = bool(
        discovery_eval["vs_persistence"]["delta"]["significant"]
        and discovery_eval["vs_icu_only"]["delta"]["significant"]
        and discovery_eval["vs_ed_placebo"]["delta"]["significant"]
    )
    heldout_pred = fit_predictions(discovery, heldout, target, icu_features, ed_plus_features, seed, ridge_alpha)
    heldout_eval = {
        "vs_persistence": eval_against(heldout, heldout_pred["ed_plus"], heldout_pred["persistence"], seed + 20_000, bootstrap_samples),
        "vs_icu_only": eval_against(heldout, heldout_pred["ed_plus"], heldout_pred["icu_only"], seed + 21_000, bootstrap_samples),
        "vs_ed_placebo": eval_against(heldout, heldout_pred["ed_plus"], heldout_pred["ed_placebo"], seed + 22_000, bootstrap_samples),
    }
    interval = None
    if selected:
        interval = interval_summary(
            discovery,
            oof["ed_plus"],
            heldout,
            heldout_pred["ed_plus"],
            level=conformal_level,
            min_calibration_rows=min_calibration_rows,
            min_test_rows=min_test_rows,
        )
    return {
        "status": "evaluated",
        "icu_feature_count": int(len(icu_features)),
        "ed_plus_feature_count": int(len(ed_plus_features)),
        "selected_ed_plus": selected,
        "discovery": discovery_eval,
        "heldout": heldout_eval,
        "interval": interval,
    }


def split_pass(report: dict[str, object], require_interval: bool = False) -> bool:
    if report.get("status") != "evaluated":
        return False
    heldout = report.get("heldout") or {}
    passed = bool(
        report.get("selected_ed_plus")
        and heldout["vs_persistence"]["delta"]["significant"]
        and heldout["vs_icu_only"]["delta"]["significant"]
        and heldout["vs_ed_placebo"]["delta"]["significant"]
    )
    if require_interval:
        passed = passed and bool((report.get("interval") or {}).get("coverage_gate_passed"))
    return passed


def summarize(reports: list[dict[str, object]]) -> dict[str, int]:
    evaluated = 0
    selected = 0
    beats_persistence = 0
    beats_icu = 0
    beats_placebo = 0
    interval = 0
    for report in reports:
        if report.get("status") != "evaluated":
            continue
        evaluated += 1
        selected += int(bool(report.get("selected_ed_plus")))
        heldout = report.get("heldout") or {}
        beats_persistence += int(bool(heldout["vs_persistence"]["delta"]["significant"]))
        beats_icu += int(bool(heldout["vs_icu_only"]["delta"]["significant"]))
        beats_placebo += int(bool(heldout["vs_ed_placebo"]["delta"]["significant"]))
        interval += int(bool((report.get("interval") or {}).get("coverage_gate_passed")))
    return {
        "evaluated_splits": int(evaluated),
        "selected_ed_plus_splits": int(selected),
        "heldout_beats_persistence_splits": int(beats_persistence),
        "heldout_beats_icu_only_splits": int(beats_icu),
        "heldout_beats_ed_placebo_splits": int(beats_placebo),
        "interval_pass_splits": int(interval),
    }


def external_report(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
    icu_features: list[str],
    ed_plus_features: list[str],
    group_column: str,
    discovery_fraction: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    if group_column not in frame or frame[group_column].nunique(dropna=True) < 2:
        return {"available": False}
    if group_column == "time_holdout_group":
        discovery_groups = {"early"}
        heldout_groups = {"late"}
    else:
        discovery_groups, heldout_groups = split_subjects(frame, seed, discovery_fraction, group_column=group_column)
    return {
        "available": True,
        **audit_split(
            rows,
            target,
            icu_features,
            ed_plus_features,
            set(discovery_groups),
            set(heldout_groups),
            group_column,
            seed,
            args.min_pairs,
            args.min_subjects,
            args.inner_folds,
            args.ridge_alpha,
            args.bootstrap_samples,
            args.conformal_level,
            args.min_calibration_rows,
            args.min_test_rows,
        ),
    }


def audit_target(
    frame: pd.DataFrame,
    target: str,
    rows: pd.DataFrame,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, object]:
    icu_features = base_feature_columns(frame, target)
    ed_features = ed_feature_columns(frame)
    ed_plus_features = sorted(set(icu_features) | set(ed_features))
    random_reports = []
    for seed in seeds:
        discovery_groups, heldout_groups = split_subjects(frame, seed, args.discovery_fraction)
        random_reports.append(audit_split(
            rows,
            target,
            icu_features,
            ed_plus_features,
            discovery_groups,
            heldout_groups,
            "subject_id",
            seed,
            args.min_pairs,
            args.min_subjects,
            args.inner_folds,
            args.ridge_alpha,
            args.bootstrap_samples,
            args.conformal_level,
            args.min_calibration_rows,
            args.min_test_rows,
        ))
    careunit = external_report(frame, rows, target, icu_features, ed_plus_features, "first_careunit", args.discovery_fraction, 8101, args)
    time = external_report(frame, rows, target, icu_features, ed_plus_features, "time_holdout_group", args.discovery_fraction, 8201, args)
    summary = summarize(random_reports)
    split_count = len(seeds)
    validated = bool(
        summary["evaluated_splits"] == split_count
        and summary["selected_ed_plus_splits"] == split_count
        and summary["heldout_beats_persistence_splits"] == split_count
        and summary["heldout_beats_icu_only_splits"] == split_count
        and summary["heldout_beats_ed_placebo_splits"] == split_count
        and split_pass(careunit)
        and split_pass(time)
    )
    interval_validated = bool(
        validated
        and summary["interval_pass_splits"] == split_count
        and split_pass(careunit, require_interval=True)
        and split_pass(time, require_interval=True)
    )
    return {
        "support": {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
            "icu_feature_count": int(len(icu_features)),
            "ed_feature_count": int(len(ed_features)),
            "ed_plus_feature_count": int(len(ed_plus_features)),
        },
        "random_patient_splits": summary,
        "careunit_holdout": {
            "available": bool(careunit.get("available")),
            "gate_passed": split_pass(careunit),
            "interval_gate_passed": split_pass(careunit, require_interval=True),
            "heldout": careunit.get("heldout") if careunit.get("available") else None,
            "interval": careunit.get("interval") if careunit.get("available") else None,
        },
        "time_holdout": {
            "available": bool(time.get("available")),
            "gate_passed": split_pass(time),
            "interval_gate_passed": split_pass(time, require_interval=True),
            "heldout": time.get("heldout") if time.get("available") else None,
            "interval": time.get("interval") if time.get("available") else None,
        },
        "validated_ed_increment": validated,
        "interval_validated": interval_validated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-suffix", default="tp1")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--conformal-level", type=float, default=0.9)
    parser.add_argument("--min-calibration-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--max-rows-per-target", type=int, default=80_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    requested = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    frame = add_time_holdout_column(frame, args.discovery_fraction)
    eligible = set(eligible_forecast_targets(frame, args.future_suffix))
    targets = tuple(target for target in requested if target in eligible)
    max_rows = None if args.max_rows_per_target <= 0 else int(args.max_rows_per_target)
    target_reports = {}
    for target in targets:
        print(f"Auditing ED-to-ICU baseline target: {target}", flush=True)
        rows, sampling = capped_subject_rows(
            forecast_rows(frame, target, args.future_suffix),
            max_rows,
            seed=stable_seed("ed-to-icu", args.future_suffix, target),
        )
        report = audit_target(frame, target, rows, seeds, args)
        report["sampling"] = sampling
        target_reports[target] = report
    validated = sorted(target for target, report in target_reports.items() if report.get("validated_ed_increment"))
    intervals = sorted(target for target, report in target_reports.items() if report.get("interval_validated"))
    output = {
        "artifact": "MIMIC-IV-ED to early-ICU baseline incremental audit",
        "cohort": "local-only ED-to-ICU transition cohort",
        "future_suffix": args.future_suffix,
        "targets_requested": list(requested),
        "targets_evaluated": list(targets),
        "patient_split_seeds": list(seeds),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "icu_stays": int(frame["stay_id"].nunique()),
        "ed_stays": int(frame["ed_stay_id"].nunique()),
        "validated": {
            "ed_increment_targets": validated,
            "interval_targets": intervals,
        },
        "counts": {
            "eligible_requested_targets": int(len(targets)),
            "validated_ed_increment_targets": int(len(validated)),
            "validated_interval_targets": int(len(intervals)),
        },
        "targets": target_reports,
        "gate": {
            "candidate": "ICU current state + prior ED trajectory ridge",
            "must_beat": [
                "persistence",
                "ICU-only ridge",
                "capacity-matched ED-feature placebo ridge",
            ],
            "holdouts": [
                "7 random patient splits",
                "first-careunit heldout",
                "chronological heldout",
            ],
        },
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "causal_claim_allowed": False,
            "counterfactual_treatment_effect_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
            "checkpoint_promotion_allowed": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "counts": output["counts"],
        "causal_claim_allowed": output["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
