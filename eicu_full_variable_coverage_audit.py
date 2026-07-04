"""Full numeric-variable coverage sweep for eICU observation targets.

This audit answers a narrower question than the disease routers:

    Did every numeric target visible in the local eICU transition cohorts pass
    through the observation gates?

For every numeric ``*_t`` variable in the configured 6h cohorts, the audit runs
three aggregate gates when support is available:

* same-time nowcasting against a median baseline and capacity-matched placebo;
* 6h factual forecasting against persistence and capacity-matched placebo;
* split-conformal interval coverage for targets with validated forecasting.

The report is intentionally aggregate-only.  It records target-level support
and pass/fail counts, never row-level predictions or patient identifiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS
from eicu_nowcasting_audit import (
    DEFAULT_SEEDS,
    fit_median,
    fit_ridge_nowcast,
    is_future_column,
    nowcast_feature_columns,
    target_age_column,
    target_column,
)
from eicu_sepsis_target_router import _bootstrap_ci, _group_folds, _round, split_subjects


DEFAULT_TASKS = (
    ("sepsis", Path("eicu_sepsis_transitions_6h.parquet")),
    ("aki", Path("eicu_aki_transitions_6h.parquet")),
    ("respiratory", Path("eicu_respiratory_transitions_6h.parquet")),
    *(
        (module, Path(f"eicu_{module}_transitions_6h.parquet"))
        for module in BODY_SYSTEM_CONFIGS
    ),
)

ID_COLUMNS = {
    "stay_id",
    "subject_id",
    "hospitalid",
    "onset",
    "onset_criteria",
    "t",
    "t_plus",
    "unittype",
    "died_after_window",
    "hrs_to_death_from_cut",
}


def is_active_like_target(target: str) -> bool:
    return target.startswith("active_") or target.endswith("_active") or target.endswith("_active_t")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def eligible_current_targets(frame: pd.DataFrame) -> tuple[str, ...]:
    targets = []
    for column in frame.columns:
        if not column.endswith("_t"):
            continue
        target = column[:-2]
        if is_active_like_target(target):
            continue
        values = numeric_series(frame, column)
        if values.notna().any():
            targets.append(target)
    return tuple(sorted(set(targets)))


def eligible_forecast_targets(frame: pd.DataFrame, future_suffix: str) -> tuple[str, ...]:
    targets = []
    for target in eligible_current_targets(frame):
        future = f"{target}_{future_suffix}"
        if future not in frame:
            continue
        if numeric_series(frame, future).notna().any():
            targets.append(target)
    return tuple(sorted(set(targets)))


def forecast_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in ID_COLUMNS:
            continue
        if is_future_column(column):
            continue
        if column.endswith("_active_t") or column.startswith("active_"):
            continue
        if column.endswith("_sources") or column.endswith("_kinds"):
            continue
        if target == "vasopressor_requirement" and column.startswith("act_"):
            continue
        if (
            column.endswith("_t")
            or column.endswith("_age_hr")
            or column.startswith("hist_")
            or column.startswith("act_")
            or column == "hours_since_onset"
        ):
            values = numeric_series(frame, column)
            if values.notna().any():
                columns.append(column)
    return sorted(set(columns))


def nowcast_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    current = target_column(target)
    if current not in frame:
        return frame.iloc[0:0].copy()
    selected = frame[numeric_series(frame, current).notna()].copy()
    selected["truth"] = numeric_series(selected, current).astype("float64")
    return selected[np.isfinite(selected["truth"])].copy()


def forecast_rows(frame: pd.DataFrame, target: str, future_suffix: str) -> pd.DataFrame:
    current = target_column(target)
    future = f"{target}_{future_suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    selected = frame[numeric_series(frame, current).notna() & numeric_series(frame, future).notna()].copy()
    selected["truth"] = numeric_series(selected, future).astype("float64")
    selected["current"] = numeric_series(selected, current).astype("float64")
    return selected[np.isfinite(selected["truth"]) & np.isfinite(selected["current"])].copy()


def prepare_features(train: pd.DataFrame, predict: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.zeros((len(train), 0), dtype=np.float64), np.zeros((len(predict), 0), dtype=np.float64)
    train_x = train[columns].apply(pd.to_numeric, errors="coerce")
    predict_x = predict[columns].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    predict_x = predict_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return (
        ((train_x - mean) / std).to_numpy(dtype=np.float64),
        ((predict_x - mean) / std).to_numpy(dtype=np.float64),
    )


def fit_ridge_forecast(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target: str,
    features: list[str],
    alpha: float,
    seed: int,
    placebo: bool = False,
) -> np.ndarray:
    current = target_column(target)
    if train.empty or len(train) < max(20, len(features) + 2):
        return np.full(len(predict), np.nan, dtype=np.float64)
    x_train, x_predict = prepare_features(train, predict, features)
    y = train["truth"].to_numpy(dtype=np.float64) - numeric_series(train, current).to_numpy(dtype=np.float64)
    if placebo:
        rng = np.random.default_rng(int(seed))
        y = rng.permutation(y)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_predict = np.c_[np.ones(len(x_predict)), x_predict]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return numeric_series(predict, current).to_numpy(dtype=np.float64) + x_predict @ beta


def cluster_delta(rows: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="float64")
    truth = rows["truth"].to_numpy(dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(baseline)
    if not finite.any():
        return pd.Series(dtype="float64")
    temp = pd.DataFrame({
        "subject_id": rows.loc[finite, "subject_id"].to_numpy(dtype=object),
        "delta": np.abs(candidate[finite] - truth[finite]) - np.abs(baseline[finite] - truth[finite]),
    })
    return temp.groupby("subject_id")["delta"].mean()


def delta_summary(rows: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    deltas = cluster_delta(rows, candidate, baseline)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_baseline": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=samples)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_baseline": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def mae(values: np.ndarray, truth: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(truth)
    if not finite.any():
        return None
    return float(np.abs(values[finite] - truth[finite]).mean())


def evaluate(rows: pd.DataFrame, real: np.ndarray, baseline: np.ndarray, placebo: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(rows)),
        "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
        "mae": {
            "baseline": _round(mae(baseline, truth)),
            "placebo_ridge": _round(mae(placebo, truth)),
            "ridge": _round(mae(real, truth)),
        },
        "delta_vs_baseline": delta_summary(rows, real, baseline, seed=seed, samples=samples),
        "delta_vs_placebo": delta_summary(rows, real, placebo, seed=seed + 1234, samples=samples),
    }


def conformal_radius(residuals: np.ndarray, level: float) -> float | None:
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return None
    ordered = np.sort(residuals)
    rank = int(np.ceil((len(ordered) + 1) * float(level))) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return float(ordered[rank])


def interval_summary(
    calibration_rows: pd.DataFrame,
    calibration_pred: np.ndarray,
    test_rows: pd.DataFrame,
    test_pred: np.ndarray,
    level: float,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, object]:
    cal_truth = calibration_rows["truth"].to_numpy(dtype=np.float64)
    test_truth = test_rows["truth"].to_numpy(dtype=np.float64)
    cal_resid = np.abs(np.asarray(calibration_pred, dtype=np.float64) - cal_truth)
    test_resid = np.abs(np.asarray(test_pred, dtype=np.float64) - test_truth)
    cal_resid = cal_resid[np.isfinite(cal_resid)]
    test_resid = test_resid[np.isfinite(test_resid)]
    radius = conformal_radius(cal_resid, level)
    coverage = None
    if radius is not None and len(test_resid):
        coverage = float((test_resid <= radius).mean())
    return {
        "level": float(level),
        "calibration_rows": int(len(cal_resid)),
        "test_rows": int(len(test_resid)),
        "radius": _round(radius),
        "coverage": _round(coverage),
        "coverage_gate_passed": bool(
            radius is not None
            and len(cal_resid) >= int(min_calibration_rows)
            and len(test_resid) >= int(min_test_rows)
            and coverage is not None
            and 0.87 <= coverage <= 0.93
        ),
    }


def oof_predictions(
    discovery: pd.DataFrame,
    target: str,
    features: list[str],
    seed: int,
    inner_folds: int,
    ridge_alpha: float,
    mode: str,
) -> dict[str, np.ndarray]:
    baseline = np.full(len(discovery), np.nan, dtype=np.float64)
    real = np.full(len(discovery), np.nan, dtype=np.float64)
    placebo = np.full(len(discovery), np.nan, dtype=np.float64)
    folds = _group_folds(discovery["subject_id"].to_numpy(dtype=object), seed=seed + 17, folds=inner_folds)
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        train = discovery.iloc[train_idx]
        val = discovery.iloc[val_idx]
        if mode == "nowcast":
            baseline[val_idx] = fit_median(train, val)
            real[val_idx] = fit_ridge_nowcast(train, val, features, ridge_alpha, seed=seed + 101 * fold_index)
            placebo[val_idx] = fit_ridge_nowcast(
                train,
                val,
                features,
                ridge_alpha,
                seed=seed + 10_000 + 101 * fold_index,
                placebo=True,
            )
        elif mode == "forecast":
            baseline[val_idx] = val["current"].to_numpy(dtype=np.float64)
            real[val_idx] = fit_ridge_forecast(train, val, target, features, ridge_alpha, seed=seed + 101 * fold_index)
            placebo[val_idx] = fit_ridge_forecast(
                train,
                val,
                target,
                features,
                ridge_alpha,
                seed=seed + 10_000 + 101 * fold_index,
                placebo=True,
            )
        else:
            raise ValueError(f"unsupported mode: {mode}")
    return {"baseline": baseline, "ridge": real, "placebo_ridge": placebo}


def fit_predictions(
    discovery: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    features: list[str],
    seed: int,
    ridge_alpha: float,
    mode: str,
) -> dict[str, np.ndarray]:
    if mode == "nowcast":
        return {
            "baseline": fit_median(discovery, heldout),
            "ridge": fit_ridge_nowcast(discovery, heldout, features, ridge_alpha, seed=seed + 20_000),
            "placebo_ridge": fit_ridge_nowcast(
                discovery,
                heldout,
                features,
                ridge_alpha,
                seed=seed + 30_000,
                placebo=True,
            ),
        }
    if mode == "forecast":
        return {
            "baseline": heldout["current"].to_numpy(dtype=np.float64),
            "ridge": fit_ridge_forecast(discovery, heldout, target, features, ridge_alpha, seed=seed + 20_000),
            "placebo_ridge": fit_ridge_forecast(
                discovery,
                heldout,
                target,
                features,
                ridge_alpha,
                seed=seed + 30_000,
                placebo=True,
            ),
        }
    raise ValueError(f"unsupported mode: {mode}")


def audit_target_split(
    rows: pd.DataFrame,
    target: str,
    features: list[str],
    discovery_groups: set[object],
    heldout_groups: set[object],
    seed: int,
    min_pairs: int,
    min_subjects: int,
    inner_folds: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    group_column: str,
    mode: str,
    conformal_level: float,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, object]:
    discovery = rows[rows[group_column].isin(discovery_groups)].copy()
    heldout = rows[rows[group_column].isin(heldout_groups)].copy()
    supported = (
        len(discovery) >= int(min_pairs)
        and discovery["subject_id"].nunique() >= int(min_subjects)
        and len(features) > 0
    )
    if not supported:
        return {
            "status": "insufficient_support",
            "discovery_rows": int(len(discovery)),
            "discovery_subjects": int(discovery["subject_id"].nunique()) if len(discovery) else 0,
            "feature_count": int(len(features)),
        }
    oof = oof_predictions(discovery, target, features, seed, inner_folds, ridge_alpha, mode)
    discovery_eval = evaluate(
        discovery,
        oof["ridge"],
        oof["baseline"],
        oof["placebo_ridge"],
        seed=seed + 40_000,
        samples=bootstrap_samples,
    )
    selected = bool(
        discovery_eval["delta_vs_baseline"]["significant"]
        and discovery_eval["delta_vs_placebo"]["significant"]
    )
    heldout_pred = fit_predictions(discovery, heldout, target, features, seed, ridge_alpha, mode)
    heldout_eval = evaluate(
        heldout,
        heldout_pred["ridge"],
        heldout_pred["baseline"],
        heldout_pred["placebo_ridge"],
        seed=seed + 50_000,
        samples=bootstrap_samples,
    )
    interval = None
    if mode == "forecast" and selected:
        interval = interval_summary(
            discovery,
            oof["ridge"],
            heldout,
            heldout_pred["ridge"],
            level=conformal_level,
            min_calibration_rows=min_calibration_rows,
            min_test_rows=min_test_rows,
        )
    return {
        "status": "evaluated",
        "feature_count": int(len(features)),
        "selected_ridge": selected,
        "discovery": discovery_eval,
        "heldout": heldout_eval,
        "interval": interval,
    }


def summarize_split_reports(reports: list[dict[str, object]], mode: str) -> dict[str, int]:
    selected = 0
    beats_base = 0
    beats_placebo = 0
    interval = 0
    evaluated = 0
    for report in reports:
        if report.get("status") != "evaluated":
            continue
        evaluated += 1
        selected += int(bool(report.get("selected_ridge")))
        heldout = report.get("heldout") or {}
        beats_base += int(bool((heldout.get("delta_vs_baseline") or {}).get("significant")))
        beats_placebo += int(bool((heldout.get("delta_vs_placebo") or {}).get("significant")))
        interval += int(bool((report.get("interval") or {}).get("coverage_gate_passed")))
    output = {
        "evaluated_splits": int(evaluated),
        "selected_ridge_splits": int(selected),
        "heldout_beats_baseline_splits": int(beats_base),
        "heldout_beats_placebo_splits": int(beats_placebo),
    }
    if mode == "forecast":
        output["interval_pass_splits"] = int(interval)
    return output


def hospital_gate_pass(report: dict[str, object], require_interval: bool = False) -> bool:
    if report.get("status") != "evaluated":
        return False
    heldout = report.get("heldout") or {}
    passed = bool(
        report.get("selected_ridge")
        and (heldout.get("delta_vs_baseline") or {}).get("significant")
        and (heldout.get("delta_vs_placebo") or {}).get("significant")
    )
    if require_interval:
        passed = passed and bool((report.get("interval") or {}).get("coverage_gate_passed"))
    return passed


def audit_target(
    frame: pd.DataFrame,
    target: str,
    rows: pd.DataFrame,
    features: list[str],
    seeds: tuple[int, ...],
    discovery_fraction: float,
    min_pairs: int,
    min_subjects: int,
    inner_folds: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    mode: str,
    conformal_level: float,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, object]:
    support = {
        "rows": int(len(rows)),
        "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
        "feature_count": int(len(features)),
    }
    random_reports = []
    for seed in seeds:
        discovery_groups, heldout_groups = split_subjects(frame, seed, discovery_fraction)
        random_reports.append(audit_target_split(
            rows,
            target,
            features,
            discovery_groups,
            heldout_groups,
            seed,
            min_pairs,
            min_subjects,
            inner_folds,
            ridge_alpha,
            bootstrap_samples,
            group_column="subject_id",
            mode=mode,
            conformal_level=conformal_level,
            min_calibration_rows=min_calibration_rows,
            min_test_rows=min_test_rows,
        ))
    hospital_report = {"available": False}
    if "hospitalid" in frame and frame["hospitalid"].nunique() >= 3:
        discovery_groups, heldout_groups = split_subjects(
            frame,
            9001,
            discovery_fraction,
            group_column="hospitalid",
        )
        hospital_report = {
            "available": True,
            **audit_target_split(
                rows,
                target,
                features,
                discovery_groups,
                heldout_groups,
                9001,
                min_pairs,
                min_subjects,
                inner_folds,
                ridge_alpha,
                bootstrap_samples,
                group_column="hospitalid",
                mode=mode,
                conformal_level=conformal_level,
                min_calibration_rows=min_calibration_rows,
                min_test_rows=min_test_rows,
            ),
        }
    summary = summarize_split_reports(random_reports, mode)
    split_count = len(seeds)
    validated = bool(
        summary["evaluated_splits"] == split_count
        and summary["selected_ridge_splits"] == split_count
        and summary["heldout_beats_baseline_splits"] == split_count
        and summary["heldout_beats_placebo_splits"] == split_count
        and hospital_gate_pass(hospital_report)
    )
    interval_validated = False
    if mode == "forecast":
        interval_validated = bool(
            validated
            and summary["interval_pass_splits"] == split_count
            and hospital_gate_pass(hospital_report, require_interval=True)
        )
    return {
        "support": support,
        "random_patient_splits": summary,
        "hospital_holdout": {
            "available": bool(hospital_report.get("available")),
            "gate_passed": hospital_gate_pass(hospital_report),
            "interval_gate_passed": hospital_gate_pass(hospital_report, require_interval=True) if mode == "forecast" else None,
            "heldout": hospital_report.get("heldout") if hospital_report.get("available") else None,
            "interval": hospital_report.get("interval") if mode == "forecast" and hospital_report.get("available") else None,
        },
        "validated": validated,
        "interval_validated": interval_validated if mode == "forecast" else None,
    }


def audit_module(
    module: str,
    cohort: Path,
    seeds: tuple[int, ...],
    discovery_fraction: float,
    min_pairs: int,
    min_subjects: int,
    inner_folds: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    future_suffix: str,
    conformal_level: float,
    min_calibration_rows: int,
    min_test_rows: int,
    max_targets: int | None,
) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    nowcast_targets = list(eligible_current_targets(frame))
    forecast_targets = list(eligible_forecast_targets(frame, future_suffix))
    if max_targets is not None:
        nowcast_targets = nowcast_targets[:max_targets]
        forecast_targets = forecast_targets[:max_targets]

    target_reports: dict[str, dict[str, object]] = {}
    all_targets = sorted(set(nowcast_targets) | set(forecast_targets))
    for target in all_targets:
        report: dict[str, object] = {
            "eligible_nowcast": target in nowcast_targets,
            "eligible_forecast": target in forecast_targets,
        }
        if target in nowcast_targets:
            rows = nowcast_rows(frame, target)
            report["nowcast"] = audit_target(
                frame,
                target,
                rows,
                nowcast_feature_columns(frame, target),
                seeds,
                discovery_fraction,
                min_pairs,
                min_subjects,
                inner_folds,
                ridge_alpha,
                bootstrap_samples,
                mode="nowcast",
                conformal_level=conformal_level,
                min_calibration_rows=min_calibration_rows,
                min_test_rows=min_test_rows,
            )
        if target in forecast_targets:
            rows = forecast_rows(frame, target, future_suffix)
            report["forecast"] = audit_target(
                frame,
                target,
                rows,
                forecast_feature_columns(frame, target),
                seeds,
                discovery_fraction,
                min_pairs,
                min_subjects,
                inner_folds,
                ridge_alpha,
                bootstrap_samples,
                mode="forecast",
                conformal_level=conformal_level,
                min_calibration_rows=min_calibration_rows,
                min_test_rows=min_test_rows,
            )
        target_reports[target] = report

    nowcast_validated = sorted(
        target
        for target, report in target_reports.items()
        if (report.get("nowcast") or {}).get("validated")
    )
    forecast_validated = sorted(
        target
        for target, report in target_reports.items()
        if (report.get("forecast") or {}).get("validated")
    )
    interval_validated = sorted(
        target
        for target, report in target_reports.items()
        if (report.get("forecast") or {}).get("interval_validated")
    )
    return {
        "module": module,
        "cohort": cohort.name,
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "eligible_current_numeric_targets": nowcast_targets,
        "eligible_forecast_numeric_targets": forecast_targets,
        "validated": {
            "nowcast_targets": nowcast_validated,
            "forecast_targets": forecast_validated,
            "interval_targets": interval_validated,
        },
        "counts": {
            "eligible_nowcast_targets": int(len(nowcast_targets)),
            "eligible_forecast_targets": int(len(forecast_targets)),
            "validated_nowcast_targets": int(len(nowcast_validated)),
            "validated_forecast_targets": int(len(forecast_validated)),
            "validated_interval_targets": int(len(interval_validated)),
        },
        "targets": target_reports,
    }


def parse_tasks(raw: str) -> tuple[tuple[str, Path], ...]:
    if not raw.strip():
        return DEFAULT_TASKS
    tasks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        module, cohort = item.split(":", 1)
        tasks.append((module, Path(cohort)))
    return tuple(tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--future-suffix", default="tp6")
    parser.add_argument("--conformal-level", type=float, default=0.9)
    parser.add_argument("--min-calibration-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--max-targets", type=int, default=-1, help="Debug limit per module; -1 means all.")
    parser.add_argument("--output", type=Path, default=Path("eicu_full_variable_coverage_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    max_targets = None if int(args.max_targets) < 0 else int(args.max_targets)
    tasks = parse_tasks(args.tasks)
    modules = [
        audit_module(
            module,
            cohort,
            seeds,
            args.discovery_fraction,
            args.min_pairs,
            args.min_subjects,
            args.inner_folds,
            args.ridge_alpha,
            args.bootstrap_samples,
            args.future_suffix,
            args.conformal_level,
            args.min_calibration_rows,
            args.min_test_rows,
            max_targets,
        )
        for module, cohort in tasks
    ]
    aggregate = {
        "eligible_nowcast_targets": int(sum(item["counts"]["eligible_nowcast_targets"] for item in modules)),
        "eligible_forecast_targets": int(sum(item["counts"]["eligible_forecast_targets"] for item in modules)),
        "validated_nowcast_targets": int(sum(item["counts"]["validated_nowcast_targets"] for item in modules)),
        "validated_forecast_targets": int(sum(item["counts"]["validated_forecast_targets"] for item in modules)),
        "validated_interval_targets": int(sum(item["counts"]["validated_interval_targets"] for item in modules)),
    }
    output = {
        "artifact": "eICU full numeric-variable coverage audit",
        "definition": "all numeric *_t targets in configured 6h transition cohorts are evaluated for nowcast, forecast, and interval gates when support exists",
        "future_suffix": args.future_suffix,
        "patient_split_seeds": list(seeds),
        "aggregate_counts": aggregate,
        "modules": modules,
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "same_time_imputation_allowed_for_validated_targets": True,
            "factual_prediction_allowed_for_validated_targets": True,
            "causal_claim_allowed": False,
            "counterfactual_treatment_effect_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "aggregate_counts": aggregate,
        "causal_claim_allowed": output["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

