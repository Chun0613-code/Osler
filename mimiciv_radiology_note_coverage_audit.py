"""Audit MIMIC-IV radiology-note findings as structured observation targets.

This audit is intentionally narrower than the generic MIMIC coverage sweep.  It
evaluates only ``rad_*`` targets produced by
``mimiciv_radiology_note_observation_extract.py``.

Same-time nowcasting excludes all ``rad_*`` features so a model cannot use
other fields extracted from the same report to recover the target.  Forecasting
may use current ``rad_*_t`` features because those reports are authored before
the anchor and are legitimate prior observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eicu_full_variable_coverage_audit import (
    ID_COLUMNS,
    audit_target_split,
    hospital_gate_pass,
    summarize_split_reports,
)
from eicu_nowcasting_audit import DEFAULT_SEEDS, is_future_column, target_age_column, target_column
from eicu_sepsis_target_router import split_subjects
from mimiciv_cross_database_coverage_audit import add_time_holdout_column


RAD_TARGETS = (
    "rad_pulmonary_edema",
    "rad_pleural_effusion",
    "rad_consolidation",
    "rad_atelectasis",
    "rad_pneumothorax",
    "rad_cardiomegaly",
)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def feature_columns(frame: pd.DataFrame, target: str, mode: str) -> list[str]:
    excluded = {
        *ID_COLUMNS,
        "first_careunit",
        "last_careunit",
        "time_holdout_group",
        target_column(target),
        target_age_column(target),
    }
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if is_future_column(column):
            continue
        if column.endswith("_sources") or column.endswith("_kinds"):
            continue
        if column.startswith("active_") or column.endswith("_active_t"):
            continue
        if mode == "nowcast" and column.startswith("rad_"):
            continue
        if column.endswith("_t") or column.endswith("_age_hr") or column.startswith("hist_"):
            if numeric_series(frame, column).notna().any():
                columns.append(column)
    return sorted(set(columns))


def nowcast_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    current = target_column(target)
    if current not in frame:
        return frame.iloc[0:0].copy()
    rows = frame[numeric_series(frame, current).notna()].copy()
    rows["truth"] = numeric_series(rows, current).astype("float64")
    return rows


def forecast_rows(frame: pd.DataFrame, target: str, future_suffix: str) -> pd.DataFrame:
    current = target_column(target)
    future = f"{target}_{future_suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    rows = frame[numeric_series(frame, current).notna() & numeric_series(frame, future).notna()].copy()
    rows["truth"] = numeric_series(rows, future).astype("float64")
    rows["current"] = numeric_series(rows, current).astype("float64")
    return rows


def external_report(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
    features: list[str],
    mode: str,
    group_column: str,
    discovery_fraction: float,
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
    if group_column not in frame or frame[group_column].nunique(dropna=True) < 2:
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
        **audit_target_split(
            rows,
            target,
            features,
            set(discovery_groups),
            set(heldout_groups),
            seed,
            min_pairs,
            min_subjects,
            inner_folds,
            ridge_alpha,
            bootstrap_samples,
            group_column=group_column,
            mode=mode,
            conformal_level=conformal_level,
            min_calibration_rows=min_calibration_rows,
            min_test_rows=min_test_rows,
        ),
    }


def audit_target(
    frame: pd.DataFrame,
    target: str,
    mode: str,
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
) -> dict[str, object]:
    rows = nowcast_rows(frame, target) if mode == "nowcast" else forecast_rows(frame, target, future_suffix)
    features = feature_columns(frame, target, mode)
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
    careunit = external_report(
        frame,
        rows,
        target,
        features,
        mode,
        "first_careunit",
        discovery_fraction,
        9001,
        min_pairs,
        min_subjects,
        inner_folds,
        ridge_alpha,
        bootstrap_samples,
        conformal_level,
        min_calibration_rows,
        min_test_rows,
    )
    time = external_report(
        frame,
        rows,
        target,
        features,
        mode,
        "time_holdout_group",
        discovery_fraction,
        9901,
        min_pairs,
        min_subjects,
        inner_folds,
        ridge_alpha,
        bootstrap_samples,
        conformal_level,
        min_calibration_rows,
        min_test_rows,
    )
    summary = summarize_split_reports(random_reports, mode)
    split_count = len(seeds)
    validated = bool(
        summary["evaluated_splits"] == split_count
        and summary["selected_ridge_splits"] == split_count
        and summary["heldout_beats_baseline_splits"] == split_count
        and summary["heldout_beats_placebo_splits"] == split_count
        and hospital_gate_pass(careunit)
        and hospital_gate_pass(time)
    )
    interval_validated = False
    if mode == "forecast":
        interval_validated = bool(
            validated
            and summary["interval_pass_splits"] == split_count
            and hospital_gate_pass(careunit, require_interval=True)
            and hospital_gate_pass(time, require_interval=True)
        )
    return {
        "support": {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
            "feature_count": int(len(features)),
        },
        "feature_policy": (
            "all rad_* features excluded for nowcast to prevent same-report leakage"
            if mode == "nowcast"
            else "current rad_* observations allowed for future factual prediction"
        ),
        "random_patient_splits": summary,
        "careunit_holdout": {
            "available": bool(careunit.get("available")),
            "gate_passed": hospital_gate_pass(careunit),
            "interval_gate_passed": hospital_gate_pass(careunit, require_interval=True) if mode == "forecast" else None,
            "heldout": careunit.get("heldout") if careunit.get("available") else None,
            "interval": careunit.get("interval") if mode == "forecast" and careunit.get("available") else None,
        },
        "time_holdout": {
            "available": bool(time.get("available")),
            "gate_passed": hospital_gate_pass(time),
            "interval_gate_passed": hospital_gate_pass(time, require_interval=True) if mode == "forecast" else None,
            "heldout": time.get("heldout") if time.get("available") else None,
            "interval": time.get("interval") if mode == "forecast" and time.get("available") else None,
        },
        "validated": validated,
        "interval_validated": interval_validated if mode == "forecast" else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("/private/tmp/mimiciv_observation_radiology_transitions_6h.parquet"))
    parser.add_argument("--output", type=Path, default=Path("mimiciv_radiology_note_coverage_audit.json"))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    frame = add_time_holdout_column(frame, args.discovery_fraction)
    reports = {}
    for target in RAD_TARGETS:
        print(f"Auditing radiology note target: {target}", flush=True)
        reports[target] = {
            "nowcast": audit_target(
                frame,
                target,
                "nowcast",
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
            ),
            "forecast": audit_target(
                frame,
                target,
                "forecast",
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
            ),
        }
    validated_nowcast = sorted(target for target, report in reports.items() if report["nowcast"]["validated"])
    validated_forecast = sorted(target for target, report in reports.items() if report["forecast"]["validated"])
    validated_interval = sorted(target for target, report in reports.items() if report["forecast"]["interval_validated"])
    output = {
        "artifact": "MIMIC-IV radiology-note structured observation coverage audit",
        "cohort": str(args.cohort),
        "targets": list(RAD_TARGETS),
        "counts": {
            "eligible_nowcast_targets": len(RAD_TARGETS),
            "eligible_forecast_targets": len(RAD_TARGETS),
            "validated_nowcast_targets": len(validated_nowcast),
            "validated_forecast_targets": len(validated_forecast),
            "validated_interval_targets": len(validated_interval),
        },
        "validated": {
            "nowcast_targets": validated_nowcast,
            "forecast_targets": validated_forecast,
            "interval_targets": validated_interval,
        },
        "target_reports": reports,
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "note_timestamp_leakage_guard": True,
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
        "counts": output["counts"],
        "validated": output["validated"],
        "causal_claim_allowed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
