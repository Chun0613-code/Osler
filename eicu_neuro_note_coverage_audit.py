"""Audit eICU neuro-note observation targets.

This uses the same aggregate gate family as the numeric whole-body layer, but
with two note-specific leakage guards:

* same-time nowcast excludes every ``neuro_*`` feature, so one neurologic
  assessment cannot trivially predict another row from the same assessment;
* 6h forecasting may use current ``neuro_*_t`` values because they are known
  before the forecast horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_full_variable_coverage_audit import (
    audit_target,
    forecast_feature_columns,
    forecast_rows,
    hospital_gate_pass,
    nowcast_rows,
)
from eicu_nowcasting_audit import DEFAULT_SEEDS, is_future_column, nowcast_feature_columns


TARGETS = (
    "neuro_gcs",
    "neuro_sedation_score",
    "neuro_delirium_present",
    "neuro_mental_abnormal",
    "neuro_pupils_abnormal",
    "neuro_motor_abnormal",
)


def _round(value, digits: int = 6):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def note_nowcast_features(frame: pd.DataFrame, target: str) -> list[str]:
    blocked_prefixes = ("neuro_",)
    return [
        column
        for column in nowcast_feature_columns(frame, target)
        if not column.startswith(blocked_prefixes)
    ]


def note_forecast_features(frame: pd.DataFrame, target: str) -> list[str]:
    columns = []
    for column in forecast_feature_columns(frame, target):
        if is_future_column(column):
            continue
        columns.append(column)
    return sorted(set(columns))


def audit_note_target(
    frame: pd.DataFrame,
    target: str,
    seeds: tuple[int, ...],
    discovery_fraction: float,
    min_pairs: int,
    min_subjects: int,
    inner_folds: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    conformal_level: float,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "eligible_nowcast": f"{target}_t" in frame,
        "eligible_forecast": f"{target}_tp6" in frame,
    }
    if report["eligible_nowcast"]:
        rows = nowcast_rows(frame, target)
        report["nowcast"] = audit_target(
            frame,
            target,
            rows,
            note_nowcast_features(frame, target),
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
        report["nowcast"]["feature_policy"] = "all neuro_* features excluded to prevent same-assessment leakage"
    if report["eligible_forecast"]:
        rows = forecast_rows(frame, target, "tp6")
        report["forecast"] = audit_target(
            frame,
            target,
            rows,
            note_forecast_features(frame, target),
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
        report["forecast"]["feature_policy"] = "current neuro_* observations allowed for future factual prediction"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("/private/tmp/eicu_acute_neuro_note_observation_transitions_6h.parquet"))
    parser.add_argument("--output", type=Path, default=Path("eicu_neuro_note_coverage_audit.json"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--conformal-level", type=float, default=0.9)
    parser.add_argument("--min-calibration-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    target_reports: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        print(f"Auditing neuro-note target: {target}", flush=True)
        target_reports[target] = audit_note_target(
            frame,
            target,
            seeds,
            args.discovery_fraction,
            args.min_pairs,
            args.min_subjects,
            args.inner_folds,
            args.ridge_alpha,
            args.bootstrap_samples,
            args.conformal_level,
            args.min_calibration_rows,
            args.min_test_rows,
        )

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
    output = {
        "artifact": "eICU neuro-note structured observation coverage audit",
        "cohort": "local-only eICU acute-neuro note-enhanced transition cohort",
        "targets": list(TARGETS),
        "counts": {
            "eligible_nowcast_targets": int(sum(bool((report.get("nowcast") or {}).get("support")) for report in target_reports.values())),
            "eligible_forecast_targets": int(sum(bool((report.get("forecast") or {}).get("support")) for report in target_reports.values())),
            "validated_nowcast_targets": int(len(nowcast_validated)),
            "validated_forecast_targets": int(len(forecast_validated)),
            "validated_interval_targets": int(len(interval_validated)),
        },
        "validated": {
            "nowcast_targets": nowcast_validated,
            "forecast_targets": forecast_validated,
            "interval_targets": interval_validated,
        },
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "note_timestamp_leakage_guard": True,
            "causal_claim_allowed": False,
            "counterfactual_treatment_effect_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
        "target_reports": target_reports,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "counts": output["counts"],
        "validated": output["validated"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
