"""Observation coverage audit for the MIMIC-IV-ED scene.

The ED audit reuses the same aggregate gates as the ICU whole-body layer, but
uses ED-appropriate external stress tests:

* seven patient-heldout random splits;
* arrival-transport heldout;
* chronological heldout.

It remains observational-only: no treatment, causal, counterfactual, clinical,
or runtime authority is granted by a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from eicu_full_variable_coverage_audit import (
    audit_target_split,
    eligible_current_targets,
    eligible_forecast_targets,
    forecast_feature_columns,
    forecast_rows,
    hospital_gate_pass,
    nowcast_feature_columns,
    nowcast_rows,
    summarize_split_reports,
)
from eicu_nowcasting_audit import DEFAULT_SEEDS
from eicu_sepsis_target_router import split_subjects
from mimiciv_cross_database_coverage_audit import add_time_holdout_column, capped_subject_rows


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def external_group_report(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
    features: list[str],
    group_column: str,
    discovery_fraction: float,
    seed: int,
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


def audit_target_ed(
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
    transport_report = external_group_report(
        frame,
        rows,
        target,
        features,
        "arrival_transport",
        discovery_fraction,
        7101,
        min_pairs,
        min_subjects,
        inner_folds,
        ridge_alpha,
        bootstrap_samples,
        mode,
        conformal_level,
        min_calibration_rows,
        min_test_rows,
    )
    time_report = external_group_report(
        frame,
        rows,
        target,
        features,
        "time_holdout_group",
        discovery_fraction,
        7201,
        min_pairs,
        min_subjects,
        inner_folds,
        ridge_alpha,
        bootstrap_samples,
        mode,
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
        and hospital_gate_pass(transport_report)
        and hospital_gate_pass(time_report)
    )
    interval_validated = False
    if mode == "forecast":
        interval_validated = bool(
            validated
            and summary["interval_pass_splits"] == split_count
            and hospital_gate_pass(transport_report, require_interval=True)
            and hospital_gate_pass(time_report, require_interval=True)
        )
    return {
        "support": support,
        "random_patient_splits": summary,
        "arrival_transport_holdout": {
            "available": bool(transport_report.get("available")),
            "gate_passed": hospital_gate_pass(transport_report),
            "interval_gate_passed": hospital_gate_pass(transport_report, require_interval=True) if mode == "forecast" else None,
            "heldout": transport_report.get("heldout") if transport_report.get("available") else None,
            "interval": transport_report.get("interval") if mode == "forecast" and transport_report.get("available") else None,
        },
        "time_holdout": {
            "available": bool(time_report.get("available")),
            "gate_passed": hospital_gate_pass(time_report),
            "interval_gate_passed": hospital_gate_pass(time_report, require_interval=True) if mode == "forecast" else None,
            "heldout": time_report.get("heldout") if time_report.get("available") else None,
            "interval": time_report.get("interval") if mode == "forecast" and time_report.get("available") else None,
        },
        "validated": validated,
        "interval_validated": interval_validated if mode == "forecast" else None,
    }


def audit_ed(
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
    max_rows_per_target: int | None,
) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    frame = add_time_holdout_column(frame, discovery_fraction)
    nowcast_targets = list(eligible_current_targets(frame))
    forecast_targets = list(eligible_forecast_targets(frame, future_suffix))
    target_reports: dict[str, dict[str, object]] = {}
    for target in sorted(set(nowcast_targets) | set(forecast_targets)):
        print(f"Auditing MIMIC-IV-ED target: {target}", flush=True)
        report: dict[str, object] = {
            "eligible_nowcast": target in nowcast_targets,
            "eligible_forecast": target in forecast_targets,
        }
        if target in nowcast_targets:
            rows, sampling = capped_subject_rows(
                nowcast_rows(frame, target),
                max_rows_per_target,
                seed=stable_seed("mimiciv-ed", "nowcast", target, future_suffix),
            )
            report["nowcast"] = audit_target_ed(
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
                "nowcast",
                conformal_level,
                min_calibration_rows,
                min_test_rows,
            )
            report["nowcast"]["sampling"] = sampling
        if target in forecast_targets:
            rows, sampling = capped_subject_rows(
                forecast_rows(frame, target, future_suffix),
                max_rows_per_target,
                seed=stable_seed("mimiciv-ed", "forecast", target, future_suffix),
            )
            report["forecast"] = audit_target_ed(
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
                "forecast",
                conformal_level,
                min_calibration_rows,
                min_test_rows,
            )
            report["forecast"]["sampling"] = sampling
        target_reports[target] = report
    validated_nowcast = sorted(
        target
        for target, report in target_reports.items()
        if (report.get("nowcast") or {}).get("validated")
    )
    validated_forecast = sorted(
        target
        for target, report in target_reports.items()
        if (report.get("forecast") or {}).get("validated")
    )
    validated_interval = sorted(
        target
        for target, report in target_reports.items()
        if (report.get("forecast") or {}).get("interval_validated")
    )
    return {
        "artifact": "MIMIC-IV-ED v2.2 ED-scene observation coverage audit",
        "cohort": "local-only ED observation transition cohort",
        "future_suffix": future_suffix,
        "patient_split_seeds": list(seeds),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "arrival_transport_groups": int(frame["arrival_transport"].nunique()) if "arrival_transport" in frame else None,
        "disposition_groups": int(frame["disposition"].nunique()) if "disposition" in frame else None,
        "max_rows_per_target": int(max_rows_per_target) if max_rows_per_target is not None else None,
        "eligible_current_numeric_targets": nowcast_targets,
        "eligible_forecast_numeric_targets": forecast_targets,
        "validated": {
            "nowcast_targets": validated_nowcast,
            "forecast_targets": validated_forecast,
            "interval_targets": validated_interval,
        },
        "counts": {
            "eligible_nowcast_targets": int(len(nowcast_targets)),
            "eligible_forecast_targets": int(len(forecast_targets)),
            "validated_nowcast_targets": int(len(validated_nowcast)),
            "validated_forecast_targets": int(len(validated_forecast)),
            "validated_interval_targets": int(len(validated_interval)),
        },
        "targets": target_reports,
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "same_time_imputation_allowed_for_validated_targets": True,
            "factual_prediction_allowed_for_validated_targets": True,
            "causal_claim_allowed": False,
            "counterfactual_treatment_effect_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
            "new_runtime_authority": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-suffix", default="tp1")
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
    max_rows_per_target = None if args.max_rows_per_target <= 0 else int(args.max_rows_per_target)
    report = audit_ed(
        args.cohort,
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
        max_rows_per_target,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "counts": report["counts"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
