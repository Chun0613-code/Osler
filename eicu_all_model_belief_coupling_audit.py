"""Audit whether all current belief models add cross-system forecast signal.

This aggregate-only audit compares:

* baseline: ordinary table/action ridge features;
* candidate: baseline plus all five current predict-update belief feature sets;
* placebo: baseline plus capacity-matched random features.

It is factual and observational only.  It grants no causal, clinical,
counterfactual, runtime, checkpoint, or active-rule authority.
"""

from __future__ import annotations

import json
import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from aki_renal_belief import renal_belief_state_v2_features
from cardiovascular_belief import cardiovascular_belief_state_features
from electrolyte_belief import electrolyte_belief_state_features
from endocrine_belief import endocrine_belief_state_features
from respiratory_belief import respiratory_belief_state_features
from eicu_body_system_target_router import _fit_ridge
from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _round, _subject_column, split_subjects


SEEDS = (7, 11, 19, 23, 37, 53, 71)
warnings.filterwarnings("ignore", category=PerformanceWarning)

COHORTS = (
    {
        "name": "respiratory_6h",
        "path": "eicu_respiratory_transitions_6h.parquet",
        "future_suffix": "tp6",
        "targets": ("o2sat", "respiratory_rate", "heart_rate", "bicarbonate", "map", "paco2", "ph"),
    },
    {
        "name": "electrolyte_6h",
        "path": "eicu_electrolyte_acid_base_transitions_6h.parquet",
        "future_suffix": "tp6",
        "targets": (
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "anion_gap",
            "creatinine",
            "magnesium",
            "phosphate",
            "map",
        ),
    },
    {
        "name": "endocrine_6h",
        "path": "eicu_endocrine_stress_transitions_6h.parquet",
        "future_suffix": "tp6",
        "targets": ("glucose", "anion_gap", "bicarbonate", "sodium", "potassium", "map", "temperature"),
    },
    {
        "name": "cardiovascular_6h",
        "path": "eicu_cardiovascular_instability_transitions_6h.parquet",
        "future_suffix": "tp6",
        "targets": ("heart_rate", "map", "lactate", "o2sat", "respiratory_rate", "creatinine"),
    },
)


def attach_all_beliefs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    builders = (
        ("renal", renal_belief_state_v2_features),
        ("cardiovascular", cardiovascular_belief_state_features),
        ("electrolyte", electrolyte_belief_state_features),
        ("respiratory", respiratory_belief_state_features),
        ("endocrine", endocrine_belief_state_features),
    )
    parts = [frame.reset_index(drop=True)]
    columns: list[str] = []
    for _name, builder in builders:
        extra = builder(frame).reset_index(drop=True)
        parts.append(extra)
        columns.extend(list(extra.columns))
    combined = pd.concat(parts, axis=1)
    columns = sorted(set(column for column in columns if column in combined))
    return combined, columns


def target_rows(frame: pd.DataFrame, target: str, suffix: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_{suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def placebo_columns(frame: pd.DataFrame, count: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    if count <= 0:
        return frame.copy(), []
    rng = np.random.default_rng(seed)
    columns = [f"placebo_all_belief_{i}" for i in range(count)]
    values = rng.normal(size=(len(frame), count))
    placebo = pd.DataFrame(values, index=frame.index, columns=columns, dtype=np.float64)
    output = pd.concat([frame, placebo], axis=1)
    return output, columns


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

    base_features = [column for column in _feature_columns(train, target) if column not in set(belief_columns)]
    usable_beliefs = [
        column for column in belief_columns
        if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    baseline = _fit_ridge(train, test, target, base_features, 10.0, suffix)
    candidate = _fit_ridge(train, test, target, sorted(set(base_features + usable_beliefs)), 10.0, suffix)

    placebo_train, placebo = placebo_columns(train, len(usable_beliefs), seed + 101)
    placebo_test, _ = placebo_columns(test, len(usable_beliefs), seed + 202)
    placebo_pred = _fit_ridge(placebo_train, placebo_test, target, sorted(set(base_features + placebo)), 10.0, suffix)

    truth = test[f"{target}_{suffix}"].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "group_column": group_column,
        "rows": int(len(test)),
        "subjects": int(test[_subject_column(test)].nunique()) if len(test) else 0,
        "feature_counts": {
            "baseline": int(len(base_features)),
            "belief": int(len(usable_beliefs)),
            "candidate": int(len(set(base_features + usable_beliefs))),
            "placebo": int(len(placebo)),
        },
        "candidate_vs_baseline": delta_summary(test, truth, candidate, baseline, seed + 11),
        "candidate_vs_placebo": delta_summary(test, truth, candidate, placebo_pred, seed + 23),
    }


def summarize(target_reports: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [report for report in target_reports if report.get("status") == "evaluated"]
    both = [
        report for report in evaluated
        if report["candidate_vs_baseline"]["significant"] and report["candidate_vs_placebo"]["significant"]
    ]
    base_deltas = [
        float(report["candidate_vs_baseline"]["point_delta"])
        for report in evaluated
        if report["candidate_vs_baseline"]["point_delta"] is not None
    ]
    placebo_deltas = [
        float(report["candidate_vs_placebo"]["point_delta"])
        for report in evaluated
        if report["candidate_vs_placebo"]["point_delta"] is not None
    ]
    return {
        "evaluated_splits": int(len(evaluated)),
        "pass_both_count": int(len(both)),
        "median_delta_vs_baseline": _round(float(np.median(base_deltas))) if base_deltas else None,
        "median_delta_vs_placebo": _round(float(np.median(placebo_deltas))) if placebo_deltas else None,
    }


def audit_cohort(spec: dict[str, object]) -> dict[str, object]:
    print(f"Loading {spec['name']}", flush=True)
    frame = pd.read_parquet(PROJECT / str(spec["path"])).reset_index(drop=True)
    frame, belief_columns = attach_all_beliefs(frame)
    print(f"Attached {len(belief_columns)} belief features to {spec['name']} rows={len(frame)}", flush=True)
    targets = {}
    for target in spec["targets"]:
        split_reports = [
            split_target_report(frame, str(target), str(spec["future_suffix"]), belief_columns, seed)
            for seed in SEEDS
        ]
        hospital_report = None
        if "hospitalid" in frame and frame["hospitalid"].nunique(dropna=True) >= 2:
            hospital_report = split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                9901,
                group_column="hospitalid",
            )
        targets[str(target)] = {
            "patient_splits": summarize(split_reports),
            "hospital_holdout": hospital_report,
        }
        print(spec["name"], target, targets[str(target)]["patient_splits"], flush=True)
    return {
        "name": spec["name"],
        "cohort": spec["path"],
        "future_suffix": spec["future_suffix"],
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "belief_feature_count": int(len(belief_columns)),
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eicu_all_model_belief_coupling_audit.json")
    args = parser.parse_args()
    cohorts = [audit_cohort(spec) for spec in COHORTS]
    validated = []
    for cohort in cohorts:
        for target, report in cohort["targets"].items():
            patient = report["patient_splits"]
            hospital = report["hospital_holdout"] or {}
            if (
                patient["pass_both_count"] == len(SEEDS)
                and hospital.get("status") == "evaluated"
                and hospital.get("candidate_vs_baseline", {}).get("significant")
                and hospital.get("candidate_vs_placebo", {}).get("significant")
            ):
                validated.append({"cohort": cohort["name"], "target": target, "patient": patient})
    output = {
        "artifact": "temporary all-model belief coupling probe",
        "definition": "table/action ridge baseline vs baseline plus all current belief model outputs vs capacity-matched placebo",
        "boundary": {
            "factual_observed_inputs_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_authority": False,
            "row_level_outputs_written": False,
        },
        "validated": validated,
        "cohorts": cohorts,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validated": validated, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
