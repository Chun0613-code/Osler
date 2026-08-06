"""Focused ablation for all-model MAP coupling hits."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
BUILDERS = {
    "renal": renal_belief_state_v2_features,
    "cardiovascular": cardiovascular_belief_state_features,
    "electrolyte": electrolyte_belief_state_features,
    "respiratory": respiratory_belief_state_features,
    "endocrine": endocrine_belief_state_features,
}
COHORTS = (
    ("respiratory_6h", "eicu_respiratory_transitions_6h.parquet", "tp6", "map"),
    ("electrolyte_6h", "eicu_electrolyte_acid_base_transitions_6h.parquet", "tp6", "map"),
)


def attach_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    parts = [frame.reset_index(drop=True)]
    groups = {}
    for name, builder in BUILDERS.items():
        extra = builder(frame).reset_index(drop=True)
        parts.append(extra)
        groups[name] = list(extra.columns)
    return pd.concat(parts, axis=1), groups


def target_rows(frame: pd.DataFrame, target: str, suffix: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_{suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def placebo(frame: pd.DataFrame, count: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    columns = [f"placebo_{i}" for i in range(count)]
    if not columns:
        return frame.copy(), columns
    values = np.random.default_rng(seed).normal(size=(len(frame), count))
    return pd.concat([frame, pd.DataFrame(values, index=frame.index, columns=columns)], axis=1), columns


def delta(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, base: np.ndarray, seed: int) -> dict[str, object]:
    mask = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(base)
    if not mask.any():
        return {"subjects": 0, "point_delta": None, "bootstrap_95_ci": [None, None], "significant": False}
    temp = rows.loc[mask, [_subject_column(rows)]].copy()
    temp["delta"] = np.abs(pred[mask] - truth[mask]) - np.abs(base[mask] - truth[mask])
    values = temp.groupby(_subject_column(rows))["delta"].mean().to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=200)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def split_report(frame: pd.DataFrame, target: str, suffix: str, columns: list[str], seed: int, group_column: str | None = None) -> dict[str, object]:
    group_column = group_column or _subject_column(frame)
    discovery_groups, heldout_groups = split_subjects(frame, seed=seed, discovery_fraction=0.67, group_column=group_column)
    train = frame[frame[group_column].isin(discovery_groups)].copy()
    test = frame[frame[group_column].isin(heldout_groups)].copy()
    train = target_rows(train, target, suffix)
    test = target_rows(test, target, suffix)
    active = next((column for column in test.columns if column.endswith("_active_t")), None)
    if active:
        test = test[test[active].fillna(False).astype(bool)].copy()
    base_features = [column for column in _feature_columns(train, target) if column not in set(columns)]
    usable = [column for column in columns if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()]
    baseline = _fit_ridge(train, test, target, base_features, 10.0, suffix)
    candidate = _fit_ridge(train, test, target, sorted(set(base_features + usable)), 10.0, suffix)
    train_p, pcols = placebo(train, len(usable), seed + 111)
    test_p, _ = placebo(test, len(usable), seed + 222)
    placebo_pred = _fit_ridge(train_p, test_p, target, sorted(set(base_features + pcols)), 10.0, suffix)
    truth = test[f"{target}_{suffix}"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(test)),
        "subjects": int(test[_subject_column(test)].nunique()) if len(test) else 0,
        "candidate_vs_baseline": delta(test, truth, candidate, baseline, seed + 11),
        "candidate_vs_placebo": delta(test, truth, candidate, placebo_pred, seed + 23),
    }


def summarize(reports: list[dict[str, object]]) -> dict[str, object]:
    passed = [
        report for report in reports
        if report["candidate_vs_baseline"]["significant"] and report["candidate_vs_placebo"]["significant"]
    ]
    b = [report["candidate_vs_baseline"]["point_delta"] for report in reports if report["candidate_vs_baseline"]["point_delta"] is not None]
    p = [report["candidate_vs_placebo"]["point_delta"] for report in reports if report["candidate_vs_placebo"]["point_delta"] is not None]
    return {
        "pass_both_count": int(len(passed)),
        "median_delta_vs_baseline": _round(float(np.median(b))) if b else None,
        "median_delta_vs_placebo": _round(float(np.median(p))) if p else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eicu_all_model_map_ablation_audit.json")
    args = parser.parse_args()
    output = {}
    for cohort_name, path, suffix, target in COHORTS:
        frame = pd.read_parquet(PROJECT / path).reset_index(drop=True)
        frame, groups = attach_groups(frame)
        group_reports = {}
        for group_name, columns in groups.items():
            reports = [split_report(frame, target, suffix, columns, seed) for seed in SEEDS]
            hospital = split_report(frame, target, suffix, columns, 9901, group_column="hospitalid")
            group_reports[group_name] = {
                "patient_splits": summarize(reports),
                "hospital_holdout": {
                    "candidate_vs_baseline": hospital["candidate_vs_baseline"],
                    "candidate_vs_placebo": hospital["candidate_vs_placebo"],
                },
            }
            print(cohort_name, group_name, group_reports[group_name]["patient_splits"], flush=True)
        output[cohort_name] = group_reports
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": args.output}, indent=2))


if __name__ == "__main__":
    main()
