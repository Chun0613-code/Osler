"""Audit patient-specific musculoskeletal / rhabdomyolysis belief features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_body_system_target_router import _fit_ridge
from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _round, split_subjects
from musculoskeletal_rhabdo_belief import (
    MUSCULOSKELETAL_BELIEF_COLUMNS,
    musculoskeletal_rhabdo_belief_state_features,
    placebo_musculoskeletal_belief_features,
)


DEFAULT_TARGETS = (
    "cpk",
    "myoglobin",
    "ldh",
    "potassium",
    "creatinine",
    "bun",
    "phosphate",
    "calcium",
    "ionized_calcium",
    "bicarbonate",
    "ph",
    "urine_output",
    "map",
    "lactate",
)
METHODS = ("baseline_ridge", "muscle_belief_ridge", "placebo_belief_ridge")


def _subject_column(frame: pd.DataFrame) -> str:
    return "subject_id" if "subject_id" in frame else "stay_id"


def _target_rows(frame: pd.DataFrame, target: str, future_suffix: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_{future_suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def attach_belief(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    extra = musculoskeletal_rhabdo_belief_state_features(frame)
    return pd.concat([frame.reset_index(drop=True), extra.reset_index(drop=True)], axis=1), list(extra.columns)


def _fit_predictions(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
    belief_columns: list[str],
) -> dict[str, np.ndarray]:
    belief_set = set(belief_columns)
    base_features = [
        column for column in _feature_columns(train, target)
        if column not in belief_set
    ]
    base_pred = _fit_ridge(train, heldout, target, base_features, ridge_alpha, future_suffix)
    belief_pred = _fit_ridge(train, heldout, target, sorted(set(base_features + belief_columns)), ridge_alpha, future_suffix)

    train_placebo = pd.concat([train, placebo_musculoskeletal_belief_features(train, seed=seed + 202)], axis=1)
    heldout_placebo = pd.concat([heldout, placebo_musculoskeletal_belief_features(heldout, seed=seed + 303)], axis=1)
    placebo_columns = [f"placebo_{column}" for column in MUSCULOSKELETAL_BELIEF_COLUMNS]
    placebo_pred = _fit_ridge(
        train_placebo,
        heldout_placebo,
        target,
        sorted(set(base_features + placebo_columns)),
        ridge_alpha,
        future_suffix,
    )
    return {
        "baseline_ridge": base_pred,
        "muscle_belief_ridge": belief_pred,
        "placebo_belief_ridge": placebo_pred,
    }


def _mae(truth: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not mask.any():
        return None
    return float(np.mean(np.abs(pred[mask] - truth[mask])))


def _delta_by_subject(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> pd.Series:
    mask = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(baseline)
    if not mask.any():
        return pd.Series(dtype="float64")
    group_column = _subject_column(rows)
    temp = rows.loc[mask, [group_column]].copy()
    temp["delta"] = np.abs(pred[mask] - truth[mask]) - np.abs(baseline[mask] - truth[mask])
    return temp.groupby(group_column)["delta"].mean()


def _delta_summary(
    rows: pd.DataFrame,
    truth: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    deltas = _delta_by_subject(rows, truth, pred, baseline)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_baseline": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=bootstrap_samples)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_baseline": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _target_report(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
    bootstrap_samples: int,
    belief_columns: list[str],
) -> dict[str, object]:
    train_target = _target_rows(train, target, future_suffix)
    heldout_target = _target_rows(heldout, target, future_suffix)
    future = f"{target}_{future_suffix}"
    truth = heldout_target[future].to_numpy(dtype=np.float64) if future in heldout_target else np.asarray([])
    predictions = _fit_predictions(train_target, heldout_target, target, future_suffix, ridge_alpha, seed, belief_columns)
    baseline = predictions["baseline_ridge"]
    placebo = predictions["placebo_belief_ridge"]
    candidate = predictions["muscle_belief_ridge"]
    return {
        "rows": int(len(heldout_target)),
        "subjects": int(heldout_target[_subject_column(heldout_target)].nunique()) if len(heldout_target) else 0,
        "mae": {method: _round(_mae(truth, predictions[method])) for method in METHODS},
        "candidate_vs_baseline": _delta_summary(
            heldout_target,
            truth,
            candidate,
            baseline,
            seed=seed + 11,
            bootstrap_samples=bootstrap_samples,
        ),
        "candidate_vs_placebo": _delta_summary(
            heldout_target,
            truth,
            candidate,
            placebo,
            seed=seed + 23,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def split_report(
    frame: pd.DataFrame,
    seed: int,
    future_suffix: str,
    discovery_fraction: float,
    ridge_alpha: float,
    bootstrap_samples: int,
    targets: tuple[str, ...],
    group_column: str | None = None,
    belief_columns: list[str] | None = None,
) -> dict[str, object]:
    group_column = group_column or _subject_column(frame)
    discovery_groups, heldout_groups = split_subjects(frame, seed=seed, discovery_fraction=discovery_fraction, group_column=group_column)
    discovery = frame[frame[group_column].isin(discovery_groups)].copy()
    heldout = frame[frame[group_column].isin(heldout_groups)].copy()
    active_column = next((column for column in heldout.columns if column.endswith("_active_t")), None)
    active = heldout[heldout[active_column].fillna(False).astype(bool)].copy() if active_column else heldout
    belief_columns = belief_columns or []
    target_reports = {
        target: _target_report(discovery, active, target, future_suffix, ridge_alpha, seed, bootstrap_samples, belief_columns)
        for target in targets
    }
    return {
        "seed": seed,
        "group_column": group_column,
        "discovery_groups": int(len(discovery_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "heldout_rows": int(len(active)),
        "targets": target_reports,
    }


def summarize_splits(reports: list[dict[str, object]], targets: tuple[str, ...]) -> dict[str, object]:
    summary = {}
    for target in targets:
        target_reports = [report["targets"][target] for report in reports if target in report["targets"]]
        base_deltas = [
            item["candidate_vs_baseline"]["point_delta"]
            for item in target_reports
            if item["candidate_vs_baseline"]["point_delta"] is not None
        ]
        placebo_deltas = [
            item["candidate_vs_placebo"]["point_delta"]
            for item in target_reports
            if item["candidate_vs_placebo"]["point_delta"] is not None
        ]
        pass_both = [
            item for item in target_reports
            if item["candidate_vs_baseline"]["significant"] and item["candidate_vs_placebo"]["significant"]
        ]
        summary[target] = {
            "evaluated_splits": int(len(target_reports)),
            "pass_both_count": int(len(pass_both)),
            "median_delta_vs_baseline": _round(float(np.median(base_deltas))) if base_deltas else None,
            "median_delta_vs_placebo": _round(float(np.median(placebo_deltas))) if placebo_deltas else None,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="eicu_musculoskeletal_rhabdo_transitions_6h.parquet")
    parser.add_argument("--future-suffix", default="tp6")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--output", default="eicu_musculoskeletal_belief_audit.json")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input).reset_index(drop=True)
    frame, belief_columns = attach_belief(frame)
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    seeds = (7, 11, 19, 23, 37, 53, 71)
    patient_splits = [
        split_report(frame, seed, args.future_suffix, args.discovery_fraction, args.ridge_alpha, args.bootstrap_samples, targets, belief_columns=belief_columns)
        for seed in seeds
    ]
    holdout = None
    if "hospitalid" in frame and frame["hospitalid"].nunique(dropna=True) >= 2:
        holdout = split_report(
            frame,
            9901,
            args.future_suffix,
            args.discovery_fraction,
            args.ridge_alpha,
            args.bootstrap_samples,
            targets,
            group_column="hospitalid",
            belief_columns=belief_columns,
        )
    summary = summarize_splits(patient_splits, targets)
    validated = []
    for target in targets:
        target_holdout = (holdout or {}).get("targets", {}).get(target, {})
        if (
            summary[target]["pass_both_count"] == len(seeds)
            and target_holdout.get("candidate_vs_baseline", {}).get("significant")
            and target_holdout.get("candidate_vs_placebo", {}).get("significant")
        ):
            validated.append(target)
    output = {
        "artifact": "Musculoskeletal/rhabdomyolysis belief audit",
        "input": args.input,
        "future_suffix": args.future_suffix,
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "belief_feature_count": int(len(belief_columns)),
        "targets": targets,
        "patient_split_summary": summary,
        "hospital_holdout": holdout,
        "validated_targets": validated,
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
    print(json.dumps({"validated_targets": validated, "summary": summary, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
