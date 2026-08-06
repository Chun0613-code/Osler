"""Audit patient-specific electrolyte / acid-base belief features.

This tests whether an online predict-update electrolyte belief state improves
downstream observable prediction beyond both:

* baseline ridge using the usual observed state/action variables; and
* a capacity-matched placebo ridge using random features of the same width.

The output is aggregate-only and grants no causal, counterfactual, clinical,
runtime, checkpoint, or active-rule authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_body_system_target_router import _fit_ridge
from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _round, split_subjects
from electrolyte_belief import (
    ELECTROLYTE_BELIEF_COLUMNS,
    electrolyte_belief_state_features,
    placebo_electrolyte_belief_features,
)


DEFAULT_TARGETS = (
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "anion_gap",
    "calcium",
    "ionized_calcium",
    "magnesium",
    "phosphate",
    "serum_osmolality",
    "map",
    "creatinine",
)
METHODS = ("baseline_ridge", "electrolyte_belief_ridge", "placebo_belief_ridge")


def _subject_column(frame: pd.DataFrame) -> str:
    return "subject_id" if "subject_id" in frame else "stay_id"


def _target_rows(frame: pd.DataFrame, target: str, future_suffix: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_{future_suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def attach_belief(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    extra = electrolyte_belief_state_features(frame)
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
    base_features = [
        column for column in _feature_columns(train, target)
        if column not in set(belief_columns)
    ]
    base_pred = _fit_ridge(
        train,
        heldout,
        target,
        base_features,
        ridge_alpha,
        future_suffix,
    )
    belief_pred = _fit_ridge(
        train,
        heldout,
        target,
        sorted(set(base_features + belief_columns)),
        ridge_alpha,
        future_suffix,
    )

    train_placebo = pd.concat([
        train,
        placebo_electrolyte_belief_features(train, seed=seed + 202),
    ], axis=1)
    heldout_placebo = pd.concat([
        heldout,
        placebo_electrolyte_belief_features(heldout, seed=seed + 303),
    ], axis=1)
    placebo_columns = [f"placebo_{column}" for column in ELECTROLYTE_BELIEF_COLUMNS]
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
        "electrolyte_belief_ridge": belief_pred,
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
    predictions = _fit_predictions(
        train_target,
        heldout_target,
        target,
        future_suffix,
        ridge_alpha,
        seed,
        belief_columns,
    )
    baseline = predictions["baseline_ridge"]
    placebo = predictions["placebo_belief_ridge"]
    candidate = predictions["electrolyte_belief_ridge"]
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
    discovery_groups, heldout_groups = split_subjects(
        frame,
        seed=seed,
        discovery_fraction=discovery_fraction,
        group_column=group_column,
    )
    discovery = frame[frame[group_column].isin(discovery_groups)].copy()
    heldout = frame[frame[group_column].isin(heldout_groups)].copy()
    belief_columns = belief_columns or list(ELECTROLYTE_BELIEF_COLUMNS)
    target_reports = {
        target: _target_report(
            discovery,
            heldout,
            target,
            future_suffix,
            ridge_alpha,
            seed=seed + 1000 * index,
            bootstrap_samples=bootstrap_samples,
            belief_columns=belief_columns,
        )
        for index, target in enumerate(targets)
    }
    passed_targets = [
        target for target, report in target_reports.items()
        if report["candidate_vs_baseline"]["significant"]
        and report["candidate_vs_placebo"]["significant"]
    ]
    return {
        "seed": int(seed),
        "group_column": group_column,
        "discovery_groups": int(len(discovery_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "overlap": int(len(discovery_groups & heldout_groups)),
        "targets": target_reports,
        "passed_targets": passed_targets,
        "passed_count": int(len(passed_targets)),
    }


def summarize_reports(reports: list[dict[str, object]], targets: tuple[str, ...]) -> dict[str, object]:
    output = {}
    for target in targets:
        baseline_wins = []
        placebo_wins = []
        baseline_deltas = []
        placebo_deltas = []
        for report in reports:
            target_report = report["targets"][target]
            base_delta = target_report["candidate_vs_baseline"]
            placebo_delta = target_report["candidate_vs_placebo"]
            baseline_wins.append(bool(base_delta["significant"]))
            placebo_wins.append(bool(placebo_delta["significant"]))
            if base_delta["point_delta"] is not None:
                baseline_deltas.append(float(base_delta["point_delta"]))
            if placebo_delta["point_delta"] is not None:
                placebo_deltas.append(float(placebo_delta["point_delta"]))
        output[target] = {
            "candidate_beats_baseline_significant_count": int(sum(baseline_wins)),
            "candidate_beats_placebo_significant_count": int(sum(placebo_wins)),
            "candidate_passes_both_count": int(sum(a and b for a, b in zip(baseline_wins, placebo_wins))),
            "median_delta_vs_baseline": _round(float(np.median(baseline_deltas))) if baseline_deltas else None,
            "median_delta_vs_placebo": _round(float(np.median(placebo_deltas))) if placebo_deltas else None,
        }
    return output


def audit(
    cohort: Path,
    *,
    future_suffix: str,
    seeds: list[int],
    targets: tuple[str, ...],
    discovery_fraction: float,
    ridge_alpha: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    frame, belief_columns = attach_belief(frame)
    random_reports = [
        split_report(
            frame,
            seed=seed,
            future_suffix=future_suffix,
            discovery_fraction=discovery_fraction,
            ridge_alpha=ridge_alpha,
            bootstrap_samples=bootstrap_samples,
            targets=targets,
            belief_columns=belief_columns,
        )
        for seed in seeds
    ]
    hospital_report = None
    if "hospitalid" in frame and frame["hospitalid"].nunique() >= 3:
        hospital_report = split_report(
            frame,
            seed=9001,
            future_suffix=future_suffix,
            discovery_fraction=discovery_fraction,
            ridge_alpha=ridge_alpha,
            bootstrap_samples=bootstrap_samples,
            targets=targets,
            group_column="hospitalid",
            belief_columns=belief_columns,
        )
    active_column = "electrolyte_acid_base_active_t"
    return {
        "cohort": cohort.name,
        "future_suffix": future_suffix,
        "belief_columns": belief_columns,
        "cohort_summary": {
            "rows": int(len(frame)),
            "subjects": int(frame[_subject_column(frame)].nunique()) if _subject_column(frame) in frame else None,
            "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
            "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
            "active_rows": int(frame[active_column].fillna(False).sum()) if active_column in frame else None,
        },
        "random_patient_splits": {
            "seeds": seeds,
            "summary": summarize_reports(random_reports, targets),
            "runs": random_reports,
        },
        "hospital_holdout": hospital_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="eicu_electrolyte_acid_base_transitions_6h.parquet")
    parser.add_argument("--future-suffix", default="tp6")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--output", default="eicu_electrolyte_belief_audit.json")
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    result = audit(
        Path(args.cohort),
        future_suffix=args.future_suffix,
        seeds=seeds,
        targets=targets,
        discovery_fraction=args.discovery_fraction,
        ridge_alpha=args.ridge_alpha,
        bootstrap_samples=args.bootstrap_samples,
    )
    report = {
        "experiment": "eICU electrolyte predict-update belief downstream observable audit",
        "belief": {
            "name": "electrolyte_acid_base_patient_specific_state",
            "columns": ELECTROLYTE_BELIEF_COLUMNS,
            "direct_hidden_state_accuracy_claim_allowed": False,
            "requires_capacity_matched_placebo": True,
            "targets": targets,
        },
        "gate": {
            "candidate": "ridge_realfit_plus_electrolyte_belief_features",
            "baseline": "ridge_realfit",
            "placebo": "ridge_realfit_plus_capacity_matched_noise_features",
            "pass_rule": "candidate must significantly beat both baseline and placebo on held-out patients",
        },
        "audit": result,
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included_in_report": False,
            "direct_hidden_state_accuracy_claim_allowed": False,
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
            "active_rule_promotion_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "future_suffix": args.future_suffix,
        "summary": result["random_patient_splits"]["summary"],
        "hospital_passed": (result["hospital_holdout"] or {}).get("passed_targets"),
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
