"""Audit temporal predict-update coupling between eICU body-system modules.

The prior static coupling audit rejected same-window upstream feature
concatenation.  This script tests a more physiologic candidate: each upstream
organ system is first compressed into temporal predict-update belief features,
then those features are added to downstream factual prediction.

Passing means only that the temporal coupling belief improves held-out factual
prediction beyond both:

* a downstream baseline ridge model without the coupling belief;
* a capacity-matched placebo with the same number of random features.

It does not mean causality, counterfactual validity, clinical authority,
checkpoint promotion, runtime treatment authority, or active-rule promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from body_temporal_coupling_belief import temporal_coupling_features
from eicu_body_system_coupling_audit import (
    COUPLING_SPECS,
    SEEDS,
    _active_column,
    _delta_summary,
    _mae,
    _target_rows,
    _target_scale,
    _with_placebo,
    summarize_reports,
)
from eicu_body_system_target_router import _fit_ridge
from eicu_sepsis_target_router import _feature_columns, _round, _subject_column, split_subjects


METHODS = ("baseline_ridge", "temporal_coupled_ridge", "placebo_ridge")


def _attach_temporal_features(frame: pd.DataFrame, edge_name: str) -> tuple[pd.DataFrame, list[str]]:
    features = temporal_coupling_features(frame, edge_name)
    feature_columns = list(features.columns)
    output = pd.concat([frame.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    return output, feature_columns


def _predict_methods(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    future_suffix: str,
    baseline_features: list[str],
    coupling_features: list[str],
    ridge_alpha: float,
    seed: int,
) -> dict[str, np.ndarray]:
    candidate_features = sorted(set(baseline_features + coupling_features))
    placebo_train, placebo_columns = _with_placebo(train, coupling_features, seed=seed + 301)
    placebo_heldout, _placebo_columns = _with_placebo(heldout, coupling_features, seed=seed + 401)
    return {
        "baseline_ridge": _fit_ridge(train, heldout, target, baseline_features, ridge_alpha, future_suffix),
        "temporal_coupled_ridge": _fit_ridge(train, heldout, target, candidate_features, ridge_alpha, future_suffix),
        "placebo_ridge": _fit_ridge(
            placebo_train,
            placebo_heldout,
            target,
            sorted(set(baseline_features + placebo_columns)),
            ridge_alpha,
            future_suffix,
        ),
    }


def _target_report(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    coupling_features: list[str],
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
    bootstrap_samples: int,
    active_only: bool,
) -> dict[str, object]:
    train_target = _target_rows(train, target, future_suffix)
    heldout_target = _target_rows(heldout, target, future_suffix)
    active_column = _active_column(heldout_target)
    if active_only and active_column is not None:
        heldout_target = heldout_target[heldout_target[active_column].fillna(False).astype(bool)].copy()

    baseline_features = [column for column in _feature_columns(train_target, target) if column not in set(coupling_features)]
    coupling_features = [column for column in coupling_features if column in train_target and column in heldout_target]
    future = f"{target}_{future_suffix}"
    truth = heldout_target[future].to_numpy(dtype=np.float64) if future in heldout_target else np.asarray([])
    scale = _target_scale(train_target, target, future_suffix) if len(train_target) else 1.0
    predictions = _predict_methods(
        train_target,
        heldout_target,
        target,
        future_suffix,
        baseline_features,
        coupling_features,
        ridge_alpha,
        seed,
    )
    temporal = predictions["temporal_coupled_ridge"]
    baseline = predictions["baseline_ridge"]
    placebo = predictions["placebo_ridge"]
    return {
        "rows": int(len(heldout_target)),
        "subjects": int(heldout_target[_subject_column(heldout_target)].nunique()) if len(heldout_target) else 0,
        "active_only": bool(active_only),
        "feature_counts": {
            "baseline": int(len(baseline_features)),
            "temporal_coupling": int(len(coupling_features)),
            "candidate": int(len(set(baseline_features + coupling_features))),
            "placebo": int(len(coupling_features)),
        },
        "temporal_coupling_features": coupling_features,
        "mae": {
            method: _round(_mae(truth, predictions[method], scale=scale))
            for method in METHODS
        },
        "candidate_vs_baseline": _delta_summary(
            heldout_target,
            truth,
            temporal,
            baseline,
            scale,
            seed=seed + 17,
            bootstrap_samples=bootstrap_samples,
        ),
        "candidate_vs_placebo": _delta_summary(
            heldout_target,
            truth,
            temporal,
            placebo,
            scale,
            seed=seed + 31,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def split_report(
    frame: pd.DataFrame,
    spec: dict[str, object],
    seed: int,
    *,
    coupling_features: list[str],
    discovery_fraction: float,
    ridge_alpha: float,
    bootstrap_samples: int,
    group_column: str | None = None,
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
    future_suffix = str(spec["future_suffix"])
    targets = {}
    for index, target in enumerate(spec["targets"]):
        target = str(target)
        targets[target] = {
            "all_windows": _target_report(
                discovery,
                heldout,
                target,
                coupling_features,
                future_suffix,
                ridge_alpha,
                seed=seed + 1000 * index,
                bootstrap_samples=bootstrap_samples,
                active_only=False,
            ),
            "active_only": _target_report(
                discovery,
                heldout,
                target,
                coupling_features,
                future_suffix,
                ridge_alpha,
                seed=seed + 2000 * index,
                bootstrap_samples=bootstrap_samples,
                active_only=True,
            ),
        }
    return {
        "seed": int(seed),
        "group_column": group_column,
        "discovery_groups": int(len(discovery_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "overlap": int(len(discovery_groups & heldout_groups)),
        "temporal_coupling_feature_count": int(len(coupling_features)),
        "targets": targets,
    }


def audit_spec(
    spec: dict[str, object],
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    discovery_fraction: float,
    ridge_alpha: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    cohort = data_dir / str(spec["cohort"])
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    frame, coupling_features = _attach_temporal_features(frame, str(spec["name"]))
    random_reports = [
        split_report(
            frame,
            spec,
            seed=seed,
            coupling_features=coupling_features,
            discovery_fraction=discovery_fraction,
            ridge_alpha=ridge_alpha,
            bootstrap_samples=bootstrap_samples,
        )
        for seed in seeds
    ]
    hospital_report = None
    if "hospitalid" in frame and frame["hospitalid"].nunique(dropna=True) >= 2:
        hospital_report = split_report(
            frame,
            spec,
            seed=1777,
            coupling_features=coupling_features,
            discovery_fraction=discovery_fraction,
            ridge_alpha=ridge_alpha,
            bootstrap_samples=bootstrap_samples,
            group_column="hospitalid",
        )
    targets = tuple(str(target) for target in spec["targets"])
    return {
        "name": spec["name"],
        "source_system": spec["source_system"],
        "target_system": spec["target_system"],
        "rationale": spec["rationale"],
        "cohort": str(spec["cohort"]),
        "future_suffix": spec["future_suffix"],
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()) if len(frame) else 0,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "targets": targets,
        "candidate": "temporal_predict_update_coupling_belief",
        "random_patient_splits": {
            "seeds": list(seeds),
            "summary": summarize_reports(random_reports, targets),
            "runs": random_reports,
        },
        "hospital_holdout": hospital_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output", default="eicu_body_system_temporal_coupling_audit.json")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--edges", default="all", help="comma-separated edge names or 'all'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    requested = {item.strip() for item in args.edges.split(",") if item.strip()} if args.edges != "all" else None
    specs = [spec for spec in COUPLING_SPECS if requested is None or str(spec["name"]) in requested]
    results = [
        audit_spec(
            spec,
            data_dir=Path(args.data_dir),
            seeds=seeds,
            discovery_fraction=args.discovery_fraction,
            ridge_alpha=args.ridge_alpha,
            bootstrap_samples=args.bootstrap_samples,
        )
        for spec in specs
    ]
    report = {
        "experiment": "eICU body-system temporal predict-update coupling audit",
        "gate": {
            "candidate": "temporal_predict_update_belief_features",
            "baseline": "ridge_without_temporal_coupling_belief",
            "placebo": "ridge_plus_capacity_matched_random_features",
            "pass_rule": "candidate must significantly beat both baseline and placebo on held-out patients",
        },
        "edges": results,
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included_in_report": False,
            "factual_predictive_signal_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
            "checkpoint_promotion_allowed": False,
            "active_rule_promotion_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    concise = {
        edge["name"]: {
            target: edge["random_patient_splits"]["summary"][target]["active_only"]["candidate_passes_both_count"]
            for target in edge["targets"]
        }
        for edge in results
    }
    print(json.dumps({
        "output": args.output,
        "edges": len(results),
        "active_pass_both_counts": concise,
        "causal_claim_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
