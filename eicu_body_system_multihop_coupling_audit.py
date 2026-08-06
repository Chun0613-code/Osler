"""Audit multi-hop body-system factual coupling in eICU.

The focused edge audits validate single directed edges.  This script asks a
harder question: once a first-hop edge is validated, does its mediated state add
incremental downstream signal beyond a direct edge?

The first tested path is:

    sepsis / immune-inflammatory burden -> MAP mediator -> renal targets

The baseline is not persistence and not a no-coupling model.  The baseline is a
direct sepsis-to-renal factual ridge model.  The candidate adds discovery-only
MAP mediator features.  It must beat both the direct baseline and a
capacity-matched placebo on held-out patients.

This remains factual and observational.  It grants no causal, counterfactual,
clinical, runtime, checkpoint-promotion, or active-rule authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from body_edge_specific_coupling_belief import focused_sepsis_cardiovascular_features
from eicu_body_system_coupling_audit import (
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
from eicu_sepsis_target_router import _feature_columns, _group_folds, _round, _subject_column, split_subjects


MULTIHOP_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "sepsis_map_renal_6h",
        "cohort": "eicu_sepsis_transitions_6h.parquet",
        "future_suffix": "tp6",
        "mediator_future_suffix": "tp6",
        "first_hop": "sepsis_immune_inflammatory_to_map",
        "mediator": "map",
        "target_system": "renal",
        "targets": ("creatinine", "bun", "urine_output"),
        "rationale": "test whether validated sepsis->MAP signal adds renal prediction beyond direct sepsis->renal features",
    },
    {
        "name": "sepsis_map6_renal_24h",
        "cohort": "eicu_sepsis_transitions_24h.parquet",
        "mediator_cohort": "eicu_sepsis_transitions_6h.parquet",
        "future_suffix": "tp24",
        "mediator_future_suffix": "tp6",
        "first_hop": "sepsis_immune_inflammatory_to_map_6h",
        "mediator": "map",
        "target_system": "renal",
        "targets": ("creatinine", "bun", "urine_output"),
        "rationale": "compose the validated 6h sepsis->MAP mediator with 24h renal targets on the validated cardio-renal time scale",
    },
    {
        "name": "sepsis_map6_renal_48h",
        "cohort": "eicu_sepsis_transitions_48h.parquet",
        "mediator_cohort": "eicu_sepsis_transitions_6h.parquet",
        "future_suffix": "tp48",
        "mediator_future_suffix": "tp6",
        "first_hop": "sepsis_immune_inflammatory_to_map_6h",
        "mediator": "map",
        "target_system": "renal",
        "targets": ("creatinine", "bun", "urine_output"),
        "rationale": "compose the validated 6h sepsis->MAP mediator with 48h renal targets on the validated cardio-renal time scale",
    },
)


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _attach_first_hop_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = focused_sepsis_cardiovascular_features(frame)
    columns = list(features.columns)
    return pd.concat([frame.reset_index(drop=True), features.reset_index(drop=True)], axis=1), columns


def _merge_mediator_future(
    frame: pd.DataFrame,
    spec: dict[str, object],
    *,
    data_dir: Path,
) -> pd.DataFrame:
    mediator_cohort = spec.get("mediator_cohort")
    if not mediator_cohort:
        return frame
    mediator = str(spec["mediator"])
    mediator_suffix = str(spec.get("mediator_future_suffix", spec["future_suffix"]))
    mediator_column = f"{mediator}_{mediator_suffix}"
    if mediator_column in frame:
        return frame
    mediator_frame = pd.read_parquet(
        data_dir / str(mediator_cohort),
        columns=["stay_id", "t", mediator_column],
    ).drop_duplicates(["stay_id", "t"])
    merged = frame.merge(mediator_frame, on=["stay_id", "t"], how="inner", validate="one_to_one")
    return merged.reset_index(drop=True)


def _mediator_predict(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    mediator: str,
    feature_columns: list[str],
    future_suffix: str,
    ridge_alpha: float,
) -> np.ndarray:
    feature_columns = [column for column in feature_columns if column in train and column in predict]
    return _fit_ridge(train, predict, mediator, feature_columns, ridge_alpha, future_suffix)


def _crossfit_discovery_mediator(
    discovery: pd.DataFrame,
    *,
    mediator: str,
    feature_columns: list[str],
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(discovery), np.nan, dtype=np.float64)
    groups = discovery[_subject_column(discovery)].to_numpy(dtype=object)
    folds = _group_folds(groups, seed=seed, folds=3)
    for train_idx, valid_idx in folds:
        train = discovery.iloc[train_idx].copy()
        valid = discovery.iloc[valid_idx].copy()
        predictions[valid_idx] = _mediator_predict(
            train,
            valid,
            mediator=mediator,
            feature_columns=feature_columns,
            future_suffix=future_suffix,
            ridge_alpha=ridge_alpha,
        )
    missing = ~np.isfinite(predictions)
    if missing.any():
        predictions[missing] = _mediator_predict(
            discovery,
            discovery.iloc[np.flatnonzero(missing)].copy(),
            mediator=mediator,
            feature_columns=feature_columns,
            future_suffix=future_suffix,
            ridge_alpha=ridge_alpha,
        )
    return predictions


def _map_mediator_features(frame: pd.DataFrame, predicted_map: np.ndarray) -> pd.DataFrame:
    current_map = pd.to_numeric(frame.get("map_t", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=np.float64)
    current_map[~np.isfinite(current_map)] = 75.0
    predicted = np.asarray(predicted_map, dtype=np.float64)
    predicted[~np.isfinite(predicted)] = current_map[~np.isfinite(predicted)]
    creatinine = pd.to_numeric(frame.get("creatinine_t", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=np.float64)
    bun = pd.to_numeric(frame.get("bun_t", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=np.float64)
    urine = pd.to_numeric(frame.get("urine_output_t", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=np.float64)
    lactate = pd.to_numeric(frame.get("lactate_t", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=np.float64)
    creatinine[~np.isfinite(creatinine)] = 1.1
    bun[~np.isfinite(bun)] = 22.0
    urine[~np.isfinite(urine)] = 60.0
    lactate[~np.isfinite(lactate)] = 1.5

    predicted_low_map = _clip01((65.0 - predicted) / 35.0)
    current_low_map = _clip01((65.0 - current_map) / 35.0)
    predicted_delta = np.clip(predicted - current_map, -80.0, 80.0)
    return pd.DataFrame({
        "mh_predicted_map_tp6": predicted,
        "mh_predicted_map_delta": predicted_delta,
        "mh_predicted_low_map": predicted_low_map,
        "mh_current_low_map": current_low_map,
        "mh_predicted_map_recovery": _clip01((predicted - current_map) / 25.0),
        "mh_predicted_map_worsening": _clip01((current_map - predicted) / 25.0),
        "mh_low_map_x_creatinine": predicted_low_map * _clip01((creatinine - 1.2) / 4.0),
        "mh_low_map_x_bun": predicted_low_map * _clip01((bun - 25.0) / 80.0),
        "mh_low_map_x_oliguria": predicted_low_map * _clip01((30.0 - urine) / 30.0),
        "mh_low_map_x_lactate": predicted_low_map * _clip01((lactate - 2.0) / 8.0),
    }, index=frame.index, dtype=np.float64)


def _attach_multihop_features(
    discovery: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    first_hop_features: list[str],
    mediator: str,
    mediator_future_suffix: str,
    ridge_alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    mediator_features = [column for column in _feature_columns(discovery, mediator) if column in set(first_hop_features) or column.endswith("_t") or column.endswith("_age_hr") or column.startswith(("hist_", "act_"))]
    discovery_pred = _crossfit_discovery_mediator(
        discovery,
        mediator=mediator,
        feature_columns=mediator_features,
        future_suffix=mediator_future_suffix,
        ridge_alpha=ridge_alpha,
        seed=seed,
    )
    heldout_pred = _mediator_predict(
        discovery,
        heldout,
        mediator=mediator,
        feature_columns=mediator_features,
        future_suffix=mediator_future_suffix,
        ridge_alpha=ridge_alpha,
    )
    discovery_mh = _map_mediator_features(discovery, discovery_pred)
    heldout_mh = _map_mediator_features(heldout, heldout_pred)
    columns = list(discovery_mh.columns)
    discovery = pd.concat([discovery.reset_index(drop=True), discovery_mh.reset_index(drop=True)], axis=1)
    heldout = pd.concat([heldout.reset_index(drop=True), heldout_mh.reset_index(drop=True)], axis=1)
    return discovery, heldout, columns


def _predict_methods(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    future_suffix: str,
    direct_features: list[str],
    multihop_features: list[str],
    ridge_alpha: float,
    seed: int,
) -> dict[str, np.ndarray]:
    direct_features = [column for column in direct_features if column in train and column in heldout]
    multihop_features = [column for column in multihop_features if column in train and column in heldout]
    candidate_features = sorted(set(direct_features + multihop_features))
    placebo_train, placebo_columns = _with_placebo(train, multihop_features, seed=seed + 701)
    placebo_heldout, _placebo_columns = _with_placebo(heldout, multihop_features, seed=seed + 801)
    return {
        "direct_sepsis_renal_ridge": _fit_ridge(train, heldout, target, direct_features, ridge_alpha, future_suffix),
        "multihop_ridge": _fit_ridge(train, heldout, target, candidate_features, ridge_alpha, future_suffix),
        "placebo_ridge": _fit_ridge(
            placebo_train,
            placebo_heldout,
            target,
            sorted(set(direct_features + placebo_columns)),
            ridge_alpha,
            future_suffix,
        ),
    }


def _scope_report(
    rows: pd.DataFrame,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    target_report_features: dict[str, object],
    scale: float,
    seed: int,
    bootstrap_samples: int,
    active_only: bool,
) -> dict[str, object]:
    candidate = predictions["multihop_ridge"]
    baseline = predictions["direct_sepsis_renal_ridge"]
    placebo = predictions["placebo_ridge"]
    return {
        "rows": int(len(rows)),
        "subjects": int(rows[_subject_column(rows)].nunique()) if len(rows) else 0,
        "active_only": bool(active_only),
        **target_report_features,
        "mae": {
            method: _round(_mae(truth, predictions[method], scale=scale))
            for method in ("direct_sepsis_renal_ridge", "multihop_ridge", "placebo_ridge")
        },
        "candidate_vs_baseline": _delta_summary(
            rows,
            truth,
            candidate,
            baseline,
            scale,
            seed=seed + 17,
            bootstrap_samples=bootstrap_samples,
        ),
        "candidate_vs_placebo": _delta_summary(
            rows,
            truth,
            candidate,
            placebo,
            scale,
            seed=seed + 31,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def _target_reports(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    first_hop_features: list[str],
    multihop_features: list[str],
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, dict[str, object]]:
    train_target = _target_rows(train, target, future_suffix)
    heldout_target = _target_rows(heldout, target, future_suffix)
    excluded = set(multihop_features)
    direct_features = [
        column for column in _feature_columns(train_target, target)
        if column not in excluded and (column in set(first_hop_features) or not column.startswith("mh_"))
    ]
    future = f"{target}_{future_suffix}"
    truth = heldout_target[future].to_numpy(dtype=np.float64) if future in heldout_target else np.asarray([])
    scale = _target_scale(train_target, target, future_suffix) if len(train_target) else 1.0
    predictions = _predict_methods(
        train_target,
        heldout_target,
        target,
        future_suffix,
        direct_features,
        multihop_features,
        ridge_alpha,
        seed,
    )
    target_report_features = {
        "feature_counts": {
            "direct": int(len(direct_features)),
            "multihop": int(len(multihop_features)),
            "candidate": int(len(set(direct_features + multihop_features))),
            "placebo": int(len(multihop_features)),
        },
        "first_hop_features": [column for column in first_hop_features if column in train_target],
        "multihop_features": [column for column in multihop_features if column in train_target],
    }
    all_report = _scope_report(
        heldout_target,
        truth,
        predictions,
        target_report_features=target_report_features,
        scale=scale,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        active_only=False,
    )
    active_column = _active_column(heldout_target)
    if active_column is not None and len(heldout_target):
        active_mask = heldout_target[active_column].fillna(False).astype(bool).to_numpy()
    else:
        active_mask = np.zeros(len(heldout_target), dtype=bool)
    active_rows = heldout_target.loc[active_mask].copy()
    active_predictions = {method: predictions[method][active_mask] for method in predictions}
    active_truth = truth[active_mask]
    active_report = _scope_report(
        active_rows,
        active_truth,
        active_predictions,
        target_report_features=target_report_features,
        scale=scale,
        seed=seed + 100,
        bootstrap_samples=bootstrap_samples,
        active_only=True,
    )
    return {"all_windows": all_report, "active_only": active_report}


def split_report(
    frame: pd.DataFrame,
    spec: dict[str, object],
    seed: int,
    *,
    first_hop_features: list[str],
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
    discovery, heldout, multihop_features = _attach_multihop_features(
        discovery,
        heldout,
        first_hop_features=first_hop_features,
        mediator=str(spec["mediator"]),
        mediator_future_suffix=str(spec.get("mediator_future_suffix", spec["future_suffix"])),
        ridge_alpha=ridge_alpha,
        seed=seed,
    )
    targets = {}
    for index, target in enumerate(spec["targets"]):
        target = str(target)
        targets[target] = _target_reports(
            discovery,
            heldout,
            target,
            first_hop_features,
            multihop_features,
            str(spec["future_suffix"]),
            ridge_alpha,
            seed=seed + 1000 * index,
            bootstrap_samples=bootstrap_samples,
        )
    return {
        "seed": int(seed),
        "group_column": group_column,
        "discovery_groups": int(len(discovery_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "overlap": int(len(discovery_groups & heldout_groups)),
        "first_hop_feature_count": int(len(first_hop_features)),
        "multihop_feature_count": int(len(multihop_features)),
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
    frame = pd.read_parquet(data_dir / str(spec["cohort"])).reset_index(drop=True)
    frame = _merge_mediator_future(frame, spec, data_dir=data_dir)
    frame, first_hop_features = _attach_first_hop_features(frame)
    random_reports = [
        split_report(
            frame,
            spec,
            seed=seed,
            first_hop_features=first_hop_features,
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
            seed=2029,
            first_hop_features=first_hop_features,
            discovery_fraction=discovery_fraction,
            ridge_alpha=ridge_alpha,
            bootstrap_samples=bootstrap_samples,
            group_column="hospitalid",
        )
    targets = tuple(str(target) for target in spec["targets"])
    return {
        "name": spec["name"],
        "first_hop": spec["first_hop"],
        "mediator": spec["mediator"],
        "target_system": spec["target_system"],
        "rationale": spec["rationale"],
        "cohort": str(spec["cohort"]),
        "mediator_cohort": str(spec.get("mediator_cohort", spec["cohort"])),
        "future_suffix": spec["future_suffix"],
        "mediator_future_suffix": spec.get("mediator_future_suffix", spec["future_suffix"]),
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()) if len(frame) else 0,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "targets": targets,
        "candidate": "direct_sepsis_renal_plus_map_mediator_features",
        "baseline": "direct_sepsis_renal_ridge",
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
    parser.add_argument("--output", default="eicu_body_system_multihop_coupling_audit.json")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--specs", default="all", help="comma-separated multihop spec names or 'all'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    requested = {item.strip() for item in args.specs.split(",") if item.strip()} if args.specs != "all" else None
    specs = [spec for spec in MULTIHOP_SPECS if requested is None or str(spec["name"]) in requested]
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
        "experiment": "eICU multi-hop body-system coupling audit",
        "gate": {
            "baseline": "direct_first_hop_to_downstream_ridge",
            "candidate": "direct_baseline_plus_discovery_only_mediator_features",
            "placebo": "direct_baseline_plus_capacity_matched_random_features",
            "pass_rule": "candidate must significantly beat both direct baseline and placebo on held-out patients",
        },
        "multihop_specs": results,
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
        spec["name"]: {
            target: spec["random_patient_splits"]["summary"][target]["active_only"]["candidate_passes_both_count"]
            for target in spec["targets"]
        }
        for spec in results
    }
    print(json.dumps({
        "output": args.output,
        "multihop_specs": len(results),
        "active_pass_both_counts": concise,
        "causal_claim_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
