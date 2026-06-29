"""Focused deep coupling audits for the most plausible body-system frontiers.

The broad coupling passes found weak but non-promotable signals.  This script
stops scanning every edge and focuses on the two frontiers that had both
physiologic plausibility and empirical signal:

* renal-electrolyte K/bicarbonate store dynamics at 6h;
* cardio-renal perfusion coupling at 24h and 48h.

The gate remains fail-closed: focused coupling features must beat both a
downstream baseline and a capacity-matched placebo on held-out patients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from body_edge_specific_coupling_belief import focused_coupling_features
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
from eicu_sepsis_target_router import _feature_columns, _round, _subject_column, split_subjects


FOCUSED_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "renal_electrolyte_store_6h",
        "cohort": "eicu_electrolyte_acid_base_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "renal",
        "target_system": "electrolyte_acid_base",
        "targets": ("potassium", "bicarbonate", "anion_gap", "sodium", "phosphate"),
        "rationale": "explicit K-store, bicarbonate-buffer, and sodium-water shared states",
    },
    {
        "name": "cardio_renal_24h",
        "cohort": "eicu_aki_transitions_24h.parquet",
        "future_suffix": "tp24",
        "source_system": "cardiovascular_perfusion",
        "target_system": "renal",
        "targets": ("creatinine", "bun", "urine_output"),
        "rationale": "perfusion shock burden coupled to renal reserve and creatinine kinetics at 24h",
    },
    {
        "name": "cardio_renal_48h",
        "cohort": "eicu_aki_transitions_48h.parquet",
        "future_suffix": "tp48",
        "source_system": "cardiovascular_perfusion",
        "target_system": "renal",
        "targets": ("creatinine", "bun", "urine_output"),
        "rationale": "perfusion shock burden coupled to renal reserve and creatinine kinetics at 48h",
    },
)


def _attach_features(frame: pd.DataFrame, spec_name: str) -> tuple[pd.DataFrame, list[str]]:
    features = focused_coupling_features(frame, spec_name)
    columns = list(features.columns)
    return pd.concat([frame.reset_index(drop=True), features.reset_index(drop=True)], axis=1), columns


def _predict_methods(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    future_suffix: str,
    baseline_features: list[str],
    focused_features: list[str],
    ridge_alpha: float,
    seed: int,
) -> dict[str, object]:
    candidate_features = sorted(set(baseline_features + focused_features))
    placebo_train, placebo_columns = _with_placebo(train, focused_features, seed=seed + 501)
    placebo_heldout, _placebo_columns = _with_placebo(heldout, focused_features, seed=seed + 601)
    return {
        "baseline_ridge": _fit_ridge(train, heldout, target, baseline_features, ridge_alpha, future_suffix),
        "focused_coupled_ridge": _fit_ridge(train, heldout, target, candidate_features, ridge_alpha, future_suffix),
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
    focused_features: list[str],
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
    bootstrap_samples: int,
    active_only: bool,
) -> dict[str, object]:
    reports = _target_reports(
        train,
        heldout,
        target,
        focused_features,
        future_suffix,
        ridge_alpha,
        seed,
        bootstrap_samples,
    )
    return reports["active_only" if active_only else "all_windows"]


def _build_scope_report(
    rows,
    truth,
    predictions,
    *,
    target_report_features: dict[str, object],
    scale: float,
    seed: int,
    bootstrap_samples: int,
    active_only: bool,
) -> dict[str, object]:
    candidate = predictions["focused_coupled_ridge"]
    baseline = predictions["baseline_ridge"]
    placebo = predictions["placebo_ridge"]
    return {
        "rows": int(len(rows)),
        "subjects": int(rows[_subject_column(rows)].nunique()) if len(rows) else 0,
        "active_only": bool(active_only),
        **target_report_features,
        "mae": {
            method: _round(_mae(truth, predictions[method], scale=scale))
            for method in ("baseline_ridge", "focused_coupled_ridge", "placebo_ridge")
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
    focused_features: list[str],
    future_suffix: str,
    ridge_alpha: float,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, dict[str, object]]:
    train_target = _target_rows(train, target, future_suffix)
    heldout_target = _target_rows(heldout, target, future_suffix)
    focused_features = [column for column in focused_features if column in train_target and column in heldout_target]
    baseline_features = [column for column in _feature_columns(train_target, target) if column not in set(focused_features)]
    future = f"{target}_{future_suffix}"
    truth = heldout_target[future].to_numpy(dtype="float64") if future in heldout_target else []
    scale = _target_scale(train_target, target, future_suffix) if len(train_target) else 1.0
    predictions = _predict_methods(
        train_target,
        heldout_target,
        target,
        future_suffix,
        baseline_features,
        focused_features,
        ridge_alpha,
        seed,
    )
    target_report_features = {
        "feature_counts": {
            "baseline": int(len(baseline_features)),
            "focused_coupling": int(len(focused_features)),
            "candidate": int(len(set(baseline_features + focused_features))),
            "placebo": int(len(focused_features)),
        },
        "focused_coupling_features": focused_features,
    }
    all_report = _build_scope_report(
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
        active_mask = [False] * len(heldout_target)
    active_rows = heldout_target.loc[active_mask].copy()
    active_predictions = {
        method: predictions[method][active_mask]
        for method in ("baseline_ridge", "focused_coupled_ridge", "placebo_ridge")
    }
    active_truth = truth[active_mask]
    active_report = _build_scope_report(
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
    focused_features: list[str],
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
        targets[target] = _target_reports(
                discovery,
                heldout,
                target,
                focused_features,
                future_suffix,
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
        "focused_coupling_feature_count": int(len(focused_features)),
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
    frame, focused_features = _attach_features(frame, str(spec["name"]))
    random_reports = [
        split_report(
            frame,
            spec,
            seed=seed,
            focused_features=focused_features,
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
            focused_features=focused_features,
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
        "candidate": "focused_deep_coupling_features",
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
    parser.add_argument("--output", default="eicu_body_system_focused_coupling_audit.json")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--specs", default="all", help="comma-separated focused spec names or 'all'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    requested = {item.strip() for item in args.specs.split(",") if item.strip()} if args.specs != "all" else None
    specs = [spec for spec in FOCUSED_SPECS if requested is None or str(spec["name"]) in requested]
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
        "experiment": "eICU focused body-system deep coupling audit",
        "gate": {
            "candidate": "focused_deep_coupling_features",
            "baseline": "ridge_without_focused_coupling_features",
            "placebo": "ridge_plus_capacity_matched_random_features",
            "pass_rule": "candidate must significantly beat both baseline and placebo on held-out patients",
        },
        "focused_specs": results,
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
        "focused_specs": len(results),
        "active_pass_both_counts": concise,
        "causal_claim_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
