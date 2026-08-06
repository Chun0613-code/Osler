"""Audit cross-system factual coupling edges in eICU transition cohorts.

This is the first depth pass after body-system breadth completion.  It tests
whether upstream organ-system state improves downstream factual prediction
beyond:

* a baseline ridge model with the upstream system features removed;
* a capacity-matched placebo with the same number of random columns.

The output is aggregate-only.  Passing an edge means "the upstream state carries
held-out factual predictive signal."  It does not mean causal influence,
counterfactual validity, clinical recommendation authority, or runtime action
authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_body_system_target_router import _fit_ridge, _future_column
from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _round, _subject_column, split_subjects


SEEDS = (7, 11, 19, 23, 37, 53, 71)
METHODS = ("baseline_ridge", "coupled_ridge", "placebo_ridge")


COUPLING_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "renal_to_electrolyte_acid_base",
        "cohort": "eicu_electrolyte_acid_base_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "renal",
        "target_system": "electrolyte_acid_base",
        "targets": ("potassium", "bicarbonate", "phosphate", "sodium", "anion_gap"),
        "upstream_features": (
            "creatinine_t",
            "creatinine_age_hr",
            "bun_t",
            "bun_age_hr",
            "urine_output_t",
            "urine_output_age_hr",
            "hist_renal_replacement*",
            "act_renal_replacement*",
            "hist_diuretics*",
            "act_diuretics*",
        ),
        "rationale": "renal filtration and renal-support state should help electrolyte and acid-base trajectories",
    },
    {
        "name": "respiratory_to_acid_base",
        "cohort": "eicu_respiratory_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "respiratory",
        "target_system": "acid_base",
        "targets": ("ph", "bicarbonate", "lactate"),
        "upstream_features": (
            "o2sat_t",
            "o2sat_age_hr",
            "respiratory_rate_t",
            "respiratory_rate_age_hr",
            "hist_ventilation*",
            "act_ventilation*",
            "hist_bronchodilator*",
            "act_bronchodilator*",
            "hist_systemic_steroid*",
            "act_systemic_steroid*",
        ),
        "rationale": "ventilation and oxygenation state should help acid-base and lactate trajectories",
    },
    {
        "name": "endocrine_to_electrolyte",
        "cohort": "eicu_endocrine_stress_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "endocrine_metabolic",
        "target_system": "electrolyte_acid_base",
        "targets": ("potassium", "sodium", "bicarbonate", "anion_gap"),
        "upstream_features": (
            "glucose_t",
            "glucose_age_hr",
            "serum_osmolality_t",
            "serum_osmolality_age_hr",
            "serum_ketones_t",
            "serum_ketones_age_hr",
            "hist_insulin*",
            "act_insulin*",
            "hist_dextrose*",
            "act_dextrose*",
            "hist_systemic_steroid*",
            "act_systemic_steroid*",
        ),
        "rationale": "glycemic/osmotic state and insulin/dextrose evidence should help electrolyte trajectories",
    },
    {
        "name": "heme_to_perfusion",
        "cohort": "eicu_coagulopathy_heme_transitions_24h.parquet",
        "future_suffix": "tp24",
        "source_system": "hematologic_coagulation",
        "target_system": "cardiovascular_perfusion",
        "targets": ("map", "lactate", "heart_rate"),
        "upstream_features": (
            "hemoglobin_t",
            "hemoglobin_age_hr",
            "hematocrit_t",
            "hematocrit_age_hr",
            "platelets_t",
            "platelets_age_hr",
            "inr_t",
            "inr_age_hr",
            "ptt_t",
            "ptt_age_hr",
            "fibrinogen_t",
            "fibrinogen_age_hr",
            "hist_transfusion*",
            "act_transfusion*",
        ),
        "rationale": "oxygen-carrying capacity, coagulation, and transfusion evidence should help perfusion targets",
    },
    {
        "name": "cardiovascular_to_renal",
        "cohort": "eicu_cardiovascular_instability_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "cardiovascular_perfusion",
        "target_system": "renal",
        "targets": ("creatinine", "urine_output"),
        "upstream_features": (
            "map_t",
            "map_age_hr",
            "heart_rate_t",
            "heart_rate_age_hr",
            "lactate_t",
            "lactate_age_hr",
            "hist_vasopressor*",
            "act_vasopressor*",
            "hist_inotrope*",
            "act_inotrope*",
            "hist_fluids*",
            "act_fluids*",
        ),
        "rationale": "perfusion and hemodynamic support should help short-horizon renal output markers",
    },
    {
        "name": "hepatic_to_coagulation_platelets",
        "cohort": "eicu_hepatic_failure_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "hepatic",
        "target_system": "hematologic_coagulation",
        "targets": ("platelets", "bicarbonate", "creatinine"),
        "upstream_features": (
            "bilirubin_t",
            "bilirubin_age_hr",
            "bilirubin_direct_t",
            "bilirubin_direct_age_hr",
            "hist_hepatic_encephalopathy_tx*",
            "act_hepatic_encephalopathy_tx*",
            "hist_renal_replacement*",
            "act_renal_replacement*",
        ),
        "rationale": "hepatic failure state should help downstream platelet/coagulation-proxy and renal-acid-base targets",
    },
    {
        "name": "immune_to_hemodynamics",
        "cohort": "eicu_immune_inflammatory_transitions_6h.parquet",
        "future_suffix": "tp6",
        "source_system": "immune_inflammatory",
        "target_system": "cardiovascular_perfusion",
        "targets": ("map", "lactate", "platelets", "albumin"),
        "upstream_features": (
            "wbc_t",
            "wbc_age_hr",
            "temperature_t",
            "temperature_age_hr",
            "crp_t",
            "crp_age_hr",
            "crp_hs_t",
            "crp_hs_age_hr",
            "esr_t",
            "esr_age_hr",
            "ferritin_t",
            "ferritin_age_hr",
            "hist_antibiotics*",
            "act_antibiotics*",
            "hist_systemic_steroid*",
            "act_systemic_steroid*",
        ),
        "rationale": "inflammatory state should help hemodynamic, platelet, and albumin trajectories",
    },
)


def _target_rows(frame: pd.DataFrame, target: str, future_suffix: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = _future_column(target, future_suffix)
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def _active_column(frame: pd.DataFrame) -> str | None:
    columns = [column for column in frame.columns if column.endswith("_active_t")]
    return columns[0] if columns else None


def _resolve_upstream_columns(frame: pd.DataFrame, target: str, patterns: tuple[str, ...]) -> list[str]:
    features = set(_feature_columns(frame, target))
    columns: set[str] = set()
    for pattern in patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            columns.update(column for column in features if column.startswith(prefix))
        elif pattern in features:
            columns.add(pattern)
    return sorted(columns)


def _target_scale(discovery: pd.DataFrame, target: str, future_suffix: str) -> float:
    future = pd.to_numeric(discovery[_future_column(target, future_suffix)], errors="coerce")
    std = float(future.std(skipna=True))
    if not np.isfinite(std) or std < 1e-6:
        current = pd.to_numeric(discovery[f"{target}_t"], errors="coerce")
        std = float(current.std(skipna=True))
    return max(std, 1.0) if np.isfinite(std) else 1.0


def _with_placebo(frame: pd.DataFrame, columns: list[str], seed: int) -> tuple[pd.DataFrame, list[str]]:
    placebo_columns = [f"placebo_coupling_{index}" for index in range(len(columns))]
    output = frame.copy()
    if not placebo_columns:
        return output, placebo_columns
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, size=(len(output), len(placebo_columns)))
    for index, column in enumerate(placebo_columns):
        output[column] = values[:, index]
    return output, placebo_columns


def _predict_methods(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    target: str,
    future_suffix: str,
    baseline_features: list[str],
    upstream_features: list[str],
    ridge_alpha: float,
    seed: int,
) -> dict[str, np.ndarray]:
    coupled_features = sorted(set(baseline_features + upstream_features))
    placebo_train, placebo_columns = _with_placebo(train, upstream_features, seed=seed + 101)
    placebo_heldout, _placebo_columns = _with_placebo(heldout, upstream_features, seed=seed + 202)
    return {
        "baseline_ridge": _fit_ridge(train, heldout, target, baseline_features, ridge_alpha, future_suffix),
        "coupled_ridge": _fit_ridge(train, heldout, target, coupled_features, ridge_alpha, future_suffix),
        "placebo_ridge": _fit_ridge(
            placebo_train,
            placebo_heldout,
            target,
            sorted(set(baseline_features + placebo_columns)),
            ridge_alpha,
            future_suffix,
        ),
    }


def _mae(truth: np.ndarray, pred: np.ndarray, scale: float | None = None) -> float | None:
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not mask.any():
        return None
    error = np.abs(pred[mask] - truth[mask])
    if scale is not None:
        error = error / max(float(scale), 1e-6)
    return float(error.mean())


def _delta_by_subject(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray, scale: float) -> pd.Series:
    mask = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(baseline)
    if not mask.any():
        return pd.Series(dtype="float64")
    subject_column = _subject_column(rows)
    temp = rows.loc[mask, [subject_column]].copy()
    temp["delta"] = (
        np.abs(pred[mask] - truth[mask]) - np.abs(baseline[mask] - truth[mask])
    ) / max(float(scale), 1e-6)
    return temp.groupby(subject_column)["delta"].mean()


def _delta_summary(
    rows: pd.DataFrame,
    truth: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    scale: float,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    deltas = _delta_by_subject(rows, truth, pred, baseline, scale)
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
    upstream_patterns: tuple[str, ...],
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
    base_features = _feature_columns(train_target, target)
    upstream_features = _resolve_upstream_columns(train_target, target, upstream_patterns)
    baseline_features = [column for column in base_features if column not in set(upstream_features)]
    future = _future_column(target, future_suffix)
    truth = heldout_target[future].to_numpy(dtype=np.float64) if future in heldout_target else np.asarray([])
    scale = _target_scale(train_target, target, future_suffix) if len(train_target) else 1.0
    predictions = _predict_methods(
        train_target,
        heldout_target,
        target,
        future_suffix,
        baseline_features,
        upstream_features,
        ridge_alpha,
        seed,
    )
    coupled = predictions["coupled_ridge"]
    baseline = predictions["baseline_ridge"]
    placebo = predictions["placebo_ridge"]
    return {
        "rows": int(len(heldout_target)),
        "subjects": int(heldout_target[_subject_column(heldout_target)].nunique()) if len(heldout_target) else 0,
        "active_only": bool(active_only),
        "feature_counts": {
            "baseline": int(len(baseline_features)),
            "upstream": int(len(upstream_features)),
            "coupled": int(len(set(baseline_features + upstream_features))),
            "placebo": int(len(upstream_features)),
        },
        "upstream_features": upstream_features,
        "mae": {
            method: _round(_mae(truth, predictions[method], scale=scale))
            for method in METHODS
        },
        "candidate_vs_baseline": _delta_summary(
            heldout_target,
            truth,
            coupled,
            baseline,
            scale,
            seed=seed + 17,
            bootstrap_samples=bootstrap_samples,
        ),
        "candidate_vs_placebo": _delta_summary(
            heldout_target,
            truth,
            coupled,
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
    upstream_patterns = tuple(spec["upstream_features"])
    targets = {}
    for index, target in enumerate(spec["targets"]):
        target = str(target)
        targets[target] = {
            "all_windows": _target_report(
                discovery,
                heldout,
                target,
                upstream_patterns,
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
                upstream_patterns,
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
        "targets": targets,
    }


def _passes_both(report: dict[str, object]) -> bool:
    return bool(
        report["candidate_vs_baseline"]["significant"]
        and report["candidate_vs_placebo"]["significant"]
    )


def summarize_reports(reports: list[dict[str, object]], targets: tuple[str, ...]) -> dict[str, object]:
    output: dict[str, object] = {}
    for target in targets:
        target_summary: dict[str, object] = {}
        for scope in ("all_windows", "active_only"):
            base_counts = []
            placebo_counts = []
            pass_counts = []
            deltas_baseline = []
            deltas_placebo = []
            for report in reports:
                target_report = report["targets"][target][scope]
                base = target_report["candidate_vs_baseline"]
                placebo = target_report["candidate_vs_placebo"]
                base_counts.append(bool(base["significant"]))
                placebo_counts.append(bool(placebo["significant"]))
                pass_counts.append(_passes_both(target_report))
                if base["point_delta"] is not None:
                    deltas_baseline.append(float(base["point_delta"]))
                if placebo["point_delta"] is not None:
                    deltas_placebo.append(float(placebo["point_delta"]))
            target_summary[scope] = {
                "candidate_beats_baseline_significant_count": int(sum(base_counts)),
                "candidate_beats_placebo_significant_count": int(sum(placebo_counts)),
                "candidate_passes_both_count": int(sum(pass_counts)),
                "median_delta_vs_baseline": _round(float(np.median(deltas_baseline))) if deltas_baseline else None,
                "median_delta_vs_placebo": _round(float(np.median(deltas_placebo))) if deltas_placebo else None,
            }
        output[target] = target_summary
    return output


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
    random_reports = [
        split_report(
            frame,
            spec,
            seed=seed,
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
            seed=991,
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
        "upstream_feature_patterns": spec["upstream_features"],
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
    parser.add_argument("--output", default="eicu_body_system_coupling_audit.json")
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
        "experiment": "eICU body-system factual coupling audit",
        "gate": {
            "candidate": "baseline_ridge_plus_upstream_system_features",
            "baseline": "ridge_without_upstream_system_features",
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
