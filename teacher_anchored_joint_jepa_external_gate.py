"""External holdout gate for the teacher-anchored hierarchical JEPA adapter.

The bounded seven-seed audit identifies target/horizon cells worth checking.
This script retrains the same candidate on hospital, care-unit, and forward-
time splits.  It is research-only and never writes a runtime registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from teacher_anchored_joint_jepa_audit import (
    TeacherAnchoredResidualAdapter,
    _cache_base_features,
    _conformal_scale_features,
    _crossfit_normalized_conformal,
    _evaluate,
    _evaluate_conformal,
    _evaluate_route_ablation,
    _patient_cluster_conformal_quantile,
    _patient_coverage_diagnostics,
    _predict_adapter,
    _select_targets,
    _train_adapter,
    _train_distributional_scale_heads,
)
from whole_body_joint_gate import (
    _holdout_available,
    _module_balanced_metrics,
    _split_by_column,
    _split_forward_time,
    _split_known_groups,
    _split_train_calibration,
    _validated_router_prediction,
)
from whole_body_joint_jepa import fit_joint, load_joint_event_examples


def _split_train_calibration_with_fraction(
    frame,
    train_rows,
    seed,
    fraction,
):
    """Create a patient-disjoint calibration pool inside training domains."""

    fraction = float(fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    train_rows = np.asarray(train_rows, dtype=np.int64)
    train_frame = frame.iloc[train_rows].reset_index(drop=True)
    fit_local, calibration_local = _split_by_column(
        train_frame, "subject_id", seed, fraction
    )
    return train_rows[fit_local], train_rows[calibration_local]


def _fit_base(frame, variables, module_names, fit_rows, *, seed, args):
    return fit_joint(
        frame,
        variables,
        module_names,
        fit_rows,
        seed=seed,
        epochs=args.base_epochs,
        batch_size=args.batch_size,
        measurement_process_mode="observed",
        cross_forecast_weight=1.0,
        masked_forecast_probability=0.15,
        module_mask_probability=0.15,
        world_model=True,
        world_model_weight=1.0,
        target_momentum=0.99,
        future_target_mode="window",
        target_min_observations=2,
        regime_balanced_loss=True,
        regime_count=3,
        state_normalization="robust",
        hospital_invariance_weight=args.domain_invariance_weight,
        domain_invariance_columns=tuple(
            value.strip()
            for value in args.domain_invariance_columns.split(",")
            if value.strip()
        ),
        target_encoder_mode="fast",
        target_horizon_regime_adapter=True,
        patient_residual_adapter=True,
        future_task_balanced_sampling=True,
        uncertainty_gate=True,
        validated_edge_adapters=True,
        validated_edge_weight=0.1,
        target_head_stage_epochs=0,
        measurement_time_head=True,
        measurement_time_weight=0.1,
        uncertainty_calibration_weight=0.05,
        include_treatment_context=True,
    )


def _direct_conformal(
    frame,
    arrays,
    calibration_rows,
    test_rows,
    target_names,
    target_indices,
    calibration_prediction,
    test_prediction,
    *,
    seed,
    method="fixed",
    nominal_coverage=0.90,
):
    scaler, _, _, _, _, future, future_mask, _, horizon = arrays[:9]
    future_raw = future * scaler.scales + scaler.medians
    output = {}
    for local_index, (target, target_index) in enumerate(
        zip(target_names, target_indices)
    ):
        common_horizons = sorted(
            set(horizon[calibration_rows]).intersection(horizon[test_rows])
        )
        for horizon_value in common_horizons:
            cal_mask = (
                (horizon[calibration_rows] == float(horizon_value))
                & np.asarray(
                    future_mask[calibration_rows, target_index], dtype=bool
                )
            )
            test_mask = (
                (horizon[test_rows] == float(horizon_value))
                & np.asarray(future_mask[test_rows, target_index], dtype=bool)
            )
            if int(cal_mask.sum()) < 30 or int(test_mask.sum()) < 30:
                continue
            cal_rows = calibration_rows[cal_mask]
            heldout_rows = test_rows[test_mask]
            cal_raw = (
                calibration_prediction[cal_mask, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            test_raw = (
                test_prediction[test_mask, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            residual = np.abs(
                cal_raw - future_raw[cal_rows, target_index]
            )
            calibration_seed = (
                seed
                + target_index * 10_007
                + int(round(float(horizon_value) * 101))
            )
            if method == "normalized":
                calibration_features = _conformal_scale_features(
                    arrays,
                    cal_rows,
                    target_index,
                    calibration_prediction[cal_mask, local_index],
                )
                test_features = _conformal_scale_features(
                    arrays,
                    heldout_rows,
                    target_index,
                    test_prediction[test_mask, local_index],
                )
                widths, calibration = _crossfit_normalized_conformal(
                    frame,
                    cal_rows,
                    residual,
                    calibration_features,
                    test_features,
                    seed=calibration_seed,
                    coverage=nominal_coverage,
                )
            elif method == "adaptive":
                from teacher_anchored_joint_jepa_audit import (
                    _adaptive_conformal_quantile,
                )

                quantile, rank, calibration = (
                    _adaptive_conformal_quantile(
                        frame,
                        cal_rows,
                        residual,
                        seed=calibration_seed,
                        target_coverage=nominal_coverage,
                    )
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration["q90_rank"] = int(rank)
            elif method == "patient_cluster":
                quantile, rank, calibration = (
                    _patient_cluster_conformal_quantile(
                        frame,
                        cal_rows,
                        residual,
                        coverage=nominal_coverage,
                    )
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration["q90_rank"] = int(rank)
            elif method == "fixed":
                from whole_body_joint_gate import (
                    _finite_sample_conformal_quantile,
                )

                quantile, rank = _finite_sample_conformal_quantile(
                    residual, coverage=nominal_coverage
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration = {
                    "method": "fixed_split_conformal",
                    "selected_nominal_coverage": float(
                        nominal_coverage
                    ),
                    "target_empirical_coverage": 0.90,
                    "calibration_rows": int(len(residual)),
                    "q90_rank": int(rank),
                    "test_labels_used": False,
                }
            else:
                raise ValueError(f"Unknown conformal method: {method}")
            observed = future_raw[heldout_rows, target_index]
            covered = (observed >= test_raw - widths) & (
                observed <= test_raw + widths
            )
            coverage = _patient_coverage_diagnostics(
                frame,
                heldout_rows,
                covered,
                seed=calibration_seed + 503,
            )
            key = (
                f"{target}@"
                f"{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            )
            output[key] = {
                "calibration_n": int(cal_mask.sum()),
                "test_n": int(test_mask.sum()),
                "q90": float(np.median(widths)),
                "q90_rank": int(calibration["q90_rank"]),
                "coverage": coverage["patient_equalized_coverage"],
                "row_coverage": coverage["row_coverage"],
                "patient_equalized_coverage": coverage[
                    "patient_equalized_coverage"
                ],
                "patient_coverage_ci95": coverage[
                    "patient_coverage_ci95"
                ],
                "test_subjects": coverage["test_subjects"],
                "pass": bool(
                    coverage["test_subjects"] >= 20
                    and 0.87
                    <= coverage["patient_equalized_coverage"]
                    <= 0.93
                ),
                "finite_sample_corrected": True,
                "adaptive_calibration": calibration,
                "calibration_rows_untouched_during_training": True,
                "test_labels_used_for_calibration": False,
            }
    return output


def _run_external_split(
    frame,
    variables,
    module_names,
    target_names,
    target_indices,
    train_rows,
    test_rows,
    *,
    seed,
    args,
):
    fit_rows, calibration_rows = _split_train_calibration_with_fraction(
        frame,
        np.asarray(train_rows, dtype=np.int64),
        seed + 1001,
        args.calibration_fraction,
    )
    model, arrays, losses = _fit_base(
        frame,
        variables,
        module_names,
        fit_rows,
        seed=seed,
        args=args,
    )
    teacher_fit, teacher_selection = _split_train_calibration(
        frame, fit_rows, seed + 9001
    )
    all_rows = np.arange(len(frame), dtype=np.int64)
    teacher, teacher_metadata = _validated_router_prediction(
        frame,
        variables,
        module_names,
        arrays,
        teacher_fit,
        teacher_selection,
        all_rows,
        include_treatment_context=True,
        seed=seed + 30_000,
    )
    features = _cache_base_features(
        model,
        arrays,
        target_indices,
        all_rows,
        batch_size=args.batch_size,
    )
    membership = (
        model.variable_module_membership[:, list(target_indices)]
        .detach()
        .cpu()
        .numpy()
        .T
    )
    sparse_experts = args.variant in {
        "sparse_experts",
        "sparse_distributional",
        "sparse_target_distributional",
    }
    distributional_scale = args.variant in {
        "sparse_distributional",
        "sparse_target_distributional",
    }
    target_specific_projections = (
        args.variant == "sparse_target_distributional"
    )
    variant_seed_offset = {
        "hierarchical": 101,
        "sparse_experts": 151,
        "sparse_distributional": 151,
        "sparse_target_distributional": 181,
    }[args.variant]
    torch.manual_seed(seed + variant_seed_offset)
    adapter = TeacherAnchoredResidualAdapter(
        target_indices=target_indices,
        target_module_membership=membership,
        latent_dim=model.latent,
        organ_dim=model.token_dim,
        module_count=len(module_names),
        dynamic_organ_route=False,
        sparse_experts=sparse_experts,
        distributional_scale=distributional_scale,
        target_specific_projections=target_specific_projections,
    )
    training = _train_adapter(
        adapter,
        features,
        arrays,
        teacher,
        fit_rows,
        target_indices,
        epochs=args.adapter_epochs,
        batch_size=args.batch_size,
        seed=seed + 40_000,
        pcgrad=False,
        route_entropy_weight=args.route_entropy_weight,
        residual_l1_weight=args.residual_l1_weight,
        expert_entropy_weight=args.expert_entropy_weight,
        distributional_weight=0.0,
    )
    scale_training = _train_distributional_scale_heads(
        adapter,
        features,
        arrays,
        teacher,
        fit_rows,
        target_indices,
        epochs=args.scale_epochs if distributional_scale else 0,
        batch_size=args.batch_size,
        seed=seed + 45_000,
    )
    calibration_output = _predict_adapter(
        adapter,
        features,
        arrays,
        teacher,
        calibration_rows,
        target_indices,
        batch_size=args.batch_size,
    )
    test_output = _predict_adapter(
        adapter,
        features,
        arrays,
        teacher,
        test_rows,
        target_indices,
        batch_size=args.batch_size,
    )
    no_expert_output = _predict_adapter(
        adapter,
        features,
        arrays,
        teacher,
        test_rows,
        target_indices,
        batch_size=args.batch_size,
        disable_sparse_experts=True,
    )
    calibration_prediction = calibration_output["value"]
    test_prediction = test_output["value"]
    values = _evaluate(
        frame,
        arrays,
        test_rows,
        target_names,
        target_indices,
        test_prediction,
        teacher,
        seed=seed,
    )
    no_expert_ablation = _evaluate_route_ablation(
        frame,
        arrays,
        test_rows,
        target_names,
        target_indices,
        test_prediction,
        no_expert_output["value"],
        seed=seed + 700_000,
        baseline_label="no_expert",
    )
    conformal = _evaluate_conformal(
        frame,
        arrays,
        calibration_rows,
        test_rows,
        target_names,
        target_indices,
        calibration_prediction,
        test_prediction,
        seed=seed,
        method=args.conformal_method,
        nominal_coverage=args.conformal_nominal_coverage,
        calibration_scale=calibration_output["scale"],
        test_scale=test_output["scale"],
    )
    current = arrays[2]
    full_prediction = current[test_rows].copy()
    full_prediction[:, list(target_indices)] = test_prediction
    module_balanced = _module_balanced_metrics(
        frame,
        variables,
        arrays,
        full_prediction,
        test_rows,
        module_names,
    )
    selected_sources = {}
    for metadata in teacher_metadata.values():
        source = str(metadata.get("selected_source", "unknown"))
        selected_sources[source] = selected_sources.get(source, 0) + 1
    return {
        "fit_rows": int(len(fit_rows)),
        "calibration_rows": int(len(calibration_rows)),
        "test_rows": int(len(test_rows)),
        "base_loss": float(losses[-1]),
        "adapter_loss_history": training["loss_history"],
        "scale_loss_history": scale_training["loss_history"],
        "scale_point_predictor_frozen": scale_training[
            "point_predictor_frozen"
        ],
        "teacher_source_counts": selected_sources,
        "values": values,
        "conformal": conformal,
        "no_expert_ablation": no_expert_ablation,
        "module_balanced": module_balanced,
    }


def _patient_candidates(bounded, variant="hierarchical"):
    return {
        key
        for key, value in bounded["aggregate"][variant].items()
        if value.get("evaluated_splits") == 7
        and value.get("beats_teacher_splits") == 7
        and value.get("beats_persistence_splits") == 7
        and value.get("conformal_pass_splits") == 7
        and (
            not variant.startswith("sparse_")
            or value.get("beats_no_expert_splits") == 7
        )
    }


def _external_pass(report, key, *, require_no_expert):
    value = report.get("values", {}).get(key, {})
    no_expert = (
        report.get("no_expert_ablation", {})
        .get(key, {})
        .get("no_expert_bootstrap", {})
        .get("pass", False)
    )
    return {
        "teacher_bootstrap": bool(
            value.get("teacher_bootstrap", {}).get("pass", False)
        ),
        "persistence_bootstrap": bool(
            value.get("persistence_bootstrap", {}).get("pass", False)
        ),
        "conformal": bool(
            report.get("conformal", {}).get(key, {}).get("pass", False)
        ),
        "module_balanced": bool(
            report.get("module_balanced", {}).get(key, {}).get("pass", False)
        ),
        **(
            {"no_expert_bootstrap": bool(no_expert)}
            if require_no_expert
            else {}
        ),
    }


def run(args):
    modules = tuple(
        value.strip() for value in args.modules.split(",") if value.strip()
    )
    horizons = tuple(
        int(value) for value in args.horizons.split(",") if value.strip()
    )
    frame, variables, module_names = load_joint_event_examples(
        args.event_examples,
        modules,
        horizons,
        args.max_stays,
        min_modules=args.min_modules,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
        history_lags_hours=tuple(
            float(value)
            for value in args.history_lags_hours.split(",")
            if value.strip()
        ),
    )
    target_names = _select_targets(args.targets, variables)
    if not target_names:
        raise ValueError("No requested target is available")
    target_indices = tuple(variables.index(target) for target in target_names)
    bounded = json.loads(args.bounded_report.read_text())
    patient_candidates = _patient_candidates(bounded, args.variant)
    progress_path = Path(f"{args.output}.progress.json")
    progress_config = {
        "event_examples": str(args.event_examples.resolve()),
        "bounded_report": str(args.bounded_report.resolve()),
        "rows": int(len(frame)),
        "targets": list(target_names),
        "horizons": list(horizons),
        "base_epochs": int(args.base_epochs),
        "adapter_epochs": int(args.adapter_epochs),
        "scale_epochs": int(args.scale_epochs),
        "variant": str(args.variant),
        "conformal_method": str(args.conformal_method),
        "conformal_nominal_coverage": float(
            args.conformal_nominal_coverage
        ),
        "calibration_fraction": float(args.calibration_fraction),
        "domain_invariance_weight": float(args.domain_invariance_weight),
        "domain_invariance_columns": [
            value.strip()
            for value in args.domain_invariance_columns.split(",")
            if value.strip()
        ],
    }
    completed = {}
    if getattr(args, "resume", False) and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config") != progress_config:
            raise ValueError(
                "Existing external-gate progress does not match this config"
            )
        completed = dict(progress.get("splits", {}))

    def checkpoint(name, value):
        completed[name] = value
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "schema": (
                        "teacher_anchored_joint_jepa_external_progress.v1"
                    ),
                    "status": "in_progress",
                    "config": progress_config,
                    "completed_splits": sorted(completed),
                    "splits": completed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    hospital = completed.get("hospital")
    if hospital is None:
        hospital_train, hospital_test = _split_by_column(
            frame, "hospitalid", 2026
        )
        hospital = _run_external_split(
            frame,
            variables,
            module_names,
            target_names,
            target_indices,
            hospital_train,
            hospital_test,
            seed=2026,
            args=args,
        )
        checkpoint("hospital", hospital)

    care_unit = completed.get("care_unit")
    if care_unit is None:
        if _holdout_available(frame, "careunit"):
            care_train, care_test = _split_known_groups(
                frame, "careunit", 2027
            )
            care_unit = _run_external_split(
                frame,
                variables,
                module_names,
                target_names,
                target_indices,
                care_train,
                care_test,
                seed=2027,
                args=args,
            )
            care_unit["available"] = True
            care_unit["heldout_groups"] = sorted(
                frame.iloc[care_test]["careunit"].astype(str).unique().tolist()
            )
        else:
            care_unit = {
                "available": False,
                "reason": "careunit_provenance_unavailable",
            }
        checkpoint("care_unit", care_unit)

    time = completed.get("time")
    if time is None:
        time_train, time_test = _split_forward_time(frame)
        time = _run_external_split(
            frame,
            variables,
            module_names,
            target_names,
            target_indices,
            time_train,
            time_test,
            seed=2028,
            args=args,
        )
        time["split_policy"] = (
            "latest 25% unique anchor times within each stay; "
            "earlier anchors train"
        )
        checkpoint("time", time)

    registry = {}
    require_no_expert = args.variant.startswith("sparse_")
    for key in sorted(patient_candidates):
        requirements = {
            "patient_7seed_teacher_and_persistence": True,
            **{
                f"hospital_{name}": value
                for name, value in _external_pass(
                    hospital, key, require_no_expert=require_no_expert
                ).items()
            },
            **{
                f"care_unit_{name}": value
                for name, value in (
                    _external_pass(
                        care_unit,
                        key,
                        require_no_expert=require_no_expert,
                    ).items()
                    if care_unit.get("available")
                    else {
                        "teacher_bootstrap": False,
                        "persistence_bootstrap": False,
                        "conformal": False,
                        "module_balanced": False,
                        **(
                            {"no_expert_bootstrap": False}
                            if require_no_expert
                            else {}
                        ),
                    }.items()
                )
            },
            **{
                f"time_{name}": value
                for name, value in _external_pass(
                    time, key, require_no_expert=require_no_expert
                ).items()
            },
        }
        registry[key] = {
            "requirements": requirements,
            "passed_external_factual_gate": bool(all(requirements.values())),
            "validated_for_runtime": False,
            "runtime_rejection_reasons": [
                "direction_delta_and_svd_scoring_pending"
            ],
            "cross_organ_claim_allowed": False,
        }
    return {
        "schema": "teacher_anchored_joint_jepa_external_gate.v1",
        "status": "research_only",
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "targets": list(target_names),
        "target_mode": (
            "all_available"
            if args.targets.strip().lower() == "all"
            else "explicit"
        ),
        "two_level_latent_hierarchy": True,
        "patient_candidate_cells": sorted(patient_candidates),
        "hospital": hospital,
        "care_unit": care_unit,
        "time": time,
        "registry": registry,
        "external_gate_passed_cells": sorted(
            key
            for key, value in registry.items()
            if value["passed_external_factual_gate"]
        ),
        "promotion_status": "candidate_only",
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--bounded-report", type=Path, required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument(
        "--targets",
        default="glucose,bicarbonate,creatinine,potassium,map,heart_rate",
        help=(
            "Comma-separated targets, or 'all' to reproduce the bounded "
            "two-level whole-body candidate on every canonical variable."
        ),
    )
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--history-lags-hours", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=500)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument(
        "--variant",
        choices=(
            "hierarchical",
            "sparse_experts",
            "sparse_distributional",
            "sparse_target_distributional",
        ),
        default="hierarchical",
    )
    parser.add_argument("--base-epochs", type=int, default=2)
    parser.add_argument("--adapter-epochs", type=int, default=3)
    parser.add_argument("--scale-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--route-entropy-weight", type=float, default=0.002)
    parser.add_argument("--expert-entropy-weight", type=float, default=0.0005)
    parser.add_argument("--residual-l1-weight", type=float, default=0.01)
    parser.add_argument(
        "--domain-invariance-weight",
        type=float,
        default=0.05,
        help="Training-only adversarial provenance-domain loss weight.",
    )
    parser.add_argument(
        "--domain-invariance-columns",
        default="hospitalid",
        help=(
            "Comma-separated provenance columns removed adversarially from "
            "the physiology latent."
        ),
    )
    parser.add_argument(
        "--conformal-method",
        choices=(
            "fixed",
            "adaptive",
            "normalized",
            "patient_cluster",
            "patient_cluster_cv_adaptive",
            "distributional",
            "distributional_mondrian",
            "distributional_adaptive",
            "distributional_asymmetric",
            "distributional_shape_adaptive",
        ),
        default="fixed",
    )
    parser.add_argument(
        "--conformal-nominal-coverage", type=float, default=0.90
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help=(
            "Patient-disjoint calibration share within training domains. "
            "The external hospital/care-unit/time test remains untouched."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed external splits from OUTPUT.progress.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "external_gate_passed_cells": report[
                    "external_gate_passed_cells"
                ],
                "status": report["promotion_status"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
