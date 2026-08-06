"""Compile teacher-anchored JEPA evidence into a fail-closed v2 registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(value):
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _patient_pass(bounded, key, variant):
    cell = bounded.get("aggregate", {}).get(variant, {}).get(key, {})
    return bool(
        cell.get("evaluated_splits") == 7
        and cell.get("beats_teacher_splits") == 7
        and cell.get("beats_persistence_splits") == 7
        and (
            not variant.startswith("sparse_")
            or cell.get("beats_no_expert_splits") == 7
        )
    )


def _external_requirements(external, key):
    registry = external.get("registry", {}).get(key, {})
    requirements = registry.get("requirements", {})
    return {
        name: bool(value) for name, value in requirements.items()
    }


def _scoring_requirements(
    scoring,
    key,
    *,
    minimum_direction_accuracy,
    minimum_delta_correlation,
    variant,
):
    reports = scoring.get("reports", [])
    cells = [
        report.get("variants", {})
        .get(variant, {})
        .get("cells", {})
        .get(key)
        for report in reports
    ]
    cells = [cell for cell in cells if isinstance(cell, dict)]
    conformal_cells = [
        report.get("variants", {})
        .get(variant, {})
        .get("conformal", {})
        .get(key, {})
        for report in reports
    ]
    split_checks = []
    for cell in cells:
        direction = cell.get("direction_accuracy")
        correlation = cell.get("delta_correlation")
        teacher = cell.get("teacher_bootstrap", {}).get("pass", False)
        persistence = cell.get("persistence_bootstrap", {}).get(
            "pass", False
        )
        split_checks.append(
            {
                "direction_accuracy": direction,
                "delta_correlation": correlation,
                "teacher_bootstrap": bool(teacher),
                "persistence_bootstrap": bool(persistence),
                "pass": bool(
                    direction is not None
                    and correlation is not None
                    and float(direction) >= minimum_direction_accuracy
                    and float(correlation) >= minimum_delta_correlation
                    and teacher
                    and persistence
                ),
            }
        )
    patient_conformal_pass = bool(
        len(conformal_cells) == 7
        and all(cell.get("pass", False) for cell in conformal_cells)
    )
    return {
        "split_count": len(split_checks),
        "splits": split_checks,
        "patient_conformal_splits": conformal_cells,
        "patient_conformal_pass": patient_conformal_pass,
        "pass": bool(
            len(split_checks) == 7
            and all(item["pass"] for item in split_checks)
        ),
    }


def _svd_requirement(scoring):
    splits = [
        {
            "seed": report.get("seed"),
            **report.get("latent_svd", {}),
        }
        for report in scoring.get("reports", [])
    ]
    return {
        "split_count": len(splits),
        "splits": splits,
        "pass": bool(
            len(splits) == 7
            and all(item.get("pass", False) for item in splits)
        ),
    }


def build_registry(
    bounded,
    external,
    scoring,
    *,
    minimum_direction_accuracy=0.55,
    minimum_delta_correlation=0.05,
    variant="hierarchical",
    input_contract_id="joint_asof_whole_body_state.v3",
    input_contract=(
        "as-of aligned whole-body state; exact anchor-linked future labels"
    ),
    variable_units=None,
    candidate_id=None,
    include_cells=None,
):
    bounded = _load(bounded)
    external = _load(external)
    scoring = _load(scoring)
    cells = sorted(bounded.get("aggregate", {}).get(variant, {}))
    if include_cells is not None:
        allowed = {str(value) for value in include_cells}
        cells = [key for key in cells if key in allowed]
    svd = _svd_requirement(scoring)
    registry = {}
    for key in cells:
        external_requirements = _external_requirements(external, key)
        external_pass = bool(
            external_requirements
            and all(external_requirements.values())
        )
        scoring_result = _scoring_requirements(
            scoring,
            key,
            minimum_direction_accuracy=minimum_direction_accuracy,
            minimum_delta_correlation=minimum_delta_correlation,
            variant=variant,
        )
        requirements = {
            "patient_7seed_teacher_and_persistence": _patient_pass(
                bounded, key, variant
            ),
            "external_hospital_care_unit_time_conformal_module": external_pass,
            "direction_delta_and_patient_bootstrap": scoring_result["pass"],
            "patient_conformal_7seed": scoring_result[
                "patient_conformal_pass"
            ],
            "svd_mode_stability": svd["pass"],
        }
        validated = bool(all(requirements.values()))
        external_cell = external.get("registry", {}).get(key, {})
        registry[key] = {
            "validated": validated,
            "validated_for_joint_runtime": validated,
            "can_move": validated,
            "status": (
                "validated_move" if validated else "fallback_persistence"
            ),
            "source_model": f"teacher_anchored_{variant}_joint_jepa",
            "requirements": requirements,
            "rejection_reasons": [
                name for name, passed in requirements.items() if not passed
            ],
            "external_requirements": external_requirements,
            "scoring": scoring_result,
            "patient_conformal": bool(
                scoring_result["patient_conformal_pass"]
            ),
            "hospital_conformal": bool(
                external_cell.get("requirements", {}).get(
                    "hospital_conformal", False
                )
            ),
            "care_unit_conformal": bool(
                external_cell.get("requirements", {}).get(
                    "care_unit_conformal", False
                )
            ),
            "time_conformal": bool(
                external_cell.get("requirements", {}).get(
                    "time_conformal", False
                )
            ),
            "cross_organ_claim_allowed": False,
            "causal_claim_allowed": False,
        }
    validated_cells = sorted(
        key for key, value in registry.items() if value["can_move"]
    )
    return {
        "schema": "whole_body_promotion_registry.v2",
        "candidate_id": (
            candidate_id
            or f"teacher_anchored_{variant}_joint_jepa"
        ),
        "input_contract_id": input_contract_id,
        "input_contract": input_contract,
        "variable_units": dict(variable_units or {}),
        "candidate_configuration": {
            "validated_router_teacher_anchor": True,
            "zero_initialized_bounded_residual": True,
            "two_level_latent_hierarchy": True,
            "level_1_shared_whole_body_latent": True,
            "level_2_canonical_organ_latent_tokens": True,
            "target_heads_above_two_level_latents": True,
            "shared_slow_context_latent": True,
            "target_specific_fast_future_latent": True,
            "dynamic_cross_organ_route": False,
            "sparse_target_horizon_experts": bool(
                variant.startswith("sparse_")
            ),
            "target_specific_distributional_scale": bool(
                variant == "sparse_distributional"
            ),
            "pcgrad": False,
            "conformal_method": bounded.get("architecture", {}).get(
                "conformal_method", "unknown"
            ),
            "conformal_nominal_coverage": bounded.get(
                "architecture", {}
            ).get("conformal_nominal_coverage"),
            "target_empirical_interval_coverage": 0.90,
        },
        "target_horizon_registry": registry,
        "validated_target_horizon_cells": validated_cells,
        "validated_target_horizon_count": len(validated_cells),
        "global_svd_gate": svd,
        "policy": {
            "minimum_direction_accuracy": minimum_direction_accuracy,
            "minimum_delta_correlation": minimum_delta_correlation,
            "requires_7_patient_splits": True,
            "requires_hospital_care_unit_time_and_conformal_gate": True,
            "requires_current_router_comparison": True,
            "cross_organ_attribution_requires_separate_ablation": True,
            "missing_evidence_action": "fallback_persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "promotion_status": (
            "validated_target_gated_joint_research_only"
            if validated_cells
            else "candidate_only"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("hierarchical", "sparse_experts", "sparse_distributional"),
        default="hierarchical",
    )
    parser.add_argument(
        "--input-contract-id",
        default="joint_asof_whole_body_state.v3",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Distinct runtime candidate identity for this exact data contract.",
    )
    parser.add_argument(
        "--input-contract",
        default=(
            "as-of aligned whole-body state; exact anchor-linked future labels"
        ),
    )
    parser.add_argument(
        "--variable-unit",
        action="append",
        default=[],
        metavar="TARGET=UNIT",
        help="Record a required clinical unit; may be repeated.",
    )
    parser.add_argument(
        "--include-cell",
        action="append",
        default=None,
        metavar="TARGET@HORIZON",
        help="Materialize only this validated cell; may be repeated.",
    )
    args = parser.parse_args()
    variable_units = {}
    for item in args.variable_unit:
        if "=" not in item:
            parser.error("--variable-unit must use TARGET=UNIT")
        target, unit = item.split("=", 1)
        if not target.strip() or not unit.strip():
            parser.error("--variable-unit requires non-empty TARGET and UNIT")
        variable_units[target.strip()] = unit.strip()
    output = build_registry(
        args.bounded,
        args.external,
        args.scoring,
        variant=args.variant,
        input_contract_id=args.input_contract_id,
        input_contract=args.input_contract,
        variable_units=variable_units,
        candidate_id=args.candidate_id,
        include_cells=args.include_cell,
    )
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validated_cells": output[
                    "validated_target_horizon_cells"
                ],
                "promotion_status": output["promotion_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
