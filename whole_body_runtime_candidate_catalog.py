"""Compile schema-safe factual runtime candidate catalogs.

The shared-module stack and joint-JEPA answer related but non-interchangeable
questions.  The shared stack consumes one module row at a time; joint-JEPA
consumes an as-of aligned whole-body state.  This tool records both contracts
without blending their outputs or treating a validation result from one schema
as evidence for the other.

The catalog is factual research infrastructure only.  It never authorizes
causal, treatment, or clinical decision claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _shared_entry(report: dict) -> dict:
    registry = report.get("runtime_registry", {})
    validated = sorted(
        cell
        for cell, value in registry.items()
        if isinstance(value, dict) and value.get("validated_for_runtime", False)
    )
    return {
        "candidate_id": "shared_module_nested_stack",
        "input_contract_id": "shared_module_row.v1",
        "input_contract": report.get(
            "data_contract",
            "module-local historical rows with true-patient evaluation identity",
        ),
        "prediction_source": "nested convex stack selected on disjoint patient calibration rows",
        "validated_cells": validated,
        "validated_cell_count": len(validated),
        "runtime_status": (
            "target_gated_factual_research_only" if validated else "candidate_only"
        ),
        "fallback": "persistence for every cell absent from validated_cells",
    }


def _joint_entry(report: dict) -> dict:
    schema = str(report.get("schema", ""))
    registry = report.get("joint_runtime_registry", {})
    schema_current = schema == "whole_body_joint_jepa_gate.v3"
    validated = sorted(
        cell
        for cell, value in registry.items()
        if schema_current
        and isinstance(value, dict)
        and value.get("validated_for_joint_runtime", False)
    )
    return {
        "candidate_id": "joint_asof_whole_body_jepa",
        "input_contract_id": "joint_asof_whole_body_state.v3",
        "input_contract": report.get(
            "joint_runtime_data_contract",
            "as-of aligned whole-body state; exact anchor-linked future labels",
        ),
        "source_gate_schema": schema,
        "schema_current": schema_current,
        "validated_cells": validated,
        "validated_cell_count": len(validated),
        "runtime_status": (
            "target_gated_factual_research_only"
            if validated
            else "candidate_only"
        ),
        "fallback": "do not invoke joint-JEPA; use the matching contract's validated source or persistence",
    }


def _promotion_registry_entry(report: dict) -> dict:
    schema = str(report.get("schema", ""))
    schema_current = schema == "whole_body_promotion_registry.v2"
    registry = report.get("target_horizon_registry", {})
    validated = sorted(
        cell
        for cell, value in registry.items()
        if schema_current
        and isinstance(value, dict)
        and value.get(
            "validated_for_joint_runtime",
            value.get("can_move", value.get("validated", False)),
        )
    )
    return {
        "candidate_id": report.get("candidate_id", "joint_jepa_registry"),
        "input_contract_id": report.get(
            "input_contract_id", "unspecified_fail_closed"
        ),
        "input_contract": report.get(
            "input_contract", "materialized target-gated joint-JEPA registry"
        ),
        "variable_units": report.get("variable_units", {}),
        "output_contract": report.get("output_contract", {}),
        "source_gate_schema": schema,
        "schema_current": schema_current,
        "validated_cells": validated,
        "validated_cell_count": len(validated),
        "runtime_status": (
            "target_gated_factual_research_only"
            if validated and report.get("promotion_status")
            == "validated_target_gated_joint_research_only"
            else "candidate_only"
        ),
        "fallback": (
            "persistence unless registry, input contract, target horizon, and units all match"
        ),
    }


def build_catalog(shared: dict, joint: dict, promotion_registries=()) -> dict:
    entries = [_shared_entry(shared), _joint_entry(joint)]
    entries.extend(_promotion_registry_entry(item) for item in promotion_registries)
    return {
        "schema": "whole_body_runtime_candidate_catalog.v1",
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "candidates": entries,
        "resolution_policy": {
            "rule": "Choose only candidates whose input_contract_id exactly matches the live feature schema.",
            "forbidden": [
                "Do not blend shared-module and joint-asof predictions during validation.",
                "Do not promote any cell absent from that candidate's validated_cells.",
                "Do not use this catalog for treatment-effect or clinical-decision claims.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument(
        "--promotion-registry",
        type=Path,
        action="append",
        default=[],
        help="Additional materialized v2 promotion registry; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = build_catalog(
        _load(args.shared),
        _load(args.joint),
        [_load(path) for path in args.promotion_registry],
    )
    args.output.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidates": [
                    {
                        "candidate_id": entry["candidate_id"],
                        "runtime_status": entry["runtime_status"],
                        "validated_cells": entry["validated_cell_count"],
                    }
                    for entry in catalog["candidates"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
