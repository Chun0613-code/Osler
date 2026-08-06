"""Materialize fail-closed registries from target-specific factual audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_UNITS = {
    "albumin": "g/dL",
    "anion_gap": "mmol/L",
    "bicarbonate": "mmol/L",
    "bun": "mg/dL",
    "calcium": "mg/dL",
    "chloride": "mmol/L",
    "creatinine": "mg/dL",
    "hematocrit": "%",
    "hemoglobin": "g/dL",
    "magnesium": "mg/dL",
    "phosphate": "mg/dL",
    "platelets": "K/uL",
    "potassium": "mmol/L",
    "sodium": "mmol/L",
    "temperature": "degC",
    "total_protein": "g/dL",
    "wbc": "K/uL",
}


def _change_threshold(cell: dict) -> float | None:
    thresholds = {
        float(split["change_threshold"])
        for split in cell.get("splits", {}).values()
        if isinstance(split, dict) and split.get("change_threshold") is not None
    }
    if len(thresholds) > 1:
        raise ValueError("A target-horizon cell contains inconsistent change thresholds")
    return next(iter(thresholds), None)


def build_registry(report: dict, *, candidate_id: str, input_contract_id: str) -> dict:
    supported_schemas = {
        "target_specific_event_hurdle_audit.v1",
        "target_specific_event_hurdle_domain_conformal_audit.v1",
    }
    if report.get("schema") not in supported_schemas:
        raise ValueError("Unsupported target-specific audit schema")
    registry = {}
    units = {}
    output_contract = {}
    for original_key, cell in sorted(report.get("reports", {}).items()):
        target = str(cell.get("target", ""))
        horizon = int(cell.get("horizon_hours", 0))
        validated = bool(cell.get("summary", {}).get("validated", False))
        interval = (
            "external-domain-heldout adaptive normalized split conformal"
            if report.get("schema")
            == "target_specific_event_hurdle_domain_conformal_audit.v1"
            else "patient-grouped crossfit adaptive normalized split conformal"
        )
        if target == "o2sat":
            runtime_target = "hypoxemia_event"
            task_type = "binary_event_probability"
            units[runtime_target] = "probability"
            output_contract[runtime_target] = {
                "task_type": task_type,
                "event_definition": report.get("hypoxemia_definition"),
                "not_equivalent_to": "continuous oxygen-saturation value forecast",
            }
        elif target in TARGET_UNITS:
            runtime_target = target
            task_type = "continuous_value_with_change_hurdle"
            units[runtime_target] = TARGET_UNITS[target]
            threshold = _change_threshold(cell)
            output_contract[runtime_target] = {
                "task_type": task_type,
                "change_definition": (
                    f"abs(future-current) >= {threshold} {TARGET_UNITS[target]}"
                    if threshold is not None
                    else None
                ),
                "change_threshold": threshold,
                "interval": interval,
                "point_prediction": "persistence-anchored target-specific change hurdle",
            }
        else:
            continue
        key = f"{runtime_target}@{horizon}h"
        patient_conformal = (
            cell.get("summary", {}).get("patient_passes")
            == cell.get("summary", {}).get("patient_required")
            == 7
        )
        hospital_conformal = bool(
            cell.get("splits", {}).get("hospital", {}).get("pass", False)
        )
        care_unit_conformal = bool(
            cell.get("splits", {}).get("careunit", {}).get("pass", False)
        )
        time_conformal = bool(
            cell.get("splits", {}).get("forward_time", {}).get("pass", False)
        )
        registry[key] = {
            "validated_for_joint_runtime": validated,
            "can_move": validated,
            "task_type": task_type,
            "source_model": candidate_id,
            "source_audit_cell": original_key,
            "patient_conformal": patient_conformal,
            "hospital_conformal": hospital_conformal,
            "care_unit_conformal": care_unit_conformal,
            "time_conformal": time_conformal,
            "requirements": {
                "patient_7seed": patient_conformal,
                "hospital_heldout": hospital_conformal,
                "careunit_heldout": care_unit_conformal,
                "forward_time_heldout": time_conformal,
                "causal_claim_allowed": False,
            },
        }
    validated_cells = sorted(
        key for key, value in registry.items() if value["validated_for_joint_runtime"]
    )
    return {
        "schema": "whole_body_promotion_registry.v2",
        "candidate_id": candidate_id,
        "input_contract_id": input_contract_id,
        "input_contract": report.get("input_policy"),
        "variable_units": units,
        "output_contract": output_contract,
        "source_audit_schema": report.get("schema"),
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "promotion_status": (
            "validated_target_gated_joint_research_only"
            if validated_cells
            else "candidate_only"
        ),
        "validated_target_horizon_cells": validated_cells,
        "target_horizon_registry": registry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--input-contract-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    output = build_registry(
        report,
        candidate_id=args.candidate_id,
        input_contract_id=args.input_contract_id,
    )
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "validated_cells": output["validated_target_horizon_cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
