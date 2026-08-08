"""Build and query the fail-closed pure-JEPA target promotion registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_registry(report: dict) -> dict:
    registry = {}
    for module, result in report.items():
        if not isinstance(result, dict) or "aggregate" not in result:
            continue
        hospital_metrics = result.get("hospital_heldout", {}).get("metrics", {})
        patient_reports = result.get("patient_heldout", {})
        module_registry = {}
        for target, aggregate in result["aggregate"].items():
            patient_ok = bool(aggregate.get("validated_patient_gate", False))
            hospital_ok = bool(
                result.get("hospital_heldout", {}).get("status") == "run"
                and hospital_metrics.get(target, {}).get("significant_vs_persistence", False)
            )
            conformal_statuses = [
                seed_report.get("conformal", {}).get(target, {}).get("status")
                for seed_report in patient_reports.values()
            ]
            conformal_ok = bool(
                len(conformal_statuses) == 7
                and all(status == "PASS" for status in conformal_statuses)
            )
            can_move = patient_ok and hospital_ok and conformal_ok
            reasons = []
            if not patient_ok:
                reasons.append("patient_gate")
            if not hospital_ok:
                reasons.append("hospital_gate")
            if not conformal_ok:
                reasons.append("conformal_gate")
            module_registry[target] = {
                "can_move": can_move,
                "status": "validated_move" if can_move else "fallback_persistence",
                "patient_gate": patient_ok,
                "hospital_gate": hospital_ok,
                "conformal_gate": conformal_ok,
                "rejection_reasons": reasons,
            }
        registry[module] = module_registry
    return {
        "policy": {
            "fail_closed": True,
            "unvalidated_target_action": "persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "modules": registry,
    }


def target_status(registry: dict, module: str, target: str) -> dict:
    return (
        registry.get("modules", {})
        .get(module, {})
        .get(
            target,
            {
                "can_move": False,
                "status": "fallback_persistence",
                "rejection_reasons": ["missing_registry_entry"],
            },
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="pure_jepa_full_gate.json")
    parser.add_argument("--output", default="pure_jepa_gate_registry.json")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    registry = build_registry(report)
    Path(args.output).write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
