"""Materialize a strict target-by-horizon whole-body runtime registry.

The training gate, scoring audit, latent stability audit and current-router
comparison are deliberately separate artifacts. This module is the only place
where they may be combined into runtime permission. Missing evidence fails
closed instead of being interpreted as a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load(value: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _scoring_by_cell(scoring: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for report in scoring.get("reports", ()):
        for key, value in report.get("cells", {}).items():
            if isinstance(value, Mapping):
                cells.setdefault(str(key), []).append(value)
    return cells


def _svd_passes(scoring: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    reports = list(scoring.get("reports", ()))
    if not reports:
        return False, {"reason": "missing_svd_reports"}
    passes = []
    details = []
    for report in reports:
        # Prefer the learned JEPA latent audit. Older scoring artifacts expose
        # only the physiological Jacobian SVD and remain compatible.
        svd = report.get("latent_svd") or report.get("svd", {})
        fit_rank = svd.get("fit_effective_rank")
        test_rank = svd.get("test_effective_rank")
        cosine = svd.get(
            "mean_top_subspace_cosine", svd.get("mean_top_mode_cosine")
        )
        valid = all(value is not None and np.isfinite(float(value)) for value in (fit_rank, test_rank, cosine))
        rank_ratio = (
            float(test_rank) / max(float(fit_rank), 1e-8) if valid else None
        )
        passed = bool(
            valid
            and float(fit_rank) >= 2.0
            and float(test_rank) >= 2.0
            and 0.50 <= rank_ratio <= 2.0
            and float(cosine) >= 0.50
        )
        passes.append(passed)
        details.append(
            {
                "seed": report.get("seed"),
                "fit_effective_rank": fit_rank,
                "test_effective_rank": test_rank,
                "rank_ratio": rank_ratio,
                "mean_top_mode_cosine": cosine,
                "mode_stability_metric": (
                    "principal_subspace_overlap"
                    if "mean_top_subspace_cosine" in svd
                    else "legacy_same_index_mode_cosine"
                ),
                "pass": passed,
            }
        )
    return bool(all(passes)), {"splits": details}


def _score_passes(
    reports: list[Mapping[str, Any]],
    *,
    minimum_direction_accuracy: float,
    minimum_delta_correlation: float,
) -> tuple[bool, dict[str, Any]]:
    if len(reports) < 7:
        return False, {
            "reason": "fewer_than_7_patient_scoring_splits",
            "split_count": len(reports),
        }
    checks = []
    for report in reports:
        direction = report.get("direction_accuracy")
        correlation = report.get("delta_correlation")
        ci = report.get("patient_delta_bootstrap_ci", {})
        high = ci.get("high") if isinstance(ci, Mapping) else None
        passed = bool(
            direction is not None
            and correlation is not None
            and high is not None
            and float(direction) >= minimum_direction_accuracy
            and float(correlation) >= minimum_delta_correlation
            and float(high) < 0.0
        )
        checks.append(
            {
                "direction_accuracy": direction,
                "delta_correlation": correlation,
                "patient_bootstrap_high": high,
                "pass": passed,
            }
        )
    return bool(all(item["pass"] for item in checks)), {"splits": checks}


def build_registry(
    gate: Mapping[str, Any] | str | Path,
    scoring: Mapping[str, Any] | str | Path,
    *,
    validated_router_comparison: Mapping[str, Any] | str | Path | None = None,
    minimum_direction_accuracy: float = 0.55,
    minimum_delta_correlation: float = 0.05,
) -> dict[str, Any]:
    """Return a v2 registry; incomplete evidence can never grant movement."""

    gate = _load(gate)
    scoring = _load(scoring)
    router = _load(validated_router_comparison)
    gate_cells = gate.get(
        "joint_runtime_registry", gate.get("target_horizon_registry", {})
    )
    score_cells = _scoring_by_cell(scoring)
    router_cells = router.get("cells", router.get("target_horizon_registry", {}))
    svd_pass, svd_detail = _svd_passes(scoring)
    gate_configuration = gate.get("model_configuration")
    scoring_configuration = scoring.get("model_configuration")
    configuration_match = bool(
        isinstance(gate_configuration, Mapping)
        and isinstance(scoring_configuration, Mapping)
        and dict(gate_configuration) == dict(scoring_configuration)
    )
    registry = {}
    for key, gate_cell in gate_cells.items():
        gate_cell = gate_cell if isinstance(gate_cell, Mapping) else {}
        score_pass, score_detail = _score_passes(
            score_cells.get(str(key), []),
            minimum_direction_accuracy=minimum_direction_accuracy,
            minimum_delta_correlation=minimum_delta_correlation,
        )
        if isinstance(router_cells, Mapping) and key in router_cells:
            router_cell = router_cells.get(key, {})
            router_pass = bool(
                isinstance(router_cell, Mapping)
                and (
                    router_cell.get("pass")
                    or router_cell.get("beats_validated_router")
                )
            )
        else:
            # New gate artifacts rebuild the current ridge/persistence router
            # inside every outer split. This is stronger than an unrelated
            # external comparison and avoids test-set source selection.
            router_cell = {
                name: gate_cell.get(name, False)
                for name in (
                    "patient_validated_router_value",
                    "patient_validated_router_bootstrap",
                    "hospital_validated_router_value",
                    "hospital_validated_router_bootstrap",
                    "care_unit_validated_router_value",
                    "care_unit_validated_router_bootstrap",
                    "time_validated_router_value",
                    "time_validated_router_bootstrap",
                )
            }
            router_pass = bool(router_cell and all(router_cell.values()))
        requirements = {
            "joint_gate": bool(
                gate_cell.get(
                    "validated_for_joint_runtime",
                    gate_cell.get("validated", False),
                )
            ),
            "direction_delta_and_patient_bootstrap": score_pass,
            "svd_mode_stability": svd_pass,
            "beats_current_validated_router": router_pass,
            "same_candidate_configuration": configuration_match,
        }
        can_move = bool(all(requirements.values()))
        rejection_reasons = [
            name for name, passed in requirements.items() if not passed
        ]
        registry[str(key)] = {
            "validated": can_move,
            "validated_for_joint_runtime": can_move,
            "can_move": can_move,
            "status": "validated_move" if can_move else "fallback_persistence",
            "requirements": requirements,
            "rejection_reasons": rejection_reasons,
            "scoring": score_detail,
            "router_comparison": dict(router_cell)
            if isinstance(router_cell, Mapping)
            else {},
            "patient_conformal": bool(gate_cell.get("patient_conformal", False)),
            "hospital_conformal": bool(gate_cell.get("hospital_conformal", False)),
            "care_unit_conformal": bool(gate_cell.get("care_unit_conformal", False)),
            "time_conformal": bool(gate_cell.get("time_conformal", False)),
        }
    validated = sorted(
        key for key, cell in registry.items() if cell["can_move"]
    )
    return {
        "schema": "whole_body_promotion_registry.v2",
        "target_horizon_registry": registry,
        "validated_target_horizon_cells": validated,
        "validated_target_horizon_count": len(validated),
        "global_svd_gate": {
            "pass": svd_pass,
            **svd_detail,
        },
        "candidate_configuration_gate": {
            "pass": configuration_match,
            "gate_configuration": gate_configuration,
            "scoring_configuration": scoring_configuration,
        },
        "policy": {
            "minimum_direction_accuracy": float(minimum_direction_accuracy),
            "minimum_delta_correlation": float(minimum_delta_correlation),
            "requires_7_patient_splits": True,
            "requires_hospital_care_unit_time_and_conformal_gate": True,
            "requires_current_router_comparison": True,
            "missing_evidence_action": "fallback_persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "promotion_status": (
            "validated_target_gated_joint_research_only"
            if validated
            else "candidate_only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--scoring", required=True, type=Path)
    parser.add_argument("--validated-router-comparison", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-direction-accuracy", type=float, default=0.55)
    parser.add_argument("--minimum-delta-correlation", type=float, default=0.05)
    args = parser.parse_args()
    output = build_registry(
        args.gate,
        args.scoring,
        validated_router_comparison=args.validated_router_comparison,
        minimum_direction_accuracy=args.minimum_direction_accuracy,
        minimum_delta_correlation=args.minimum_delta_correlation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "validated_cells": output["validated_target_horizon_count"],
                "promotion_status": output["promotion_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
