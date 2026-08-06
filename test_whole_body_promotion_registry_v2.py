import unittest

from whole_body_promotion_registry_v2 import build_registry


def scoring_report(direction=0.7, correlation=0.3, high=-0.01):
    return {
        "cells": {
            "map@6h": {
                "direction_accuracy": direction,
                "delta_correlation": correlation,
                "patient_delta_bootstrap_ci": {"high": high},
            }
        },
        "svd": {
            "fit_effective_rank": 4.0,
            "test_effective_rank": 3.5,
            "mean_top_mode_cosine": 0.1,
            "mean_top_subspace_cosine": 0.8,
        },
    }

MODEL_CONFIGURATION = {"source_contract": "canonical_event_ledger", "world_model": True}


class PromotionRegistryV2Tests(unittest.TestCase):
    def test_all_evidence_is_required(self):
        gate = {
            "model_configuration": MODEL_CONFIGURATION,
            "target_horizon_registry": {
                "map@6h": {"validated_for_joint_runtime": True}
            }
        }
        scoring = {
            "model_configuration": MODEL_CONFIGURATION,
            "reports": [scoring_report() for _ in range(7)],
        }
        result = build_registry(
            gate,
            scoring,
            validated_router_comparison={
                "cells": {"map@6h": {"beats_validated_router": True}}
            },
        )
        self.assertTrue(result["target_horizon_registry"]["map@6h"]["can_move"])

    def test_missing_router_comparison_fails_closed(self):
        gate = {
            "model_configuration": MODEL_CONFIGURATION,
            "target_horizon_registry": {
                "map@6h": {"validated_for_joint_runtime": True}
            }
        }
        scoring = {
            "model_configuration": MODEL_CONFIGURATION,
            "reports": [scoring_report() for _ in range(7)],
        }
        result = build_registry(gate, scoring)
        cell = result["target_horizon_registry"]["map@6h"]
        self.assertFalse(cell["can_move"])
        self.assertIn("beats_current_validated_router", cell["rejection_reasons"])

    def test_nested_router_evidence_can_come_from_the_joint_gate(self):
        router_evidence = {
            name: True
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
        gate = {
            "model_configuration": MODEL_CONFIGURATION,
            "target_horizon_registry": {
                "map@6h": {
                    "validated_for_joint_runtime": True,
                    **router_evidence,
                }
            }
        }
        scoring = {
            "model_configuration": MODEL_CONFIGURATION,
            "reports": [scoring_report() for _ in range(7)],
        }
        result = build_registry(gate, scoring)
        self.assertTrue(result["target_horizon_registry"]["map@6h"]["can_move"])

    def test_latent_collapse_rejects_every_cell(self):
        gate = {
            "model_configuration": MODEL_CONFIGURATION,
            "target_horizon_registry": {
                "map@6h": {"validated_for_joint_runtime": True}
            }
        }
        reports = [scoring_report() for _ in range(7)]
        for report in reports:
            report["svd"]["test_effective_rank"] = 1.0
        result = build_registry(
            gate,
            {"model_configuration": MODEL_CONFIGURATION, "reports": reports},
            validated_router_comparison={
                "cells": {"map@6h": {"beats_validated_router": True}}
            },
        )
        self.assertFalse(result["target_horizon_registry"]["map@6h"]["can_move"])

    def test_rotation_invariant_subspace_metric_controls_the_gate(self):
        gate = {
            "model_configuration": MODEL_CONFIGURATION,
            "target_horizon_registry": {
                "map@6h": {"validated_for_joint_runtime": True}
            }
        }
        reports = [scoring_report() for _ in range(7)]
        result = build_registry(
            gate,
            {"model_configuration": MODEL_CONFIGURATION, "reports": reports},
            validated_router_comparison={
                "cells": {"map@6h": {"beats_validated_router": True}}
            },
        )
        self.assertTrue(result["global_svd_gate"]["pass"])

    def test_mismatched_candidate_configuration_fails_closed(self):
        gate = {
            "model_configuration": MODEL_CONFIGURATION,
            "target_horizon_registry": {
                "map@6h": {"validated_for_joint_runtime": True}
            },
        }
        scoring = {
            "model_configuration": {
                "source_contract": "legacy_module_transition",
                "world_model": True,
            },
            "reports": [scoring_report() for _ in range(7)],
        }
        result = build_registry(
            gate,
            scoring,
            validated_router_comparison={
                "cells": {"map@6h": {"beats_validated_router": True}}
            },
        )
        cell = result["target_horizon_registry"]["map@6h"]
        self.assertFalse(cell["can_move"])
        self.assertIn("same_candidate_configuration", cell["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
