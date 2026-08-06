import unittest
from pathlib import Path

from teacher_anchored_joint_jepa_registry import build_registry
from teacher_anchored_joint_jepa_external_gate import _patient_candidates
from whole_body_joint_runtime import forecast_joint_values


class TeacherAnchoredJointJEPARegistryTests(unittest.TestCase):
    def test_external_candidates_require_patient_conformal(self):
        bounded = {
            "aggregate": {
                "hierarchical": {
                    "passes": {
                        "evaluated_splits": 7,
                        "beats_teacher_splits": 7,
                        "beats_persistence_splits": 7,
                        "conformal_pass_splits": 7,
                    },
                    "point_only": {
                        "evaluated_splits": 7,
                        "beats_teacher_splits": 7,
                        "beats_persistence_splits": 7,
                        "conformal_pass_splits": 6,
                    },
                }
            }
        }
        self.assertEqual(_patient_candidates(bounded), {"passes"})

    def _payloads(self):
        key = "map@3h"
        bounded = {
            "aggregate": {
                "hierarchical": {
                    key: {
                        "evaluated_splits": 7,
                        "beats_teacher_splits": 7,
                        "beats_persistence_splits": 7,
                    }
                }
            }
        }
        external = {
            "registry": {
                key: {
                    "requirements": {
                        "hospital_conformal": True,
                        "care_unit_conformal": True,
                        "time_conformal": True,
                        "hospital_bootstrap": True,
                    }
                }
            }
        }
        scoring = {
            "reports": [
                {
                    "seed": seed,
                    "latent_svd": {"pass": True},
                    "variants": {
                        "hierarchical": {
                            "cells": {
                                key: {
                                    "direction_accuracy": 0.60,
                                    "delta_correlation": 0.20,
                                    "teacher_bootstrap": {"pass": True},
                                    "persistence_bootstrap": {"pass": True},
                                }
                            },
                            "conformal": {key: {"pass": True}},
                        }
                    },
                }
                for seed in range(7)
            ]
        }
        return bounded, external, scoring

    def test_complete_evidence_validates_only_the_cell(self):
        output = build_registry(*self._payloads())
        self.assertEqual(
            output["validated_target_horizon_cells"], ["map@3h"]
        )
        self.assertFalse(
            output["target_horizon_registry"]["map@3h"][
                "cross_organ_claim_allowed"
            ]
        )

    def test_registry_records_input_contract_and_units(self):
        output = build_registry(
            *self._payloads(),
            input_contract_id="joint_interval_urine.v1",
            variable_units={"urine_output": "mL/hour"},
            candidate_id="aligned_treatment_interval_urine_jepa",
        )
        self.assertEqual(
            output["candidate_id"],
            "aligned_treatment_interval_urine_jepa",
        )
        self.assertEqual(output["input_contract_id"], "joint_interval_urine.v1")
        self.assertEqual(output["variable_units"]["urine_output"], "mL/hour")

    def test_missing_split_fails_closed(self):
        bounded, external, scoring = self._payloads()
        scoring["reports"].pop()
        output = build_registry(bounded, external, scoring)
        self.assertEqual(output["validated_target_horizon_cells"], [])

    def test_sparse_distributional_variant_requires_matched_ablation(self):
        bounded, external, scoring = self._payloads()
        cell = bounded["aggregate"].pop("hierarchical")
        cell["map@3h"].update(
            {"conformal_pass_splits": 7, "beats_no_expert_splits": 7}
        )
        bounded["aggregate"]["sparse_distributional"] = cell
        for report in scoring["reports"]:
            report["variants"]["sparse_distributional"] = (
                report["variants"].pop("hierarchical")
            )
        external["registry"]["map@3h"]["requirements"][
            "hospital_no_expert_bootstrap"
        ] = True
        output = build_registry(
            bounded,
            external,
            scoring,
            variant="sparse_distributional",
        )
        self.assertEqual(output["validated_target_horizon_cells"], ["map@3h"])
        self.assertTrue(
            output["candidate_configuration"][
                "target_specific_distributional_scale"
            ]
        )

    def test_materialized_registry_moves_only_full_cohort_cells(self):
        registry = Path(
            "teacher_anchored_joint_jepa_validated_registry_20260731.json"
        )
        self.assertTrue(registry.exists())
        result_6h = forecast_joint_values(
            prediction=[[1.7, 90.0, 80.0]],
            current=[[1.6, 75.0, 70.0]],
            variables=("creatinine", "heart_rate", "map"),
            horizons=6.0,
            registry=registry,
            checkpoint_status=(
                "validated_target_gated_joint_research_only"
            ),
        )
        self.assertTrue(
            result_6h["forecasts"][0]["creatinine"]["can_move"]
        )
        self.assertFalse(
            result_6h["forecasts"][0]["heart_rate"]["can_move"]
        )
        self.assertFalse(result_6h["forecasts"][0]["map"]["can_move"])
        result_12h = forecast_joint_values(
            prediction=[[1.8, 90.0]],
            current=[[1.6, 75.0]],
            variables=("creatinine", "heart_rate"),
            horizons=12.0,
            registry=registry,
            checkpoint_status=(
                "validated_target_gated_joint_research_only"
            ),
        )
        self.assertTrue(
            result_12h["forecasts"][0]["creatinine"]["can_move"]
        )
        self.assertFalse(
            result_12h["forecasts"][0]["heart_rate"]["can_move"]
        )
        result_3h = forecast_joint_values(
            prediction=[[1.7, 85.0]],
            current=[[1.6, 75.0]],
            variables=("creatinine", "heart_rate"),
            horizons=3.0,
            registry=registry,
            checkpoint_status=(
                "validated_target_gated_joint_research_only"
            ),
        )
        self.assertFalse(
            result_3h["forecasts"][0]["creatinine"]["can_move"]
        )
        self.assertTrue(
            result_3h["forecasts"][0]["heart_rate"]["can_move"]
        )


if __name__ == "__main__":
    unittest.main()
