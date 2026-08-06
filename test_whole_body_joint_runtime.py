import json
import unittest
from pathlib import Path

import numpy as np

from whole_body_joint_runtime import (
    forecast_joint_values,
    forecast_sparse_irregular_value,
)


def joint_registry(cells):
    return {
        "schema": "whole_body_promotion_registry.v2",
        "target_horizon_registry": cells,
    }


class WholeBodyJointRuntimeTests(unittest.TestCase):
    def test_validated_glucose_registries_move_source_specific_cells(self):
        root = Path(__file__).resolve().parent
        aligned = (
            root
            / "aligned_treatment_glucose_hierarchical_validated_registry_20260803.json"
        )
        for horizon in (1.0, 12.0):
            result = forecast_joint_values(
                prediction=[[118.0]],
                current=[[130.0]],
                variables=("glucose",),
                horizons=horizon,
                registry=aligned,
                checkpoint_status=(
                    "validated_target_gated_joint_research_only"
                ),
                input_contract_id=(
                    "joint_asof_whole_body_state_aligned_treatment_"
                    "interval_urine.v1"
                ),
                variable_units={"glucose": "mg/dL"},
            )
            self.assertTrue(
                result["forecasts"][0]["glucose"]["can_move"]
            )

        legacy = (
            root
            / "two_level_glucose3_sparse_distributional_validated_registry_20260803.json"
        )
        result = forecast_joint_values(
            prediction=[[121.0]],
            current=[[130.0]],
            variables=("glucose",),
            horizons=3.0,
            registry=legacy,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id="joint_asof_whole_body_state.v3",
            variable_units={"glucose": "mg/dL"},
        )
        self.assertTrue(result["forecasts"][0]["glucose"]["can_move"])

    def test_validated_respiratory_registries_move_new_cells(self):
        root = Path(__file__).resolve().parent
        contract = (
            "joint_asof_whole_body_state_aligned_treatment_"
            "interval_urine.v1"
        )
        for filename, horizon in (
            ("respiratory3_validated_registry_20260803.json", 3.0),
            ("respiratory24_validated_registry_20260803.json", 24.0),
        ):
            result = forecast_joint_values(
                prediction=[[18.0]],
                current=[[20.0]],
                variables=("respiratory_rate",),
                horizons=horizon,
                registry=root / filename,
                checkpoint_status=(
                    "validated_target_gated_joint_research_only"
                ),
                input_contract_id=contract,
                variable_units={"respiratory_rate": "breaths/minute"},
            )["forecasts"][0]["respiratory_rate"]
            self.assertTrue(result["can_move"])
            self.assertEqual(
                result["source"],
                "teacher_anchored_sparse_distributional_joint_jepa",
            )

    def test_domain_calibrated_hurdle_cell_is_runtime_usable(self):
        root = Path(__file__).resolve().parent
        cell = forecast_joint_values(
            prediction=[[145.0]],
            current=[[130.0]],
            variables=("platelets",),
            horizons=1.0,
            registry=(
                root
                / "platelets1_domain_hurdle_validated_registry_20260805.json"
            ),
            checkpoint_status="validated_target_gated_joint_research_only",
            lower=[[100.0]],
            upper=[[190.0]],
            input_contract_id=(
                "joint_asof_whole_body_state_aligned_treatment_"
                "interval_urine.v1"
            ),
            variable_units={"platelets": "K/uL"},
        )["forecasts"][0]["platelets"]
        self.assertTrue(cell["can_move"])
        self.assertEqual(cell["forecast_point"], 145.0)
        self.assertEqual(
            cell["source_model"],
            "platelets_domain_calibrated_change_hurdle_joint_jepa",
        )
        self.assertEqual(
            cell["task_type"], "continuous_value_with_change_hurdle"
        )
        self.assertEqual(cell["interval_status"], "calibrated")

    def test_all_new_target_specific_hurdle_cells_are_runtime_usable(self):
        root = Path(__file__).resolve().parent
        filenames = (
            "platelets3_hurdle_validated_registry_20260805.json",
            "hemoglobin_hurdle_validated_registry_20260805.json",
            "hematocrit_hurdle_validated_registry_20260805.json",
            "bicarbonate24_hurdle_validated_registry_20260805.json",
            "bun24_hurdle_validated_registry_20260805.json",
            "platelets1_domain_hurdle_validated_registry_20260805.json",
        )
        observed_cells = set()
        for filename in filenames:
            path = root / filename
            registry = json.loads(path.read_text(encoding="utf-8"))
            for key in registry["validated_target_horizon_cells"]:
                target, rendered_horizon = key.split("@", 1)
                horizon = float(rendered_horizon.removesuffix("h"))
                unit = registry["variable_units"][target]
                cell = forecast_joint_values(
                    prediction=[[2.0]],
                    current=[[1.0]],
                    variables=(target,),
                    horizons=horizon,
                    registry=path,
                    checkpoint_status=(
                        "validated_target_gated_joint_research_only"
                    ),
                    lower=[[0.5]],
                    upper=[[2.5]],
                    input_contract_id=registry["input_contract_id"],
                    variable_units={target: unit},
                )["forecasts"][0][target]
                self.assertTrue(cell["can_move"], key)
                self.assertEqual(
                    cell["source_model"], registry["candidate_id"], key
                )
                self.assertEqual(
                    cell["task_type"],
                    "continuous_value_with_change_hurdle",
                    key,
                )
                self.assertEqual(cell["interval_status"], "calibrated", key)
                observed_cells.add(key)
        self.assertEqual(
            observed_cells,
            {
                "platelets@1h",
                "platelets@3h",
                "hemoglobin@3h",
                "hemoglobin@6h",
                "hemoglobin@12h",
                "hematocrit@3h",
                "hematocrit@6h",
                "bicarbonate@24h",
                "bun@24h",
            },
        )

    def test_respiratory_registry_falls_back_on_unit_mismatch(self):
        result = forecast_joint_values(
            prediction=[[18.0]],
            current=[[20.0]],
            variables=("respiratory_rate",),
            horizons=3.0,
            registry=(
                Path(__file__).resolve().parent
                / "respiratory3_validated_registry_20260803.json"
            ),
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id=(
                "joint_asof_whole_body_state_aligned_treatment_"
                "interval_urine.v1"
            ),
            variable_units={"respiratory_rate": "breaths/second"},
        )["forecasts"][0]["respiratory_rate"]
        self.assertFalse(result["can_move"])
        self.assertIn("variable_unit_mismatch", result["rejection_reasons"])

    def test_aligned_glucose_registry_rejects_wrong_contract_and_horizon(self):
        registry = (
            Path(__file__).resolve().parent
            / "aligned_treatment_glucose_hierarchical_validated_registry_20260803.json"
        )
        wrong_contract = forecast_joint_values(
            prediction=[[118.0]],
            current=[[130.0]],
            variables=("glucose",),
            horizons=1.0,
            registry=registry,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id="joint_asof_whole_body_state.v3",
            variable_units={"glucose": "mg/dL"},
        )["forecasts"][0]["glucose"]
        self.assertFalse(wrong_contract["can_move"])
        self.assertIn(
            "input_contract_mismatch", wrong_contract["rejection_reasons"]
        )

        wrong_horizon = forecast_joint_values(
            prediction=[[118.0]],
            current=[[130.0]],
            variables=("glucose",),
            horizons=3.0,
            registry=registry,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id=(
                "joint_asof_whole_body_state_aligned_treatment_"
                "interval_urine.v1"
            ),
            variable_units={"glucose": "mg/dL"},
        )["forecasts"][0]["glucose"]
        self.assertFalse(wrong_horizon["can_move"])
        self.assertIn(
            "target_horizon_not_validated",
            wrong_horizon["rejection_reasons"],
        )

    def test_aligned_treatment_urine_registry_is_contract_gated(self):
        registry_path = (
            Path(__file__).resolve().parent
            / "aligned_treatment_interval_urine_validated_registry_20260802.json"
        )
        matched = forecast_joint_values(
            prediction=[[72.0]],
            current=[[60.0]],
            variables=("urine_output",),
            horizons=1.0,
            registry=registry_path,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id=(
                "joint_asof_whole_body_state_aligned_treatment_"
                "interval_urine.v1"
            ),
            variable_units={"urine_output": "mL/hour"},
        )
        self.assertTrue(matched["forecasts"][0]["urine_output"]["can_move"])

        legacy = forecast_joint_values(
            prediction=[[72.0]],
            current=[[60.0]],
            variables=("urine_output",),
            horizons=1.0,
            registry=registry_path,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id="joint_asof_whole_body_state_interval_urine.v1",
            variable_units={"urine_output": "mL/hour"},
        )
        cell = legacy["forecasts"][0]["urine_output"]
        self.assertFalse(cell["can_move"])
        self.assertIn("input_contract_mismatch", cell["rejection_reasons"])

    def test_unvalidated_cells_fall_back_to_persistence(self):
        result = forecast_joint_values(
            prediction=np.asarray([[110.0, 2.0]], dtype=float),
            current=np.asarray([[70.0, 1.2]], dtype=float),
            variables=("map", "creatinine"),
            horizons=6.0,
            registry=joint_registry({
                    "map@6h": {"validated": True},
                    "creatinine@6h": {"validated": False},
                }),
            checkpoint_status="validated_target_gated_joint_research_only",
        )
        self.assertEqual(result["forecasts"][0]["map"]["source"], "joint_jepa_validated")
        self.assertEqual(result["forecasts"][0]["creatinine"]["source"], "persistence_fallback")
        self.assertEqual(result["prediction_raw"], [[110.0, 1.2]])

    def test_candidate_checkpoint_cannot_move_any_cell(self):
        result = forecast_joint_values(
            prediction=np.asarray([[110.0]], dtype=float),
            current=np.asarray([[70.0]], dtype=float),
            variables=("map",),
            horizons=6.0,
            registry=joint_registry({"map@6h": {"validated": True}}),
            checkpoint_status="candidate_only",
        )
        entry = result["forecasts"][0]["map"]
        self.assertEqual(entry["source"], "persistence_fallback")
        self.assertIn("checkpoint_candidate_only", entry["rejection_reasons"])

    def test_nonfinite_candidate_falls_back_even_when_validated(self):
        result = forecast_joint_values(
            prediction=np.asarray([[np.nan]], dtype=float),
            current=np.asarray([[70.0]], dtype=float),
            variables=("map",),
            horizons=6.0,
            registry=joint_registry({"map@6h": {"validated": True}}),
            checkpoint_status="validated_target_gated_joint_research_only",
        )
        self.assertEqual(result["prediction_raw"], [[70.0]])
        self.assertFalse(result["forecasts"][0]["map"]["can_move"])

    def test_v2_contract_exposes_provenance_without_granting_causality(self):
        result = forecast_joint_values(
            prediction=np.asarray([[95.0]], dtype=float),
            current=np.asarray([[90.0]], dtype=float),
            variables=("glucose",),
            horizons=6.0,
            registry=joint_registry({
                    "glucose@6h": {
                        "validated": True,
                        "patient_conformal": True,
                        "hospital_conformal": True,
                        "care_unit_conformal": True,
                        "time_conformal": True,
                    }
                }),
            checkpoint_status="validated_target_gated_joint_research_only",
            lower=np.asarray([[85.0]]),
            upper=np.asarray([[105.0]]),
            observed=np.asarray([[True]]),
            measurement_age_hours=np.asarray([[0.5]]),
            actual_horizons=np.asarray([[5.75]]),
            organ_latent_sources=("endocrine",),
            observed_treatment_context=({"insulin_rate": 2.0},),
        )
        self.assertEqual(result["schema"], "whole_body_state_forecast.v2")
        cell = result["forecasts"][0]["glucose"]
        self.assertEqual(cell["observed_value"], 90.0)
        self.assertEqual(cell["forecast_point"], 95.0)
        self.assertEqual(cell["actual_horizon"], 5.75)
        self.assertEqual(cell["measurement_age"], 0.5)
        self.assertEqual(cell["organ_latent_source"], "endocrine")
        self.assertEqual(cell["interval_status"], "calibrated")
        self.assertFalse(cell["causal_claim_allowed"])

    def test_missing_anchor_uses_only_explicitly_validated_nowcast(self):
        result = forecast_joint_values(
            prediction=np.asarray([[2.0, 3.0]]),
            current=np.asarray([[np.nan, np.nan]]),
            variables=("creatinine", "bun"),
            horizons=24.0,
            registry=joint_registry({
                    "creatinine@24h": {"validated": True},
                    "bun@24h": {"validated": True},
                }),
            checkpoint_status="validated_target_gated_joint_research_only",
            observed=np.asarray([[False, False]]),
            nowcast=np.asarray([[1.4, 22.0]]),
            nowcast_validated=np.asarray([[True, False]]),
        )
        creatinine = result["forecasts"][0]["creatinine"]
        bun = result["forecasts"][0]["bun"]
        self.assertEqual(creatinine["current_estimate_source"], "validated_nowcast")
        self.assertEqual(creatinine["nowcast"], 1.4)
        self.assertTrue(creatinine["can_move"])
        self.assertEqual(bun["status"], "missing")
        self.assertIsNone(bun["forecast_point"])

    def test_sparse_target_requires_irregular_event_validation(self):
        result = forecast_joint_values(
            prediction=np.asarray([[1.8]]),
            current=np.asarray([[1.2]]),
            variables=("inr",),
            horizons=12.0,
            registry=joint_registry({
                    "inr@12h": {"validated": True},
                }),
            checkpoint_status="validated_target_gated_joint_research_only",
        )
        cell = result["forecasts"][0]["inr"]
        self.assertFalse(cell["can_move"])
        self.assertIn(
            "requires_irregular_time_event_validation",
            cell["rejection_reasons"],
        )

    def test_interval_evidence_is_fail_closed(self):
        result = forecast_joint_values(
            prediction=np.asarray([[95.0]]),
            current=np.asarray([[90.0]]),
            variables=("glucose",),
            horizons=6.0,
            registry=joint_registry({
                    "glucose@6h": {
                        "validated": True,
                        "patient_conformal": True,
                        "hospital_conformal": True,
                    }
                }),
            checkpoint_status="validated_target_gated_joint_research_only",
            lower=np.asarray([[85.0]]),
            upper=np.asarray([[105.0]]),
        )
        self.assertEqual(
            result["forecasts"][0]["glucose"]["interval_status"],
            "needs_calibration_audit",
        )

    def test_raw_gate_cannot_be_used_as_runtime_registry(self):
        result = forecast_joint_values(
            prediction=np.asarray([[95.0]]),
            current=np.asarray([[90.0]]),
            variables=("glucose",),
            horizons=6.0,
            registry={"target_horizon_registry": {"glucose@6h": {"validated": True}}},
            checkpoint_status="validated_target_gated_joint_research_only",
        )
        cell = result["forecasts"][0]["glucose"]
        self.assertFalse(cell["can_move"])
        self.assertIn(
            "joint_promotion_registry_v2_required",
            cell["rejection_reasons"],
        )

    def test_contract_and_unit_mismatch_fail_closed(self):
        registry = joint_registry({"urine_output@3h": {"validated": True}})
        registry["input_contract_id"] = "joint_interval_urine.v1"
        registry["variable_units"] = {"urine_output": "mL/hour"}
        mismatch = forecast_joint_values(
            prediction=[[80.0]],
            current=[[60.0]],
            variables=("urine_output",),
            horizons=3.0,
            registry=registry,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id="joint_asof_whole_body_state.v3",
            variable_units={"urine_output": "mL"},
        )
        cell = mismatch["forecasts"][0]["urine_output"]
        self.assertFalse(cell["can_move"])
        self.assertIn("input_contract_mismatch", cell["rejection_reasons"])
        self.assertIn("variable_unit_mismatch", cell["rejection_reasons"])

        matched = forecast_joint_values(
            prediction=[[80.0]],
            current=[[60.0]],
            variables=("urine_output",),
            horizons=3.0,
            registry=registry,
            checkpoint_status="validated_target_gated_joint_research_only",
            input_contract_id="joint_interval_urine.v1",
            variable_units={"urine_output": "mL/hour"},
        )
        self.assertTrue(matched["forecasts"][0]["urine_output"]["can_move"])

    def test_sparse_runtime_requires_source_specific_registry(self):
        result = forecast_sparse_irregular_value(
            target="inr",
            query_horizon_hours=12.0,
            current=1.2,
            prediction=1.1,
            actual_horizon_hours=13.5,
            registry={
                "schema": "sparse_irregular_promotion_registry.v1",
                "target_horizon_registry": {
                    "inr@12h": {
                        "irregular_time_forecast_validated": True,
                        "source_model": "sparse_irregular_log_event",
                        "requirements": {"conformal": True},
                    }
                },
            },
            checkpoint_status="validated_sparse_event_research_only",
            lower=0.9,
            upper=1.3,
        )
        self.assertTrue(result["can_move"])
        self.assertEqual(result["source_model"], "sparse_irregular_log_event")
        self.assertEqual(result["interval_status"], "calibrated")


if __name__ == "__main__":
    unittest.main()
