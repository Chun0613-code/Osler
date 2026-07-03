import unittest

from osler_jepa.observation_layer import (
    DENIED_AUTHORITIES,
    observation_readiness,
    rollout_readiness,
    nowcast_state_grid,
    trajectory_prediction_grid,
    uncertainty_calibration_policy,
    whole_body_state_forecast_template,
    whole_body_observation_capabilities,
)


class ObservationLayerTests(unittest.TestCase):
    def test_validated_multihop_and_candidates_are_explicit(self):
        capabilities = {
            capability.name: capability
            for capability in whole_body_observation_capabilities()
        }

        self.assertTrue(capabilities["sepsis_map6_renal24_multihop"].is_validated)
        self.assertEqual(
            capabilities["sepsis_map6_renal24_multihop"].targets,
            ("creatinine", "bun"),
        )
        self.assertEqual(
            capabilities["sepsis_map6_renal24_multihop"].horizon_hours,
            (24,),
        )

        self.assertFalse(
            capabilities["respiratory_acid_base_observed_coupling"].is_validated
        )
        self.assertFalse(capabilities["sepsis_map6_renal48_multihop"].is_validated)

        nowcast = capabilities["whole_body_same_time_nowcasting"]
        self.assertTrue(nowcast.is_validated)
        self.assertEqual(nowcast.horizon_hours, (0,))
        self.assertIn("same_time_state_completion", nowcast.allowed_uses)
        self.assertIn("bilirubin_direct", nowcast.targets)

        mimiciv = capabilities["mimiciv_cross_database_observation_validation"]
        self.assertTrue(mimiciv.is_validated)
        self.assertEqual(mimiciv.scope, "external_database_observation_validation")
        self.assertEqual(mimiciv.horizon_hours, (0, 6))
        self.assertIn("14_target_calibrated_interval_cells", mimiciv.targets)

        ed = capabilities["mimiciv_ed_scene_observation_validation"]
        self.assertTrue(ed.is_validated)
        self.assertEqual(ed.scope, "ed_scene_observation_validation")
        self.assertEqual(ed.horizon_hours, (0, 1, 3, 6))
        self.assertIn("6h_6_target_forecast_interval_cells", ed.targets)

        ed_to_icu = capabilities["mimiciv_ed_to_icu_baseline_candidate"]
        self.assertFalse(ed_to_icu.is_validated)
        self.assertEqual(ed_to_icu.scope, "ed_to_early_icu_baseline")
        self.assertEqual(ed_to_icu.horizon_hours, (1, 3, 6))
        self.assertIn("heart_rate_map_3h_6h_near_miss", ed_to_icu.targets)
        self.assertNotIn("factual_prediction", ed_to_icu.allowed_uses)

        nhanes = capabilities["nhanes_healthy_population_nowcast_validation"]
        self.assertTrue(nhanes.is_validated)
        self.assertEqual(nhanes.scope, "external_population_nowcast_validation")
        self.assertEqual(nhanes.horizon_hours, (0,))
        self.assertIn("29_cross_sectional_nowcast_targets", nhanes.targets)
        self.assertIn("hs_crp_fallback", nhanes.targets)
        self.assertIn("same_time_state_completion", nhanes.allowed_uses)
        self.assertNotIn("factual_prediction", nhanes.allowed_uses)
        self.assertIn("no forecast", nhanes.evidence)

        radiology_notes = capabilities["mimiciv_radiology_note_structured_observation_candidate"]
        self.assertFalse(radiology_notes.is_validated)
        self.assertEqual(radiology_notes.scope, "note_backed_measurement_depth")
        self.assertIn("rad_pleural_effusion", radiology_notes.targets)
        self.assertNotIn("factual_prediction", radiology_notes.allowed_uses)
        self.assertIn("structured_note_observation", radiology_notes.allowed_uses)

        neuro_notes = capabilities["eicu_neuro_note_structured_observation_candidate"]
        self.assertFalse(neuro_notes.is_validated)
        self.assertIn("neuro_gcs", neuro_notes.targets)
        self.assertIn("neuro_delirium_present", neuro_notes.targets)
        self.assertNotIn("same_time_state_completion", neuro_notes.allowed_uses)
        self.assertIn("EICU_NEURO_NOTE_ALL_ICU_FINDINGS.md", neuro_notes.evidence)
        self.assertIn("100,862 subjects", neuro_notes.evidence)

        cardiovascular_belief = capabilities["cardiovascular_heart_rate_belief_state"]
        self.assertTrue(cardiovascular_belief.is_validated)
        self.assertEqual(cardiovascular_belief.scope, "predict_update_belief_state")
        self.assertEqual(cardiovascular_belief.horizon_hours, (6,))
        self.assertIn("heart_rate", cardiovascular_belief.targets)
        self.assertIn("7/7 patient splits", cardiovascular_belief.evidence)
        self.assertIn("factual_prediction", cardiovascular_belief.allowed_uses)

        cardiovascular_remaining = capabilities["cardiovascular_belief_remaining_targets_candidate"]
        self.assertFalse(cardiovascular_remaining.is_validated)
        self.assertEqual(cardiovascular_remaining.scope, "predict_update_belief_state")
        self.assertIn("map", cardiovascular_remaining.targets)
        self.assertNotIn("heart_rate", cardiovascular_remaining.targets)
        self.assertNotIn("factual_prediction", cardiovascular_remaining.allowed_uses)

    def test_readiness_is_observation_only_and_fail_closed(self):
        readiness = observation_readiness()

        self.assertGreaterEqual(readiness["validated_count"], 8)
        self.assertGreaterEqual(readiness["candidate_count"], 1)

        boundary = readiness["safety_boundary"]
        self.assertTrue(boundary["observation_and_factual_prediction_allowed"])
        self.assertFalse(boundary["causal_claim_allowed"])
        self.assertFalse(boundary["counterfactual_claim_allowed"])
        self.assertFalse(boundary["clinical_claim_allowed"])
        self.assertFalse(boundary["runtime_decision_authority"])
        self.assertFalse(boundary["checkpoint_promotion_allowed"])
        self.assertFalse(boundary["active_rule_promotion_allowed"])

        for capability in readiness["validated_capabilities"]:
            self.assertTrue(
                "factual_prediction" in capability["allowed_uses"]
                or "same_time_state_completion" in capability["allowed_uses"]
            )
            for authority in DENIED_AUTHORITIES:
                self.assertIn(authority, capability["denied_authorities"])

    def test_trajectory_grid_moves_only_validated_target_horizons(self):
        cells = {
            (cell.target, cell.horizon_hours): cell
            for cell in trajectory_prediction_grid()
        }

        self.assertEqual(cells[("glucose", 6)].status, "validated")
        self.assertTrue(cells[("glucose", 6)].can_move)
        self.assertEqual(cells[("glucose", 1)].source, "persistence")
        self.assertFalse(cells[("glucose", 1)].can_move)

        self.assertEqual(cells[("creatinine", 24)].status, "validated")
        self.assertEqual(cells[("creatinine", 48)].status, "validated")
        self.assertFalse(cells[("creatinine", 6)].can_move)
        self.assertEqual(cells[("creatinine", 6)].status, "fallback_wrong_horizon")

        self.assertFalse(cells[("bilirubin", 24)].can_move)
        self.assertEqual(
            cells[("bilirubin", 24)].status,
            "fallback_observability_limited",
        )

    def test_whole_body_state_forecast_template_unifies_three_objects(self):
        nowcasts = nowcast_state_grid()
        self.assertEqual(len(nowcasts), 71)
        self.assertTrue(all(cell.time_axis == "current" for cell in nowcasts))
        self.assertTrue(all(cell.can_estimate for cell in nowcasts))
        self.assertTrue(all(not cell.can_move for cell in nowcasts))
        self.assertIn(
            ("hepatic_failure", "bilirubin_direct"),
            {(cell.module, cell.target) for cell in nowcasts},
        )

        template = whole_body_state_forecast_template()
        self.assertEqual(template["object_name"], "whole_body_state_forecast")
        self.assertFalse(template["runtime_values_included"])
        self.assertEqual(
            template["current_state"]["validated_nowcast_module_target_cells"],
            71,
        )
        self.assertGreaterEqual(
            template["future_trajectory"]["validated_cell_count"],
            154,
        )
        future_cells = template["future_trajectory"]["cells"]
        self.assertIn(
            ("respiratory", "paco2", 3),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["status"] == "validated"
            },
        )
        self.assertIn(
            ("sepsis", "platelets", 6),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["status"] == "validated"
            },
        )
        self.assertIn(
            ("sepsis", "platelets", 6),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["interval_status"] == "calibrated"
            },
        )
        self.assertFalse(template["safety_boundary"]["causal_claim_allowed"])
        self.assertFalse(template["safety_boundary"]["clinical_claim_allowed"])
        self.assertFalse(template["safety_boundary"]["runtime_decision_authority"])
        self.assertIn(
            "mimiciv_cross_database_coverage_audit.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "mimiciv_ed_observation_coverage_audit_6h.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "mimiciv_ed_to_icu_baseline_audit_6h.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "nhanes_2017_2018_nowcast_audit.json",
            template["source_artifacts"],
        )

    def test_uncertainty_policy_requires_calibration_before_intervals(self):
        policy = uncertainty_calibration_policy()

        self.assertEqual(policy["primary_coverage_target"], 0.9)
        self.assertTrue(policy["coverage_gate"]["required"])
        self.assertTrue(policy["coverage_gate"]["hospital_heldout_required"])
        self.assertIn("needs_calibration_audit", policy["missing_calibration_policy"])

        readiness = rollout_readiness()
        self.assertIn(1, readiness["horizons_hours"])
        self.assertIn(48, readiness["horizons_hours"])
        self.assertGreater(readiness["validated_cell_count"], 0)
        self.assertGreater(readiness["fallback_or_closed_cell_count"], 0)
        self.assertTrue(readiness["trajectory_policy"]["no_recursive_unvalidated_rollout"])


if __name__ == "__main__":
    unittest.main()
