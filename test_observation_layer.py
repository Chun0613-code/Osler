import unittest

from osler_jepa.observation_layer import (
    DENIED_AUTHORITIES,
    observation_readiness,
    rollout_readiness,
    trajectory_prediction_grid,
    uncertainty_calibration_policy,
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
            self.assertIn("factual_prediction", capability["allowed_uses"])
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
