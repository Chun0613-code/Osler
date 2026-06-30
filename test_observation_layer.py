import unittest

from osler_jepa.observation_layer import (
    DENIED_AUTHORITIES,
    observation_readiness,
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


if __name__ == "__main__":
    unittest.main()
