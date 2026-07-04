import unittest

from personalization_demo.runtime import capabilities, forecast, make_frame, sample_payload


class PersonalizationDemoTests(unittest.TestCase):
    def test_sample_forecast_uses_real_belief_contract(self):
        output = forecast(sample_payload())
        self.assertGreaterEqual(len(output["forecasts"]), 15)
        systems = {item["system"] for item in output["forecasts"]}
        self.assertIn("renal", systems)
        self.assertIn("cardiovascular", systems)
        self.assertIn("electrolyte", systems)
        self.assertIn("respiratory", systems)
        self.assertIn("endocrine", systems)
        targets = {item["target"] for item in output["forecasts"]}
        self.assertIn("glucose", targets)
        self.assertIn("o2sat", targets)
        self.assertIn("creatinine", targets)
        self.assertIn("heart_rate", targets)
        personalized = [
            item for item in output["forecasts"]
            if item.get("personalized") and item["personalized"]["point"] is not None
        ]
        self.assertTrue(personalized)
        self.assertFalse(output["contract"]["ridge_coefficients_serialized"])
        self.assertFalse(output["safety_boundary"]["clinical_claim_allowed"])
        self.assertFalse(output["safety_boundary"]["causal_claim_allowed"])

    def test_capabilities_are_fail_closed(self):
        contract = capabilities()
        self.assertFalse(contract["forecast_policy"]["ridge_coefficients_serialized"])
        self.assertFalse(contract["safety_boundary"]["clinical_claim_allowed"])
        self.assertFalse(contract["safety_boundary"]["counterfactual_claim_allowed"])
        self.assertFalse(contract["safety_boundary"]["treatment_recommendation_allowed"])

    def test_make_frame_computes_map_from_sbp_dbp(self):
        frame = make_frame({
            "trajectory": [
                {"hours_since_onset": 0, "sbp": 90, "dbp": 60},
            ],
        })
        self.assertAlmostEqual(frame.loc[0, "map_t"], 70.0)

    def test_make_frame_keeps_missing_values_missing(self):
        frame = make_frame({
            "trajectory": [
                {"hours_since_onset": 0, "glucose": "", "potassium": None},
            ],
        })
        self.assertTrue(frame["glucose_t"].isna().all())
        self.assertTrue(frame["potassium_t"].isna().all())
        self.assertFalse((frame[["glucose_t", "potassium_t"]] == 0).any().any())

    def test_make_frame_derives_anion_gap_from_basic_chemistry(self):
        frame = make_frame({
            "trajectory": [
                {"hours_since_onset": 0, "sodium": 140, "chloride": 104, "bicarbonate": 20},
            ],
        })
        self.assertAlmostEqual(frame.loc[0, "anion_gap_t"], 16.0)


if __name__ == "__main__":
    unittest.main()
