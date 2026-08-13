import unittest

from demo.forecast_api import (
    SCHEMA_VERSION,
    capabilities_response,
    forecast_response,
    validation_summary_response,
)
from personalization_demo.runtime import sample_payload


class ForecastApiContractTests(unittest.TestCase):
    def test_sample_has_exactly_twelve_validated_artifact_cells(self):
        response = forecast_response(sample_payload())
        validated = [
            item for item in response["forecasts"]
            if item["tier"] == "validated_artifact"
        ]
        self.assertEqual(len(validated), 12)
        self.assertTrue(all(item["lower"] is not None for item in validated))
        self.assertTrue(all(item["upper"] is not None for item in validated))
        self.assertEqual(response["schema_version"], SCHEMA_VERSION)
        self.assertIn("model_version", response)
        self.assertEqual(response["case_source"], "user_supplied")
        self.assertIn("safety_boundary", response)

    def test_non_artifact_outputs_never_receive_intervals(self):
        response = forecast_response(sample_payload())
        non_artifacts = [
            item for item in response["forecasts"]
            if item["tier"] != "validated_artifact"
        ]
        self.assertTrue(non_artifacts)
        self.assertTrue(all(item["lower"] is None for item in non_artifacts))
        self.assertTrue(all(item["upper"] is None for item in non_artifacts))

    def test_unknown_cell_fails_closed(self):
        payload = sample_payload()
        payload["requested_cells"] = [{"target": "lactate", "horizon_hours": 24}]
        response = forecast_response(payload)
        self.assertEqual(len(response["forecasts"]), 1)
        item = response["forecasts"][0]
        self.assertEqual(item["tier"], "unsupported")
        self.assertIsNone(item["personalized"])
        self.assertIsNone(item["lower"])
        self.assertIsNone(item["upper"])

    def test_future_observation_after_anchor_is_rejected(self):
        payload = sample_payload()
        payload["anchor_hour"] = 4
        with self.assertRaisesRegex(ValueError, "after anchor_hour"):
            forecast_response(payload)

    def test_future_named_field_is_rejected(self):
        payload = sample_payload()
        payload["trajectory"][-1]["future_creatinine"] = 2.0
        with self.assertRaisesRegex(ValueError, "future/outcome field"):
            forecast_response(payload)

    def test_validation_summary_is_aggregate_only(self):
        summary = validation_summary_response()
        aggregate = summary["interval_width_reduction"]
        self.assertEqual(aggregate["scope"], "held_out_cohort_aggregate_only")
        self.assertFalse(aggregate["case_level_claim_allowed"])
        self.assertEqual(aggregate["validation_runs"], 120)
        self.assertLess(aggregate["maximum_percent"], 35.0)

    def test_forecast_has_no_drug_ranking_authority(self):
        safety = capabilities_response()["safety_boundary"]
        self.assertFalse(safety["treatment_recommendation_allowed"])
        self.assertFalse(safety["automated_prescribing_allowed"])
        self.assertFalse(safety["drug_ranking_changes_allowed"])


if __name__ == "__main__":
    unittest.main()
