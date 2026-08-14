import unittest

from demo.forecast_api import (
    SCHEMA_VERSION,
    capabilities_response,
    forecast_response,
    monitoring_cases_response,
    replay_forecast_request,
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

    def test_anchor_must_equal_prefix_last_observation(self):
        payload = sample_payload()
        payload["anchor_hour"] = 9
        with self.assertRaisesRegex(ValueError, "must equal"):
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

    def test_replay_requests_are_strict_prefixes(self):
        case = monitoring_cases_response()["cases"][0]
        events = case["observation_events"]
        for visible_count in range(1, len(events) + 1):
            request = replay_forecast_request(case, visible_count)
            self.assertEqual(len(request["trajectory"]), visible_count)
            self.assertEqual(
                request["trajectory"],
                [event["observation"] for event in events[:visible_count]],
            )
            self.assertEqual(
                request["anchor_hour"],
                request["trajectory"][-1]["hours_since_onset"],
            )
            future_rows = [event["observation"] for event in events[visible_count:]]
            self.assertFalse(any(row in request["trajectory"] for row in future_rows))

    def test_monitoring_case_contract_is_ordered_and_unit_complete(self):
        response = monitoring_cases_response()
        self.assertIn("model_version", response)
        self.assertIn("safety_boundary", response)
        case = response["cases"][0]
        self.assertIn("source_metadata", case)
        self.assertIn("safety_boundary", case)
        replay = case["replay_metadata"]
        self.assertTrue(replay["retrospective"])
        self.assertTrue(replay["not_live"])
        self.assertEqual(replay["initial_visible_count"], 1)
        self.assertEqual(replay["recommended_step"], "next_observation")
        self.assertFalse(replay["forecast_observation_matching"]["supported"])
        units = replay["variable_units"]
        hours = []
        for event in case["observation_events"]:
            hours.append(event["available_at_hour"])
            self.assertEqual(
                event["available_at_hour"],
                event["observation"]["hours_since_onset"],
            )
            self.assertEqual(
                set(event["observation"]).difference(units),
                set(),
            )
            self.assertIn(event["signal"], {"stable", "watch", "research_signal"})
        self.assertEqual(hours, sorted(hours))
        self.assertEqual(len(hours), len(set(hours)))


if __name__ == "__main__":
    unittest.main()
