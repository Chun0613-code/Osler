import json
import unittest

from demo.build_forecast_api_fixtures import (
    DEFAULT_FIXTURE_PATH,
    build_forecast_api_fixtures,
)


class ForecastApiFixtureTests(unittest.TestCase):
    def test_committed_fixture_matches_live_contract_exactly(self):
        committed = json.loads(DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(committed, build_forecast_api_fixtures())

    def test_prefix_fixtures_never_include_future_observations(self):
        fixtures = build_forecast_api_fixtures()
        cases = fixtures["get_monitoring_cases"]["response"]["cases"]
        events = cases[0]["observation_events"]
        first = fixtures["post_forecast_first_prefix"]["request"]
        second = fixtures["post_forecast_second_prefix"]["request"]
        self.assertEqual(len(first["trajectory"]), 1)
        self.assertEqual(len(second["trajectory"]), 2)
        self.assertEqual(first["trajectory"], [events[0]["observation"]])
        self.assertEqual(
            second["trajectory"],
            [events[0]["observation"], events[1]["observation"]],
        )
        self.assertEqual(first["anchor_hour"], events[0]["available_at_hour"])
        self.assertEqual(second["anchor_hour"], events[1]["available_at_hour"])
        self.assertNotIn(events[2]["observation"], second["trajectory"])

    def test_fixture_covers_validated_unsupported_and_unhealthy_states(self):
        fixtures = build_forecast_api_fixtures()
        second = fixtures["post_forecast_second_prefix"]["response"]
        validated = [
            item for item in second["forecasts"]
            if item["tier"] == "validated_artifact"
        ]
        illustrative = [
            item for item in second["forecasts"]
            if item["tier"] == "illustrative"
        ]
        unsupported = fixtures["post_forecast_unsupported"]["response"]["forecasts"]
        unhealthy = fixtures["get_model_health_unhealthy"]
        self.assertEqual(len(validated), 12)
        self.assertTrue(all(item["lower"] is not None for item in validated))
        self.assertTrue(all(item["lower"] is None for item in illustrative))
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0]["tier"], "unsupported")
        self.assertIsNone(unsupported[0]["population"])
        self.assertIsNone(unsupported[0]["personalized"])
        self.assertIsNone(unsupported[0]["lower"])
        self.assertIsNone(unsupported[0]["upper"])
        self.assertEqual(unhealthy["status"], 503)
        self.assertFalse(unhealthy["response"]["ready"])


if __name__ == "__main__":
    unittest.main()
