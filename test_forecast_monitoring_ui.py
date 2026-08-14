from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parent / "demo" / "static" / "forecast-monitoring.js"
).read_text(encoding="utf-8")


class ForecastMonitoringUiContractTests(unittest.TestCase):
    def test_replay_uses_backend_events_units_and_display_name(self):
        self.assertIn("c.observation_events.map", SCRIPT)
        self.assertIn("c.replay_metadata.variable_units", SCRIPT)
        self.assertIn("c.display_name", SCRIPT)
        self.assertNotIn("forecast_payload", SCRIPT)

    def test_backend_disabled_matching_has_no_ui_tolerance_fallback(self):
        self.assertIn("configured.supported === false", SCRIPT)
        self.assertIn("Awaiting / no configured matching", SCRIPT)
        self.assertNotIn("DEFAULT_MATCH_TOLERANCE", SCRIPT)
        self.assertNotIn("UI default tolerance", SCRIPT)

    def test_degraded_health_gates_every_forecast_request(self):
        self.assertIn("if (!healthy()) throw new Error('Forecast service unavailable')", SCRIPT)
        self.assertIn("if (healthy() && c)", SCRIPT)
        self.assertIn("if (!healthy()) return", SCRIPT)
        self.assertIn("integrity_load", SCRIPT)
        self.assertIn("inference_smoke", SCRIPT)

    def test_fixture_is_explicit_labeled_and_uses_committed_shape(self):
        self.assertIn("data-fm=\"fixture-load\"", SCRIPT)
        self.assertIn("/api/forecast/fixtures", SCRIPT)
        self.assertIn("post_forecast_", SCRIPT)
        self.assertIn("Cached research demonstration", SCRIPT)
        self.assertNotIn("/static/forecast-fixtures.json", SCRIPT)

    def test_null_intervals_are_not_coerced_and_coverage_is_target_specific(self):
        self.assertIn("v === null || v === undefined || v === ''", SCRIPT)
        self.assertIn("target-specific calibrated research interval", SCRIPT)
        self.assertIn("extend beyond physiological support", SCRIPT)
        self.assertNotIn("90% research interval", SCRIPT)


if __name__ == "__main__":
    unittest.main()
