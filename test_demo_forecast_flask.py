import unittest

from demo.demo_app import SAMPLE_CASES, app
from demo.forecast_api import replay_forecast_request


class DemoForecastFlaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_contract_endpoints_are_available(self):
        capabilities = self.client.get("/api/forecast/capabilities")
        validation = self.client.get("/api/forecast/validation-summary")
        health = self.client.get("/api/health/models")
        self.assertEqual(capabilities.status_code, 200)
        self.assertEqual(validation.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.get_json()["ready"])

    def test_public_monitoring_case_returns_twelve_validated_cells(self):
        cases_response = self.client.get("/api/monitoring/cases")
        self.assertEqual(cases_response.status_code, 200)
        case = cases_response.get_json()["cases"][0]
        self.assertEqual(case["case_source"], "eicu_crd_demo_2.0.1")
        payload = replay_forecast_request(case, 2)
        response = self.client.post("/api/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        validated = [
            item for item in body["forecasts"]
            if item["tier"] == "validated_artifact"
        ]
        self.assertEqual(len(validated), 12)
        self.assertEqual(body["case_source"], "eicu_crd_demo_2.0.1")

    def test_future_data_is_rejected_over_http(self):
        case = self.client.get("/api/monitoring/cases").get_json()["cases"][0]
        payload = replay_forecast_request(case, 2)
        payload["trajectory"].append(case["observation_events"][2]["observation"])
        response = self.client.post("/api/forecast", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("after anchor_hour", response.get_json()["error"])

    def test_forecast_does_not_change_drug_ranking(self):
        analyze_payload = {"fields": SAMPLE_CASES[0], "use_openfda": False}
        before = self.client.post("/api/analyze", json=analyze_payload)
        self.assertEqual(before.status_code, 200)
        case = self.client.get("/api/monitoring/cases").get_json()["cases"][0]
        payload = replay_forecast_request(case, 2)
        forecast = self.client.post("/api/forecast", json=payload)
        self.assertEqual(forecast.status_code, 200)
        after = self.client.post("/api/analyze", json=analyze_payload)
        self.assertEqual(after.status_code, 200)
        before_drugs = [
            item["drug"] for item in before.get_json()["result"]["candidates"]
        ]
        after_drugs = [
            item["drug"] for item in after.get_json()["result"]["candidates"]
        ]
        self.assertEqual(after_drugs, before_drugs)

    def test_flask_serializes_full_replay_contract(self):
        response = self.client.get("/api/monitoring/cases")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        case = body["cases"][0]
        self.assertTrue(case["replay_metadata"]["retrospective"])
        self.assertTrue(case["replay_metadata"]["not_live"])
        self.assertEqual(len(case["observation_events"]), 3)
        self.assertIn("source_metadata", case)
        self.assertIn("safety_boundary", case)


if __name__ == "__main__":
    unittest.main()
