import unittest

from target_specific_event_hurdle_registry import build_registry


class TargetSpecificEventHurdleRegistryTests(unittest.TestCase):
    def _report(self, target="o2sat", validated=True):
        passed = {
            "pass": validated,
            "change_threshold": 10.0 if target == "platelets" else 0.1,
        }
        return {
            "schema": "target_specific_event_hurdle_audit.v1",
            "hypoxemia_definition": "future_o2sat <= 92",
            "creatinine_change_definition": "abs(future-current) >= 0.1 mg/dL",
            "input_policy": "measurement-pure",
            "reports": {
                f"{target}@1h": {
                    "target": target,
                    "horizon_hours": 1,
                    "summary": {
                        "validated": validated,
                        "patient_passes": 7 if validated else 6,
                        "patient_required": 7,
                    },
                    "splits": {
                        "hospital": passed,
                        "careunit": passed,
                        "forward_time": passed,
                    },
                }
            },
        }

    def test_o2_event_is_not_registered_as_continuous_o2sat(self):
        output = build_registry(
            self._report(), candidate_id="o2_event", input_contract_id="joint.v1"
        )
        self.assertEqual(output["validated_target_horizon_cells"], ["hypoxemia_event@1h"])
        self.assertNotIn("o2sat@1h", output["target_horizon_registry"])
        self.assertEqual(
            output["output_contract"]["hypoxemia_event"]["task_type"],
            "binary_event_probability",
        )

    def test_failed_cell_stays_candidate_only(self):
        output = build_registry(
            self._report(validated=False),
            candidate_id="o2_event",
            input_contract_id="joint.v1",
        )
        self.assertEqual(output["validated_target_horizon_cells"], [])
        self.assertEqual(output["promotion_status"], "candidate_only")

    def test_generic_hurdle_target_keeps_units_threshold_and_contract(self):
        output = build_registry(
            self._report(target="platelets"),
            candidate_id="platelets_hurdle",
            input_contract_id="joint.v1",
        )
        self.assertEqual(output["validated_target_horizon_cells"], ["platelets@1h"])
        self.assertEqual(output["variable_units"], {"platelets": "K/uL"})
        contract = output["output_contract"]["platelets"]
        self.assertEqual(contract["change_threshold"], 10.0)
        self.assertEqual(contract["task_type"], "continuous_value_with_change_hurdle")
        self.assertIn("adaptive normalized", contract["interval"])
        cell = output["target_horizon_registry"]["platelets@1h"]
        self.assertEqual(cell["source_model"], "platelets_hurdle")
        self.assertTrue(cell["patient_conformal"])
        self.assertTrue(cell["hospital_conformal"])
        self.assertTrue(cell["care_unit_conformal"])
        self.assertTrue(cell["time_conformal"])

    def test_inconsistent_change_thresholds_fail_closed(self):
        report = self._report(target="platelets")
        cell = report["reports"]["platelets@1h"]
        cell["splits"]["extra"] = {"pass": True, "change_threshold": 20.0}
        with self.assertRaises(ValueError):
            build_registry(
                report,
                candidate_id="platelets_hurdle",
                input_contract_id="joint.v1",
            )

    def test_domain_conformal_schema_keeps_distinct_interval_contract(self):
        report = self._report(target="platelets")
        report["schema"] = "target_specific_event_hurdle_domain_conformal_audit.v1"
        output = build_registry(
            report,
            candidate_id="platelets_domain_hurdle",
            input_contract_id="joint.v1",
        )
        self.assertIn(
            "external-domain-heldout",
            output["output_contract"]["platelets"]["interval"],
        )


if __name__ == "__main__":
    unittest.main()
