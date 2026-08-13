import unittest

from personalization_demo.runtime import capabilities, forecast, make_frame, sample_payload
from osler_jepa.state_completion import complete_current_state


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
        precision = [
            item for item in personalized
            if item["interval_status"] == "validated_patient_specific_conformal"
        ]
        self.assertEqual(
            {(item["target"], item["horizon_hours"]) for item in precision},
            {
                ("creatinine", 3),
                ("creatinine", 12),
                ("creatinine", 24),
                ("bun", 3),
                ("bun", 6),
                ("bun", 12),
                ("bun", 24),
                ("bun", 48),
                ("urine_output", 6),
                ("urine_output", 12),
                ("map", 3),
                ("map", 6),
            },
        )
        self.assertTrue(all(item["lower"] < item["personalized"]["point"] for item in precision))
        self.assertTrue(all(item["upper"] > item["personalized"]["point"] for item in precision))
        uncalibrated = [item for item in personalized if item not in precision]
        self.assertTrue(all(item["lower"] is None for item in uncalibrated))
        self.assertTrue(all(item["upper"] is None for item in uncalibrated))
        self.assertEqual(
            output["contract"]["serialized_precision_cells"],
            [
                "bun@12h",
                "bun@24h",
                "bun@3h",
                "bun@48h",
                "bun@6h",
                "creatinine@12h",
                "creatinine@24h",
                "creatinine@3h",
                "map@3h",
                "map@6h",
                "urine_output@12h",
                "urine_output@6h",
            ],
        )
        self.assertFalse(output["safety_boundary"]["clinical_claim_allowed"])
        self.assertFalse(output["safety_boundary"]["causal_claim_allowed"])

    def test_capabilities_are_fail_closed(self):
        contract = capabilities()
        self.assertEqual(
            len(contract["forecast_policy"]["serialized_precision_cells"]), 12
        )
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

    def test_completion_layer_reports_source_for_red_cell_family(self):
        completions = complete_current_state({"hemoglobin": 10.0})
        self.assertAlmostEqual(completions["hematocrit"]["point_estimate"], 30.0)
        self.assertEqual(
            completions["hematocrit"]["source"],
            "same_group_calibrated_completion",
        )
        self.assertFalse(completions["hematocrit"]["clinical_claim_allowed"])

    def test_completion_layer_handles_more_formula_sibling_families(self):
        completions = complete_current_state({
            "bilirubin_total": 3.2,
            "bilirubin_direct": 1.1,
            "total_protein": 7.0,
            "albumin": 3.1,
            "total_cholesterol": 180.0,
            "hdl_cholesterol": 45.0,
            "triglycerides": 150.0,
            "ck_mb": 5.0,
            "cpk": 250.0,
            "pao2": 80.0,
            "fio2": 0.4,
            "respiratory_rate": 20.0,
            "tidal_volume": 500.0,
        })

        self.assertAlmostEqual(completions["bilirubin_indirect"]["point_estimate"], 2.1)
        self.assertAlmostEqual(completions["globulin"]["point_estimate"], 3.9)
        self.assertAlmostEqual(completions["non_hdl_cholesterol"]["point_estimate"], 135.0)
        self.assertAlmostEqual(completions["ldl_cholesterol"]["point_estimate"], 105.0)
        self.assertAlmostEqual(completions["ck_mb_index"]["point_estimate"], 2.0)
        self.assertAlmostEqual(completions["pf_ratio"]["point_estimate"], 200.0)
        self.assertAlmostEqual(completions["minute_ventilation"]["point_estimate"], 10.0)
        for key in (
            "bilirubin_indirect",
            "globulin",
            "non_hdl_cholesterol",
            "ldl_cholesterol",
            "ck_mb_index",
            "pf_ratio",
            "minute_ventilation",
        ):
            self.assertFalse(completions[key]["clinical_claim_allowed"])

    def test_completion_layer_handles_red_cell_indices(self):
        completions = complete_current_state({"rbc": 4.5, "mcv": 90.0, "mch": 30.0})
        self.assertAlmostEqual(completions["hematocrit"]["point_estimate"], 40.5)
        self.assertAlmostEqual(completions["hemoglobin"]["point_estimate"], 13.5)

        completions = complete_current_state({"hemoglobin": 13.5, "hematocrit": 40.5})
        self.assertAlmostEqual(completions["mchc"]["point_estimate"], 33.3333333333)


if __name__ == "__main__":
    unittest.main()
