import unittest

import pandas as pd

from aki_mechanism import predict_aki_mechanism
from eicu_aki_transition_extract import classify_aki_action


class EicuAkiRouterTests(unittest.TestCase):
    def test_classifies_core_aki_action_terms(self):
        self.assertIn("fluids", classify_aki_action("normal saline bolus"))
        self.assertIn("vasopressor", classify_aki_action("norepinephrine infusion"))
        self.assertIn("diuretics", classify_aki_action("furosemide IV"))
        self.assertIn("renal_replacement", classify_aki_action("CRRT"))
        self.assertIn("nephrotoxin", classify_aki_action("vancomycin IVPB"))

    def test_unknown_terms_do_not_create_actions(self):
        self.assertEqual(classify_aki_action("acetaminophen tablet"), ())

    def test_renal_mechanism_returns_nan_for_unsupported_target(self):
        frame = pd.DataFrame({"creatinine_t": [2.0], "bun_t": [40.0]})
        prediction = predict_aki_mechanism(frame, "sodium")
        self.assertTrue(pd.isna(prediction[0]))

    def test_renal_mechanism_predicts_slow_accumulation_under_stress(self):
        frame = pd.DataFrame({
            "creatinine_t": [2.0],
            "bun_t": [45.0],
            "urine_output_t": [5.0],
            "map_t": [52.0],
            "hist_vasopressor": [1.0],
            "hist_nephrotoxin": [1.0],
            "hist_renal_replacement": [0.0],
            "act_renal_replacement": [0.0],
        })

        creatinine = predict_aki_mechanism(frame, "creatinine")
        bun = predict_aki_mechanism(frame, "bun")

        self.assertGreater(creatinine[0], frame.loc[0, "creatinine_t"])
        self.assertGreater(bun[0], frame.loc[0, "bun_t"])

    def test_renal_mechanism_allows_clearance_under_rrt(self):
        frame = pd.DataFrame({
            "creatinine_t": [4.0],
            "bun_t": [90.0],
            "urine_output_t": [5.0],
            "map_t": [52.0],
            "hist_vasopressor": [1.0],
            "hist_nephrotoxin": [1.0],
            "hist_renal_replacement": [1.0],
            "act_renal_replacement": [1.0],
        })

        creatinine = predict_aki_mechanism(frame, "creatinine")
        bun = predict_aki_mechanism(frame, "bun")

        self.assertLess(creatinine[0], frame.loc[0, "creatinine_t"])
        self.assertLess(bun[0], frame.loc[0, "bun_t"])


if __name__ == "__main__":
    unittest.main()
