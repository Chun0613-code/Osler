import unittest

import pandas as pd

from aki_mechanism import predict_aki_mechanism
from aki_renal_belief import (
    RENAL_BELIEF_COLUMNS,
    placebo_belief_features,
    renal_belief_features,
)
from eicu_aki_transition_extract import classify_aki_action
from eicu_sepsis_target_router import _feature_columns


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

    def test_feature_columns_exclude_long_horizon_future_targets(self):
        frame = pd.DataFrame({
            "creatinine_t": [1.2, 1.3],
            "creatinine_tp24": [1.8, 1.9],
            "bun_tp48": [60.0, 62.0],
            "map_age_hr": [1.0, 2.0],
            "act_fluids": [0, 1],
        })

        features = _feature_columns(frame)

        self.assertIn("creatinine_t", features)
        self.assertIn("map_age_hr", features)
        self.assertIn("act_fluids", features)
        self.assertNotIn("creatinine_tp24", features)
        self.assertNotIn("bun_tp48", features)

    def test_renal_belief_features_are_fixed_and_finite(self):
        frame = pd.DataFrame({
            "creatinine_t": [2.0, None],
            "bun_t": [45.0, None],
            "urine_output_t": [5.0, None],
            "map_t": [52.0, None],
            "creatinine_age_hr": [2.0, None],
            "bun_age_hr": [2.0, None],
            "urine_output_age_hr": [1.0, None],
            "hist_renal_replacement": [0.0, 1.0],
            "act_renal_replacement": [0.0, 0.0],
            "hist_vasopressor": [1.0, 0.0],
            "act_vasopressor": [0.0, 0.0],
        })

        features = renal_belief_features(frame)

        self.assertEqual(tuple(features.columns), RENAL_BELIEF_COLUMNS)
        self.assertFalse(features.isna().any().any())

    def test_placebo_belief_is_capacity_matched(self):
        frame = pd.DataFrame({"creatinine_t": [1.0, 2.0, 3.0]})

        placebo = placebo_belief_features(frame, seed=7)

        self.assertEqual(placebo.shape, (len(frame), len(RENAL_BELIEF_COLUMNS)))
        self.assertTrue(all(column.startswith("placebo_") for column in placebo.columns))


if __name__ == "__main__":
    unittest.main()
