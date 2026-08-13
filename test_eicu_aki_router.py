import unittest

import pandas as pd

from aki_mechanism import predict_aki_mechanism
from aki_renal_belief import (
    RENAL_BELIEF_COLUMNS,
    RENAL_STATE_BELIEF_COLUMNS,
    RENAL_STATE_V2_BELIEF_COLUMNS,
    RENAL_STATE_V3_BELIEF_COLUMNS,
    CreatinineKineticsBelief,
    RenalReserveBelief,
    placebo_belief_features,
    renal_belief_features,
    renal_belief_state_features,
    renal_belief_state_v2_features,
    renal_belief_state_v3_features,
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

    def test_renal_reserve_belief_predict_update(self):
        row = pd.Series({
            "creatinine_t": 2.0,
            "bun_t": 45.0,
            "urine_output_t": 5.0,
            "map_t": 52.0,
            "creatinine_age_hr": 2.0,
            "bun_age_hr": 2.0,
            "urine_output_age_hr": 1.0,
            "hist_vasopressor": 1.0,
            "hist_nephrotoxin": 1.0,
            "hist_renal_replacement": 0.0,
        })

        belief = RenalReserveBelief.from_row(row)
        predicted = belief.predict(row, delta_hours=4.0)
        updated = predicted.update(row)

        self.assertGreater(belief.mean, 0.0)
        self.assertGreater(predicted.variance, belief.variance)
        self.assertGreater(updated.mean, 0.0)
        self.assertLessEqual(updated.variance, predicted.variance)

    def test_renal_belief_state_features_are_fixed_and_finite(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1],
            "hours_since_onset": [0.0, 4.0],
            "creatinine_t": [2.0, 2.2],
            "bun_t": [45.0, 48.0],
            "urine_output_t": [5.0, 8.0],
            "map_t": [52.0, 60.0],
            "creatinine_age_hr": [2.0, 2.0],
            "bun_age_hr": [2.0, 2.0],
            "urine_output_age_hr": [1.0, 1.0],
            "hist_vasopressor": [1.0, 1.0],
            "hist_nephrotoxin": [1.0, 1.0],
            "hist_renal_replacement": [0.0, 0.0],
        })

        features = renal_belief_state_features(frame)

        self.assertEqual(tuple(features.columns), RENAL_STATE_BELIEF_COLUMNS)
        self.assertFalse(features.isna().any().any())
        self.assertNotEqual(
            features.loc[0, "state_belief_renal_reserve_mean"],
            features.loc[1, "state_belief_renal_reserve_mean"],
        )

    def test_creatinine_kinetics_belief_updates_slope(self):
        first = pd.Series({
            "creatinine_t": 1.2,
            "creatinine_age_hr": 1.0,
            "bun_t": 20.0,
            "urine_output_t": 80.0,
            "map_t": 75.0,
        })
        second = pd.Series({
            "creatinine_t": 1.8,
            "creatinine_age_hr": 1.0,
            "bun_t": 35.0,
            "urine_output_t": 20.0,
            "map_t": 60.0,
        })

        belief = CreatinineKineticsBelief.from_row(first)
        predicted = belief.predict(first, delta_hours=6.0)
        updated, innovation, confidence = predicted.update(second, delta_hours=6.0)

        self.assertGreater(innovation, 0.0)
        self.assertGreater(confidence, 0.0)
        self.assertGreater(updated.slope, 0.0)

    def test_renal_belief_state_v2_features_include_creatinine_dimension(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1],
            "hours_since_onset": [0.0, 6.0],
            "creatinine_t": [1.2, 1.8],
            "bun_t": [20.0, 35.0],
            "urine_output_t": [80.0, 20.0],
            "map_t": [75.0, 60.0],
            "creatinine_age_hr": [1.0, 1.0],
            "bun_age_hr": [2.0, 2.0],
            "urine_output_age_hr": [1.0, 1.0],
            "hist_vasopressor": [0.0, 1.0],
            "hist_nephrotoxin": [0.0, 1.0],
            "hist_renal_replacement": [0.0, 0.0],
        })

        features = renal_belief_state_v2_features(frame)

        self.assertEqual(tuple(features.columns), RENAL_STATE_V2_BELIEF_COLUMNS)
        self.assertFalse(features.isna().any().any())
        self.assertGreater(features.loc[1, "state2_belief_creatinine_slope"], 0.0)

    def test_renal_belief_state_v3_features_include_causal_urine_kinetics(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1, 1],
            "hours_since_onset": [0.0, 3.0, 6.0],
            "creatinine_t": [1.2, 1.3, 1.4],
            "bun_t": [20.0, 22.0, 24.0],
            "urine_output_t": [90.0, 45.0, 20.0],
            "map_t": [75.0, 68.0, 62.0],
            "creatinine_age_hr": [1.0, 1.0, 1.0],
            "bun_age_hr": [2.0, 2.0, 2.0],
            "urine_output_age_hr": [1.0, 1.0, 1.0],
        })

        full = renal_belief_state_v3_features(frame)
        prefix = renal_belief_state_v3_features(frame.iloc[:2])

        self.assertEqual(tuple(full.columns), RENAL_STATE_V3_BELIEF_COLUMNS)
        self.assertFalse(full.isna().any().any())
        self.assertLess(full.loc[1, "state3_belief_urine_slope"], 0.0)
        pd.testing.assert_series_equal(full.loc[1], prefix.loc[1])


if __name__ == "__main__":
    unittest.main()
