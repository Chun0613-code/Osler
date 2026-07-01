import unittest

import numpy as np
import pandas as pd

from eicu_conformal_coverage_audit import _module_api, conformal_quantile
from eicu_full_variable_coverage_audit import (
    eligible_current_targets,
    eligible_forecast_targets,
    forecast_feature_columns,
)
from eicu_intermediate_horizon_audit import horizon_label, horizon_suffix, module_paths
from eicu_nowcasting_audit import is_future_column, module_targets, nowcast_feature_columns


class ForecastAuditHelperTests(unittest.TestCase):
    def test_horizon_suffix_and_label_are_stable(self):
        self.assertEqual(horizon_suffix(1), "tp1")
        self.assertEqual(horizon_suffix(3.0), "tp3")
        self.assertEqual(horizon_suffix(1.5), "tp1p5")
        self.assertEqual(horizon_label(12), "12h")
        self.assertEqual(horizon_label(1.5), "1p5h")

    def test_conformal_quantile_uses_finite_sorted_residuals(self):
        residuals = np.asarray([0.1, 0.4, np.nan, 0.2, 0.3])
        self.assertEqual(conformal_quantile(residuals, 0.5), 0.3)
        self.assertEqual(conformal_quantile(residuals, 0.9), 0.4)
        self.assertIsNone(conformal_quantile(np.asarray([np.nan]), 0.9))

    def test_expanded_module_paths_and_apis_are_available(self):
        respiratory = module_paths("respiratory", 3)
        self.assertEqual(respiratory["cohort"].name, "eicu_respiratory_transitions_3h.parquet")
        body = module_paths("cardiovascular_instability", 12)
        self.assertEqual(body["router_report"].name, "eicu_cardiovascular_instability_target_router_12h.json")
        self.assertEqual(_module_api("respiratory")["active_scope"], "active_respiratory")
        self.assertEqual(_module_api("cardiovascular_instability")["active_scope"], "active")

    def test_nowcasting_feature_filter_stays_same_time_only(self):
        self.assertTrue(is_future_column("glucose_tp6"))
        self.assertTrue(is_future_column("creatinine_tp24"))
        self.assertTrue(is_future_column("fio2_tp1.5"))
        self.assertFalse(is_future_column("glucose_t"))
        self.assertFalse(is_future_column("hist_glucose_mean"))

        frame = pd.DataFrame({
            "subject_id": [1],
            "hospitalid": [2],
            "glucose_t": [120.0],
            "glucose_age_hr": [0.1],
            "map_t": [72.0],
            "map_age_hr": [0.2],
            "glucose_tp6": [150.0],
            "act_insulin_iv": [1.0],
            "active_sepsis": [True],
            "respiratory_active_t": [False],
            "hgb_sources": ["lab"],
            "hist_glucose_mean": [140.0],
            "hours_since_onset": [3.0],
        })
        features = nowcast_feature_columns(frame, "glucose")

        self.assertIn("map_t", features)
        self.assertIn("map_age_hr", features)
        self.assertIn("hist_glucose_mean", features)
        self.assertIn("hours_since_onset", features)
        self.assertNotIn("glucose_t", features)
        self.assertNotIn("glucose_age_hr", features)
        self.assertNotIn("glucose_tp6", features)
        self.assertNotIn("act_insulin_iv", features)
        self.assertNotIn("active_sepsis", features)
        self.assertNotIn("respiratory_active_t", features)
        self.assertNotIn("hgb_sources", features)

    def test_nowcasting_module_targets_are_registered(self):
        self.assertGreater(len(module_targets("sepsis")), 0)
        self.assertGreater(len(module_targets("respiratory")), 0)
        self.assertGreater(len(module_targets("cardiovascular_instability")), 0)
        with self.assertRaises(ValueError):
            module_targets("not_a_module")

    def test_full_variable_coverage_target_inventory_excludes_labels(self):
        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "hospitalid": [1, 1],
            "map_t": [70.0, 72.0],
            "map_tp6": [75.0, 76.0],
            "map_age_hr": [0.1, 0.2],
            "bilirubin_t": [1.2, np.nan],
            "sepsis_active": [True, False],
            "respiratory_active_t": [True, True],
            "act_fluid": [1.0, 0.0],
            "glucose_tp6": [130.0, 140.0],
            "hist_map_mean": [71.0, 73.0],
        })
        self.assertEqual(eligible_current_targets(frame), ("bilirubin", "map"))
        self.assertEqual(eligible_forecast_targets(frame, "tp6"), ("map",))

        features = forecast_feature_columns(frame, "map")
        self.assertIn("map_t", features)
        self.assertIn("map_age_hr", features)
        self.assertIn("act_fluid", features)
        self.assertIn("hist_map_mean", features)
        self.assertNotIn("map_tp6", features)
        self.assertNotIn("respiratory_active_t", features)


if __name__ == "__main__":
    unittest.main()
