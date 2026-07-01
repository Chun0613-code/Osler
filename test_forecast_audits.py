import unittest

import numpy as np

from eicu_conformal_coverage_audit import _module_api, conformal_quantile
from eicu_intermediate_horizon_audit import horizon_label, horizon_suffix, module_paths


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


if __name__ == "__main__":
    unittest.main()
