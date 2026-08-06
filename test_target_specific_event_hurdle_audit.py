import unittest

import numpy as np
import pandas as pd

from target_specific_event_hurdle_audit import (
    _adaptive_conformal_width,
    _adaptive_normalized_conformal_widths,
    _exclude_sibling_features,
    _feature_frame,
    _finite_conformal_width,
    _patient_bootstrap,
)


class TargetSpecificEventHurdleAuditTests(unittest.TestCase):
    def test_feature_frame_excludes_future_and_provenance(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["a", "a"],
                "anchor_time": pd.to_datetime(
                    ["2026-01-01", "2026-01-01 01:00"], format="mixed"
                ),
                "glucose_t": [100.0, 110.0],
                "glucose_age_hr": [0.0, 0.0],
                "_observed_glucose": [1.0, 1.0],
                "future_glucose": [110.0, 120.0],
                "hospitalid": [1, 1],
                "careunit": ["MICU", "MICU"],
                "hist_insulin_last_rate": [0.0, 1.0],
            }
        )
        features = _feature_frame(frame)
        self.assertIn("glucose_t", features)
        self.assertIn("slope1h__glucose_t", features)
        self.assertIn("hist_insulin_last_rate", features)
        self.assertNotIn("future_glucose", features)
        self.assertNotIn("hospitalid", features)
        self.assertNotIn("careunit", features)

    def test_conformal_width_uses_finite_sample_rank(self):
        width = _finite_conformal_width(np.arange(10, dtype=float), 0.90)
        self.assertEqual(width, 9.0)

    def test_patient_bootstrap_equalizes_rows_per_patient(self):
        frame = pd.DataFrame({"subject_id": ["a", "a", "a", "b", "b", "b"]})
        candidate = np.zeros(6)
        comparator = np.ones(6)
        report = _patient_bootstrap(
            frame,
            np.arange(6),
            candidate,
            comparator,
            seed=7,
        )
        self.assertEqual(report["patient_equal_delta"], -1.0)
        self.assertFalse(report["pass"])

    def test_adaptive_conformal_uses_only_calibration_patients(self):
        frame = pd.DataFrame(
            {"subject_id": np.repeat([f"p{i}" for i in range(40)], 3)}
        )
        residual = np.linspace(0.0, 1.0, len(frame))
        width, selected, coverage = _adaptive_conformal_width(
            frame, np.arange(len(frame)), residual, seed=9
        )
        self.assertTrue(np.isfinite(width))
        self.assertGreaterEqual(selected, 0.80)
        self.assertLessEqual(selected, 0.94)
        self.assertGreater(coverage, 0.80)

    def test_hemoglobin_excludes_hematocrit_siblings(self):
        features = pd.DataFrame(
            {
                "hemoglobin_t": [10.0],
                "hematocrit_t": [30.0],
                "lag1__hematocrit_t": [29.0],
                "platelets_t": [200.0],
            }
        )
        clean = _exclude_sibling_features(features, "hemoglobin")
        self.assertIn("hemoglobin_t", clean)
        self.assertIn("platelets_t", clean)
        self.assertNotIn("hematocrit_t", clean)
        self.assertNotIn("lag1__hematocrit_t", clean)

    def test_normalized_conformal_small_calibration_falls_back(self):
        frame = pd.DataFrame({"subject_id": ["a"] * 40})
        residual = np.linspace(0.01, 1.0, 40)
        features = np.column_stack([residual, residual ** 2])
        widths, report = _adaptive_normalized_conformal_widths(
            frame,
            np.arange(40),
            residual,
            features,
            features[:3],
            seed=3,
        )
        self.assertEqual(len(widths), 3)
        self.assertEqual(report["selected_nominal_coverage"], 0.90)


if __name__ == "__main__":
    unittest.main()
