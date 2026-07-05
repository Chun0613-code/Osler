import math
import unittest

import numpy as np

from osler_jepa.waveform_features import (
    aggregate_feature_summary,
    robust_univariate_features,
    waveform_array_features,
    WaveformFeatureRecord,
)


class WaveformFeatureTests(unittest.TestCase):
    def test_robust_univariate_features_ignore_nan(self):
        features = robust_univariate_features(np.array([1.0, 2.0, np.nan, 4.0]), "x")

        self.assertAlmostEqual(features["x_valid_fraction"], 0.75)
        self.assertAlmostEqual(features["x_median"], 2.0)
        self.assertGreater(features["x_iqr"], 0)

    def test_empty_channel_returns_nan_features(self):
        features = robust_univariate_features(np.array([np.nan, np.nan]), "x")

        self.assertEqual(features["x_valid_fraction"], 0.0)
        self.assertTrue(math.isnan(features["x_median"]))

    def test_waveform_array_features_group_core_signals(self):
        values = np.array(
            [
                [0.1, 70.0, 0.5, 0.2],
                [0.2, 80.0, 0.7, 0.3],
                [0.3, 60.0, 0.4, 0.4],
            ]
        )
        features = waveform_array_features(values, ["II", "ABP", "Pleth", "Resp"])

        self.assertIn("wave_ecg_1_median", features)
        self.assertIn("wave_arterial_pressure_1_median", features)
        self.assertIn("wave_arterial_pressure_1_hypotension_fraction_lt65", features)
        self.assertIn("wave_pleth_1_iqr", features)
        self.assertIn("wave_respiration_1_std", features)

    def test_aggregate_summary_has_no_prediction_authority(self):
        record = WaveformFeatureRecord(
            record_name="r1",
            source_path="/tmp/r1",
            sampling_frequency_hz=125.0,
            seconds_read=60.0,
            signals=("II", "ABP"),
            signal_groups=("arterial_pressure", "ecg"),
            features={"wave_ecg_1_median": 0.1},
        )

        summary = aggregate_feature_summary([record])

        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(summary["signal_group_counts"]["ecg"], 1)
        self.assertFalse(summary["boundary"]["prediction_authority"])
        self.assertFalse(summary["boundary"]["raw_samples_committed"])


if __name__ == "__main__":
    unittest.main()
