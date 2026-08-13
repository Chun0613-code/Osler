import unittest

import numpy as np
import pandas as pd

from osler_jepa.interval_personalization import (
    mondrian_online_adaptive_interval_widths,
    online_adaptive_interval_widths,
    selective_interval_widths,
)


class IntervalPersonalizationTests(unittest.TestCase):
    def test_selective_policy_abstains_on_high_predicted_scale(self):
        scores = np.linspace(0.2, 1.2, 80)
        scales = np.concatenate((np.full(40, 0.5), np.full(40, 2.0)))
        subjects = np.asarray([f"p{i // 2}" for i in range(80)])
        test_scales = np.asarray([0.4, 0.6, 1.5, 2.5])

        result = selective_interval_widths(
            scores,
            scales,
            subjects,
            test_scales,
            selection_fraction=0.5,
        )

        self.assertEqual(result.selected.tolist(), [True, False, False, False])
        self.assertLess(result.widths[0], result.widths[-1])

    def test_online_update_waits_until_outcome_is_observable(self):
        calibration_scores = np.linspace(0.1, 2.0, 100)
        calibration_subjects = np.asarray([f"c{i // 2}" for i in range(100)])
        anchors = pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"]
        )
        future_times = pd.to_datetime(
            ["2026-01-01 02:00", "2026-01-01 04:00", "2026-01-01 05:00"]
        )

        result = online_adaptive_interval_widths(
            calibration_scores,
            calibration_subjects,
            np.asarray(["p1", "p1", "p1"]),
            anchors,
            future_times,
            np.ones(3),
            np.zeros(3),
            np.asarray([10.0, 0.0, 0.0]),
            min_resolved_history=1,
            gamma=0.05,
        )

        self.assertAlmostEqual(result.widths[0], result.widths[1])
        self.assertGreater(result.widths[2], result.widths[1])
        self.assertEqual(result.prior_resolved_count.tolist(), [0, 0, 1])
        self.assertEqual(result.personalized.tolist(), [False, False, True])

    def test_same_measurement_is_not_counted_more_than_once(self):
        calibration_scores = np.linspace(0.1, 2.0, 100)
        calibration_subjects = np.asarray([f"c{i // 2}" for i in range(100)])
        anchors = pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 03:00"]
        )
        future_times = pd.to_datetime(
            ["2026-01-01 02:00", "2026-01-01 02:00", "2026-01-01 05:00"]
        )

        result = online_adaptive_interval_widths(
            calibration_scores,
            calibration_subjects,
            np.asarray(["p1", "p1", "p1"]),
            anchors,
            future_times,
            np.ones(3),
            np.zeros(3),
            np.asarray([10.0, 10.0, 0.0]),
            min_resolved_history=1,
        )

        self.assertEqual(result.prior_resolved_count.tolist(), [0, 0, 1])

    def test_mondrian_policy_uses_only_matching_anchor_regime(self):
        calibration_scores = np.concatenate((np.ones(60), np.full(60, 10.0)))
        calibration_subjects = np.asarray([f"c{i}" for i in range(120)])
        calibration_regimes = np.asarray(["early"] * 60 + ["late"] * 60)
        anchors = pd.to_datetime(["2026-01-01", "2026-01-02"])
        futures = anchors + pd.to_timedelta(1, unit="h")

        result = mondrian_online_adaptive_interval_widths(
            calibration_scores,
            calibration_subjects,
            calibration_regimes,
            np.asarray(["p1", "p2"]),
            np.asarray(["early", "late"]),
            anchors,
            futures,
            np.ones(2),
            np.zeros(2),
            np.zeros(2),
        )

        self.assertLess(result.widths[0], result.widths[1])
        self.assertAlmostEqual(result.widths[1] / result.widths[0], 10.0)


if __name__ == "__main__":
    unittest.main()
