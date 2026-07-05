import unittest

import numpy as np
import pandas as pd

from mimiciv_treatment_context_accuracy_audit import (
    audit_split,
    feature_sets,
    fit_ridge_residual,
)


class MimicIVTreatmentContextAccuracyAuditTests(unittest.TestCase):
    def test_feature_sets_separate_treatment_context_from_baseline(self):
        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "stay_id": [10, 20],
            "first_careunit": ["MICU", "SICU"],
            "glucose_t": [100.0, 120.0],
            "glucose_age_hr": [0.1, 0.2],
            "glucose_tp6": [130.0, 150.0],
            "map_t": [70.0, 72.0],
            "map_age_hr": [0.1, 0.2],
            "hist_insulin": [0, 1],
            "act_insulin": [1, 0],
            "act_insulin_evidence_count": [2, 0],
            "active_sepsis": [False, False],
        })

        sets = feature_sets(frame, "glucose")

        self.assertIn("map_t", sets["baseline"])
        self.assertNotIn("hist_insulin", sets["baseline"])
        self.assertNotIn("act_insulin", sets["baseline"])
        self.assertIn("hist_insulin", sets["candidate"])
        self.assertIn("act_insulin", sets["candidate"])
        self.assertIn("act_insulin_evidence_count", sets["treatment"])

    def test_fit_ridge_residual_supports_equal_capacity_noise(self):
        rows = []
        for subject in range(30):
            for idx in range(2):
                current = 100.0 + subject * 0.5 + idx
                treatment = float(subject % 2)
                rows.append({
                    "subject_id": subject,
                    "glucose_t": current,
                    "glucose_tp6": current + 5.0 * treatment,
                    "map_t": 70.0 + idx,
                    "act_insulin": treatment,
                    "truth": current + 5.0 * treatment,
                    "current": current,
                })
        frame = pd.DataFrame(rows)
        train = frame.iloc[:40].copy()
        predict = frame.iloc[40:].copy()

        baseline = fit_ridge_residual(
            train,
            predict,
            "glucose",
            ["map_t"],
            alpha=1.0,
            seed=1,
        )
        placebo = fit_ridge_residual(
            train,
            predict,
            "glucose",
            ["map_t"],
            alpha=1.0,
            seed=1,
            noise_count=1,
        )
        candidate = fit_ridge_residual(
            train,
            predict,
            "glucose",
            ["map_t", "act_insulin"],
            alpha=1.0,
            seed=1,
        )

        self.assertEqual(len(baseline), len(predict))
        self.assertEqual(len(placebo), len(predict))
        self.assertLess(
            np.abs(candidate - predict["truth"].to_numpy(dtype=float)).mean(),
            np.abs(baseline - predict["truth"].to_numpy(dtype=float)).mean(),
        )

    def test_audit_split_requires_candidate_to_beat_baseline_and_placebo(self):
        rows = []
        for subject in range(80):
            group = "train" if subject < 50 else "test"
            treatment = float(subject % 2)
            for idx in range(2):
                current = 100.0 + idx
                future = current + 10.0 * treatment
                rows.append({
                    "subject_id": subject,
                    "split_group": group,
                    "glucose_t": current,
                    "glucose_tp6": future,
                    "map_t": 70.0 + idx,
                    "act_insulin": treatment,
                    "truth": future,
                    "current": current,
                })
        frame = pd.DataFrame(rows)
        report = audit_split(
            frame,
            "glucose",
            ["map_t"],
            ["map_t", "act_insulin"],
            1,
            {"train"},
            {"test"},
            "split_group",
            seed=7,
            alpha=1.0,
            inner_folds=3,
            bootstrap_samples=100,
            min_pairs=20,
            min_subjects=10,
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertTrue(report["selected_candidate"])
        self.assertTrue(report["heldout"]["candidate_vs_baseline"]["significant"])
        self.assertTrue(report["heldout"]["candidate_vs_placebo"]["significant"])


if __name__ == "__main__":
    unittest.main()
