import unittest

from eicu_nowcasting_audit import (
    is_leakage_sibling_column,
    leakage_sibling_targets,
    nowcast_feature_columns,
)


class EicuNowcastingLeakageGuardTests(unittest.TestCase):
    def test_red_cell_sibling_targets_are_guarded(self):
        self.assertEqual(
            leakage_sibling_targets("hemoglobin"),
            {"hematocrit", "rbc", "mcv", "mch", "mchc", "rdw"},
        )
        self.assertTrue(is_leakage_sibling_column("hematocrit_t", "hemoglobin"))
        self.assertTrue(is_leakage_sibling_column("rbc_age_hr", "hemoglobin"))
        self.assertFalse(is_leakage_sibling_column("platelets_t", "hemoglobin"))

    def test_formula_and_assay_sibling_targets_are_guarded(self):
        self.assertEqual(
            leakage_sibling_targets("anion_gap"),
            {"sodium", "chloride", "bicarbonate"},
        )
        self.assertIn("hdl_cholesterol", leakage_sibling_targets("ldl_cholesterol"))
        self.assertIn("triglycerides", leakage_sibling_targets("ldl_cholesterol"))
        self.assertIn("cpk", leakage_sibling_targets("ck_mb_index"))
        self.assertIn("ck_mb", leakage_sibling_targets("ck_mb_index"))
        self.assertIn("crp_hs", leakage_sibling_targets("crp"))
        self.assertTrue(is_leakage_sibling_column("total_cholesterol_t", "ldl_cholesterol"))
        self.assertTrue(is_leakage_sibling_column("cpk_age_hr", "ck_mb_index"))
        self.assertFalse(is_leakage_sibling_column("heart_rate_t", "ck_mb_index"))

    def test_feature_columns_exclude_formula_siblings(self):
        import pandas as pd

        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "stay_id": [10, 20],
            "hemoglobin_t": [10.0, 11.0],
            "hematocrit_t": [30.0, 33.0],
            "rbc_t": [3.4, 3.7],
            "platelets_t": [200.0, 210.0],
            "hours_since_onset": [0.0, 1.0],
        })
        features = nowcast_feature_columns(frame, "hemoglobin")
        self.assertNotIn("hematocrit_t", features)
        self.assertNotIn("rbc_t", features)
        self.assertIn("platelets_t", features)

    def test_feature_columns_exclude_lipid_and_muscle_siblings(self):
        import pandas as pd

        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "stay_id": [10, 20],
            "ldl_cholesterol_t": [90.0, 100.0],
            "total_cholesterol_t": [160.0, 180.0],
            "hdl_cholesterol_t": [45.0, 50.0],
            "triglycerides_t": [120.0, 150.0],
            "cpk_t": [100.0, 200.0],
            "ck_mb_t": [4.0, 8.0],
            "ck_mb_index_t": [4.0, 4.0],
            "heart_rate_t": [80.0, 88.0],
            "hours_since_onset": [0.0, 1.0],
        })
        lipid_features = nowcast_feature_columns(frame, "ldl_cholesterol")
        self.assertNotIn("total_cholesterol_t", lipid_features)
        self.assertNotIn("hdl_cholesterol_t", lipid_features)
        self.assertNotIn("triglycerides_t", lipid_features)
        self.assertIn("heart_rate_t", lipid_features)

        muscle_features = nowcast_feature_columns(frame, "ck_mb_index")
        self.assertNotIn("cpk_t", muscle_features)
        self.assertNotIn("ck_mb_t", muscle_features)
        self.assertIn("heart_rate_t", muscle_features)


if __name__ == "__main__":
    unittest.main()
