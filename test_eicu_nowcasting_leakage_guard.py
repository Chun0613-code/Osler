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
            {"hematocrit", "rbc"},
        )
        self.assertTrue(is_leakage_sibling_column("hematocrit_t", "hemoglobin"))
        self.assertTrue(is_leakage_sibling_column("rbc_age_hr", "hemoglobin"))
        self.assertFalse(is_leakage_sibling_column("platelets_t", "hemoglobin"))

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


if __name__ == "__main__":
    unittest.main()
