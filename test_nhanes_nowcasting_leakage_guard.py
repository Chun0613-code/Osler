import unittest

from nhanes_nowcast_audit import leak_guarded_features


class NhanesNowcastingLeakageGuardTests(unittest.TestCase):
    def test_lipid_siblings_are_excluded_from_ldl_features(self):
        features, excluded = leak_guarded_features(
            "ldl_cholesterol",
            [
                "age",
                "sex",
                "total_cholesterol",
                "hdl_cholesterol",
                "triglycerides",
                "glucose",
            ],
        )

        self.assertEqual(
            excluded,
            ["hdl_cholesterol", "non_hdl_cholesterol", "total_cholesterol", "triglycerides"],
        )
        self.assertEqual(features, ["age", "sex", "glucose"])

    def test_body_size_and_red_cell_siblings_are_excluded(self):
        body_features, body_excluded = leak_guarded_features(
            "bmi",
            ["age", "weight", "height", "waist", "glucose"],
        )
        self.assertEqual(body_excluded, ["height", "waist", "weight"])
        self.assertEqual(body_features, ["age", "glucose"])

        red_features, red_excluded = leak_guarded_features(
            "hemoglobin",
            ["age", "hematocrit", "rbc", "mcv", "platelets"],
        )
        self.assertEqual(red_excluded, ["hematocrit", "mch", "mchc", "mcv", "rbc", "rdw"])
        self.assertEqual(red_features, ["age", "platelets"])

    def test_inflammatory_siblings_are_excluded(self):
        features, excluded = leak_guarded_features(
            "hs_crp",
            ["age", "sex", "albumin", "glucose", "crp", "crp_hs"],
        )
        self.assertEqual(excluded, ["crp", "crp_hs", "esr", "ferritin"])
        self.assertEqual(features, ["age", "sex", "albumin", "glucose"])


if __name__ == "__main__":
    unittest.main()
