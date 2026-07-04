import unittest

import pandas as pd

from eicu_respiratory_transition_extract import classify_respiratory_action
from eicu_sepsis_target_router import _feature_columns


class EicuRespiratoryRouterTests(unittest.TestCase):
    def test_classifies_core_respiratory_action_terms(self):
        self.assertIn("ventilation", classify_respiratory_action("mechanical ventilation"))
        self.assertIn("bronchodilator", classify_respiratory_action("albuterol nebulizer"))
        self.assertIn("systemic_steroid", classify_respiratory_action("methylprednisolone IV"))
        self.assertIn("antibiotics", classify_respiratory_action("ceftriaxone"))
        self.assertIn("fluids", classify_respiratory_action("normal saline bolus"))
        self.assertIn("vasopressor", classify_respiratory_action("norepinephrine infusion"))

    def test_unknown_terms_do_not_create_actions(self):
        self.assertEqual(classify_respiratory_action("acetaminophen tablet"), ())

    def test_feature_columns_exclude_future_respiration_targets(self):
        frame = pd.DataFrame({
            "o2sat_t": [88.0, 94.0],
            "o2sat_tp6": [92.0, 95.0],
            "respiratory_rate_tp24": [18.0, 20.0],
            "o2sat_age_hr": [0.5, 1.0],
            "act_ventilation": [1, 0],
            "hist_ventilation": [0, 1],
        })

        features = _feature_columns(frame, "o2sat")

        self.assertIn("o2sat_t", features)
        self.assertIn("o2sat_age_hr", features)
        self.assertIn("act_ventilation", features)
        self.assertIn("hist_ventilation", features)
        self.assertNotIn("o2sat_tp6", features)
        self.assertNotIn("respiratory_rate_tp24", features)


if __name__ == "__main__":
    unittest.main()
