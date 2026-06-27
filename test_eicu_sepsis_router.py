import unittest

import pandas as pd

from eicu_sepsis_target_router import _feature_columns
from eicu_sepsis_transition_extract import classify_sepsis_action


class EicuSepsisRouterTests(unittest.TestCase):
    def test_classifies_core_sepsis_action_terms(self):
        self.assertIn("vasopressor", classify_sepsis_action("norepinephrine infusion"))
        self.assertIn("antibiotics", classify_sepsis_action("vancomycin IVPB"))
        self.assertIn("fluids", classify_sepsis_action("lactated ringer bolus"))
        self.assertIn("ventilation", classify_sepsis_action("mechanical ventilation"))
        self.assertIn("renal_replacement", classify_sepsis_action("CRRT"))

    def test_vasopressor_requirement_target_cannot_use_future_actions(self):
        frame = pd.DataFrame({
            "map_t": [62.0, 71.0],
            "map_age_hr": [0.5, 0.2],
            "vasopressor_requirement_t": [1.0, 0.0],
            "act_vasopressor": [1.0, 0.0],
            "act_vasopressor_evidence_count": [2.0, 0.0],
            "act_fluids": [1.0, 1.0],
            "hist_vasopressor": [1.0, 0.0],
            "hours_since_onset": [4.0, 8.0],
        })

        vaso_columns = _feature_columns(frame, "vasopressor_requirement")
        map_columns = _feature_columns(frame, "map")

        self.assertNotIn("act_vasopressor", vaso_columns)
        self.assertNotIn("act_vasopressor_evidence_count", vaso_columns)
        self.assertNotIn("act_fluids", vaso_columns)
        self.assertIn("hist_vasopressor", vaso_columns)
        self.assertIn("act_vasopressor", map_columns)


if __name__ == "__main__":
    unittest.main()
