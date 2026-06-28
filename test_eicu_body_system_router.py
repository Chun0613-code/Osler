import unittest

import pandas as pd

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS, get_body_system_config
from eicu_body_system_transition_extract import add_active_column, classify_action
from eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from eicu_sepsis_target_router import _feature_columns


class EicuBodySystemRouterTests(unittest.TestCase):
    def test_body_system_contracts_cover_major_missing_systems(self):
        self.assertIn("cardiovascular_instability", BODY_SYSTEM_CONFIGS)
        self.assertIn("acute_neuro", BODY_SYSTEM_CONFIGS)
        self.assertIn("hepatic_failure", BODY_SYSTEM_CONFIGS)
        self.assertIn("coagulopathy_heme", BODY_SYSTEM_CONFIGS)

    def test_classifies_cardiovascular_actions(self):
        config = get_body_system_config("cardiovascular_instability")

        self.assertIn("vasopressor", classify_action(config, "norepinephrine infusion"))
        self.assertIn("inotrope", classify_action(config, "dobutamine drip"))
        self.assertIn("diuretics", classify_action(config, "furosemide IV"))
        self.assertIn("antiarrhythmic", classify_action(config, "amiodarone bolus"))

    def test_classifies_neuro_actions(self):
        config = get_body_system_config("acute_neuro")

        self.assertIn("antiepileptic", classify_action(config, "levetiracetam"))
        self.assertIn("osmotherapy", classify_action(config, "mannitol"))
        self.assertIn("ventilation", classify_action(config, "mechanical ventilation"))

    def test_classifies_hepatic_and_heme_actions(self):
        hepatic = get_body_system_config("hepatic_failure")
        heme = get_body_system_config("coagulopathy_heme")

        self.assertIn("hepatic_encephalopathy_tx", classify_action(hepatic, "lactulose"))
        self.assertIn("transfusion", classify_action(heme, "packed red blood cells"))
        self.assertIn("transfusion", classify_action(heme, "Volume-Transfuse plasma"))
        self.assertIn("anticoagulant", classify_action(heme, "heparin infusion"))
        self.assertIn("antiplatelet", classify_action(heme, "clopidogrel"))

    def test_heme_labs_are_first_class_state_variables(self):
        expected = {
            "Hgb": "hemoglobin",
            "Hct": "hematocrit",
            "PT - INR": "inr",
            "PTT": "ptt",
            "fibrinogen": "fibrinogen",
        }

        for lab_name, state_name in expected.items():
            self.assertEqual(LAB_TO_STATE[lab_name], state_name)
            self.assertIn(state_name, PLAUSIBLE)

    def test_coagulopathy_targets_include_first_class_heme_variables(self):
        config = get_body_system_config("coagulopathy_heme")

        for target in ("hemoglobin", "hematocrit", "inr", "ptt", "fibrinogen"):
            self.assertIn(target, config.targets)
            self.assertIn(target, config.state_vars)

    def test_active_rules_create_disease_specific_column(self):
        config = get_body_system_config("cardiovascular_instability")
        frame = pd.DataFrame({
            "map_t": [60.0, 80.0],
            "lactate_t": [1.0, 1.0],
            "heart_rate_t": [80.0, 85.0],
            "creatinine_t": [1.0, 1.0],
            "urine_output_t": [100.0, 100.0],
            "hist_vasopressor": [0, 0],
            "hist_inotrope": [0, 0],
        })

        active = add_active_column(frame, config)

        self.assertTrue(active.loc[0, "cardiovascular_instability_active_t"])
        self.assertFalse(active.loc[1, "cardiovascular_instability_active_t"])

    def test_feature_columns_exclude_body_system_future_targets(self):
        frame = pd.DataFrame({
            "map_t": [62.0, 71.0],
            "map_tp6": [68.0, 72.0],
            "lactate_tp24": [2.0, 1.8],
            "hist_vasopressor": [1.0, 0.0],
            "act_vasopressor": [1.0, 0.0],
        })

        features = _feature_columns(frame, "map")

        self.assertIn("map_t", features)
        self.assertIn("hist_vasopressor", features)
        self.assertIn("act_vasopressor", features)
        self.assertNotIn("map_tp6", features)
        self.assertNotIn("lactate_tp24", features)


if __name__ == "__main__":
    unittest.main()
