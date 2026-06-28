import unittest
from pathlib import Path
import tempfile

import pandas as pd

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS, get_body_system_config
from eicu_body_system_transition_extract import add_active_column, classify_action, read_restricted_stay_ids
from eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from eicu_sepsis_target_router import _feature_columns
from heme_coag_belief import (
    CoagulationReserveBelief,
    HEME_COAG_BELIEF_COLUMNS,
    HEME_COAG_STATE_COLUMNS,
    heme_coag_belief_features,
    heme_coag_state_features,
    placebo_heme_coag_belief_features,
)


class EicuBodySystemRouterTests(unittest.TestCase):
    def test_body_system_contracts_cover_major_missing_systems(self):
        self.assertIn("electrolyte_acid_base", BODY_SYSTEM_CONFIGS)
        self.assertIn("endocrine_stress", BODY_SYSTEM_CONFIGS)
        self.assertIn("gi_pancreatic_nutrition", BODY_SYSTEM_CONFIGS)
        self.assertIn("cardiac_injury", BODY_SYSTEM_CONFIGS)
        self.assertIn("musculoskeletal_rhabdo", BODY_SYSTEM_CONFIGS)
        self.assertIn("immune_inflammatory", BODY_SYSTEM_CONFIGS)
        self.assertIn("cardiovascular_instability", BODY_SYSTEM_CONFIGS)
        self.assertIn("acute_neuro", BODY_SYSTEM_CONFIGS)
        self.assertIn("hepatic_failure", BODY_SYSTEM_CONFIGS)
        self.assertIn("coagulopathy_heme", BODY_SYSTEM_CONFIGS)

    def test_expanded_labs_are_first_class_state_variables(self):
        expected = {
            "anion gap": "anion_gap",
            "chloride": "chloride",
            "calcium": "calcium",
            "magnesium": "magnesium",
            "phosphate": "phosphate",
            "albumin": "albumin",
            "AST (SGOT)": "ast",
            "lipase": "lipase",
            "troponin - I": "troponin_i",
            "CPK": "cpk",
            "CRP": "crp",
        }

        for lab_name, state_name in expected.items():
            self.assertEqual(LAB_TO_STATE[lab_name], state_name)
            self.assertIn(state_name, PLAUSIBLE)

    def test_classifies_new_body_system_actions(self):
        electrolyte = get_body_system_config("electrolyte_acid_base")
        endocrine = get_body_system_config("endocrine_stress")
        gi = get_body_system_config("gi_pancreatic_nutrition")
        cardiac = get_body_system_config("cardiac_injury")
        muscle = get_body_system_config("musculoskeletal_rhabdo")
        immune = get_body_system_config("immune_inflammatory")

        self.assertIn("magnesium_repletion", classify_action(electrolyte, "magnesium sulfate"))
        self.assertIn("bicarbonate", classify_action(electrolyte, "sodium bicarbonate"))
        self.assertIn("insulin", classify_action(endocrine, "regular insulin infusion"))
        self.assertIn("dextrose", classify_action(endocrine, "dextrose 50%"))
        self.assertIn("systemic_steroid", classify_action(endocrine, "hydrocortisone"))
        self.assertIn("ppi", classify_action(gi, "pantoprazole"))
        self.assertIn("octreotide", classify_action(gi, "octreotide"))
        self.assertIn("nutrition", classify_action(gi, "TPN"))
        self.assertIn("antiarrhythmic", classify_action(cardiac, "amiodarone"))
        self.assertIn("anticoagulant", classify_action(cardiac, "heparin"))
        self.assertIn("inotrope", classify_action(cardiac, "dobutamine"))
        self.assertNotIn("anticoagulant", classify_action(cardiac, "troponin"))
        self.assertIn("bicarbonate", classify_action(muscle, "sodium bicarbonate"))
        self.assertIn("renal_replacement", classify_action(muscle, "crrt"))
        self.assertIn("systemic_steroid", classify_action(immune, "methylprednisolone"))
        self.assertIn("antibiotics", classify_action(immune, "vancomycin"))

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

    def test_heme_coag_belief_features_are_fixed_and_finite(self):
        frame = pd.DataFrame({
            "hemoglobin_t": [6.8, 12.0],
            "hematocrit_t": [21.0, 36.0],
            "platelets_t": [45.0, 220.0],
            "inr_t": [2.5, 1.1],
            "ptt_t": [80.0, 32.0],
            "fibrinogen_t": [90.0, 320.0],
            "map_t": [58.0, 82.0],
            "lactate_t": [4.5, 1.0],
            "hist_transfusion": [1.0, 0.0],
            "act_anticoagulant": [1.0, 0.0],
        })

        features = heme_coag_belief_features(frame)

        self.assertEqual(tuple(features.columns), HEME_COAG_BELIEF_COLUMNS)
        self.assertFalse(features.isna().any().any())
        self.assertGreater(features.loc[0, "belief_bleeding_burden"], features.loc[1, "belief_bleeding_burden"])
        self.assertLess(features.loc[0, "belief_coagulation_reserve"], features.loc[1, "belief_coagulation_reserve"])

    def test_heme_coag_placebo_is_capacity_matched(self):
        frame = pd.DataFrame({"hemoglobin_t": [7.0, 10.0, 12.0]})

        placebo = placebo_heme_coag_belief_features(frame, seed=17)

        self.assertEqual(placebo.shape, (len(frame), len(HEME_COAG_BELIEF_COLUMNS)))
        self.assertTrue(all(column.startswith("placebo_") for column in placebo.columns))

    def test_coagulation_reserve_belief_predict_update(self):
        row = pd.Series({
            "hemoglobin_t": 7.5,
            "hematocrit_t": 23.0,
            "platelets_t": 70.0,
            "inr_t": 1.8,
            "ptt_t": 55.0,
            "fibrinogen_t": 150.0,
            "hist_transfusion": 1.0,
            "act_anticoagulant": 0.0,
        })

        belief = CoagulationReserveBelief.from_row(row)
        predicted = belief.predict(row, delta_hours=6.0)
        updated = predicted.update(row)

        self.assertGreaterEqual(updated.mean, 0.0)
        self.assertLessEqual(updated.mean, 1.0)
        self.assertGreater(predicted.variance, belief.variance)

    def test_heme_coag_state_features_are_fixed_and_finite(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1],
            "hours_since_onset": [0.0, 6.0],
            "hemoglobin_t": [10.0, 7.8],
            "hematocrit_t": [30.0, 23.0],
            "platelets_t": [180.0, 80.0],
            "inr_t": [1.1, 2.0],
            "ptt_t": [32.0, 70.0],
            "fibrinogen_t": [300.0, 120.0],
            "hist_transfusion": [0.0, 1.0],
        })

        features = heme_coag_state_features(frame)

        self.assertEqual(tuple(features.columns), HEME_COAG_STATE_COLUMNS)
        self.assertFalse(features.isna().any().any())
        self.assertNotEqual(
            features.loc[0, "state_belief_coag_reserve_mean"],
            features.loc[1, "state_belief_coag_reserve_mean"],
        )

    def test_restricted_stay_ids_are_read_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stay_ids.csv"
            pd.DataFrame({"stay_id": [10, 10, 12, None]}).to_csv(path, index=False)

            stay_ids = read_restricted_stay_ids(path)

        self.assertEqual(stay_ids, {10, 12})


if __name__ == "__main__":
    unittest.main()
