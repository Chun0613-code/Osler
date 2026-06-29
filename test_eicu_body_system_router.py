import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS, get_body_system_config
from eicu_body_system_coupling_audit import _resolve_upstream_columns, summarize_reports
from eicu_body_system_transition_extract import (
    add_active_column,
    classify_action,
    classify_blood_product_subtype,
    evidence_window_summary,
    read_restricted_stay_ids,
    unit_like_count,
)
from eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from eicu_sepsis_target_router import _feature_columns
from body_temporal_coupling_belief import (
    temporal_coupling_features,
    temporal_resp_acid_base_features,
)
from body_edge_specific_coupling_belief import (
    edge_renal_electrolyte_buffering_features,
    edge_specific_coupling_features,
)
from heme_coag_belief import (
    CoagulationReserveBelief,
    HEME_COAG_BELIEF_COLUMNS,
    HEME_COAG_STATE_COLUMNS,
    heme_coag_belief_features,
    heme_coag_state_features,
    placebo_heme_coag_belief_features,
)
from osler_jepa.body_coupling import (
    body_system_coupling_edges,
    coupling_readiness_from_audit,
)


class EicuBodySystemRouterTests(unittest.TestCase):
    def test_body_system_contracts_cover_major_missing_systems(self):
        self.assertIn("integumentary_skin_wound", BODY_SYSTEM_CONFIGS)
        self.assertIn("toxic_metabolic", BODY_SYSTEM_CONFIGS)
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

    def test_classifies_skin_and_toxic_metabolic_actions(self):
        skin = get_body_system_config("integumentary_skin_wound")
        toxic = get_body_system_config("toxic_metabolic")

        self.assertIn("wound_care", classify_action(skin, "wound vac dressing change"))
        self.assertIn("topical_antimicrobial", classify_action(skin, "silver sulfadiazine cream"))
        self.assertIn("antibiotics", classify_action(skin, "vancomycin"))
        self.assertIn("nutrition", classify_action(skin, "TPN"))
        self.assertIn("antidote", classify_action(toxic, "naloxone"))
        self.assertIn("antidote", classify_action(toxic, "N-acetylcysteine"))
        self.assertIn("decontamination", classify_action(toxic, "activated charcoal"))
        self.assertIn("toxic_support", classify_action(toxic, "lipid emulsion"))
        self.assertIn("bicarbonate", classify_action(toxic, "sodium bicarbonate"))
        self.assertIn("renal_replacement", classify_action(toxic, "CRRT"))

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

    def test_classifies_blood_product_subtypes_and_units(self):
        self.assertEqual(classify_blood_product_subtype("packed red blood cells"), "prbc")
        self.assertEqual(classify_blood_product_subtype("fresh frozen plasma"), "plasma")
        self.assertEqual(classify_blood_product_subtype("platelet pheresis"), "platelet")
        self.assertEqual(classify_blood_product_subtype("cryoprecipitate"), "cryo")
        self.assertEqual(classify_blood_product_subtype("blood product"), "unknown")
        self.assertAlmostEqual(unit_like_count(300.0, "prbc"), 1.0)
        self.assertAlmostEqual(unit_like_count(2.0, "plasma"), 2.0)

    def test_transfusion_window_summary_keeps_subtype_and_dose_features(self):
        config = get_body_system_config("coagulopathy_heme")
        lookup = {
            "transfusion": {
                "starts": pd.Series([-2.0, 1.0, 2.0]).to_numpy(dtype="float64"),
                "ends": pd.Series([-1.0, 2.0, 3.0]).to_numpy(dtype="float64"),
                "dose_observed": pd.Series([True, True, False]).to_numpy(dtype=bool),
                "product_subtype": pd.Series(["prbc", "plasma", "unknown"]).to_numpy(dtype=object),
                "dose_amount": pd.Series([300.0, 250.0, float("nan")]).to_numpy(dtype="float64"),
                "unit_like_count": pd.Series([1.0, 1.0, float("nan")]).to_numpy(dtype="float64"),
            }
        }

        summary = evidence_window_summary(lookup, config, anchor_hour=0.0, horizon_hours=6.0)

        self.assertEqual(summary["hist_transfusion_prbc"], 1)
        self.assertEqual(summary["act_transfusion_plasma"], 1)
        self.assertEqual(summary["act_transfusion_unknown"], 1)
        self.assertEqual(summary["hist_transfusion_prbc_unit_like_count"], 1.0)
        self.assertEqual(summary["act_transfusion_plasma_volume_like_ml"], 250.0)
        self.assertEqual(summary["act_transfusion_plasma_dose_observed"], 1)

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

    def test_coupling_resolves_upstream_wildcard_features(self):
        frame = pd.DataFrame({
            "potassium_t": [4.0, 5.0],
            "potassium_tp6": [4.1, 4.8],
            "creatinine_t": [1.0, 2.0],
            "creatinine_age_hr": [1.0, 2.0],
            "act_renal_replacement": [0, 1],
            "act_renal_replacement_evidence_count": [0, 2],
            "act_renal_replacement_dose_observed": [0, 0],
            "map_t": [70.0, 65.0],
        })

        columns = _resolve_upstream_columns(
            frame,
            "potassium",
            ("creatinine_t", "creatinine_age_hr", "act_renal_replacement*"),
        )

        self.assertIn("creatinine_t", columns)
        self.assertIn("creatinine_age_hr", columns)
        self.assertIn("act_renal_replacement", columns)
        self.assertIn("act_renal_replacement_evidence_count", columns)
        self.assertNotIn("potassium_tp6", columns)

    def test_coupling_summary_counts_pass_both(self):
        reports = [{
            "targets": {
                "potassium": {
                    "active_only": {
                        "candidate_vs_baseline": {"significant": True, "point_delta": -0.1},
                        "candidate_vs_placebo": {"significant": True, "point_delta": -0.2},
                    },
                    "all_windows": {
                        "candidate_vs_baseline": {"significant": True, "point_delta": -0.1},
                        "candidate_vs_placebo": {"significant": False, "point_delta": 0.0},
                    },
                }
            }
        }]

        summary = summarize_reports(reports, ("potassium",))

        self.assertEqual(summary["potassium"]["active_only"]["candidate_passes_both_count"], 1)
        self.assertEqual(summary["potassium"]["all_windows"]["candidate_passes_both_count"], 0)

    def test_body_coupling_contract_includes_key_organ_edges(self):
        edges = {edge.name: edge for edge in body_system_coupling_edges()}

        self.assertIn("renal_to_electrolyte_acid_base", edges)
        self.assertIn("respiratory_to_acid_base", edges)
        self.assertIn("cardiovascular_to_renal", edges)
        self.assertIn("potassium", edges["renal_to_electrolyte_acid_base"].targets)
        self.assertEqual(edges["heme_to_perfusion"].horizon_hours, 24)

    def test_coupling_readiness_keeps_weak_signal_candidate_only(self):
        audit = {
            "edges": [{
                "name": "endocrine_to_electrolyte",
                "source_system": "endocrine_metabolic",
                "target_system": "electrolyte_acid_base",
                "targets": ("potassium", "sodium"),
                "random_patient_splits": {
                    "summary": {
                        "potassium": {
                            "active_only": {"candidate_passes_both_count": 1},
                        },
                        "sodium": {
                            "active_only": {"candidate_passes_both_count": 0},
                        },
                    },
                },
            }],
        }

        readiness = coupling_readiness_from_audit(audit)

        self.assertEqual(readiness["promoted_edges"], [])
        self.assertEqual(readiness["weak_candidate_edges"][0]["weak_candidate_targets"], ["potassium"])
        self.assertFalse(readiness["causal_claim_allowed"])

    def test_coupling_readiness_rejects_zero_pass_edges(self):
        audit = {
            "edges": [{
                "name": "immune_to_hemodynamics",
                "source_system": "immune_inflammatory",
                "target_system": "cardiovascular_perfusion",
                "targets": ("map",),
                "random_patient_splits": {
                    "summary": {
                        "map": {
                            "active_only": {"candidate_passes_both_count": 0},
                        },
                    },
                },
            }],
        }

        readiness = coupling_readiness_from_audit(audit)

        self.assertEqual(readiness["promoted_edges"], [])
        self.assertEqual(readiness["rejected_edges"][0]["name"], "immune_to_hemodynamics")

    def test_temporal_coupling_features_are_online_and_finite(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1, 1],
            "hours_since_onset": [0.0, 6.0, 12.0],
            "o2sat_t": [96.0, 88.0, 92.0],
            "o2sat_age_hr": [1.0, 1.0, 2.0],
            "respiratory_rate_t": [18.0, 34.0, 24.0],
            "respiratory_rate_age_hr": [1.0, 1.0, 2.0],
            "hist_ventilation": [0.0, 0.0, 1.0],
            "act_ventilation": [0.0, 1.0, 1.0],
            "hist_bronchodilator": [0.0, 0.0, 0.0],
            "act_bronchodilator": [0.0, 1.0, 0.0],
            "hist_systemic_steroid": [0.0, 0.0, 1.0],
            "act_systemic_steroid": [0.0, 0.0, 0.0],
        })

        features = temporal_resp_acid_base_features(frame)

        self.assertIn("tc_resp_burden_mean", features.columns)
        self.assertFalse(features.isna().all(axis=None))
        self.assertTrue(np.isfinite(features.to_numpy(dtype="float64")).all())
        self.assertNotEqual(features.loc[1, "tc_resp_burden_innovation"], 0.0)

    def test_temporal_coupling_dispatcher_rejects_unknown_edge(self):
        with self.assertRaises(KeyError):
            temporal_coupling_features(pd.DataFrame({"stay_id": [1]}), "not_a_body_edge")

    def test_edge_specific_renal_coupling_features_include_interactions(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1, 1],
            "hours_since_onset": [0.0, 6.0, 12.0],
            "creatinine_t": [1.1, 1.8, 2.2],
            "creatinine_age_hr": [1.0, 1.0, 1.0],
            "bun_t": [20.0, 35.0, 45.0],
            "bun_age_hr": [1.0, 1.0, 1.0],
            "urine_output_t": [100.0, 25.0, 15.0],
            "urine_output_age_hr": [1.0, 1.0, 1.0],
            "map_t": [75.0, 60.0, 58.0],
            "potassium_t": [4.2, 5.6, 5.9],
            "potassium_age_hr": [1.0, 1.0, 1.0],
            "bicarbonate_t": [24.0, 15.0, 12.0],
            "bicarbonate_age_hr": [1.0, 1.0, 1.0],
            "ph_t": [7.35, 7.20, 7.12],
            "ph_age_hr": [1.0, 1.0, 1.0],
            "anion_gap_t": [12.0, 22.0, 28.0],
            "anion_gap_age_hr": [1.0, 1.0, 1.0],
            "sodium_t": [140.0, 130.0, 128.0],
            "sodium_age_hr": [1.0, 1.0, 1.0],
            "serum_osmolality_t": [290.0, 310.0, 318.0],
            "serum_osmolality_age_hr": [1.0, 1.0, 1.0],
            "hist_renal_replacement": [0.0, 0.0, 0.0],
            "act_renal_replacement": [0.0, 0.0, 1.0],
            "hist_diuretics": [0.0, 0.0, 0.0],
            "act_diuretics": [0.0, 1.0, 1.0],
            "hist_potassium_repletion": [0.0, 0.0, 0.0],
            "act_potassium_repletion": [0.0, 0.0, 0.0],
            "hist_bicarbonate": [0.0, 0.0, 0.0],
            "act_bicarbonate": [0.0, 0.0, 1.0],
            "hist_fluids": [0.0, 1.0, 1.0],
            "act_fluids": [0.0, 1.0, 0.0],
        })

        features = edge_renal_electrolyte_buffering_features(frame)

        self.assertIn("esc_renal_low_reserve_x_acid_deficit", features.columns)
        self.assertIn("esc_renal_potassium_handling_stress_mean", features.columns)
        self.assertTrue(np.isfinite(features.to_numpy(dtype="float64")).all())
        self.assertGreater(
            features.loc[2, "esc_renal_acid_buffer_deficit_mean"],
            features.loc[0, "esc_renal_acid_buffer_deficit_mean"],
        )

    def test_edge_specific_coupling_dispatcher_rejects_unknown_edge(self):
        with self.assertRaises(KeyError):
            edge_specific_coupling_features(pd.DataFrame({"stay_id": [1]}), "not_a_body_edge")

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
