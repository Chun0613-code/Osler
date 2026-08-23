import unittest

import pandas as pd

from mimiciv_observation_transition_extract import (
    ACTION_KEYS,
    assemble_treatment_context,
    classify_treatment_label,
    empty_treatment_evidence,
    treatment_window_summary,
)


class MimicIVTreatmentContextTests(unittest.TestCase):
    def test_classifies_common_mimic_treatments(self):
        self.assertIn("insulin", classify_treatment_label("Insulin - Regular infusion"))
        self.assertIn("vasopressor", classify_treatment_label("Norepinephrine"))
        self.assertIn("potassium_repletion", classify_treatment_label("Potassium Chloride KCL"))
        self.assertIn("renal_replacement", classify_treatment_label("CRRT continuous renal replacement"))
        self.assertIn("ventilation", classify_treatment_label("mechanical ventilation"))

    def test_treatment_window_summary_emits_history_and_future_context(self):
        lookup = {
            "insulin": {
                "starts": pd.Series([-2.0, 1.0]).to_numpy(dtype="float64"),
                "ends": pd.Series([-1.0, 2.0]).to_numpy(dtype="float64"),
                "sources": pd.Series(["mimic_emar", "mimic_emar"]).to_numpy(dtype=object),
                "dose_observed": pd.Series([True, True]).to_numpy(dtype=bool),
                "dose_values": pd.Series([3.0, 5.0]).to_numpy(dtype="float64"),
                "dose_dimensions": pd.Series(["drug_units", "drug_units"]).to_numpy(dtype=object),
                "rate_values": pd.Series([1.0, 2.0]).to_numpy(dtype="float64"),
                "rate_dimensions": pd.Series(["drug_units_per_hour", "drug_units_per_hour"]).to_numpy(dtype=object),
                "routes": pd.Series(["sc", "sc"]).to_numpy(dtype=object),
            }
        }

        summary = treatment_window_summary(lookup, anchor_hour=0.0, horizon_hours=6.0)

        self.assertEqual(summary["hist_insulin"], 1)
        self.assertEqual(summary["act_insulin"], 1)
        self.assertEqual(summary["hist_insulin_evidence_count"], 1)
        self.assertEqual(summary["act_insulin_evidence_count"], 1)
        self.assertEqual(summary["act_insulin_dose_observed"], 1)
        self.assertAlmostEqual(summary["hist_insulin_dose_drug_units_sum"], 3.0)
        self.assertAlmostEqual(
            summary["hist_insulin_rate_drug_units_per_hour_time_weighted_mean"],
            1.0,
        )
        self.assertIn("vasopressor", ACTION_KEYS)
        self.assertEqual(summary["act_vasopressor"], 0)

    def test_assemble_treatment_context_aligns_to_anchor_times(self):
        anchors = pd.DataFrame({
            "stay_id": [1],
            "t": [pd.Timestamp("2100-01-01 12:00:00")],
        })
        evidence = empty_treatment_evidence()
        evidence.loc[0] = {
            "stay_id": 1,
            "starttime": pd.Timestamp("2100-01-01 13:00:00"),
            "endtime": pd.Timestamp("2100-01-01 14:00:00"),
            "action": "vasopressor",
            "source": "mimic_inputevents",
            "dose_observed": True,
            "amount_like": 10.0,
            "rate_like": 0.2,
            "original_label": "Norepinephrine",
        }

        context = assemble_treatment_context(anchors, evidence, horizon_hours=6.0)

        self.assertEqual(int(context.loc[0, "act_vasopressor"]), 1)
        self.assertEqual(int(context.loc[0, "hist_vasopressor"]), 0)
        self.assertEqual(int(context.loc[0, "act_vasopressor_dose_observed"]), 1)

    def test_open_inputevent_keeps_rate_but_not_future_total_amount(self):
        lookup = {
            "vasopressor": {
                "starts": pd.Series([-1.0]).to_numpy(dtype="float64"),
                "ends": pd.Series([2.0]).to_numpy(dtype="float64"),
                "sources": pd.Series(["mimic_inputevents"]).to_numpy(dtype=object),
                "dose_observed": pd.Series([True]).to_numpy(dtype=bool),
                "dose_values": pd.Series([12.0]).to_numpy(dtype="float64"),
                "dose_dimensions": pd.Series(["mass_mg"]).to_numpy(dtype=object),
                "rate_values": pd.Series([1.5]).to_numpy(dtype="float64"),
                "rate_dimensions": pd.Series(["mass_mcg_per_kg_min"]).to_numpy(dtype=object),
                "routes": pd.Series(["iv"]).to_numpy(dtype=object),
            }
        }

        summary = treatment_window_summary(lookup, anchor_hour=0.0, horizon_hours=6.0)

        self.assertEqual(summary["hist_vasopressor"], 1)
        self.assertEqual(summary["hist_vasopressor_dose_mass_mg_sum"], 0.0)
        self.assertAlmostEqual(
            summary["hist_vasopressor_rate_mass_mcg_per_kg_min_time_weighted_mean"],
            1.5,
        )
        self.assertAlmostEqual(
            summary["hist_vasopressor_current_rate_mass_mcg_per_kg_min"],
            1.5,
        )

    def test_open_fluid_infusion_tracks_only_pre_anchor_delivered_volume(self):
        lookup = {
            "fluids": {
                "starts": pd.Series([-2.0]).to_numpy(dtype="float64"),
                "ends": pd.Series([2.0]).to_numpy(dtype="float64"),
                "sources": pd.Series(["mimic_inputevents"]).to_numpy(dtype=object),
                "dose_observed": pd.Series([True]).to_numpy(dtype=bool),
                "dose_values": pd.Series([400.0]).to_numpy(dtype="float64"),
                "dose_dimensions": pd.Series(["volume_ml"]).to_numpy(dtype=object),
                "rate_values": pd.Series([100.0]).to_numpy(dtype="float64"),
                "rate_dimensions": pd.Series(["volume_ml_per_hour"]).to_numpy(dtype=object),
                "routes": pd.Series(["iv"]).to_numpy(dtype=object),
            }
        }

        summary = treatment_window_summary(lookup, anchor_hour=0.0, horizon_hours=6.0)

        self.assertEqual(summary["hist_fluids_dose_volume_ml_sum"], 0.0)
        self.assertAlmostEqual(summary["hist_fluids_delivered_volume_ml_sum"], 200.0)
        self.assertAlmostEqual(summary["hist_fluids_current_rate_volume_ml_per_hour"], 100.0)


if __name__ == "__main__":
    unittest.main()
