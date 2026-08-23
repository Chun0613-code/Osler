import unittest

import pandas as pd

from osler_jepa.urine_output import (
    derive_interval_urine_rate,
    derive_timestamped_urine_rate,
)


class UrineOutputTests(unittest.TestCase):
    def test_uses_actual_collection_interval_and_excludes_non_volume(self):
        frame = pd.DataFrame(
            {
                "patientunitstayid": [1, 1, 1, 1, 1],
                "intakeoutputoffset": [0, 120, 180, 180, 240],
                "cellpath": [
                    "flowsheet|I&O|Output (ml)|Urine",
                ] * 5,
                "celllabel": [
                    "Urine",
                    "Urine",
                    "Urine",
                    "Urine Output-Foley",
                    "Urine Count",
                ],
                "cellvaluenumeric": [100.0, 200.0, 20.0, 30.0, 1.0],
            }
        )
        result = derive_interval_urine_rate(
            frame, id_column="patientunitstayid"
        )
        self.assertEqual(result["offset"].tolist(), [120, 180])
        self.assertEqual(result["valuenum"].tolist(), [100.0, 50.0])
        self.assertEqual(result["collection_interval_hr"].tolist(), [2.0, 1.0])
        self.assertTrue(result["quality"].eq("derived").all())

    def test_excludes_unknown_first_interval_and_retrospective_summary(self):
        frame = pd.DataFrame(
            {
                "patientunitstayid": [1, 1, 2],
                "intakeoutputoffset": [0, 1500, 60],
                "cellpath": [
                    "flowsheet|I&O|Output (ml)|Urine",
                ] * 3,
                "celllabel": ["Urine", "Urine", "Urine"],
                "cellvaluenumeric": [100.0, 500.0, 100.0],
            }
        )
        result = derive_interval_urine_rate(
            frame, id_column="patientunitstayid", maximum_interval_hours=24.0
        )
        self.assertTrue(result.empty)

    def test_timestamped_mimic_volume_becomes_rate_and_zero_is_observed(self):
        frame = pd.DataFrame(
            {
                "stay_id": [1, 1, 1, 2],
                "charttime": pd.to_datetime(
                    [
                        "2026-01-01 00:00",
                        "2026-01-01 02:00",
                        "2026-01-01 03:00",
                        "2026-01-01 00:00",
                    ]
                ),
                "volume_ml": [100.0, 200.0, 0.0, 700.0],
            }
        )

        result = derive_timestamped_urine_rate(
            frame,
            id_column="stay_id",
            time_column="charttime",
            value_column="volume_ml",
        )

        self.assertEqual(result["stay_id"].tolist(), [1, 1])
        self.assertEqual(result["valuenum"].tolist(), [100.0, 0.0])
        self.assertEqual(result["collection_interval_hr"].tolist(), [2.0, 1.0])
        self.assertTrue(result["quality"].eq("derived_interval_rate").all())


if __name__ == "__main__":
    unittest.main()
