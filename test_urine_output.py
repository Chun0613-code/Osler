import unittest

import pandas as pd

from osler_jepa.urine_output import derive_interval_urine_rate


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


if __name__ == "__main__":
    unittest.main()
