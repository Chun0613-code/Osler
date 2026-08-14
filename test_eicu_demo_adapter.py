import tempfile
import unittest
from pathlib import Path

import pandas as pd

from demo.eicu_demo_adapter import build_monitoring_case


class EicuDemoAdapterTests(unittest.TestCase):
    def test_adapter_uses_only_rows_at_or_before_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{
                "patientunitstayid": 7,
                "gender": "Female",
                "age": "60",
                "unittype": "Med-Surg ICU",
                "apacheadmissiondx": "source label",
            }]).to_csv(root / "patient.csv", index=False)
            pd.DataFrame([
                {"patientunitstayid": 7, "labresultoffset": 60, "labname": "BUN", "labresult": 20},
                {"patientunitstayid": 7, "labresultoffset": 60, "labname": "creatinine", "labresult": 1.0},
                {"patientunitstayid": 7, "labresultoffset": 120, "labname": "BUN", "labresult": 22},
                {"patientunitstayid": 7, "labresultoffset": 120, "labname": "creatinine", "labresult": 1.2},
                {"patientunitstayid": 7, "labresultoffset": 180, "labname": "creatinine", "labresult": 9.9},
            ]).to_csv(root / "lab.csv", index=False)
            pd.DataFrame([
                {
                    "patientunitstayid": 7,
                    "observationoffset": offset,
                    "heartrate": value,
                    "respiration": 18,
                    "sao2": 97,
                    "systemicmean": 70,
                    "temperature": 37,
                }
                for offset, value in ((50, 80), (110, 82), (170, 200))
            ]).to_csv(root / "vitalPeriodic.csv", index=False)
            pd.DataFrame([
                {
                    "patientunitstayid": 7,
                    "intakeoutputoffset": offset,
                    "cellpath": "flowsheet|I&O|Output (ml)|Urine",
                    "celllabel": "Urine",
                    "cellvaluenumeric": value,
                }
                for offset, value in ((30, 30), (90, 60), (150, 900))
            ]).to_csv(root / "intakeOutput.csv", index=False)

            case = build_monitoring_case(root, 7, anchor_offset_minutes=120)
            events = case["observation_events"]
            trajectory = [event["observation"] for event in events]
            self.assertEqual([row["hours_since_onset"] for row in trajectory], [1.0, 2.0])
            self.assertEqual(trajectory[-1]["creatinine"], 1.2)
            self.assertEqual(trajectory[-1]["heart_rate"], 82.0)
            self.assertNotIn(9.9, [row.get("creatinine") for row in trajectory])
            self.assertEqual(case["case_source"], "eicu_crd_demo_2.0.1")
            self.assertTrue(case["replay_metadata"]["retrospective"])
            self.assertTrue(case["replay_metadata"]["not_live"])
            self.assertEqual(
                [event["available_at_hour"] for event in events],
                [1.0, 2.0],
            )
            units = case["replay_metadata"]["variable_units"]
            self.assertTrue(all(set(row).issubset(units) for row in trajectory))


if __name__ == "__main__":
    unittest.main()
