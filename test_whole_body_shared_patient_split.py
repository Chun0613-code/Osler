"""Regression checks for true-patient splitting in shared module cohorts."""

from __future__ import annotations

import unittest

import pandas as pd

from whole_body_shared_gate import _patient_split_column, _split_by_column


class SharedPatientSplitTests(unittest.TestCase):
    def test_uses_original_patient_id_when_history_ids_are_namespaced(self):
        frame = pd.DataFrame({
            "subject_id": ["sepsis::10", "aki::10", "sepsis::11", "aki::11"],
            "_patient_subject_id": ["10", "10", "11", "11"],
        })
        self.assertEqual(_patient_split_column(frame), "_patient_subject_id")
        train, test = _split_by_column(frame, _patient_split_column(frame), seed=7, fraction=0.5)
        train_patients = set(frame.iloc[train]["_patient_subject_id"])
        test_patients = set(frame.iloc[test]["_patient_subject_id"])
        self.assertFalse(train_patients & test_patients)
        for patient in set(frame["_patient_subject_id"]):
            in_train = bool((frame.iloc[train]["_patient_subject_id"] == patient).any())
            in_test = bool((frame.iloc[test]["_patient_subject_id"] == patient).any())
            self.assertNotEqual(in_train, in_test)


if __name__ == "__main__":
    unittest.main()
