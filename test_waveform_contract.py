import tempfile
import unittest
from pathlib import Path

from mimiciv_waveform_manifest import scan_waveform_headers
from osler_jepa.waveform import (
    classify_signal_name,
    parse_wfdb_header_text,
    summarize_waveform_headers,
    waveform_feature_contract,
    waveform_readiness,
)


class WaveformContractTests(unittest.TestCase):
    def test_signal_classification_covers_core_mimic_waveform_labels(self):
        self.assertEqual(classify_signal_name("II"), "ecg")
        self.assertEqual(classify_signal_name("ABP"), "arterial_pressure")
        self.assertEqual(classify_signal_name("ART"), "arterial_pressure")
        self.assertEqual(classify_signal_name("PLETH"), "pleth")
        self.assertEqual(classify_signal_name("RESP"), "respiration")
        self.assertEqual(classify_signal_name("SpO2"), "oxygen_saturation")
        self.assertIsNone(classify_signal_name("unknown_signal"))

    def test_wfdb_header_parser_extracts_manifest_safe_fields(self):
        header = parse_wfdb_header_text(
            "\n".join(
                [
                    "p000001-2183-05-01-11-45 4 125 1250",
                    "p000001.dat 16 200/mV 11 1024 0 0 0 II",
                    "p000001.dat 16 200/mmHg 11 1024 0 0 0 ABP",
                    "p000001.dat 16 200/mV 11 1024 0 0 0 PLETH",
                    "p000001.dat 16 200/mV 11 1024 0 0 0 RESP",
                ]
            )
        )

        self.assertEqual(header.signal_count, 4)
        self.assertEqual(header.sampling_frequency_hz, 125.0)
        self.assertEqual(header.sample_count, 1250)
        self.assertEqual(header.duration_seconds, 10.0)
        self.assertEqual(header.signals, ("II", "ABP", "PLETH", "RESP"))
        self.assertEqual(
            header.signal_groups,
            ("arterial_pressure", "ecg", "pleth", "respiration"),
        )

    def test_manifest_summary_and_readiness_are_candidate_only(self):
        header = parse_wfdb_header_text(
            "\n".join(
                [
                    "record_a 3 250 2500",
                    "a.dat 16 200/mV 11 1024 0 0 0 II",
                    "a.dat 16 200/mmHg 11 1024 0 0 0 ART",
                    "a.dat 16 200/mV 11 1024 0 0 0 PLETH",
                ]
            )
        )

        summary = summarize_waveform_headers([header])
        readiness = waveform_readiness(summary)

        self.assertEqual(summary["record_count"], 1)
        self.assertIn("ecg", summary["signal_group_counts"])
        self.assertIn("arterial_pressure", summary["candidate_feature_groups"])
        self.assertTrue(readiness["data_available"])
        self.assertEqual(readiness["status"], "candidate_only")
        self.assertTrue(readiness["accuracy_audit_required"])
        self.assertFalse(readiness["factual_prediction_authority"])

    def test_no_manifest_fails_closed(self):
        readiness = waveform_readiness(None)

        self.assertFalse(readiness["data_available"])
        self.assertFalse(readiness["factual_prediction_authority"])
        self.assertIn("fail closed", readiness["reason"])

    def test_feature_contract_denies_forecast_authority_before_gate(self):
        contract = waveform_feature_contract()

        self.assertEqual(contract["status"], "candidate_only")
        self.assertFalse(contract["promotion_gate"]["claim_allowed_before_gate"])
        self.assertIn("heart_rate", contract["candidate_targets"])
        self.assertIn("counterfactual_treatment_effect", contract["denied_authorities"])

    def test_manifest_scanner_emits_aggregate_only_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.hea").write_text(
                "\n".join(
                    [
                        "sample 2 125 1250",
                        "sample.dat 16 200/mV 11 1024 0 0 0 II",
                        "sample.dat 16 200/mmHg 11 1024 0 0 0 ABP",
                    ]
                )
            )

            manifest = scan_waveform_headers(root)

        self.assertEqual(manifest["record_count"], 1)
        self.assertEqual(manifest["headers_scanned"], 1)
        self.assertFalse(manifest["boundary"]["raw_samples_read"])
        self.assertFalse(manifest["boundary"]["row_level_manifest_emitted"])
        self.assertFalse(manifest["boundary"]["forecast_feature_authority"])


if __name__ == "__main__":
    unittest.main()
