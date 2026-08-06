import json
import unittest
from pathlib import Path

from whole_body_runtime_candidate_catalog import build_catalog


class WholeBodyRuntimeCandidateCatalogTests(unittest.TestCase):
    def test_current_catalog_contains_46_unique_validated_cells(self):
        root = Path(__file__).resolve().parent
        payload = json.loads(
            (root / "whole_body_runtime_candidate_catalog_v9.json").read_text(
                encoding="utf-8"
            )
        )
        validated = {
            cell
            for candidate in payload["candidates"]
            for cell in candidate.get("validated_cells", [])
        }
        targets = {cell.split("@", 1)[0] for cell in validated}

        self.assertEqual(len(validated), 46)
        self.assertEqual(len(targets), 15)
        self.assertTrue(payload["factual_only"])
        self.assertFalse(payload["causal_claim_allowed"])
        self.assertFalse(payload["clinical_promotion_allowed"])

    def test_current_catalog_keeps_source_contracts_separate(self):
        root = Path(__file__).resolve().parent
        payload = json.loads(
            (root / "whole_body_runtime_candidate_catalog_v9.json").read_text(
                encoding="utf-8"
            )
        )
        by_id = {item["candidate_id"]: item for item in payload["candidates"]}

        shared = by_id["shared_module_nested_stack"]
        self.assertEqual(shared["input_contract_id"], "shared_module_row.v1")
        self.assertEqual(shared["validated_cell_count"], 25)

        aligned = by_id[
            "teacher_anchored_aligned_treatment_glucose_hierarchical_joint_jepa"
        ]
        self.assertEqual(
            aligned["input_contract_id"],
            "joint_asof_whole_body_state_aligned_treatment_interval_urine.v1",
        )
        self.assertEqual(aligned["validated_cells"], ["glucose@12h", "glucose@1h"])

        platelets = by_id[
            "platelets_domain_calibrated_change_hurdle_joint_jepa"
        ]
        self.assertEqual(platelets["validated_cells"], ["platelets@1h"])
        self.assertIn(
            "external-domain-heldout",
            platelets["output_contract"]["platelets"]["interval"],
        )

    def test_materialized_registry_keeps_its_contract_and_units(self):
        registry = {
            "schema": "whole_body_promotion_registry.v2",
            "candidate_id": "interval_urine_joint_jepa",
            "input_contract_id": "joint_interval_urine.v1",
            "input_contract": "interval-normalized urine output",
            "variable_units": {"urine_output": "mL/hour"},
            "output_contract": {
                "urine_output": {"task_type": "continuous_value"}
            },
            "promotion_status": "validated_target_gated_joint_research_only",
            "target_horizon_registry": {
                "urine_output@3h": {"validated_for_joint_runtime": True},
                "urine_output@6h": {"validated_for_joint_runtime": False},
            },
        }
        catalog = build_catalog({}, {}, [registry])
        entry = catalog["candidates"][-1]
        self.assertEqual(entry["validated_cells"], ["urine_output@3h"])
        self.assertEqual(entry["input_contract_id"], "joint_interval_urine.v1")
        self.assertEqual(entry["variable_units"]["urine_output"], "mL/hour")
        self.assertEqual(
            entry["output_contract"]["urine_output"]["task_type"],
            "continuous_value",
        )
        self.assertEqual(
            entry["runtime_status"], "target_gated_factual_research_only"
        )

    def test_wrong_registry_schema_is_candidate_only(self):
        catalog = build_catalog(
            {},
            {},
            [
                {
                    "schema": "raw_gate.v1",
                    "promotion_status": "validated_target_gated_joint_research_only",
                    "target_horizon_registry": {
                        "urine_output@3h": {"validated": True}
                    },
                }
            ],
        )
        entry = catalog["candidates"][-1]
        self.assertEqual(entry["validated_cells"], [])
        self.assertEqual(entry["runtime_status"], "candidate_only")


if __name__ == "__main__":
    unittest.main()
