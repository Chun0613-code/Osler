import hashlib
import json
import unittest
from pathlib import Path

import torch

from pure_jepa_runtime import load_checkpoint


class ModelReleaseManifestTests(unittest.TestCase):
    def test_all_released_models_match_manifest_and_declared_status(self):
        root = Path(__file__).resolve().parent
        manifest = json.loads(
            (root / "MODEL_RELEASE_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["causal_claim_allowed"])
        self.assertFalse(manifest["clinical_promotion_allowed"])
        self.assertEqual(len(manifest["artifacts"]), 16)

        for artifact in manifest["artifacts"]:
            path = root / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            self.assertEqual(path.stat().st_size, artifact["bytes"], artifact["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["sha256"], artifact["path"])

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint.get("model_type"), artifact["model_type"])
            self.assertEqual(checkpoint.get("promotion_status"), artifact["status"])

            if artifact["model_type"] == "SparseAwarePureJEPABodyModel":
                model, scaler, metadata = load_checkpoint(path)
                self.assertEqual(tuple(scaler.variables), tuple(metadata["variables"]))
                self.assertFalse(model.training)

    def test_only_joint_checkpoint_is_candidate_only(self):
        root = Path(__file__).resolve().parent
        manifest = json.loads(
            (root / "MODEL_RELEASE_MANIFEST.json").read_text(encoding="utf-8")
        )
        candidates = [
            artifact for artifact in manifest["artifacts"]
            if artifact["status"] == "candidate_only"
        ]
        self.assertEqual(
            [artifact["path"] for artifact in candidates],
            ["whole_body_joint_jepa_hospital_invariant_15module_smoke_20260717.pt"],
        )


if __name__ == "__main__":
    unittest.main()
