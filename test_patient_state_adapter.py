import unittest
from pathlib import Path

import numpy as np
import torch

from osler_jepa.adjacent_hierarchy import AdjacentThreeLevelResidualAdapter
from osler_jepa.patient_state import (
    NeuralBeliefFilter,
    PatientStateAdapter,
    PatientStateResidualForecaster,
    load_patient_state_artifact,
    predict_patient_state_artifact,
)
from renal_patient_state_precision_registry import build_registry
from patient_state_adapter_audit import (
    _interval_narrowing_claim_allowed,
    _width_comparison,
)


class PatientStateAdapterTest(unittest.TestCase):
    def test_masked_padding_does_not_change_history_state(self):
        torch.manual_seed(7)
        model = NeuralBeliefFilter(input_dim=4, hidden=8)
        observed = torch.randn(2, 3, 4)
        padded = torch.cat((observed, torch.randn(2, 2, 4) * 100.0), dim=1)
        state_a, uncertainty_a = model(observed, torch.ones(2, 3, dtype=torch.bool))
        state_b, uncertainty_b = model(
            padded,
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool),
        )
        torch.testing.assert_close(state_a, state_b)
        torch.testing.assert_close(uncertainty_a, uncertainty_b)

    def test_zero_initialized_adapter_preserves_organ_anchor(self):
        torch.manual_seed(11)
        adapter = PatientStateAdapter(
            organ_dim=5,
            history_input_dim=4,
            belief_dim=3,
            hidden_dim=8,
        )
        organs = torch.randn(3, 2, 5)
        output = adapter(
            organ_latents=organs,
            organ_presence=torch.ones(3, 2),
            patient_history=torch.randn(3, 2, 4, 4),
            patient_history_mask=torch.ones(3, 2, 4, dtype=torch.bool),
            patient_belief=torch.randn(3, 2, 3),
        )
        torch.testing.assert_close(output["organ_latents"], organs)
        self.assertEqual(tuple(output["patient_state"].shape), (3, 2, 8))
        self.assertTrue(torch.all(output["uncertainty"] > 0.0))

    def test_absent_organ_cannot_receive_patient_residual(self):
        adapter = PatientStateAdapter(
            organ_dim=3,
            history_input_dim=2,
            belief_dim=1,
            hidden_dim=6,
        )
        with torch.no_grad():
            adapter.organ_residual.bias.fill_(1.0)
        output = adapter(
            organ_latents=torch.zeros(1, 2, 3),
            organ_presence=torch.tensor([[1.0, 0.0]]),
            patient_history=torch.zeros(1, 2, 2, 2),
            patient_history_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            patient_belief=torch.zeros(1, 2, 1),
        )
        self.assertTrue(torch.any(output["organ_residual"][:, 0] != 0.0))
        torch.testing.assert_close(
            output["organ_residual"][:, 1],
            torch.zeros_like(output["organ_residual"][:, 1]),
        )

    def test_formal_hierarchy_accepts_patient_state_only_at_organ_layer(self):
        model = AdjacentThreeLevelResidualAdapter(
            target_indices=(0,),
            target_module_membership=np.ones((1, 1), dtype=np.float32),
            latent_dim=4,
            organ_dim=3,
            module_count=1,
            organ_names=("kidneys",),
            adapter_dim=6,
            patient_history_dim=2,
            patient_belief_dim=1,
        )
        output = model(
            teacher_anchor=torch.tensor([[0.4], [0.7]]),
            context_latent=torch.randn(2, 4),
            target_latent=torch.randn(2, 1, 4),
            organ_latents=torch.randn(2, 1, 3),
            organ_presence=torch.ones(2, 1),
            current=torch.randn(2, 1),
            current_mask=torch.ones(2, 1),
            ages=torch.zeros(2, 1),
            horizon=torch.tensor([24.0, 24.0]),
            patient_history=torch.randn(2, 1, 3, 2),
            patient_history_mask=torch.ones(2, 1, 3, dtype=torch.bool),
            patient_belief=torch.randn(2, 1, 1),
        )
        self.assertIn("patient_state", output)
        self.assertEqual(
            model.patient_state_adapter.architecture_contract["placement"],
            "organ_layer",
        )
        parameter_names = set(dict(model.patient_state_adapter.named_parameters()))
        self.assertFalse(any("body" in name or "system" in name for name in parameter_names))

    def test_residual_forecaster_starts_at_population_anchor(self):
        model = PatientStateResidualForecaster(feature_dim=4, belief_dim=2, hidden=8)
        anchor = torch.tensor([0.2, -0.4, 1.1])
        prediction, scale = model(
            torch.randn(3, 4),
            torch.randn(3, 5, 5),
            torch.ones(3, 5, dtype=torch.bool),
            torch.randn(3, 2),
            anchor,
            torch.randn(3),
        )
        torch.testing.assert_close(prediction, anchor)
        self.assertTrue(torch.all(scale > 0.0))

    def test_promoted_artifact_is_loadable_when_present(self):
        artifact = Path("patient_state_creatinine24_v1")
        if not artifact.exists():
            self.skipTest("promoted artifact is not present")
        loaded = load_patient_state_artifact(artifact)
        self.assertEqual(loaded.metadata["target"], "creatinine")
        self.assertEqual(loaded.metadata["horizon_hours"], 24)
        self.assertEqual(
            len(loaded.preprocessing["conformal_alpha"]),
            len(loaded.preprocessing["conformal_quantile"]),
        )

    def test_matched_interval_width_requires_coverage_and_narrowing(self):
        subjects = np.repeat(np.arange(80), 2)
        narrower = _width_comparison(
            subjects,
            np.ones(160, dtype=np.float64),
            np.full(160, 2.0, dtype=np.float64),
            seed=17,
            bootstrap_samples=500,
        )
        self.assertTrue(narrower["candidate_significantly_narrower"])
        self.assertLess(narrower["patient_bootstrap_95_ci"][1], 0.0)
        self.assertTrue(
            _interval_narrowing_claim_allowed(
                narrower,
                candidate_coverage=0.90,
                population_coverage=0.90,
            )
        )
        self.assertFalse(
            _interval_narrowing_claim_allowed(
                narrower,
                candidate_coverage=0.90,
                population_coverage=0.86,
            )
        )

    def test_renal_precision_registry_is_fail_closed(self):
        registry = build_registry(Path(".").resolve())
        self.assertEqual(registry["evaluated_cells"], 20)
        self.assertEqual(
            registry["precision_promoted_cells"],
            [
                "creatinine@3h",
                "creatinine@12h",
                "creatinine@24h",
                "bun@3h",
                "bun@6h",
                "bun@12h",
                "bun@24h",
                "bun@48h",
                "urine_output@6h",
                "urine_output@12h",
                "map@3h",
                "map@6h",
            ],
        )
        self.assertTrue(registry["cells"]["creatinine@24h"]["precision_promoted"])
        self.assertIsNone(registry["cells"]["urine_output@1h"]["artifact"])

    def test_temporal_mondrian_artifact_selects_anchor_regime(self):
        loaded = load_patient_state_artifact("patient_state_precision_map3_v1")
        self.assertEqual(loaded.metadata["conformal_policy"], "temporal_mondrian")
        feature_count = len(loaded.metadata["feature_columns"])
        belief_count = int(loaded.preprocessing["belief_center"].shape[0])
        common = {
            "current_values": np.full(feature_count, np.nan, dtype=np.float32),
            "history_values": np.full((2, feature_count), np.nan, dtype=np.float32),
            "history_elapsed_hours": np.asarray([-3.0, 0.0], dtype=np.float32),
            "history_mask": np.asarray([True, True]),
            "belief_values": np.zeros(belief_count, dtype=np.float32),
            "current_target": 70.0,
        }
        early = predict_patient_state_artifact(
            loaded, anchor_hours_since_onset=12.0, **common
        )
        late = predict_patient_state_artifact(
            loaded, anchor_hours_since_onset=96.0, **common
        )
        self.assertGreater(early.half_width, 0.0)
        self.assertGreater(late.half_width, 0.0)
        self.assertNotEqual(early.half_width, late.half_width)

    def test_precision_artifact_returns_finite_validated_interval(self):
        loaded = load_patient_state_artifact("patient_state_precision_bun3_v1")
        feature_count = len(loaded.metadata["feature_columns"])
        belief_count = int(loaded.preprocessing["belief_center"].shape[0])
        prediction = predict_patient_state_artifact(
            loaded,
            current_values=np.full(feature_count, np.nan, dtype=np.float32),
            history_values=np.full((2, feature_count), np.nan, dtype=np.float32),
            history_elapsed_hours=np.asarray([-3.0, 0.0], dtype=np.float32),
            history_mask=np.asarray([True, True]),
            belief_values=np.zeros(belief_count, dtype=np.float32),
            current_target=25.0,
            anchor_hours_since_onset=12.0,
        )
        self.assertTrue(np.isfinite(prediction.personalized_point))
        self.assertGreater(prediction.half_width, 0.0)
        self.assertLess(prediction.lower, prediction.personalized_point)
        self.assertGreater(prediction.upper, prediction.personalized_point)


if __name__ == "__main__":
    unittest.main()
