import unittest

import numpy as np
import pandas as pd
import torch

from teacher_anchored_joint_jepa_audit import (
    TeacherAnchoredResidualAdapter,
    _adaptive_patient_cluster_conformal_quantile,
    _adaptive_conformal_quantile,
    _crossfit_adaptive_patient_cluster_conformal_quantile,
    _crossfit_normalized_conformal,
    _distributional_adaptive_widths,
    _distributional_mondrian_widths,
    _distributional_asymmetric_widths,
    _distributional_shape_adaptive_widths,
    _future_label_support,
    _latent_svd_stability,
    _patient_cluster_conformal_quantile,
    _patient_coverage_diagnostics,
    _pcgrad,
    _sparsemax,
    _select_targets,
    _train_distributional_scale_heads,
)


class TeacherAnchoredJointJEPAAuditTests(unittest.TestCase):
    def test_all_target_selection_preserves_canonical_variable_order(self):
        variables = ("glucose", "creatinine", "heart_rate")
        self.assertEqual(_select_targets("all", variables), variables)
        self.assertEqual(
            _select_targets("heart_rate,missing,glucose", variables),
            ("heart_rate", "glucose"),
        )

    def test_future_label_support_records_unsupported_cells(self):
        frame = pd.DataFrame(
            {
                "subject_id": np.repeat(np.arange(25), 2),
                "horizon_hours": np.tile([3, 6], 25),
                "future_glucose": np.arange(50, dtype=float),
                "future_cortisol": np.nan,
            }
        )
        report = _future_label_support(
            frame, ("glucose", "cortisol"), (3, 6), minimum_rows=20
        )
        self.assertEqual(report["glucose"]["evaluable_horizons"], ["3h", "6h"])
        self.assertEqual(
            report["cortisol"]["support_limited_horizons"], ["3h", "6h"]
        )

    def test_distributional_mondrian_uses_calibration_state_only(self):
        subjects = np.repeat(np.arange(60), 4)
        frame = pd.DataFrame({"subject_id": subjects})
        calibration_rows = np.arange(180)
        test_rows = np.arange(180, 240)
        current = np.linspace(-2.0, 2.0, 240)
        residual = 0.5 + np.abs(current[:180])
        widths, metadata = _distributional_mondrian_widths(
            frame,
            calibration_rows,
            test_rows,
            residual,
            np.ones(180),
            np.ones(60),
            current[:180],
            current[180:],
            np.ones(180, dtype=bool),
            np.ones(60, dtype=bool),
        )
        self.assertEqual(widths.shape, (60,))
        self.assertTrue(np.all(widths > 0.0))
        self.assertTrue(metadata["cutpoints_fit_on_calibration_only"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertGreaterEqual(len(metadata["strata"]), 1)

    def test_distributional_adaptive_uses_disjoint_calibration_patients(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(90), 4)})
        rows = np.arange(360)
        rng = np.random.default_rng(41)
        residual = np.abs(rng.normal(0.0, 1.0, size=360))
        widths, metadata = _distributional_adaptive_widths(
            frame,
            rows,
            residual,
            np.ones(360),
            np.linspace(0.8, 1.2, 24),
            seed=43,
        )
        self.assertEqual(widths.shape, (24,))
        self.assertTrue(np.all(widths > 0.0))
        self.assertTrue(metadata["patient_disjoint_final_calibration"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertFalse(metadata["scale_head_test_labels_used"])
        self.assertGreaterEqual(metadata["selected_nominal_coverage"], 0.75)
        self.assertLessEqual(metadata["selected_nominal_coverage"], 0.90)

    def test_distributional_asymmetric_calibrates_tails_separately(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(30), 4)})
        rows = np.arange(120)
        lower, upper, metadata = _distributional_asymmetric_widths(
            frame,
            rows,
            np.linspace(-0.2, 0.8, 120),
            np.linspace(-0.1, 2.0, 120),
            np.ones(12),
        )
        self.assertEqual(lower.shape, (12,))
        self.assertEqual(upper.shape, (12,))
        self.assertTrue(np.all(upper > lower))
        self.assertFalse(metadata["test_labels_used"])
        self.assertAlmostEqual(metadata["tail_coverage"], 0.95)

    def test_distributional_shape_selection_keeps_final_patients_disjoint(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(80), 4)})
        rows = np.arange(320)
        rng = np.random.default_rng(43)
        error = rng.normal(0.4, 0.6, size=320)
        lower, upper, metadata = _distributional_shape_adaptive_widths(
            frame,
            rows,
            error,
            np.ones(320),
            np.ones(20),
            seed=47,
        )
        self.assertEqual(lower.shape, (20,))
        self.assertEqual(upper.shape, (20,))
        self.assertTrue(metadata["selection_final_patient_disjoint"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertIn(metadata["selected_shape"], ("symmetric", "asymmetric"))

    def _adapter(
        self,
        dynamic=True,
        sparse=False,
        distributional=False,
        target_specific=False,
    ):
        return TeacherAnchoredResidualAdapter(
            target_indices=(0, 1),
            target_module_membership=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            latent_dim=4,
            organ_dim=3,
            module_count=3,
            adapter_dim=8,
            dynamic_organ_route=dynamic,
            sparse_experts=sparse,
            distributional_scale=distributional,
            target_specific_projections=target_specific,
        )

    def _inputs(self):
        return {
            "teacher_anchor": torch.tensor([[0.2, -0.4], [0.3, -0.2]]),
            "context_latent": torch.randn(2, 4),
            "target_latent": torch.randn(2, 2, 4),
            "organ_latents": torch.randn(2, 3, 3),
            "organ_presence": torch.ones(2, 3),
            "current": torch.randn(2, 2),
            "current_mask": torch.ones(2, 2),
            "ages": torch.zeros(2, 2),
            "horizon": torch.tensor([6.0, 12.0]),
        }

    def test_zero_initialized_adapter_is_exact_teacher_anchor(self):
        adapter = self._adapter(dynamic=True)
        output = adapter(**self._inputs())
        torch.testing.assert_close(
            output["value"], self._inputs()["teacher_anchor"]
        )
        torch.testing.assert_close(
            output["residual"], torch.zeros_like(output["residual"])
        )

    def test_dynamic_route_excludes_targets_own_organ(self):
        adapter = self._adapter(dynamic=True)
        output = adapter(**self._inputs())
        self.assertTrue(
            torch.all(output["route_weights"][:, 0, 0] == 0.0)
        )
        self.assertTrue(
            torch.all(output["route_weights"][:, 1, 1] == 0.0)
        )
        torch.testing.assert_close(
            output["route_weights"].sum(dim=-1),
            torch.ones(2, 2),
        )

    def test_disabling_route_preserves_teacher_at_initialization(self):
        adapter = self._adapter(dynamic=True)
        output = adapter(**self._inputs(), disable_organ_route=True)
        torch.testing.assert_close(
            output["value"], self._inputs()["teacher_anchor"]
        )
        torch.testing.assert_close(
            output["route_weights"],
            torch.zeros_like(output["route_weights"]),
        )

    def test_sparsemax_produces_exact_sparse_simplex_weights(self):
        weights = _sparsemax(
            torch.tensor([[3.0, 1.0, -2.0], [0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
        self.assertTrue(torch.any(weights == 0.0))
        self.assertTrue(torch.all(weights >= 0.0))

    def test_sparse_experts_are_target_specific_and_normalized(self):
        adapter = self._adapter(dynamic=False, sparse=True)
        output = adapter(**self._inputs())
        torch.testing.assert_close(
            output["expert_weights"].sum(dim=-1), torch.ones(2, 2)
        )
        self.assertTrue(torch.all(output["expert_weights"] >= 0.0))
        torch.testing.assert_close(
            output["value"], self._inputs()["teacher_anchor"]
        )

    def test_sparse_expert_masks_missing_own_organ(self):
        adapter = self._adapter(dynamic=False, sparse=True)
        inputs = self._inputs()
        inputs["organ_presence"][:, 0] = 0.0
        output = adapter(**inputs)
        self.assertTrue(
            torch.all(output["expert_weights"][:, 0, 2] == 0.0)
        )

    def test_target_specific_projection_has_matched_disable_path(self):
        adapter = self._adapter(
            dynamic=False,
            sparse=True,
            target_specific=True,
        )
        inputs = self._inputs()
        initial = adapter(**inputs)
        disabled = adapter(
            **inputs, disable_target_specific_projections=True
        )
        torch.testing.assert_close(initial["value"], disabled["value"])
        with torch.no_grad():
            adapter.target_fast_adapters[0][-1].bias.fill_(0.5)
            adapter.residual_heads[0][-1].weight.fill_(0.1)
        enabled = adapter(**inputs)
        disabled = adapter(
            **inputs, disable_target_specific_projections=True
        )
        self.assertFalse(torch.equal(enabled["value"], disabled["value"]))

    def test_disabling_sparse_experts_restores_hierarchical_mixture(self):
        adapter = self._adapter(dynamic=False, sparse=True)
        output = adapter(**self._inputs(), disable_sparse_experts=True)
        torch.testing.assert_close(
            output["expert_weights"][..., 0]
            + output["expert_weights"][..., 1],
            torch.ones(2, 2),
        )
        self.assertTrue(
            torch.all(output["expert_weights"][..., 2] == 0.0)
        )

    def test_distributional_head_starts_positive_without_moving_teacher(self):
        adapter = self._adapter(
            dynamic=False, sparse=True, distributional=True
        )
        inputs = self._inputs()
        output = adapter(**inputs)
        torch.testing.assert_close(output["value"], inputs["teacher_anchor"])
        self.assertTrue(torch.all(output["scale"] > 0.0))
        torch.testing.assert_close(
            output["scale"],
            torch.full_like(output["scale"], 0.50),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    def test_distributional_loss_does_not_backprop_into_scale_inputs(self):
        adapter = self._adapter(
            dynamic=False, sparse=True, distributional=True
        )
        inputs = self._inputs()
        inputs["context_latent"].requires_grad_(True)
        output = adapter(**inputs)
        output["scale"].sum().backward()
        gradient = inputs["context_latent"].grad
        self.assertTrue(gradient is None or torch.all(gradient == 0.0))

    def test_scale_stage_does_not_change_point_predictor(self):
        torch.manual_seed(59)
        adapter = self._adapter(
            dynamic=False, sparse=True, distributional=True
        )
        rows = 24
        arrays = (
            None,
            None,
            np.zeros((rows, 2), dtype=np.float32),
            np.ones((rows, 2), dtype=np.float32),
            np.zeros((rows, 2), dtype=np.float32),
            np.random.default_rng(61).normal(size=(rows, 2)).astype(
                np.float32
            ),
            np.ones((rows, 2), dtype=np.float32),
            None,
            np.full(rows, 6.0, dtype=np.float32),
        )
        features = {
            "context": torch.randn(rows, 4),
            "target": torch.randn(rows, 2, 4),
            "organ": torch.randn(rows, 3, 3),
            "presence": torch.ones(rows, 3),
        }
        teacher = np.zeros((rows, 2), dtype=np.float32)
        point_before = {
            name: value.detach().clone()
            for name, value in adapter.named_parameters()
            if not name.startswith("scale_heads.")
        }
        scale_before = {
            name: value.detach().clone()
            for name, value in adapter.named_parameters()
            if name.startswith("scale_heads.")
        }
        report = _train_distributional_scale_heads(
            adapter,
            features,
            arrays,
            teacher,
            np.arange(rows),
            (0, 1),
            epochs=2,
            batch_size=8,
            seed=67,
        )
        self.assertTrue(report["point_predictor_frozen"])
        for name, value in adapter.named_parameters():
            if name in point_before:
                torch.testing.assert_close(value, point_before[name])
        self.assertTrue(
            any(
                not torch.equal(value, scale_before[name])
                for name, value in adapter.named_parameters()
                if name in scale_before
            )
        )

    def test_pcgrad_projects_a_direct_conflict(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        left = parameter[0] + parameter[1]
        right = -parameter[0] + 0.25 * parameter[1]
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        before, after = _pcgrad(
            [left, right],
            [parameter],
            np.random.default_rng(7),
        )
        self.assertTrue(any(value < 0.0 for value in before))
        self.assertTrue(all(value >= -1e-6 for value in after))
        self.assertIsNotNone(parameter.grad)

    def test_latent_svd_detects_stable_noncollapsed_modes(self):
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(240, 3))
        projection = rng.normal(size=(3, 12))
        values = latent @ projection + rng.normal(scale=0.01, size=(240, 12))
        features = {
            "context": torch.from_numpy(values[:, :4]).float(),
            "target": torch.from_numpy(values[:, 4:].reshape(240, 2, 4)).float(),
        }
        report = _latent_svd_stability(
            features,
            np.arange(0, 120),
            np.arange(120, 240),
        )
        self.assertTrue(report["pass"])
        self.assertGreater(report["fit_effective_rank"], 2.0)
        self.assertGreater(report["mean_top_subspace_cosine"], 0.5)

    def test_adaptive_conformal_uses_patient_disjoint_calibration(self):
        subject_ids = np.repeat(np.arange(30), 5)
        frame = pd.DataFrame({"subject_id": subject_ids})
        residuals = (
            np.linspace(0.05, 2.0, len(subject_ids))
            + (subject_ids % 3) * 0.01
        )
        quantile, rank, metadata = _adaptive_conformal_quantile(
            frame,
            np.arange(len(frame)),
            residuals,
            seed=17,
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(
            metadata["method"],
            "patient_grouped_cv_adaptive_split_conformal",
        )
        self.assertTrue(
            metadata["patient_disjoint_final_calibration"]
        )
        self.assertFalse(metadata["test_labels_used"])
        self.assertGreater(metadata["tuning_subjects"], 0)
        self.assertGreater(
            metadata["final_calibration_subjects"], 0
        )
        self.assertLessEqual(
            metadata["selected_nominal_coverage"], 0.90
        )

    def test_adaptive_conformal_falls_back_when_support_is_small(self):
        frame = pd.DataFrame({"subject_id": np.arange(20)})
        quantile, rank, metadata = _adaptive_conformal_quantile(
            frame,
            np.arange(20),
            np.linspace(0.1, 1.0, 20),
            seed=19,
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(
            metadata["method"], "fixed_split_conformal_fallback"
        )

    def test_normalized_conformal_crossfits_scale_by_patient(self):
        rng = np.random.default_rng(23)
        subject_ids = np.repeat(np.arange(40), 5)
        feature = rng.normal(size=(len(subject_ids), 3))
        residuals = np.abs(
            rng.normal(
                scale=0.5 + 0.8 * np.abs(feature[:, 0]),
                size=len(subject_ids),
            )
        )
        test_features = np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        widths, metadata = _crossfit_normalized_conformal(
            pd.DataFrame({"subject_id": subject_ids}),
            np.arange(len(subject_ids)),
            residuals,
            feature,
            test_features,
            seed=29,
        )
        self.assertEqual(
            metadata["method"],
            "patient_grouped_crossfit_normalized_conformal",
        )
        self.assertEqual(widths.shape, (2,))
        self.assertTrue(np.all(widths > 0.0))
        self.assertFalse(metadata["test_labels_used"])
        self.assertTrue(metadata["patient_grouped_scale_crossfit"])

    def test_patient_cluster_conformal_equalizes_long_stays(self):
        frame = pd.DataFrame(
            {
                "subject_id": (
                    ["dense"] * 100
                    + ["sparse_1"] * 2
                    + ["sparse_2"] * 2
                    + ["sparse_3"] * 2
                )
            }
        )
        residuals = np.asarray(
            [0.1] * 100 + [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        )
        quantile, rank, metadata = (
            _patient_cluster_conformal_quantile(
                frame,
                np.arange(len(frame)),
                residuals,
                coverage=0.75,
            )
        )
        self.assertGreaterEqual(quantile, 1.2)
        self.assertGreater(rank, 0)
        self.assertEqual(metadata["calibration_subjects"], 4)
        self.assertTrue(metadata["equal_total_weight_per_patient"])

    def test_patient_coverage_is_not_row_dominated(self):
        frame = pd.DataFrame(
            {"subject_id": ["dense"] * 100 + ["sparse"] * 2}
        )
        covered = np.asarray([True] * 100 + [False, False])
        report = _patient_coverage_diagnostics(
            frame,
            np.arange(len(frame)),
            covered,
            seed=31,
            bootstrap_samples=100,
        )
        self.assertGreater(report["row_coverage"], 0.95)
        self.assertAlmostEqual(
            report["patient_equalized_coverage"], 0.5
        )
        self.assertEqual(report["test_subjects"], 2)

    def test_adaptive_patient_cluster_uses_disjoint_final_patients(self):
        rng = np.random.default_rng(43)
        subjects = np.repeat(np.arange(60), 4)
        frame = pd.DataFrame({"subject_id": subjects})
        residuals = np.abs(rng.normal(size=len(frame)))
        quantile, rank, metadata = (
            _adaptive_patient_cluster_conformal_quantile(
                frame,
                np.arange(len(frame)),
                residuals,
                seed=47,
            )
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(
            metadata["method"],
            "patient_grouped_cv_adaptive_patient_cluster_conformal",
        )
        self.assertTrue(metadata["patient_disjoint_final_calibration"])
        self.assertFalse(metadata["test_labels_used"])

    def test_crossfit_adaptive_patient_cluster_uses_full_calibration(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(80), 4)})
        rows = np.arange(320)
        residuals = np.linspace(0.05, 2.0, len(rows))
        quantile, rank, metadata = (
            _crossfit_adaptive_patient_cluster_conformal_quantile(
                frame,
                rows,
                residuals,
                seed=41,
            )
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(metadata["calibration_rows"], len(rows))
        self.assertTrue(
            metadata["all_calibration_rows_used_after_crossfit_selection"]
        )
        self.assertFalse(metadata["test_labels_used"])
        self.assertLessEqual(metadata["selected_nominal_coverage"], 0.90)


if __name__ == "__main__":
    unittest.main()
