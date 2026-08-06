import unittest

import numpy as np
import pandas as pd
import torch

from whole_body_joint_jepa import (
    JointWholeBodyJEPA,
    JointWholeBodyJEPAWorldModel,
    _asof_align_parts,
    _canonicalize_audit_provenance,
    fit_joint,
)
from eicu_sepsis_transition_extract import target_tolerance_for_horizon


class WholeBodyJointAlignmentTests(unittest.TestCase):
    def test_conformal_quantile_uses_finite_sample_order_statistic(self):
        from whole_body_joint_gate import _finite_sample_conformal_quantile

        scores = np.arange(1.0, 101.0)
        quantile, rank = _finite_sample_conformal_quantile(scores, 0.90)
        self.assertEqual(rank, 91)
        self.assertEqual(quantile, 91.0)

    def test_patient_conformal_uses_pooled_coverage_and_wilson_compatibility(self):
        from whole_body_joint_gate import _aggregate_conformal

        results = [
            {"x@6h": {"coverage": 0.91, "test_n": 100, "pass": True}}
            for _ in range(6)
        ]
        results.append(
            {"x@6h": {"coverage": 0.86, "test_n": 100, "pass": False}}
        )
        aggregate = _aggregate_conformal(results)["x@6h"]
        self.assertEqual(aggregate["point_pass_splits"], 6)
        self.assertEqual(aggregate["calibration_compatible_splits"], 7)
        self.assertTrue(aggregate["pooled_coverage_pass"])
        self.assertTrue(aggregate["pass_all"])

    def test_patient_conformal_rejects_true_miscalibration(self):
        from whole_body_joint_gate import _aggregate_conformal

        results = [
            {"x@6h": {"coverage": 0.70, "test_n": 500, "pass": False}}
            for _ in range(7)
        ]
        aggregate = _aggregate_conformal(results)["x@6h"]
        self.assertFalse(aggregate["pooled_coverage_pass"])
        self.assertFalse(aggregate["pass_all"])

    def test_patient_conformal_preserves_an_in_band_point_estimate(self):
        from whole_body_joint_gate import _aggregate_conformal

        results = [
            {"x@6h": {"coverage": 0.87, "test_n": 100_000, "pass": True}}
            for _ in range(7)
        ]
        aggregate = _aggregate_conformal(results)["x@6h"]
        self.assertEqual(aggregate["calibration_compatible_splits"], 7)
        self.assertTrue(aggregate["pass_all"])

    def test_care_unit_and_database_are_preserved_only_as_audit_provenance(self):
        frame = pd.DataFrame(
            {
                "unittype": ["MICU", ""],
                "first_careunit": ["", "SICU"],
                "database": ["eicu", "mimiciv"],
            }
        )
        result = _canonicalize_audit_provenance(frame)
        self.assertEqual(result["careunit"].tolist(), ["MICU", "SICU"])
        self.assertEqual(result["database"].tolist(), ["eicu", "mimiciv"])
        self.assertIn("careunit", result.attrs["audit_provenance_columns"])
        self.assertIn("careunit", result.attrs["physiology_encoder_excludes"])

    def test_fit_joint_accepts_care_unit_domain_invariance(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3", "s4"],
                "stay_id": ["a", "b", "c", "d"],
                "hospitalid": [1, 1, 2, 2],
                "careunit": ["MICU", "SICU", "MICU", "SICU"],
                "database": ["eicu"] * 4,
                "t": pd.date_range("2025-01-01", periods=4, freq="h"),
                "_time_hours": np.arange(4, dtype=float),
                "_source_horizon": np.full(4, 1.0),
                "horizon_hours": np.full(4, 1.0),
                "_module_present_0": np.ones(4),
                "present__body": np.ones(4),
                "hours_since_onset": np.arange(4, dtype=float),
                "x_t": [1.0, 2.0, 3.0, 4.0],
                "x_mask_t": np.ones(4),
                "x_age_hr": np.zeros(4),
                "future_x": [1.1, 2.1, 3.1, 4.1],
            }
        )
        frame.attrs["module_variables"] = {"body": ("x",)}
        model, _, _ = fit_joint(
            frame,
            ("x",),
            ("body",),
            np.arange(4),
            epochs=0,
            hospital_invariance_weight=0.05,
            domain_invariance_columns=("careunit",),
        )
        self.assertIsInstance(model, JointWholeBodyJEPA)

    def test_care_unit_holdout_keeps_whole_units_out_of_training(self):
        from whole_body_joint_gate import _split_known_groups

        frame = pd.DataFrame(
            {
                "careunit": ["MICU", "MICU", "SICU", "SICU", "__unknown_careunit__"],
            }
        )
        train, test = _split_known_groups(frame, "careunit", seed=7, fraction=0.5)
        heldout = set(frame.iloc[test]["careunit"])
        trained_known = set(frame.iloc[train]["careunit"]) - {"__unknown_careunit__"}
        self.assertTrue(heldout)
        self.assertTrue(heldout.isdisjoint(trained_known))
        self.assertIn(4, train)

    def test_forward_time_holdout_never_trains_on_later_anchor(self):
        from whole_body_joint_gate import _split_forward_time

        frame = pd.DataFrame(
            {
                "stay_id": ["a"] * 4 + ["b"] * 4,
                "_time_hours": [0.0, 1.0, 2.0, 3.0] * 2,
            }
        )
        train, test = _split_forward_time(frame, fraction=0.25)
        for stay_id in ("a", "b"):
            train_times = frame.iloc[train].loc[
                frame.iloc[train]["stay_id"] == stay_id, "_time_hours"
            ]
            test_times = frame.iloc[test].loc[
                frame.iloc[test]["stay_id"] == stay_id, "_time_hours"
            ]
            self.assertLess(float(train_times.max()), float(test_times.min()))

    def test_validated_router_uses_disjoint_selection_before_outer_test(self):
        from whole_body_joint_gate import _validated_router_prediction
        from whole_body_joint_jepa import _joint_arrays

        rng = np.random.default_rng(11)
        rows = 180
        driver = rng.normal(size=rows)
        current = rng.normal(size=rows)
        frame = pd.DataFrame(
            {
                "subject_id": [f"s{index}" for index in range(rows)],
                "stay_id": [f"stay{index}" for index in range(rows)],
                "t": pd.date_range("2025-01-01", periods=rows, freq="h"),
                "_time_hours": np.arange(rows, dtype=float),
                "_source_horizon": np.full(rows, 6.0),
                "_module_present_0": np.ones(rows),
                "present__body": np.ones(rows),
                "hours_since_onset": np.arange(rows, dtype=float),
                "x_t": current,
                "z_t": driver,
                "x_mask_t": np.ones(rows),
                "z_mask_t": np.ones(rows),
                "x_age_hr": np.zeros(rows),
                "z_age_hr": np.zeros(rows),
                "future_x": current + 2.0 * driver,
                "future_z": driver,
            }
        )
        fit_rows = np.arange(0, 100)
        selection_rows = np.arange(100, 140)
        test_rows = np.arange(140, rows)
        arrays = _joint_arrays(
            frame,
            ("x", "z"),
            ("body",),
            fit_rows,
        )
        prediction, metadata = _validated_router_prediction(
            frame,
            ("x", "z"),
            ("body",),
            arrays,
            fit_rows,
            selection_rows,
            test_rows,
            include_treatment_context=False,
            seed=19,
        )
        self.assertEqual(
            metadata["x@6h"]["selected_source"],
            "ridge_realfit",
        )
        scaler = arrays[0]
        raw_prediction = prediction * scaler.scales + scaler.medians
        mae = np.mean(
            np.abs(
                raw_prediction[:, 0]
                - frame.iloc[test_rows]["future_x"].to_numpy(dtype=float)
            )
        )
        self.assertLess(float(mae), 0.2)

    def test_asof_inputs_are_causal_and_future_labels_are_not_shifted(self):
        part_a = pd.DataFrame(
            {
                "subject_id": ["s1", "s1"],
                "stay_id": ["stay1", "stay1"],
                "hospitalid": [1, 1],
                "t": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
                "_source_horizon": [6, 6],
                "x_t": [1.0, 2.0],
                "x_age_hr": [0.0, 0.0],
                "future_x": [2.0, 3.0],
                "_row_time_hr": [0.0, 1.0],
            }
        )
        part_b = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "hospitalid": [1],
                "t": pd.to_datetime(["2025-01-01 00:30"]),
                "_source_horizon": [6],
                "y_t": [11.0],
                "y_age_hr": [0.0],
                "future_y": [99.0],
                "_row_time_hr": [0.5],
            }
        )
        aligned = _asof_align_parts(
            [part_a, part_b],
            ("a", "b"),
            {"a": ("x",), "b": ("y",)},
            ("x", "y"),
            max_age_hours=6.0,
            future_label_tolerance_hours=0.0,
            min_modules=2,
        )
        row = aligned.loc[aligned["t"] == pd.Timestamp("2025-01-01 01:00")].iloc[0]
        self.assertEqual(row["_module_count"], 2)
        self.assertAlmostEqual(row["y_t"], 11.0)
        self.assertAlmostEqual(row["y_age_hr"], 0.5)
        self.assertEqual(int(row["_source_module_x"]), 0)
        self.assertEqual(int(row["_source_module_y"]), 1)
        self.assertTrue(np.isnan(row["future_y"]))
        self.assertAlmostEqual(row["future_x"], 3.0)

    def test_freshest_duplicate_value_keeps_its_actual_source_module(self):
        part_a = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "hospitalid": [1],
                "t": pd.to_datetime(["2025-01-01 00:00"]),
                "_source_horizon": [6],
                "x_t": [1.0],
                "x_age_hr": [0.0],
                "_row_time_hr": [0.0],
            }
        )
        part_b = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "hospitalid": [1],
                "t": pd.to_datetime(["2025-01-01 00:30"]),
                "_source_horizon": [6],
                "x_t": [2.0],
                "x_age_hr": [0.0],
                "_row_time_hr": [0.5],
            }
        )
        aligned = _asof_align_parts(
            [part_a, part_b],
            ("a", "b"),
            {"a": ("x",), "b": ("x",)},
            ("x",),
            max_age_hours=6.0,
            future_label_tolerance_hours=0.0,
            min_modules=0,
        )
        row = aligned.loc[aligned["t"] == pd.Timestamp("2025-01-01 00:30")].iloc[0]
        self.assertAlmostEqual(row["x_t"], 2.0)
        self.assertEqual(int(row["_source_module_x"]), 1)

    def test_horizon_window_adds_only_a_bounded_future_label(self):
        part_a = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "hospitalid": [1],
                "t": pd.to_datetime(["2025-01-01 01:00"]),
                "_source_horizon": [6],
                "x_t": [2.0],
                "x_age_hr": [0.0],
                "future_x": [3.0],
                "_row_time_hr": [1.0],
            }
        )
        part_b = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "hospitalid": [1],
                "t": pd.to_datetime(["2025-01-01 00:30"]),
                "_source_horizon": [6],
                "y_t": [11.0],
                "y_age_hr": [0.0],
                "future_y": [99.0],
                "_row_time_hr": [0.5],
            }
        )
        aligned = _asof_align_parts(
            [part_a, part_b],
            ("a", "b"),
            {"a": ("x",), "b": ("y",)},
            ("x", "y"),
            max_age_hours=6.0,
            future_label_tolerance_hours=0.5,
            min_modules=2,
        )
        row = aligned.iloc[0]
        self.assertAlmostEqual(row["y_t"], 11.0)
        self.assertAlmostEqual(row["y_age_hr"], 0.5)
        self.assertAlmostEqual(row["future_y"], 99.0)

    def test_short_horizon_tolerance_cannot_reuse_current_observation(self):
        self.assertAlmostEqual(target_tolerance_for_horizon(1.0), 0.5)
        self.assertAlmostEqual(target_tolerance_for_horizon(3.0), 1.5)
        self.assertAlmostEqual(target_tolerance_for_horizon(6.0), 2.0)

    def test_horizon_scaled_age_policy_rejects_stale_short_horizon_input(self):
        part = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "hospitalid": [1],
                "t": pd.to_datetime(["2025-01-01 01:00"]),
                "_source_horizon": [1],
                "x_t": [2.0],
                "x_age_hr": [1.0],
                "future_x": [3.0],
                "_row_time_hr": [1.0],
            }
        )
        aligned = _asof_align_parts(
            [part],
            ("a",),
            {"a": ("x",)},
            ("x",),
            max_age_hours=6.0,
            future_label_tolerance_hours=0.0,
            min_modules=0,
            age_policy="horizon_scaled",
        )
        self.assertTrue(np.isnan(aligned.iloc[0]["x_t"]))
        self.assertEqual(int(aligned.iloc[0]["_module_count"]), 0)

    def test_neutralized_measurement_process_removes_masks_and_ages(self):
        from whole_body_joint_jepa import _joint_arrays

        frame = pd.DataFrame(
            {
                "subject_id": ["s1", "s1"],
                "stay_id": ["stay1", "stay1"],
                "t": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
                "_time_hours": [0.0, 1.0],
                "_source_horizon": [1, 1],
                "_module_present_0": [1.0, 0.0],
                "hours_since_onset": [0.0, 1.0],
                "x_t": [1.0, np.nan],
                "x_age_hr": [0.0, np.nan],
                "future_x": [2.0, 2.0],
            }
        )
        arrays = _joint_arrays(
            frame,
            ("x",),
            ("a",),
            np.array([0, 1]),
            measurement_process_mode="neutralized",
        )
        _, input_matrix, _, mask, ages, _, _, _, _, presence = arrays[:10]
        self.assertTrue(np.all(mask == 1.0))
        self.assertTrue(np.all(ages == 0.0))
        self.assertTrue(np.all(presence == 1.0))
        self.assertTrue(np.allclose(input_matrix[:, 1], 1.0))

    def test_raw_future_delta_and_label_weight_reach_training_arrays(self):
        from whole_body_joint_jepa import _joint_arrays

        frame = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "stay_id": ["stay1"],
                "t": pd.to_datetime(["2025-01-01 00:00"]),
                "_time_hours": [0.0],
                "_source_horizon": [3.0],
                "_module_present_0": [1.0],
                "hours_since_onset": [0.0],
                "x_t": [1.0],
                "y_t": [2.0],
                "x_age_hr": [0.0],
                "y_age_hr": [0.0],
                "future_x": [1.5],
                "future_y": [2.5],
                "future_delta_t_hr_x": [2.25],
                "future_delta_t_hr_y": [2.75],
                "future_label_weight_x": [1.0],
                "future_label_weight_y": [0.0],
            }
        )
        arrays = _joint_arrays(
            frame,
            ("x", "y"),
            ("a",),
            np.array([0]),
        )
        future_mask = arrays[6]
        future_delta_t = arrays[12]
        future_window_delta_t = arrays[13]
        self.assertAlmostEqual(float(future_delta_t[0, 0]), 2.25)
        self.assertAlmostEqual(float(future_delta_t[0, 1]), 2.75)
        self.assertEqual(float(future_mask[0, 0]), 1.0)
        self.assertEqual(float(future_mask[0, 1]), 0.0)
        self.assertAlmostEqual(float(future_window_delta_t[0, 0, 0]), 2.25)
        # Timing metadata remains auditable even when the quality gate masks
        # the corresponding value out of JEPA supervision.
        self.assertAlmostEqual(float(future_window_delta_t[0, 0, 1]), 2.75)

    def test_next_measurement_time_labels_are_separate_from_future_values(self):
        from whole_body_joint_jepa import _joint_arrays

        frame = pd.DataFrame(
            {
                "subject_id": ["s1", "s1"],
                "stay_id": ["stay1", "stay1"],
                "t": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
                "_time_hours": [0.0, 1.0],
                "_source_horizon": [3.0, 3.0],
                "_module_present_0": [1.0, 1.0],
                "hours_since_onset": [0.0, 1.0],
                "x_t": [1.0, 1.2],
                "x_age_hr": [0.0, 0.0],
                "future_x": [1.5, 1.7],
                "next_measurement_time_hr_x": [2.5, np.nan],
            }
        )
        arrays = _joint_arrays(frame, ("x",), ("a",), np.arange(2))
        self.assertAlmostEqual(float(arrays[15][0, 0]), 2.5)
        self.assertEqual(float(arrays[16][0, 0]), 1.0)
        self.assertTrue(np.isnan(arrays[15][1, 0]))
        self.assertEqual(float(arrays[16][1, 0]), 0.0)

    def test_observation_process_loss_does_not_update_physiology_encoder(self):
        model = JointWholeBodyJEPA(
            n_state=2,
            input_dim=8,
            module_count=1,
            hidden=12,
            latent=8,
            token_dim=8,
            measurement_time_head=True,
        )
        observations = torch.randn(4, 3, 8)
        current = torch.randn(4, 2)
        mask = torch.ones(4, 2)
        ages = torch.zeros(4, 2)
        presence = torch.ones(4, 1)
        horizon = torch.full((4,), 6.0)
        output = model(
            observations, current, mask, ages, presence, horizon
        )
        process_loss = (
            output["observation_logit"].sum()
            + output["next_measurement_time_log"].sum()
        )
        process_loss.backward()

        physiology_parameters = [
            *model.belief_filter.parameters(),
            *model.variable_mixer.parameters(),
            *model.module_mixer.parameters(),
            *model.state_encoder.parameters(),
        ]
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.all(parameter.grad == 0))
                for parameter in physiology_parameters
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in model.observation_heads.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in model.measurement_time_heads.parameters()
            )
        )

    def test_time_only_placebo_removes_patient_physiology_but_keeps_capacity(self):
        model = JointWholeBodyJEPA(
            n_state=2,
            input_dim=8,
            module_count=1,
            hidden=12,
            latent=8,
            token_dim=8,
            belief_input_mode="placebo_time_only",
        )
        observations = torch.zeros((2, 3, 8), dtype=torch.float32)
        observations[0, :, 0] = 1.0
        observations[1, :, 0] = 100.0
        observations[:, :, -1] = torch.tensor([0.0, 0.5, 1.0])
        current = torch.ones((2, 2), dtype=torch.float32)
        mask = torch.ones((2, 2), dtype=torch.float32)
        ages = torch.zeros((2, 2), dtype=torch.float32)
        presence = torch.ones((2, 1), dtype=torch.float32)
        _, belief, _, _, _ = model.encode_context_components(
            observations,
            current,
            mask,
            ages,
            presence,
        )
        self.assertTrue(torch.allclose(belief[0], belief[1]))

    def test_attribution_validated_group_masks_only_creatinine_at_12h(self):
        from whole_body_joint_jepa import (
            _build_source_group_adapter_specs,
            _validated_edge_batch_view,
        )

        modules = (
            "aki",
            "electrolyte_acid_base",
            "endocrine_stress",
            "gi_pancreatic_nutrition",
            "musculoskeletal_rhabdo",
        )
        variables = ("creatinine", "bun", "potassium")
        membership = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        specs = _build_source_group_adapter_specs(
            modules,
            variables,
            {name: tuple(variable for variable, owned in zip(variables, row) if owned)
             for name, row in zip(modules, membership)},
            allowed_keys=("metabolic_renal->creatinine@12h",),
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["allowed_horizons"], [12.0])
        current = np.asarray([[1.2, 30.0, 4.0], [1.4, 31.0, 4.1]], dtype=np.float32)
        mask = np.ones_like(current)
        ages = np.zeros_like(current)
        presence = np.ones((2, len(modules)), dtype=np.float32)
        input_matrix = np.concatenate(
            [current, mask, ages, np.zeros((2, 1), dtype=np.float32), presence, np.zeros((2, 1), dtype=np.float32)],
            axis=1,
        )
        view = _validated_edge_batch_view(
            np.asarray([0, 1]), current, mask, ages, presence, input_matrix,
            np.asarray([[0], [1]]), np.ones_like(current), np.asarray([12.0, 6.0]),
            specs, membership,
        )
        self.assertEqual(float(view[0][0, 0]), 0.0)
        self.assertAlmostEqual(float(view[0][0, 1]), 30.0)
        self.assertEqual(float(view[5][0, 0]), 1.0)
        self.assertEqual(float(view[5][1, 0]), 0.0)

    def test_dense_future_interpolation_requires_a_tight_post_anchor_bracket(self):
        from eicu_raw_measurement_time_augment import _future_label_arrays

        anchors = pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "_anchor_hour": [0.0, 0.0, 0.0],
                "_augment_horizon": [1.0, 1.0, 1.0],
            }
        )
        values, times, deltas, quality, _, _, weights, left, right, gaps = (
            _future_label_arrays(
                anchors,
                "heart_rate",
                {
                    # Tight post-anchor bracket: interpolate at exactly 1h.
                    1: (np.array([0.9, 1.1]), np.array([80.0, 100.0])),
                    # A raw event at the target remains a raw exact label.
                    2: (np.array([0.8, 1.0, 1.2]), np.array([70.0, 90.0, 110.0])),
                    # A raw event outside the 1h dense tolerance remains missing.
                    3: (np.array([1.8]), np.array([110.0])),
                },
                future_label_mode="interpolate_dense",
                interpolated_label_weight=0.50,
            )
        )
        self.assertEqual(quality[0], "interpolated_dense")
        self.assertAlmostEqual(float(values[0]), 90.0)
        self.assertAlmostEqual(float(times[0]), 1.0)
        self.assertAlmostEqual(float(deltas[0]), 1.0)
        self.assertAlmostEqual(float(left[0]), 0.9)
        self.assertAlmostEqual(float(right[0]), 1.1)
        self.assertAlmostEqual(float(gaps[0]), 0.2)
        self.assertAlmostEqual(float(weights[0]), 0.5)
        self.assertEqual(quality[1], "exact")
        self.assertAlmostEqual(float(values[1]), 90.0)
        self.assertAlmostEqual(float(weights[1]), 1.0)
        self.assertEqual(quality[2], "missing")
        self.assertAlmostEqual(float(weights[2]), 0.0)

    def test_joint_model_exposes_a_finite_body_latent_for_consistency_audit(self):
        model = JointWholeBodyJEPA(
            n_state=2,
            input_dim=9,
            module_count=1,
            hidden=8,
            latent=6,
            token_dim=8,
            variable_module_ids=[0, 0],
        )
        output = model(
            torch.randn(3, 4, 9),
            torch.randn(3, 2),
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 1),
            torch.tensor([1.0, 6.0, 12.0]),
        )
        self.assertEqual(tuple(output["latent"].shape), (3, 6))
        self.assertEqual(tuple(output["belief_state"].shape), (3, 8))
        self.assertEqual(tuple(output["belief_uncertainty"].shape), (3, 8))
        self.assertEqual(tuple(output["organ_latents"].shape), (3, 1, 8))
        self.assertEqual(tuple(output["organ_presence"].shape), (3, 1))
        self.assertEqual(tuple(output["target_temporal_latent"].shape), (3, 2, 6))
        self.assertEqual(tuple(output["target_temporal_gate"].shape), (3, 2))
        self.assertTrue(torch.isfinite(output["latent"]).all())

    def test_true_jepa_world_model_has_context_predictor_and_target_latents(self):
        model = JointWholeBodyJEPAWorldModel(
            n_state=2,
            input_dim=9,
            module_count=1,
            hidden=8,
            latent=6,
            token_dim=8,
            variable_module_ids=[0, 0],
        )
        output = model(
            torch.randn(3, 4, 9),
            torch.randn(3, 2),
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 1),
            torch.tensor([1.0, 6.0, 12.0]),
            future=torch.randn(3, 2),
            future_mask=torch.ones(3, 2),
        )
        self.assertEqual(tuple(output["context_latent"].shape), (3, 6))
        self.assertEqual(tuple(output["predicted_latent"].shape), (3, 6))
        self.assertEqual(tuple(output["predicted_trajectory_latent"].shape), (3, 1, 6))
        self.assertEqual(tuple(output["target_latent"].shape), (3, 6))
        self.assertEqual(tuple(output["target_trajectory_latent"].shape), (3, 1, 6))
        self.assertEqual(tuple(output["predicted_target_latent"].shape), (3, 2, 6))
        self.assertEqual(tuple(output["target_specific_latent"].shape), (3, 2, 6))
        self.assertEqual(tuple(output["target_specific_available"].shape), (3, 2))
        self.assertEqual(tuple(output["belief_state"].shape), (3, 8))
        self.assertEqual(tuple(output["belief_uncertainty"].shape), (3, 8))
        self.assertEqual(tuple(output["organ_latents"].shape), (3, 1, 8))
        self.assertEqual(tuple(output["organ_presence"].shape), (3, 1))
        self.assertEqual(tuple(output["target_temporal_latent"].shape), (3, 2, 6))
        self.assertTrue(torch.isfinite(output["predicted_latent"]).all())

    def test_joint_model_measurement_time_head_is_separate_and_positive(self):
        model = JointWholeBodyJEPA(
            n_state=2,
            input_dim=9,
            module_count=1,
            hidden=8,
            latent=6,
            token_dim=8,
            variable_module_ids=[0, 0],
            measurement_time_head=True,
        )
        output = model(
            torch.randn(3, 4, 9),
            torch.randn(3, 2),
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 1),
            torch.tensor([1.0, 6.0, 12.0]),
        )
        self.assertIn("next_measurement_time_log", output)
        self.assertEqual(tuple(output["next_measurement_time_log"].shape), (3, 2))
        predicted_hours = torch.nn.functional.softplus(
            output["next_measurement_time_log"]
        )
        self.assertTrue(torch.isfinite(predicted_hours).all())
        self.assertTrue((predicted_hours >= 0.0).all())

    def test_true_jepa_world_model_encodes_a_future_trajectory_prefix(self):
        model = JointWholeBodyJEPAWorldModel(
            n_state=2,
            input_dim=9,
            module_count=1,
            hidden=8,
            latent=6,
            token_dim=8,
            variable_module_ids=[0, 0],
        )
        output = model(
            torch.randn(3, 4, 9),
            torch.randn(3, 2),
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 1),
            torch.tensor([1.0, 6.0, 12.0]),
            future=torch.randn(3, 2),
            future_mask=torch.ones(3, 2),
            future_window=torch.randn(3, 2, 2),
            future_window_mask=torch.ones(3, 2, 2),
            window_horizons=torch.tensor([1.0, 6.0]),
        )
        self.assertEqual(tuple(output["target_latent"].shape), (3, 6))
        self.assertEqual(tuple(output["predicted_trajectory_latent"].shape), (3, 2, 6))
        self.assertEqual(tuple(output["target_trajectory_latent"].shape), (3, 2, 6))
        self.assertEqual(
            tuple(output["predicted_target_trajectory_latent"].shape), (3, 2, 2, 6)
        )
        self.assertEqual(
            tuple(output["target_specific_trajectory_latent"].shape), (3, 2, 2, 6)
        )
        self.assertEqual(tuple(output["target_specific_available"].shape), (3, 2, 2))
        self.assertTrue(torch.all(output["target_available"]))
        self.assertTrue(torch.isfinite(output["target_latent"]).all())


if __name__ == "__main__":
    unittest.main()
