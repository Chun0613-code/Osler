"""Strict gate for the genuinely joint whole-body JEPA.

In addition to persistence and conformal checks, this gate runs a
cross-system ablation: for selected dense targets, the same trained model is
evaluated after all other body variables are masked from both history and the
current state.  A candidate is called coupled only when the full body input
beats that own-target-only ablation on held-out data.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from osler_jepa.cohort_signature import cohort_signature

from nonlinear_latent_rollout import set_seed
from whole_body_joint_jepa import (
    _build_supervision_strata,
    fit_joint,
    load_joint_cohort,
    load_joint_event_examples,
)


SEEDS = (7, 19, 31, 43, 59, 71, 83)
PATIENT_BOOTSTRAP_SAMPLES = 1000
PATIENT_BOOTSTRAP_MIN_SUBJECTS = 10
CROSS_SYSTEM_TARGETS = {
    "glucose",
    "potassium",
    "bicarbonate",
    "creatinine",
    "sodium",
    "chloride",
    "map",
    "heart_rate",
    "o2sat",
    "respiratory_rate",
    "lactate",
}

JOINT_MOVEMENT_REQUIREMENTS = (
    "patient_value",
    "patient_bootstrap",
    "patient_validated_router_value",
    "patient_validated_router_bootstrap",
    "patient_module_balanced",
    "hospital_value",
    "hospital_validated_router_value",
    "hospital_validated_router_bootstrap",
    "patient_conformal",
    "hospital_conformal",
    "hospital_module_balanced",
    "care_unit_value",
    "care_unit_validated_router_value",
    "care_unit_validated_router_bootstrap",
    "care_unit_conformal",
    "care_unit_module_balanced",
    "time_value",
    "time_validated_router_value",
    "time_validated_router_bootstrap",
    "time_conformal",
    "time_module_balanced",
)


def apply_joint_movement_policy(
    entry,
    *,
    cross_system_evaluated,
):
    """Separate forecast permission from cross-organ attribution permission."""

    entry = dict(entry)
    entry["cross_system_evaluated"] = bool(cross_system_evaluated)
    entry["cross_system_attribution_validated"] = bool(
        cross_system_evaluated
        and entry.get("patient_cross_system", False)
        and entry.get("hospital_cross_system", False)
        and entry.get("care_unit_cross_system", False)
        and entry.get("time_cross_system", False)
    )
    entry["cross_organ_claim_allowed"] = entry[
        "cross_system_attribution_validated"
    ]
    entry["validated_for_joint_runtime"] = bool(
        all(entry.get(name, False) for name in JOINT_MOVEMENT_REQUIREMENTS)
    )
    entry["validated"] = entry["validated_for_joint_runtime"]
    return entry

BELIEF_DOWNSTREAM_GROUPS = {
    "renal_reserve": {"creatinine", "bun", "urine_output"},
    "glycemic_response": {"glucose"},
    "acid_base_reserve": {"bicarbonate", "anion_gap", "ph", "paco2"},
    "cardiorespiratory_reserve": {
        "map",
        "heart_rate",
        "o2sat",
        "respiratory_rate",
        "lactate",
    },
}


def _belief_group_for_target(target: str) -> str | None:
    for group, targets in BELIEF_DOWNSTREAM_GROUPS.items():
        if target in targets:
            return group
    return None


def _split_by_column(frame: pd.DataFrame, column: str, seed: int, fraction: float = 0.25):
    values = frame[column].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    selected = set(values[: max(1, int(round(len(values) * fraction)))].tolist())
    test = frame[column].isin(selected).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _split_train_calibration(frame, train_rows, seed):
    train_frame = frame.iloc[train_rows]
    fit_local, cal_local = _split_by_column(train_frame.reset_index(drop=True), "subject_id", seed, 0.20)
    return train_rows[fit_local], train_rows[cal_local]


def _split_forward_time(
    frame: pd.DataFrame,
    fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the latest anchor times within each stay.

    MIMIC and eICU de-identification make global calendar ordering
    incomparable. A within-stay forward split still tests the required
    direction: learn only from earlier physiology and evaluate on later
    physiology, without putting a later anchor in the training partition.
    """

    train_mask = np.zeros(len(frame), dtype=bool)
    test_mask = np.zeros(len(frame), dtype=bool)
    for _, group in frame.groupby("stay_id", sort=False):
        positions = group.index.to_numpy(dtype=np.int64)
        times = pd.to_numeric(
            group.get("_time_hours", group.get("hours_since_onset")),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_times = np.unique(times[np.isfinite(times)])
        if len(finite_times) < 2:
            train_mask[positions] = True
            continue
        test_count = max(1, int(round(len(finite_times) * float(fraction))))
        test_count = min(test_count, len(finite_times) - 1)
        boundary = finite_times[-test_count]
        local_test = np.isfinite(times) & (times >= boundary)
        test_mask[positions[local_test]] = True
        train_mask[positions[~local_test]] = True
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _holdout_available(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum_groups: int = 2,
) -> bool:
    if column not in frame:
        return False
    values = frame[column].dropna().astype(str).str.strip()
    values = values[
        (values != "")
        & ~values.str.startswith("__unknown_")
        & (values.str.lower() != "nan")
    ]
    return int(values.nunique()) >= int(minimum_groups)


def _split_known_groups(
    frame: pd.DataFrame,
    column: str,
    seed: int,
    fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[column].fillna("").astype(str).str.strip()
    known = (
        (values != "")
        & ~values.str.startswith("__unknown_")
        & (values.str.lower() != "nan")
    )
    groups = values[known].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    selected = set(
        groups[: max(1, int(round(len(groups) * float(fraction))))].tolist()
    )
    test = known & values.isin(selected)
    return np.flatnonzero(~test.to_numpy()), np.flatnonzero(test.to_numpy())


def _ablate_target(
    arrays,
    target_index,
    n_state,
    module_count,
    variable_module_membership=None,
):
    (
        scaler,
        input_matrix,
        current,
        mask,
        ages,
        future,
        future_mask,
        history,
        horizon,
        presence,
        future_window,
        future_window_mask,
        future_delta_t,
        future_window_delta_t,
        window_horizons,
    ) = arrays[:15]
    # The measurement-process labels are not inputs to an ablation view, but
    # keep them attached so optional observation-time scoring remains aligned.
    extra_arrays = tuple(arrays[15:])
    current = current.copy()
    mask = mask.copy()
    ages = ages.copy()
    presence = presence.copy()
    keep = np.zeros(n_state, dtype=bool)
    keep[target_index] = True
    current[:, ~keep] = 0.0
    mask[:, ~keep] = 0.0
    ages[:, ~keep] = 7.0
    # A fair own-target baseline keeps only the module(s) owning the target.
    # Retaining every module-presence flag leaks the full body topology into
    # the ablation and makes the cross-system comparison too optimistic.
    presence[:, :] = 0.0
    if variable_module_membership is None:
        owner = int(min(target_index, max(0, module_count - 1)))
        presence[:, owner] = 1.0
    else:
        owners = np.asarray(variable_module_membership, dtype=np.float32)[:, target_index] > 0.5
        if owners.any():
            presence[:, owners] = 1.0
    onset = input_matrix[:, 3 * n_state : 3 * n_state + 1]
    presence_start = 3 * n_state + 1
    presence_end = presence_start + int(module_count)
    time_feature = input_matrix[:, presence_end:]
    ablated_input = np.concatenate(
        [current, mask, ages, onset, presence, time_feature], axis=1
    ).astype(np.float32)
    return (
        scaler,
        ablated_input,
        current,
        mask,
        ages,
        future,
        future_mask,
        history,
        horizon,
        presence,
        future_window,
        future_window_mask,
        future_delta_t,
        future_window_delta_t,
        window_horizons,
        *extra_arrays,
    )


def _forward(model, arrays, rows, batch_size=4096, output_key="value"):
    _, input_matrix, current, mask, ages, _, _, history, horizon, presence = arrays[:10]
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            output.append(
                model(
                    torch.from_numpy(input_matrix[history[batch]]).float(),
                    torch.from_numpy(current[batch]).float(),
                    torch.from_numpy(mask[batch]).float(),
                    torch.from_numpy(ages[batch]).float(),
                    torch.from_numpy(presence[batch]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )[output_key].numpy()
            )
    return np.concatenate(output, axis=0) if output else np.empty((0, model.n_state))


def _forward_many(
    model,
    keyed_arrays,
    rows,
    batch_size=256,
    output_keys=("value", "reconstruction"),
    view_batch_size=4,
):
    """Run several ablated input views together without changing their outputs.

    Gate evaluation repeatedly scores the same rows after removing one target at
    a time. Views are concatenated in small chunks to control peak memory while
    keeping each view isolated.
    """
    model.eval()
    keys = [key for key, _ in keyed_arrays]
    collected = {
        key: {output_key: [] for output_key in output_keys}
        for key in keys
    }
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for view_start in range(0, len(keyed_arrays), max(1, int(view_batch_size))):
                view_chunk = keyed_arrays[
                    view_start : view_start + max(1, int(view_batch_size))
                ]
                inputs = []
                currents = []
                masks = []
                ages = []
                presences = []
                horizons = []
                for _, arrays in view_chunk:
                    _, input_matrix, current, mask, ages_array, _, _, history, horizon, presence = arrays[:10]
                    inputs.append(torch.from_numpy(input_matrix[history[batch]]).float())
                    currents.append(torch.from_numpy(current[batch]).float())
                    masks.append(torch.from_numpy(mask[batch]).float())
                    ages.append(torch.from_numpy(ages_array[batch]).float())
                    presences.append(torch.from_numpy(presence[batch]).float())
                    horizons.append(torch.from_numpy(horizon[batch]).float())

                output = model(
                    torch.cat(inputs, dim=0),
                    torch.cat(currents, dim=0),
                    torch.cat(masks, dim=0),
                    torch.cat(ages, dim=0),
                    torch.cat(presences, dim=0),
                    torch.cat(horizons, dim=0),
                )
                offset = 0
                width = len(batch)
                for key, _ in view_chunk:
                    end = offset + width
                    for output_key in output_keys:
                        collected[key][output_key].append(
                            output[output_key][offset:end].cpu().numpy()
                        )
                    offset = end

    return {
        key: {
            output_key: (
                np.concatenate(parts, axis=0)
                if parts
                else np.empty((0, model.n_state))
            )
            for output_key, parts in outputs.items()
        }
        for key, outputs in collected.items()
    }


def _patient_paired_bootstrap(
    frame,
    selected,
    candidate_error,
    persistence_error,
    *,
    seed=20260720,
):
    """Give every patient one paired candidate-minus-persistence contribution."""

    subjects = frame.iloc[np.asarray(selected, dtype=np.int64)]["subject_id"].astype(str).to_numpy()
    subject_deltas = []
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        subject_deltas.append(float(np.mean(candidate_error[subject_mask] - persistence_error[subject_mask])))
    subject_deltas = np.asarray(subject_deltas, dtype=np.float64)
    count = len(subject_deltas)
    if count == 0:
        return {
            "n_subjects": 0,
            "patient_equal_delta": None,
            "bootstrap_low": None,
            "bootstrap_high": None,
            "pass": False,
        }
    point = float(subject_deltas.mean())
    if count < PATIENT_BOOTSTRAP_MIN_SUBJECTS:
        return {
            "n_subjects": int(count),
            "patient_equal_delta": point,
            "bootstrap_low": None,
            "bootstrap_high": None,
            "pass": False,
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, count, size=(PATIENT_BOOTSTRAP_SAMPLES, count))
    bootstrap_means = subject_deltas[draws].mean(axis=1)
    low = float(np.quantile(bootstrap_means, 0.025))
    high = float(np.quantile(bootstrap_means, 0.975))
    return {
        "n_subjects": int(count),
        "patient_equal_delta": point,
        "bootstrap_low": low,
        "bootstrap_high": high,
        "pass": bool(high < 0.0),
    }


def _router_feature_matrix(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    module_names: tuple[str, ...],
    *,
    include_treatment_context: bool,
) -> np.ndarray:
    """Build the measurement-pure feature view used by the validated router.

    Audit provenance is deliberately absent. Only anchor-time values, masks,
    ages, body-module presence and optionally pre-anchor treatment history are
    eligible.
    """

    columns: list[np.ndarray] = []
    for suffix in ("_t", "_mask_t", "_age_hr"):
        for variable in variables:
            series = pd.to_numeric(
                frame.get(
                    f"{variable}{suffix}",
                    pd.Series(np.nan, index=frame.index),
                ),
                errors="coerce",
            )
            columns.append(series.to_numpy(dtype=np.float64))
    columns.append(
        pd.to_numeric(
            frame.get(
                "hours_since_onset",
                frame.get("_time_hours", pd.Series(0.0, index=frame.index)),
            ),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    )
    for module in module_names:
        columns.append(
            pd.to_numeric(
                frame.get(
                    f"present__{module}",
                    pd.Series(0.0, index=frame.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
        )
    if include_treatment_context:
        for column in sorted(value for value in frame.columns if value.startswith("hist_")):
            columns.append(
                pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            )
    return np.column_stack(columns).astype(np.float64)


def _ridge_residual_prediction(
    features: np.ndarray,
    current: np.ndarray,
    future: np.ndarray,
    fit_rows: np.ndarray,
    prediction_rows: np.ndarray,
    *,
    alpha: float = 10.0,
) -> np.ndarray:
    """Fit one discovery-only residual ridge and predict raw future values."""

    usable = np.asarray(fit_rows, dtype=np.int64)
    usable = usable[
        np.isfinite(current[usable])
        & np.isfinite(future[usable])
    ]
    output = current[np.asarray(prediction_rows, dtype=np.int64)].copy()
    if len(usable) < max(80, min(features.shape[1] + 10, 250)):
        return output
    x_train = features[usable]
    available = np.isfinite(x_train).any(axis=0)
    if not available.any():
        return output
    x_train = x_train[:, available]
    medians = np.nanmedian(x_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    scale = np.nanstd(x_train, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x_train = np.c_[
        np.ones(len(x_train)),
        (x_train - medians) / scale,
    ]
    residual = future[usable] - current[usable]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    gram = x_train.T @ x_train + penalty
    rhs = x_train.T @ residual
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ rhs
    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    x_test = features[prediction_rows][:, available]
    x_test = np.where(np.isfinite(x_test), x_test, medians)
    x_test = np.c_[
        np.ones(len(x_test)),
        (x_test - medians) / scale,
    ]
    valid = np.isfinite(current[prediction_rows])
    output[valid] = current[prediction_rows[valid]] + x_test[valid] @ beta
    return output


def _validated_router_prediction(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    module_names: tuple[str, ...],
    arrays,
    fit_rows: np.ndarray,
    selection_rows: np.ndarray,
    prediction_rows: np.ndarray,
    *,
    include_treatment_context: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """Rebuild the current ridge/persistence router without test leakage.

    Ridge models are fit on ``fit_rows``. A cell may leave persistence only
    when it wins a patient-level paired bootstrap on the disjoint
    ``selection_rows``. The outer prediction rows never select the source.
    """

    (
        scaler,
        _,
        current_scaled,
        current_mask,
        _,
        future_scaled,
        future_mask,
        _,
        horizon,
        _,
    ) = arrays[:10]
    current = current_scaled * scaler.scales + scaler.medians
    future = future_scaled * scaler.scales + scaler.medians
    current = np.where(current_mask > 0.5, current, np.nan)
    future = np.where(future_mask > 0.5, future, np.nan)
    features = _router_feature_matrix(
        frame,
        variables,
        module_names,
        include_treatment_context=include_treatment_context,
    )
    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    output_raw = current[prediction_rows].copy()
    metadata: dict[str, dict[str, object]] = {}
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    selection_rows = np.asarray(selection_rows, dtype=np.int64)
    for index, target in enumerate(variables):
        for horizon_value in sorted(set(horizon[prediction_rows])):
            fit_cell = fit_rows[horizon[fit_rows] == float(horizon_value)]
            selection_cell = selection_rows[
                horizon[selection_rows] == float(horizon_value)
            ]
            prediction_local = np.flatnonzero(
                horizon[prediction_rows] == float(horizon_value)
            )
            prediction_cell = prediction_rows[prediction_local]
            key = (
                f"{target}@"
                f"{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            )
            if len(selection_cell) == 0 or len(prediction_cell) == 0:
                continue
            selection_ridge = _ridge_residual_prediction(
                features,
                current[:, index],
                future[:, index],
                fit_cell,
                selection_cell,
            )
            selection_observed = (
                np.isfinite(current[selection_cell, index])
                & np.isfinite(future[selection_cell, index])
            )
            selected = selection_cell[selection_observed]
            if len(selected) < 30:
                metadata[key] = {
                    "selected_source": "persistence",
                    "selection_rows": int(len(selected)),
                    "reason": "insufficient_disjoint_selection_rows",
                }
                continue
            ridge_error = np.abs(
                selection_ridge[selection_observed]
                - future[selected, index]
            )
            persistence_error = np.abs(
                current[selected, index] - future[selected, index]
            )
            bootstrap = _patient_paired_bootstrap(
                frame,
                selected,
                ridge_error,
                persistence_error,
                seed=seed + index * 101 + int(round(float(horizon_value))),
            )
            selected_source = "ridge_realfit" if bootstrap["pass"] else "persistence"
            metadata[key] = {
                "selected_source": selected_source,
                "selection_rows": int(len(selected)),
                "selection_subjects": int(bootstrap["n_subjects"]),
                "selection_patient_equal_delta_vs_persistence": bootstrap[
                    "patient_equal_delta"
                ],
                "selection_bootstrap_low": bootstrap["bootstrap_low"],
                "selection_bootstrap_high": bootstrap["bootstrap_high"],
                "reason": (
                    "discovery_only_ridge_passed_patient_bootstrap"
                    if selected_source == "ridge_realfit"
                    else "ridge_did_not_pass_disjoint_patient_bootstrap"
                ),
            }
            if selected_source != "ridge_realfit":
                continue
            output_raw[prediction_local, index] = _ridge_residual_prediction(
                features,
                current[:, index],
                future[:, index],
                fit_cell,
                prediction_cell,
            )
    output_scaled = (output_raw - scaler.medians) / scaler.scales
    return output_scaled.astype(np.float32), metadata


def _metrics(
    frame,
    variables,
    arrays,
    prediction,
    rows,
    self_prediction_by_index=None,
    fit_rows=None,
    validated_router_prediction=None,
    capacity_placebo_prediction=None,
):
    scaler, _, current, _, _, future, future_mask, _, horizon, _ = arrays[:10]
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    pred_raw = prediction * scaler.scales + scaler.medians
    router_raw = (
        validated_router_prediction * scaler.scales + scaler.medians
        if validated_router_prediction is not None
        else None
    )
    placebo_raw = (
        capacity_placebo_prediction * scaler.scales + scaler.medians
        if capacity_placebo_prediction is not None
        else None
    )
    results = {}
    for index, target in enumerate(variables):
        self_raw = None
        if self_prediction_by_index and index in self_prediction_by_index:
            self_raw = self_prediction_by_index[index] * scaler.scales + scaler.medians
        for horizon_value in sorted(set(horizon[rows])):
            row_mask = (
                (horizon[rows] == float(horizon_value))
                & np.asarray(future_mask[rows, index], dtype=bool)
            )
            if row_mask.sum() < 30:
                continue
            selected = rows[row_mask]
            local = np.flatnonzero(row_mask)
            candidate_error = np.abs(pred_raw[local, index] - future_raw[selected, index])
            persistence_error = np.abs(current_raw[selected, index] - future_raw[selected, index])
            candidate = float(candidate_error.mean())
            persistence = float(persistence_error.mean())
            bootstrap = _patient_paired_bootstrap(
                frame,
                selected,
                candidate_error,
                persistence_error,
                seed=20260720 + int(index) * 101 + int(round(float(horizon_value))),
            )
            current_delta = future_raw[selected, index] - current_raw[selected, index]
            predicted_delta = pred_raw[local, index] - current_raw[selected, index]
            nonzero = np.abs(current_delta) > 1e-8
            observed_training = None
            if fit_rows is not None:
                observed_training = pd.to_numeric(
                    frame.iloc[np.asarray(fit_rows)][f"{target}_t"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                observed_training = observed_training[np.isfinite(observed_training)]
            target_scale = 0.0
            if observed_training is not None and len(observed_training) >= 4:
                target_scale = float(np.subtract(*np.quantile(observed_training, [0.75, 0.25])))
                if target_scale < 1e-6:
                    target_scale = float(np.std(observed_training))
            target_scale = max(target_scale, float(scaler.scales[index]), 1e-6)
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            entry = {
                "target": target,
                "horizon_hours": float(horizon_value),
                "n": int(len(selected)),
                "candidate_mae": candidate,
                "persistence_mae": persistence,
                "delta_vs_persistence": candidate - persistence,
                "pass_value": candidate < persistence,
                "patient_equal_delta_vs_persistence": bootstrap["patient_equal_delta"],
                "patient_bootstrap_low": bootstrap["bootstrap_low"],
                "patient_bootstrap_high": bootstrap["bootstrap_high"],
                "patient_bootstrap_n_subjects": bootstrap["n_subjects"],
                "pass_patient_bootstrap": bootstrap["pass"],
                "normalized_delta_vs_persistence": float((candidate - persistence) / target_scale),
                "direction_accuracy": float(np.mean(np.sign(predicted_delta[nonzero]) == np.sign(current_delta[nonzero]))) if nonzero.any() else None,
                "delta_correlation": float(np.corrcoef(predicted_delta, current_delta)[0, 1]) if np.std(predicted_delta) > 1e-9 and np.std(current_delta) > 1e-9 else None,
            }
            if self_raw is not None:
                self_mae = float(np.abs(self_raw[local, index] - future_raw[selected, index]).mean())
                entry["self_only_mae"] = self_mae
                entry["delta_vs_self_only"] = candidate - self_mae
                entry["pass_cross_system"] = candidate < self_mae
            if router_raw is not None:
                router_error = np.abs(
                    router_raw[local, index] - future_raw[selected, index]
                )
                router_mae = float(router_error.mean())
                router_bootstrap = _patient_paired_bootstrap(
                    frame,
                    selected,
                    candidate_error,
                    router_error,
                    seed=20260721
                    + int(index) * 101
                    + int(round(float(horizon_value))),
                )
                entry["validated_router_mae"] = router_mae
                entry["delta_vs_validated_router"] = candidate - router_mae
                entry["pass_validated_router_value"] = candidate < router_mae
                entry["validated_router_patient_equal_delta"] = router_bootstrap[
                    "patient_equal_delta"
                ]
                entry["validated_router_bootstrap_low"] = router_bootstrap[
                    "bootstrap_low"
                ]
                entry["validated_router_bootstrap_high"] = router_bootstrap[
                    "bootstrap_high"
                ]
                entry["pass_validated_router_bootstrap"] = router_bootstrap["pass"]
            if placebo_raw is not None:
                placebo_error = np.abs(
                    placebo_raw[local, index] - future_raw[selected, index]
                )
                placebo_mae = float(placebo_error.mean())
                placebo_bootstrap = _patient_paired_bootstrap(
                    frame,
                    selected,
                    candidate_error,
                    placebo_error,
                    seed=20260722
                    + int(index) * 101
                    + int(round(float(horizon_value))),
                )
                entry["capacity_placebo_mae"] = placebo_mae
                entry["delta_vs_capacity_placebo"] = candidate - placebo_mae
                entry["pass_capacity_placebo_value"] = candidate < placebo_mae
                entry["capacity_placebo_patient_equal_delta"] = (
                    placebo_bootstrap["patient_equal_delta"]
                )
                entry["capacity_placebo_bootstrap_low"] = placebo_bootstrap[
                    "bootstrap_low"
                ]
                entry["capacity_placebo_bootstrap_high"] = placebo_bootstrap[
                    "bootstrap_high"
                ]
                entry["pass_capacity_placebo_bootstrap"] = placebo_bootstrap[
                    "pass"
                ]
            results[key] = entry
    return results


def _regime_metrics(variables, arrays, prediction, rows, regime_ids, delta_bin_ids):
    """Report factual error by target, actual-delay bin, and anchor regime.

    These are diagnostic strata only. They do not change the promotion gate,
    because tiny strata are useful for inspection but not reliable promotion
    units.
    """

    scaler, _, current, _, _, future, future_mask, _, horizon, _ = arrays[:10]
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    pred_raw = prediction * scaler.scales + scaler.medians
    output = {}
    for index, target in enumerate(variables):
        for horizon_value in sorted(set(horizon[rows])):
            for delta_bin in sorted(set(delta_bin_ids[rows, index].tolist())):
                if int(delta_bin) < 0:
                    continue
                for regime in sorted(set(regime_ids[rows, index].tolist())):
                    if int(regime) < 0:
                        continue
                    row_mask = (
                        (horizon[rows] == float(horizon_value))
                        & np.asarray(future_mask[rows, index], dtype=bool)
                        & (delta_bin_ids[rows, index] == int(delta_bin))
                        & (regime_ids[rows, index] == int(regime))
                    )
                    if not row_mask.any():
                        continue
                    selected = rows[row_mask]
                    local = np.flatnonzero(row_mask)
                    candidate = float(np.abs(pred_raw[local, index] - future_raw[selected, index]).mean())
                    persistence = float(np.abs(current_raw[selected, index] - future_raw[selected, index]).mean())
                    key = (
                        f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
                        f"|delta_bin={int(delta_bin)}|regime={int(regime)}"
                    )
                    output[key] = {
                        "target": target,
                        "horizon_hours": float(horizon_value),
                        "delta_bin": int(delta_bin),
                        "regime": int(regime),
                        "n": int(len(selected)),
                        "candidate_mae": candidate,
                        "persistence_mae": persistence,
                        "delta_vs_persistence": candidate - persistence,
                        "pass_value": candidate < persistence,
                    }
    return output


def _reconstruction_metrics(variables, arrays, prediction_by_index, rows):
    scaler, _, current, mask, _, _, _, _, horizon, _ = arrays[:10]
    current_raw = current * scaler.scales + scaler.medians
    results = {}
    for index, prediction in prediction_by_index.items():
        target = variables[index]
        pred_raw = prediction * scaler.scales + scaler.medians
        for horizon_value in sorted(set(horizon[rows])):
            row_mask = (
                (horizon[rows] == float(horizon_value))
                & np.asarray(mask[rows, index], dtype=bool)
            )
            if row_mask.sum() < 30:
                continue
            selected = rows[row_mask]
            local = np.flatnonzero(row_mask)
            candidate = float(np.abs(pred_raw[local, index] - current_raw[selected, index]).mean())
            baseline = float(np.abs(scaler.medians[index] - current_raw[selected, index]).mean())
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            results[key] = {
                "target": target,
                "horizon_hours": float(horizon_value),
                "n": int(len(selected)),
                "reconstruction_mae": candidate,
                "median_baseline_mae": baseline,
                "delta_vs_median": candidate - baseline,
                "pass_reconstruction": candidate < baseline,
            }
    return results


def _aggregate_reconstruction(results):
    keys = sorted(set().union(*(result.keys() for result in results)))
    return {
        key: {
            "n_splits": sum(key in result for result in results),
            "pass_splits": sum(bool(result.get(key, {}).get("pass_reconstruction", False)) for result in results),
            "median_delta_vs_median": float(
                np.median([result[key]["delta_vs_median"] for result in results if key in result])
            ),
            "pass_all_splits": bool(
                all(key in result for result in results)
                and all(result[key].get("pass_reconstruction", False) for result in results)
            ),
        }
        for key in keys
    }


def _finite_sample_conformal_quantile(scores, coverage=0.90):
    """Return the distribution-free split-conformal order statistic."""

    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    rank = int(np.ceil((len(values) + 1) * float(coverage)))
    rank = min(max(rank, 1), len(values))
    ordered = np.sort(values)
    return float(ordered[rank - 1]), int(rank)


def _conformal(
    frame,
    variables,
    arrays,
    model,
    calibration_rows,
    test_rows,
    *,
    mondrian=False,
):
    scaler, _, current, current_mask, _, future, future_mask, _, horizon, _ = arrays[:10]
    cal_pred = _forward(model, arrays, calibration_rows)
    test_pred = _forward(model, arrays, test_rows)
    cal_scale = _forward(
        model, arrays, calibration_rows, output_key="value_scale"
    )
    test_scale = _forward(
        model, arrays, test_rows, output_key="value_scale"
    )
    future_raw = future * scaler.scales + scaler.medians
    current_raw = current * scaler.scales + scaler.medians
    cal_raw = cal_pred * scaler.scales + scaler.medians
    test_raw = test_pred * scaler.scales + scaler.medians
    cal_scale_raw = np.maximum(
        cal_scale * np.abs(scaler.scales),
        1e-6,
    )
    test_scale_raw = np.maximum(
        test_scale * np.abs(scaler.scales),
        1e-6,
    )
    output = {}
    for index, target in enumerate(variables):
        for horizon_value in sorted(set(horizon[calibration_rows]).intersection(horizon[test_rows])):
            cal_mask = (
                (horizon[calibration_rows] == float(horizon_value))
                & np.asarray(future_mask[calibration_rows, index], dtype=bool)
            )
            test_mask = (
                (horizon[test_rows] == float(horizon_value))
                & np.asarray(future_mask[test_rows, index], dtype=bool)
            )
            if cal_mask.sum() < 30 or test_mask.sum() < 30:
                continue
            residual = np.abs(
                cal_raw[cal_mask, index]
                - future_raw[calibration_rows[cal_mask], index]
            )
            normalized_residual = residual / cal_scale_raw[cal_mask, index]
            q90, q90_rank = _finite_sample_conformal_quantile(
                normalized_residual,
                coverage=0.90,
            )
            observed = future_raw[test_rows[test_mask], index]
            predicted = test_raw[test_mask, index]
            width = q90 * test_scale_raw[test_mask, index]
            calibration_mode = "global_normalized"
            regime_cutpoints = []
            regime_q90 = []
            # A single global quantile is brittle for heavy-tailed laboratory
            # values: a normal AST and a markedly elevated AST have very
            # different residual distributions.  Anchor-value Mondrian
            # conformal keeps coverage conditional on a measurement-pure,
            # pre-forecast physiological regime.  It is used only when all
            # bins have enough calibration support; otherwise the global
            # normalized interval remains the fail-closed fallback.
            cal_rows_cell = calibration_rows[cal_mask]
            test_rows_cell = test_rows[test_mask]
            cal_anchor = current_raw[cal_rows_cell, index]
            test_anchor = current_raw[test_rows_cell, index]
            cal_anchor_observed = np.asarray(
                current_mask[cal_rows_cell, index], dtype=bool
            ) & np.isfinite(cal_anchor)
            if bool(mondrian) and int(cal_anchor_observed.sum()) >= 180:
                cuts = np.unique(
                    np.quantile(
                        cal_anchor[cal_anchor_observed],
                        [1.0 / 3.0, 2.0 / 3.0],
                    )
                )
                if len(cuts) == 2:
                    cal_regime = np.searchsorted(cuts, cal_anchor, side="right")
                    test_regime = np.searchsorted(cuts, test_anchor, side="right")
                    supported = all(
                        int(
                            (
                                cal_anchor_observed
                                & (cal_regime == regime)
                            ).sum()
                        )
                        >= 50
                        for regime in range(3)
                    )
                    if supported:
                        q_by_regime = []
                        for regime in range(3):
                            regime_mask = (
                                cal_anchor_observed
                                & (cal_regime == regime)
                            )
                            regime_q, _ = _finite_sample_conformal_quantile(
                                normalized_residual[regime_mask],
                                coverage=0.90,
                            )
                            q_by_regime.append(regime_q)
                        finite_test_anchor = np.asarray(
                            current_mask[test_rows_cell, index], dtype=bool
                        ) & np.isfinite(test_anchor)
                        if bool(finite_test_anchor.any()):
                            regime_width = np.asarray(q_by_regime)[
                                test_regime[finite_test_anchor]
                            ] * test_scale_raw[test_mask, index][
                                finite_test_anchor
                            ]
                            width[finite_test_anchor] = regime_width
                            calibration_mode = "anchor_regime_normalized"
                            regime_cutpoints = [
                                float(value) for value in cuts
                            ]
                            regime_q90 = q_by_regime
            coverage = float(
                (
                    (observed >= predicted - width)
                    & (observed <= predicted + width)
                ).mean()
            )
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            output[key] = {
                "coverage": coverage,
                "calibration_n": int(cal_mask.sum()),
                "test_n": int(test_mask.sum()),
                "q90": q90,
                "q90_rank": q90_rank,
                "finite_sample_corrected": True,
                "normalized": True,
                "calibration_mode": calibration_mode,
                "regime_cutpoints": regime_cutpoints,
                "regime_q90": regime_q90,
                "median_test_scale": float(
                    np.median(test_scale_raw[test_mask, index])
                ),
                "median_interval_half_width": float(np.median(width)),
                "pass": bool(0.87 <= coverage <= 0.93),
            }
    return output


def _module_balanced_metrics(frame, variables, arrays, prediction, rows, module_names):
    """Require a joint prediction to work across multiple observed body modules.

    This is deliberately a coverage guard rather than another aggregate score.
    A cell is eligible only when at least three modules have enough held-out
    patients, and at least 80% of those module-conditioned subsets improve on
    persistence.  It prevents a joint candidate from earning runtime use by
    improving only one frequent organ-pair signature.
    """

    scaler, _, current, _, _, future, future_mask, _, horizon, presence = arrays[:10]
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    prediction_raw = prediction * scaler.scales + scaler.medians
    output = {}
    for index, target in enumerate(variables):
        for horizon_value in sorted(set(horizon[rows])):
            observed = (
                (horizon[rows] == float(horizon_value))
                & np.asarray(future_mask[rows, index], dtype=bool)
            )
            if not observed.any():
                continue
            local_rows = np.flatnonzero(observed)
            selected = rows[local_rows]
            modules = []
            for module_index, module_name in enumerate(module_names):
                keep = presence[selected, module_index] > 0.5
                module_rows = selected[keep]
                if len(module_rows) < 30:
                    continue
                subject_count = int(frame.iloc[module_rows]["subject_id"].astype(str).nunique())
                if subject_count < 20:
                    continue
                local = local_rows[keep]
                candidate = float(
                    np.abs(prediction_raw[local, index] - future_raw[module_rows, index]).mean()
                )
                persistence = float(
                    np.abs(current_raw[module_rows, index] - future_raw[module_rows, index]).mean()
                )
                modules.append(
                    {
                        "module": str(module_name),
                        "rows": int(len(module_rows)),
                        "subjects": subject_count,
                        "delta_vs_persistence": candidate - persistence,
                        "pass": bool(candidate < persistence),
                    }
                )
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            pass_rate = float(np.mean([entry["pass"] for entry in modules])) if modules else 0.0
            output[key] = {
                "eligible_modules": int(len(modules)),
                "pass_rate": pass_rate,
                "pass": bool(len(modules) >= 3 and pass_rate >= 0.80),
                "modules": modules,
            }
    return output


def _aggregate(results):
    keys = sorted(set().union(*(result.keys() for result in results)))
    output = {}
    for key in keys:
        cells = [result[key] for result in results if key in result]
        output[key] = {
            "n_splits": len(cells),
            "value_pass_splits": int(sum(cell.get("pass_value", False) for cell in cells)),
            "cross_system_pass_splits": int(sum(cell.get("pass_cross_system", False) for cell in cells)),
            "patient_bootstrap_pass_splits": int(
                sum(cell.get("pass_patient_bootstrap", False) for cell in cells)
            ),
            "validated_router_value_pass_splits": int(
                sum(cell.get("pass_validated_router_value", False) for cell in cells)
            ),
            "validated_router_bootstrap_pass_splits": int(
                sum(
                    cell.get("pass_validated_router_bootstrap", False)
                    for cell in cells
                )
            ),
            "capacity_placebo_value_pass_splits": int(
                sum(
                    cell.get("pass_capacity_placebo_value", False)
                    for cell in cells
                )
            ),
            "capacity_placebo_bootstrap_pass_splits": int(
                sum(
                    cell.get("pass_capacity_placebo_bootstrap", False)
                    for cell in cells
                )
            ),
            "median_delta_vs_persistence": float(np.median([cell["delta_vs_persistence"] for cell in cells])),
            "median_patient_delta_vs_persistence": float(
                np.median(
                    [
                        cell["patient_equal_delta_vs_persistence"]
                        for cell in cells
                        if cell.get("patient_equal_delta_vs_persistence") is not None
                    ]
                )
            ) if any(cell.get("patient_equal_delta_vs_persistence") is not None for cell in cells) else None,
            "median_delta_vs_self_only": float(
                np.median([cell["delta_vs_self_only"] for cell in cells if "delta_vs_self_only" in cell])
            )
            if any("delta_vs_self_only" in cell for cell in cells)
            else None,
            "median_delta_vs_validated_router": float(
                np.median(
                    [
                        cell["delta_vs_validated_router"]
                        for cell in cells
                        if "delta_vs_validated_router" in cell
                    ]
                )
            )
            if any("delta_vs_validated_router" in cell for cell in cells)
            else None,
            "median_delta_vs_capacity_placebo": float(
                np.median(
                    [
                        cell["delta_vs_capacity_placebo"]
                        for cell in cells
                        if "delta_vs_capacity_placebo" in cell
                    ]
                )
            )
            if any("delta_vs_capacity_placebo" in cell for cell in cells)
            else None,
            "pass_all_value_splits": bool(len(cells) == len(results) and all(cell.get("pass_value", False) for cell in cells)),
            "pass_all_patient_bootstrap": bool(
                len(cells) == len(results)
                and all(cell.get("pass_patient_bootstrap", False) for cell in cells)
            ),
            "pass_all_validated_router_value": bool(
                len(cells) == len(results)
                and all(
                    cell.get("pass_validated_router_value", False)
                    for cell in cells
                )
            ),
            "pass_all_validated_router_bootstrap": bool(
                len(cells) == len(results)
                and all(
                    cell.get("pass_validated_router_bootstrap", False)
                    for cell in cells
                )
            ),
            "pass_all_capacity_placebo_value": bool(
                len(cells) == len(results)
                and all(
                    cell.get("pass_capacity_placebo_value", False)
                    for cell in cells
                )
            ),
            "pass_all_capacity_placebo_bootstrap": bool(
                len(cells) == len(results)
                and all(
                    cell.get("pass_capacity_placebo_bootstrap", False)
                    for cell in cells
                )
            ),
            "pass_all_cross_system_splits": bool(
                len(cells) == len(results)
                and all(cell.get("pass_cross_system", False) for cell in cells if "pass_cross_system" in cell)
                and any("pass_cross_system" in cell for cell in cells)
            ),
        }
    return output


def _aggregate_conformal(results):
    keys = sorted(set().union(*(result.keys() for result in results)))
    output = {}
    z = 1.959963984540054
    for key in keys:
        cells = [result[key] for result in results if key in result]
        compatible = []
        weighted_covered = 0.0
        total = 0
        for cell in cells:
            n = int(cell.get("test_n", 0))
            coverage = float(cell.get("coverage", np.nan))
            if n <= 0 or not np.isfinite(coverage):
                compatible.append(False)
                continue
            successes = int(round(coverage * n))
            proportion = successes / n
            denominator = 1.0 + (z * z) / n
            center = (
                proportion + (z * z) / (2.0 * n)
            ) / denominator
            half_width = (
                z
                * np.sqrt(
                    proportion * (1.0 - proportion) / n
                    + (z * z) / (4.0 * n * n)
                )
                / denominator
            )
            point_in_band = 0.87 <= coverage <= 0.93
            compatible.append(
                bool(
                    point_in_band
                    or center - half_width <= 0.90 <= center + half_width
                )
            )
            weighted_covered += coverage * n
            total += n
        pooled_coverage = (
            float(weighted_covered / total) if total > 0 else None
        )
        complete = len(cells) == len(results)
        calibration_compatible = bool(
            complete and compatible and all(compatible)
        )
        pooled_pass = bool(
            pooled_coverage is not None
            and 0.87 <= pooled_coverage <= 0.93
        )
        output[key] = {
            "n_splits": len(cells),
            "split_coverages": [
                float(cell["coverage"]) for cell in cells
            ],
            "split_test_n": [
                int(cell["test_n"]) for cell in cells
            ],
            "pass_splits": sum(
                bool(cell.get("pass", False)) for cell in cells
            ),
            "point_pass_splits": sum(
                bool(cell.get("pass", False)) for cell in cells
            ),
            "calibration_compatible_splits": sum(compatible),
            "pooled_test_n": int(total),
            "pooled_coverage": pooled_coverage,
            "pooled_coverage_pass": pooled_pass,
            # The target remains exactly 90% with the same 0.87-0.93 pooled
            # boundary. A split passes when its point coverage is already in
            # that band, or when Wilson compatibility shows that a small
            # excursion is consistent with finite-sample fluctuation.
            "pass_all": bool(
                complete and pooled_pass and calibration_compatible
            ),
        }
    return output


def _aggregate_module_balance(results):
    keys = sorted(set().union(*(result.keys() for result in results)))
    return {
        key: {
            "n_splits": sum(key in result for result in results),
            "pass_splits": sum(bool(result.get(key, {}).get("pass", False)) for result in results),
            "pass_all": bool(
                all(key in result and result[key].get("pass", False) for result in results)
            ),
        }
        for key in keys
    }


def _evaluate_external_holdout(
    frame,
    variables,
    module_names,
    train_rows,
    test_rows,
    *,
    seed,
    fit_options,
    cross_indices,
    variable_module_membership,
    regime_count,
    include_treatment_context,
    belief_placebo_gate,
    mondrian_conformal,
):
    """Fit and score one non-patient external split with identical gates."""

    if len(train_rows) == 0 or len(test_rows) == 0:
        return {
            "available": False,
            "reason": "empty_train_or_test_partition",
            "values": {},
            "conformal": {},
            "module_balanced": {},
            "regime_metrics": {},
            "reconstruction": {},
        }
    fit_rows, calibration_rows = _split_train_calibration(
        frame, np.asarray(train_rows, dtype=np.int64), seed + 1001
    )
    model, arrays, losses = fit_joint(
        frame,
        variables,
        module_names,
        fit_rows,
        seed=seed,
        **fit_options,
    )
    regime_ids, delta_bin_ids, regime_metadata = _build_supervision_strata(
        arrays, fit_rows, regime_count=regime_count
    )
    prediction = _forward(model, arrays, test_rows)
    placebo_model = None
    placebo_arrays = None
    placebo_prediction = None
    if belief_placebo_gate:
        placebo_options = dict(fit_options)
        placebo_options["belief_input_mode"] = "placebo_time_only"
        placebo_model, placebo_arrays, _ = fit_joint(
            frame,
            variables,
            module_names,
            fit_rows,
            seed=seed + 50_000,
            **placebo_options,
        )
        placebo_prediction = _forward(
            placebo_model, placebo_arrays, test_rows
        )
    router_prediction, router_selection = _validated_router_prediction(
        frame,
        variables,
        module_names,
        arrays,
        fit_rows,
        calibration_rows,
        test_rows,
        include_treatment_context=include_treatment_context,
        seed=seed + 30_000,
    )
    ablated_views = [
        (
            index,
            _ablate_target(
                arrays,
                index,
                len(variables),
                len(module_names),
                variable_module_membership,
            ),
        )
        for index in sorted(cross_indices)
    ]
    batched_ablations = _forward_many(model, ablated_views, test_rows)
    self_predictions = {
        index: batched_ablations[index]["value"]
        for index in sorted(cross_indices)
    }
    reconstruction_predictions = {
        index: batched_ablations[index]["reconstruction"]
        for index in sorted(cross_indices)
    }
    output = {
        "available": True,
        "train_rows": int(len(train_rows)),
        "fit_rows": int(len(fit_rows)),
        "calibration_rows": int(len(calibration_rows)),
        "test_rows": int(len(test_rows)),
        "loss": float(losses[-1]) if losses else None,
        "values": _metrics(
            frame,
            variables,
            arrays,
            prediction,
            test_rows,
            self_predictions,
            fit_rows,
            router_prediction,
            placebo_prediction,
        ),
        "validated_router_selection": router_selection,
        "conformal": _conformal(
            frame,
            variables,
            arrays,
            model,
            calibration_rows,
            test_rows,
            mondrian=mondrian_conformal,
        ),
        "module_balanced": _module_balanced_metrics(
            frame,
            variables,
            arrays,
            prediction,
            test_rows,
            module_names,
        ),
        "regime_metrics": _regime_metrics(
            variables,
            arrays,
            prediction,
            test_rows,
            regime_ids,
            delta_bin_ids,
        ),
        "regime_metadata": regime_metadata,
        "reconstruction": _reconstruction_metrics(
            variables,
            arrays,
            reconstruction_predictions,
            test_rows,
        ),
    }
    del batched_ablations, ablated_views, model, arrays
    if placebo_model is not None:
        del placebo_model, placebo_arrays
    gc.collect()
    return output


def run_gate(
    frame,
    variables,
    module_names,
    epochs=2,
    batch_size=2048,
    measurement_process_mode="observed",
    measurement_time_head=False,
    measurement_time_weight=0.10,
    uncertainty_calibration_weight=0.05,
    mondrian_conformal=False,
    include_treatment_context=False,
    cross_forecast_weight=1.0,
    masked_forecast_probability=0.50,
    module_mask_probability=0.75,
    nowcast_pretrain_epochs=0,
    measurement_consistency_weight=0.0,
    world_model=False,
    world_model_weight=1.0,
    target_momentum=0.99,
    future_target_mode="window",
    target_min_observations=2,
    regime_balanced_loss=False,
    regime_count=3,
    state_normalization="standard",
    hospital_invariance_weight=0.0,
    target_encoder_mode="fast",
    target_horizon_regime_adapter=False,
    patient_residual_adapter=False,
    future_task_balanced_sampling=False,
    coupling_preservation_weight=0.0,
    uncertainty_gate=False,
    validated_edge_adapters=False,
    validated_edge_weight=0.0,
    validated_edge_stage_epochs=0,
    validated_edge_targets=None,
    target_head_stage_epochs=0,
    source_group_adapters=False,
    source_group_contextual_regime_gate=False,
    source_group_targeted_treatment_context=False,
    source_group_adapter_weight=0.0,
    source_group_adapter_stage_epochs=0,
    source_group_adapter_targets=None,
    cross_targets=None,
    extended_holdouts=True,
    belief_input_mode="observed",
    belief_placebo_gate=False,
):
    if belief_placebo_gate and belief_input_mode != "observed":
        raise ValueError(
            "belief_placebo_gate requires belief_input_mode='observed'"
        )
    patient_metrics, patient_conformal, patient_reconstruction = [], [], []
    patient_module_balanced = []
    patient_regime_metrics = []
    patient_router_selection = []
    cross_target_names = set(CROSS_SYSTEM_TARGETS if cross_targets is None else cross_targets)
    cross_indices = {
        index for index, target in enumerate(variables) if target in cross_target_names
    }
    module_variables = frame.attrs.get("module_variables", {})
    variable_module_membership = np.zeros((len(module_names), len(variables)), dtype=np.float32)
    for index, variable in enumerate(variables):
        for module_index, module in enumerate(module_names):
            if variable in set(module_variables.get(module, ())):
                variable_module_membership[module_index, index] = 1.0
    fit_options = {
        "epochs": epochs,
        "batch_size": batch_size,
        "measurement_process_mode": measurement_process_mode,
        "measurement_time_head": measurement_time_head,
        "measurement_time_weight": measurement_time_weight,
        "uncertainty_calibration_weight": uncertainty_calibration_weight,
        "include_treatment_context": include_treatment_context,
        "cross_forecast_weight": cross_forecast_weight,
        "masked_forecast_probability": masked_forecast_probability,
        "module_mask_probability": module_mask_probability,
        "nowcast_pretrain_epochs": nowcast_pretrain_epochs,
        "measurement_consistency_weight": measurement_consistency_weight,
        "world_model": world_model,
        "world_model_weight": world_model_weight,
        "target_momentum": target_momentum,
        "future_target_mode": future_target_mode,
        "target_min_observations": target_min_observations,
        "regime_balanced_loss": regime_balanced_loss,
        "regime_count": regime_count,
        "state_normalization": state_normalization,
        "hospital_invariance_weight": hospital_invariance_weight,
        "target_encoder_mode": target_encoder_mode,
        "target_horizon_regime_adapter": target_horizon_regime_adapter,
        "patient_residual_adapter": patient_residual_adapter,
        "future_task_balanced_sampling": future_task_balanced_sampling,
        "coupling_preservation_weight": coupling_preservation_weight,
        "uncertainty_gate": uncertainty_gate,
        "validated_edge_adapters": validated_edge_adapters,
        "validated_edge_weight": validated_edge_weight,
        "validated_edge_stage_epochs": validated_edge_stage_epochs,
        "target_head_stage_epochs": target_head_stage_epochs,
        "validated_edge_targets": validated_edge_targets,
        "source_group_adapters": source_group_adapters,
        "source_group_contextual_regime_gate": source_group_contextual_regime_gate,
        "source_group_targeted_treatment_context": source_group_targeted_treatment_context,
        "source_group_adapter_weight": source_group_adapter_weight,
        "source_group_adapter_stage_epochs": source_group_adapter_stage_epochs,
        "source_group_adapter_targets": source_group_adapter_targets,
        "belief_input_mode": belief_input_mode,
    }
    for seed in SEEDS:
        train_rows, test_rows = _split_by_column(frame, "subject_id", seed)
        fit_rows, calibration_rows = _split_train_calibration(frame, train_rows, seed + 1001)
        model, arrays, losses = fit_joint(
            frame,
            variables,
            module_names,
            fit_rows,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            measurement_process_mode=measurement_process_mode,
            measurement_time_head=measurement_time_head,
            measurement_time_weight=measurement_time_weight,
            uncertainty_calibration_weight=uncertainty_calibration_weight,
            include_treatment_context=include_treatment_context,
            cross_forecast_weight=cross_forecast_weight,
            masked_forecast_probability=masked_forecast_probability,
            module_mask_probability=module_mask_probability,
            nowcast_pretrain_epochs=nowcast_pretrain_epochs,
            measurement_consistency_weight=measurement_consistency_weight,
            world_model=world_model,
            world_model_weight=world_model_weight,
            target_momentum=target_momentum,
            future_target_mode=future_target_mode,
            target_min_observations=target_min_observations,
            regime_balanced_loss=regime_balanced_loss,
            regime_count=regime_count,
            state_normalization=state_normalization,
            hospital_invariance_weight=hospital_invariance_weight,
            target_encoder_mode=target_encoder_mode,
            target_horizon_regime_adapter=target_horizon_regime_adapter,
            patient_residual_adapter=patient_residual_adapter,
            future_task_balanced_sampling=future_task_balanced_sampling,
            coupling_preservation_weight=coupling_preservation_weight,
            uncertainty_gate=uncertainty_gate,
            validated_edge_adapters=validated_edge_adapters,
            validated_edge_weight=validated_edge_weight,
            validated_edge_stage_epochs=validated_edge_stage_epochs,
            target_head_stage_epochs=target_head_stage_epochs,
            validated_edge_targets=validated_edge_targets,
            source_group_adapters=source_group_adapters,
            source_group_contextual_regime_gate=source_group_contextual_regime_gate,
            source_group_targeted_treatment_context=source_group_targeted_treatment_context,
            source_group_adapter_weight=source_group_adapter_weight,
            source_group_adapter_stage_epochs=source_group_adapter_stage_epochs,
            source_group_adapter_targets=source_group_adapter_targets,
            belief_input_mode=belief_input_mode,
        )
        regime_ids, delta_bin_ids, regime_metadata = _build_supervision_strata(
            arrays, fit_rows, regime_count=regime_count
        )
        prediction = _forward(model, arrays, test_rows)
        placebo_prediction = None
        placebo_model = None
        placebo_arrays = None
        if belief_placebo_gate:
            placebo_options = dict(fit_options)
            placebo_options["belief_input_mode"] = "placebo_time_only"
            placebo_model, placebo_arrays, _ = fit_joint(
                frame,
                variables,
                module_names,
                fit_rows,
                seed=seed + 50_000,
                **placebo_options,
            )
            placebo_prediction = _forward(
                placebo_model, placebo_arrays, test_rows
            )
        router_prediction, router_selection = _validated_router_prediction(
            frame,
            variables,
            module_names,
            arrays,
            fit_rows,
            calibration_rows,
            test_rows,
            include_treatment_context=include_treatment_context,
            seed=seed + 30_000,
        )
        self_predictions = {}
        reconstruction_predictions = {}
        ablated_views = [
            (
                index,
                _ablate_target(
                    arrays,
                    index,
                    len(variables),
                    len(module_names),
                    variable_module_membership,
                ),
            )
            for index in sorted(cross_indices)
        ]
        batched_ablations = _forward_many(model, ablated_views, test_rows)
        for index in sorted(cross_indices):
            self_predictions[index] = batched_ablations[index]["value"]
            reconstruction_predictions[index] = batched_ablations[index]["reconstruction"]
        patient_metrics.append(
            _metrics(
                frame,
                variables,
                arrays,
                prediction,
                test_rows,
                self_predictions,
                fit_rows,
                router_prediction,
                placebo_prediction,
            )
        )
        patient_router_selection.append(router_selection)
        patient_regime_metrics.append(
            _regime_metrics(variables, arrays, prediction, test_rows, regime_ids, delta_bin_ids)
        )
        patient_reconstruction.append(_reconstruction_metrics(variables, arrays, reconstruction_predictions, test_rows))
        patient_conformal.append(
            _conformal(
                frame,
                variables,
                arrays,
                model,
                calibration_rows,
                test_rows,
                mondrian=mondrian_conformal,
            )
        )
        patient_module_balanced.append(
            _module_balanced_metrics(frame, variables, arrays, prediction, test_rows, module_names)
        )
        print(json.dumps({"scope": "patient", "seed": seed, "cells": len(patient_metrics[-1]), "loss": losses[-1]}), flush=True)
        # Each split owns a large joint array plus several ablation views.
        # Release them before fitting the next seed so full-body gates do not
        # fail from allocator growth rather than from model behavior.
        del batched_ablations, ablated_views, model, arrays
        if placebo_model is not None:
            del placebo_model, placebo_arrays
        gc.collect()

    hospital_train, hospital_test = _split_by_column(frame, "hospitalid", 2026)
    fit_rows, calibration_rows = _split_train_calibration(frame, hospital_train, 3027)
    model, arrays, losses = fit_joint(
        frame,
        variables,
        module_names,
        fit_rows,
        seed=2026,
        epochs=epochs,
        batch_size=batch_size,
        measurement_process_mode=measurement_process_mode,
        measurement_time_head=measurement_time_head,
        measurement_time_weight=measurement_time_weight,
        uncertainty_calibration_weight=uncertainty_calibration_weight,
        include_treatment_context=include_treatment_context,
        cross_forecast_weight=cross_forecast_weight,
        masked_forecast_probability=masked_forecast_probability,
        module_mask_probability=module_mask_probability,
        nowcast_pretrain_epochs=nowcast_pretrain_epochs,
        measurement_consistency_weight=measurement_consistency_weight,
        world_model=world_model,
        world_model_weight=world_model_weight,
        target_momentum=target_momentum,
        future_target_mode=future_target_mode,
        target_min_observations=target_min_observations,
        regime_balanced_loss=regime_balanced_loss,
        regime_count=regime_count,
        state_normalization=state_normalization,
        hospital_invariance_weight=hospital_invariance_weight,
        target_encoder_mode=target_encoder_mode,
        target_horizon_regime_adapter=target_horizon_regime_adapter,
        patient_residual_adapter=patient_residual_adapter,
        future_task_balanced_sampling=future_task_balanced_sampling,
        coupling_preservation_weight=coupling_preservation_weight,
        uncertainty_gate=uncertainty_gate,
        validated_edge_adapters=validated_edge_adapters,
        validated_edge_weight=validated_edge_weight,
        validated_edge_stage_epochs=validated_edge_stage_epochs,
        target_head_stage_epochs=target_head_stage_epochs,
        validated_edge_targets=validated_edge_targets,
        source_group_adapters=source_group_adapters,
        source_group_contextual_regime_gate=source_group_contextual_regime_gate,
        source_group_targeted_treatment_context=source_group_targeted_treatment_context,
        source_group_adapter_weight=source_group_adapter_weight,
        source_group_adapter_stage_epochs=source_group_adapter_stage_epochs,
        source_group_adapter_targets=source_group_adapter_targets,
        belief_input_mode=belief_input_mode,
    )
    hospital_regime_ids, hospital_delta_bin_ids, hospital_regime_metadata = _build_supervision_strata(
        arrays, fit_rows, regime_count=regime_count
    )
    prediction = _forward(model, arrays, hospital_test)
    hospital_placebo_prediction = None
    hospital_placebo_model = None
    hospital_placebo_arrays = None
    if belief_placebo_gate:
        placebo_options = dict(fit_options)
        placebo_options["belief_input_mode"] = "placebo_time_only"
        (
            hospital_placebo_model,
            hospital_placebo_arrays,
            _,
        ) = fit_joint(
            frame,
            variables,
            module_names,
            fit_rows,
            seed=52_026,
            **placebo_options,
        )
        hospital_placebo_prediction = _forward(
            hospital_placebo_model,
            hospital_placebo_arrays,
            hospital_test,
        )
    hospital_router_prediction, hospital_router_selection = (
        _validated_router_prediction(
            frame,
            variables,
            module_names,
            arrays,
            fit_rows,
            calibration_rows,
            hospital_test,
            include_treatment_context=include_treatment_context,
            seed=32_026,
        )
    )
    self_predictions = {}
    reconstruction_predictions = {}
    ablated_views = [
        (
            index,
            _ablate_target(
                arrays,
                index,
                len(variables),
                len(module_names),
                variable_module_membership,
            ),
        )
        for index in sorted(cross_indices)
    ]
    batched_ablations = _forward_many(model, ablated_views, hospital_test)
    for index in sorted(cross_indices):
        self_predictions[index] = batched_ablations[index]["value"]
        reconstruction_predictions[index] = batched_ablations[index]["reconstruction"]
    hospital_metrics = _metrics(
        frame,
        variables,
        arrays,
        prediction,
        hospital_test,
        self_predictions,
        fit_rows,
        hospital_router_prediction,
        hospital_placebo_prediction,
    )
    hospital_regime_metrics = _regime_metrics(
        variables, arrays, prediction, hospital_test,
        hospital_regime_ids, hospital_delta_bin_ids,
    )
    hospital_reconstruction = _reconstruction_metrics(
        variables, arrays, reconstruction_predictions, hospital_test
    )
    hospital_conformal = _conformal(
        frame,
        variables,
        arrays,
        model,
        calibration_rows,
        hospital_test,
        mondrian=mondrian_conformal,
    )
    hospital_module_balanced = _module_balanced_metrics(
        frame, variables, arrays, prediction, hospital_test, module_names
    )
    del model, arrays, batched_ablations, ablated_views
    if hospital_placebo_model is not None:
        del hospital_placebo_model, hospital_placebo_arrays
    gc.collect()
    unavailable_external = {
        "available": False,
        "reason": "extended_holdouts_disabled",
        "values": {},
        "conformal": {},
        "module_balanced": {},
        "regime_metrics": {},
        "reconstruction": {},
    }
    care_unit_heldout = dict(unavailable_external)
    time_heldout = dict(unavailable_external)
    if extended_holdouts:
        if _holdout_available(frame, "careunit"):
            care_train, care_test = _split_known_groups(
                frame, "careunit", 2027
            )
            care_unit_heldout = _evaluate_external_holdout(
                frame,
                variables,
                module_names,
                care_train,
                care_test,
                seed=2027,
                fit_options=fit_options,
                cross_indices=cross_indices,
                variable_module_membership=variable_module_membership,
                regime_count=regime_count,
                include_treatment_context=include_treatment_context,
                belief_placebo_gate=belief_placebo_gate,
                mondrian_conformal=mondrian_conformal,
            )
            care_unit_heldout["heldout_groups"] = sorted(
                frame.iloc[care_test]["careunit"].astype(str).unique().tolist()
            )
        else:
            care_unit_heldout = dict(unavailable_external)
            care_unit_heldout["reason"] = "careunit_provenance_unavailable"

        time_train, time_test = _split_forward_time(frame)
        time_heldout = _evaluate_external_holdout(
            frame,
            variables,
            module_names,
            time_train,
            time_test,
            seed=2028,
            fit_options=fit_options,
            cross_indices=cross_indices,
            variable_module_membership=variable_module_membership,
            regime_count=regime_count,
            include_treatment_context=include_treatment_context,
            belief_placebo_gate=belief_placebo_gate,
            mondrian_conformal=mondrian_conformal,
        )
        time_heldout["split_policy"] = (
            "latest 25% unique anchor times within each stay; "
            "all earlier anchors remain in training"
        )
    patient_aggregate = _aggregate(patient_metrics)
    patient_conf = _aggregate_conformal(patient_conformal)
    patient_module_aggregate = _aggregate_module_balance(patient_module_balanced)
    patient_reconstruction_aggregate = _aggregate_reconstruction(patient_reconstruction)
    belief_incremental_registry = {}
    if belief_placebo_gate:
        for key, value in patient_aggregate.items():
            hospital_value = hospital_metrics.get(key, {})
            care_value = care_unit_heldout.get("values", {}).get(key, {})
            time_value = time_heldout.get("values", {}).get(key, {})
            target = str(key).split("@", 1)[0]
            group = _belief_group_for_target(target)
            requirements = {
                "patient_value": value.get(
                    "pass_all_capacity_placebo_value", False
                ),
                "patient_bootstrap": value.get(
                    "pass_all_capacity_placebo_bootstrap", False
                ),
                "hospital_value": hospital_value.get(
                    "pass_capacity_placebo_value", False
                ),
                "hospital_bootstrap": hospital_value.get(
                    "pass_capacity_placebo_bootstrap", False
                ),
                "care_unit_value": bool(
                    care_unit_heldout.get("available")
                    and care_value.get("pass_capacity_placebo_value", False)
                ),
                "care_unit_bootstrap": bool(
                    care_unit_heldout.get("available")
                    and care_value.get("pass_capacity_placebo_bootstrap", False)
                ),
                "time_value": bool(
                    time_heldout.get("available")
                    and time_value.get("pass_capacity_placebo_value", False)
                ),
                "time_bootstrap": bool(
                    time_heldout.get("available")
                    and time_value.get("pass_capacity_placebo_bootstrap", False)
                ),
                "downstream_group_predeclared": group is not None,
            }
            belief_incremental_registry[key] = {
                "belief_group": group,
                "requirements": requirements,
                "validated": bool(all(requirements.values())),
                "semantics": (
                    "patient history carries incremental factual signal; "
                    "the latent is not a measured physiological quantity"
                ),
            }
    reconstruction_registry = {}
    reconstruction_validated = []
    for key, value in patient_reconstruction_aggregate.items():
        entry = {
            "patient_reconstruction": value.get("pass_all_splits", False),
            "hospital_reconstruction": hospital_reconstruction.get(key, {}).get(
                "pass_reconstruction", False
            ),
        }
        entry["validated"] = bool(all(entry.values()))
        reconstruction_registry[key] = entry
        if entry["validated"]:
            reconstruction_validated.append(key)
    registry = {}
    validated = []
    for key, value in patient_aggregate.items():
        target_name = str(key).split("@", 1)[0]
        cross_system_evaluated = target_name in cross_target_names
        h = hospital_metrics.get(key, {})
        hc = hospital_conformal.get(key, {})
        care_value = care_unit_heldout.get("values", {}).get(key, {})
        care_conformal = care_unit_heldout.get("conformal", {}).get(key, {})
        care_module = care_unit_heldout.get("module_balanced", {}).get(key, {})
        time_value = time_heldout.get("values", {}).get(key, {})
        time_conformal = time_heldout.get("conformal", {}).get(key, {})
        time_module = time_heldout.get("module_balanced", {}).get(key, {})
        entry = {
            "patient_value": value.get("pass_all_value_splits", False),
            "patient_bootstrap": value.get("pass_all_patient_bootstrap", False),
            "patient_validated_router_value": value.get(
                "pass_all_validated_router_value", False
            ),
            "patient_validated_router_bootstrap": value.get(
                "pass_all_validated_router_bootstrap", False
            ),
            "patient_cross_system": value.get("pass_all_cross_system_splits", False),
            "patient_module_balanced": patient_module_aggregate.get(key, {}).get("pass_all", False),
            "hospital_value": h.get("pass_value", False),
            "hospital_validated_router_value": h.get(
                "pass_validated_router_value", False
            ),
            "hospital_validated_router_bootstrap": h.get(
                "pass_validated_router_bootstrap", False
            ),
            "hospital_cross_system": h.get("pass_cross_system", False),
            "patient_conformal": patient_conf.get(key, {}).get("pass_all", False),
            "hospital_conformal": hc.get("pass", False),
            "hospital_module_balanced": hospital_module_balanced.get(key, {}).get("pass", False),
            "care_unit_value": bool(
                care_unit_heldout.get("available")
                and care_value.get("pass_value", False)
            ),
            "care_unit_validated_router_value": bool(
                care_unit_heldout.get("available")
                and care_value.get("pass_validated_router_value", False)
            ),
            "care_unit_validated_router_bootstrap": bool(
                care_unit_heldout.get("available")
                and care_value.get("pass_validated_router_bootstrap", False)
            ),
            "care_unit_cross_system": bool(
                care_unit_heldout.get("available")
                and care_value.get("pass_cross_system", False)
            ),
            "care_unit_conformal": bool(
                care_unit_heldout.get("available")
                and care_conformal.get("pass", False)
            ),
            "care_unit_module_balanced": bool(
                care_unit_heldout.get("available")
                and care_module.get("pass", False)
            ),
            "time_value": bool(
                time_heldout.get("available")
                and time_value.get("pass_value", False)
            ),
            "time_validated_router_value": bool(
                time_heldout.get("available")
                and time_value.get("pass_validated_router_value", False)
            ),
            "time_validated_router_bootstrap": bool(
                time_heldout.get("available")
                and time_value.get("pass_validated_router_bootstrap", False)
            ),
            "time_cross_system": bool(
                time_heldout.get("available")
                and time_value.get("pass_cross_system", False)
            ),
            "time_conformal": bool(
                time_heldout.get("available")
                and time_conformal.get("pass", False)
            ),
            "time_module_balanced": bool(
                time_heldout.get("available")
                and time_module.get("pass", False)
            ),
        }
        # Cross-system ablation controls whether the runtime may attribute a
        # forecast to other organs. It is not a universal accuracy gate:
        # target-specific JEPA heads are allowed to win using their own organ
        # history, and targets outside CROSS_SYSTEM_TARGETS are not silently
        # rejected merely because no attribution audit was scheduled.
        entry = apply_joint_movement_policy(
            entry,
            cross_system_evaluated=cross_system_evaluated,
        )
        registry[key] = entry
        if entry["validated_for_joint_runtime"]:
            validated.append(key)
    strict = bool(registry and all(entry["validated"] for entry in registry.values()))
    if strict:
        promotion_status = "validated_joint_research_only"
    elif validated:
        # The shared latent is allowed per target-horizon cell.  Sparse or
        # unstable cells remain fail-closed and must use their fallback source.
        promotion_status = "validated_target_gated_joint_research_only"
    else:
        promotion_status = "candidate_only"
    return {
        "schema": "whole_body_joint_jepa_gate.v3",
        "modules": list(module_names),
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "mean_modules_per_row": float(frame["_module_count"].mean()),
        "cross_system_ablation_targets": sorted(cross_target_names.intersection(variables)),
        "patient_heldout": {
            "seeds": list(SEEDS),
            "aggregate": patient_aggregate,
            "conformal": patient_conf,
            "module_balanced": patient_module_aggregate,
            "validated_router_selection": patient_router_selection,
        },
        "patient_bootstrap_policy": {
            "primary_metric": "equal-weighted per-patient candidate-minus-persistence MAE",
            "samples": PATIENT_BOOTSTRAP_SAMPLES,
            "min_subjects": PATIENT_BOOTSTRAP_MIN_SUBJECTS,
            "pass_rule": "95% paired bootstrap upper bound < 0",
        },
        "patient_reconstruction": patient_reconstruction_aggregate,
        "patient_regime_metrics": patient_regime_metrics,
        "hospital_heldout": {
            "values": hospital_metrics,
            "conformal": hospital_conformal,
            "module_balanced": hospital_module_balanced,
            "validated_router_selection": hospital_router_selection,
        },
        "hospital_regime_metrics": hospital_regime_metrics,
        "hospital_reconstruction": hospital_reconstruction,
        "care_unit_heldout": care_unit_heldout,
        "time_heldout": time_heldout,
        "extended_holdouts_enabled": bool(extended_holdouts),
        "target_horizon_registry": registry,
        "joint_runtime_registry": registry,
        "cross_system_nowcast_registry": reconstruction_registry,
        "belief_incremental_registry": belief_incremental_registry,
        "belief_placebo_gate_enabled": bool(belief_placebo_gate),
        "belief_downstream_groups": {
            key: sorted(value)
            for key, value in BELIEF_DOWNSTREAM_GROUPS.items()
        },
        "validated_target_horizon_cells": sorted(validated),
        "validated_target_horizon_count": len(validated),
        "validated_joint_runtime_cells": sorted(validated),
        "validated_joint_runtime_cell_count": len(validated),
        "joint_runtime_data_contract": "as-of aligned whole-body state; exact anchor-linked future labels; target-gated joint-JEPA only",
        "validated_cross_system_nowcast_cells": sorted(reconstruction_validated),
        "validated_cross_system_nowcast_count": len(reconstruction_validated),
        "promotion_status": promotion_status,
        "measurement_process_mode": measurement_process_mode,
        "measurement_time_head": bool(measurement_time_head),
        "measurement_time_weight": float(measurement_time_weight),
        "mondrian_conformal": bool(mondrian_conformal),
        "treatment_context": bool(include_treatment_context),
        "state_normalization": state_normalization,
        "hospital_invariance_weight": float(hospital_invariance_weight),
        "hospital_adversarial_training_only": bool(float(hospital_invariance_weight) > 0.0),
        "hospital_id_inference_feature": False,
        "target_encoder_mode": target_encoder_mode,
        "target_horizon_regime_adapter": bool(target_horizon_regime_adapter),
        "patient_residual_adapter": bool(patient_residual_adapter),
        "future_task_balanced_sampling": bool(future_task_balanced_sampling),
        "coupling_preservation_weight": float(coupling_preservation_weight),
        "uncertainty_gate": bool(uncertainty_gate),
        "validated_edge_adapters": bool(validated_edge_adapters),
        "validated_edge_weight": float(validated_edge_weight),
        "validated_edge_stage_epochs": int(validated_edge_stage_epochs),
        "target_head_stage_epochs": int(target_head_stage_epochs),
        "validated_edge_targets": sorted(validated_edge_targets or ()),
        "source_group_adapters": bool(source_group_adapters),
        "source_group_contextual_regime_gate": bool(source_group_contextual_regime_gate),
        "source_group_targeted_treatment_context": bool(
            source_group_targeted_treatment_context
        ),
        "source_group_adapter_weight": float(source_group_adapter_weight),
        "source_group_adapter_stage_epochs": int(source_group_adapter_stage_epochs),
        "source_group_adapter_targets": sorted(source_group_adapter_targets or ()),
        "belief_input_mode": str(belief_input_mode),
        "cross_forecast_weight": float(cross_forecast_weight),
        "masked_forecast_probability": float(masked_forecast_probability),
        "module_mask_probability": float(module_mask_probability),
        "nowcast_pretrain_epochs": int(nowcast_pretrain_epochs),
        "measurement_consistency_weight": float(measurement_consistency_weight),
        "world_model": bool(world_model),
        "world_model_weight": float(world_model_weight),
        "target_momentum": float(target_momentum),
        "future_target_mode": future_target_mode,
        "target_min_observations": int(target_min_observations),
        "regime_balanced_loss": bool(regime_balanced_loss),
        "regime_count": int(regime_count),
        "regime_definition": "target-specific anchor-current quantile regime; diagnostic strata only",
        "future_delta_target": True,
        "measurement_pure_target": True,
        "cross_forecast_curriculum": True,
        "joint_latent_validated": strict,
        "joint_latent_partially_validated": bool(validated),
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "fail_closed_reason": None if strict else "only cells beating persistence and the nested validated router while passing patient, hospital, care-unit, forward-time, conformal, cross-system ablation, and module-balanced gates may move; all other cells fallback",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--event-examples",
        type=Path,
        default=None,
        help=(
            "Canonical event-ledger examples parquet. When supplied, this "
            "replaces legacy module transition labels as the joint cohort."
        ),
    )
    parser.add_argument(
        "--organ-contract-mode",
        choices=("canonical_organs", "overlap"),
        default="canonical_organs",
        help=(
            "Canonical organ ownership prevents disease-router variables from "
            "being copied into nearly every organ token."
        ),
    )
    parser.add_argument(
        "--history-mode",
        choices=("multiscale", "recent"),
        default="multiscale",
    )
    parser.add_argument("--history-steps", type=int, default=5)
    parser.add_argument(
        "--history-lags-hours",
        default="48,24,12,8,6,4,3,2,1,0",
    )
    parser.add_argument("--modules", default="sepsis,aki,respiratory,integumentary_skin_wound,toxic_metabolic,electrolyte_acid_base,endocrine_stress,gi_pancreatic_nutrition,cardiac_injury,musculoskeletal_rhabdo,immune_inflammatory,cardiovascular_instability,acute_neuro,hepatic_failure,coagulopathy_heme")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--alignment", choices=("exact", "asof"), default="asof")
    parser.add_argument("--asof-max-age-hours", type=float, default=6.0)
    parser.add_argument(
        "--asof-age-policy",
        choices=("fixed", "horizon_scaled"),
        default="horizon_scaled",
    )
    parser.add_argument("--future-label-tolerance-hours", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cross-forecast-weight", type=float, default=1.0)
    parser.add_argument("--masked-forecast-probability", type=float, default=0.50)
    parser.add_argument("--module-mask-probability", type=float, default=0.75)
    parser.add_argument(
        "--nowcast-pretrain-epochs",
        type=int,
        default=0,
        help="Masked same-anchor reconstruction epochs before future forecasting.",
    )
    parser.add_argument(
        "--measurement-consistency-weight",
        type=float,
        default=0.0,
        help="Weight for latent consistency across two independent missingness views.",
    )
    parser.add_argument(
        "--world-model",
        action="store_true",
        help="Train the true JEPA latent-prediction candidate with an EMA target encoder.",
    )
    parser.add_argument("--world-model-weight", type=float, default=1.0)
    parser.add_argument("--target-momentum", type=float, default=0.99)
    parser.add_argument(
        "--future-target-mode",
        choices=("window", "single"),
        default="window",
        help="JEPA target: causal future trajectory prefix or one future row.",
    )
    parser.add_argument(
        "--target-min-observations",
        type=int,
        default=2,
        help="Minimum observed variables required for a future JEPA step.",
    )
    parser.add_argument(
        "--regime-balanced-loss",
        action="store_true",
        help="Balance future loss by target, actual delta-time bin, and anchor regime.",
    )
    parser.add_argument("--regime-count", type=int, default=3)
    parser.add_argument(
        "--measurement-process-mode",
        choices=("observed", "neutralized"),
        default="observed",
    )
    parser.add_argument(
        "--measurement-time-head",
        action="store_true",
        help="Predict the next timestamped measurement delay as a separate observation-process task.",
    )
    parser.add_argument(
        "--measurement-time-weight",
        type=float,
        default=0.10,
        help="Loss weight for the separate next-measurement-time head.",
    )
    parser.add_argument(
        "--uncertainty-calibration-weight",
        type=float,
        default=0.05,
        help=(
            "Train detached target-specific scale heads for normalized "
            "conformal intervals without moving point forecasts."
        ),
    )
    parser.add_argument(
        "--mondrian-conformal",
        action="store_true",
        help=(
            "Calibrate normalized conformal residuals separately by "
            "measurement-pure anchor-value regime when support is sufficient."
        ),
    )
    parser.add_argument(
        "--include-treatment-context",
        action="store_true",
        help=(
            "Use only hist_* treatment evidence available at or before the "
            "anchor; act_* future-window actions remain excluded."
        ),
    )
    parser.add_argument(
        "--state-normalization",
        choices=("standard", "robust"),
        default="standard",
        help="Training-only global normalization; robust uses median/MAD.",
    )
    parser.add_argument(
        "--hospital-invariance-weight",
        type=float,
        default=0.0,
        help="Training-only adversarial hospital-domain loss weight.",
    )
    parser.add_argument(
        "--target-encoder-mode",
        choices=("full", "fast"),
        default="fast",
        help="Future JEPA target encoder: full mixers or fast stop-gradient view.",
    )
    parser.add_argument(
        "--target-horizon-regime-adapter",
        action="store_true",
        help="Enable target-specific horizon x anchor-regime residual adapters.",
    )
    parser.add_argument(
        "--patient-residual-adapter",
        action="store_true",
        help="Enable target-specific residuals derived from each patient's history latent.",
    )
    parser.add_argument(
        "--future-task-balanced-sampling",
        action="store_true",
        help="Sample future task cells by inverse target-horizon-regime frequency.",
    )
    parser.add_argument(
        "--coupling-preservation-weight",
        type=float,
        default=0.0,
        help="Explicit loss weight for target-ablated cross-organ future views.",
    )
    parser.add_argument(
        "--uncertainty-gate",
        action="store_true",
        help="Gate forecast movement by the patient belief uncertainty; otherwise use persistence directly.",
    )
    parser.add_argument(
        "--validated-edge-adapters",
        action="store_true",
        help="Enable adapters only for the validated cardio-renal and sepsis-MAP edges.",
    )
    parser.add_argument(
        "--validated-edge-weight",
        type=float,
        default=0.0,
        help="Train validated edge adapters on source-present/target-masked future views.",
    )
    parser.add_argument(
        "--validated-edge-stage-epochs",
        type=int,
        default=0,
        help="Freeze the body/local heads and train validated edge adapters separately.",
    )
    parser.add_argument(
        "--target-head-stage-epochs",
        type=int,
        default=0,
        help=(
            "Freeze the shared whole-body latent and specialize every "
            "target-specific temporal/value/uncertainty JEPA branch."
        ),
    )
    parser.add_argument(
        "--validated-edge-targets",
        default=None,
        help="Comma-separated source_module->target residual adapters; defaults to every pre-validated edge.",
    )
    parser.add_argument(
        "--source-group-adapters",
        action="store_true",
        help="Enable attribution-backed source-group adapters for a separate candidate experiment.",
    )
    parser.add_argument(
        "--source-group-contextual-regime-gate",
        action="store_true",
        help=(
            "Give an attribution-backed group adapter direct as-of source values "
            "and a patient-state movement gate; experiment-only."
        ),
    )
    parser.add_argument(
        "--source-group-targeted-treatment-context",
        action="store_true",
        help=(
            "Keep selected hist_* treatment context out of the shared body latent "
            "and expose it only to the matching source-group adapter."
        ),
    )
    parser.add_argument(
        "--source-group-adapter-weight",
        type=float,
        default=0.0,
        help="Joint-training loss weight for source-group adapters; use zero with a separate adapter stage.",
    )
    parser.add_argument(
        "--source-group-adapter-stage-epochs",
        type=int,
        default=0,
        help="Freeze the base model and fit source-group adapters for this many epochs.",
    )
    parser.add_argument(
        "--source-group-adapter-targets",
        default=None,
        help="Comma-separated attribution-backed group adapters, e.g. metabolic_renal->creatinine@12h.",
    )
    parser.add_argument(
        "--cross-targets",
        default=None,
        help="Comma-separated targets for cross-system ablation; default evaluates all eligible targets.",
    )
    parser.add_argument(
        "--skip-extended-holdouts",
        action="store_true",
        help=(
            "Skip care-unit and within-stay forward-time gates for a bounded "
            "smoke only. Such a report cannot promote any runtime cell."
        ),
    )
    parser.add_argument(
        "--belief-input-mode",
        choices=("observed", "current_only", "placebo_time_only"),
        default="observed",
        help=(
            "Observed patient history, current-state-only ablation, or a "
            "capacity-matched time-only placebo."
        ),
    )
    parser.add_argument(
        "--belief-placebo-gate",
        action="store_true",
        help=(
            "Train an equal-capacity time-only placebo on every split; neural "
            "patient history is validated only when it beats that placebo."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("whole_body_joint_jepa_gate.json"))
    args = parser.parse_args()
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    history_lags_hours = tuple(
        float(value)
        for value in args.history_lags_hours.split(",")
        if value.strip()
    )
    if args.event_examples is not None:
        frame, variables, module_names = load_joint_event_examples(
            args.event_examples,
            modules,
            horizons,
            args.max_stays,
            min_modules=args.min_modules,
            organ_contract_mode=args.organ_contract_mode,
            history_mode=args.history_mode,
            history_steps=args.history_steps,
            history_lags_hours=history_lags_hours,
        )
    else:
        frame, variables, module_names = load_joint_cohort(
            args.data_root,
            modules,
            horizons,
            args.max_stays,
            min_modules=args.min_modules,
            alignment=args.alignment,
            asof_max_age_hours=args.asof_max_age_hours,
            future_label_tolerance_hours=args.future_label_tolerance_hours,
            asof_age_policy=args.asof_age_policy,
            include_treatment_context=args.include_treatment_context,
        )
    report = run_gate(
        frame,
        variables,
        module_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        measurement_process_mode=args.measurement_process_mode,
        measurement_time_head=args.measurement_time_head,
        measurement_time_weight=args.measurement_time_weight,
        uncertainty_calibration_weight=args.uncertainty_calibration_weight,
        mondrian_conformal=args.mondrian_conformal,
        include_treatment_context=args.include_treatment_context,
        cross_forecast_weight=args.cross_forecast_weight,
        masked_forecast_probability=args.masked_forecast_probability,
        module_mask_probability=args.module_mask_probability,
        nowcast_pretrain_epochs=args.nowcast_pretrain_epochs,
        measurement_consistency_weight=args.measurement_consistency_weight,
        world_model=args.world_model,
        world_model_weight=args.world_model_weight,
        target_momentum=args.target_momentum,
        future_target_mode=args.future_target_mode,
        target_min_observations=args.target_min_observations,
        regime_balanced_loss=args.regime_balanced_loss,
        regime_count=args.regime_count,
        state_normalization=args.state_normalization,
        hospital_invariance_weight=args.hospital_invariance_weight,
        target_encoder_mode=args.target_encoder_mode,
        target_horizon_regime_adapter=args.target_horizon_regime_adapter,
        patient_residual_adapter=args.patient_residual_adapter,
        future_task_balanced_sampling=args.future_task_balanced_sampling,
        coupling_preservation_weight=args.coupling_preservation_weight,
        uncertainty_gate=args.uncertainty_gate,
        validated_edge_adapters=args.validated_edge_adapters,
        validated_edge_weight=args.validated_edge_weight,
        validated_edge_stage_epochs=args.validated_edge_stage_epochs,
        target_head_stage_epochs=args.target_head_stage_epochs,
        validated_edge_targets=(
            tuple(value.strip() for value in args.validated_edge_targets.split(",") if value.strip())
            if args.validated_edge_targets
            else None
        ),
        source_group_adapters=args.source_group_adapters,
        source_group_contextual_regime_gate=args.source_group_contextual_regime_gate,
        source_group_targeted_treatment_context=args.source_group_targeted_treatment_context,
        source_group_adapter_weight=args.source_group_adapter_weight,
        source_group_adapter_stage_epochs=args.source_group_adapter_stage_epochs,
        source_group_adapter_targets=(
            tuple(value.strip() for value in args.source_group_adapter_targets.split(",") if value.strip())
            if args.source_group_adapter_targets
            else None
        ),
        cross_targets=(
            tuple(value.strip() for value in args.cross_targets.split(",") if value.strip())
            if args.cross_targets
            else None
        ),
        extended_holdouts=not args.skip_extended_holdouts,
        belief_input_mode=args.belief_input_mode,
        belief_placebo_gate=args.belief_placebo_gate,
    )
    report["alignment_mode"] = args.alignment
    report["source_contract"] = (
        "canonical_event_ledger"
        if args.event_examples is not None
        else "legacy_module_transition"
    )
    report["event_examples"] = (
        str(args.event_examples.resolve())
        if args.event_examples is not None
        else None
    )
    report["asof_max_age_hours"] = float(args.asof_max_age_hours)
    report["asof_age_policy"] = args.asof_age_policy
    report["future_label_tolerance_hours"] = float(args.future_label_tolerance_hours)
    report["measurement_time_head"] = bool(args.measurement_time_head)
    report["measurement_time_weight"] = float(args.measurement_time_weight)
    report["uncertainty_calibration_weight"] = float(
        args.uncertainty_calibration_weight
    )
    report["mondrian_conformal"] = bool(args.mondrian_conformal)
    report["treatment_context"] = bool(args.include_treatment_context)
    report["future_target_mode"] = args.future_target_mode
    report["target_min_observations"] = int(args.target_min_observations)
    report["state_normalization"] = args.state_normalization
    report["hospital_invariance_weight"] = float(args.hospital_invariance_weight)
    report["target_encoder_mode"] = args.target_encoder_mode
    report["target_horizon_regime_adapter"] = bool(
        args.target_horizon_regime_adapter
    )
    report["patient_residual_adapter"] = bool(args.patient_residual_adapter)
    report["future_task_balanced_sampling"] = bool(
        args.future_task_balanced_sampling
    )
    report["coupling_preservation_weight"] = float(
        args.coupling_preservation_weight
    )
    report["uncertainty_gate"] = bool(args.uncertainty_gate)
    report["validated_edge_adapters"] = bool(args.validated_edge_adapters)
    report["validated_edge_weight"] = float(args.validated_edge_weight)
    report["validated_edge_stage_epochs"] = int(args.validated_edge_stage_epochs)
    report["target_head_stage_epochs"] = int(args.target_head_stage_epochs)
    report["validated_edge_targets"] = (
        [value.strip() for value in args.validated_edge_targets.split(",") if value.strip()]
        if args.validated_edge_targets
        else None
    )
    report["source_group_adapters"] = bool(args.source_group_adapters)
    report["source_group_contextual_regime_gate"] = bool(
        args.source_group_contextual_regime_gate
    )
    report["source_group_targeted_treatment_context"] = bool(
        args.source_group_targeted_treatment_context
    )
    report["source_group_adapter_weight"] = float(args.source_group_adapter_weight)
    report["source_group_adapter_stage_epochs"] = int(args.source_group_adapter_stage_epochs)
    report["source_group_adapter_targets"] = (
        [value.strip() for value in args.source_group_adapter_targets.split(",") if value.strip()]
        if args.source_group_adapter_targets
        else None
    )
    report["cross_targets"] = (
        [value.strip() for value in args.cross_targets.split(",") if value.strip()]
        if args.cross_targets
        else None
    )
    report["extended_holdouts_enabled"] = not args.skip_extended_holdouts
    report["belief_input_mode"] = args.belief_input_mode
    report["belief_placebo_gate_enabled"] = bool(args.belief_placebo_gate)
    report["model_configuration"] = {
        "cohort_signature": cohort_signature(frame, variables),
        "source_contract": report["source_contract"],
        "organ_contract_mode": str(args.organ_contract_mode),
        "history_mode": str(args.history_mode),
        "history_steps": int(args.history_steps),
        "history_lags_hours": list(history_lags_hours),
        "modules": list(module_names),
        "horizons": list(horizons),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "cross_forecast_weight": float(args.cross_forecast_weight),
        "masked_forecast_probability": float(
            args.masked_forecast_probability
        ),
        "module_mask_probability": float(args.module_mask_probability),
        "world_model": bool(args.world_model),
        "world_model_weight": float(args.world_model_weight),
        "target_momentum": float(args.target_momentum),
        "future_target_mode": str(args.future_target_mode),
        "target_min_observations": int(args.target_min_observations),
        "regime_balanced_loss": bool(args.regime_balanced_loss),
        "regime_count": int(args.regime_count),
        "measurement_process_mode": str(args.measurement_process_mode),
        "measurement_time_head": bool(args.measurement_time_head),
        "measurement_time_weight": float(args.measurement_time_weight),
        "uncertainty_calibration_weight": float(
            args.uncertainty_calibration_weight
        ),
        "mondrian_conformal": bool(args.mondrian_conformal),
        "include_treatment_context": bool(args.include_treatment_context),
        "state_normalization": str(args.state_normalization),
        "hospital_invariance_weight": float(args.hospital_invariance_weight),
        "target_encoder_mode": str(args.target_encoder_mode),
        "target_horizon_regime_adapter": bool(args.target_horizon_regime_adapter),
        "patient_residual_adapter": bool(args.patient_residual_adapter),
        "future_task_balanced_sampling": bool(args.future_task_balanced_sampling),
        "coupling_preservation_weight": float(args.coupling_preservation_weight),
        "uncertainty_gate": bool(args.uncertainty_gate),
        "validated_edge_adapters": bool(args.validated_edge_adapters),
        "validated_edge_weight": float(args.validated_edge_weight),
        "validated_edge_stage_epochs": int(args.validated_edge_stage_epochs),
        "target_head_stage_epochs": int(args.target_head_stage_epochs),
        "belief_input_mode": str(args.belief_input_mode),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "promotion_status": report["promotion_status"], "validated_cells": report["validated_target_horizon_count"], "rows": report["rows"], "mean_modules_per_row": report["mean_modules_per_row"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
