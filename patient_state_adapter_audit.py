"""Validate an organ-local patient-state adapter against matched controls.

The experiment is factual and patient-held-out.  A population ridge forecast
is the anchor.  The candidate may add only a bounded residual derived from a
causal neural history state and the renal predict-update belief.  The placebo
has identical architecture and parameter count, but receives patient-mismatched
history and belief tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aki_renal_belief import (
    renal_belief_state_v2_features,
    renal_belief_state_v3_features,
)
from eicu_sepsis_target_router import _bootstrap_ci, split_subjects
from osler_jepa.interval_personalization import (
    mondrian_online_adaptive_interval_widths,
    online_adaptive_interval_widths,
    patient_weighted_quantile,
)
from osler_jepa.patient_state import PatientStateResidualForecaster


SEEDS = (7, 11, 19, 23, 37, 53, 71)
RENAL_VARIABLES = (
    "creatinine",
    "bun",
    "urine_output",
    "potassium",
    "bicarbonate",
    "sodium",
    "map",
)
TREATMENT_COLUMNS = (
    "hist_fluids",
    "hist_vasopressor",
    "hist_diuretics",
    "hist_renal_replacement",
    "hist_nephrotoxin",
)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for variable in RENAL_VARIABLES:
        columns.extend((f"{variable}_t", f"{variable}_age_hr"))
    columns.extend(column for column in TREATMENT_COLUMNS if column in frame)
    return [column for column in columns if column in frame]


def _balanced_rows(
    frame: pd.DataFrame,
    *,
    max_rows_per_subject: int,
    max_subjects: int | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = frame["subject_id"].astype(str).unique()
    if max_subjects is not None and len(subjects) > max_subjects:
        subjects = rng.choice(subjects, size=max_subjects, replace=False)
        frame = frame[frame["subject_id"].astype(str).isin(subjects)]
    selected = []
    for _, rows in frame.groupby("subject_id", sort=False):
        rows = rows.sort_values("t")
        if len(rows) <= max_rows_per_subject:
            selected.extend(rows.index.tolist())
            continue
        positions = np.linspace(0, len(rows) - 1, max_rows_per_subject)
        selected.extend(rows.index[np.round(positions).astype(int)].tolist())
    return frame.loc[selected].sort_values(["stay_id", "t"]).reset_index(drop=True)


def _history_indices(frame: pd.DataFrame, steps: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full((len(frame), steps), -1, dtype=np.int64)
    mask = np.zeros((len(frame), steps), dtype=bool)
    for _, rows in frame.groupby("stay_id", sort=False):
        ordered = rows.sort_values("t").index.to_numpy(dtype=np.int64)
        for position, row_index in enumerate(ordered):
            history = ordered[max(0, position - steps + 1) : position + 1]
            indices[row_index, -len(history) :] = history
            mask[row_index, -len(history) :] = True
    return indices, mask


@dataclass
class PreparedData:
    frame: pd.DataFrame
    current: np.ndarray
    history_indices: np.ndarray
    history_mask: np.ndarray
    beliefs: np.ndarray
    belief_name: str
    feature_columns: list[str]


def prepare_data(
    cohort: Path,
    *,
    history_steps: int,
    max_rows_per_subject: int,
    max_subjects: int | None,
    seed: int,
    renal_belief_version: str,
) -> PreparedData:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    frame = _balanced_rows(
        frame,
        max_rows_per_subject=max_rows_per_subject,
        max_subjects=max_subjects,
        seed=seed,
    )
    belief_functions = {
        "v2": renal_belief_state_v2_features,
        "v3": renal_belief_state_v3_features,
    }
    belief_name = f"renal_belief_state_{renal_belief_version}"
    belief = belief_functions[renal_belief_version](frame).apply(
        pd.to_numeric, errors="coerce"
    )
    columns = _feature_columns(frame)
    current = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    indices, mask = _history_indices(frame, history_steps)
    return PreparedData(
        frame=frame,
        current=current,
        history_indices=indices,
        history_mask=mask,
        beliefs=belief.to_numpy(dtype=np.float32),
        belief_name=belief_name,
        feature_columns=columns,
    )


def _fit_scaler(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values[train], axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25 = np.nanquantile(values[train], 0.25, axis=0)
    q75 = np.nanquantile(values[train], 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _model_arrays(
    data: PreparedData,
    rows: np.ndarray,
    current_center: np.ndarray,
    current_scale: np.ndarray,
    belief_center: np.ndarray,
    belief_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = np.nan_to_num(
        (data.current[rows] - current_center) / current_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    history_rows = data.history_indices[rows]
    safe_indices = np.maximum(history_rows, 0)
    history_values = data.current[safe_indices]
    history_values = np.nan_to_num(
        (history_values - current_center) / current_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    anchor_times = pd.to_datetime(data.frame.loc[rows, "t"]).to_numpy()
    event_times = pd.to_datetime(data.frame.loc[safe_indices.reshape(-1), "t"]).to_numpy()
    event_times = event_times.reshape(safe_indices.shape)
    elapsed = (
        (event_times - anchor_times[:, None]) / np.timedelta64(1, "h")
    ).astype(np.float32)
    elapsed = np.clip(elapsed / 48.0, -1.0, 0.0)
    history = np.concatenate((history_values, elapsed[..., None]), axis=-1)
    history_mask = data.history_mask[rows]
    history[~history_mask] = 0.0
    belief = np.nan_to_num(
        (data.beliefs[rows] - belief_center) / belief_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    return current, history, history_mask, belief


def _mismatched_patient_rows(frame: pd.DataFrame, rows: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subjects = frame.loc[rows, "subject_id"].astype(str).to_numpy()
    order = rng.permutation(len(rows))
    for _ in range(8):
        collisions = subjects == subjects[order]
        if not collisions.any():
            break
        order[collisions] = rng.permutation(order[collisions])
    return rows[order]


def _train_model(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchor: np.ndarray,
    current_target: np.ndarray,
    truth: np.ndarray,
    *,
    seed: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> PatientStateResidualForecaster:
    torch.manual_seed(seed)
    current, history, history_mask, belief = arrays
    model = PatientStateResidualForecaster(
        current.shape[1], belief.shape[1], hidden
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(current),
        torch.from_numpy(history),
        torch.from_numpy(history_mask),
        torch.from_numpy(belief),
        torch.from_numpy(anchor.astype(np.float32)),
        torch.from_numpy(current_target.astype(np.float32)),
        torch.from_numpy(truth.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            current_b, history_b, mask_b, belief_b, anchor_b, target_b, truth_b = [
                value.to(device) for value in batch
            ]
            prediction, scale = model(
                current_b, history_b, mask_b, belief_b, anchor_b, target_b
            )
            error = prediction - truth_b
            point_loss = nn.functional.smooth_l1_loss(prediction, truth_b)
            scale_loss = (error.detach().abs() / scale + torch.log(scale)).mean()
            loss = point_loss + 0.10 * scale_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


@torch.no_grad()
def _predict(
    model: PatientStateResidualForecaster,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchor: np.ndarray,
    current_target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    current, history, history_mask, belief = arrays
    dataset = TensorDataset(
        torch.from_numpy(current),
        torch.from_numpy(history),
        torch.from_numpy(history_mask),
        torch.from_numpy(belief),
        torch.from_numpy(anchor.astype(np.float32)),
        torch.from_numpy(current_target.astype(np.float32)),
    )
    predictions, scales = [], []
    model.eval()
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        current_b, history_b, mask_b, belief_b, anchor_b, target_b = [
            value.to(device) for value in batch
        ]
        point, scale = model(current_b, history_b, mask_b, belief_b, anchor_b, target_b)
        predictions.append(point.cpu().numpy())
        scales.append(scale.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(scales)


def _subject_delta(
    subjects: np.ndarray,
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    temp = pd.DataFrame(
        {
            "subject": subjects.astype(str),
            "delta": np.abs(candidate - truth) - np.abs(reference - truth),
        }
    )
    return temp.groupby("subject", sort=False)["delta"].mean().to_numpy()


def _comparison(subjects, truth, candidate, reference, seed, bootstrap_samples):
    delta = _subject_delta(subjects, truth, candidate, reference)
    ci = _bootstrap_ci(delta, seed=seed, samples=bootstrap_samples)
    return {
        "patient_equalized_delta_mae": round(float(delta.mean()), 6),
        "patient_bootstrap_95_ci": ci,
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _patient_equalized_mae(
    subjects: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    temp = pd.DataFrame(
        {
            "subject": np.asarray(subjects, dtype=str),
            "error": np.abs(np.asarray(prediction) - np.asarray(truth)),
        }
    )
    return float(temp.groupby("subject", sort=False)["error"].mean().mean())


def _select_residual_gain(
    subjects: np.ndarray,
    truth: np.ndarray,
    anchor: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Select bounded residual shrinkage using calibration patients only."""

    gains = np.linspace(0.0, 1.0, 9)
    scored = []
    residual = np.asarray(prediction) - np.asarray(anchor)
    for gain in gains:
        blended = np.asarray(anchor) + float(gain) * residual
        scored.append(
            (_patient_equalized_mae(subjects, truth, blended), -float(gain), float(gain))
        )
    return min(scored)[2]


def _select_nested_nominal_coverage(
    scores: np.ndarray,
    subjects: np.ndarray,
    anchor_times: np.ndarray,
    future_times: np.ndarray,
    scales: np.ndarray,
    points: np.ndarray,
    truth: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    """Select nominal coverage on an inner patient-heldout calibration split."""

    unique = np.unique(np.asarray(subjects, dtype=str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(1, int(round(0.67 * len(unique))))
    fit_subjects = set(unique[:cut])
    fit = np.asarray([subject in fit_subjects for subject in subjects], dtype=bool)
    heldout = ~fit
    if fit.sum() < 30 or heldout.sum() < 30:
        return 0.90, float("nan")
    candidates = np.linspace(0.86, 0.94, 17)
    choices = []
    for nominal in candidates:
        result = online_adaptive_interval_widths(
            np.asarray(scores)[fit],
            np.asarray(subjects)[fit],
            np.asarray(subjects)[heldout],
            np.asarray(anchor_times)[heldout],
            np.asarray(future_times)[heldout],
            np.asarray(scales)[heldout],
            np.asarray(points)[heldout],
            np.asarray(truth)[heldout],
            coverage=float(nominal),
        )
        observed_coverage = float(
            np.mean(
                np.abs(np.asarray(points)[heldout] - np.asarray(truth)[heldout])
                <= result.widths
            )
        )
        choices.append(
            (abs(observed_coverage - 0.90), abs(float(nominal) - 0.90), float(nominal), observed_coverage)
        )
    _, _, nominal, observed = min(choices)
    return nominal, observed


def _fit_population_residual_scale(
    current_fit: np.ndarray,
    truth_fit: np.ndarray,
    prediction_fit: np.ndarray,
):
    """Fit a calibration-safe heteroscedastic scale from fit patients only."""

    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=20.0),
    )
    log_error = np.log1p(np.abs(np.asarray(prediction_fit) - np.asarray(truth_fit)))
    model.fit(current_fit, log_error)
    return model


def _predict_population_residual_scale(model, current: np.ndarray) -> np.ndarray:
    log_scale = np.clip(model.predict(current), -6.0, 12.0)
    return np.maximum(np.expm1(log_scale), 1e-3).astype(np.float64)


def _temporal_regimes(hours_since_onset) -> np.ndarray:
    hours = np.asarray(hours_since_onset, dtype=np.float64)
    return np.where(hours < 24.0, "early_0_24h", np.where(hours < 72.0, "middle_24_72h", "late_72h_plus"))


def _width_comparison(
    subjects: np.ndarray,
    candidate_width: np.ndarray,
    population_width: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    temp = pd.DataFrame(
        {
            "subject": subjects.astype(str),
            "delta": candidate_width - population_width,
        }
    )
    patient_delta = temp.groupby("subject", sort=False)["delta"].mean().to_numpy()
    ci = _bootstrap_ci(patient_delta, seed=seed, samples=bootstrap_samples)
    return {
        "candidate_mean_half_width": round(float(np.mean(candidate_width)), 6),
        "population_mean_half_width": round(float(np.mean(population_width)), 6),
        "candidate_median_half_width": round(float(np.median(candidate_width)), 6),
        "population_median_half_width": round(float(np.median(population_width)), 6),
        "patient_equalized_delta_half_width": round(float(patient_delta.mean()), 6),
        "patient_bootstrap_95_ci": ci,
        "candidate_significantly_narrower": bool(ci is not None and ci[1] < 0.0),
    }


def _interval_narrowing_claim_allowed(
    width_comparison: dict[str, object],
    *,
    candidate_coverage: float,
    population_coverage: float,
) -> bool:
    return bool(
        0.87 <= candidate_coverage <= 0.93
        and 0.87 <= population_coverage <= 0.93
        and width_comparison["candidate_significantly_narrower"]
    )


def _split_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
    group_column: str,
    late_stage_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    discovery, heldout = split_subjects(
        frame,
        seed=seed,
        discovery_fraction=0.67,
        group_column=group_column,
    )
    train = np.flatnonzero(frame[group_column].isin(discovery).to_numpy())
    test_mask = frame[group_column].isin(heldout).to_numpy()
    if late_stage_only:
        threshold = float(frame.loc[test_mask, "hours_since_onset"].quantile(0.67))
        test_mask &= frame["hours_since_onset"].to_numpy(dtype=float) >= threshold
    return train, np.flatnonzero(test_mask)


def _run_split(
    data: PreparedData,
    *,
    target: str,
    horizon_hours: int,
    seed: int,
    group_column: str,
    late_stage_only: bool,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    bootstrap_samples: int,
    device: torch.device,
    calibrate_residual_gain: bool,
    calibrate_candidate_nominal_coverage: bool,
    population_residual_scale: bool,
    temporal_mondrian_conformal: bool,
) -> dict[str, object]:
    frame = data.frame
    future_column = f"{target}_tp{horizon_hours}"
    current_column = f"{target}_t"
    eligible = frame[current_column].notna() & frame[future_column].notna()
    eligible_rows = np.flatnonzero(eligible.to_numpy())
    eligible_frame = frame.loc[eligible_rows].reset_index(drop=True)
    local_train, local_test = _split_rows(
        eligible_frame,
        seed=seed,
        group_column=group_column,
        late_stage_only=late_stage_only,
    )
    train = eligible_rows[local_train]
    test = eligible_rows[local_test]
    train_subjects = frame.loc[train, "subject_id"].astype(str).unique()
    rng = np.random.default_rng(seed + 101)
    rng.shuffle(train_subjects)
    calibration_count = max(1, int(round(0.20 * len(train_subjects))))
    calibration_subjects = set(train_subjects[:calibration_count])
    calibration_mask = frame.loc[train, "subject_id"].astype(str).isin(calibration_subjects)
    calibration = train[calibration_mask.to_numpy()]
    fit = train[~calibration_mask.to_numpy()]

    current_center, current_scale = _fit_scaler(data.current, fit)
    belief_center, belief_scale = _fit_scaler(data.beliefs, fit)
    baseline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    y_fit_physical = frame.loc[fit, future_column].to_numpy(dtype=np.float64)
    baseline.fit(data.current[fit], y_fit_physical)
    baseline_fit = baseline.predict(data.current[fit])
    baseline_calibration = baseline.predict(data.current[calibration])
    baseline_test = baseline.predict(data.current[test])
    if population_residual_scale:
        population_scale_model = _fit_population_residual_scale(
            data.current[fit], y_fit_physical, baseline_fit
        )
        population_scale_calibration = _predict_population_residual_scale(
            population_scale_model, data.current[calibration]
        )
        population_scale_test = _predict_population_residual_scale(
            population_scale_model, data.current[test]
        )
    else:
        population_scale_calibration = np.ones(len(calibration), dtype=np.float64)
        population_scale_test = np.ones(len(test), dtype=np.float64)

    target_center = float(np.nanmedian(y_fit_physical))
    target_scale = float(np.nanquantile(y_fit_physical, 0.75) - np.nanquantile(y_fit_physical, 0.25)) / 1.349
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0
    current_target_fit = frame.loc[fit, current_column].to_numpy(dtype=np.float32)
    current_target_calibration = frame.loc[calibration, current_column].to_numpy(dtype=np.float32)
    current_target_test = frame.loc[test, current_column].to_numpy(dtype=np.float32)
    normalized_truth = ((y_fit_physical - target_center) / target_scale).astype(np.float32)
    normalized_anchor = ((baseline_fit - target_center) / target_scale).astype(np.float32)
    normalized_current = ((current_target_fit - target_center) / target_scale).astype(np.float32)

    fit_arrays = _model_arrays(
        data, fit, current_center, current_scale, belief_center, belief_scale
    )
    calibration_arrays = _model_arrays(
        data, calibration, current_center, current_scale, belief_center, belief_scale
    )
    test_arrays = _model_arrays(
        data, test, current_center, current_scale, belief_center, belief_scale
    )
    candidate = _train_model(
        fit_arrays,
        normalized_anchor,
        normalized_current,
        normalized_truth,
        seed=seed,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    mismatched_fit = _mismatched_patient_rows(frame, fit, seed + 201)
    placebo_fit_arrays = _model_arrays(
        data, mismatched_fit, current_center, current_scale, belief_center, belief_scale
    )
    placebo_fit_arrays = (fit_arrays[0],) + placebo_fit_arrays[1:]
    placebo = _train_model(
        placebo_fit_arrays,
        normalized_anchor,
        normalized_current,
        normalized_truth,
        seed=seed,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    calibration_anchor = ((baseline_calibration - target_center) / target_scale).astype(np.float32)
    test_anchor = ((baseline_test - target_center) / target_scale).astype(np.float32)
    calibration_current = ((current_target_calibration - target_center) / target_scale).astype(np.float32)
    test_current = ((current_target_test - target_center) / target_scale).astype(np.float32)
    candidate_calibration, scale_calibration = _predict(
        candidate,
        calibration_arrays,
        calibration_anchor,
        calibration_current,
        device,
        batch_size,
    )
    candidate_test, scale_test = _predict(
        candidate, test_arrays, test_anchor, test_current, device, batch_size
    )

    mismatched_test = _mismatched_patient_rows(frame, test, seed + 303)
    mismatched_calibration = _mismatched_patient_rows(
        frame, calibration, seed + 302
    )
    placebo_calibration_arrays_raw = _model_arrays(
        data,
        mismatched_calibration,
        current_center,
        current_scale,
        belief_center,
        belief_scale,
    )
    placebo_calibration_arrays = (
        calibration_arrays[0],
    ) + placebo_calibration_arrays_raw[1:]
    placebo_calibration, _ = _predict(
        placebo,
        placebo_calibration_arrays,
        calibration_anchor,
        calibration_current,
        device,
        batch_size,
    )
    placebo_test_arrays_raw = _model_arrays(
        data, mismatched_test, current_center, current_scale, belief_center, belief_scale
    )
    placebo_test_arrays = (test_arrays[0],) + placebo_test_arrays_raw[1:]
    placebo_test, _ = _predict(
        placebo, placebo_test_arrays, test_anchor, test_current, device, batch_size
    )

    candidate_calibration = candidate_calibration * target_scale + target_center
    candidate_test = candidate_test * target_scale + target_center
    placebo_calibration = placebo_calibration * target_scale + target_center
    placebo_test = placebo_test * target_scale + target_center
    scale_calibration = scale_calibration * target_scale
    scale_test = scale_test * target_scale
    truth_calibration = frame.loc[calibration, future_column].to_numpy(dtype=np.float64)
    truth_test = frame.loc[test, future_column].to_numpy(dtype=np.float64)
    subjects_calibration = frame.loc[calibration, "subject_id"].astype(str).to_numpy()
    subjects_test = frame.loc[test, "subject_id"].astype(str).to_numpy()
    candidate_gain = 1.0
    placebo_gain = 1.0
    if calibrate_residual_gain:
        candidate_gain = _select_residual_gain(
            subjects_calibration,
            truth_calibration,
            baseline_calibration,
            candidate_calibration,
        )
        placebo_gain = _select_residual_gain(
            subjects_calibration,
            truth_calibration,
            baseline_calibration,
            placebo_calibration,
        )
        candidate_calibration = baseline_calibration + candidate_gain * (
            candidate_calibration - baseline_calibration
        )
        candidate_test = baseline_test + candidate_gain * (
            candidate_test - baseline_test
        )
        placebo_test = baseline_test + placebo_gain * (
            placebo_test - baseline_test
        )
    calibration_scores = np.abs(candidate_calibration - truth_calibration) / np.maximum(
        scale_calibration, 1e-6
    )
    candidate_nominal = 0.90
    candidate_inner_coverage = None
    if calibrate_candidate_nominal_coverage:
        candidate_nominal, candidate_inner_coverage = _select_nested_nominal_coverage(
            calibration_scores,
            subjects_calibration,
            frame.loc[calibration, "t"].to_numpy(),
            frame.loc[calibration, "t_plus"].to_numpy(),
            np.maximum(scale_calibration, 1e-6),
            candidate_calibration,
            truth_calibration,
            seed=seed + 701,
        )
    if temporal_mondrian_conformal:
        calibration_regimes = _temporal_regimes(
            frame.loc[calibration, "hours_since_onset"].to_numpy()
        )
        test_regimes = _temporal_regimes(
            frame.loc[test, "hours_since_onset"].to_numpy()
        )
        online = mondrian_online_adaptive_interval_widths(
            calibration_scores,
            subjects_calibration,
            calibration_regimes,
            subjects_test,
            test_regimes,
            frame.loc[test, "t"],
            frame.loc[test, "t_plus"],
            np.maximum(scale_test, 1e-6),
            candidate_test,
            truth_test,
            coverage=candidate_nominal,
        )
    else:
        online = online_adaptive_interval_widths(
            calibration_scores,
            subjects_calibration,
            subjects_test,
            frame.loc[test, "t"],
            frame.loc[test, "t_plus"],
            np.maximum(scale_test, 1e-6),
            candidate_test,
            truth_test,
            coverage=candidate_nominal,
        )
    population_calibration_scores = np.abs(
        baseline_calibration - truth_calibration
    ) / population_scale_calibration
    population_nominal = 0.90
    if temporal_mondrian_conformal:
        population_online = mondrian_online_adaptive_interval_widths(
            population_calibration_scores,
            subjects_calibration,
            calibration_regimes,
            subjects_test,
            test_regimes,
            frame.loc[test, "t"],
            frame.loc[test, "t_plus"],
            population_scale_test,
            baseline_test,
            truth_test,
            coverage=population_nominal,
        )
    else:
        population_online = online_adaptive_interval_widths(
            population_calibration_scores,
            subjects_calibration,
            subjects_test,
            frame.loc[test, "t"],
            frame.loc[test, "t_plus"],
            population_scale_test,
            baseline_test,
            truth_test,
            coverage=population_nominal,
        )
    covered = np.abs(candidate_test - truth_test) <= online.widths
    population_covered = (
        np.abs(baseline_test - truth_test) <= population_online.widths
    )
    personalized = online.personalized
    width_comparison = _width_comparison(
        subjects_test,
        online.widths,
        population_online.widths,
        seed=seed + 607,
        bootstrap_samples=bootstrap_samples,
    )
    candidate_coverage_passes = bool(0.87 <= covered.mean() <= 0.93)
    population_coverage_passes = bool(
        0.87 <= population_covered.mean() <= 0.93
    )
    candidate_vs_baseline = _comparison(
        subjects_test,
        truth_test,
        candidate_test,
        baseline_test,
        seed + 401,
        bootstrap_samples,
    )
    candidate_vs_placebo = _comparison(
        subjects_test,
        truth_test,
        candidate_test,
        placebo_test,
        seed + 503,
        bootstrap_samples,
    )
    return {
        "seed": int(seed),
        "group_column": group_column,
        "late_stage_only": bool(late_stage_only),
        "rows": int(len(test)),
        "subjects": int(len(np.unique(subjects_test))),
        "parameter_count_candidate": int(sum(p.numel() for p in candidate.parameters())),
        "parameter_count_placebo": int(sum(p.numel() for p in placebo.parameters())),
        "mae": {
            "population_baseline": round(float(np.mean(np.abs(baseline_test - truth_test))), 6),
            "patient_state_candidate": round(float(np.mean(np.abs(candidate_test - truth_test))), 6),
            "capacity_matched_placebo": round(float(np.mean(np.abs(placebo_test - truth_test))), 6),
        },
        "candidate_vs_population": candidate_vs_baseline,
        "candidate_vs_placebo": candidate_vs_placebo,
        "adaptive_conformal": {
            "coverage": round(float(covered.mean()), 6),
            "mean_half_width": round(float(np.mean(online.widths)), 6),
            "personalized_rows": int(personalized.sum()),
            "personalized_rate": round(float(personalized.mean()), 6),
            "personalized_coverage": (
                round(float(covered[personalized].mean()), 6)
                if personalized.any()
                else None
            ),
            "passes_0_87_to_0_93": candidate_coverage_passes,
        },
        "matched_population_conformal": {
            "coverage": round(float(population_covered.mean()), 6),
            "mean_half_width": round(float(np.mean(population_online.widths)), 6),
            "median_half_width": round(float(np.median(population_online.widths)), 6),
            "personalized_rows": int(population_online.personalized.sum()),
            "passes_0_87_to_0_93": population_coverage_passes,
        },
        "matched_interval_width": {
            **width_comparison,
            "both_coverage_gates_pass": bool(
                candidate_coverage_passes and population_coverage_passes
            ),
            "narrower_interval_claim_allowed": bool(
                _interval_narrowing_claim_allowed(
                    width_comparison,
                    candidate_coverage=float(covered.mean()),
                    population_coverage=float(population_covered.mean()),
                )
            ),
        },
        "development_selected_policy": {
            "candidate_residual_gain": round(float(candidate_gain), 6),
            "placebo_residual_gain": round(float(placebo_gain), 6),
            "candidate_nominal_coverage": round(float(candidate_nominal), 6),
            "population_nominal_coverage": round(float(population_nominal), 6),
            "population_residual_scale": bool(population_residual_scale),
            "temporal_mondrian_conformal": bool(temporal_mondrian_conformal),
            "candidate_inner_coverage": (
                round(float(candidate_inner_coverage), 6)
                if candidate_inner_coverage is not None
                and np.isfinite(candidate_inner_coverage)
                else None
            ),
        },
        "passes_point_gate": bool(
            candidate_vs_baseline["significant"]
            and candidate_vs_placebo["significant"]
        ),
    }


def _summary(runs: list[dict[str, object]], expected: int) -> dict[str, object]:
    return {
        "runs": len(runs),
        "point_gate_passes": int(sum(run["passes_point_gate"] for run in runs)),
        "conformal_gate_passes": int(
            sum(run["adaptive_conformal"]["passes_0_87_to_0_93"] for run in runs)
        ),
        "all_point_gates_pass": bool(
            len(runs) == expected and all(run["passes_point_gate"] for run in runs)
        ),
        "all_conformal_gates_pass": bool(
            len(runs) == expected
            and all(run["adaptive_conformal"]["passes_0_87_to_0_93"] for run in runs)
        ),
        "matched_interval_narrowing_passes": int(
            sum(
                run["matched_interval_width"]["narrower_interval_claim_allowed"]
                for run in runs
            )
        ),
        "all_matched_interval_narrowing_gates_pass": bool(
            len(runs) == expected
            and all(
                run["matched_interval_width"]["narrower_interval_claim_allowed"]
                for run in runs
            )
        ),
        "median_patient_equalized_delta_half_width": round(
            float(
                np.median(
                    [
                        run["matched_interval_width"][
                            "patient_equalized_delta_half_width"
                        ]
                        for run in runs
                    ]
                )
            ),
            6,
        ),
        "median_delta_vs_population": round(
            float(np.median([run["candidate_vs_population"]["patient_equalized_delta_mae"] for run in runs])),
            6,
        ),
        "median_delta_vs_placebo": round(
            float(np.median([run["candidate_vs_placebo"]["patient_equalized_delta_mae"] for run in runs])),
            6,
        ),
    }


def _fit_final_artifact(
    data: PreparedData,
    *,
    target: str,
    horizon_hours: int,
    artifact_dir: Path,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    validation_report: Path,
    calibrate_candidate_nominal_coverage: bool,
    temporal_mondrian_conformal: bool,
) -> dict[str, object]:
    frame = data.frame
    future_column = f"{target}_tp{horizon_hours}"
    current_column = f"{target}_t"
    eligible = np.flatnonzero(
        (frame[current_column].notna() & frame[future_column].notna()).to_numpy()
    )
    subjects = frame.loc[eligible, "subject_id"].astype(str).unique()
    rng = np.random.default_rng(1701)
    rng.shuffle(subjects)
    calibration_count = max(1, int(round(0.15 * len(subjects))))
    calibration_subjects = set(subjects[:calibration_count])
    calibration_mask = frame.loc[eligible, "subject_id"].astype(str).isin(
        calibration_subjects
    )
    calibration = eligible[calibration_mask.to_numpy()]
    fit = eligible[~calibration_mask.to_numpy()]

    current_center, current_scale = _fit_scaler(data.current, fit)
    belief_center, belief_scale = _fit_scaler(data.beliefs, fit)
    baseline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    truth_fit = frame.loc[fit, future_column].to_numpy(dtype=np.float64)
    baseline.fit(data.current[fit], truth_fit)
    baseline_fit = baseline.predict(data.current[fit])
    baseline_calibration = baseline.predict(data.current[calibration])
    target_center = float(np.nanmedian(truth_fit))
    target_scale = float(
        np.nanquantile(truth_fit, 0.75) - np.nanquantile(truth_fit, 0.25)
    ) / 1.349
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0
    fit_arrays = _model_arrays(
        data, fit, current_center, current_scale, belief_center, belief_scale
    )
    model = _train_model(
        fit_arrays,
        ((baseline_fit - target_center) / target_scale).astype(np.float32),
        (
            (frame.loc[fit, current_column].to_numpy(dtype=np.float32) - target_center)
            / target_scale
        ).astype(np.float32),
        ((truth_fit - target_center) / target_scale).astype(np.float32),
        seed=1701,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    calibration_arrays = _model_arrays(
        data, calibration, current_center, current_scale, belief_center, belief_scale
    )
    calibration_prediction, calibration_scale = _predict(
        model,
        calibration_arrays,
        ((baseline_calibration - target_center) / target_scale).astype(np.float32),
        (
            (
                frame.loc[calibration, current_column].to_numpy(dtype=np.float32)
                - target_center
            )
            / target_scale
        ).astype(np.float32),
        device,
        batch_size,
    )
    calibration_prediction = calibration_prediction * target_scale + target_center
    calibration_scale = calibration_scale * target_scale
    calibration_truth = frame.loc[calibration, future_column].to_numpy(dtype=np.float64)
    scores = np.abs(calibration_prediction - calibration_truth) / np.maximum(
        calibration_scale, 1e-6
    )
    calibration_ids = frame.loc[calibration, "subject_id"].astype(str).to_numpy()
    conformal_target_coverage = 0.90
    if calibrate_candidate_nominal_coverage:
        conformal_target_coverage, _ = _select_nested_nominal_coverage(
            scores,
            calibration_ids,
            frame.loc[calibration, "t"].to_numpy(),
            frame.loc[calibration, "t_plus"].to_numpy(),
            np.maximum(calibration_scale, 1e-6),
            calibration_prediction,
            calibration_truth,
            seed=2402,
        )
    alpha_grid = np.linspace(0.02, 0.25, 24)
    quantiles = np.asarray(
        [
            patient_weighted_quantile(scores, calibration_ids, coverage=1.0 - alpha)
            for alpha in alpha_grid
        ],
        dtype=np.float64,
    )
    conformal_arrays: dict[str, np.ndarray] = {}
    if temporal_mondrian_conformal:
        calibration_regimes = _temporal_regimes(
            frame.loc[calibration, "hours_since_onset"].to_numpy()
        )
        regime_names = np.asarray(
            ("early_0_24h", "middle_24_72h", "late_72h_plus")
        )
        regime_quantiles = []
        for regime in regime_names:
            selected = calibration_regimes == regime
            if int(selected.sum()) < 30:
                regime_quantiles.append(quantiles.copy())
                continue
            regime_quantiles.append(
                np.asarray(
                    [
                        patient_weighted_quantile(
                            scores[selected],
                            calibration_ids[selected],
                            coverage=1.0 - alpha,
                        )
                        for alpha in alpha_grid
                    ],
                    dtype=np.float64,
                )
            )
        conformal_arrays = {
            "conformal_regime_names": regime_names,
            "conformal_quantile_by_regime": np.stack(regime_quantiles),
        }

    artifact_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "feature_dim": int(data.current.shape[1]),
                "belief_dim": int(data.beliefs.shape[1]),
                "hidden": int(hidden),
            },
        },
        artifact_dir / "model.pt",
    )
    joblib.dump(baseline, artifact_dir / "population_anchor.joblib")
    np.savez_compressed(
        artifact_dir / "preprocessing_and_conformal.npz",
        current_center=current_center,
        current_scale=current_scale,
        belief_center=belief_center,
        belief_scale=belief_scale,
        target_center=np.asarray([target_center]),
        target_scale=np.asarray([target_scale]),
        conformal_alpha=alpha_grid,
        conformal_quantile=quantiles,
        **conformal_arrays,
    )
    metadata = {
        "schema": "organ_patient_state_artifact.v1",
        "target": target,
        "horizon_hours": int(horizon_hours),
        "organ": "kidneys",
        "feature_columns": data.feature_columns,
        "belief": data.belief_name,
        "conformal_target_coverage": float(conformal_target_coverage),
        "conformal_policy": (
            "temporal_mondrian" if temporal_mondrian_conformal else "global"
        ),
        "temporal_regime_cutpoints_hours": (
            [24.0, 72.0] if temporal_mondrian_conformal else None
        ),
        "history_steps": int(data.history_indices.shape[1]),
        "fit_rows": int(len(fit)),
        "fit_subjects": int(frame.loc[fit, "subject_id"].nunique()),
        "calibration_rows": int(len(calibration)),
        "calibration_subjects": int(frame.loc[calibration, "subject_id"].nunique()),
        "validation_report": validation_report.name,
        "factual_only": True,
        "causal_claim_allowed": False,
        "direct_hidden_state_truth_claim_allowed": False,
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    artifact_files = (
        "metadata.json",
        "model.pt",
        "population_anchor.joblib",
        "preprocessing_and_conformal.npz",
    )
    manifest = "\n".join(
        f"{hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()}  {name}"
        for name in artifact_files
    )
    (artifact_dir / "MANIFEST.sha256").write_text(
        manifest + "\n", encoding="ascii"
    )
    return {
        "directory": str(artifact_dir.resolve()),
        "schema": metadata["schema"],
        "model": "model.pt",
        "population_anchor": "population_anchor.joblib",
        "preprocessing_and_conformal": "preprocessing_and_conformal.npz",
        "metadata": "metadata.json",
        "manifest": "MANIFEST.sha256",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("eicu_aki_transitions_24h.parquet"))
    parser.add_argument("--target", choices=RENAL_VARIABLES, default="creatinine")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        choices=(1, 3, 6, 12, 24, 48),
        default=24,
    )
    parser.add_argument("--output", type=Path, default=Path("patient_state_adapter_audit.json"))
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--max-rows-per-subject", type=int, default=8)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--calibrate-residual-gain", action="store_true")
    parser.add_argument(
        "--calibrate-candidate-nominal-coverage", action="store_true"
    )
    parser.add_argument("--population-residual-scale", action="store_true")
    parser.add_argument("--temporal-mondrian-conformal", action="store_true")
    parser.add_argument(
        "--renal-belief-version", choices=("v2", "v3"), default="v2"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    device = _device(args.device)
    data = prepare_data(
        args.cohort,
        history_steps=args.history_steps,
        max_rows_per_subject=args.max_rows_per_subject,
        max_subjects=args.max_subjects,
        seed=17,
        renal_belief_version=args.renal_belief_version,
    )
    common = {
        "data": data,
        "target": args.target,
        "horizon_hours": args.horizon_hours,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "bootstrap_samples": args.bootstrap_samples,
        "device": device,
        "calibrate_residual_gain": args.calibrate_residual_gain,
        "calibrate_candidate_nominal_coverage": args.calibrate_candidate_nominal_coverage,
        "population_residual_scale": args.population_residual_scale,
        "temporal_mondrian_conformal": args.temporal_mondrian_conformal,
    }
    patient_runs = [
        _run_split(seed=seed, group_column="subject_id", late_stage_only=False, **common)
        for seed in seeds
    ]
    external_runs = []
    if not args.skip_external:
        external_runs.extend(
            [
                _run_split(seed=9001, group_column="hospitalid", late_stage_only=False, **common),
                _run_split(seed=9002, group_column="unittype", late_stage_only=False, **common),
                _run_split(seed=9003, group_column="subject_id", late_stage_only=True, **common),
            ]
        )
    patient_summary = _summary(patient_runs, len(seeds))
    external_summary = _summary(external_runs, 3) if external_runs else None
    promoted = bool(
        patient_summary["all_point_gates_pass"]
        and patient_summary["all_conformal_gates_pass"]
        and external_summary is not None
        and external_summary["all_point_gates_pass"]
        and external_summary["all_conformal_gates_pass"]
    )
    interval_narrowing_claim_allowed = bool(
        patient_summary["all_matched_interval_narrowing_gates_pass"]
        and external_summary is not None
        and external_summary["all_matched_interval_narrowing_gates_pass"]
    )
    precision_promoted = bool(promoted and interval_narrowing_claim_allowed)
    report = {
        "schema": "organ_patient_state_adapter_audit.v1",
        "target": args.target,
        "horizon_hours": int(args.horizon_hours),
        "cohort": str(args.cohort.resolve()),
        "cohort_summary": {
            "rows": int(len(data.frame)),
            "subjects": int(data.frame["subject_id"].nunique()),
            "hospitals": int(data.frame["hospitalid"].nunique()),
            "care_units": int(data.frame["unittype"].nunique()),
        },
        "architecture": {
            "population_anchor": "ridge_current_state",
            "patient_history": "causal_neural_predict_update_filter",
            "mechanistic_belief": data.belief_name,
            "adapter_placement": "kidney_organ_token_only",
            "residual_bounded": True,
            "capacity_matched_placebo": "same_model_with_patient_mismatched_history_and_belief",
            "parameter_counts_equal": all(
                run["parameter_count_candidate"] == run["parameter_count_placebo"]
                for run in patient_runs + external_runs
            ),
        },
        "patient_heldout": {
            "summary": patient_summary,
            "runs": patient_runs,
        },
        "external_heldout": {
            "summary": external_summary,
            "runs": external_runs,
        },
        "promotion": {
            "status": "promoted" if promoted else "candidate_only",
            "patient_state_adapter_allowed": promoted,
            "precision_status": (
                "promoted" if precision_promoted else "candidate_only"
            ),
            "narrower_patient_interval_allowed": precision_promoted,
            "requires_point_and_conformal_pass_on_all_patient_and_external_gates": True,
        },
        "interval_narrowing": {
            "claim_allowed": interval_narrowing_claim_allowed,
            "requires_equal_coverage_gate": True,
            "requires_patient_level_paired_bootstrap": True,
        },
        "boundary": {
            "factual_only": True,
            "direct_hidden_state_truth_claim_allowed": False,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "raw_rows_or_patient_ids_included": False,
        },
    }
    if precision_promoted and args.artifact_dir is not None:
        report["artifact"] = _fit_final_artifact(
            data,
            target=args.target,
            horizon_hours=args.horizon_hours,
            artifact_dir=args.artifact_dir,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            validation_report=args.output,
            calibrate_candidate_nominal_coverage=(
                args.calibrate_candidate_nominal_coverage
            ),
            temporal_mondrian_conformal=args.temporal_mondrian_conformal,
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
