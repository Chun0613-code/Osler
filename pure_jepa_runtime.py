"""Fail-closed runtime for saved pure-JEPA body-system checkpoints.

The runtime deliberately separates two decisions:

* the neural model may produce a factual candidate for every target; and
* the promotion registry decides whether that candidate is allowed to move
  away from the latest observed value.

Unvalidated targets therefore return persistence, even when a checkpoint is
available.  This module does not infer treatment effects and does not expose
clinical or causal authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import StateScaler
from pure_jepa_body_system import PureJEPABodyModel
from pure_jepa_gate_registry import target_status
from sparse_aware_pure_jepa import SparseAwarePureJEPABodyModel


HISTORY_LENGTH = 5
NONNEGATIVE_TARGETS = {
    "albumin", "anion_gap", "amylase", "bilirubin", "bilirubin_direct", "bnp",
    "bun", "calcium", "chloride", "ck_mb", "cortisol", "creatinine", "cpk",
    "crp", "ferritin", "fibrinogen", "free_t4", "glucose", "hematocrit",
    "hemoglobin", "inr", "ionized_calcium", "lactate", "ldh", "lipase",
    "magnesium", "phosphate", "platelets", "potassium", "ptt", "serum_ketones",
    "serum_osmolality", "sodium", "total_protein", "troponin_i", "troponin_t",
    "tsh", "triglycerides", "urine_output", "wbc",
}


def _scaler_from_dict(payload: dict[str, Any]) -> StateScaler:
    return StateScaler(
        tuple(payload["variables"]),
        np.asarray(payload["medians"], dtype=np.float32),
        np.asarray(payload["scales"], dtype=np.float32),
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[
    PureJEPABodyModel, StateScaler, dict[str, Any]
]:
    """Load a pure-JEPA body checkpoint without loading any hand-built belief."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    scaler = _scaler_from_dict(checkpoint["scaler"])
    model_class = (
        SparseAwarePureJEPABodyModel
        if checkpoint.get("model_type") == "SparseAwarePureJEPABodyModel"
        else PureJEPABodyModel
    )
    model = model_class(
        n_state=len(checkpoint["variables"]),
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint["hidden"]),
        latent=int(checkpoint["latent"]),
        measurement_time_head=bool(
            checkpoint.get("architecture", {}).get("next_measurement_time_head", False)
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, scaler, checkpoint


def _numeric_column(frame: pd.DataFrame, variable: str, suffix: str) -> np.ndarray:
    column = f"{variable}{suffix}"
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
    if suffix == "_t" and variable in frame.columns:
        return pd.to_numeric(frame[variable], errors="coerce").to_numpy(dtype=np.float32)
    return np.full(len(frame), np.nan, dtype=np.float32)


def _history_features(frame: pd.DataFrame, scaler: StateScaler) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build the exact measurement/mask/age/onset input used in training."""

    values = np.column_stack([
        _numeric_column(frame, variable, "_t") for variable in scaler.variables
    ]).astype(np.float32)
    mask = np.isfinite(values).astype(np.float32)
    filled = np.where(np.isfinite(values), values, scaler.medians)
    current = (filled - scaler.medians) / scaler.scales
    ages = np.column_stack([
        _numeric_column(frame, variable, "_age_hr")
        for variable in scaler.variables
    ]).astype(np.float32)
    ages = np.nan_to_num(ages, nan=168.0, posinf=168.0, neginf=0.0)
    ages = np.clip(ages, 0.0, 168.0) / 24.0
    onset = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    onset = np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0)
    onset = np.clip(onset, 0.0, 168.0) / 24.0
    return current.astype(np.float32), mask, ages, onset[:, None]


def _last_history(frame: pd.DataFrame, length: int = HISTORY_LENGTH) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("history_frame must contain at least one pre-anchor observation")
    working = frame.copy()
    if "_time_hours" in working.columns:
        working = working.sort_values("_time_hours", kind="stable")
    elif "hours_since_onset" in working.columns:
        working = working.sort_values("hours_since_onset", kind="stable")
    working = working.tail(max(1, int(length))).reset_index(drop=True)
    if len(working) < length:
        padding = pd.concat([working.iloc[[0]]] * (length - len(working)), ignore_index=True)
        working = pd.concat([padding, working], ignore_index=True)
    return working


def _horizon_conformal(conformal: dict[str, Any], horizon_hours: float) -> dict[str, Any] | None:
    """Return only an interval calibrated for the requested horizon."""
    by_horizon = conformal.get("by_horizon", {})
    if not isinstance(by_horizon, dict):
        return None
    candidates = []
    for key, value in by_horizon.items():
        try:
            distance = abs(float(key) - float(horizon_hours))
        except (TypeError, ValueError):
            continue
        candidates.append((distance, value))
    if not candidates:
        return None
    distance, value = min(candidates, key=lambda item: item[0])
    if distance > 1e-6 or value.get("status") != "PASS":
        return None
    return value


def forecast_history(
    history_frame: pd.DataFrame,
    checkpoint: str | Path | tuple[PureJEPABodyModel, StateScaler, dict[str, Any]],
    *,
    module: str,
    horizon_hours: float,
    registry: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    """Forecast from observations available at an anchor time.

    ``history_frame`` must contain only rows available at or before the anchor.
    The function cannot establish that provenance from values alone, so callers
    must apply their timestamp policy before invoking it.
    """

    if isinstance(checkpoint, (str, Path)):
        model, scaler, metadata = load_checkpoint(checkpoint, device=device)
    else:
        model, scaler, metadata = checkpoint
    history = _last_history(history_frame)
    current_z, masks, ages, onset = _history_features(history, scaler)
    input_matrix = np.concatenate([current_z, masks, ages, onset], axis=1)
    current = torch.from_numpy(current_z[-1:]).to(device=device, dtype=torch.float32)
    observations = torch.from_numpy(input_matrix[None, ...]).to(
        device=device, dtype=torch.float32
    )
    horizon = torch.tensor([float(horizon_hours)], device=device, dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(observations, current, horizon)
    if isinstance(raw_output, dict):
        prediction_z = raw_output["value"].cpu().numpy()[0]
        observation_probability = torch.sigmoid(raw_output["observation_logit"])[0].cpu().numpy()
        stable_probability = torch.sigmoid(raw_output["stable_logit"])[0].cpu().numpy()
        normalized_scale = torch.nn.functional.softplus(raw_output["log_scale"])[0].cpu().numpy()
        uncertainty = normalized_scale * scaler.scales
    else:
        prediction_z = raw_output.cpu().numpy()[0]
        observation_probability = None
        stable_probability = None
        uncertainty = None
    prediction = prediction_z * scaler.scales + scaler.medians
    latest_values = current_z[-1] * scaler.scales + scaler.medians
    checkpoint_status = str(
        metadata.get("promotion_status")
        or metadata.get("architecture", {}).get("promotion_status", "candidate_only")
    )
    checkpoint_approved = checkpoint_status in {"validated", "promoted", "validated_research_only"}
    validated_targets = set(metadata.get("validated_targets", []))
    measurement_time_targets = set(
        metadata.get("measurement_process", {}).get(
            "next_measurement_time_validated_targets", []
        )
    )
    observation_targets = set(
        metadata.get("measurement_process", {}).get(
            "observation_process_validated_targets", []
        )
    )
    stability_targets = set(
        metadata.get("measurement_process", {}).get(
            "stability_validated_targets", []
        )
    )

    targets = {}
    for index, target in enumerate(scaler.variables):
        status = target_status(registry, module, target)
        target_allowed = not validated_targets or target in validated_targets
        allowed = bool(status.get("can_move", False) and checkpoint_approved and target_allowed)
        rejection_reasons = list(status.get("rejection_reasons", []))
        if not checkpoint_approved:
            rejection_reasons.append("checkpoint_candidate_only")
        if not target_allowed:
            rejection_reasons.append("target_not_in_validated_targets")
        point = float(prediction[index] if allowed else latest_values[index])
        conformal = metadata.get("conformal", {}).get(target, {})
        horizon_interval = _horizon_conformal(conformal, horizon_hours)
        q90 = horizon_interval.get("q90") if horizon_interval else None
        interval_ready = bool(
            allowed
            and conformal.get("status") == "PASS"
            and q90 is not None
            and np.isfinite(float(q90))
        )
        lower = float(point - float(q90)) if interval_ready else None
        upper = float(point + float(q90)) if interval_ready else None
        if lower is not None and target in NONNEGATIVE_TARGETS:
            lower = max(0.0, lower)
        targets[target] = {
            "point": point if np.isfinite(point) else None,
            "lower": lower,
            "upper": upper,
            "source": "pure_jepa_validated" if allowed else "persistence_fallback",
            "can_move": allowed,
            "status": status.get("status", "fallback_persistence"),
            "rejection_reasons": rejection_reasons,
            "interval_status": "calibrated" if interval_ready else "needs_horizon_calibration_artifact",
        }
        if observation_probability is not None:
            targets[target]["observation_probability"] = float(observation_probability[index])
            targets[target]["stable_probability"] = float(stable_probability[index])
            targets[target]["uncertainty_scale"] = float(uncertainty[index])
            targets[target]["observation_probability_status"] = (
                "validated" if target in observation_targets else "candidate_only"
            )
            targets[target]["stable_probability_status"] = (
                "validated" if target in stability_targets else "candidate_only"
            )
            targets[target]["observation_probability_semantics"] = (
                "probability_target_is_observed_at_requested_horizon_not_next_measurement_time"
            )
            if "next_measurement_time_log" in raw_output:
                predicted_hours = float(
                    np.expm1(
                        np.log1p(np.exp(np.clip(
                            raw_output["next_measurement_time_log"][0, index].cpu().item(),
                            -40.0, 40.0,
                        )))
                    )
                )
                targets[target]["next_measurement_time_hours"] = predicted_hours
                targets[target]["next_measurement_time_status"] = (
                    "validated" if target in measurement_time_targets else "candidate_only"
                )

    return {
        "module": module,
        "horizon_hours": float(horizon_hours),
        "history_rows_used": int(len(history)),
        "variables": targets,
        "architecture": metadata.get("architecture", {}),
        "factual_only": True,
        "next_measurement_time_available": False,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load the target promotion registry generated by the full gate."""

    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
