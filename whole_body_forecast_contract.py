"""Unified factual whole-body forecast query contract.

The body-system checkpoints remain independently gated.  This module only
unifies their query and output schema; it does not pretend that 15 separately
trained modules share a validated latent state.  Missing checkpoints,
unvalidated targets, and missing history all fail closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pure_jepa_runtime import forecast_history, load_checkpoint, load_registry


def _latest_history_mapping(history: Any) -> Mapping[str, Any]:
    if hasattr(history, "iloc") and len(history):
        return history.iloc[-1].to_dict()
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        if history and isinstance(history[-1], Mapping):
            return history[-1]
    if isinstance(history, Mapping):
        return history
    return {}


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _upgrade_target_cell(
    target: str,
    cell: Mapping[str, Any],
    latest: Mapping[str, Any],
    module: str,
    horizon: float,
) -> dict[str, Any]:
    observed_value = _finite_or_none(
        latest.get(f"{target}_t", latest.get(target))
    )
    measurement_age = _finite_or_none(
        latest.get(f"{target}_age_hr", latest.get(f"{target}_measurement_age"))
    )
    context = {
        key: value
        for key, value in latest.items()
        if str(key).startswith("hist_") and _finite_or_none(value) not in (None, 0.0)
    }
    rejection_reasons = list(cell.get("rejection_reasons", ()))
    point = _finite_or_none(cell.get("point"))
    source = str(cell.get("source", "persistence_fallback"))
    return {
        "observed_value": observed_value,
        "nowcast": None,
        "forecast_point": point,
        "lower": _finite_or_none(cell.get("lower")),
        "upper": _finite_or_none(cell.get("upper")),
        "actual_horizon": float(horizon),
        "measurement_age": measurement_age,
        "source_model": source,
        "organ_latent_source": module,
        "observed_treatment_context": context,
        "can_move": bool(cell.get("can_move", False)),
        "fallback_reason": rejection_reasons[0] if rejection_reasons else None,
        "rejection_reasons": rejection_reasons,
        "interval_status": str(
            cell.get("interval_status", "needs_calibration_audit")
        ),
        "observation_probability": cell.get("observation_probability"),
        "stable_probability": cell.get("stable_probability"),
        "next_measurement_time_hours": cell.get("next_measurement_time_hours"),
        "causal_claim_allowed": False,
    }


def forecast_whole_body(
    histories: Mapping[str, Any],
    checkpoints: Mapping[str, str | Path],
    registry: Mapping[str, Any] | str | Path,
    horizons: Sequence[float] = (1, 3, 6, 12, 24, 48),
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return one queryable whole-body object without cross-module leakage."""
    if isinstance(registry, (str, Path)):
        registry = load_registry(registry)
    modules: dict[str, Any] = {}
    for module, checkpoint_path in checkpoints.items():
        module_entry: dict[str, Any] = {
            "module": module,
            "status": "missing_history" if module not in histories else "candidate",
            "forecasts": {},
        }
        if module not in histories:
            modules[module] = module_entry
            continue
        latest = _latest_history_mapping(histories[module])
        try:
            loaded = load_checkpoint(checkpoint_path, device=device)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            module_entry["status"] = "checkpoint_unavailable"
            module_entry["error"] = str(exc)
            modules[module] = module_entry
            continue
        for horizon in horizons:
            result = forecast_history(
                histories[module], loaded, module=module,
                horizon_hours=float(horizon), registry=registry, device=device,
            )
            result["targets_v2"] = {
                target: _upgrade_target_cell(
                    target, cell, latest, module, float(horizon)
                )
                for target, cell in result.get("variables", {}).items()
            }
            module_entry["forecasts"][str(horizon)] = result
        module_entry["status"] = "factual_research_only"
        module_entry["shared_latent_validated"] = False
        modules[module] = module_entry
    return {
        "schema": "whole_body_state_forecast.v2",
        "modules": modules,
        "horizons_hours": [float(value) for value in horizons],
        "semantics": {
            "forecast": "future factual trajectory under observed context only",
            "nowcast_and_forecast_are_separate": True,
            "measurement_process_is_separate": True,
            "organ_latent_is_model_derived_not_ground_truth": True,
            "treatment_context": "strictly pre-anchor observed factual input only",
            "unvalidated_target_action": "persistence_fallback",
            "shared_latent_validated": False,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }
