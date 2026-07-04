"""Reconcile JEPA shadow forecasts with later observed DKA physiology.

This module scores factual forecasts only when the observed treatment exposure
and elapsed time match the stored forecast contract. It never updates weights,
promotes rules, or turns observational agreement into a causal claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

import numpy as np

from osler_jepa.shadow import build_dka_shadow_state


def pseudonymize_subject(
    subject_key: Optional[str], salt: Optional[str] = None
) -> Optional[str]:
    """Create a stable local cohort key without storing the source identifier."""
    if subject_key is None or not str(subject_key).strip():
        return None
    secret = salt or os.environ.get("OSLER_JEPA_LEDGER_SALT")
    if not secret:
        raise ValueError(
            "OSLER_JEPA_LEDGER_SALT is required when a subject key is supplied"
        )
    message = f"osler-jepa-shadow-v1|{subject_key}".encode("utf-8")
    return hmac.new(
        str(secret).encode("utf-8"), message, hashlib.sha256
    ).hexdigest()


def _model_contracts():
    """Load numerical contracts lazily to avoid package initialization cycles."""
    from dka_action_contract import ACTION_KEYS, expand_action
    from dka_world_model import A_SCALE, S_STD, STATE_KEYS
    from osler_jepa.symbolic import PHYSICAL_STABLE_THRESHOLDS

    return (
        ACTION_KEYS, expand_action, A_SCALE, S_STD, STATE_KEYS,
        PHYSICAL_STABLE_THRESHOLDS,
    )


def _direction(delta: float, threshold: float) -> str:
    if delta < -threshold:
        return "decrease"
    if delta > threshold:
        return "increase"
    return "stable"


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def load_shadow_forecast(ledger_path: str | Path, event_id: str) -> Dict[str, Any]:
    """Return the latest matching observed shadow forecast from a JSONL ledger."""
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(f"shadow ledger not found: {path}")
    found = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if (
                record.get("record_type") == "shadow_forecast"
                and record.get("event_id") == event_id
            ):
                found = record
    if found is None:
        raise KeyError(f"shadow forecast event not found: {event_id}")
    if found.get("status") != "observed":
        raise ValueError(
            f"shadow event {event_id} is not an observed forecast: "
            f"{found.get('status')}"
        )
    return found


def _select_observation(forecast, candidate: Optional[str]):
    observations = forecast.get("observations") or []
    if candidate is not None:
        wanted = candidate.strip().lower()
        observations = [
            observation for observation in observations
            if str(observation.get("candidate") or "").strip().lower() == wanted
        ]
    if not observations:
        raise KeyError("no matching forecast observation")
    if len(observations) > 1:
        raise ValueError("candidate is required when a forecast has multiple observations")
    return observations[0]


def _action_alignment(expected, actual, tolerance: float):
    ACTION_KEYS, expand_action, A_SCALE, _, _, _ = _model_contracts()
    expected_vector = expand_action(expected).astype(np.float32)
    actual_vector = expand_action(actual).astype(np.float32)
    normalized_error = np.abs(expected_vector - actual_vector) / A_SCALE
    return {
        "matches": bool(np.all(normalized_error <= tolerance)),
        "tolerance_fraction_of_training_scale": float(tolerance),
        "max_normalized_difference": round(float(normalized_error.max()), 6),
        "expected_mean_exposure": {
            name: round(float(value), 6)
            for name, value in zip(ACTION_KEYS, expected_vector)
        },
        "observed_mean_exposure": {
            name: round(float(value), 6)
            for name, value in zip(ACTION_KEYS, actual_vector)
        },
        "per_action_normalized_difference": {
            name: round(float(value), 6)
            for name, value in zip(ACTION_KEYS, normalized_error)
        },
    }


def _schedule_vector(schedule, time_hours: float):
    _, expand_action, _, _, _, _ = _model_contracts()
    current = expand_action({}).astype(np.float32)
    for event in sorted(schedule, key=lambda item: float(item.get("hours", 0.0))):
        if float(event.get("hours", 0.0)) > time_hours + 1e-9:
            break
        current = expand_action(event.get("action") or {}).astype(np.float32)
    return current


def _mean_schedule_exposure(schedule, horizon_hours: float):
    ACTION_KEYS, _, _, _, _, _ = _model_contracts()
    horizon = max(float(horizon_hours), 1e-6)
    points = sorted({
        0.0, horizon,
        *[
            min(horizon, max(0.0, float(event.get("hours", 0.0))))
            for event in schedule
        ],
    })
    total = np.zeros(len(ACTION_KEYS), dtype=np.float32)
    for start, stop in zip(points[:-1], points[1:]):
        if stop > start:
            total += _schedule_vector(schedule, (start + stop) / 2.0) * (stop - start)
    return total / horizon


def _temporal_action_alignment(expected_schedule, actual_schedule, horizon, tolerance):
    ACTION_KEYS, _, A_SCALE, _, _, _ = _model_contracts()
    points = sorted({
        0.0, float(horizon),
        *[
            min(float(horizon), max(0.0, float(event.get("hours", 0.0))))
            for event in [*expected_schedule, *actual_schedule]
        ],
    })
    maximum = np.zeros(len(ACTION_KEYS), dtype=np.float32)
    for start, stop in zip(points[:-1], points[1:]):
        if stop <= start:
            continue
        midpoint = (start + stop) / 2.0
        difference = np.abs(
            _schedule_vector(expected_schedule, midpoint)
            - _schedule_vector(actual_schedule, midpoint)
        ) / A_SCALE
        maximum = np.maximum(maximum, difference)
    return {
        "verified": True,
        "matches": bool(np.all(maximum <= tolerance)),
        "max_normalized_difference": round(float(maximum.max()), 6),
        "per_action_max_normalized_difference": {
            name: round(float(value), 6)
            for name, value in zip(ACTION_KEYS, maximum)
        },
    }


def reconcile_shadow_forecast(
    forecast: Mapping[str, Any],
    future_fields: Mapping[str, Any],
    actual_action: Mapping[str, Any] | list[float],
    elapsed_hours: float,
    candidate: Optional[str] = None,
    action_tolerance: float = 0.10,
    horizon_tolerance_hours: float = 1.0,
    subject_key: Optional[str] = None,
    subject_salt: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a stored factual forecast with later measured physiology."""
    observation = _select_observation(forecast, candidate)
    subject_group_hash = pseudonymize_subject(subject_key, subject_salt)
    initial_state = (forecast.get("state_contract") or {}).get("state") or {}
    predicted_state = observation.get("predicted_final_state") or {}
    expected_action = observation.get("effective_action_summary")
    expected_schedule = observation.get("effective_action_schedule") or []
    if not expected_action:
        raise ValueError("forecast lacks effective_action_summary")

    actual_contract = build_dka_shadow_state(future_fields)
    expected_hours = float(observation.get("horizon_hours") or 0.0)
    elapsed_hours = float(elapsed_hours)
    actual_schedule = (
        actual_action.get("schedule")
        if isinstance(actual_action, Mapping) and "schedule" in actual_action
        else None
    )
    actual_mean = (
        _mean_schedule_exposure(actual_schedule, expected_hours)
        if actual_schedule is not None else actual_action
    )
    action_alignment = _action_alignment(
        expected_action, actual_mean, float(action_tolerance)
    )
    if expected_schedule and actual_schedule is not None:
        temporal_alignment = _temporal_action_alignment(
            expected_schedule, actual_schedule, expected_hours,
            float(action_tolerance),
        )
    else:
        temporal_alignment = {
            "verified": False,
            "matches": False,
            "reason": "both forecast and observed start/stop schedules are required",
        }
    action_alignment["temporal_alignment"] = temporal_alignment
    horizon_difference = abs(elapsed_hours - expected_hours)
    horizon_matches = horizon_difference <= float(horizon_tolerance_hours)

    record: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "shadow_outcome_reconciliation",
        "reconciliation_id": str(uuid4()),
        "forecast_event_id": forecast.get("event_id"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": observation.get("candidate"),
        "checkpoint": forecast.get("checkpoint"),
        "subject_group_hash": subject_group_hash,
        "privacy": {
            "subject_identifier_stored": False,
            "grouping_method": (
                "hmac_sha256_local_salt" if subject_group_hash else "not_supplied"
            ),
        },
        "forecast_provenance": {
            "transition_validation_status": (
                observation.get("transition_validation") or {}
            ).get("status"),
            "prolog_decision": (
                observation.get("prolog_reasoning") or {}
            ).get("decision"),
            "symbolic_disagreement": bool(observation.get("disagreement")),
        },
        "forecast_type": "factual_observed-treatment forecast",
        "research_only": True,
        "causal_claim_allowed": False,
        "clinical_decision_authority": False,
        "online_weight_update_allowed": False,
        "automatic_rule_promotion_allowed": False,
        "eligible_for_offline_training_review": False,
        "action_alignment": action_alignment,
        "horizon_alignment": {
            "matches": horizon_matches,
            "expected_hours": expected_hours,
            "observed_hours": elapsed_hours,
            "absolute_difference_hours": round(horizon_difference, 6),
            "tolerance_hours": float(horizon_tolerance_hours),
        },
        "future_observation_contract": actual_contract,
    }
    if not action_alignment["matches"]:
        record.update({
            "status": "action_mismatch",
            "reason": "observed treatment exposure does not match the forecast path",
        })
        return record
    if not temporal_alignment["verified"]:
        record.update({
            "status": "action_timing_unverified",
            "reason": "mean exposure matches but treatment timing was not supplied",
        })
        return record
    if not temporal_alignment["matches"]:
        record.update({
            "status": "action_timing_mismatch",
            "reason": "observed treatment start/stop timing differs from the forecast path",
        })
        return record
    if not horizon_matches:
        record.update({
            "status": "horizon_mismatch",
            "reason": "future observation is outside the forecast time tolerance",
        })
        return record

    actual_state = actual_contract["state"]
    _, _, _, S_STD, STATE_KEYS, PHYSICAL_STABLE_THRESHOLDS = _model_contracts()
    common = [
        name for name in STATE_KEYS
        if name in initial_state and name in predicted_state and name in actual_state
    ]
    if not common:
        record.update({
            "status": "insufficient_outcome",
            "reason": "no measured future state overlaps the stored forecast",
        })
        return record

    per_state = {}
    jepa_errors = []
    persistence_errors = []
    changed_jepa = []
    changed_persistence = []
    for name in common:
        index = STATE_KEYS.index(name)
        initial = float(initial_state[name])
        predicted = float(predicted_state[name])
        actual = float(actual_state[name])
        scale = max(float(S_STD[index]), 1e-6)
        threshold = float(PHYSICAL_STABLE_THRESHOLDS[index])
        actual_direction = _direction(actual - initial, threshold)
        predicted_direction = _direction(predicted - initial, threshold)
        persistence_direction = "stable"
        jepa_error = abs(actual - predicted)
        persistence_error = abs(actual - initial)
        jepa_normalized = jepa_error / scale
        persistence_normalized = persistence_error / scale
        jepa_errors.append(jepa_normalized)
        persistence_errors.append(persistence_normalized)
        if actual_direction != "stable":
            changed_jepa.append(predicted_direction == actual_direction)
            changed_persistence.append(False)
        per_state[name] = {
            "initial": round(initial, 6),
            "predicted": round(predicted, 6),
            "observed": round(actual, 6),
            "jepa_absolute_error": round(jepa_error, 6),
            "persistence_absolute_error": round(persistence_error, 6),
            "jepa_normalized_error": round(jepa_normalized, 6),
            "persistence_normalized_error": round(persistence_normalized, 6),
            "normalized_improvement_over_persistence": round(
                persistence_normalized - jepa_normalized, 6
            ),
            "observed_direction": actual_direction,
            "jepa_direction": predicted_direction,
            "persistence_direction": persistence_direction,
            "jepa_direction_correct": predicted_direction == actual_direction,
            "persistence_direction_correct": persistence_direction == actual_direction,
            "measurement_provenance": actual_contract["provenance"].get(name),
        }

    jepa_mean = float(np.mean(jepa_errors))
    persistence_mean = float(np.mean(persistence_errors))
    improvement = persistence_mean - jepa_mean
    winner = "tie"
    if improvement > 1e-6:
        winner = "jepa"
    elif improvement < -1e-6:
        winner = "persistence"
    outcome = future_fields.get("outcome") or {}
    if "alive" in future_fields and "alive" not in outcome:
        outcome = {**outcome, "alive": future_fields.get("alive")}
    terminal_outcome = None
    if outcome.get("alive") is not None:
        observed_dead = not bool(outcome.get("alive"))
        predicted_risk = (observation.get("predicted_risk") or {}).get(
            "intervention"
        )
        terminal_outcome = {
            "observed_dead": observed_dead,
            "death_cause": outcome.get("death_cause"),
            "predicted_death_probability": predicted_risk,
            "brier_score": (
                round((float(predicted_risk) - float(observed_dead)) ** 2, 6)
                if predicted_risk is not None else None
            ),
        }
    record.update({
        "status": "scored",
        "per_state": per_state,
        "summary": {
            "observed_state_count": len(common),
            "jepa_mean_normalized_absolute_error": round(jepa_mean, 6),
            "persistence_mean_normalized_absolute_error": round(
                persistence_mean, 6
            ),
            "normalized_improvement_over_persistence": round(improvement, 6),
            "winner": winner,
            "changed_state_count": len(changed_jepa),
            "jepa_changed_state_direction_accuracy": (
                round(float(np.mean(changed_jepa)), 6) if changed_jepa else None
            ),
            "persistence_changed_state_direction_accuracy": (
                round(float(np.mean(changed_persistence)), 6)
                if changed_persistence else None
            ),
        },
        "eligible_for_offline_training_review": (
            len(common) >= 4 and subject_group_hash is not None
        ),
        "eligible_for_cohort_evaluation": subject_group_hash is not None,
        "review_gate": (
            "human_review_required_before any dataset inclusion, weight update, "
            "or rule-candidate generation"
        ),
    })
    if terminal_outcome is not None:
        record["terminal_outcome"] = terminal_outcome
    return record


def reconcile_from_ledger(
    ledger_path: str | Path,
    event_id: str,
    future_fields: Mapping[str, Any],
    actual_action: Mapping[str, Any] | list[float],
    elapsed_hours: float,
    candidate: Optional[str] = None,
    append: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    forecast = load_shadow_forecast(ledger_path, event_id)
    record = reconcile_shadow_forecast(
        forecast,
        future_fields,
        actual_action,
        elapsed_hours,
        candidate=candidate,
        **kwargs,
    )
    if append:
        _append_jsonl(Path(ledger_path), record)
    return record
