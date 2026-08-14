"""Framework-independent HTTP contract for factual Osler forecasts."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from personalization_demo import runtime


SCHEMA_VERSION = "osler.forecast.v1"
MONITORING_CASES_PATH = Path(__file__).resolve().parent / "monitoring_cases.json"
ALLOWED_REPLAY_SIGNALS = frozenset({"stable", "watch", "research_signal"})


def forecast_response(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    case_source = payload.get("case_source", "user_supplied")
    if not isinstance(case_source, str) or not case_source.strip():
        raise ValueError("case_source must be a non-empty string")
    result = runtime.forecast(payload)
    result["schema_version"] = SCHEMA_VERSION
    return result


def capabilities_response() -> dict[str, Any]:
    result = runtime.capabilities()
    result.update({
        "schema_version": SCHEMA_VERSION,
        "request_schema": {
            "trajectory": "required non-empty array of observations",
            "anchor_hour": "optional; observations after it are rejected",
            "requested_cells": (
                "optional array of {target, horizon_hours}; unknown cells abstain"
            ),
            "case_source": "optional provenance label; defaults to user_supplied",
        },
        "response_tiers": {
            "validated_artifact": (
                "exact serialized target/horizon artifact with calibrated interval"
            ),
            "illustrative": (
                "belief-derived research illustration without calibrated interval"
            ),
            "unsupported": "abstention; no point forecast or interval",
        },
        "replay_policy": {
            "retrospective_only": True,
            "live_monitoring_supported": False,
            "request_uses_trajectory_prefix_only": True,
            "anchor_equals_prefix_last_observation": True,
            "allowed_signals": sorted(ALLOWED_REPLAY_SIGNALS),
            "clinical_alerts_supported": False,
        },
    })
    return result


@lru_cache(maxsize=1)
def _monitoring_cases_document() -> dict[str, Any]:
    document = json.loads(MONITORING_CASES_PATH.read_text(encoding="utf-8"))
    for case in document.get("cases", []):
        _validate_monitoring_case(case)
    return document


def monitoring_cases_response() -> dict[str, Any]:
    """Return the full retrospective replay timeline for frontend playback."""

    contract = capabilities_response()
    return {
        **deepcopy(_monitoring_cases_document()),
        "model_version": contract["model_version"],
        "safety_boundary": contract["safety_boundary"],
    }


def replay_forecast_request(
    case: dict[str, Any], visible_count: int
) -> dict[str, Any]:
    """Construct the only supported replay request: an observation prefix."""

    _validate_monitoring_case(case)
    events = case["observation_events"]
    if isinstance(visible_count, bool) or not isinstance(visible_count, int):
        raise ValueError("visible_count must be an integer")
    if visible_count < 1 or visible_count > len(events):
        raise ValueError("visible_count is outside the replay timeline")
    prefix = events[:visible_count]
    trajectory = [deepcopy(event["observation"]) for event in prefix]
    anchor_hour = float(trajectory[-1]["hours_since_onset"])
    return {
        "patient_id": case["id"],
        "case_source": case["case_source"],
        "anchor_hour": anchor_hour,
        "trajectory": trajectory,
    }


def _validate_monitoring_case(case: dict[str, Any]) -> None:
    metadata = case.get("replay_metadata")
    events = case.get("observation_events")
    if not isinstance(metadata, dict):
        raise ValueError("monitoring case is missing replay_metadata")
    if metadata.get("retrospective") is not True or metadata.get("not_live") is not True:
        raise ValueError("monitoring replay must be retrospective and not_live")
    if metadata.get("recommended_step") != "next_observation":
        raise ValueError("monitoring replay recommended_step must be next_observation")
    if not isinstance(events, list) or not events:
        raise ValueError("monitoring case must include observation_events")
    initial = metadata.get("initial_visible_count")
    if isinstance(initial, bool) or not isinstance(initial, int):
        raise ValueError("initial_visible_count must be an integer")
    if initial < 1 or initial > len(events):
        raise ValueError("initial_visible_count is outside the replay timeline")
    units = metadata.get("variable_units")
    if not isinstance(units, dict) or not units:
        raise ValueError("monitoring replay must declare variable_units")
    previous_hour = -math.inf
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("observation"), dict):
            raise ValueError("each replay event must include an observation object")
        available = float(event.get("available_at_hour"))
        observation_hour = float(event["observation"].get("hours_since_onset"))
        if available <= previous_hour:
            raise ValueError("observation_events must be strictly time ordered")
        if not math.isclose(available, observation_hour, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("available_at_hour must match the observation time")
        if event.get("signal") not in ALLOWED_REPLAY_SIGNALS:
            raise ValueError("replay event uses an unsupported signal")
        missing_units = set(event["observation"]).difference(units)
        if missing_units:
            raise ValueError(f"observation variables are missing units: {sorted(missing_units)}")
        previous_hour = available


@lru_cache(maxsize=1)
def validation_summary_response() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "renal_patient_state_precision_registry_20260813.json").read_text(
            encoding="utf-8"
        )
    )
    reductions = []
    cells = []
    for cell_name, cell in registry["cells"].items():
        if not cell.get("precision_promoted") or not cell.get("artifact"):
            continue
        report = json.loads((root / cell["validation_report"]).read_text(encoding="utf-8"))
        run_count = 0
        for split in ("patient_heldout", "external_heldout"):
            for run in report[split]["runs"]:
                widths = run["matched_interval_width"]
                population = float(widths["population_mean_half_width"])
                candidate = float(widths["candidate_mean_half_width"])
                reductions.append(100.0 * (population - candidate) / population)
                run_count += 1
        cells.append({
            "cell": cell_name,
            "artifact": cell["artifact"],
            "validation_runs": run_count,
            "patient_heldout_gates": cell["patient_narrowing_gates"],
            "external_heldout_gates": cell["external_narrowing_gates"],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": runtime.MODEL_VERSION,
        "serialized_validated_artifact_count": len(cells),
        "cells": cells,
        "interval_width_reduction": {
            "scope": "held_out_cohort_aggregate_only",
            "case_level_claim_allowed": False,
            "validation_runs": len(reductions),
            "minimum_percent": round(min(reductions), 1),
            "median_percent": round(median(reductions), 1),
            "reported_median_percent_approx": 11.8,
            "maximum_percent": round(max(reductions), 1),
            "plain_language": (
                "Approximately 1.1%-34.5% across 120 held-out runs; "
                "median approximately 11.8%. This is not a real-time "
                "case-level interval-shrinkage claim."
            ),
        },
        "safety_boundary": runtime.capabilities()["safety_boundary"],
    }


def model_health_response() -> dict[str, Any]:
    result = runtime.model_health()
    result["schema_version"] = SCHEMA_VERSION
    return result
