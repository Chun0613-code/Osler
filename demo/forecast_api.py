"""Framework-independent HTTP contract for factual Osler forecasts."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from statistics import median
from typing import Any

from personalization_demo import runtime


SCHEMA_VERSION = "osler.forecast.v1"


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
    })
    return result


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
