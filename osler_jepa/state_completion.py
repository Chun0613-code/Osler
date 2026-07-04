"""Current-state completion helpers for the observation layer.

The helpers deliberately separate deterministic/near-deterministic completion
from cross-system ridge nowcasting.  A high score from hemoglobin->hematocrit
should not be counted as evidence that the whole-body nowcast model understands
unrelated physiology.
"""

from __future__ import annotations

import math
from typing import Any


def complete_current_state(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return same-time completions with explicit provenance.

    Inputs are plain physiologic variable names, without ``_t`` suffixes.  The
    function never fills a missing value with zero.  It only emits values when
    the required sibling measurements are present and finite.
    """

    values = {key: _number(value) for key, value in observation.items()}
    output: dict[str, dict[str, Any]] = {}

    if _missing(values.get("map")) and _present(values.get("sbp")) and _present(values.get("dbp")):
        sbp = float(values["sbp"])
        dbp = float(values["dbp"])
        output["map"] = _entry(
            dbp + (sbp - dbp) / 3.0,
            "derived_formula_completion",
            ("sbp", "dbp"),
            "dbp + (sbp - dbp) / 3",
            "high_when_inputs_observed",
        )

    if (
        _missing(values.get("anion_gap"))
        and _present(values.get("sodium"))
        and _present(values.get("chloride"))
        and _present(values.get("bicarbonate"))
    ):
        output["anion_gap"] = _entry(
            float(values["sodium"]) - float(values["chloride"]) - float(values["bicarbonate"]),
            "derived_formula_completion",
            ("sodium", "chloride", "bicarbonate"),
            "sodium - chloride - bicarbonate",
            "high_when_inputs_observed",
        )

    if (
        _missing(values.get("serum_osmolality"))
        and _present(values.get("sodium"))
        and _present(values.get("glucose"))
        and _present(values.get("bun"))
    ):
        output["serum_osmolality"] = _entry(
            2.0 * float(values["sodium"]) + float(values["glucose"]) / 18.0 + float(values["bun"]) / 2.8,
            "derived_formula_completion",
            ("sodium", "glucose", "bun"),
            "2*sodium + glucose/18 + bun/2.8",
            "moderate_formula_estimate",
        )

    output.update(_complete_red_cell_group(values))
    return output


def _complete_red_cell_group(values: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    hgb = values.get("hemoglobin")
    hct = values.get("hematocrit")
    rbc = values.get("rbc")

    if _missing(hct) and _present(hgb):
        output["hematocrit"] = _entry(
            float(hgb) * 3.0,
            "same_group_calibrated_completion",
            ("hemoglobin",),
            "hematocrit ~= 3 * hemoglobin",
            "moderate_same_group_estimate",
        )
    if _missing(hgb) and _present(hct):
        output["hemoglobin"] = _entry(
            float(hct) / 3.0,
            "same_group_calibrated_completion",
            ("hematocrit",),
            "hemoglobin ~= hematocrit / 3",
            "moderate_same_group_estimate",
        )
    if _missing(rbc) and _present(hgb):
        output["rbc"] = _entry(
            float(hgb) / 3.0,
            "same_group_calibrated_completion",
            ("hemoglobin",),
            "rbc proxy ~= hemoglobin / 3",
            "low_same_group_proxy",
        )
    return output


def _entry(
    value: float,
    source: str,
    inputs: tuple[str, ...],
    formula: str,
    confidence: str,
) -> dict[str, Any]:
    return {
        "point_estimate": float(value),
        "source": source,
        "inputs": inputs,
        "formula": formula,
        "confidence": confidence,
        "clinical_claim_allowed": False,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _present(value: float | None) -> bool:
    return value is not None


def _missing(value: float | None) -> bool:
    return value is None

