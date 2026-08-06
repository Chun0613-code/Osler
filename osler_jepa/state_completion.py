"""Current-state completion helpers for the observation layer.

The helpers deliberately separate deterministic/near-deterministic completion
from cross-system ridge nowcasting.  A high score from hemoglobin->hematocrit
should not be counted as evidence that the whole-body nowcast model understands
unrelated physiology.
"""

from __future__ import annotations

import math
from typing import Any


DERIVED_COMPLETION_RULES: dict[str, dict[str, object]] = {
    "map": {
        "source": "derived_formula_completion",
        "inputs": ("sbp", "dbp"),
        "formula": "dbp + (sbp - dbp) / 3",
        "confidence": "high_when_inputs_observed",
    },
    "pulse_pressure": {
        "source": "derived_formula_completion",
        "inputs": ("sbp", "dbp"),
        "formula": "sbp - dbp",
        "confidence": "high_when_inputs_observed",
    },
    "anion_gap": {
        "source": "derived_formula_completion",
        "inputs": ("sodium", "chloride", "bicarbonate"),
        "formula": "sodium - chloride - bicarbonate",
        "confidence": "high_when_inputs_observed",
    },
    "serum_osmolality": {
        "source": "derived_formula_completion",
        "inputs": ("sodium", "glucose", "bun"),
        "formula": "2*sodium + glucose/18 + bun/2.8",
        "confidence": "moderate_formula_estimate",
    },
    "bilirubin_indirect": {
        "source": "derived_formula_completion",
        "inputs": ("bilirubin_total", "bilirubin_direct"),
        "formula": "bilirubin_total - bilirubin_direct",
        "confidence": "high_when_inputs_observed",
    },
    "globulin": {
        "source": "derived_formula_completion",
        "inputs": ("total_protein", "albumin"),
        "formula": "total_protein - albumin",
        "confidence": "moderate_formula_estimate",
    },
    "non_hdl_cholesterol": {
        "source": "derived_formula_completion",
        "inputs": ("total_cholesterol", "hdl_cholesterol"),
        "formula": "total_cholesterol - hdl_cholesterol",
        "confidence": "high_when_inputs_observed",
    },
    "ldl_cholesterol": {
        "source": "derived_formula_completion",
        "inputs": ("total_cholesterol", "hdl_cholesterol", "triglycerides"),
        "formula": "total_cholesterol - hdl_cholesterol - triglycerides/5",
        "confidence": "moderate_formula_estimate",
    },
    "corrected_calcium": {
        "source": "derived_formula_completion",
        "inputs": ("calcium", "albumin"),
        "formula": "calcium + 0.8*(4 - albumin)",
        "confidence": "moderate_formula_estimate",
    },
    "ck_mb_index": {
        "source": "derived_formula_completion",
        "inputs": ("ck_mb", "cpk"),
        "formula": "100 * ck_mb / cpk",
        "confidence": "moderate_formula_estimate",
    },
    "pf_ratio": {
        "source": "derived_formula_completion",
        "inputs": ("pao2", "fio2"),
        "formula": "pao2 / fio2",
        "confidence": "moderate_formula_estimate",
    },
    "minute_ventilation": {
        "source": "derived_formula_completion",
        "inputs": ("respiratory_rate", "tidal_volume"),
        "formula": "respiratory_rate * tidal_volume / 1000",
        "confidence": "moderate_formula_estimate",
    },
}

SAME_GROUP_CALIBRATED_COMPLETION_GROUPS: dict[str, dict[str, object]] = {
    "blood_pressure": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("sbp", "dbp", "map", "pulse_pressure"),
        "rationale": "blood-pressure family; MAP and pulse pressure are derived from SBP/DBP",
    },
    "red_cell_indices": {
        "source": "same_group_calibrated_completion",
        "targets": ("hemoglobin", "hematocrit", "rbc", "mcv", "mch", "mchc", "rdw"),
        "rationale": "near-linear red-cell measurement family; not cross-system inference",
    },
    "white_cell_differential": {
        "source": "same_group_calibrated_completion",
        "targets": (
            "wbc",
            "neutrophils",
            "lymphocytes",
            "monocytes",
            "eosinophils",
            "basophils",
            "bands",
        ),
        "rationale": "white-cell count and differential family; same-panel completion only",
    },
    "acid_base": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("ph", "paco2", "bicarbonate", "base_excess"),
        "rationale": "acid-base family; Henderson-Hasselbalch-coupled variables",
    },
    "anion_gap_panel": {
        "source": "derived_formula_completion",
        "targets": ("anion_gap", "sodium", "chloride", "bicarbonate"),
        "rationale": "anion gap is derived from sodium, chloride, and bicarbonate",
    },
    "osmolality_panel": {
        "source": "derived_formula_completion",
        "targets": ("serum_osmolality", "osmolality", "sodium", "glucose", "bun"),
        "rationale": "calculated osmolality is derived from sodium, glucose, and BUN",
    },
    "oxygenation": {
        "source": "same_group_calibrated_completion",
        "targets": ("o2sat", "oxygen_saturation", "pao2", "fio2", "pf_ratio"),
        "rationale": "oxygenation family; same-system completion only",
    },
    "ventilation": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("respiratory_rate", "tidal_volume", "minute_ventilation", "minute_volume"),
        "rationale": "minute ventilation is coupled to rate and tidal volume",
    },
    "renal_function": {
        "source": "same_group_calibrated_completion",
        "targets": ("creatinine", "bun", "egfr", "urine_output"),
        "rationale": "renal-function family; not evidence of cross-system nowcast",
    },
    "bilirubin_fraction": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("bilirubin", "bilirubin_total", "bilirubin_direct", "bilirubin_indirect"),
        "rationale": "bilirubin fractions belong to the same measurement family",
    },
    "protein_fraction": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("albumin", "prealbumin", "total_protein", "globulin"),
        "rationale": "globulin is derived from total protein and albumin",
    },
    "coagulation_time": {
        "source": "same_group_calibrated_completion",
        "targets": ("pt", "inr", "pt_inr"),
        "rationale": "INR is standardized from PT; same-family only",
    },
    "partial_thromboplastin": {
        "source": "same_group_calibrated_completion",
        "targets": ("ptt", "aptt"),
        "rationale": "PTT/aPTT naming variants and same assay family",
    },
    "calcium_fraction": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("calcium", "ionized_calcium", "corrected_calcium", "albumin"),
        "rationale": "calcium fractions and albumin correction are same-panel completion",
    },
    "lipids": {
        "source": "derived_formula_or_same_group_completion",
        "targets": (
            "total_cholesterol",
            "hdl_cholesterol",
            "ldl_cholesterol",
            "triglycerides",
            "non_hdl_cholesterol",
        ),
        "rationale": "lipid fractions include deterministic and near-deterministic relationships",
    },
    "cardiac_injury_markers": {
        "source": "same_group_calibrated_completion",
        "targets": ("troponin_i", "troponin_t", "ck_mb", "ck_mb_index", "bnp"),
        "rationale": "cardiac injury biomarker family; same-family evidence only",
    },
    "muscle_injury_markers": {
        "source": "derived_formula_or_same_group_completion",
        "targets": ("cpk", "ck_mb", "ck_mb_index", "myoglobin", "ldh"),
        "rationale": "muscle injury enzyme family; CK-MB index is derived from CK-MB and CPK",
    },
    "pancreatic_enzymes": {
        "source": "same_group_calibrated_completion",
        "targets": ("amylase", "lipase"),
        "rationale": "pancreatic enzyme family; not cross-system inference",
    },
    "inflammatory_markers": {
        "source": "same_group_calibrated_completion",
        "targets": ("crp", "crp_hs", "hs_crp", "esr", "ferritin"),
        "rationale": "inflammatory marker family; same-family evidence only",
    },
    "thyroid_axis": {
        "source": "same_group_calibrated_completion",
        "targets": ("tsh", "free_t4"),
        "rationale": "thyroid-axis measurement family; same-axis completion only",
    },
    "body_size": {
        "source": "same_group_calibrated_completion",
        "targets": ("bmi", "weight", "height", "waist"),
        "rationale": "body-size measurement family with deterministic or near-deterministic relationships",
    },
}

LEAKAGE_SIBLING_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(group["targets"])
    for group in SAME_GROUP_CALIBRATED_COMPLETION_GROUPS.values()
)


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
            **_rule("map"),
        )
    if _missing(values.get("pulse_pressure")) and _present(values.get("sbp")) and _present(values.get("dbp")):
        output["pulse_pressure"] = _entry(
            float(values["sbp"]) - float(values["dbp"]),
            **_rule("pulse_pressure"),
        )

    if (
        _missing(values.get("anion_gap"))
        and _present(values.get("sodium"))
        and _present(values.get("chloride"))
        and _present(values.get("bicarbonate"))
    ):
        output["anion_gap"] = _entry(
            float(values["sodium"]) - float(values["chloride"]) - float(values["bicarbonate"]),
            **_rule("anion_gap"),
        )

    if (
        _missing(values.get("serum_osmolality"))
        and _present(values.get("sodium"))
        and _present(values.get("glucose"))
        and _present(values.get("bun"))
    ):
        osmolality = 2.0 * float(values["sodium"]) + float(values["glucose"]) / 18.0 + float(values["bun"]) / 2.8
        output["serum_osmolality"] = _entry(osmolality, **_rule("serum_osmolality"))
    if (
        _missing(values.get("osmolality"))
        and _present(values.get("sodium"))
        and _present(values.get("glucose"))
        and _present(values.get("bun"))
    ):
        osmolality = 2.0 * float(values["sodium"]) + float(values["glucose"]) / 18.0 + float(values["bun"]) / 2.8
        output["osmolality"] = _entry(osmolality, **_rule("serum_osmolality"))

    total_bili = _first_present(values, ("bilirubin_total", "bilirubin"))
    direct_bili = values.get("bilirubin_direct")
    if _missing(values.get("bilirubin_indirect")) and _present(total_bili) and _present(direct_bili):
        output["bilirubin_indirect"] = _entry(
            float(total_bili) - float(direct_bili),
            **_rule("bilirubin_indirect"),
        )
    if _missing(values.get("globulin")) and _present(values.get("total_protein")) and _present(values.get("albumin")):
        output["globulin"] = _entry(
            float(values["total_protein"]) - float(values["albumin"]),
            **_rule("globulin"),
        )
    if (
        _missing(values.get("non_hdl_cholesterol"))
        and _present(values.get("total_cholesterol"))
        and _present(values.get("hdl_cholesterol"))
    ):
        output["non_hdl_cholesterol"] = _entry(
            float(values["total_cholesterol"]) - float(values["hdl_cholesterol"]),
            **_rule("non_hdl_cholesterol"),
        )
    if (
        _missing(values.get("ldl_cholesterol"))
        and _present(values.get("total_cholesterol"))
        and _present(values.get("hdl_cholesterol"))
        and _present(values.get("triglycerides"))
        and float(values["triglycerides"]) < 400.0
    ):
        output["ldl_cholesterol"] = _entry(
            float(values["total_cholesterol"]) - float(values["hdl_cholesterol"]) - float(values["triglycerides"]) / 5.0,
            **_rule("ldl_cholesterol"),
        )
    if (
        _missing(values.get("corrected_calcium"))
        and _present(values.get("calcium"))
        and _present(values.get("albumin"))
    ):
        output["corrected_calcium"] = _entry(
            float(values["calcium"]) + 0.8 * (4.0 - float(values["albumin"])),
            **_rule("corrected_calcium"),
        )
    if (
        _missing(values.get("ck_mb_index"))
        and _present(values.get("ck_mb"))
        and _present(values.get("cpk"))
        and float(values["cpk"]) != 0.0
    ):
        output["ck_mb_index"] = _entry(
            100.0 * float(values["ck_mb"]) / float(values["cpk"]),
            **_rule("ck_mb_index"),
        )
    if (
        _missing(values.get("pf_ratio"))
        and _present(values.get("pao2"))
        and _present(values.get("fio2"))
        and float(values["fio2"]) != 0.0
    ):
        output["pf_ratio"] = _entry(
            float(values["pao2"]) / float(values["fio2"]),
            **_rule("pf_ratio"),
        )
    if (
        _missing(values.get("minute_ventilation"))
        and _present(values.get("respiratory_rate"))
        and _present(values.get("tidal_volume"))
    ):
        output["minute_ventilation"] = _entry(
            float(values["respiratory_rate"]) * float(values["tidal_volume"]) / 1000.0,
            **_rule("minute_ventilation"),
        )

    output.update(_complete_red_cell_group(values))
    return output


def _complete_red_cell_group(values: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    hgb = values.get("hemoglobin")
    hct = values.get("hematocrit")
    rbc = values.get("rbc")
    mcv = values.get("mcv")
    mch = values.get("mch")
    mchc = values.get("mchc")

    if _missing(hct) and _present(rbc) and _present(mcv):
        output["hematocrit"] = _entry(
            float(rbc) * float(mcv) / 10.0,
            "same_group_calibrated_completion",
            ("rbc", "mcv"),
            "hematocrit ~= rbc * mcv / 10",
            "moderate_same_group_estimate",
        )
    elif _missing(hct) and _present(hgb):
        output["hematocrit"] = _entry(
            float(hgb) * 3.0,
            "same_group_calibrated_completion",
            ("hemoglobin",),
            "hematocrit ~= 3 * hemoglobin",
            "moderate_same_group_estimate",
        )
    if _missing(hgb) and _present(rbc) and _present(mch):
        output["hemoglobin"] = _entry(
            float(rbc) * float(mch) / 10.0,
            "same_group_calibrated_completion",
            ("rbc", "mch"),
            "hemoglobin ~= rbc * mch / 10",
            "moderate_same_group_estimate",
        )
    elif _missing(hgb) and _present(hct):
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
    if _missing(mchc) and _present(hgb) and _present(hct) and float(hct) != 0.0:
        output["mchc"] = _entry(
            float(hgb) * 100.0 / float(hct),
            "same_group_calibrated_completion",
            ("hemoglobin", "hematocrit"),
            "mchc ~= hemoglobin * 100 / hematocrit",
            "moderate_same_group_estimate",
        )
    if _missing(mch) and _present(hgb) and _present(rbc) and float(rbc) != 0.0:
        output["mch"] = _entry(
            float(hgb) * 10.0 / float(rbc),
            "same_group_calibrated_completion",
            ("hemoglobin", "rbc"),
            "mch ~= hemoglobin * 10 / rbc",
            "moderate_same_group_estimate",
        )
    if _missing(mcv) and _present(hct) and _present(rbc) and float(rbc) != 0.0:
        output["mcv"] = _entry(
            float(hct) * 10.0 / float(rbc),
            "same_group_calibrated_completion",
            ("hematocrit", "rbc"),
            "mcv ~= hematocrit * 10 / rbc",
            "moderate_same_group_estimate",
        )
    return output


def _rule(name: str) -> dict[str, Any]:
    rule = DERIVED_COMPLETION_RULES[name]
    return {
        "source": str(rule["source"]),
        "inputs": tuple(rule["inputs"]),
        "formula": str(rule["formula"]),
        "confidence": str(rule["confidence"]),
    }


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


def _first_present(values: dict[str, float | None], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if _present(values.get(key)):
            return values[key]
    return None
