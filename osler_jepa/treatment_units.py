"""Unit-safe factual treatment features for MIMIC-IV administration events.

Raw MIMIC administration tables mix volumes, masses, equivalents, and drug
units in the same numeric columns.  These helpers preserve dimensional meaning
before treatment history reaches a forecasting model.  They intentionally do
not estimate treatment effects or infer undocumented doses.
"""

from __future__ import annotations

import re
from typing import Final

import numpy as np


DOSE_DIMENSIONS: Final[tuple[str, ...]] = (
    "volume_ml",
    "mass_mg",
    "amount_mmol",
    "charge_meq",
    "drug_units",
    "energy_kcal",
)

RATE_DIMENSIONS: Final[tuple[str, ...]] = (
    "volume_ml_per_hour",
    "mass_mg_per_hour",
    "amount_mmol_per_hour",
    "charge_meq_per_hour",
    "drug_units_per_hour",
    "mass_mcg_per_kg_min",
    "volume_ml_per_kg_hour",
)


ACTION_DOSE_DIMENSIONS: Final[dict[str, tuple[str, ...]]] = {
    "albumin": ("volume_ml", "mass_mg"),
    "bicarbonate": ("charge_meq", "amount_mmol", "mass_mg"),
    "dextrose": ("mass_mg", "volume_ml"),
    "diuretics": ("mass_mg",),
    "fluids": ("volume_ml",),
    "inotrope": ("mass_mg", "drug_units"),
    "insulin": ("drug_units",),
    "magnesium_repletion": ("amount_mmol", "mass_mg"),
    "nutrition": ("volume_ml", "mass_mg", "energy_kcal"),
    "potassium_repletion": ("charge_meq", "amount_mmol"),
    "systemic_steroid": ("mass_mg",),
    "transfusion": ("volume_ml",),
    "vasopressor": ("mass_mg", "drug_units"),
}

ACTION_RATE_DIMENSIONS: Final[dict[str, tuple[str, ...]]] = {
    "albumin": ("volume_ml_per_hour",),
    "bicarbonate": (
        "charge_meq_per_hour",
        "amount_mmol_per_hour",
        "mass_mg_per_hour",
    ),
    "dextrose": ("volume_ml_per_hour", "mass_mg_per_hour"),
    "diuretics": ("mass_mg_per_hour",),
    "fluids": ("volume_ml_per_hour", "volume_ml_per_kg_hour"),
    "inotrope": (
        "mass_mg_per_hour",
        "drug_units_per_hour",
        "mass_mcg_per_kg_min",
    ),
    "insulin": ("drug_units_per_hour",),
    "magnesium_repletion": ("amount_mmol_per_hour", "mass_mg_per_hour"),
    "nutrition": ("volume_ml_per_hour", "mass_mg_per_hour"),
    "potassium_repletion": ("charge_meq_per_hour", "amount_mmol_per_hour"),
    "systemic_steroid": ("mass_mg_per_hour",),
    "transfusion": ("volume_ml_per_hour",),
    "vasopressor": (
        "mass_mg_per_hour",
        "drug_units_per_hour",
        "mass_mcg_per_kg_min",
    ),
}

PRECISE_ACTIONS: Final[tuple[str, ...]] = tuple(
    sorted(set(ACTION_DOSE_DIMENSIONS) | set(ACTION_RATE_DIMENSIONS))
)


TARGET_TREATMENT_ACTIONS: Final[dict[str, tuple[str, ...]]] = {
    "heart_rate": (
        "albumin",
        "diuretics",
        "fluids",
        "inotrope",
        "transfusion",
        "vasopressor",
    ),
    "hematocrit": (
        "albumin",
        "diuretics",
        "fluids",
        "transfusion",
    ),
    "hemoglobin": (
        "albumin",
        "diuretics",
        "fluids",
        "transfusion",
    ),
    "creatinine": (
        "albumin",
        "diuretics",
        "fluids",
        "inotrope",
        "nephrotoxin",
        "renal_replacement",
        "vasopressor",
    ),
    "glucose": (
        "dextrose",
        "insulin",
        "nutrition",
        "systemic_steroid",
    ),
    "potassium": (
        "bicarbonate",
        "diuretics",
        "insulin",
        "magnesium_repletion",
        "potassium_repletion",
        "renal_replacement",
    ),
    "map": (
        "albumin",
        "diuretics",
        "fluids",
        "inotrope",
        "transfusion",
        "vasopressor",
    ),
    "bicarbonate": (
        "bicarbonate",
        "fluids",
        "insulin",
        "renal_replacement",
        "ventilation",
    ),
    "urine_output": (
        "albumin",
        "diuretics",
        "fluids",
        "inotrope",
        "renal_replacement",
        "vasopressor",
    ),
}


def _unit_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    return re.sub(r"[\s_.]+", "", text)


def canonical_dose(value: object, unit: object) -> tuple[float, str | None]:
    """Return a numeric dose in a stable dimensional unit."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan"), None
    if not np.isfinite(number) or number <= 0.0:
        return float("nan"), None
    text = _unit_text(unit)
    if text in {"ml", "milliliter", "milliliters"}:
        return number, "volume_ml"
    if text in {"l", "liter", "liters"}:
        return number * 1000.0, "volume_ml"
    if text in {"mcg", "ug", "microgram", "micrograms"}:
        return number / 1000.0, "mass_mg"
    if text in {"mg", "milligram", "milligrams"}:
        return number, "mass_mg"
    if text in {"g", "gm", "gram", "grams"}:
        return number * 1000.0, "mass_mg"
    if text in {"mmol", "millimole", "millimoles"}:
        return number, "amount_mmol"
    if text in {"meq", "milliequivalent", "milliequivalents"}:
        return number, "charge_meq"
    if text in {"unit", "units", "u", "iu"}:
        return number, "drug_units"
    if text in {"millionunits", "millionunit"}:
        return number * 1_000_000.0, "drug_units"
    if text in {"kcal", "kilocalorie", "kilocalories"}:
        return number, "energy_kcal"
    return float("nan"), None


def canonical_rate(value: object, unit: object) -> tuple[float, str | None]:
    """Return a rate in a stable unit while preserving weight normalization."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan"), None
    if not np.isfinite(number) or number <= 0.0:
        return float("nan"), None
    text = _unit_text(unit).replace("per", "/")
    aliases = {
        "ml/hr": (number, "volume_ml_per_hour"),
        "ml/hour": (number, "volume_ml_per_hour"),
        "ml/min": (number * 60.0, "volume_ml_per_hour"),
        "l/hr": (number * 1000.0, "volume_ml_per_hour"),
        "l/hour": (number * 1000.0, "volume_ml_per_hour"),
        "mcg/hr": (number / 1000.0, "mass_mg_per_hour"),
        "mcg/hour": (number / 1000.0, "mass_mg_per_hour"),
        "mcg/min": (number * 0.06, "mass_mg_per_hour"),
        "ug/hr": (number / 1000.0, "mass_mg_per_hour"),
        "ug/hour": (number / 1000.0, "mass_mg_per_hour"),
        "ug/min": (number * 0.06, "mass_mg_per_hour"),
        "mg/hr": (number, "mass_mg_per_hour"),
        "mg/hour": (number, "mass_mg_per_hour"),
        "mg/min": (number * 60.0, "mass_mg_per_hour"),
        "g/hr": (number * 1000.0, "mass_mg_per_hour"),
        "g/hour": (number * 1000.0, "mass_mg_per_hour"),
        "grams/hour": (number * 1000.0, "mass_mg_per_hour"),
        "mmol/hr": (number, "amount_mmol_per_hour"),
        "mmol/hour": (number, "amount_mmol_per_hour"),
        "meq/hr": (number, "charge_meq_per_hour"),
        "meq/hour": (number, "charge_meq_per_hour"),
        "unit/hr": (number, "drug_units_per_hour"),
        "units/hr": (number, "drug_units_per_hour"),
        "unit/hour": (number, "drug_units_per_hour"),
        "units/hour": (number, "drug_units_per_hour"),
        "mcg/kg/min": (number, "mass_mcg_per_kg_min"),
        "ug/kg/min": (number, "mass_mcg_per_kg_min"),
        "ml/kg/hr": (number, "volume_ml_per_kg_hour"),
        "ml/kg/hour": (number, "volume_ml_per_kg_hour"),
    }
    return aliases.get(text, (float("nan"), None))


def route_category(route: object, source: object) -> str:
    text = str(route or "").strip().lower()
    if "subcut" in text or text in {"sc", "sq"}:
        return "sc"
    if "oral" in text or text in {"po", "ng", "gt"}:
        return "oral"
    if "intraven" in text or text in {"iv", "iv push", "iv drip"}:
        return "iv"
    if str(source or "") == "mimic_inputevents":
        return "iv"
    return "other"


def treatment_feature_actions(target: str) -> tuple[str, ...]:
    return TARGET_TREATMENT_ACTIONS.get(str(target), PRECISE_ACTIONS)
