"""Runtime helpers for the Osler personalized belief demo.

This module is intentionally small and explicit.  It calls the five validated
belief modules directly, then builds a transparent demo forecast:

* population forecast = persistence baseline from the patient's latest value;
* personalized forecast = persistence plus a belief-derived adjustment from
  that patient's own trajectory;
* interval = conservative demo band, not a clinical confidence interval.

The belief features are the real validated predict-update feature builders.
The demo does not claim to serve the audit ridge coefficients because those
coefficients are not serialized as production artifacts yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aki_renal_belief import renal_belief_state_v2_features
from cardiovascular_belief import cardiovascular_belief_state_features
from electrolyte_belief import electrolyte_belief_state_features
from endocrine_belief import endocrine_belief_state_features
from respiratory_belief import respiratory_belief_state_features
from osler_jepa.state_completion import complete_current_state


@dataclass(frozen=True)
class TargetSpec:
    target: str
    horizon_hours: int
    interval_half_width: float
    max_abs_delta: float
    system: str
    status: str = "validated_demo_runtime"


@dataclass(frozen=True)
class BeliefSystemSpec:
    name: str
    display_name: str
    belief_fn: Any
    targets: tuple[TargetSpec, ...]
    key_features: tuple[str, ...]


SYSTEMS: tuple[BeliefSystemSpec, ...] = (
    BeliefSystemSpec(
        name="renal",
        display_name="Renal Reserve",
        belief_fn=renal_belief_state_v2_features,
        targets=(
            TargetSpec("creatinine", 24, 0.45, 0.8, "renal"),
            TargetSpec("bun", 24, 7.0, 12.0, "renal"),
        ),
        key_features=(
            "state2_belief_renal_reserve_mean",
            "state2_belief_renal_reserve_sd",
            "state2_belief_renal_stress",
            "state2_belief_creatinine_slope",
            "state2_belief_creatinine_innovation",
        ),
    ),
    BeliefSystemSpec(
        name="cardiovascular",
        display_name="Cardiovascular Perfusion",
        belief_fn=cardiovascular_belief_state_features,
        targets=(TargetSpec("heart_rate", 6, 12.0, 18.0, "cardiovascular"),),
        key_features=(
            "cv_belief_shock_burden_mean",
            "cv_belief_shock_burden_sd",
            "cv_belief_shock_burden_innovation",
            "cv_belief_tachycardia_burden",
            "cv_belief_low_map_burden",
        ),
    ),
    BeliefSystemSpec(
        name="electrolyte",
        display_name="Electrolyte / Acid-Base",
        belief_fn=electrolyte_belief_state_features,
        targets=(
            TargetSpec("potassium", 6, 0.55, 1.2, "electrolyte"),
            TargetSpec("bicarbonate", 6, 3.0, 6.0, "electrolyte"),
            TargetSpec("anion_gap", 6, 4.0, 8.0, "electrolyte"),
            TargetSpec("creatinine", 6, 0.5, 0.8, "electrolyte"),
        ),
        key_features=(
            "el_belief_potassium_slope_per_hr",
            "el_belief_bicarbonate_slope_per_hr",
            "el_belief_anion_gap_slope_per_hr",
            "el_belief_creatinine_slope_per_hr",
            "el_belief_potassium_high_burden",
            "el_belief_acid_burden",
        ),
    ),
    BeliefSystemSpec(
        name="respiratory",
        display_name="Respiratory / Gas Exchange",
        belief_fn=respiratory_belief_state_features,
        targets=(
            TargetSpec("o2sat", 6, 4.0, 10.0, "respiratory"),
            TargetSpec("respiratory_rate", 6, 6.0, 12.0, "respiratory"),
            TargetSpec("heart_rate", 6, 12.0, 18.0, "respiratory"),
            TargetSpec("bicarbonate", 6, 3.0, 6.0, "respiratory"),
        ),
        key_features=(
            "resp_belief_oxygenation_mean",
            "resp_belief_oxygenation_sd",
            "resp_belief_o2sat_slope_per_hr",
            "resp_belief_respiratory_rate_slope_per_hr",
            "resp_belief_bicarbonate_slope_per_hr",
            "resp_belief_hypoxemia_burden",
        ),
    ),
    BeliefSystemSpec(
        name="endocrine",
        display_name="Endocrine / Glycemic Stress",
        belief_fn=endocrine_belief_state_features,
        targets=(
            TargetSpec("glucose", 6, 65.0, 120.0, "endocrine"),
            TargetSpec("anion_gap", 6, 4.0, 8.0, "endocrine"),
            TargetSpec("bicarbonate", 6, 3.0, 6.0, "endocrine"),
            TargetSpec("sodium", 6, 4.0, 8.0, "endocrine"),
            TargetSpec("potassium", 6, 0.55, 1.2, "endocrine"),
            TargetSpec("map", 6, 14.0, 24.0, "endocrine"),
        ),
        key_features=(
            "endo_belief_glycemic_stress_mean",
            "endo_belief_osmotic_ketotic_stress_mean",
            "endo_belief_adrenal_hemodynamic_stress_mean",
            "endo_belief_glucose_slope_per_hr",
            "endo_belief_anion_gap_slope_per_hr",
            "endo_belief_map_slope_per_hr",
        ),
    ),
)

SYSTEM_BY_NAME = {system.name: system for system in SYSTEMS}

ALIASES = {
    "spo2": "o2sat",
    "spO2": "o2sat",
    "rr": "respiratory_rate",
    "resp_rate": "respiratory_rate",
    "hr": "heart_rate",
    "temp": "temperature",
    "hco3": "bicarbonate",
    "bicarb": "bicarbonate",
    "ag": "anion_gap",
    "cr": "creatinine",
    "scr": "creatinine",
}


def sample_payload() -> dict[str, Any]:
    """Return a small synthetic trajectory for the demo UI."""

    return {
        "patient_id": "synthetic-demo-patient",
        "trajectory": [
            {
                "hours_since_onset": 0,
                "glucose": 310,
                "anion_gap": 22,
                "bicarbonate": 14,
                "potassium": 4.9,
                "sodium": 134,
                "creatinine": 1.4,
                "bun": 28,
                "map": 68,
                "heart_rate": 118,
                "o2sat": 91,
                "respiratory_rate": 28,
                "paco2": 31,
                "ph": 7.28,
                "temperature": 38.1,
                "act_insulin": 0,
                "act_fluids": 1,
                "act_ventilation": 0,
            },
            {
                "hours_since_onset": 4,
                "glucose": 255,
                "anion_gap": 19,
                "bicarbonate": 16,
                "potassium": 4.4,
                "sodium": 136,
                "creatinine": 1.5,
                "bun": 30,
                "map": 72,
                "heart_rate": 108,
                "o2sat": 93,
                "respiratory_rate": 24,
                "paco2": 34,
                "ph": 7.32,
                "temperature": 37.7,
                "hist_insulin": 1,
                "act_insulin": 1,
                "hist_fluids": 1,
                "act_fluids": 0,
            },
            {
                "hours_since_onset": 8,
                "glucose": 210,
                "anion_gap": 17,
                "bicarbonate": 18,
                "potassium": 4.1,
                "sodium": 137,
                "creatinine": 1.55,
                "bun": 31,
                "map": 75,
                "heart_rate": 100,
                "o2sat": 95,
                "respiratory_rate": 21,
                "paco2": 37,
                "ph": 7.36,
                "temperature": 37.2,
                "hist_insulin": 1,
                "act_insulin": 1,
                "hist_fluids": 1,
                "act_fluids": 0,
            },
        ],
    }


def capabilities() -> dict[str, Any]:
    """Return the public demo contract."""

    return {
        "object": "osler_personalized_belief_demo",
        "validated_belief_systems": [
            {
                "name": system.name,
                "display_name": system.display_name,
                "targets": [
                    {
                        "target": target.target,
                        "horizon_hours": target.horizon_hours,
                        "status": target.status,
                    }
                    for target in system.targets
                ],
            }
            for system in SYSTEMS
        ],
        "forecast_policy": {
            "population_forecast": "persistence baseline from latest observed value",
            "personalized_forecast": "belief-derived adjustment from the patient's own trajectory",
            "ridge_coefficients_serialized": False,
            "interval_policy": "conservative demo bands from validation-scale errors, not clinical intervals",
        },
        "safety_boundary": {
            "clinical_claim_allowed": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "treatment_recommendation_allowed": False,
            "patient_ids_persisted": False,
        },
    }


def make_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("trajectory") or payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("payload must include a non-empty trajectory list")
    normalized = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError("each trajectory row must be an object")
        row: dict[str, Any] = {"stay_id": "demo", "hours_since_onset": float(index)}
        for key, value in raw.items():
            canonical = ALIASES.get(key, key)
            if canonical in {"stay_id", "subject_id", "patient_id"}:
                continue
            if canonical in {"t", "hour", "hours", "hours_since_onset"}:
                row["hours_since_onset"] = _as_float(value, default=float(index))
                continue
            if canonical.endswith("_t") or canonical.endswith("_age_hr"):
                row[canonical] = _as_float(value)
            elif canonical.startswith("act_") or canonical.startswith("hist_"):
                row[canonical] = _as_float(value, default=0.0)
            else:
                row[f"{canonical}_t"] = _as_float(value)
        if "map_t" not in row and "sbp_t" in row and "dbp_t" in row:
            row["map_t"] = (row["sbp_t"] + 2.0 * row["dbp_t"]) / 3.0
        if (
            "anion_gap_t" not in row
            and "sodium_t" in row
            and "chloride_t" in row
            and "bicarbonate_t" in row
        ):
            row["anion_gap_t"] = row["sodium_t"] - row["chloride_t"] - row["bicarbonate_t"]
        completions = complete_current_state({
            key[:-2]: value
            for key, value in row.items()
            if key.endswith("_t")
        })
        for target, payload in completions.items():
            row.setdefault(f"{target}_t", payload["point_estimate"])
        normalized.append(row)
    frame = pd.DataFrame(normalized)
    frame["stay_id"] = "demo"
    frame = frame.sort_values("hours_since_onset").reset_index(drop=True)
    return frame


def forecast(payload: dict[str, Any]) -> dict[str, Any]:
    """Return population and personalized belief-derived forecasts."""

    frame = make_frame(payload)
    outputs = []
    belief_states = {}
    for system in SYSTEMS:
        features = system.belief_fn(frame).reset_index(drop=True)
        latest_features = features.iloc[-1].to_dict() if not features.empty else {}
        belief_states[system.name] = {
            key: _clean_number(latest_features.get(key))
            for key in system.key_features
            if key in latest_features
        }
        latest_row = frame.iloc[-1]
        for target in system.targets:
            current = _latest_value(frame, target.target)
            if current is None:
                outputs.append({
                    "system": system.name,
                    "target": target.target,
                    "horizon_hours": target.horizon_hours,
                    "status": "missing_current_value",
                    "population": None,
                    "personalized": None,
                    "lower": None,
                    "upper": None,
                    "can_personalize": False,
                    "reason": f"no {target.target}_t value in trajectory",
                })
                continue
            adjustment, source = _belief_adjustment(system.name, target, latest_features, latest_row)
            personalized = float(current + adjustment)
            lower = float(personalized - target.interval_half_width)
            upper = float(personalized + target.interval_half_width)
            outputs.append({
                "system": system.name,
                "target": target.target,
                "horizon_hours": target.horizon_hours,
                "status": target.status,
                "population": {
                    "point": float(current),
                    "source": "persistence_population_baseline",
                },
                "personalized": {
                    "point": personalized,
                    "delta_vs_population": float(adjustment),
                    "source": source,
                },
                "lower": lower,
                "upper": upper,
                "interval_status": "demo_validation_scale_band_not_clinical_interval",
                "can_personalize": True,
            })
    return {
        "patient_id": payload.get("patient_id"),
        "latest_hour": float(frame["hours_since_onset"].iloc[-1]),
        "input_rows": int(len(frame)),
        "forecasts": outputs,
        "belief_states": belief_states,
        "contract": capabilities()["forecast_policy"],
        "safety_boundary": capabilities()["safety_boundary"],
    }


def _belief_adjustment(
    system_name: str,
    target: TargetSpec,
    features: dict[str, Any],
    row: pd.Series,
) -> tuple[float, str]:
    slope_keys = [
        f"el_belief_{target.target}_slope_per_hr",
        f"resp_belief_{target.target}_slope_per_hr",
        f"endo_belief_{target.target}_slope_per_hr",
    ]
    if system_name == "renal" and target.target == "creatinine":
        slope_keys.insert(0, "state2_belief_creatinine_slope")
    for key in slope_keys:
        value = _as_float(features.get(key))
        if value is not None and np.isfinite(value):
            delta = 0.65 * value * target.horizon_hours
            return _clip_delta(delta, target.max_abs_delta), f"{key} x horizon"

    if system_name == "renal":
        stress = _as_float(features.get("state2_belief_renal_stress"), 0.0) or 0.0
        reserve = _as_float(features.get("state2_belief_renal_reserve_mean"), 0.5) or 0.5
        sign = 1.0 if target.target in {"creatinine", "bun"} else -1.0
        scale = 0.04 * target.horizon_hours
        return _clip_delta(sign * scale * (stress - reserve), target.max_abs_delta), "renal reserve/stress belief"

    if system_name == "cardiovascular" and target.target == "heart_rate":
        tachy = _as_float(features.get("cv_belief_tachycardia_burden"), 0.0) or 0.0
        shock = _as_float(features.get("cv_belief_shock_burden_innovation"), 0.0) or 0.0
        return _clip_delta(8.0 * tachy + 6.0 * shock, target.max_abs_delta), "perfusion shock/tachycardia belief"

    if system_name == "respiratory" and target.target == "heart_rate":
        hypoxemia = _as_float(features.get("resp_belief_hypoxemia_burden"), 0.0) or 0.0
        tachypnea = _as_float(features.get("resp_belief_tachypnea_burden"), 0.0) or 0.0
        return _clip_delta(7.0 * hypoxemia + 4.0 * tachypnea, target.max_abs_delta), "respiratory burden belief"

    if system_name == "endocrine" and target.target == "map":
        adrenal = _as_float(features.get("endo_belief_adrenal_hemodynamic_stress_mean"), 0.0) or 0.0
        low_map = _as_float(features.get("endo_belief_low_map_burden"), 0.0) or 0.0
        return _clip_delta(-10.0 * adrenal - 8.0 * low_map, target.max_abs_delta), "adrenal/hemodynamic stress belief"

    return 0.0, "no target-specific belief delta; persistence retained"


def _latest_value(frame: pd.DataFrame, target: str) -> float | None:
    column = f"{target}_t"
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _clip_delta(value: float, max_abs_delta: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -max_abs_delta, max_abs_delta))


def _as_float(value: Any, default: float | None = np.nan) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _clean_number(value: Any) -> float | None:
    number = _as_float(value, default=None)
    return None if number is None else float(number)
