"""Explicit whole-body latent coupling features.

This layer sits above the validated per-system belief states.  It does not
invent a new black-box model; it combines already-computed belief outputs into
interpretable cross-system axes such as hemodynamic-renal load,
respiratory-acid-base mismatch, and endocrine-electrolyte stress.

The features are factual observation-layer research features only.  They carry
no causal, counterfactual, clinical, or treatment-planning authority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = default
    return values


def _clip01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def _norm_state(values: np.ndarray, center: float = 0.7, width: float = 1.0) -> np.ndarray:
    return _clip01((np.asarray(values, dtype=np.float64) - center) / max(width, 1e-6))


def whole_body_coupling_latent_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return explicit cross-system latent coupling features.

    The input is expected to contain the per-system belief columns from
    ``eicu_all_model_belief_coupling_audit``.  Missing columns are treated as
    neutral defaults, which lets the same function run across eICU and MIMIC-IV
    cohorts with different measurement coverage.
    """

    renal_reserve = _num(frame, "state2_belief_renal_reserve_mean", 1.0)
    renal_stress = _num(frame, "state2_belief_renal_stress", 0.0)
    creatinine_slope = _num(frame, "state2_belief_creatinine_slope", 0.0)
    low_renal_reserve = _clip01(1.0 - renal_reserve / 1.4)
    rising_creatinine = _clip01(np.maximum(creatinine_slope, 0.0) / 0.08)

    cv_shock = _num(frame, "cv_belief_shock_burden_mean", 0.0)
    cv_low_map = _num(frame, "cv_belief_low_map_burden", 0.0)
    cv_lactate = _num(frame, "cv_belief_lactate_burden", 0.0)
    cv_tachy = _num(frame, "cv_belief_tachycardia_burden", 0.0)

    el_sodium_water = _num(frame, "el_belief_sodium_water_mean", 0.0)
    el_potassium_store = _num(frame, "el_belief_potassium_store_mean", 0.0)
    el_acid_base = _num(frame, "el_belief_acid_base_buffer_mean", 0.0)
    el_osmotic_renal = _num(frame, "el_belief_osmotic_renal_mean", 0.0)
    el_acid = _num(frame, "el_belief_acid_burden", 0.0)
    el_anion_gap = _num(frame, "el_belief_anion_gap_burden", 0.0)
    el_k_low = _num(frame, "el_belief_potassium_low_burden", 0.0)
    el_k_high = _num(frame, "el_belief_potassium_high_burden", 0.0)

    resp_oxygen = _num(frame, "resp_belief_oxygenation_mean", 0.0)
    resp_vent = _num(frame, "resp_belief_ventilation_co2_mean", 0.0)
    resp_comp = _num(frame, "resp_belief_acid_base_compensation_mean", 0.0)
    resp_hypoxemia = _num(frame, "resp_belief_hypoxemia_burden", 0.0)
    resp_tachypnea = _num(frame, "resp_belief_tachypnea_burden", 0.0)
    resp_co2 = _num(frame, "resp_belief_co2_retention_burden", 0.0)

    endo_glycemic = _num(frame, "endo_belief_glycemic_stress_mean", 0.0)
    endo_osmotic = _num(frame, "endo_belief_osmotic_ketotic_stress_mean", 0.0)
    endo_adrenal = _num(frame, "endo_belief_adrenal_hemodynamic_stress_mean", 0.0)
    endo_hyperglycemia = _num(frame, "endo_belief_hyperglycemia_burden", 0.0)
    endo_ketotic = _num(frame, "endo_belief_ketotic_gap_burden", 0.0)
    endo_low_bicarb = _num(frame, "endo_belief_low_bicarbonate_burden", 0.0)

    hemodynamic = np.clip(0.45 * cv_shock + 0.25 * cv_low_map + 0.20 * cv_lactate + 0.10 * _norm_state(endo_adrenal), 0.0, 2.0)
    respiratory = np.clip(0.35 * resp_hypoxemia + 0.25 * resp_tachypnea + 0.20 * resp_co2 + 0.20 * _norm_state(resp_oxygen), 0.0, 2.0)
    metabolic = np.clip(0.35 * _norm_state(endo_glycemic) + 0.25 * _norm_state(endo_osmotic) + 0.20 * endo_hyperglycemia + 0.20 * endo_ketotic, 0.0, 2.0)
    electrolyte = np.clip(0.28 * _norm_state(el_sodium_water) + 0.22 * _norm_state(el_potassium_store) + 0.22 * _norm_state(el_acid_base) + 0.14 * el_k_low + 0.14 * el_k_high, 0.0, 2.0)
    renal = np.clip(0.40 * low_renal_reserve + 0.30 * renal_stress + 0.20 * rising_creatinine + 0.10 * _norm_state(el_osmotic_renal), 0.0, 2.0)

    global_instability = np.clip(
        0.24 * hemodynamic
        + 0.20 * respiratory
        + 0.20 * metabolic
        + 0.18 * electrolyte
        + 0.18 * renal,
        0.0,
        2.0,
    )

    output = pd.DataFrame({
        "wbc_latent_hemodynamic_axis": hemodynamic,
        "wbc_latent_respiratory_axis": respiratory,
        "wbc_latent_metabolic_axis": metabolic,
        "wbc_latent_electrolyte_axis": electrolyte,
        "wbc_latent_renal_axis": renal,
        "wbc_latent_global_instability": global_instability,
        "wbc_hemodynamic_x_low_renal_reserve": hemodynamic * low_renal_reserve,
        "wbc_hemodynamic_x_rising_creatinine": hemodynamic * rising_creatinine,
        "wbc_hemodynamic_x_respiratory": hemodynamic * respiratory,
        "wbc_hemodynamic_x_metabolic": hemodynamic * metabolic,
        "wbc_shock_x_hypoxemia": cv_shock * resp_hypoxemia,
        "wbc_shock_x_lactate": cv_shock * cv_lactate,
        "wbc_adrenal_x_low_map": _norm_state(endo_adrenal) * cv_low_map,
        "wbc_adrenal_x_hemodynamic": _norm_state(endo_adrenal) * hemodynamic,
        "wbc_resp_x_acid_base": respiratory * _norm_state(el_acid_base),
        "wbc_resp_co2_x_acid": resp_co2 * el_acid,
        "wbc_hypoxemia_x_lactate": resp_hypoxemia * cv_lactate,
        "wbc_metabolic_x_sodium_water": metabolic * _norm_state(el_sodium_water),
        "wbc_metabolic_x_potassium_store": metabolic * _norm_state(el_potassium_store),
        "wbc_glycemic_x_potassium_shift": _norm_state(endo_glycemic) * (el_k_low + el_k_high),
        "wbc_ketotic_x_anion_gap": endo_ketotic * el_anion_gap,
        "wbc_low_bicarb_x_acid_buffer": endo_low_bicarb * _norm_state(el_acid_base),
        "wbc_renal_x_potassium_store": renal * _norm_state(el_potassium_store),
        "wbc_renal_x_acid_base": renal * _norm_state(el_acid_base),
        "wbc_renal_x_sodium_water": renal * _norm_state(el_sodium_water),
        "wbc_global_x_hemodynamic": global_instability * hemodynamic,
        "wbc_global_x_respiratory": global_instability * respiratory,
        "wbc_global_x_metabolic": global_instability * metabolic,
        "wbc_global_x_electrolyte": global_instability * electrolyte,
        "wbc_global_x_renal": global_instability * renal,
        "wbc_axis_spread": np.nanstd(np.vstack([hemodynamic, respiratory, metabolic, electrolyte, renal]), axis=0),
    }, index=frame.index, dtype=np.float64)

    return output.replace([np.inf, -np.inf], np.nan)
