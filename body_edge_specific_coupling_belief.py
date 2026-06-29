"""Edge-specific shared-state features for whole-body coupling audits.

The generic temporal coupling layer produced weak but non-promotable signals.
This module tries the next, more physiologic step: each directed edge receives a
bespoke shared state that combines upstream reserve/burden with downstream
state.  The features are still online-compatible and future-safe: they use only
current/past observations inside the stay.

These are research candidate states.  They are not measured physiology and are
usable only if they improve downstream observable prediction beyond both a
baseline and a capacity-matched placebo.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from aki_renal_belief import renal_belief_state_v2_features
from body_temporal_coupling_belief import (
    _action_any,
    _clean,
    _clip01,
    _finite_clip,
    _freshness,
    _num,
    _temporal_scalar_features,
    temporal_endocrine_electrolyte_features,
    temporal_heme_perfusion_features,
    temporal_inflammatory_hemodynamic_features,
    temporal_perfusion_renal_features,
    temporal_resp_acid_base_features,
)
from heme_coag_belief import heme_coag_state_features


EDGE_SPECIFIC_COUPLING_FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def edge_renal_electrolyte_buffering_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Renal reserve coupled to acid-base, potassium, and sodium-water state."""

    renal = renal_belief_state_v2_features(frame).add_prefix("esc_renal_")
    reserve = _finite_clip(
        renal["esc_renal_state2_belief_renal_reserve_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.8,
    )
    creatinine_slope = _finite_clip(
        renal["esc_renal_state2_belief_creatinine_slope"].to_numpy(dtype=np.float64),
        -0.1,
        0.14,
        0.0,
    )
    potassium = _finite_clip(_num(frame, "potassium_t"), 1.5, 9.0, 4.2)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 2.0, 45.0, 24.0)
    ph = _finite_clip(_num(frame, "ph_t"), 6.7, 7.8, 7.35)
    anion_gap = _finite_clip(_num(frame, "anion_gap_t"), 0.0, 50.0, 12.0)
    sodium = _finite_clip(_num(frame, "sodium_t"), 105.0, 180.0, 140.0)
    osmolality = _finite_clip(_num(frame, "serum_osmolality_t"), 240.0, 430.0, 290.0)
    urine = _finite_clip(_num(frame, "urine_output_t"), 0.0, 500.0, 60.0)

    rrt = _action_any(frame, "renal_replacement")
    diuretics = _action_any(frame, "diuretics")
    fluids = _action_any(frame, "fluids")
    potassium_repletion = _action_any(frame, "potassium_repletion")
    bicarbonate_tx = _action_any(frame, "bicarbonate")

    acid_deficit = np.clip(
        0.42 * _clip01((22.0 - bicarbonate) / 14.0)
        + 0.28 * _clip01((7.32 - ph) / 0.32)
        + 0.20 * _clip01((anion_gap - 14.0) / 18.0)
        + 0.10 * np.clip(1.0 - reserve / 1.5, 0.0, 1.0),
        0.0,
        2.0,
    )
    acid_confidence = _freshness(frame, ("bicarbonate_age_hr", "ph_age_hr", "anion_gap_age_hr", "creatinine_age_hr"))
    acid_drift = -0.006 * rrt - 0.004 * bicarbonate_tx + 0.004 * np.maximum(creatinine_slope, 0.0)
    acid_state = _temporal_scalar_features(
        frame,
        prefix="esc_renal_acid_buffer_deficit",
        observation=acid_deficit,
        confidence=acid_confidence,
        drift_per_hour=acid_drift,
        low=0.0,
        high=2.0,
    )

    k_excess = _clip01((potassium - 5.0) / 2.0)
    k_deficit = _clip01((3.5 - potassium) / 1.5)
    renal_clearance_limited = np.clip(1.0 - reserve / 1.5, 0.0, 1.0)
    k_handling = np.clip(
        0.32 * k_excess * renal_clearance_limited
        + 0.24 * k_deficit
        + 0.18 * _clip01((urine - 120.0) / 220.0)
        + 0.16 * diuretics
        + 0.10 * potassium_repletion,
        0.0,
        2.0,
    )
    k_confidence = _freshness(frame, ("potassium_age_hr", "creatinine_age_hr", "urine_output_age_hr"))
    k_drift = -0.006 * rrt - 0.004 * diuretics + 0.006 * potassium_repletion
    k_state = _temporal_scalar_features(
        frame,
        prefix="esc_renal_potassium_handling_stress",
        observation=k_handling,
        confidence=k_confidence,
        drift_per_hour=k_drift,
        low=0.0,
        high=2.0,
    )

    sodium_water = np.clip(
        0.30 * _clip01(np.abs(sodium - 140.0) / 18.0)
        + 0.25 * _clip01((osmolality - 300.0) / 65.0)
        + 0.20 * _clip01((30.0 - urine) / 30.0)
        + 0.15 * fluids
        + 0.10 * diuretics,
        0.0,
        2.0,
    )
    sodium_confidence = _freshness(frame, ("sodium_age_hr", "serum_osmolality_age_hr", "urine_output_age_hr"))
    sodium_drift = -0.004 * rrt - 0.002 * fluids + 0.004 * diuretics
    sodium_state = _temporal_scalar_features(
        frame,
        prefix="esc_renal_sodium_water_imbalance",
        observation=sodium_water,
        confidence=sodium_confidence,
        drift_per_hour=sodium_drift,
        low=0.0,
        high=2.0,
    )

    interactions = pd.DataFrame({
        "esc_renal_reserve_x_acid_deficit": reserve * acid_deficit,
        "esc_renal_reserve_x_k_handling": reserve * k_handling,
        "esc_renal_reserve_x_sodium_water": reserve * sodium_water,
        "esc_renal_low_reserve_x_acid_deficit": renal_clearance_limited * acid_deficit,
        "esc_renal_low_reserve_x_k_excess": renal_clearance_limited * k_excess,
        "esc_renal_creatinine_slope_x_acid_deficit": np.maximum(creatinine_slope, 0.0) * acid_deficit,
    }, index=frame.index)

    return _clean(pd.concat([renal, acid_state, k_state, sodium_state, interactions], axis=1))


def edge_immune_hemodynamic_capillary_features(frame: pd.DataFrame) -> pd.DataFrame:
    immune = temporal_inflammatory_hemodynamic_features(frame).add_prefix("esc_immune_")
    inflammatory = _finite_clip(
        immune["esc_immune_tc_inflammatory_burden_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.4,
    )
    albumin = _finite_clip(_num(frame, "albumin_t"), 1.0, 6.0, 3.0)
    platelets = _finite_clip(_num(frame, "platelets_t"), 1.0, 1000.0, 180.0)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.1, 30.0, 1.5)
    vasopressor = _action_any(frame, "vasopressor")
    fluids = _action_any(frame, "fluids")
    antibiotics = _action_any(frame, "antibiotics")
    steroid = _action_any(frame, "systemic_steroid")

    capillary = np.clip(
        0.30 * inflammatory
        + 0.22 * _clip01((3.0 - albumin) / 1.5)
        + 0.18 * _clip01((100.0 - platelets) / 90.0)
        + 0.15 * _clip01((65.0 - map_value) / 30.0)
        + 0.10 * _clip01((lactate - 2.0) / 8.0)
        + 0.05 * fluids,
        0.0,
        2.0,
    )
    confidence = _freshness(frame, ("wbc_age_hr", "temperature_age_hr", "albumin_age_hr", "platelets_age_hr", "map_age_hr"))
    drift = -0.004 * antibiotics - 0.003 * steroid + 0.006 * vasopressor + 0.003 * fluids
    capillary_state = _temporal_scalar_features(
        frame,
        prefix="esc_immune_capillary_leak_shock",
        observation=capillary,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    interactions = pd.DataFrame({
        "esc_immune_inflammation_x_low_map": inflammatory * _clip01((65.0 - map_value) / 30.0),
        "esc_immune_inflammation_x_lactate": inflammatory * _clip01((lactate - 2.0) / 8.0),
        "esc_immune_inflammation_x_albumin_deficit": inflammatory * _clip01((3.0 - albumin) / 1.5),
        "esc_immune_capillary_x_platelet_deficit": capillary * _clip01((100.0 - platelets) / 90.0),
        "esc_immune_treatment_context": np.clip(0.4 * antibiotics + 0.3 * steroid + 0.3 * vasopressor, 0.0, 1.0),
    }, index=frame.index)
    return _clean(pd.concat([immune, capillary_state, interactions], axis=1))


def edge_respiratory_co2_ventilation_features(frame: pd.DataFrame) -> pd.DataFrame:
    resp = temporal_resp_acid_base_features(frame).add_prefix("esc_resp_")
    ph = _finite_clip(_num(frame, "ph_t"), 6.7, 7.8, 7.35)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 2.0, 45.0, 24.0)
    rr = _finite_clip(_num(frame, "respiratory_rate_t"), 2.0, 80.0, 18.0)
    o2sat = _finite_clip(_num(frame, "o2sat_t"), 0.0, 100.0, 96.0)
    ventilation = _action_any(frame, "ventilation")
    bronchodilator = _action_any(frame, "bronchodilator")

    co2_proxy = np.clip(
        0.34 * _clip01((7.35 - ph) / 0.25)
        + 0.24 * _clip01((28.0 - bicarbonate) / 18.0)
        + 0.22 * _clip01((rr - 24.0) / 24.0)
        + 0.12 * _clip01((92.0 - o2sat) / 22.0)
        + 0.08 * ventilation,
        0.0,
        2.0,
    )
    confidence = _freshness(frame, ("ph_age_hr", "bicarbonate_age_hr", "respiratory_rate_age_hr", "o2sat_age_hr"))
    drift = -0.010 * ventilation - 0.004 * bronchodilator
    co2_state = _temporal_scalar_features(
        frame,
        prefix="esc_resp_co2_ventilation_mismatch",
        observation=co2_proxy,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    interactions = pd.DataFrame({
        "esc_resp_co2_x_acidemia": co2_proxy * _clip01((7.35 - ph) / 0.25),
        "esc_resp_co2_x_low_bicarbonate": co2_proxy * _clip01((22.0 - bicarbonate) / 14.0),
        "esc_resp_co2_x_tachypnea": co2_proxy * _clip01((rr - 24.0) / 24.0),
        "esc_resp_ventilation_context": ventilation,
    }, index=frame.index)
    return _clean(pd.concat([resp, co2_state, interactions], axis=1))


def edge_heme_oxygen_delivery_features(frame: pd.DataFrame) -> pd.DataFrame:
    heme = heme_coag_state_features(frame).add_prefix("esc_heme_")
    hemoglobin = _finite_clip(_num(frame, "hemoglobin_t"), 1.0, 25.0, 10.0)
    hematocrit = _finite_clip(_num(frame, "hematocrit_t"), 3.0, 75.0, 30.0)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.1, 30.0, 1.5)
    heart_rate = _finite_clip(_num(frame, "heart_rate_t"), 20.0, 240.0, 85.0)
    transfusion = _action_any(frame, "transfusion")
    vasopressor = _action_any(frame, "vasopressor")

    anemia = 0.65 * _clip01((8.0 - hemoglobin) / 3.0) + 0.35 * _clip01((24.0 - hematocrit) / 9.0)
    perfusion_debt = _clip01((65.0 - map_value) / 30.0)
    lactate_debt = _clip01((lactate - 2.0) / 8.0)
    tachy = _clip01((heart_rate - 110.0) / 55.0)
    oxygen_debt = np.clip(
        0.36 * anemia
        + 0.25 * lactate_debt
        + 0.18 * perfusion_debt
        + 0.12 * tachy
        + 0.09 * transfusion,
        0.0,
        2.0,
    )
    confidence = _freshness(frame, ("hemoglobin_age_hr", "hematocrit_age_hr", "map_age_hr", "lactate_age_hr"))
    drift = -0.010 * transfusion + 0.004 * vasopressor
    oxygen_state = _temporal_scalar_features(
        frame,
        prefix="esc_heme_oxygen_delivery_debt",
        observation=oxygen_debt,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    interactions = pd.DataFrame({
        "esc_heme_anemia_x_lactate": anemia * lactate_debt,
        "esc_heme_anemia_x_low_map": anemia * perfusion_debt,
        "esc_heme_oxygen_debt_x_tachycardia": oxygen_debt * tachy,
        "esc_heme_transfusion_context": transfusion,
    }, index=frame.index)
    return _clean(pd.concat([heme, oxygen_state, interactions], axis=1))


def edge_perfusion_renal_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    perfusion = temporal_perfusion_renal_features(frame).add_prefix("esc_cardio_")
    renal = renal_belief_state_v2_features(frame).add_prefix("esc_cardio_renal_")
    shock = _finite_clip(
        perfusion["esc_cardio_tc_perfusion_shock_burden_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.5,
    )
    reserve = _finite_clip(
        renal["esc_cardio_renal_state2_belief_renal_reserve_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.8,
    )
    interactions = pd.DataFrame({
        "esc_cardio_shock_x_low_renal_reserve": shock * np.clip(1.0 - reserve / 1.5, 0.0, 1.0),
        "esc_cardio_shock_x_creatinine_slope": shock * np.maximum(
            renal["esc_cardio_renal_state2_belief_creatinine_slope"].to_numpy(dtype=np.float64),
            0.0,
        ),
    }, index=frame.index)
    return _clean(pd.concat([perfusion, renal, interactions], axis=1))


def edge_hepatic_coagulation_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    # The current eICU hepatic state lacks reliable INR/ammonia trajectories in
    # this cohort, so this remains intentionally conservative.
    from body_temporal_coupling_belief import temporal_hepatic_coag_features

    hepatic = temporal_hepatic_coag_features(frame).add_prefix("esc_hepatic_")
    platelets = _finite_clip(_num(frame, "platelets_t"), 1.0, 1000.0, 180.0)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 2.0, 45.0, 24.0)
    hepatic_mean = _finite_clip(
        hepatic["esc_hepatic_tc_hepatic_burden_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.3,
    )
    interactions = pd.DataFrame({
        "esc_hepatic_burden_x_platelet_deficit": hepatic_mean * _clip01((100.0 - platelets) / 90.0),
        "esc_hepatic_burden_x_low_bicarbonate": hepatic_mean * _clip01((22.0 - bicarbonate) / 14.0),
    }, index=frame.index)
    return _clean(pd.concat([hepatic, interactions], axis=1))


def edge_endocrine_electrolyte_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    endocrine = temporal_endocrine_electrolyte_features(frame).add_prefix("esc_endocrine_")
    potassium = _finite_clip(_num(frame, "potassium_t"), 1.5, 9.0, 4.2)
    sodium = _finite_clip(_num(frame, "sodium_t"), 105.0, 180.0, 140.0)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 2.0, 45.0, 24.0)
    osmotic = _finite_clip(
        endocrine["esc_endocrine_tc_endocrine_osmotic_burden_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.3,
    )
    interactions = pd.DataFrame({
        "esc_endocrine_osmotic_x_potassium_excess": osmotic * _clip01((potassium - 5.0) / 2.0),
        "esc_endocrine_osmotic_x_potassium_deficit": osmotic * _clip01((3.5 - potassium) / 1.5),
        "esc_endocrine_osmotic_x_sodium_deviation": osmotic * _clip01(np.abs(sodium - 140.0) / 18.0),
        "esc_endocrine_osmotic_x_low_bicarbonate": osmotic * _clip01((22.0 - bicarbonate) / 14.0),
    }, index=frame.index)
    return _clean(pd.concat([endocrine, interactions], axis=1))


EDGE_SPECIFIC_COUPLING_FEATURE_BUILDERS.update({
    "renal_to_electrolyte_acid_base": edge_renal_electrolyte_buffering_features,
    "respiratory_to_acid_base": edge_respiratory_co2_ventilation_features,
    "endocrine_to_electrolyte": edge_endocrine_electrolyte_shared_features,
    "heme_to_perfusion": edge_heme_oxygen_delivery_features,
    "cardiovascular_to_renal": edge_perfusion_renal_shared_features,
    "hepatic_to_coagulation_platelets": edge_hepatic_coagulation_shared_features,
    "immune_to_hemodynamics": edge_immune_hemodynamic_capillary_features,
})


def edge_specific_coupling_features(frame: pd.DataFrame, edge_name: str) -> pd.DataFrame:
    """Return edge-specific shared-state features for a named body edge."""

    try:
        builder = EDGE_SPECIFIC_COUPLING_FEATURE_BUILDERS[edge_name]
    except KeyError as exc:
        raise KeyError(f"Unknown edge-specific coupling edge: {edge_name}") from exc
    features = builder(frame)
    if len(features) != len(frame):
        raise ValueError(f"Edge-specific coupling feature length mismatch for {edge_name}")
    return features
