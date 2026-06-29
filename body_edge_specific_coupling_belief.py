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
    _hours,
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
FOCUSED_COUPLING_FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def _fast_online_state_features(
    frame: pd.DataFrame,
    *,
    prefix: str,
    observation: np.ndarray,
    confidence: np.ndarray,
    low: float = 0.0,
    high: float = 2.0,
) -> pd.DataFrame:
    """Vectorized online state approximation for very large cohorts.

    This keeps the future-safety property of the slower predict-update helper:
    each row uses its current observation plus the immediately previous
    observation from the same stay.  It avoids per-row DataFrame writes, which
    are too slow for 500k-row long-horizon cohorts.
    """

    observation = np.clip(np.asarray(observation, dtype=np.float64), low, high)
    observation[~np.isfinite(observation)] = 0.0
    confidence = np.clip(np.asarray(confidence, dtype=np.float64), 0.05, 1.0)
    confidence[~np.isfinite(confidence)] = 0.05
    previous = observation.copy()
    if len(frame):
        stay = frame["stay_id"].to_numpy() if "stay_id" in frame else np.arange(len(frame))
        hour = _hours(frame)
        order = np.lexsort((hour, stay))
        same_stay = stay[order][1:] == stay[order][:-1]
        previous[order[1:][same_stay]] = observation[order[:-1][same_stay]]
    innovation = observation - previous
    mean = np.clip(0.65 * observation + 0.35 * previous, low, high)
    sd = np.sqrt(np.clip(0.025 + (1.0 - confidence) * 0.45 + 0.05 * np.abs(innovation), 0.01, 2.0))
    return pd.DataFrame({
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_observation": observation,
        f"{prefix}_innovation": innovation,
        f"{prefix}_delta_since_prior": innovation,
        f"{prefix}_observation_confidence": confidence,
    }, index=frame.index, dtype=np.float64)


def _fast_previous_slope(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    low: float,
    high: float,
) -> np.ndarray:
    """Return current-minus-previous slope by stay using only past rows."""

    values = np.asarray(values, dtype=np.float64)
    values[~np.isfinite(values)] = 0.0
    slope = np.zeros(len(values), dtype=np.float64)
    if not len(frame):
        return slope
    stay = frame["stay_id"].to_numpy() if "stay_id" in frame else np.arange(len(frame))
    hour = _hours(frame)
    order = np.lexsort((hour, stay))
    same_stay = stay[order][1:] == stay[order][:-1]
    current_positions = order[1:][same_stay]
    previous_positions = order[:-1][same_stay]
    delta_hours = np.maximum(hour[current_positions] - hour[previous_positions], 0.25)
    slope[current_positions] = (values[current_positions] - values[previous_positions]) / delta_hours
    return np.clip(slope, low, high)


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


def focused_renal_electrolyte_store_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit renal-electrolyte K and bicarbonate buffering store features."""

    base = edge_renal_electrolyte_buffering_features(frame)
    renal_reserve = _finite_clip(
        base["esc_renal_state2_belief_renal_reserve_mean"].to_numpy(dtype=np.float64),
        0.0,
        2.0,
        0.8,
    )
    creatinine_slope = _finite_clip(
        base["esc_renal_state2_belief_creatinine_slope"].to_numpy(dtype=np.float64),
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
    glucose = _finite_clip(_num(frame, "glucose_t"), 20.0, 1200.0, 140.0)
    urine = _finite_clip(_num(frame, "urine_output_t"), 0.0, 500.0, 60.0)

    rrt = _action_any(frame, "renal_replacement")
    diuretics = _action_any(frame, "diuretics")
    fluids = _action_any(frame, "fluids")
    k_repletion = _action_any(frame, "potassium_repletion")
    bicarbonate_tx = _action_any(frame, "bicarbonate")

    acidemia_shift = _clip01((7.35 - ph) / 0.35)
    renal_clearance_limited = np.clip(1.0 - renal_reserve / 1.5, 0.0, 1.0)
    osmotic_diuresis = _clip01((glucose - 180.0) / 360.0) * _clip01(urine / 180.0)
    k_intracellular_deficit = np.clip(
        0.38 * _clip01((3.7 - potassium) / 1.4)
        + 0.25 * osmotic_diuresis
        + 0.18 * diuretics
        + 0.12 * renal_clearance_limited
        - 0.10 * k_repletion
        - 0.08 * rrt,
        0.0,
        2.0,
    )
    k_extracellular_excess = np.clip(
        0.42 * _clip01((potassium - 5.0) / 2.0)
        + 0.24 * acidemia_shift
        + 0.22 * renal_clearance_limited
        - 0.12 * rrt,
        0.0,
        2.0,
    )
    k_store_observation = np.clip(0.55 * k_intracellular_deficit + 0.45 * k_extracellular_excess, 0.0, 2.0)
    k_store_state = _temporal_scalar_features(
        frame,
        prefix="fbc_k_store_instability",
        observation=k_store_observation,
        confidence=_freshness(frame, ("potassium_age_hr", "ph_age_hr", "creatinine_age_hr", "urine_output_age_hr")),
        drift_per_hour=-0.006 * rrt - 0.004 * k_repletion + 0.004 * osmotic_diuresis,
        low=0.0,
        high=2.0,
    )

    buffer_depletion = np.clip(
        0.36 * _clip01((22.0 - bicarbonate) / 14.0)
        + 0.24 * acidemia_shift
        + 0.22 * _clip01((anion_gap - 14.0) / 18.0)
        + 0.12 * renal_clearance_limited
        + 0.06 * np.maximum(creatinine_slope, 0.0) / 0.08,
        0.0,
        2.0,
    )
    buffer_state = _temporal_scalar_features(
        frame,
        prefix="fbc_bicarbonate_buffer_depletion",
        observation=buffer_depletion,
        confidence=_freshness(frame, ("bicarbonate_age_hr", "ph_age_hr", "anion_gap_age_hr", "creatinine_age_hr")),
        drift_per_hour=-0.006 * bicarbonate_tx - 0.004 * rrt + 0.003 * np.maximum(creatinine_slope, 0.0),
        low=0.0,
        high=2.0,
    )

    sodium_water_store = np.clip(
        0.32 * _clip01(np.abs(sodium - 140.0) / 18.0)
        + 0.28 * _clip01((osmolality - 300.0) / 65.0)
        + 0.20 * _clip01((30.0 - urine) / 30.0)
        + 0.10 * fluids
        + 0.10 * diuretics,
        0.0,
        2.0,
    )
    sodium_state = _temporal_scalar_features(
        frame,
        prefix="fbc_sodium_water_store_stress",
        observation=sodium_water_store,
        confidence=_freshness(frame, ("sodium_age_hr", "serum_osmolality_age_hr", "urine_output_age_hr")),
        drift_per_hour=-0.004 * rrt - 0.002 * fluids + 0.004 * diuretics,
        low=0.0,
        high=2.0,
    )

    interactions = pd.DataFrame({
        "fbc_low_reserve_x_k_store_instability": renal_clearance_limited * k_store_observation,
        "fbc_low_reserve_x_buffer_depletion": renal_clearance_limited * buffer_depletion,
        "fbc_k_shift_x_buffer_depletion": acidemia_shift * buffer_depletion,
        "fbc_k_store_x_sodium_water": k_store_observation * sodium_water_store,
        "fbc_creatinine_slope_x_buffer_depletion": np.maximum(creatinine_slope, 0.0) * buffer_depletion,
    }, index=frame.index)
    return _clean(pd.concat([base, k_store_state, buffer_state, sodium_state, interactions], axis=1))


def focused_cardio_renal_long_horizon_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Perfusion-to-renal shared state for 24h/48h AKI coupling audits."""

    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.1, 30.0, 1.5)
    heart_rate = _finite_clip(_num(frame, "heart_rate_t"), 20.0, 240.0, 85.0)
    urine = _finite_clip(_num(frame, "urine_output_t"), 0.0, 500.0, 60.0)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.2, 20.0, 1.2)
    bun = _finite_clip(_num(frame, "bun_t"), 2.0, 220.0, 22.0)

    vasopressor = _action_any(frame, "vasopressor")
    inotrope = _action_any(frame, "inotrope")
    fluids = _action_any(frame, "fluids")
    diuretics = _action_any(frame, "diuretics")
    rrt = _action_any(frame, "renal_replacement")
    nephrotoxin = _action_any(frame, "nephrotoxin")

    low_map = _clip01((65.0 - map_value) / 35.0)
    lactate_burden = _clip01((lactate - 2.0) / 8.0)
    tachy = _clip01((heart_rate - 110.0) / 55.0)
    shock_observation = np.clip(0.40 * low_map + 0.30 * lactate_burden + 0.12 * tachy + 0.18 * vasopressor, 0.0, 2.0)
    shock_state = _fast_online_state_features(
        frame,
        prefix="fbc_cardio_shock_burden",
        observation=shock_observation,
        confidence=_freshness(frame, ("map_age_hr", "lactate_age_hr", "heart_rate_age_hr")),
        low=0.0,
        high=2.0,
    )
    creatinine_clearance_proxy = np.clip(1.2 / np.maximum(creatinine, 0.2), 0.0, 2.5)
    urine_support = np.clip(urine / 120.0, 0.0, 2.0)
    perfusion_support = np.clip((map_value - 50.0) / 45.0, 0.0, 2.0)
    renal_stress = np.clip(
        0.35 * _clip01((30.0 - urine) / 30.0)
        + 0.25 * low_map
        + 0.15 * vasopressor
        + 0.10 * nephrotoxin
        + 0.10 * _clip01((creatinine - 1.2) / 4.0)
        + 0.05 * _clip01((bun - 25.0) / 80.0),
        0.0,
        1.5,
    )
    renal_reserve = np.clip(
        0.45 * creatinine_clearance_proxy
        + 0.25 * urine_support
        + 0.20 * perfusion_support
        + 0.10 * rrt
        - 0.20 * renal_stress,
        0.0,
        2.0,
    )
    creatinine_slope = _fast_previous_slope(frame, creatinine, low=-0.10, high=0.14)
    shock = shock_state["fbc_cardio_shock_burden_mean"].to_numpy(dtype=np.float64)
    low_reserve = np.clip(1.0 - renal_reserve / 1.5, 0.0, 1.0)
    hypoperfusion = np.clip(
        0.38 * low_map
        + 0.28 * lactate_burden
        + 0.18 * vasopressor
        + 0.10 * inotrope
        + 0.06 * _clip01((30.0 - urine) / 30.0),
        0.0,
        2.0,
    )
    renal_afterload = np.clip(
        0.36 * hypoperfusion
        + 0.28 * low_reserve
        + 0.16 * _clip01((creatinine - 1.2) / 4.0)
        + 0.10 * _clip01((bun - 25.0) / 80.0)
        + 0.10 * nephrotoxin,
        0.0,
        2.0,
    )
    afterload_state = _fast_online_state_features(
        frame,
        prefix="fbc_cardio_renal_afterload",
        observation=renal_afterload,
        confidence=_freshness(frame, ("map_age_hr", "lactate_age_hr", "creatinine_age_hr", "urine_output_age_hr")),
        low=0.0,
        high=2.0,
    )
    recovery_drive = np.clip(
        0.30 * fluids
        + 0.22 * rrt
        + 0.18 * diuretics
        + 0.15 * np.clip(renal_reserve / 1.5, 0.0, 1.0)
        - 0.20 * hypoperfusion,
        0.0,
        2.0,
    )
    recovery_state = _fast_online_state_features(
        frame,
        prefix="fbc_cardio_renal_recovery_drive",
        observation=recovery_drive,
        confidence=_freshness(frame, ("creatinine_age_hr", "urine_output_age_hr", "map_age_hr")),
        low=0.0,
        high=2.0,
    )
    interactions = pd.DataFrame({
        "fbc_vector_renal_reserve": renal_reserve,
        "fbc_vector_creatinine_slope": creatinine_slope,
        "fbc_vector_renal_stress": renal_stress,
        "fbc_vector_hypoperfusion": hypoperfusion,
        "fbc_shock_x_low_renal_reserve": shock * low_reserve,
        "fbc_hypoperfusion_x_creatinine_slope": hypoperfusion * np.maximum(creatinine_slope, 0.0),
        "fbc_afterload_x_current_creatinine": renal_afterload * _clip01((creatinine - 1.2) / 4.0),
        "fbc_afterload_x_oliguria": renal_afterload * _clip01((30.0 - urine) / 30.0),
        "fbc_recovery_x_low_reserve": recovery_drive * low_reserve,
    }, index=frame.index)
    return _clean(pd.concat([shock_state, afterload_state, recovery_state, interactions], axis=1))


def focused_sepsis_cardiovascular_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Sepsis/immune burden coupled to cardiovascular tone and lactate clearance."""

    wbc = _finite_clip(_num(frame, "wbc_t"), 0.1, 120.0, 8.0)
    temperature = _finite_clip(_num(frame, "temperature_t"), 30.0, 43.0, 37.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.1, 30.0, 1.5)
    platelets = _finite_clip(_num(frame, "platelets_t"), 1.0, 1000.0, 220.0)
    albumin = _finite_clip(_num(frame, "albumin_t"), 0.5, 6.0, 3.3)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    heart_rate = _finite_clip(_num(frame, "heart_rate_t"), 20.0, 240.0, 85.0)
    respiratory_rate = _finite_clip(_num(frame, "respiratory_rate_t"), 4.0, 70.0, 18.0)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.2, 20.0, 1.1)

    antibiotics = _action_any(frame, "antibiotics")
    vasopressor = _action_any(frame, "vasopressor")
    fluids = _action_any(frame, "fluids")
    ventilation = _action_any(frame, "ventilation")
    steroid = _action_any(frame, "systemic_steroid")

    leukocyte_stress = np.maximum(_clip01((wbc - 12.0) / 24.0), _clip01((4.0 - wbc) / 3.0))
    fever_stress = np.maximum(_clip01((temperature - 38.3) / 2.7), _clip01((36.0 - temperature) / 4.0))
    inflammatory_observation = np.clip(
        0.26 * leukocyte_stress
        + 0.20 * fever_stress
        + 0.20 * _clip01((lactate - 2.0) / 8.0)
        + 0.14 * _clip01((100.0 - platelets) / 90.0)
        + 0.10 * _clip01((3.0 - albumin) / 1.5)
        + 0.10 * antibiotics,
        0.0,
        2.0,
    )
    inflammatory_state = _fast_online_state_features(
        frame,
        prefix="fbc_sepsis_inflammatory_burden",
        observation=inflammatory_observation,
        confidence=_freshness(frame, ("wbc_age_hr", "temperature_age_hr", "lactate_age_hr", "platelets_age_hr")),
        low=0.0,
        high=2.0,
    )

    vasoplegia_observation = np.clip(
        0.35 * _clip01((65.0 - map_value) / 35.0)
        + 0.22 * vasopressor
        + 0.18 * _clip01((lactate - 2.0) / 8.0)
        + 0.12 * _clip01((heart_rate - 110.0) / 55.0)
        + 0.08 * fluids
        + 0.05 * steroid,
        0.0,
        2.0,
    )
    vasoplegia_state = _fast_online_state_features(
        frame,
        prefix="fbc_sepsis_vasoplegia_burden",
        observation=vasoplegia_observation,
        confidence=_freshness(frame, ("map_age_hr", "lactate_age_hr", "heart_rate_age_hr")),
        low=0.0,
        high=2.0,
    )

    capillary_leak = np.clip(
        0.30 * _clip01((3.0 - albumin) / 1.5)
        + 0.22 * _clip01((100.0 - platelets) / 90.0)
        + 0.18 * _clip01((lactate - 2.0) / 8.0)
        + 0.12 * _clip01((creatinine - 1.5) / 3.0)
        + 0.10 * fluids
        + 0.08 * ventilation,
        0.0,
        2.0,
    )
    capillary_state = _fast_online_state_features(
        frame,
        prefix="fbc_sepsis_capillary_leak",
        observation=capillary_leak,
        confidence=_freshness(frame, ("albumin_age_hr", "platelets_age_hr", "lactate_age_hr", "creatinine_age_hr")),
        low=0.0,
        high=2.0,
    )

    inflammation = inflammatory_state["fbc_sepsis_inflammatory_burden_mean"].to_numpy(dtype=np.float64)
    vasoplegia = vasoplegia_state["fbc_sepsis_vasoplegia_burden_mean"].to_numpy(dtype=np.float64)
    leak = capillary_state["fbc_sepsis_capillary_leak_mean"].to_numpy(dtype=np.float64)
    interactions = pd.DataFrame({
        "fbc_sepsis_inflammation_x_low_map": inflammation * _clip01((65.0 - map_value) / 35.0),
        "fbc_sepsis_inflammation_x_lactate": inflammation * _clip01((lactate - 2.0) / 8.0),
        "fbc_sepsis_vasoplegia_x_tachycardia": vasoplegia * _clip01((heart_rate - 110.0) / 55.0),
        "fbc_sepsis_leak_x_lactate": leak * _clip01((lactate - 2.0) / 8.0),
        "fbc_sepsis_leak_x_hypoalbumin": leak * _clip01((3.0 - albumin) / 1.5),
        "fbc_sepsis_respiratory_stress": _clip01((respiratory_rate - 24.0) / 18.0) + 0.25 * ventilation,
        "fbc_sepsis_treatment_context": np.clip(0.35 * antibiotics + 0.30 * fluids + 0.25 * vasopressor + 0.10 * steroid, 0.0, 1.0),
    }, index=frame.index)
    return _clean(pd.concat([inflammatory_state, vasoplegia_state, capillary_state, interactions], axis=1))


def focused_hepato_renal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Hepatic burden coupled to renal perfusion and hepatorenal stress."""

    bilirubin = _finite_clip(_num(frame, "bilirubin_t"), 0.0, 60.0, 1.0)
    bilirubin_direct = _finite_clip(_num(frame, "bilirubin_direct_t"), 0.0, 50.0, 0.4)
    platelets = _finite_clip(_num(frame, "platelets_t"), 1.0, 1000.0, 220.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.1, 30.0, 1.5)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 2.0, 45.0, 24.0)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.2, 20.0, 1.1)
    bun = _finite_clip(_num(frame, "bun_t"), 2.0, 220.0, 22.0)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    sodium = _finite_clip(_num(frame, "sodium_t"), 105.0, 180.0, 140.0)
    potassium = _finite_clip(_num(frame, "potassium_t"), 1.5, 9.0, 4.2)

    fluids = _action_any(frame, "fluids")
    vasopressor = _action_any(frame, "vasopressor")
    antibiotics = _action_any(frame, "antibiotics")
    rrt = _action_any(frame, "renal_replacement")
    encephalopathy_tx = _action_any(frame, "hepatic_encephalopathy_tx")

    cholestatic_burden = _clip01((bilirubin - 2.0) / 10.0)
    direct_burden = _clip01((bilirubin_direct - 1.0) / 8.0)
    synthetic_proxy = _clip01((120.0 - platelets) / 100.0)
    shock_liver_proxy = _clip01((lactate - 2.0) / 8.0) * _clip01((70.0 - map_value) / 35.0)
    hepatic_observation = np.clip(
        0.32 * cholestatic_burden
        + 0.20 * direct_burden
        + 0.18 * synthetic_proxy
        + 0.18 * shock_liver_proxy
        + 0.12 * encephalopathy_tx,
        0.0,
        2.0,
    )
    hepatic_state = _fast_online_state_features(
        frame,
        prefix="fbc_hepatic_burden",
        observation=hepatic_observation,
        confidence=_freshness(frame, ("bilirubin_age_hr", "bilirubin_direct_age_hr", "platelets_age_hr", "lactate_age_hr")),
        low=0.0,
        high=2.0,
    )

    renal_clearance_pressure = np.clip(
        0.30 * _clip01((creatinine - 1.2) / 4.0)
        + 0.20 * _clip01((bun - 25.0) / 80.0)
        + 0.18 * _clip01((65.0 - map_value) / 35.0)
        + 0.12 * vasopressor
        + 0.10 * _clip01((130.0 - sodium) / 20.0)
        + 0.10 * _clip01((potassium - 5.0) / 2.0),
        0.0,
        2.0,
    )
    renal_pressure_state = _fast_online_state_features(
        frame,
        prefix="fbc_hepato_renal_pressure",
        observation=renal_clearance_pressure,
        confidence=_freshness(frame, ("creatinine_age_hr", "bun_age_hr", "map_age_hr", "sodium_age_hr")),
        low=0.0,
        high=2.0,
    )

    acid_perfusion_stress = np.clip(
        0.30 * _clip01((22.0 - bicarbonate) / 14.0)
        + 0.28 * _clip01((lactate - 2.0) / 8.0)
        + 0.18 * _clip01((65.0 - map_value) / 35.0)
        + 0.12 * cholestatic_burden
        + 0.12 * antibiotics,
        0.0,
        2.0,
    )
    acid_perfusion_state = _fast_online_state_features(
        frame,
        prefix="fbc_hepato_renal_acid_perfusion_stress",
        observation=acid_perfusion_stress,
        confidence=_freshness(frame, ("bicarbonate_age_hr", "lactate_age_hr", "map_age_hr", "bilirubin_age_hr")),
        low=0.0,
        high=2.0,
    )

    hepatic = hepatic_state["fbc_hepatic_burden_mean"].to_numpy(dtype=np.float64)
    renal_pressure = renal_pressure_state["fbc_hepato_renal_pressure_mean"].to_numpy(dtype=np.float64)
    acid_stress = acid_perfusion_state["fbc_hepato_renal_acid_perfusion_stress_mean"].to_numpy(dtype=np.float64)
    interactions = pd.DataFrame({
        "fbc_hepatic_x_creatinine": hepatic * _clip01((creatinine - 1.2) / 4.0),
        "fbc_hepatic_x_bun": hepatic * _clip01((bun - 25.0) / 80.0),
        "fbc_hepatic_x_low_map": hepatic * _clip01((65.0 - map_value) / 35.0),
        "fbc_hepatic_x_hyponatremia": hepatic * _clip01((130.0 - sodium) / 20.0),
        "fbc_renal_pressure_x_lactate": renal_pressure * _clip01((lactate - 2.0) / 8.0),
        "fbc_acid_stress_x_hepatic": acid_stress * hepatic,
        "fbc_hepato_renal_treatment_context": np.clip(0.30 * fluids + 0.25 * vasopressor + 0.20 * rrt + 0.15 * antibiotics + 0.10 * encephalopathy_tx, 0.0, 1.0),
    }, index=frame.index)
    return _clean(pd.concat([hepatic_state, renal_pressure_state, acid_perfusion_state, interactions], axis=1))


FOCUSED_COUPLING_FEATURE_BUILDERS.update({
    "renal_electrolyte_store_6h": focused_renal_electrolyte_store_features,
    "cardio_renal_24h": focused_cardio_renal_long_horizon_features,
    "cardio_renal_48h": focused_cardio_renal_long_horizon_features,
    "sepsis_cardio_6h": focused_sepsis_cardiovascular_features,
    "hepato_renal_6h": focused_hepato_renal_features,
})


def focused_coupling_features(frame: pd.DataFrame, focus_name: str) -> pd.DataFrame:
    """Return focused deep-coupling features for a named physiology frontier."""

    try:
        builder = FOCUSED_COUPLING_FEATURE_BUILDERS[focus_name]
    except KeyError as exc:
        raise KeyError(f"Unknown focused coupling edge: {focus_name}") from exc
    features = builder(frame)
    if len(features) != len(frame):
        raise ValueError(f"Focused coupling feature length mismatch for {focus_name}")
    return features
