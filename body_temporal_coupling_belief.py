"""Temporal predict-update features for cross-body-system coupling audits.

The static coupling audit showed that simply concatenating upstream organ-system
features is not enough to validate whole-body interaction.  This module builds a
slightly more physiologic candidate: each upstream system is compressed into an
online-compatible burden/reserve belief that evolves within a stay through a
predict -> update loop.

These features are inferred research states.  They are not measured labs, do not
claim causal effects, and are usable only if downstream observable prediction
beats both a baseline and a capacity-matched placebo.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from aki_renal_belief import renal_belief_state_v2_features
from heme_coag_belief import heme_coag_state_features


TEMPORAL_COUPLING_FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def _num(frame: pd.DataFrame, column: str, default=np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _finite_clip(values: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output)] = fill
    return np.clip(output, low, high)


def _clip01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def _action_any(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return (
        (_num(frame, f"hist_{stem}", default=0.0) > 0.0)
        | (_num(frame, f"act_{stem}", default=0.0) > 0.0)
    ).astype(np.float64)


def _freshness(frame: pd.DataFrame, columns: tuple[str, ...], default_age: float = 72.0) -> np.ndarray:
    ages = []
    for column in columns:
        if column in frame:
            ages.append(_finite_clip(_num(frame, column), 0.0, 72.0, default_age))
    if not ages:
        return np.full(len(frame), 0.15, dtype=np.float64)
    stacked = np.vstack(ages)
    return np.clip(1.0 - np.nanmean(stacked, axis=0) / 72.0, 0.05, 1.0)


def _hours(frame: pd.DataFrame) -> np.ndarray:
    if "hours_since_onset" in frame:
        return _finite_clip(_num(frame, "hours_since_onset"), -1.0e6, 1.0e6, 0.0)
    if "t" in frame:
        return _finite_clip(_num(frame, "t"), -1.0e6, 1.0e6, 0.0)
    return np.arange(len(frame), dtype=np.float64)


def _temporal_scalar_features(
    frame: pd.DataFrame,
    *,
    prefix: str,
    observation: np.ndarray,
    confidence: np.ndarray,
    drift_per_hour: np.ndarray | None = None,
    low: float = 0.0,
    high: float = 2.0,
    process_base: float = 0.006,
) -> pd.DataFrame:
    """Return an online predict-update scalar state for each stay."""

    columns = (
        f"{prefix}_mean",
        f"{prefix}_sd",
        f"{prefix}_observation",
        f"{prefix}_innovation",
        f"{prefix}_delta_since_prior",
        f"{prefix}_observation_confidence",
    )
    if frame.empty:
        return pd.DataFrame(np.nan, index=frame.index, columns=columns, dtype=np.float64)
    output = np.full((len(frame), len(columns)), np.nan, dtype=np.float64)

    work = pd.DataFrame({
        "_position": np.arange(len(frame), dtype=np.int64),
        "_stay": frame["stay_id"].to_numpy() if "stay_id" in frame else np.arange(len(frame)),
        "_hour": _hours(frame),
    }, index=frame.index)
    observation = np.clip(np.asarray(observation, dtype=np.float64), low, high)
    confidence = np.clip(np.asarray(confidence, dtype=np.float64), 0.05, 1.0)
    drift = np.zeros(len(frame), dtype=np.float64) if drift_per_hour is None else np.asarray(drift_per_hour, dtype=np.float64)
    drift[~np.isfinite(drift)] = 0.0

    for _stay, group in work.groupby("_stay", sort=False):
        ordered = group.sort_values("_hour")
        mean: float | None = None
        variance: float | None = None
        previous_hour: float | None = None
        previous_position: int | None = None
        for index, row in ordered.iterrows():
            position = int(row["_position"])
            obs = float(observation[position]) if np.isfinite(observation[position]) else 0.0
            conf = float(confidence[position]) if np.isfinite(confidence[position]) else 0.05
            obs_var = float(np.clip(0.025 + (1.0 - conf) * 0.45, 0.02, 1.0))
            if mean is None or variance is None:
                prior_mean = obs
                prior_var = obs_var
                innovation = 0.0
                delta_since_prior = 0.0
            else:
                hour = float(row["_hour"])
                delta_hours = max(0.0, hour - (previous_hour if previous_hour is not None else hour))
                previous_drift = drift[previous_position] if previous_position is not None else 0.0
                prior_mean = float(np.clip(mean + previous_drift * delta_hours, low, high))
                process_var = (process_base + 0.012 * abs(previous_drift)) * max(delta_hours, 0.25)
                prior_var = float(np.clip(variance + process_var, 0.01, 2.0))
                innovation = float(obs - prior_mean)
                delta_since_prior = innovation

            posterior_var = 1.0 / (1.0 / max(prior_var, 1e-6) + 1.0 / max(obs_var, 1e-6))
            posterior_mean = posterior_var * (prior_mean / max(prior_var, 1e-6) + obs / max(obs_var, 1e-6))
            mean = float(np.clip(posterior_mean, low, high))
            variance = float(np.clip(posterior_var, 0.006, 2.0))
            output[position, 0] = mean
            output[position, 1] = float(np.sqrt(max(variance, 1e-6)))
            output[position, 2] = obs
            output[position, 3] = innovation
            output[position, 4] = delta_since_prior
            output[position, 5] = conf
            previous_hour = float(row["_hour"])
            previous_position = position

    return pd.DataFrame(output, index=frame.index, columns=columns, dtype=np.float64)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def temporal_renal_electrolyte_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = renal_belief_state_v2_features(frame).add_prefix("tc_renal_")
    return _clean(features)


def temporal_resp_acid_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    o2sat = _finite_clip(_num(frame, "o2sat_t"), 0.0, 100.0, 96.0)
    rr = _finite_clip(_num(frame, "respiratory_rate_t"), 2.0, 80.0, 18.0)
    ventilation = _action_any(frame, "ventilation")
    bronchodilator = _action_any(frame, "bronchodilator")
    steroid = _action_any(frame, "systemic_steroid")
    oxygen_deficit = _clip01((92.0 - o2sat) / 22.0)
    tachypnea = _clip01((rr - 22.0) / 22.0)
    observation = np.clip(0.45 * oxygen_deficit + 0.35 * tachypnea + 0.20 * ventilation, 0.0, 2.0)
    confidence = _freshness(frame, ("o2sat_age_hr", "respiratory_rate_age_hr"))
    drift = -0.012 * ventilation - 0.004 * bronchodilator + 0.004 * steroid
    state = _temporal_scalar_features(
        frame,
        prefix="tc_resp_burden",
        observation=observation,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    extras = pd.DataFrame({
        "tc_resp_oxygenation_deficit": oxygen_deficit,
        "tc_resp_tachypnea_burden": tachypnea,
        "tc_resp_ventilation_context": ventilation,
        "tc_resp_bronchodilator_context": bronchodilator,
        "tc_resp_steroid_context": steroid,
    }, index=frame.index)
    return _clean(pd.concat([state, extras], axis=1))


def temporal_endocrine_electrolyte_features(frame: pd.DataFrame) -> pd.DataFrame:
    glucose = _finite_clip(_num(frame, "glucose_t"), 20.0, 1200.0, 140.0)
    osmolality = _finite_clip(_num(frame, "serum_osmolality_t"), 240.0, 420.0, 290.0)
    ketones = _finite_clip(_num(frame, "serum_ketones_t"), 0.0, 12.0, 0.0)
    insulin = _action_any(frame, "insulin")
    dextrose = _action_any(frame, "dextrose")
    steroid = _action_any(frame, "systemic_steroid")
    hyperglycemic = _clip01((glucose - 180.0) / 420.0)
    hypoglycemic = _clip01((80.0 - glucose) / 50.0)
    osmotic = _clip01((osmolality - 300.0) / 60.0)
    ketone_burden = _clip01(ketones / 6.0)
    observation = np.clip(0.45 * hyperglycemic + 0.20 * hypoglycemic + 0.20 * osmotic + 0.15 * ketone_burden, 0.0, 2.0)
    confidence = _freshness(frame, ("glucose_age_hr", "serum_osmolality_age_hr", "serum_ketones_age_hr"))
    drift = -0.020 * insulin + 0.014 * dextrose + 0.006 * steroid
    state = _temporal_scalar_features(
        frame,
        prefix="tc_endocrine_osmotic_burden",
        observation=observation,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    extras = pd.DataFrame({
        "tc_endocrine_hyperglycemic_burden": hyperglycemic,
        "tc_endocrine_hypoglycemic_burden": hypoglycemic,
        "tc_endocrine_osmolality_burden": osmotic,
        "tc_endocrine_ketone_burden": ketone_burden,
        "tc_endocrine_insulin_context": insulin,
        "tc_endocrine_dextrose_context": dextrose,
        "tc_endocrine_steroid_context": steroid,
    }, index=frame.index)
    return _clean(pd.concat([state, extras], axis=1))


def temporal_heme_perfusion_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = heme_coag_state_features(frame).add_prefix("tc_heme_")
    return _clean(features)


def temporal_perfusion_renal_features(frame: pd.DataFrame) -> pd.DataFrame:
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.1, 30.0, 1.5)
    heart_rate = _finite_clip(_num(frame, "heart_rate_t"), 20.0, 240.0, 85.0)
    vasopressor = _action_any(frame, "vasopressor")
    inotrope = _action_any(frame, "inotrope")
    fluids = _action_any(frame, "fluids")
    low_map = _clip01((65.0 - map_value) / 35.0)
    lactate_burden = _clip01((lactate - 2.0) / 8.0)
    tachy = _clip01((heart_rate - 110.0) / 55.0)
    observation = np.clip(0.40 * low_map + 0.30 * lactate_burden + 0.15 * tachy + 0.15 * vasopressor, 0.0, 2.0)
    confidence = _freshness(frame, ("map_age_hr", "lactate_age_hr", "heart_rate_age_hr"))
    drift = -0.010 * fluids + 0.008 * vasopressor + 0.004 * inotrope
    state = _temporal_scalar_features(
        frame,
        prefix="tc_perfusion_shock_burden",
        observation=observation,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    extras = pd.DataFrame({
        "tc_perfusion_low_map_burden": low_map,
        "tc_perfusion_lactate_burden": lactate_burden,
        "tc_perfusion_tachycardia_burden": tachy,
        "tc_perfusion_vasopressor_context": vasopressor,
        "tc_perfusion_inotrope_context": inotrope,
        "tc_perfusion_fluid_context": fluids,
    }, index=frame.index)
    return _clean(pd.concat([state, extras], axis=1))


def temporal_hepatic_coag_features(frame: pd.DataFrame) -> pd.DataFrame:
    bilirubin = _finite_clip(_num(frame, "bilirubin_t"), 0.0, 60.0, 1.0)
    direct = _finite_clip(_num(frame, "bilirubin_direct_t"), 0.0, 60.0, 0.3)
    encephalopathy_tx = _action_any(frame, "hepatic_encephalopathy_tx")
    renal_replacement = _action_any(frame, "renal_replacement")
    bili_burden = _clip01((bilirubin - 2.0) / 12.0)
    direct_burden = _clip01((direct - 1.0) / 8.0)
    observation = np.clip(0.55 * bili_burden + 0.25 * direct_burden + 0.15 * encephalopathy_tx + 0.05 * renal_replacement, 0.0, 2.0)
    confidence = _freshness(frame, ("bilirubin_age_hr", "bilirubin_direct_age_hr"))
    drift = -0.002 * encephalopathy_tx + 0.002 * renal_replacement
    state = _temporal_scalar_features(
        frame,
        prefix="tc_hepatic_burden",
        observation=observation,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    extras = pd.DataFrame({
        "tc_hepatic_bilirubin_burden": bili_burden,
        "tc_hepatic_direct_bilirubin_burden": direct_burden,
        "tc_hepatic_encephalopathy_tx_context": encephalopathy_tx,
        "tc_hepatic_renal_replacement_context": renal_replacement,
    }, index=frame.index)
    return _clean(pd.concat([state, extras], axis=1))


def temporal_inflammatory_hemodynamic_features(frame: pd.DataFrame) -> pd.DataFrame:
    wbc = _finite_clip(_num(frame, "wbc_t"), 0.1, 120.0, 10.0)
    temp = _finite_clip(_num(frame, "temperature_t"), 30.0, 43.0, 37.0)
    crp = _finite_clip(_num(frame, "crp_t"), 0.0, 500.0, 0.0)
    ferritin = _finite_clip(_num(frame, "ferritin_t"), 0.0, 50000.0, 0.0)
    antibiotics = _action_any(frame, "antibiotics")
    steroid = _action_any(frame, "systemic_steroid")
    leukocyte_burden = np.maximum(_clip01((wbc - 12.0) / 30.0), _clip01((4.0 - wbc) / 4.0))
    temp_burden = np.maximum(_clip01((temp - 38.0) / 3.0), _clip01((36.0 - temp) / 4.0))
    crp_burden = _clip01(crp / 150.0)
    ferritin_burden = _clip01(ferritin / 5000.0)
    observation = np.clip(0.35 * leukocyte_burden + 0.30 * temp_burden + 0.20 * crp_burden + 0.15 * ferritin_burden, 0.0, 2.0)
    confidence = _freshness(frame, ("wbc_age_hr", "temperature_age_hr", "crp_age_hr", "ferritin_age_hr"))
    drift = -0.006 * antibiotics - 0.004 * steroid
    state = _temporal_scalar_features(
        frame,
        prefix="tc_inflammatory_burden",
        observation=observation,
        confidence=confidence,
        drift_per_hour=drift,
        low=0.0,
        high=2.0,
    )
    extras = pd.DataFrame({
        "tc_inflammatory_leukocyte_burden": leukocyte_burden,
        "tc_inflammatory_temperature_burden": temp_burden,
        "tc_inflammatory_crp_burden": crp_burden,
        "tc_inflammatory_ferritin_burden": ferritin_burden,
        "tc_inflammatory_antibiotic_context": antibiotics,
        "tc_inflammatory_steroid_context": steroid,
    }, index=frame.index)
    return _clean(pd.concat([state, extras], axis=1))


TEMPORAL_COUPLING_FEATURE_BUILDERS.update({
    "renal_to_electrolyte_acid_base": temporal_renal_electrolyte_features,
    "respiratory_to_acid_base": temporal_resp_acid_base_features,
    "endocrine_to_electrolyte": temporal_endocrine_electrolyte_features,
    "heme_to_perfusion": temporal_heme_perfusion_features,
    "cardiovascular_to_renal": temporal_perfusion_renal_features,
    "hepatic_to_coagulation_platelets": temporal_hepatic_coag_features,
    "immune_to_hemodynamics": temporal_inflammatory_hemodynamic_features,
})


def temporal_coupling_features(frame: pd.DataFrame, edge_name: str) -> pd.DataFrame:
    """Return temporal coupling belief features for a named edge."""

    try:
        builder = TEMPORAL_COUPLING_FEATURE_BUILDERS[edge_name]
    except KeyError as exc:
        raise KeyError(f"Unknown temporal coupling edge: {edge_name}") from exc
    features = builder(frame)
    if len(features) != len(frame):
        raise ValueError(f"Temporal coupling feature length mismatch for {edge_name}")
    return features
