"""Patient-specific immune / inflammatory belief features.

This module extends the validated predict-update personalization pattern to
immune-inflammatory physiology.  It compresses each patient's own fever,
leukocyte, lactate/perfusion, respiratory stress, renal stress, and observed
infection-treatment context into online-compatible features.

The features are inferred research states, not measured clinical variables.
They can be used only when a downstream observable audit shows improvement
beyond both a baseline model and a capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_TRACKED_LEVELS = (
    "wbc",
    "temperature",
    "lactate",
    "map",
    "heart_rate",
    "respiratory_rate",
    "o2sat",
    "creatinine",
    "platelets",
    "albumin",
)

IMMUNE_STATE_STEMS = (
    "inflammatory_activation",
    "septic_perfusion_stress",
    "host_response_stress",
)

IMMUNE_BELIEF_COLUMNS = tuple(
    column
    for stem in IMMUNE_STATE_STEMS
    for column in (
        f"immune_belief_{stem}_mean",
        f"immune_belief_{stem}_sd",
        f"immune_belief_{stem}_innovation",
        f"immune_belief_{stem}_confidence",
    )
) + tuple(
    column
    for var in CORE_TRACKED_LEVELS
    for column in (
        f"immune_belief_{var}_slope_per_hr",
        f"immune_belief_{var}_innovation",
    )
) + (
    "immune_belief_fever_burden",
    "immune_belief_hypothermia_burden",
    "immune_belief_leukocytosis_burden",
    "immune_belief_leukopenia_burden",
    "immune_belief_lactate_burden",
    "immune_belief_low_map_burden",
    "immune_belief_tachycardia_burden",
    "immune_belief_tachypnea_burden",
    "immune_belief_hypoxemia_burden",
    "immune_belief_renal_stress_burden",
    "immune_belief_thrombocytopenia_burden",
    "immune_belief_hypoalbuminemia_burden",
    "immune_belief_antibiotic_context",
    "immune_belief_fluid_context",
    "immune_belief_vasopressor_context",
    "immune_belief_steroid_context",
)


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


def _coerce_scalar(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


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
        return np.full(len(frame), 0.10, dtype=np.float64)
    stacked = np.vstack(ages)
    return np.clip(1.0 - np.nanmean(stacked, axis=0) / 72.0, 0.05, 1.0)


def _hours(frame: pd.DataFrame) -> np.ndarray:
    if "hours_since_onset" in frame:
        return _finite_clip(_num(frame, "hours_since_onset"), -1.0e6, 1.0e6, 0.0)
    if "t_hour" in frame:
        return _finite_clip(_num(frame, "t_hour"), -1.0e6, 1.0e6, 0.0)
    if "onset_hour" in frame and "t" in frame:
        return _finite_clip(_num(frame, "t") - _num(frame, "onset_hour"), -1.0e6, 1.0e6, 0.0)
    return np.arange(len(frame), dtype=np.float64)


def _scalar_state(
    frame: pd.DataFrame,
    *,
    prefix: str,
    observation: np.ndarray,
    confidence: np.ndarray,
    drift_per_hour: np.ndarray,
    low: float = 0.0,
    high: float = 2.0,
) -> pd.DataFrame:
    columns = (
        f"immune_belief_{prefix}_mean",
        f"immune_belief_{prefix}_sd",
        f"immune_belief_{prefix}_innovation",
        f"immune_belief_{prefix}_confidence",
    )
    output = np.full((len(frame), len(columns)), np.nan, dtype=np.float64)
    if frame.empty:
        return pd.DataFrame(output, index=frame.index, columns=columns)

    work = pd.DataFrame({
        "_pos": np.arange(len(frame), dtype=np.int64),
        "_stay": frame["stay_id"].to_numpy() if "stay_id" in frame else np.arange(len(frame)),
        "_hour": _hours(frame),
    }, index=frame.index)
    obs_values = np.clip(np.asarray(observation, dtype=np.float64), low, high)
    obs_values[~np.isfinite(obs_values)] = low
    conf_values = np.clip(np.asarray(confidence, dtype=np.float64), 0.05, 1.0)
    drift = np.asarray(drift_per_hour, dtype=np.float64)
    drift[~np.isfinite(drift)] = 0.0

    for _stay, group in work.groupby("_stay", sort=False):
        ordered = group.sort_values("_hour")
        mean: float | None = None
        variance: float | None = None
        previous_hour: float | None = None
        previous_pos: int | None = None
        for _index, row in ordered.iterrows():
            pos = int(row["_pos"])
            obs = float(obs_values[pos])
            conf = float(conf_values[pos])
            obs_var = float(np.clip(0.025 + (1.0 - conf) * 0.45, 0.02, 1.0))
            if mean is None or variance is None:
                prior_mean = obs
                prior_var = obs_var
                innovation = 0.0
            else:
                hour = float(row["_hour"])
                delta_hours = max(0.0, hour - (previous_hour if previous_hour is not None else hour))
                previous_drift = drift[previous_pos] if previous_pos is not None else 0.0
                prior_mean = float(np.clip(mean + previous_drift * delta_hours, low, high))
                process_var = (0.006 + 0.015 * abs(previous_drift)) * max(delta_hours, 0.25)
                prior_var = float(np.clip(variance + process_var, 0.01, 2.0))
                innovation = float(obs - prior_mean)

            posterior_var = 1.0 / (1.0 / max(prior_var, 1e-6) + 1.0 / max(obs_var, 1e-6))
            posterior_mean = posterior_var * (prior_mean / max(prior_var, 1e-6) + obs / max(obs_var, 1e-6))
            mean = float(np.clip(posterior_mean, low, high))
            variance = float(np.clip(posterior_var, 0.006, 2.0))
            output[pos, 0] = mean
            output[pos, 1] = float(np.sqrt(max(variance, 1e-6)))
            output[pos, 2] = innovation
            output[pos, 3] = conf
            previous_hour = float(row["_hour"])
            previous_pos = pos
    return pd.DataFrame(output, index=frame.index, columns=columns, dtype=np.float64)


def _tracked_kinetics(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        0.0,
        index=frame.index,
        columns=[
            column
            for var in CORE_TRACKED_LEVELS
            for column in (
                f"immune_belief_{var}_slope_per_hr",
                f"immune_belief_{var}_innovation",
            )
        ],
        dtype=np.float64,
    )
    if frame.empty:
        return output
    work = pd.DataFrame({
        "_hour": _hours(frame),
        "_stay": frame["stay_id"].to_numpy() if "stay_id" in frame else np.arange(len(frame)),
    }, index=frame.index)
    for _stay, group in work.groupby("_stay", sort=False):
        ordered = group.sort_values("_hour")
        previous_values: dict[str, float] = {}
        previous_slopes: dict[str, float] = {}
        previous_hour: float | None = None
        for index, row in ordered.iterrows():
            hour = float(row["_hour"])
            delta_hours = max(0.25, hour - (previous_hour if previous_hour is not None else hour))
            for var in CORE_TRACKED_LEVELS:
                column = f"{var}_t"
                if column not in frame:
                    continue
                value = _coerce_scalar(frame.at[index, column])
                if not np.isfinite(value):
                    continue
                previous_value = previous_values.get(var)
                previous_slope = previous_slopes.get(var, 0.0)
                if previous_value is None:
                    innovation = 0.0
                    slope = 0.0
                else:
                    predicted = previous_value + previous_slope * delta_hours
                    innovation = float(value - predicted)
                    raw_slope = float((value - previous_value) / delta_hours)
                    slope = float(np.clip(0.70 * previous_slope + 0.30 * raw_slope, -25.0, 25.0))
                output.at[index, f"immune_belief_{var}_slope_per_hr"] = slope
                output.at[index, f"immune_belief_{var}_innovation"] = innovation
                previous_values[var] = value
                previous_slopes[var] = slope
            previous_hour = hour
    return output


def immune_inflammatory_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible immune-inflammatory belief features."""

    wbc = _finite_clip(_num(frame, "wbc_t"), 0.1, 80.0, 10.0)
    temperature = _finite_clip(_num(frame, "temperature_t"), 30.0, 43.0, 37.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.2, 25.0, 1.4)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    heart_rate = _finite_clip(_num(frame, "heart_rate_t"), 20.0, 240.0, 85.0)
    respiratory_rate = _finite_clip(_num(frame, "respiratory_rate_t"), 2.0, 80.0, 18.0)
    o2sat = _finite_clip(_num(frame, "o2sat_t"), 35.0, 100.0, 96.0)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.1, 15.0, 1.0)
    platelets = _finite_clip(_num(frame, "platelets_t"), 5.0, 1000.0, 250.0)
    albumin = _finite_clip(_num(frame, "albumin_t"), 0.5, 7.0, 3.5)

    antibiotics = _action_any(frame, "antibiotics")
    fluids = _action_any(frame, "fluids")
    vasopressor = _action_any(frame, "vasopressor")
    steroid = _action_any(frame, "systemic_steroid")

    fever = _clip01((temperature - 38.0) / 3.0)
    hypothermia = _clip01((36.0 - temperature) / 4.0)
    leukocytosis = _clip01((wbc - 12.0) / 24.0)
    leukopenia = _clip01((4.0 - wbc) / 4.0)
    lactate_burden = _clip01((lactate - 2.0) / 8.0)
    low_map = _clip01((65.0 - map_value) / 35.0)
    tachycardia = _clip01((heart_rate - 100.0) / 60.0)
    tachypnea = _clip01((respiratory_rate - 22.0) / 24.0)
    hypoxemia = _clip01((92.0 - o2sat) / 28.0)
    renal_stress = _clip01((creatinine - 1.3) / 3.0)
    thrombocytopenia = _clip01((150.0 - platelets) / 120.0)
    hypoalbuminemia = _clip01((3.0 - albumin) / 1.8)

    inflammatory_observation = np.clip(
        0.24 * fever
        + 0.14 * hypothermia
        + 0.24 * leukocytosis
        + 0.14 * leukopenia
        + 0.12 * thrombocytopenia
        + 0.12 * hypoalbuminemia,
        0.0,
        2.0,
    )
    perfusion_observation = np.clip(
        0.34 * lactate_burden
        + 0.24 * low_map
        + 0.16 * vasopressor
        + 0.12 * renal_stress
        + 0.08 * fluids
        + 0.06 * antibiotics,
        0.0,
        2.0,
    )
    host_response_observation = np.clip(
        0.22 * tachycardia
        + 0.20 * tachypnea
        + 0.16 * hypoxemia
        + 0.14 * fever
        + 0.10 * hypothermia
        + 0.08 * lactate_burden
        + 0.06 * steroid
        + 0.04 * antibiotics,
        0.0,
        2.0,
    )

    inflammatory_confidence = _freshness(frame, ("wbc_age_hr", "temperature_age_hr", "platelets_age_hr", "albumin_age_hr"))
    perfusion_confidence = _freshness(frame, ("lactate_age_hr", "map_age_hr", "creatinine_age_hr"))
    host_confidence = _freshness(frame, ("heart_rate_age_hr", "respiratory_rate_age_hr", "o2sat_age_hr", "temperature_age_hr"))

    kinetics = _tracked_kinetics(frame)
    inflammatory_drift = 0.03 * kinetics["immune_belief_wbc_slope_per_hr"].to_numpy() + 0.05 * kinetics["immune_belief_temperature_slope_per_hr"].to_numpy()
    perfusion_drift = 0.04 * kinetics["immune_belief_lactate_slope_per_hr"].to_numpy() - 0.006 * kinetics["immune_belief_map_slope_per_hr"].to_numpy()
    host_drift = (
        0.008 * kinetics["immune_belief_heart_rate_slope_per_hr"].to_numpy()
        + 0.015 * kinetics["immune_belief_respiratory_rate_slope_per_hr"].to_numpy()
        - 0.010 * kinetics["immune_belief_o2sat_slope_per_hr"].to_numpy()
    )

    states = [
        _scalar_state(
            frame,
            prefix="inflammatory_activation",
            observation=inflammatory_observation,
            confidence=inflammatory_confidence,
            drift_per_hour=inflammatory_drift,
        ),
        _scalar_state(
            frame,
            prefix="septic_perfusion_stress",
            observation=perfusion_observation,
            confidence=perfusion_confidence,
            drift_per_hour=perfusion_drift,
        ),
        _scalar_state(
            frame,
            prefix="host_response_stress",
            observation=host_response_observation,
            confidence=host_confidence,
            drift_per_hour=host_drift,
        ),
        kinetics,
        pd.DataFrame({
            "immune_belief_fever_burden": fever,
            "immune_belief_hypothermia_burden": hypothermia,
            "immune_belief_leukocytosis_burden": leukocytosis,
            "immune_belief_leukopenia_burden": leukopenia,
            "immune_belief_lactate_burden": lactate_burden,
            "immune_belief_low_map_burden": low_map,
            "immune_belief_tachycardia_burden": tachycardia,
            "immune_belief_tachypnea_burden": tachypnea,
            "immune_belief_hypoxemia_burden": hypoxemia,
            "immune_belief_renal_stress_burden": renal_stress,
            "immune_belief_thrombocytopenia_burden": thrombocytopenia,
            "immune_belief_hypoalbuminemia_burden": hypoalbuminemia,
            "immune_belief_antibiotic_context": antibiotics,
            "immune_belief_fluid_context": fluids,
            "immune_belief_vasopressor_context": vasopressor,
            "immune_belief_steroid_context": steroid,
        }, index=frame.index, dtype=np.float64),
    ]
    output = pd.concat(states, axis=1)
    for column in IMMUNE_BELIEF_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output[list(IMMUNE_BELIEF_COLUMNS)].replace([np.inf, -np.inf], np.nan)


def placebo_immune_belief_features(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(frame), len(IMMUNE_BELIEF_COLUMNS))),
        index=frame.index,
        columns=[f"placebo_{column}" for column in IMMUNE_BELIEF_COLUMNS],
        dtype=np.float64,
    )
