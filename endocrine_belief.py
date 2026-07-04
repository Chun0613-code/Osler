"""Patient-specific endocrine / glycemic-stress belief features.

This module extends the validated predict-update personalization pattern to
endocrine-metabolic physiology.  It compresses each patient's own glucose,
osmotic/ketotic, acid-base, electrolyte, hemodynamic, temperature, and observed
treatment context into online-compatible features.

The features are inferred research states, not measured clinical variables.
They can be used only when a downstream observable audit shows improvement
beyond both a baseline ridge model and a capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_TRACKED_LEVELS = (
    "glucose",
    "serum_osmolality",
    "anion_gap",
    "bicarbonate",
    "sodium",
    "potassium",
    "map",
    "temperature",
)

ENDOCRINE_STATE_STEMS = (
    "glycemic_stress",
    "osmotic_ketotic_stress",
    "adrenal_hemodynamic_stress",
)

ENDOCRINE_BELIEF_COLUMNS = tuple(
    column
    for stem in ENDOCRINE_STATE_STEMS
    for column in (
        f"endo_belief_{stem}_mean",
        f"endo_belief_{stem}_sd",
        f"endo_belief_{stem}_innovation",
        f"endo_belief_{stem}_confidence",
    )
) + tuple(
    column
    for var in CORE_TRACKED_LEVELS
    for column in (
        f"endo_belief_{var}_slope_per_hr",
        f"endo_belief_{var}_innovation",
    )
) + (
    "endo_belief_hyperglycemia_burden",
    "endo_belief_hypoglycemia_burden",
    "endo_belief_osmotic_burden",
    "endo_belief_ketotic_gap_burden",
    "endo_belief_low_bicarbonate_burden",
    "endo_belief_hypernatremia_burden",
    "endo_belief_hyponatremia_burden",
    "endo_belief_hyperkalemia_burden",
    "endo_belief_hypokalemia_burden",
    "endo_belief_low_map_burden",
    "endo_belief_fever_burden",
    "endo_belief_hypothermia_burden",
    "endo_belief_insulin_context",
    "endo_belief_dextrose_context",
    "endo_belief_fluid_context",
    "endo_belief_steroid_context",
    "endo_belief_vasopressor_context",
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
        f"endo_belief_{prefix}_mean",
        f"endo_belief_{prefix}_sd",
        f"endo_belief_{prefix}_innovation",
        f"endo_belief_{prefix}_confidence",
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
                f"endo_belief_{var}_slope_per_hr",
                f"endo_belief_{var}_innovation",
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
                value = pd.to_numeric(pd.Series([frame.at[index, column]]), errors="coerce").iloc[0]
                if not np.isfinite(value):
                    continue
                value = float(value)
                previous_value = previous_values.get(var)
                previous_slope = previous_slopes.get(var, 0.0)
                if previous_value is None:
                    innovation = 0.0
                    slope = 0.0
                else:
                    predicted = previous_value + previous_slope * delta_hours
                    innovation = float(value - predicted)
                    raw_slope = float((value - previous_value) / delta_hours)
                    slope = float(np.clip(0.70 * previous_slope + 0.30 * raw_slope, -80.0, 80.0))
                output.at[index, f"endo_belief_{var}_slope_per_hr"] = slope
                output.at[index, f"endo_belief_{var}_innovation"] = innovation
                previous_values[var] = value
                previous_slopes[var] = slope
            previous_hour = hour
    return output


def endocrine_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible endocrine-metabolic belief features."""

    glucose = _finite_clip(_num(frame, "glucose_t"), 20.0, 900.0, 140.0)
    osmolality = _finite_clip(_num(frame, "serum_osmolality_t"), 240.0, 420.0, 295.0)
    ketones = _finite_clip(_num(frame, "serum_ketones_t"), 0.0, 10.0, 0.0)
    anion_gap = _finite_clip(_num(frame, "anion_gap_t"), 0.0, 60.0, 12.0)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 3.0, 50.0, 24.0)
    sodium = _finite_clip(_num(frame, "sodium_t"), 105.0, 180.0, 140.0)
    potassium = _finite_clip(_num(frame, "potassium_t"), 1.5, 9.0, 4.2)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    temperature = _finite_clip(_num(frame, "temperature_t"), 30.0, 43.0, 37.0)

    insulin = _action_any(frame, "insulin")
    dextrose = _action_any(frame, "dextrose")
    fluids = _action_any(frame, "fluids")
    steroid = _action_any(frame, "systemic_steroid")
    vasopressor = _action_any(frame, "vasopressor")

    hyperglycemia = _clip01((glucose - 180.0) / 220.0)
    hypoglycemia = _clip01((80.0 - glucose) / 50.0)
    osmotic = _clip01((osmolality - 300.0) / 60.0)
    ketotic_gap = _clip01(0.55 * _clip01((anion_gap - 14.0) / 18.0) + 0.45 * _clip01(ketones / 4.0))
    low_bicarbonate = _clip01((22.0 - bicarbonate) / 12.0)
    hypernatremia = _clip01((sodium - 146.0) / 14.0)
    hyponatremia = _clip01((132.0 - sodium) / 17.0)
    hyperkalemia = _clip01((potassium - 5.2) / 2.0)
    hypokalemia = _clip01((3.4 - potassium) / 1.4)
    low_map = _clip01((68.0 - map_value) / 28.0)
    fever = _clip01((temperature - 38.2) / 2.6)
    hypothermia = _clip01((35.5 - temperature) / 3.0)

    glycemic_obs = _clip01(
        0.44 * hyperglycemia
        + 0.24 * hypoglycemia
        + 0.10 * osmotic
        + 0.08 * steroid
        + 0.07 * insulin
        + 0.07 * dextrose
    )
    osmotic_ketotic_obs = _clip01(
        0.30 * osmotic
        + 0.28 * ketotic_gap
        + 0.18 * low_bicarbonate
        + 0.12 * hypernatremia
        + 0.07 * hyperkalemia
        + 0.05 * fluids
    )
    adrenal_hemodynamic_obs = _clip01(
        0.30 * low_map
        + 0.18 * fever
        + 0.14 * hypothermia
        + 0.16 * steroid
        + 0.14 * vasopressor
        + 0.08 * hypoglycemia
    )

    glycemic_drift = -0.014 * insulin + 0.010 * dextrose + 0.006 * steroid - 0.004 * fluids
    osmotic_ketotic_drift = -0.010 * insulin - 0.006 * fluids + 0.004 * steroid
    adrenal_hemodynamic_drift = -0.006 * steroid - 0.004 * fluids + 0.004 * vasopressor

    frames = [
        _scalar_state(
            frame,
            prefix="glycemic_stress",
            observation=glycemic_obs,
            confidence=_freshness(frame, ("glucose_age_hr", "serum_osmolality_age_hr")),
            drift_per_hour=glycemic_drift,
        ),
        _scalar_state(
            frame,
            prefix="osmotic_ketotic_stress",
            observation=osmotic_ketotic_obs,
            confidence=_freshness(frame, ("anion_gap_age_hr", "bicarbonate_age_hr", "serum_osmolality_age_hr")),
            drift_per_hour=osmotic_ketotic_drift,
        ),
        _scalar_state(
            frame,
            prefix="adrenal_hemodynamic_stress",
            observation=adrenal_hemodynamic_obs,
            confidence=_freshness(frame, ("map_age_hr", "temperature_age_hr", "glucose_age_hr")),
            drift_per_hour=adrenal_hemodynamic_drift,
        ),
        _tracked_kinetics(frame),
        pd.DataFrame({
            "endo_belief_hyperglycemia_burden": hyperglycemia,
            "endo_belief_hypoglycemia_burden": hypoglycemia,
            "endo_belief_osmotic_burden": osmotic,
            "endo_belief_ketotic_gap_burden": ketotic_gap,
            "endo_belief_low_bicarbonate_burden": low_bicarbonate,
            "endo_belief_hypernatremia_burden": hypernatremia,
            "endo_belief_hyponatremia_burden": hyponatremia,
            "endo_belief_hyperkalemia_burden": hyperkalemia,
            "endo_belief_hypokalemia_burden": hypokalemia,
            "endo_belief_low_map_burden": low_map,
            "endo_belief_fever_burden": fever,
            "endo_belief_hypothermia_burden": hypothermia,
            "endo_belief_insulin_context": insulin,
            "endo_belief_dextrose_context": dextrose,
            "endo_belief_fluid_context": fluids,
            "endo_belief_steroid_context": steroid,
            "endo_belief_vasopressor_context": vasopressor,
        }, index=frame.index),
    ]
    output = pd.concat(frames, axis=1).reindex(columns=ENDOCRINE_BELIEF_COLUMNS)
    return output.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def placebo_endocrine_belief_features(
    frame: pd.DataFrame,
    *,
    seed: int,
    columns: tuple[str, ...] = ENDOCRINE_BELIEF_COLUMNS,
) -> pd.DataFrame:
    """Return capacity-matched random features for placebo control."""

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(len(frame), len(columns)))
    return pd.DataFrame(
        data,
        index=frame.index,
        columns=[f"placebo_{column}" for column in columns],
        dtype=np.float64,
    )

