"""Patient-specific respiratory / gas-exchange belief features.

This module extends the validated predict-update personalization pattern to
respiratory physiology.  It compresses each patient's own oxygenation,
ventilation/CO2, acid-base compensation, ventilator support, and local
trajectory into online-compatible features.

The features are inferred research states, not measured clinical variables.
They can be used only when a downstream observable audit shows improvement
beyond both a baseline ridge model and a capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_TRACKED_LEVELS = (
    "o2sat",
    "respiratory_rate",
    "paco2",
    "ph",
    "bicarbonate",
    "minute_ventilation",
)

RESPIRATORY_STATE_STEMS = (
    "oxygenation",
    "ventilation_co2",
    "acid_base_compensation",
)

RESPIRATORY_BELIEF_COLUMNS = tuple(
    column
    for stem in RESPIRATORY_STATE_STEMS
    for column in (
        f"resp_belief_{stem}_mean",
        f"resp_belief_{stem}_sd",
        f"resp_belief_{stem}_innovation",
        f"resp_belief_{stem}_confidence",
    )
) + tuple(
    column
    for var in CORE_TRACKED_LEVELS
    for column in (
        f"resp_belief_{var}_slope_per_hr",
        f"resp_belief_{var}_innovation",
    )
) + (
    "resp_belief_hypoxemia_burden",
    "resp_belief_tachypnea_burden",
    "resp_belief_bradypnea_burden",
    "resp_belief_co2_retention_burden",
    "resp_belief_low_co2_burden",
    "resp_belief_acidemia_burden",
    "resp_belief_alkalemia_burden",
    "resp_belief_fio2_support",
    "resp_belief_peep_support",
    "resp_belief_minute_ventilation_burden",
    "resp_belief_ventilation_context",
    "resp_belief_bronchodilator_context",
    "resp_belief_steroid_context",
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
        f"resp_belief_{prefix}_mean",
        f"resp_belief_{prefix}_sd",
        f"resp_belief_{prefix}_innovation",
        f"resp_belief_{prefix}_confidence",
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
                f"resp_belief_{var}_slope_per_hr",
                f"resp_belief_{var}_innovation",
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
                    slope = float(np.clip(0.70 * previous_slope + 0.30 * raw_slope, -20.0, 20.0))
                output.at[index, f"resp_belief_{var}_slope_per_hr"] = slope
                output.at[index, f"resp_belief_{var}_innovation"] = innovation
                previous_values[var] = value
                previous_slopes[var] = slope
            previous_hour = hour
    return output


def respiratory_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible respiratory belief features for each row."""

    o2sat = _finite_clip(_num(frame, "o2sat_t"), 35.0, 100.0, 96.0)
    rr = _finite_clip(_num(frame, "respiratory_rate_t"), 2.0, 80.0, 18.0)
    paco2 = _finite_clip(_num(frame, "paco2_t"), 5.0, 200.0, 40.0)
    fio2 = _finite_clip(_num(frame, "fio2_t"), 20.0, 100.0, 21.0)
    peep = _finite_clip(_num(frame, "peep_t"), 0.0, 40.0, 0.0)
    minute_ventilation = _finite_clip(_num(frame, "minute_ventilation_t"), 0.5, 40.0, 7.0)
    ph = _finite_clip(_num(frame, "ph_t"), 6.7, 7.8, 7.4)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 3.0, 50.0, 24.0)
    invasive = (_num(frame, "vent_mode_invasive_t", default=0.0) > 0.0).astype(np.float64)
    noninvasive = (_num(frame, "vent_mode_noninvasive_t", default=0.0) > 0.0).astype(np.float64)

    ventilation = _action_any(frame, "ventilation")
    bronchodilator = _action_any(frame, "bronchodilator")
    steroid = _action_any(frame, "systemic_steroid")
    vasopressor = _action_any(frame, "vasopressor")

    hypoxemia = _clip01((92.0 - o2sat) / 28.0)
    tachypnea = _clip01((rr - 24.0) / 24.0)
    bradypnea = _clip01((10.0 - rr) / 8.0)
    co2_retention = _clip01((paco2 - 48.0) / 32.0)
    low_co2 = _clip01((32.0 - paco2) / 18.0)
    acidemia = _clip01((7.32 - ph) / 0.30)
    alkalemia = _clip01((ph - 7.48) / 0.22)
    fio2_support = _clip01((fio2 - 30.0) / 50.0)
    peep_support = _clip01(peep / 16.0)
    minute_ventilation_burden = _clip01((minute_ventilation - 12.0) / 14.0) + 0.25 * _clip01((4.0 - minute_ventilation) / 3.0)
    minute_ventilation_burden = _clip01(minute_ventilation_burden)
    ventilation_context = _clip01(0.55 * ventilation + 0.30 * invasive + 0.15 * noninvasive)

    oxygenation_obs = _clip01(
        0.40 * hypoxemia
        + 0.20 * fio2_support
        + 0.20 * peep_support
        + 0.10 * ventilation_context
        + 0.10 * vasopressor
    )
    ventilation_obs = _clip01(
        0.35 * tachypnea
        + 0.15 * bradypnea
        + 0.25 * co2_retention
        + 0.10 * low_co2
        + 0.15 * ventilation_context
    )
    acid_base_obs = _clip01(
        0.25 * acidemia
        + 0.20 * alkalemia
        + 0.25 * co2_retention
        + 0.15 * low_co2
        + 0.15 * _clip01((22.0 - bicarbonate) / 12.0)
    )

    oxygenation_drift = -0.010 * ventilation - 0.006 * peep_support - 0.004 * bronchodilator + 0.004 * vasopressor
    ventilation_drift = -0.008 * ventilation - 0.004 * bronchodilator + 0.003 * steroid
    acid_base_drift = -0.006 * ventilation + 0.004 * steroid

    frames = [
        _scalar_state(
            frame,
            prefix="oxygenation",
            observation=oxygenation_obs,
            confidence=_freshness(frame, ("o2sat_age_hr", "fio2_age_hr", "peep_age_hr")),
            drift_per_hour=oxygenation_drift,
        ),
        _scalar_state(
            frame,
            prefix="ventilation_co2",
            observation=ventilation_obs,
            confidence=_freshness(frame, ("respiratory_rate_age_hr", "paco2_age_hr", "minute_ventilation_age_hr")),
            drift_per_hour=ventilation_drift,
        ),
        _scalar_state(
            frame,
            prefix="acid_base_compensation",
            observation=acid_base_obs,
            confidence=_freshness(frame, ("ph_age_hr", "bicarbonate_age_hr", "paco2_age_hr")),
            drift_per_hour=acid_base_drift,
        ),
        _tracked_kinetics(frame),
        pd.DataFrame({
            "resp_belief_hypoxemia_burden": hypoxemia,
            "resp_belief_tachypnea_burden": tachypnea,
            "resp_belief_bradypnea_burden": bradypnea,
            "resp_belief_co2_retention_burden": co2_retention,
            "resp_belief_low_co2_burden": low_co2,
            "resp_belief_acidemia_burden": acidemia,
            "resp_belief_alkalemia_burden": alkalemia,
            "resp_belief_fio2_support": fio2_support,
            "resp_belief_peep_support": peep_support,
            "resp_belief_minute_ventilation_burden": minute_ventilation_burden,
            "resp_belief_ventilation_context": ventilation_context,
            "resp_belief_bronchodilator_context": bronchodilator,
            "resp_belief_steroid_context": steroid,
        }, index=frame.index),
    ]
    output = pd.concat(frames, axis=1).reindex(columns=RESPIRATORY_BELIEF_COLUMNS)
    return output.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def placebo_respiratory_belief_features(
    frame: pd.DataFrame,
    *,
    seed: int,
    columns: tuple[str, ...] = RESPIRATORY_BELIEF_COLUMNS,
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
