"""Patient-specific electrolyte / acid-base belief features.

This module extends the predict-update personalization pattern beyond renal
and cardiovascular physiology.  It builds online-compatible hidden-state
features from each patient's own sodium/water, potassium, acid-base, divalent
mineral, osmotic, renal, and treatment-context trajectory.

The features are inferred research states, not measured labs.  They can be used
only when a downstream observable audit shows improvement beyond both a
population ridge baseline and a capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_TRACKED_LEVELS = (
    "sodium",
    "potassium",
    "bicarbonate",
    "anion_gap",
    "chloride",
    "calcium",
    "magnesium",
    "phosphate",
    "serum_osmolality",
    "creatinine",
)

ELECTROLYTE_STATE_STEMS = (
    "sodium_water",
    "potassium_store",
    "acid_base_buffer",
    "divalent_mineral",
    "osmotic_renal",
)

ELECTROLYTE_BELIEF_COLUMNS = tuple(
    column
    for stem in ELECTROLYTE_STATE_STEMS
    for column in (
        f"el_belief_{stem}_mean",
        f"el_belief_{stem}_sd",
        f"el_belief_{stem}_innovation",
        f"el_belief_{stem}_confidence",
    )
) + tuple(
    column
    for var in CORE_TRACKED_LEVELS
    for column in (
        f"el_belief_{var}_slope_per_hr",
        f"el_belief_{var}_innovation",
    )
) + (
    "el_belief_sodium_low_burden",
    "el_belief_sodium_high_burden",
    "el_belief_potassium_low_burden",
    "el_belief_potassium_high_burden",
    "el_belief_acid_burden",
    "el_belief_alkalosis_burden",
    "el_belief_anion_gap_burden",
    "el_belief_chloride_high_burden",
    "el_belief_divalent_depletion_burden",
    "el_belief_osmotic_burden",
    "el_belief_renal_context",
    "el_belief_repletion_context",
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
        f"el_belief_{prefix}_mean",
        f"el_belief_{prefix}_sd",
        f"el_belief_{prefix}_innovation",
        f"el_belief_{prefix}_confidence",
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
                f"el_belief_{var}_slope_per_hr",
                f"el_belief_{var}_innovation",
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
                output.at[index, f"el_belief_{var}_slope_per_hr"] = slope
                output.at[index, f"el_belief_{var}_innovation"] = innovation
                previous_values[var] = value
                previous_slopes[var] = slope
            previous_hour = hour
    return output


def electrolyte_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible electrolyte belief features for each row."""

    sodium = _finite_clip(_num(frame, "sodium_t"), 105.0, 180.0, 140.0)
    potassium = _finite_clip(_num(frame, "potassium_t"), 1.5, 9.0, 4.2)
    chloride = _finite_clip(_num(frame, "chloride_t"), 70.0, 145.0, 103.0)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 3.0, 50.0, 24.0)
    anion_gap = _finite_clip(_num(frame, "anion_gap_t"), 0.0, 45.0, 12.0)
    calcium = _finite_clip(_num(frame, "calcium_t"), 3.0, 16.0, 8.8)
    ionized_calcium = _finite_clip(_num(frame, "ionized_calcium_t"), 0.4, 2.4, 1.15)
    magnesium = _finite_clip(_num(frame, "magnesium_t"), 0.5, 8.0, 2.0)
    phosphate = _finite_clip(_num(frame, "phosphate_t"), 0.3, 14.0, 3.2)
    osmolality = _finite_clip(_num(frame, "serum_osmolality_t"), 220.0, 430.0, 290.0)
    ph = _finite_clip(_num(frame, "ph_t"), 6.7, 7.8, 7.4)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.2, 20.0, 1.2)
    bun = _finite_clip(_num(frame, "bun_t"), 2.0, 220.0, 20.0)
    glucose = _finite_clip(_num(frame, "glucose_t"), 20.0, 1200.0, 140.0)

    fluids = _action_any(frame, "fluids")
    diuretics = _action_any(frame, "diuretics")
    rrt = _action_any(frame, "renal_replacement")
    potassium_repletion = _action_any(frame, "potassium_repletion")
    magnesium_repletion = _action_any(frame, "magnesium_repletion")
    phosphate_repletion = _action_any(frame, "phosphate_repletion")
    calcium_repletion = _action_any(frame, "calcium_repletion")
    bicarbonate_action = _action_any(frame, "bicarbonate")
    insulin = _action_any(frame, "insulin")
    dextrose = _action_any(frame, "dextrose")

    sodium_low = _clip01((132.0 - sodium) / 16.0)
    sodium_high = _clip01((sodium - 148.0) / 18.0)
    potassium_low = _clip01((3.4 - potassium) / 1.2)
    potassium_high = _clip01((potassium - 5.2) / 1.8)
    acid_burden = _clip01((20.0 - bicarbonate) / 12.0) + 0.35 * _clip01((7.32 - ph) / 0.22)
    acid_burden = _clip01(acid_burden)
    alkalosis_burden = _clip01((bicarbonate - 30.0) / 12.0) + 0.25 * _clip01((ph - 7.48) / 0.18)
    alkalosis_burden = _clip01(alkalosis_burden)
    anion_gap_burden = _clip01((anion_gap - 16.0) / 14.0)
    chloride_high = _clip01((chloride - 112.0) / 16.0)
    divalent_depletion = _clip01(
        0.25 * _clip01((8.0 - calcium) / 2.0)
        + 0.25 * _clip01((1.05 - ionized_calcium) / 0.35)
        + 0.25 * _clip01((1.7 - magnesium) / 0.7)
        + 0.25 * _clip01((2.3 - phosphate) / 1.2)
    )
    osmotic_burden = _clip01((osmolality - 300.0) / 55.0) + 0.20 * _clip01((glucose - 220.0) / 380.0)
    osmotic_burden = _clip01(osmotic_burden)
    renal_context = _clip01(0.65 * _clip01((creatinine - 1.5) / 4.5) + 0.35 * _clip01((bun - 35.0) / 95.0) + 0.25 * rrt)
    repletion_context = _clip01(
        0.20 * potassium_repletion
        + 0.20 * magnesium_repletion
        + 0.20 * phosphate_repletion
        + 0.20 * calcium_repletion
        + 0.20 * bicarbonate_action
    )

    sodium_water_obs = _clip01(0.45 * sodium_low + 0.45 * sodium_high + 0.10 * osmotic_burden)
    potassium_obs = _clip01(0.45 * potassium_low + 0.45 * potassium_high + 0.10 * renal_context)
    acid_base_obs = _clip01(0.38 * acid_burden + 0.22 * alkalosis_burden + 0.25 * anion_gap_burden + 0.15 * chloride_high)
    divalent_obs = _clip01(0.75 * divalent_depletion + 0.25 * repletion_context)
    osmotic_renal_obs = _clip01(0.45 * osmotic_burden + 0.35 * renal_context + 0.20 * sodium_high)

    sodium_drift = -0.004 * fluids + 0.004 * diuretics - 0.004 * rrt
    potassium_drift = 0.010 * potassium_repletion - 0.008 * insulin - 0.005 * rrt + 0.003 * renal_context
    acid_drift = -0.009 * bicarbonate_action - 0.005 * rrt + 0.004 * insulin
    divalent_drift = -0.008 * repletion_context + 0.004 * diuretics - 0.003 * rrt
    osmotic_drift = -0.008 * fluids - 0.006 * rrt - 0.004 * insulin + 0.004 * dextrose

    frames = [
        _scalar_state(
            frame,
            prefix="sodium_water",
            observation=sodium_water_obs,
            confidence=_freshness(frame, ("sodium_age_hr", "serum_osmolality_age_hr")),
            drift_per_hour=sodium_drift,
        ),
        _scalar_state(
            frame,
            prefix="potassium_store",
            observation=potassium_obs,
            confidence=_freshness(frame, ("potassium_age_hr", "creatinine_age_hr")),
            drift_per_hour=potassium_drift,
        ),
        _scalar_state(
            frame,
            prefix="acid_base_buffer",
            observation=acid_base_obs,
            confidence=_freshness(frame, ("bicarbonate_age_hr", "anion_gap_age_hr", "ph_age_hr")),
            drift_per_hour=acid_drift,
        ),
        _scalar_state(
            frame,
            prefix="divalent_mineral",
            observation=divalent_obs,
            confidence=_freshness(frame, ("calcium_age_hr", "magnesium_age_hr", "phosphate_age_hr")),
            drift_per_hour=divalent_drift,
        ),
        _scalar_state(
            frame,
            prefix="osmotic_renal",
            observation=osmotic_renal_obs,
            confidence=_freshness(frame, ("serum_osmolality_age_hr", "glucose_age_hr", "creatinine_age_hr")),
            drift_per_hour=osmotic_drift,
        ),
        _tracked_kinetics(frame),
        pd.DataFrame({
            "el_belief_sodium_low_burden": sodium_low,
            "el_belief_sodium_high_burden": sodium_high,
            "el_belief_potassium_low_burden": potassium_low,
            "el_belief_potassium_high_burden": potassium_high,
            "el_belief_acid_burden": acid_burden,
            "el_belief_alkalosis_burden": alkalosis_burden,
            "el_belief_anion_gap_burden": anion_gap_burden,
            "el_belief_chloride_high_burden": chloride_high,
            "el_belief_divalent_depletion_burden": divalent_depletion,
            "el_belief_osmotic_burden": osmotic_burden,
            "el_belief_renal_context": renal_context,
            "el_belief_repletion_context": repletion_context,
        }, index=frame.index),
    ]
    output = pd.concat(frames, axis=1).reindex(columns=ELECTROLYTE_BELIEF_COLUMNS)
    return output.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def placebo_electrolyte_belief_features(
    frame: pd.DataFrame,
    *,
    seed: int,
    columns: tuple[str, ...] = ELECTROLYTE_BELIEF_COLUMNS,
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
