"""Patient-specific musculoskeletal / rhabdomyolysis belief features.

This module adds an online-compatible belief state for muscle-injury,
renal-electrolyte release, and perfusion-clearance stress.  It is bounded to
observed labs, vitals, and factual treatment context.  The features are
candidate research states until a downstream observable audit beats both a
baseline model and a capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_TRACKED_LEVELS = (
    "cpk",
    "myoglobin",
    "ldh",
    "potassium",
    "creatinine",
    "bun",
    "phosphate",
    "calcium",
    "ionized_calcium",
    "bicarbonate",
    "ph",
    "urine_output",
    "map",
    "lactate",
)

MUSCLE_STATE_STEMS = (
    "muscle_injury_burden",
    "renal_electrolyte_release",
    "perfusion_clearance_stress",
)

MUSCULOSKELETAL_BELIEF_COLUMNS = tuple(
    column
    for stem in MUSCLE_STATE_STEMS
    for column in (
        f"muscle_belief_{stem}_mean",
        f"muscle_belief_{stem}_sd",
        f"muscle_belief_{stem}_innovation",
        f"muscle_belief_{stem}_confidence",
    )
) + tuple(
    column
    for var in CORE_TRACKED_LEVELS
    for column in (
        f"muscle_belief_{var}_slope_per_hr",
        f"muscle_belief_{var}_innovation",
    )
) + (
    "muscle_belief_cpk_burden",
    "muscle_belief_myoglobin_burden",
    "muscle_belief_ldh_burden",
    "muscle_belief_hyperkalemia_burden",
    "muscle_belief_renal_stress_burden",
    "muscle_belief_azotemia_burden",
    "muscle_belief_hyperphosphatemia_burden",
    "muscle_belief_hypocalcemia_burden",
    "muscle_belief_acid_base_burden",
    "muscle_belief_oliguria_burden",
    "muscle_belief_low_map_burden",
    "muscle_belief_lactate_burden",
    "muscle_belief_fluids_context",
    "muscle_belief_bicarbonate_context",
    "muscle_belief_renal_replacement_context",
    "muscle_belief_diuretics_context",
    "muscle_belief_calcium_repletion_context",
    "muscle_belief_potassium_repletion_context",
    "muscle_belief_vasopressor_context",
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
        f"muscle_belief_{prefix}_mean",
        f"muscle_belief_{prefix}_sd",
        f"muscle_belief_{prefix}_innovation",
        f"muscle_belief_{prefix}_confidence",
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
                f"muscle_belief_{var}_slope_per_hr",
                f"muscle_belief_{var}_innovation",
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
                    slope = float(np.clip(0.70 * previous_slope + 0.30 * raw_slope, -200.0, 200.0))
                output.at[index, f"muscle_belief_{var}_slope_per_hr"] = slope
                output.at[index, f"muscle_belief_{var}_innovation"] = innovation
                previous_values[var] = value
                previous_slopes[var] = slope
            previous_hour = hour
    return output


def musculoskeletal_rhabdo_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible musculoskeletal/rhabdomyolysis belief features."""

    cpk = _finite_clip(_num(frame, "cpk_t"), 0.0, 250000.0, 120.0)
    myoglobin = _finite_clip(_num(frame, "myoglobin_t"), 0.0, 100000.0, 70.0)
    ldh = _finite_clip(_num(frame, "ldh_t"), 0.0, 20000.0, 250.0)
    potassium = _finite_clip(_num(frame, "potassium_t"), 1.5, 9.0, 4.2)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.1, 20.0, 1.0)
    bun = _finite_clip(_num(frame, "bun_t"), 1.0, 200.0, 18.0)
    phosphate = _finite_clip(_num(frame, "phosphate_t"), 0.5, 15.0, 3.5)
    calcium = _finite_clip(_num(frame, "calcium_t"), 4.0, 15.0, 8.8)
    ionized_calcium = _finite_clip(_num(frame, "ionized_calcium_t"), 0.3, 2.5, 1.15)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 3.0, 50.0, 24.0)
    urine_output = _finite_clip(_num(frame, "urine_output_t"), 0.0, 1000.0, 45.0)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.2, 25.0, 1.4)

    fluids = _action_any(frame, "fluids")
    bicarbonate_tx = _action_any(frame, "bicarbonate")
    renal_replacement = _action_any(frame, "renal_replacement")
    diuretics = _action_any(frame, "diuretics")
    calcium_repletion = _action_any(frame, "calcium_repletion")
    potassium_repletion = _action_any(frame, "potassium_repletion")
    vasopressor = _action_any(frame, "vasopressor")

    cpk_burden = _clip01((cpk - 1000.0) / 9000.0)
    myoglobin_burden = _clip01((myoglobin - 500.0) / 5000.0)
    ldh_burden = _clip01((ldh - 500.0) / 2500.0)
    hyperkalemia = _clip01((potassium - 5.0) / 2.5)
    renal_stress = _clip01((creatinine - 1.3) / 3.0)
    azotemia = _clip01((bun - 25.0) / 50.0)
    hyperphosphatemia = _clip01((phosphate - 5.0) / 3.0)
    hypocalcemia = np.maximum(_clip01((8.0 - calcium) / 2.0), _clip01((1.05 - ionized_calcium) / 0.35))
    acid_base = _clip01((20.0 - bicarbonate) / 12.0)
    oliguria = _clip01((30.0 - urine_output) / 30.0)
    low_map = _clip01((65.0 - map_value) / 35.0)
    lactate_burden = _clip01((lactate - 2.0) / 8.0)

    muscle_observation = np.clip(
        0.36 * cpk_burden
        + 0.26 * myoglobin_burden
        + 0.18 * ldh_burden
        + 0.08 * hyperkalemia
        + 0.05 * hyperphosphatemia
        + 0.04 * hypocalcemia
        + 0.03 * fluids,
        0.0,
        2.0,
    )
    release_observation = np.clip(
        0.20 * cpk_burden
        + 0.18 * myoglobin_burden
        + 0.20 * hyperkalemia
        + 0.16 * hyperphosphatemia
        + 0.12 * hypocalcemia
        + 0.08 * renal_stress
        + 0.04 * calcium_repletion
        + 0.02 * potassium_repletion,
        0.0,
        2.0,
    )
    perfusion_observation = np.clip(
        0.22 * renal_stress
        + 0.18 * azotemia
        + 0.16 * oliguria
        + 0.14 * low_map
        + 0.12 * lactate_burden
        + 0.08 * acid_base
        + 0.04 * renal_replacement
        + 0.03 * vasopressor
        + 0.02 * diuretics
        + 0.01 * bicarbonate_tx,
        0.0,
        2.0,
    )

    muscle_confidence = _freshness(frame, ("cpk_age_hr", "myoglobin_age_hr", "ldh_age_hr", "potassium_age_hr"))
    release_confidence = _freshness(frame, ("potassium_age_hr", "phosphate_age_hr", "calcium_age_hr", "ionized_calcium_age_hr", "creatinine_age_hr"))
    perfusion_confidence = _freshness(frame, ("creatinine_age_hr", "bun_age_hr", "urine_output_age_hr", "map_age_hr", "lactate_age_hr"))

    kinetics = _tracked_kinetics(frame)
    muscle_drift = (
        0.0004 * kinetics["muscle_belief_cpk_slope_per_hr"].to_numpy()
        + 0.0005 * kinetics["muscle_belief_myoglobin_slope_per_hr"].to_numpy()
        + 0.0008 * kinetics["muscle_belief_ldh_slope_per_hr"].to_numpy()
    )
    release_drift = (
        0.08 * kinetics["muscle_belief_potassium_slope_per_hr"].to_numpy()
        + 0.05 * kinetics["muscle_belief_phosphate_slope_per_hr"].to_numpy()
        - 0.04 * kinetics["muscle_belief_calcium_slope_per_hr"].to_numpy()
        + 0.03 * kinetics["muscle_belief_creatinine_slope_per_hr"].to_numpy()
    )
    perfusion_drift = (
        0.05 * kinetics["muscle_belief_creatinine_slope_per_hr"].to_numpy()
        + 0.02 * kinetics["muscle_belief_bun_slope_per_hr"].to_numpy()
        - 0.004 * kinetics["muscle_belief_urine_output_slope_per_hr"].to_numpy()
        - 0.006 * kinetics["muscle_belief_map_slope_per_hr"].to_numpy()
        + 0.03 * kinetics["muscle_belief_lactate_slope_per_hr"].to_numpy()
    )

    parts = [
        _scalar_state(
            frame,
            prefix="muscle_injury_burden",
            observation=muscle_observation,
            confidence=muscle_confidence,
            drift_per_hour=muscle_drift,
        ),
        _scalar_state(
            frame,
            prefix="renal_electrolyte_release",
            observation=release_observation,
            confidence=release_confidence,
            drift_per_hour=release_drift,
        ),
        _scalar_state(
            frame,
            prefix="perfusion_clearance_stress",
            observation=perfusion_observation,
            confidence=perfusion_confidence,
            drift_per_hour=perfusion_drift,
        ),
        kinetics,
        pd.DataFrame({
            "muscle_belief_cpk_burden": cpk_burden,
            "muscle_belief_myoglobin_burden": myoglobin_burden,
            "muscle_belief_ldh_burden": ldh_burden,
            "muscle_belief_hyperkalemia_burden": hyperkalemia,
            "muscle_belief_renal_stress_burden": renal_stress,
            "muscle_belief_azotemia_burden": azotemia,
            "muscle_belief_hyperphosphatemia_burden": hyperphosphatemia,
            "muscle_belief_hypocalcemia_burden": hypocalcemia,
            "muscle_belief_acid_base_burden": acid_base,
            "muscle_belief_oliguria_burden": oliguria,
            "muscle_belief_low_map_burden": low_map,
            "muscle_belief_lactate_burden": lactate_burden,
            "muscle_belief_fluids_context": fluids,
            "muscle_belief_bicarbonate_context": bicarbonate_tx,
            "muscle_belief_renal_replacement_context": renal_replacement,
            "muscle_belief_diuretics_context": diuretics,
            "muscle_belief_calcium_repletion_context": calcium_repletion,
            "muscle_belief_potassium_repletion_context": potassium_repletion,
            "muscle_belief_vasopressor_context": vasopressor,
        }, index=frame.index, dtype=np.float64),
    ]
    output = pd.concat(parts, axis=1)
    for column in MUSCULOSKELETAL_BELIEF_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output[list(MUSCULOSKELETAL_BELIEF_COLUMNS)].replace([np.inf, -np.inf], np.nan)


def placebo_musculoskeletal_belief_features(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(frame), len(MUSCULOSKELETAL_BELIEF_COLUMNS))),
        index=frame.index,
        columns=[f"placebo_{column}" for column in MUSCULOSKELETAL_BELIEF_COLUMNS],
        dtype=np.float64,
    )
