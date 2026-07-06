"""Patient-specific GI / pancreatic / nutrition belief features.

This module adds an online-compatible belief state for gastrointestinal,
pancreatic, and nutrition physiology.  It is intentionally bounded: the state
uses observed labs and treatment context only, and may be used only when a
downstream audit beats both baseline and capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_TRACKED_LEVELS = (
    "lipase",
    "amylase",
    "triglycerides",
    "albumin",
    "total_protein",
    "calcium",
    "glucose",
    "lactate",
    "map",
    "bilirubin",
    "bicarbonate",
    "creatinine",
)

GI_STATE_STEMS = (
    "pancreatic_injury",
    "nutrition_reserve",
    "gut_perfusion_metabolic_stress",
)

GI_BELIEF_COLUMNS = tuple(
    column
    for stem in GI_STATE_STEMS
    for column in (
        f"gi_belief_{stem}_mean",
        f"gi_belief_{stem}_sd",
        f"gi_belief_{stem}_innovation",
        f"gi_belief_{stem}_confidence",
    )
) + tuple(
    column
    for var in CORE_TRACKED_LEVELS
    for column in (
        f"gi_belief_{var}_slope_per_hr",
        f"gi_belief_{var}_innovation",
    )
) + (
    "gi_belief_lipase_burden",
    "gi_belief_amylase_burden",
    "gi_belief_hypertriglyceridemia_burden",
    "gi_belief_hypoalbuminemia_burden",
    "gi_belief_low_total_protein_burden",
    "gi_belief_hypocalcemia_burden",
    "gi_belief_hyperglycemia_burden",
    "gi_belief_lactate_burden",
    "gi_belief_low_map_burden",
    "gi_belief_bilirubin_burden",
    "gi_belief_acid_base_burden",
    "gi_belief_renal_stress_burden",
    "gi_belief_fluids_context",
    "gi_belief_nutrition_context",
    "gi_belief_ppi_context",
    "gi_belief_octreotide_context",
    "gi_belief_albumin_context",
    "gi_belief_antibiotic_context",
    "gi_belief_vasopressor_context",
    "gi_belief_insulin_context",
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
        f"gi_belief_{prefix}_mean",
        f"gi_belief_{prefix}_sd",
        f"gi_belief_{prefix}_innovation",
        f"gi_belief_{prefix}_confidence",
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
                f"gi_belief_{var}_slope_per_hr",
                f"gi_belief_{var}_innovation",
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
                    slope = float(np.clip(0.70 * previous_slope + 0.30 * raw_slope, -80.0, 80.0))
                output.at[index, f"gi_belief_{var}_slope_per_hr"] = slope
                output.at[index, f"gi_belief_{var}_innovation"] = innovation
                previous_values[var] = value
                previous_slopes[var] = slope
            previous_hour = hour
    return output


def gi_nutrition_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible GI/pancreatic/nutrition belief features."""

    lipase = _finite_clip(_num(frame, "lipase_t"), 0.0, 20000.0, 30.0)
    amylase = _finite_clip(_num(frame, "amylase_t"), 0.0, 10000.0, 50.0)
    triglycerides = _finite_clip(_num(frame, "triglycerides_t"), 10.0, 5000.0, 150.0)
    albumin = _finite_clip(_num(frame, "albumin_t"), 0.5, 7.0, 3.5)
    total_protein = _finite_clip(_num(frame, "total_protein_t"), 1.0, 12.0, 6.5)
    calcium = _finite_clip(_num(frame, "calcium_t"), 4.0, 15.0, 8.8)
    glucose = _finite_clip(_num(frame, "glucose_t"), 20.0, 900.0, 140.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.2, 25.0, 1.4)
    map_value = _finite_clip(_num(frame, "map_t"), 20.0, 180.0, 75.0)
    bilirubin = _finite_clip(_num(frame, "bilirubin_t"), 0.0, 50.0, 0.8)
    bicarbonate = _finite_clip(_num(frame, "bicarbonate_t"), 3.0, 50.0, 24.0)
    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.1, 15.0, 1.0)

    fluids = _action_any(frame, "fluids")
    nutrition = _action_any(frame, "nutrition")
    ppi = _action_any(frame, "ppi")
    octreotide = _action_any(frame, "octreotide")
    albumin_tx = _action_any(frame, "albumin")
    antibiotics = _action_any(frame, "antibiotics")
    vasopressor = _action_any(frame, "vasopressor")
    insulin = _action_any(frame, "insulin")

    lipase_burden = _clip01((lipase - 180.0) / 1200.0)
    amylase_burden = _clip01((amylase - 180.0) / 900.0)
    triglyceride_burden = _clip01((triglycerides - 500.0) / 1200.0)
    hypoalbuminemia = _clip01((3.0 - albumin) / 1.8)
    low_total_protein = _clip01((5.5 - total_protein) / 2.5)
    hypocalcemia = _clip01((8.0 - calcium) / 2.0)
    hyperglycemia = _clip01((glucose - 200.0) / 250.0)
    lactate_burden = _clip01((lactate - 2.0) / 8.0)
    low_map = _clip01((65.0 - map_value) / 35.0)
    bilirubin_burden = _clip01((bilirubin - 2.0) / 8.0)
    acid_base_burden = _clip01((20.0 - bicarbonate) / 12.0)
    renal_stress = _clip01((creatinine - 1.3) / 3.0)

    pancreatic_observation = np.clip(
        0.34 * lipase_burden
        + 0.24 * amylase_burden
        + 0.20 * triglyceride_burden
        + 0.10 * hypocalcemia
        + 0.06 * lactate_burden
        + 0.06 * fluids,
        0.0,
        2.0,
    )
    nutrition_observation = np.clip(
        0.34 * hypoalbuminemia
        + 0.24 * low_total_protein
        + 0.16 * hypocalcemia
        + 0.10 * nutrition
        + 0.08 * albumin_tx
        + 0.08 * ppi,
        0.0,
        2.0,
    )
    gut_perfusion_observation = np.clip(
        0.26 * lactate_burden
        + 0.20 * low_map
        + 0.14 * bilirubin_burden
        + 0.12 * acid_base_burden
        + 0.10 * renal_stress
        + 0.08 * vasopressor
        + 0.04 * antibiotics
        + 0.03 * octreotide
        + 0.03 * insulin,
        0.0,
        2.0,
    )

    pancreatic_confidence = _freshness(frame, ("lipase_age_hr", "amylase_age_hr", "triglycerides_age_hr", "calcium_age_hr"))
    nutrition_confidence = _freshness(frame, ("albumin_age_hr", "total_protein_age_hr", "calcium_age_hr"))
    gut_confidence = _freshness(frame, ("lactate_age_hr", "map_age_hr", "bilirubin_age_hr", "bicarbonate_age_hr", "creatinine_age_hr"))

    kinetics = _tracked_kinetics(frame)
    pancreatic_drift = (
        0.001 * kinetics["gi_belief_lipase_slope_per_hr"].to_numpy()
        + 0.001 * kinetics["gi_belief_amylase_slope_per_hr"].to_numpy()
        + 0.002 * kinetics["gi_belief_triglycerides_slope_per_hr"].to_numpy()
    )
    nutrition_drift = (
        -0.05 * kinetics["gi_belief_albumin_slope_per_hr"].to_numpy()
        -0.03 * kinetics["gi_belief_total_protein_slope_per_hr"].to_numpy()
        -0.02 * kinetics["gi_belief_calcium_slope_per_hr"].to_numpy()
    )
    gut_drift = (
        0.04 * kinetics["gi_belief_lactate_slope_per_hr"].to_numpy()
        -0.006 * kinetics["gi_belief_map_slope_per_hr"].to_numpy()
        + 0.02 * kinetics["gi_belief_bilirubin_slope_per_hr"].to_numpy()
        -0.02 * kinetics["gi_belief_bicarbonate_slope_per_hr"].to_numpy()
    )

    parts = [
        _scalar_state(
            frame,
            prefix="pancreatic_injury",
            observation=pancreatic_observation,
            confidence=pancreatic_confidence,
            drift_per_hour=pancreatic_drift,
        ),
        _scalar_state(
            frame,
            prefix="nutrition_reserve",
            observation=nutrition_observation,
            confidence=nutrition_confidence,
            drift_per_hour=nutrition_drift,
        ),
        _scalar_state(
            frame,
            prefix="gut_perfusion_metabolic_stress",
            observation=gut_perfusion_observation,
            confidence=gut_confidence,
            drift_per_hour=gut_drift,
        ),
        kinetics,
        pd.DataFrame({
            "gi_belief_lipase_burden": lipase_burden,
            "gi_belief_amylase_burden": amylase_burden,
            "gi_belief_hypertriglyceridemia_burden": triglyceride_burden,
            "gi_belief_hypoalbuminemia_burden": hypoalbuminemia,
            "gi_belief_low_total_protein_burden": low_total_protein,
            "gi_belief_hypocalcemia_burden": hypocalcemia,
            "gi_belief_hyperglycemia_burden": hyperglycemia,
            "gi_belief_lactate_burden": lactate_burden,
            "gi_belief_low_map_burden": low_map,
            "gi_belief_bilirubin_burden": bilirubin_burden,
            "gi_belief_acid_base_burden": acid_base_burden,
            "gi_belief_renal_stress_burden": renal_stress,
            "gi_belief_fluids_context": fluids,
            "gi_belief_nutrition_context": nutrition,
            "gi_belief_ppi_context": ppi,
            "gi_belief_octreotide_context": octreotide,
            "gi_belief_albumin_context": albumin_tx,
            "gi_belief_antibiotic_context": antibiotics,
            "gi_belief_vasopressor_context": vasopressor,
            "gi_belief_insulin_context": insulin,
        }, index=frame.index, dtype=np.float64),
    ]
    output = pd.concat(parts, axis=1)
    for column in GI_BELIEF_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output[list(GI_BELIEF_COLUMNS)].replace([np.inf, -np.inf], np.nan)


def placebo_gi_belief_features(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(frame), len(GI_BELIEF_COLUMNS))),
        index=frame.index,
        columns=[f"placebo_{column}" for column in GI_BELIEF_COLUMNS],
        dtype=np.float64,
    )
