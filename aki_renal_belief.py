"""Interpretable AKI renal belief features for candidate-only audits.

The values in this module are not measured labs.  They are transparent
patient-specific proxies for renal reserve, perfusion stress, azotemia load, and
observation confidence.  A belief is useful only if it improves downstream
observable prediction versus both a baseline and a capacity-matched placebo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from aki_mechanism import renal_stress_index


RENAL_BELIEF_COLUMNS = (
    "belief_renal_reserve",
    "belief_azotemia_load",
    "belief_oliguria",
    "belief_perfusion_deficit",
    "belief_rrt_context",
    "belief_observation_confidence",
)

RENAL_STATE_BELIEF_COLUMNS = (
    "state_belief_renal_reserve_mean",
    "state_belief_renal_reserve_sd",
    "state_belief_renal_stress",
    "state_belief_observation_confidence",
    "state_belief_delta_since_prior",
)


def _num(frame: pd.DataFrame, column: str, default=np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _finite_clip(values: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output)] = fill
    return np.clip(output, low, high)


def _scalar(row: pd.Series, column: str, default: float = np.nan) -> float:
    value = row.get(column, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _flag(row: pd.Series, column: str) -> bool:
    return _scalar(row, column, 0.0) > 0.0


def _row_observation(row: pd.Series) -> tuple[float, float, float, float]:
    """Return reserve observation, variance, stress, and confidence."""

    creatinine = float(np.clip(_scalar(row, "creatinine_t", 1.2), 0.2, 12.0))
    bun = float(np.clip(_scalar(row, "bun_t", 20.0), 2.0, 180.0))
    urine = float(np.clip(_scalar(row, "urine_output_t", 60.0), 0.0, 500.0))
    map_value = float(np.clip(_scalar(row, "map_t", 75.0), 25.0, 160.0))
    creatinine_age = float(np.clip(_scalar(row, "creatinine_age_hr", 24.0), 0.0, 72.0))
    bun_age = float(np.clip(_scalar(row, "bun_age_hr", 24.0), 0.0, 72.0))
    urine_age = float(np.clip(_scalar(row, "urine_output_age_hr", 24.0), 0.0, 72.0))

    creatinine_clearance_proxy = np.clip(1.2 / max(creatinine, 0.2), 0.0, 2.5)
    urine_support = np.clip(urine / 120.0, 0.0, 2.0)
    perfusion_support = np.clip((map_value - 50.0) / 45.0, 0.0, 2.0)
    hist_rrt = 1.0 if _flag(row, "hist_renal_replacement") else 0.0
    hist_vasopressor = 1.0 if _flag(row, "hist_vasopressor") else 0.0
    hist_nephrotoxin = 1.0 if _flag(row, "hist_nephrotoxin") else 0.0
    stress = float(np.clip(
        0.35 * np.clip((30.0 - urine) / 30.0, 0.0, 1.0)
        + 0.25 * np.clip((65.0 - map_value) / 25.0, 0.0, 1.5)
        + 0.15 * hist_vasopressor
        + 0.10 * hist_nephrotoxin
        + 0.10 * np.clip((creatinine - 1.2) / 4.0, 0.0, 1.0)
        + 0.05 * np.clip((bun - 25.0) / 80.0, 0.0, 1.0),
        0.0,
        1.5,
    ))
    confidence = float(np.clip(
        1.0
        - 0.35 * (creatinine_age / 72.0)
        - 0.25 * (bun_age / 72.0)
        - 0.20 * (urine_age / 72.0),
        0.05,
        1.0,
    ))
    reserve = float(np.clip(
        0.45 * creatinine_clearance_proxy
        + 0.25 * urine_support
        + 0.20 * perfusion_support
        + 0.10 * hist_rrt
        - 0.20 * stress,
        0.0,
        2.0,
    ))
    variance = float(np.clip(0.04 + (1.0 - confidence) * 0.40 + stress * 0.08, 0.03, 1.0))
    return reserve, variance, stress, confidence


@dataclass(frozen=True)
class RenalReserveBelief:
    """Gaussian belief over patient-specific renal reserve/GFR proxy.

    This is an interpretable latent belief, not a measured GFR.  It can be used
    only when downstream observable prediction improves versus baseline and
    placebo.
    """

    mean: float
    variance: float
    source: str = "renal_observation_prior"

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(float(self.variance), 1e-6))

    @classmethod
    def from_row(cls, row: pd.Series) -> "RenalReserveBelief":
        mean, variance, _stress, _confidence = _row_observation(row)
        return cls(mean=mean, variance=variance, source="current_observation")

    def predict(self, row: pd.Series, delta_hours: float) -> "RenalReserveBelief":
        hours = max(0.0, float(delta_hours))
        _reserve, _variance, stress, _confidence = _row_observation(row)
        hist_rrt = 1.0 if _flag(row, "hist_renal_replacement") else 0.0
        drift = (-0.010 * stress + 0.006 * hist_rrt) * hours
        mean = float(np.clip(self.mean + drift, 0.0, 2.0))
        process_variance = (0.006 + 0.018 * stress + 0.004 * hist_rrt) * max(hours, 0.25)
        return RenalReserveBelief(
            mean=mean,
            variance=float(np.clip(self.variance + process_variance, 0.03, 2.0)),
            source="predicted_from_previous_belief",
        )

    def update(self, row: pd.Series) -> "RenalReserveBelief":
        observation, observation_variance, _stress, _confidence = _row_observation(row)
        prior_var = max(float(self.variance), 1e-6)
        obs_var = max(float(observation_variance), 1e-6)
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        posterior_mean = posterior_var * (self.mean / prior_var + observation / obs_var)
        return RenalReserveBelief(
            mean=float(np.clip(posterior_mean, 0.0, 2.0)),
            variance=float(np.clip(posterior_var, 0.01, 2.0)),
            source="predict_update_observation",
        )

    def step(self, row: pd.Series, delta_hours: float) -> "RenalReserveBelief":
        return self.predict(row, delta_hours).update(row)

    def to_features(self, row: pd.Series, prior_mean: float | None = None) -> dict[str, float]:
        _reserve, _variance, stress, confidence = _row_observation(row)
        return {
            "state_belief_renal_reserve_mean": float(self.mean),
            "state_belief_renal_reserve_sd": float(self.standard_deviation),
            "state_belief_renal_stress": float(stress),
            "state_belief_observation_confidence": float(confidence),
            "state_belief_delta_since_prior": (
                float(self.mean - prior_mean) if prior_mean is not None else 0.0
            ),
        }


def renal_belief_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return candidate renal belief features for downstream prediction gates."""

    creatinine = _finite_clip(_num(frame, "creatinine_t"), 0.2, 12.0, 1.2)
    bun = _finite_clip(_num(frame, "bun_t"), 2.0, 180.0, 20.0)
    urine = _finite_clip(_num(frame, "urine_output_t"), 0.0, 500.0, 60.0)
    map_value = _finite_clip(_num(frame, "map_t"), 25.0, 160.0, 75.0)
    creatinine_age = _finite_clip(_num(frame, "creatinine_age_hr"), 0.0, 72.0, 24.0)
    bun_age = _finite_clip(_num(frame, "bun_age_hr"), 0.0, 72.0, 24.0)
    urine_age = _finite_clip(_num(frame, "urine_output_age_hr"), 0.0, 72.0, 24.0)

    stress = renal_stress_index(frame)
    rrt = (
        (_num(frame, "hist_renal_replacement", default=0.0) > 0.0)
        | (_num(frame, "act_renal_replacement", default=0.0) > 0.0)
    ).astype(np.float64)
    vasopressor = (
        (_num(frame, "hist_vasopressor", default=0.0) > 0.0)
        | (_num(frame, "act_vasopressor", default=0.0) > 0.0)
    ).astype(np.float64)

    creatinine_clearance_proxy = np.clip(1.2 / np.maximum(creatinine, 0.2), 0.0, 2.5)
    urine_support = np.clip(urine / 120.0, 0.0, 2.0)
    perfusion_support = np.clip((map_value - 50.0) / 45.0, 0.0, 2.0)
    reserve = np.clip(
        0.45 * creatinine_clearance_proxy
        + 0.25 * urine_support
        + 0.20 * perfusion_support
        + 0.10 * rrt
        - 0.20 * stress,
        0.0,
        2.0,
    )

    azotemia = np.clip(
        0.50 * ((creatinine - 1.0) / 4.0)
        + 0.35 * ((bun - 20.0) / 80.0)
        + 0.15 * stress,
        0.0,
        2.0,
    )
    oliguria = np.clip((30.0 - urine) / 30.0, 0.0, 1.0)
    perfusion_deficit = np.clip((65.0 - map_value) / 25.0, 0.0, 1.5) + 0.25 * vasopressor
    observation_confidence = np.clip(
        1.0
        - 0.35 * (creatinine_age / 72.0)
        - 0.25 * (bun_age / 72.0)
        - 0.20 * (urine_age / 72.0),
        0.0,
        1.0,
    )

    return pd.DataFrame({
        "belief_renal_reserve": reserve,
        "belief_azotemia_load": azotemia,
        "belief_oliguria": oliguria,
        "belief_perfusion_deficit": perfusion_deficit,
        "belief_rrt_context": rrt,
        "belief_observation_confidence": observation_confidence,
    }, index=frame.index)


def renal_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return explicit predict-update renal reserve belief state features."""

    output = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=RENAL_STATE_BELIEF_COLUMNS,
        dtype=np.float64,
    )
    if frame.empty:
        return output
    sort_column = "hours_since_onset" if "hours_since_onset" in frame else None
    for _stay_id, group in frame.groupby("stay_id", sort=False):
        ordered = group.sort_values(sort_column) if sort_column else group
        belief: RenalReserveBelief | None = None
        previous_hour: float | None = None
        previous_row: pd.Series | None = None
        for index, row in ordered.iterrows():
            hour = _scalar(row, "hours_since_onset", 0.0)
            if belief is None:
                belief = RenalReserveBelief.from_row(row)
                prior_mean = None
            else:
                delta_hours = max(0.0, hour - (previous_hour if previous_hour is not None else hour))
                prior = belief.predict(previous_row if previous_row is not None else row, delta_hours)
                prior_mean = prior.mean
                belief = prior.update(row)
            features = belief.to_features(row, prior_mean=prior_mean)
            for column, value in features.items():
                output.loc[index, column] = value
            previous_hour = hour
            previous_row = row
    return output.astype(np.float64)


def placebo_belief_features(
    frame: pd.DataFrame,
    seed: int,
    columns: tuple[str, ...] = RENAL_BELIEF_COLUMNS,
) -> pd.DataFrame:
    """Return capacity-matched uninformative placebo columns.

    The placebo has the same number of columns as the belief features, but it is
    seeded noise with no access to outcomes.  This controls for the extra ridge
    capacity added by the candidate belief.
    """

    rng = np.random.default_rng(int(seed))
    values = rng.normal(0.0, 1.0, size=(len(frame), len(columns)))
    return pd.DataFrame(values, columns=[f"placebo_{name}" for name in columns], index=frame.index)
