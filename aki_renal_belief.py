"""Interpretable AKI renal belief features for candidate-only audits.

The values in this module are not measured labs.  They are transparent
patient-specific proxies for renal reserve, perfusion stress, azotemia load, and
observation confidence.  A belief is useful only if it improves downstream
observable prediction versus both a baseline and a capacity-matched placebo.
"""

from __future__ import annotations

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


def _num(frame: pd.DataFrame, column: str, default=np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _finite_clip(values: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output)] = fill
    return np.clip(output, low, high)


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


def placebo_belief_features(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return capacity-matched uninformative placebo columns.

    The placebo has the same number of columns as the belief features, but it is
    seeded noise with no access to outcomes.  This controls for the extra ridge
    capacity added by the candidate belief.
    """

    rng = np.random.default_rng(int(seed))
    values = rng.normal(0.0, 1.0, size=(len(frame), len(RENAL_BELIEF_COLUMNS)))
    return pd.DataFrame(values, columns=[f"placebo_{name}" for name in RENAL_BELIEF_COLUMNS], index=frame.index)

