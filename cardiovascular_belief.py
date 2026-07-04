"""Patient-specific cardiovascular/perfusion belief features.

This is the second candidate personalization layer after the validated AKI
renal belief state.  It compresses a patient's own MAP, lactate, heart-rate,
and treatment-context trajectory into an online-compatible predict-update
perfusion/shock burden.  The state is not a measured clinical variable and must
not be interpreted directly.  It is useful only if a downstream observable audit
shows that it improves prediction beyond both a population ridge baseline and a
capacity-matched placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from body_temporal_coupling_belief import temporal_perfusion_renal_features


CARDIOVASCULAR_BELIEF_COLUMNS = (
    "cv_belief_shock_burden_mean",
    "cv_belief_shock_burden_sd",
    "cv_belief_shock_burden_observation",
    "cv_belief_shock_burden_innovation",
    "cv_belief_shock_burden_delta_since_prior",
    "cv_belief_shock_burden_observation_confidence",
    "cv_belief_low_map_burden",
    "cv_belief_lactate_burden",
    "cv_belief_tachycardia_burden",
    "cv_belief_vasopressor_context",
    "cv_belief_inotrope_context",
    "cv_belief_fluid_context",
)


def cardiovascular_belief_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online-compatible cardiovascular belief features for each row.

    The implementation reuses the already-tested temporal perfusion builder but
    renames the features into a system-local namespace.  The builder groups by
    stay and orders by anchor time, so each row only uses current and previous
    observations within that stay.
    """

    features = temporal_perfusion_renal_features(frame).rename(
        columns={
            "tc_perfusion_shock_burden_mean": "cv_belief_shock_burden_mean",
            "tc_perfusion_shock_burden_sd": "cv_belief_shock_burden_sd",
            "tc_perfusion_shock_burden_observation": "cv_belief_shock_burden_observation",
            "tc_perfusion_shock_burden_innovation": "cv_belief_shock_burden_innovation",
            "tc_perfusion_shock_burden_delta_since_prior": "cv_belief_shock_burden_delta_since_prior",
            "tc_perfusion_shock_burden_observation_confidence": "cv_belief_shock_burden_observation_confidence",
            "tc_perfusion_low_map_burden": "cv_belief_low_map_burden",
            "tc_perfusion_lactate_burden": "cv_belief_lactate_burden",
            "tc_perfusion_tachycardia_burden": "cv_belief_tachycardia_burden",
            "tc_perfusion_vasopressor_context": "cv_belief_vasopressor_context",
            "tc_perfusion_inotrope_context": "cv_belief_inotrope_context",
            "tc_perfusion_fluid_context": "cv_belief_fluid_context",
        }
    )
    output = features.reindex(columns=CARDIOVASCULAR_BELIEF_COLUMNS)
    return output.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def placebo_cardiovascular_belief_features(
    frame: pd.DataFrame,
    *,
    seed: int,
    columns: tuple[str, ...] = CARDIOVASCULAR_BELIEF_COLUMNS,
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
