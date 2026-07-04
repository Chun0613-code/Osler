"""Conservative AKI mechanism candidate for factual routing.

This module provides a small, sourced-prior renal accumulation candidate for
the AKI router.  It is deliberately narrow:

* it predicts only creatinine and BUN;
* it does not fit parameters to the eICU cohort;
* it never estimates causal treatment effects;
* it returns NaN for unsupported or unsafe rows so the router can fall back to
  persistence.

The physiological prior is qualitative rather than tuned: impaired urine
output, hypotension/vasopressor context, and renal-replacement context change
the expected six-hour drift of creatinine/BUN.  Constants are conservative
order-of-magnitude priors, not cohort-fitted coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_TARGETS = ("creatinine", "bun")


def _num(frame: pd.DataFrame, column: str, default=np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _num(frame, column, default=0.0)
    return np.isfinite(values) & (values > 0.0)


def renal_stress_index(frame: pd.DataFrame) -> np.ndarray:
    """Return a bounded renal stress proxy in [0, 1].

    This is not a learned latent state.  It is a transparent summary of current
    factual observations: low urine output, low MAP, vasopressor context,
    nephrotoxin context, and already elevated creatinine/BUN.
    """

    stress = np.zeros(len(frame), dtype=np.float64)

    urine = _num(frame, "urine_output_t")
    stress += np.where(np.isfinite(urine) & (urine < 30.0), 0.30, 0.0)
    stress += np.where(np.isfinite(urine) & (urine < 10.0), 0.15, 0.0)

    map_value = _num(frame, "map_t")
    stress += np.where(np.isfinite(map_value) & (map_value < 65.0), 0.20, 0.0)
    stress += np.where(np.isfinite(map_value) & (map_value < 55.0), 0.10, 0.0)

    stress += np.where(_flag(frame, "hist_vasopressor"), 0.15, 0.0)
    stress += np.where(_flag(frame, "hist_nephrotoxin"), 0.10, 0.0)

    creatinine = _num(frame, "creatinine_t")
    stress += np.where(np.isfinite(creatinine) & (creatinine >= 2.0), 0.10, 0.0)

    bun = _num(frame, "bun_t")
    stress += np.where(np.isfinite(bun) & (bun >= 40.0), 0.05, 0.0)

    return np.clip(stress, 0.0, 1.0)


def predict_aki_mechanism(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Predict a six-hour factual target value from renal mechanism priors.

    The candidate is intentionally modest.  In non-RRT windows it allows slow
    accumulation proportional to renal stress.  In RRT-context windows it allows
    slow clearance.  The router decides whether this source is useful on held-out
    patients; it is never promoted by construction.
    """

    if target not in SUPPORTED_TARGETS:
        return np.full(len(frame), np.nan, dtype=np.float64)

    current = _num(frame, f"{target}_t")
    output = np.full(len(frame), np.nan, dtype=np.float64)
    valid = np.isfinite(current)
    if not valid.any():
        return output

    stress = renal_stress_index(frame)
    rrt = _flag(frame, "hist_renal_replacement") | _flag(frame, "act_renal_replacement")

    if target == "creatinine":
        # Conservative six-hour drift: small rise under stress, small clearance
        # under RRT.  Clipped to avoid implausible mechanistic jumps.
        delta = 0.03 + 0.12 * stress
        delta = np.where(rrt, -0.08 * np.maximum(current - 1.0, 0.0), delta)
        prediction = current + np.clip(delta, -0.50, 0.50)
        output[valid] = np.clip(prediction[valid], 0.05, 30.0)
        return output

    # BUN changes more visibly than creatinine but is still slow at six hours.
    delta = 0.8 + 2.4 * stress
    delta = np.where(rrt, -0.10 * np.maximum(current - 20.0, 0.0), delta)
    prediction = current + np.clip(delta, -20.0, 20.0)
    output[valid] = np.clip(prediction[valid], 1.0, 300.0)
    return output
