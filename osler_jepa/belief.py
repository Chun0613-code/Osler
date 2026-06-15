"""Interpretable belief states used by the Osler-JEPA runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from dka_action_contract import ACTION_INDEX, expand_action
from dka_body import estimate_potassium_store


@dataclass(frozen=True)
class PotassiumStoreBelief:
    """Gaussian belief over latent total-body potassium reserve.

    The prediction step accounts for documented replacement and estimated renal
    loss. The update step fuses the mechanistic serum/pH proxy and, when present,
    the JEPA latent-state estimate. This is a research belief, not a measured lab.
    """

    mean: float
    variance: float
    source: str = "mechanistic_prior"

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(self.variance, 1e-6))

    @classmethod
    def from_state(cls, state: dict) -> "PotassiumStoreBelief":
        provided = state.get("K_store")
        if provided is not None and np.isfinite(provided):
            return cls(
                mean=float(np.clip(provided, 40.0, 170.0)),
                variance=16.0,
                source="provided_belief",
            )
        mean = estimate_potassium_store(
            state.get("Ke", 4.0),
            state.get("pH", 7.35),
            state.get("creatinine", 1.2),
            state.get("urine_output", 100.0),
        )
        return cls(mean=mean, variance=100.0)

    def predict(self, action, state: dict, delta_hours: float):
        values = expand_action(action)
        hours = max(0.0, float(delta_hours))
        kcl_rate = float(values[ACTION_INDEX["kcl"]])
        urine_output = max(0.0, float(state.get("urine_output", 100.0)))
        glucose = max(0.0, float(state.get("G", 140.0)))
        creatinine = max(0.1, float(state.get("creatinine", 1.2)))

        replacement = 0.45 * kcl_rate * hours
        osmotic_drive = max(0.0, (glucose - 180.0) / 300.0)
        urine_drive = max(0.0, (urine_output - 30.0) / 120.0)
        renal_clearance = float(np.clip(1.4 / creatinine, 0.25, 1.5))
        estimated_loss = 1.8 * osmotic_drive * urine_drive * renal_clearance * hours
        mean = float(np.clip(
            self.mean + replacement - estimated_loss, 40.0, 170.0
        ))
        process_variance = (2.0 + 2.0 * osmotic_drive + 0.5 * kcl_rate / 10.0) * hours
        return PotassiumStoreBelief(
            mean=mean,
            variance=float(min(400.0, self.variance + process_variance)),
            source="predicted_from_actions_and_renal_loss",
        )

    def update(self, state: dict, model_estimate: float | None = None):
        proxy = estimate_potassium_store(
            state.get("Ke", 4.0),
            state.get("pH", 7.35),
            state.get("creatinine", 1.2),
            state.get("urine_output", 100.0),
        )
        # Serum potassium is an indirect and acid-base-dependent observation of
        # total-body reserve, so its measurement variance remains deliberately high.
        posterior_mean, posterior_variance = _kalman_update(
            self.mean, self.variance, proxy, 64.0
        )
        source = "serum_proxy_update"
        if model_estimate is not None and np.isfinite(model_estimate):
            posterior_mean, posterior_variance = _kalman_update(
                posterior_mean, posterior_variance,
                float(np.clip(model_estimate, 40.0, 170.0)), 36.0,
            )
            source = "serum_proxy_plus_jepa_update"
        return PotassiumStoreBelief(
            mean=float(np.clip(posterior_mean, 40.0, 170.0)),
            variance=float(max(posterior_variance, 1.0)),
            source=source,
        )

    def step(self, action, state: dict, delta_hours: float,
             model_estimate: float | None = None):
        return self.predict(action, state, delta_hours).update(
            state, model_estimate=model_estimate
        )

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "standard_deviation": round(self.standard_deviation, 4),
            "source": self.source,
            "measured": False,
        }


def _kalman_update(prior_mean, prior_variance, observation, observation_variance):
    gain = prior_variance / max(prior_variance + observation_variance, 1e-6)
    mean = prior_mean + gain * (observation - prior_mean)
    variance = (1.0 - gain) * prior_variance
    return mean, variance
