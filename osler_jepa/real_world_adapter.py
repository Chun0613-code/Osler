"""Runtime for the guarded factual DKA real-world residual adapter."""

from __future__ import annotations

import joblib
import numpy as np

from dka_world_model import (
    A_DIM, S_MEAN, S_STD, STATE_KEYS, s2vec,
)
from osler_jepa.symbolic import PHYSICAL_STABLE_THRESHOLDS
from real_world_improvement import apply_temperature, episode_features


AGE_FEATURE_COUNT = 10


def build_features(state, base_normalized, normalized_action_sequence, history,
                   measurement_ages=None, active_dka=True,
                   hours_since_onset=0.0):
    current = s2vec(state)
    history = np.zeros(A_DIM * 2, dtype=np.float32) if history is None else np.asarray(
        history, dtype=np.float32
    )
    exposure, episode = episode_features(
        np.asarray(normalized_action_sequence, dtype=np.float32)
    )
    ages = np.full(AGE_FEATURE_COUNT, 6.0 / 24.0, dtype=np.float32)
    if measurement_ages is not None:
        supplied = np.asarray(measurement_ages, dtype=np.float32)
        ages[:min(len(supplied), AGE_FEATURE_COUNT)] = np.clip(
            supplied[:AGE_FEATURE_COUNT], 0.0, 24.0
        ) / 24.0
    return np.concatenate([
        current,
        np.asarray(base_normalized, dtype=np.float32) - current,
        exposure,
        history,
        episode,
        ages,
        np.asarray([
            float(active_dka),
            min(max(float(hours_since_onset), 0.0), 72.0) / 72.0,
        ], dtype=np.float32),
    ]).astype(np.float32)


class RealWorldAdapter:
    """Apply a state-wise adapter selected by patient-group cross-validation.

    This may correct factual forecasts under an observed treatment sequence. It
    must not be used to claim an intervention's causal effect.
    """

    def __init__(self, artifact_path):
        self.path = str(artifact_path)
        self.payload = joblib.load(artifact_path)

    def calibrate_symbolic(self, probabilities):
        temperature = self.payload["symbolic_calibration"]["temperature"]
        return apply_temperature(np.asarray(probabilities), temperature)

    def apply(self, state, base_normalized, normalized_action_sequence, history=None,
              measurement_ages=None, active_dka=True, hours_since_onset=0.0):
        feature = build_features(
            state, base_normalized, normalized_action_sequence, history,
            measurement_ages, active_dka, hours_since_onset,
        )
        current_normalized = s2vec(state)
        current_physical = current_normalized * S_STD + S_MEAN
        base_physical = np.asarray(base_normalized) * S_STD + S_MEAN
        adjusted = base_physical.copy()
        details = {}
        for name, specification in self.payload["models"].items():
            index = int(specification["target_index"])
            method = specification["method"]
            if method == "persistence":
                values = np.asarray([current_physical[index]])
            elif method == "base_jepa":
                values = np.asarray([base_physical[index]])
            else:
                values = np.asarray([
                    base_physical[index] + float(member["model"].predict(
                        ((feature - member["mean"]) / member["std"])[None, :]
                    )[0]) * S_STD[index]
                    for member in specification["members"]
                ])
            adjusted[index] = values.mean()
            threshold = PHYSICAL_STABLE_THRESHOLDS[index]
            directions = np.where(
                values - current_physical[index] < -threshold, 0,
                np.where(values - current_physical[index] > threshold, 2, 1),
            )
            agreement = max(float(np.mean(directions == value)) for value in (0, 1, 2))
            details[name] = {
                "source": method,
                "prediction": round(float(adjusted[index]), 4),
                "ensemble_std": round(float(values.std()), 4),
                "direction_agreement": round(agreement, 4),
                "abstain": bool(
                    method == "adapter"
                    and agreement < self.payload["symbolic_calibration"][
                        "recommended_min_ensemble_agreement"
                    ]
                ),
            }
        return {
            "research_only": True,
            "forecast_type": "factual_observed-treatment forecast",
            "causal_intervention_claim_allowed": False,
            "state": {
                name: round(float(adjusted[index]), 4)
                for index, name in enumerate(STATE_KEYS)
            },
            "adapter_details": details,
        }
