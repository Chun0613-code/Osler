"""Constrained residual dynamics for the mechanistic DKA ODE.

The network may correct selected concentration derivatives. Conserved pools,
administered-dose depots, volume balance, and injury accumulation remain owned
by DKABody and cannot be modified by the learned residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dka_action_contract import ACTION_KEYS, ACTION_SCALE, expand_action
from dka_world_model_contract import STATE_KEYS


STATE_MEAN = np.asarray(
    [250, 7.2, 15, 18, 4.2, 80, 13, 15, 138, 295, 1.2, 100, 4, 110, 2],
    dtype=np.float32,
)
STATE_STD = np.asarray(
    [200, 0.3, 10, 10, 1.5, 30, 4, 25, 10, 35, 1.5, 180, 4, 35, 5],
    dtype=np.float32,
)
MAINTENANCE_SCALE = np.asarray([500, 500, 100, 100, 20], dtype=np.float32)
MAINTENANCE_KEYS = (
    "_free_water_ml", "_oral_intake_ml", "_enteral_nutrition_ml",
    "_parenteral_nutrition_ml", "_nutrition_carbohydrate_g",
)
RESIDUAL_KEYS = ("G", "Ket", "HCO3", "Ke", "Na", "Cr")
MAX_ABS_RATE = np.asarray([50.0, 1.5, 2.0, 0.45, 1.0, 0.25], dtype=np.float32)
FORBIDDEN_KEYS = (
    "Ki", "V", "I", "insulin_rapid_depot", "insulin_intermediate_depot",
    "insulin_basal_depot", "osmotic_injury", "critical_burdens",
)


def residual_features(observation, action):
    values = np.asarray([
        float(observation.get(name, STATE_MEAN[index]))
        for index, name in enumerate(STATE_KEYS)
    ], dtype=np.float32)
    normalized_state = (values - STATE_MEAN) / STATE_STD
    normalized_action = expand_action(action) / ACTION_SCALE
    maintenance = np.asarray([
        float(action.get(name, 0.0)) if isinstance(action, dict) else 0.0
        for name in MAINTENANCE_KEYS
    ], dtype=np.float32) / MAINTENANCE_SCALE
    return np.concatenate([normalized_state, normalized_action, maintenance])


class ResidualRateNetwork(nn.Module):
    def __init__(self, input_dim=None, bottleneck=8):
        super().__init__()
        input_dim = input_dim or len(STATE_KEYS) + len(ACTION_KEYS) + len(MAINTENANCE_KEYS)
        self.network = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, len(RESIDUAL_KEYS)),
            nn.Tanh(),
        )
        self.register_buffer("bounds", torch.as_tensor(MAX_ABS_RATE))

    def forward(self, features):
        return self.network(features) * self.bounds


def project_residual(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, -MAX_ABS_RATE, MAX_ABS_RATE)
    return dict(zip(RESIDUAL_KEYS, values.astype(float).tolist()))


class GreyBoxResidualRuntime:
    def __init__(self, model, feature_mean, feature_std, metadata=None):
        self.model = model.eval()
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-4)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        model = ResidualRateNetwork(
            input_dim=len(payload["feature_mean"]),
            bottleneck=int(payload["bottleneck"]),
        )
        model.load_state_dict(payload["state_dict"])
        return cls(
            model, payload["feature_mean"], payload["feature_std"],
            metadata=payload.get("metadata", {}),
        )

    def correction(self, observation, action):
        features = residual_features(observation, action)
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            values = self.model(torch.as_tensor(normalized).unsqueeze(0))[0].numpy()
        return project_residual(values)

    def schema(self):
        return {
            "residual_keys": list(RESIDUAL_KEYS),
            "forbidden_keys": list(FORBIDDEN_KEYS),
            "max_abs_rate_per_hour": dict(zip(RESIDUAL_KEYS, MAX_ABS_RATE.tolist())),
            "candidate_only": True,
            "causal_claim_allowed": False,
            **self.metadata,
        }
