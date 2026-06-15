"""Structured intervention schema and time-aware neural action encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


VALID_ROUTES = {
    "iv", "subcutaneous", "im", "oral", "inhaled", "sublingual", "unknown",
}
VALID_INSULIN_FORMULATIONS = {
    "regular", "rapid", "intermediate_nph", "basal", "unknown",
}


@dataclass(frozen=True)
class Intervention:
    name: str
    dose: float
    unit: str
    route: str
    formulation: str | None = None
    start_hour: float = 0.0
    duration_hours: float = 0.5
    indication: str | None = None
    source: str = "simulator"
    confidence: str = "mechanistic"

    def __post_init__(self):
        if self.dose < 0:
            raise ValueError("Intervention dose cannot be negative")
        if self.route not in VALID_ROUTES:
            raise ValueError(f"Unsupported route: {self.route}")
        if self.formulation is not None and self.formulation not in VALID_INSULIN_FORMULATIONS:
            raise ValueError(f"Unsupported insulin formulation: {self.formulation}")
        if self.duration_hours <= 0:
            raise ValueError("Intervention duration must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalActionEncoder(nn.Module):
    """Encode dose plus elapsed time and transition interval.

    Route/formulation are represented by separate DKA action channels before this
    encoder. The structured contract preserves the original administration label.
    """

    def __init__(self, action_dim: int, embedding_dim: int = 32, max_horizon_hours: float = 24.0):
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.max_horizon_hours = max_horizon_hours
        self.network = nn.Sequential(
            nn.Linear(action_dim + 4, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, embedding_dim),
        )

    def forward(self, action, delta_hours=0.5, elapsed_hours=0.0):
        batch_shape = action.shape[:-1]
        device, dtype = action.device, action.dtype

        def feature(value):
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
            return torch.broadcast_to(tensor, batch_shape).unsqueeze(-1)

        delta = feature(delta_hours) / self.max_horizon_hours
        elapsed = feature(elapsed_hours) / self.max_horizon_hours
        phase = elapsed * (2.0 * torch.pi)
        features = torch.cat([action, delta, elapsed, torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.network(features)
