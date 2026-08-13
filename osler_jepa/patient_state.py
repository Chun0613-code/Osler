"""Patient-specific predict-update state for the anatomical organ layer.

The adapter combines causal measurement history with an optional mechanistic
belief vector.  It may update an organ token, but it has no body or system
projection, so it cannot bypass the formal system <-> organ <-> body graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn


class NeuralBeliefFilter(nn.Module):
    """Compact causal predict-update filter with explicit padded-step masks."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.observation_projection = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.innovation_projection = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.update_gate = nn.Linear(2 * hidden, hidden)
        self.decay_rate = nn.Parameter(torch.full((hidden,), -1.0))
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.SiLU(),
            nn.Linear(max(8, hidden // 2), hidden),
        )

    def forward(
        self,
        observations: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, steps, features]")
        batch_size, steps, _ = observations.shape
        if observation_mask is None:
            observation_mask = torch.ones(
                (batch_size, steps),
                dtype=torch.bool,
                device=observations.device,
            )
        if observation_mask.shape != (batch_size, steps):
            raise ValueError("observation_mask must have shape [batch, steps]")

        time = observations[:, :, -1:].detach()
        belief = observations.new_zeros((batch_size, self.hidden))
        previous_time = time[:, 0]
        has_previous = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=observations.device
        )
        for step in range(steps):
            active = observation_mask[:, step : step + 1].bool()
            current_time = time[:, step]
            delta = torch.where(
                has_previous,
                (current_time - previous_time).abs().clamp(0.0, 1.0),
                torch.zeros_like(current_time),
            )
            decay = torch.exp(-nn.functional.softplus(self.decay_rate) * delta)
            predicted = belief * decay
            observation = self.observation_projection(observations[:, step])
            combined = torch.cat([predicted, observation], dim=-1)
            innovation = torch.tanh(self.innovation_projection(combined))
            gate = torch.sigmoid(self.update_gate(combined))
            updated = predicted + gate * (innovation - predicted)
            belief = torch.where(active, updated, belief)
            previous_time = torch.where(active, current_time, previous_time)
            has_previous = has_previous | active
        uncertainty = nn.functional.softplus(self.uncertainty_head(belief)) + 1e-4
        return belief, uncertainty


class PatientStateAdapter(nn.Module):
    """Fuse neural history and explicit belief into one organ-local state."""

    architecture_contract = {
        "placement": "organ_layer",
        "allowed_output": "organ_token_residual",
        "forbidden_outputs": ("system_token", "body_token"),
        "causal_history_only": True,
    }

    def __init__(
        self,
        *,
        organ_dim: int,
        history_input_dim: int,
        belief_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.organ_dim = int(organ_dim)
        self.history_input_dim = int(history_input_dim)
        self.belief_dim = int(belief_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_filter = NeuralBeliefFilter(history_input_dim, hidden_dim)
        self.organ_projection = nn.Linear(organ_dim, hidden_dim)
        self.belief_projection = nn.Sequential(
            nn.Linear(max(1, belief_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.organ_residual = nn.Linear(hidden_dim, organ_dim)
        nn.init.zeros_(self.organ_residual.weight)
        nn.init.zeros_(self.organ_residual.bias)
        self.residual_gate = nn.Parameter(torch.full((organ_dim,), -2.0))
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        organ_latents: torch.Tensor,
        organ_presence: torch.Tensor,
        patient_history: torch.Tensor,
        patient_history_mask: torch.Tensor,
        patient_belief: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if organ_latents.ndim != 3:
            raise ValueError("organ_latents must have shape [batch, organs, features]")
        batch, organs, organ_dim = organ_latents.shape
        if organ_dim != self.organ_dim:
            raise ValueError("organ latent width does not match adapter")
        expected_history = (batch, organs)
        if patient_history.shape[:2] != expected_history:
            raise ValueError("patient_history must align with batch and organs")
        if patient_history.shape[-1] != self.history_input_dim:
            raise ValueError("patient history feature width does not match adapter")
        if patient_history_mask.shape != patient_history.shape[:3]:
            raise ValueError("patient_history_mask must align with history steps")
        if patient_belief is None:
            patient_belief = organ_latents.new_zeros((batch, organs, max(1, self.belief_dim)))
        if patient_belief.shape != (batch, organs, max(1, self.belief_dim)):
            raise ValueError("patient_belief must align with batch, organs, and belief width")

        flat_history = patient_history.reshape(
            batch * organs,
            patient_history.shape[2],
            self.history_input_dim,
        )
        flat_mask = patient_history_mask.reshape(batch * organs, -1)
        history_state, history_uncertainty = self.history_filter(
            flat_history, flat_mask
        )
        organ_state = self.organ_projection(
            organ_latents.reshape(batch * organs, organ_dim)
        )
        belief_state = self.belief_projection(
            patient_belief.reshape(batch * organs, -1)
        )
        patient_state = self.fusion(
            torch.cat((organ_state, history_state, belief_state), dim=-1)
        )
        residual = torch.tanh(self.organ_residual(patient_state))
        residual = residual * torch.sigmoid(self.residual_gate)
        present = organ_presence.reshape(batch * organs, 1).to(residual.dtype)
        residual = residual * present
        updated = organ_latents + residual.reshape(batch, organs, organ_dim)
        uncertainty = nn.functional.softplus(
            self.uncertainty_head(patient_state.detach())
        ) + history_uncertainty.detach().mean(dim=-1, keepdim=True)
        return {
            "organ_latents": updated,
            "organ_residual": residual.reshape(batch, organs, organ_dim),
            "patient_state": patient_state.reshape(batch, organs, self.hidden_dim),
            "uncertainty": uncertainty.reshape(batch, organs),
        }


class PatientStateResidualForecaster(nn.Module):
    """Bounded organ-local patient residual around a population anchor."""

    def __init__(self, feature_dim: int, belief_dim: int, hidden: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.belief_dim = int(belief_dim)
        self.hidden = int(hidden)
        self.organ_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.patient_state = PatientStateAdapter(
            organ_dim=hidden,
            history_input_dim=feature_dim + 1,
            belief_dim=belief_dim,
            hidden_dim=hidden,
        )
        self.point_head = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        belief: torch.Tensor,
        anchor: torch.Tensor,
        current_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        organ = self.organ_projection(current).unsqueeze(1)
        state = self.patient_state(
            organ_latents=organ,
            organ_presence=torch.ones(
                (len(current), 1), dtype=current.dtype, device=current.device
            ),
            patient_history=history.unsqueeze(1),
            patient_history_mask=history_mask.unsqueeze(1),
            patient_belief=belief.unsqueeze(1),
        )
        patient_token = state["organ_latents"][:, 0]
        residual = 2.5 * torch.tanh(
            self.point_head(
                torch.cat(
                    (patient_token, anchor[:, None], current_target[:, None]),
                    dim=-1,
                )
            )[:, 0]
        )
        scale = nn.functional.softplus(
            self.scale_head(patient_token.detach())[:, 0]
        ) + 0.05
        return anchor + residual, scale


@dataclass(frozen=True)
class LoadedPatientStateArtifact:
    model: PatientStateResidualForecaster
    population_anchor: object
    preprocessing: dict[str, np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class PatientStatePrediction:
    population_point: float
    personalized_point: float
    lower: float
    upper: float
    half_width: float
    coverage: float


def load_patient_state_artifact(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPatientStateArtifact:
    """Load one validated patient-state artifact without patient-level data."""

    directory = Path(directory)
    manifest_path = directory / "MANIFEST.sha256"
    if not manifest_path.exists():
        raise ValueError("patient-state artifact has no integrity manifest")
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        artifact_path = directory / name
        if not artifact_path.is_file():
            raise ValueError(f"patient-state artifact is missing {name}")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"patient-state artifact integrity check failed: {name}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "organ_patient_state_artifact.v1":
        raise ValueError("unsupported patient-state artifact schema")
    payload = torch.load(
        directory / "model.pt",
        map_location=torch.device(device),
        weights_only=True,
    )
    config = payload["model_config"]
    model = PatientStateResidualForecaster(
        feature_dim=int(config["feature_dim"]),
        belief_dim=int(config["belief_dim"]),
        hidden=int(config["hidden"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    with np.load(directory / "preprocessing_and_conformal.npz") as values:
        preprocessing = {name: values[name].copy() for name in values.files}
    required = {
        "current_center",
        "current_scale",
        "belief_center",
        "belief_scale",
        "target_center",
        "target_scale",
        "conformal_alpha",
        "conformal_quantile",
    }
    missing = required.difference(preprocessing)
    if missing:
        raise ValueError(f"patient-state artifact is missing arrays: {sorted(missing)}")
    return LoadedPatientStateArtifact(
        model=model,
        population_anchor=joblib.load(directory / "population_anchor.joblib"),
        preprocessing=preprocessing,
        metadata=metadata,
    )


def predict_patient_state_artifact(
    artifact: LoadedPatientStateArtifact,
    *,
    current_values: np.ndarray,
    history_values: np.ndarray,
    history_elapsed_hours: np.ndarray,
    history_mask: np.ndarray,
    belief_values: np.ndarray,
    current_target: float,
    anchor_hours_since_onset: float | None = None,
    coverage: float = 0.90,
    device: str | torch.device = "cpu",
) -> PatientStatePrediction:
    """Run one validated patient-state artifact with its exact preprocessing."""

    current = np.asarray(current_values, dtype=np.float32).reshape(1, -1)
    history = np.asarray(history_values, dtype=np.float32)
    elapsed = np.asarray(history_elapsed_hours, dtype=np.float32).reshape(-1, 1)
    mask = np.asarray(history_mask, dtype=bool).reshape(1, -1)
    belief = np.asarray(belief_values, dtype=np.float32).reshape(1, -1)
    if history.ndim != 2 or len(history) != len(elapsed):
        raise ValueError("history values and elapsed times must align")
    if history.shape[1] != current.shape[1]:
        raise ValueError("current and history feature widths must match")
    if mask.shape[1] != len(history):
        raise ValueError("history mask must align with history steps")

    prep = artifact.preprocessing
    normalized_current = np.nan_to_num(
        (current - prep["current_center"]) / prep["current_scale"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    normalized_history = np.nan_to_num(
        (history - prep["current_center"]) / prep["current_scale"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    normalized_history = np.concatenate(
        (normalized_history, np.clip(elapsed / 48.0, -1.0, 0.0)), axis=1
    )[None, ...]
    normalized_history[~mask] = 0.0
    normalized_belief = np.nan_to_num(
        (belief - prep["belief_center"]) / prep["belief_scale"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    population = float(artifact.population_anchor.predict(current)[0])
    target_center = float(prep["target_center"][0])
    target_scale = float(prep["target_scale"][0])
    anchor = np.asarray([(population - target_center) / target_scale], dtype=np.float32)
    normalized_target = np.asarray(
        [(float(current_target) - target_center) / target_scale], dtype=np.float32
    )
    model_device = torch.device(device)
    artifact.model.to(model_device).eval()
    with torch.no_grad():
        point, scale = artifact.model(
            torch.from_numpy(normalized_current).to(model_device),
            torch.from_numpy(normalized_history).to(model_device),
            torch.from_numpy(mask).to(model_device),
            torch.from_numpy(normalized_belief).to(model_device),
            torch.from_numpy(anchor).to(model_device),
            torch.from_numpy(normalized_target).to(model_device),
        )
    personalized = float(point.cpu().numpy()[0] * target_scale + target_center)
    physical_scale = float(scale.cpu().numpy()[0] * target_scale)
    alpha = 1.0 - float(coverage)
    alpha_grid = prep["conformal_alpha"].astype(np.float64)
    quantile_grid = prep["conformal_quantile"].astype(np.float64)
    if artifact.metadata.get("conformal_policy") == "temporal_mondrian":
        if anchor_hours_since_onset is None:
            raise ValueError(
                "temporal Mondrian artifact requires anchor_hours_since_onset"
            )
        names = prep["conformal_regime_names"].astype(str)
        by_regime = prep["conformal_quantile_by_regime"].astype(np.float64)
        hour = float(anchor_hours_since_onset)
        regime = (
            "early_0_24h"
            if hour < 24.0
            else "middle_24_72h" if hour < 72.0 else "late_72h_plus"
        )
        matches = np.flatnonzero(names == regime)
        if len(matches) != 1:
            raise ValueError(f"artifact lacks conformal regime: {regime}")
        quantile_grid = by_regime[int(matches[0])]
    quantile = float(quantile_grid[int(np.argmin(np.abs(alpha_grid - alpha)))])
    half_width = max(0.0, physical_scale * quantile)
    return PatientStatePrediction(
        population_point=population,
        personalized_point=personalized,
        lower=personalized - half_width,
        upper=personalized + half_width,
        half_width=half_width,
        coverage=float(coverage),
    )


__all__ = [
    "NeuralBeliefFilter",
    "PatientStateAdapter",
    "PatientStateResidualForecaster",
    "LoadedPatientStateArtifact",
    "PatientStatePrediction",
    "load_patient_state_artifact",
    "predict_patient_state_artifact",
]
