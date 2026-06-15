"""
Action-conditioned JEPA world model for the numerical DKA body.

The model predicts physiology after an intervention, not merely the next
observation.  Training utilities in train_intervention_jepa.py supervise both
one-step transitions and open-loop multi-step rollouts from branched,
counterfactual simulator trajectories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dka_body import DKABody, DKAPatientProfile, estimate_potassium_store
from dka_action_contract import (
    ACTION_KEYS as ROUTE_ACTION_KEYS,
    ACTION_SCALE,
    expand_action,
)
from osler_jepa.actions import TemporalActionEncoder
from osler_jepa.actions import TREATMENT_EVENT_DIM, TREATMENT_EVENT_KEYS
from osler_jepa.ontology import OSLER_STATE_ONTOLOGY
from osler_jepa.symbolic import RULE_IDS, schema as symbolic_schema
from dka_world_model_contract import STATE_KEYS as CONTRACT_STATE_KEYS


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

STATE_KEYS = list(CONTRACT_STATE_KEYS)
S_MEAN = np.array(
    [250, 7.2, 15, 18, 4.2, 80, 13, 15.0, 138, 295, 1.2, 100, 4.0, 110, 2.0],
    dtype=np.float32,
)
S_STD = np.array(
    [200, 0.3, 10, 10, 1.5, 30, 4, 25.0, 10, 35, 1.5, 180, 4.0, 35, 5.0],
    dtype=np.float32,
)
S_DIM = len(STATE_KEYS)

ACTION_KEYS = list(ROUTE_ACTION_KEYS)
A_SCALE = ACTION_SCALE.copy()
A_DIM = len(ACTION_KEYS)
ACT_GRID = [
    expand_action(action).tolist() for action in (
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0],
        [10, 0, 0, 0, 0], [20, 0, 0, 0, 0],
        {"insulin_rapid_sc": 8}, {"insulin_rapid_sc": 16},
        {"insulin_intermediate_sc": 20}, {"insulin_intermediate_sc": 40},
        {"insulin_basal_sc": 20}, {"insulin_basal_sc": 40},
        [0, 250, 0, 0, 0], [0, 500, 0, 0, 0],
        [0, 0, 10, 0, 0], [0, 0, 20, 0, 0],
        [0, 0, 0, 25, 0], [0, 0, 0, 50, 0],
        [0, 0, 0, 0, 5], [0, 0, 0, 0, 10], [0, 0, 0, 0, 20],
        [4, 500, 10, 0, 0], [6, 500, 20, 0, 0],
        {"insulin_rapid_sc": 16, "fluids": 250},
    )
]
ACTION_EMBED_DIM = 32
HISTORY_HOURS = 6.0
H_DIM = A_DIM * 2
MAX_OBSERVATION_AGE_HOURS = 24.0

Z_DIM = 48
DT = 0.5
HORIZON = 48


def s2vec(observation):
    defaults = dict(zip(STATE_KEYS, S_MEAN.tolist()))
    defaults.update(observation)
    if "osmolality" not in observation:
        defaults["osmolality"] = 2.0 * defaults["Na"] + defaults["G"] / 18.0
    if "BHB" not in observation:
        defaults["BHB"] = max(0.0, defaults["anion_gap"] - 12.0) * 0.75
    if "K_store" not in observation:
        defaults["K_store"] = estimate_potassium_store(
            defaults["Ke"], defaults["pH"], defaults["creatinine"],
            defaults["urine_output"],
        )
    if "osmotic_injury" not in observation:
        defaults["osmotic_injury"] = 0.0
    values = np.array([defaults[key] for key in STATE_KEYS], dtype=np.float32)
    return (values - S_MEAN) / S_STD


def observation_context(observation, ages=None):
    """Return normalized state, observed-value mask, and observation ages.

    Missing values are imputed by ``s2vec`` but remain explicitly marked as
    unobserved. Ages are hours since measurement and are clipped by the encoder.
    """
    ages = ages or {}
    vector = s2vec(observation)
    mask = np.array(
        [1.0 if key in observation and observation[key] is not None else 0.0
         for key in STATE_KEYS],
        dtype=np.float32,
    )
    age = np.array([
        float(ages.get(key, 0.0 if mask[index] else MAX_OBSERVATION_AGE_HOURS))
        for index, key in enumerate(STATE_KEYS)
    ], dtype=np.float32)
    return vector, mask, np.clip(age, 0.0, MAX_OBSERVATION_AGE_HOURS)


def vec2state(vector):
    values = np.asarray(vector, dtype=np.float32) * S_STD + S_MEAN
    return dict(zip(STATE_KEYS, values.tolist()))


def a2vec(action):
    return expand_action(action) / A_SCALE


def vec2action(vector):
    return (np.asarray(vector, dtype=np.float32) * A_SCALE).tolist()


def treatment_history_features(actions, dt=DT, history_hours=HISTORY_HOURS,
                               durations=None):
    """Encode prior physical action rates as exposure and recency features."""
    values = np.asarray(actions, dtype=np.float32)
    if values.size == 0:
        return np.concatenate([
            np.zeros(A_DIM, dtype=np.float32),
            np.ones(A_DIM, dtype=np.float32),
        ])
    values = values.reshape(-1, values.shape[-1])
    padded = np.zeros((len(values), A_DIM), dtype=np.float32)
    padded[:, :min(values.shape[1], A_DIM)] = values[:, :A_DIM]
    normalized = padded / A_SCALE
    if durations is None:
        durations = np.full(len(normalized), float(dt), dtype=np.float32)
    else:
        durations = np.asarray(durations, dtype=np.float32).reshape(-1)
        if len(durations) != len(normalized):
            raise ValueError("Action history and duration history must align")
    cumulative = np.cumsum(durations[::-1])
    keep = max(1, int(np.searchsorted(cumulative, history_hours, side="left") + 1))
    normalized = normalized[-keep:]
    durations = durations[-keep:]
    exposure = (normalized * durations[:, None]).sum(axis=0) / history_hours
    recency = np.ones(A_DIM, dtype=np.float32)
    for index in range(A_DIM):
        active = np.flatnonzero(normalized[:, index] > 1e-6)
        if len(active):
            elapsed = durations[int(active[-1]) + 1:].sum()
            recency[index] = min(1.0, elapsed / history_hours)
    return np.concatenate([exposure, recency]).astype(np.float32)


def randomized_dka(body, rng=None, randomize_profile=True):
    rng = rng or np.random.default_rng()
    if randomize_profile:
        body.set_profile(DKAPatientProfile.sample(rng))
    else:
        body.reset()
    body.G = float(rng.uniform(260, 720))
    body.Ke = float(rng.uniform(3.2, 6.5))
    body.HCO3 = float(rng.uniform(6, 18))
    body.Ket = float(rng.uniform(5, 18))
    body.V = float(body.volume_setpoint * rng.uniform(0.62, 0.94))
    body.Ki = float(120.0 * body.profile.potassium_store_scale * rng.uniform(0.75, 1.0))
    body.I = float(rng.uniform(body.profile.endogenous_insulin, 8.0))
    body.insulin_rapid_depot = float(rng.uniform(0.0, 12.0)) if rng.random() < 0.25 else 0.0
    body.insulin_intermediate_depot = float(rng.uniform(0.0, 24.0)) if rng.random() < 0.18 else 0.0
    body.insulin_basal_depot = float(rng.uniform(0.0, 35.0)) if rng.random() < 0.25 else 0.0
    body.Na = float(rng.uniform(128.0, 150.0))
    body.Cr = float(np.clip(
        body.profile.baseline_creatinine / body.profile.renal_reserve
        * rng.uniform(0.9, 2.0), 0.4, 5.0,
    ))
    body.urine_output_ml_hr = float(rng.uniform(10.0, 250.0))
    body.osmotic_injury = float(rng.uniform(0.0, 4.0))
    body.t = 0.0
    body.alive = True
    body.death_cause = None
    return body.observe()


def collect(n_roll=2500, seed=SEED):
    """Backward-compatible one-step transition collector."""
    rng = np.random.default_rng(seed)
    body = DKABody(rng=rng)
    states, actions, next_states = [], [], []
    for _ in range(n_roll):
        observation = randomized_dka(body, rng)
        for _ in range(HORIZON):
            action = ACT_GRID[int(rng.integers(len(ACT_GRID)))]
            next_observation, _, dead, _ = body.step(action, dt=DT)
            states.append(s2vec(observation))
            actions.append(a2vec(action))
            next_states.append(s2vec(next_observation))
            observation = next_observation
            if dead:
                break
    return (
        torch.from_numpy(np.asarray(states, dtype=np.float32)),
        torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        torch.from_numpy(np.asarray(next_states, dtype=np.float32)),
    )


def mlp(input_dim, output_dim, hidden=192):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, output_dim),
    )


class LatentPredictor(nn.Module):
    """Residual latent dynamics P(z_t, a_t) -> z_(t+1)."""

    def __init__(self, latent_dim=Z_DIM, action_dim=ACTION_EMBED_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.delta = mlp(latent_dim + action_dim, latent_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, latent_and_action):
        latent = latent_and_action[..., :self.latent_dim]
        delta = self.delta(latent_and_action)
        return latent + torch.tanh(self.residual_scale) * delta


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.AEnc = TemporalActionEncoder(A_DIM, ACTION_EMBED_DIM)
        self.EventEnc = nn.Linear(TREATMENT_EVENT_DIM, ACTION_EMBED_DIM, bias=False)
        nn.init.zeros_(self.EventEnc.weight)
        self.E = mlp(S_DIM, Z_DIM)
        self.HEnc = nn.Sequential(
            nn.Linear(H_DIM, 96, bias=False),
            nn.SiLU(),
            nn.Linear(96, Z_DIM, bias=False),
        )
        self.ObsEnc = nn.Linear(S_DIM * 2, Z_DIM, bias=False)
        nn.init.zeros_(self.ObsEnc.weight)
        self.P = LatentPredictor()
        self.D = mlp(Z_DIM, S_DIM)
        self.R = mlp(Z_DIM, 1, hidden=96)
        symbolic_dim = Z_DIM + ACTION_EMBED_DIM
        self.DirectionHead = mlp(symbolic_dim, S_DIM * 3, hidden=128)
        self.StatusHead = mlp(symbolic_dim, 3, hidden=96)
        self.ProofPathHead = mlp(symbolic_dim, len(RULE_IDS), hidden=96)
        self.RuleProposalHead = mlp(symbolic_dim, S_DIM * 3, hidden=128)
        self.Ebar = mlp(S_DIM, Z_DIM)
        self.Ebar.load_state_dict(self.E.state_dict())
        for parameter in self.Ebar.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def ema(self, tau=0.995):
        for target, online in zip(self.Ebar.parameters(), self.E.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)

    def predict_latent(self, latent, action, delta_hours=DT, elapsed_hours=0.0,
                       treatment_events=None):
        action_embedding = self.AEnc(action, delta_hours, elapsed_hours)
        if treatment_events is not None:
            action_embedding = action_embedding + self.EventEnc(
                treatment_events.to(dtype=action.dtype)
            )
        return self.P(torch.cat([latent, action_embedding], dim=-1))

    def encode_state(self, state, history=None, observation_mask=None,
                     observation_age=None):
        if observation_mask is None:
            observation_mask = torch.ones_like(state)
        if observation_age is None:
            observation_age = torch.zeros_like(state)
        observation_mask = observation_mask.to(dtype=state.dtype)
        observation_age = observation_age.to(dtype=state.dtype)
        # ``state`` contains explicit imputations or belief estimates. Reliability
        # is carried separately by mask and age; training code zeroes synthetically
        # hidden ground truth before calling this method to prevent leakage.
        latent = self.E(state)
        observation_features = torch.cat([
            observation_mask - 1.0,
            observation_age.clamp(0.0, MAX_OBSERVATION_AGE_HOURS)
            / MAX_OBSERVATION_AGE_HOURS,
        ], dim=-1)
        latent = latent + self.ObsEnc(observation_features)
        if history is not None:
            latent = latent + self.HEnc(history)
        return latent

    def predict_step(self, state, action, delta_hours=DT, elapsed_hours=0.0,
                     history=None, observation_mask=None, observation_age=None,
                     treatment_events=None):
        latent = self.encode_state(
            state, history, observation_mask, observation_age
        )
        next_latent = self.predict_latent(
            latent, action, delta_hours, elapsed_hours, treatment_events
        )
        return self.D(next_latent), self.R(next_latent).squeeze(-1), next_latent

    def symbolic_outputs(self, context_latent, action, delta_hours=DT,
                         elapsed_hours=0.0, treatment_events=None):
        """Decode a transition into Osler-readable symbolic predictions."""
        action_embedding = self.AEnc(action, delta_hours, elapsed_hours)
        if treatment_events is not None:
            action_embedding = action_embedding + self.EventEnc(
                treatment_events.to(dtype=action.dtype)
            )
        features = torch.cat([context_latent, action_embedding], dim=-1)
        return {
            "direction_logits": self.DirectionHead(features).reshape(
                *features.shape[:-1], S_DIM, 3
            ),
            "status_logits": self.StatusHead(features),
            "proof_logits": self.ProofPathHead(features),
            "proposal_logits": self.RuleProposalHead(features).reshape(
                *features.shape[:-1], S_DIM, 3
            ),
        }

    def rollout(self, initial_state, action_sequence, initial_history=None,
                observation_mask=None, observation_age=None, delta_hours=None,
                treatment_events=None):
        """Open-loop rollout. action_sequence shape: [batch, time, A_DIM]."""
        latent = self.encode_state(
            initial_state, initial_history, observation_mask, observation_age
        )
        states, risk_logits, latents = [], [], []
        elapsed = initial_state.new_zeros(initial_state.shape[:-1])
        for step in range(action_sequence.shape[1]):
            step_delta = self._rollout_delta(
                delta_hours, step, action_sequence.shape[0], initial_state
            )
            latent = self.predict_latent(
                latent,
                action_sequence[:, step],
                delta_hours=step_delta,
                elapsed_hours=elapsed,
                treatment_events=self._rollout_event(
                    treatment_events, step, action_sequence.shape[0], initial_state
                ),
            )
            elapsed = elapsed + step_delta
            states.append(self.D(latent))
            risk_logits.append(self.R(latent).squeeze(-1))
            latents.append(latent)
        return (
            torch.stack(states, dim=1),
            torch.stack(risk_logits, dim=1),
            torch.stack(latents, dim=1),
        )

    @staticmethod
    def _rollout_delta(delta_hours, step, batch_size, reference):
        if delta_hours is None:
            return reference.new_full((batch_size,), DT)
        values = torch.as_tensor(
            delta_hours, dtype=reference.dtype, device=reference.device
        )
        if values.ndim == 0:
            return values.expand(batch_size)
        if values.ndim == 1:
            return values[step].expand(batch_size)
        return values[:, step]

    @staticmethod
    def _rollout_event(treatment_events, step, batch_size, reference):
        if treatment_events is None:
            return None
        values = torch.as_tensor(
            treatment_events, dtype=reference.dtype, device=reference.device
        )
        if values.ndim == 1:
            return values.expand(batch_size, -1)
        if values.ndim == 2:
            return values[step].expand(batch_size, -1)
        return values[:, step]


def vicreg(latent, gamma=1.0):
    if latent.shape[0] < 2:
        return latent.new_tensor(0.0)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    variance = torch.relu(gamma - std).mean()
    centered = latent - latent.mean(dim=0)
    covariance = centered.T @ centered / max(latent.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diagonal.square().sum() / latent.shape[1]
    return variance + covariance_loss


def train(model, states, actions, next_states, epochs=14, bs=256, device=None):
    """Backward-compatible one-step trainer used by the Osler mechanism demo."""
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    states, actions, next_states = states.to(device), actions.to(device), next_states.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-5
    )
    count = len(states)
    for epoch in range(epochs):
        permutation = torch.randperm(count, device=device)
        total = 0.0
        batches = 0
        for start in range(0, count, bs):
            index = permutation[start:start + bs]
            state, action, target_state = states[index], actions[index], next_states[index]
            latent = model.E(state)
            predicted_latent = model.predict_latent(latent, action)
            with torch.no_grad():
                target_latent = model.Ebar(target_state)
            loss = (
                nn.functional.mse_loss(predicted_latent, target_latent)
                + nn.functional.mse_loss(model.D(latent), state)
                + nn.functional.mse_loss(model.D(predicted_latent), target_state)
                + 0.1 * vicreg(latent)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.ema()
            total += float(loss.detach())
            batches += 1
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                latent_std = model.E(states[: min(2000, count)]).std(dim=0).mean()
            print(
                f"  ep {epoch + 1:2d} loss={total / max(batches, 1):.4f} "
                f"z_std={float(latent_std):.3f}"
            )
    return model


def save_checkpoint(model, path, metadata=None):
    payload = {
        "model_state": model.state_dict(),
        "state_keys": STATE_KEYS,
        "action_keys": ACTION_KEYS,
        "state_mean": S_MEAN,
        "state_std": S_STD,
        "action_scale": A_SCALE,
        "history": {
            "hours": HISTORY_HOURS,
            "features": [
                *[f"exposure_{name}" for name in ACTION_KEYS],
                *[f"recency_{name}" for name in ACTION_KEYS],
            ],
        },
        "observation_contract": {
            "features": ["value", "observed_mask", "age_hours"],
            "max_age_hours": MAX_OBSERVATION_AGE_HOURS,
            "missing_value_imputation": "normalized_population_mean",
        },
        "action_encoder": {
            "type": "route_formulation_channels_plus_time_and_lifecycle_events",
            "embedding_dim": ACTION_EMBED_DIM,
            "insulin_channels": list(ACTION_KEYS[:4]),
            "treatment_event_features": list(TREATMENT_EVENT_KEYS),
        },
        "symbolic_interface": symbolic_schema(
            STATE_KEYS, ACTION_KEYS, OSLER_STATE_ONTOLOGY
        ),
        "metadata": metadata or {},
    }
    torch.save(payload, Path(path))


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = WorldModel().to(device)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        notes = []
        if "ObsEnc.weight" in incompatible.missing_keys:
            notes.append(
                "The observation-context encoder is zero-initialized, so complete "
                "fresh observations preserve legacy checkpoint behavior."
            )
        if "EventEnc.weight" in incompatible.missing_keys:
            notes.append(
                "The treatment-event encoder is zero-initialized, preserving "
                "legacy rate-only checkpoint behavior."
            )
        if any("Head" in key for key in incompatible.missing_keys):
            notes.append("Legacy symbolic heads use their initialized weights.")
        payload["compatibility"] = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "notes": notes,
        }
    model.eval()
    return model, payload


def reward_batch(normalized_state):
    physical = normalized_state * torch.as_tensor(
        S_STD, device=normalized_state.device
    ) + torch.as_tensor(S_MEAN, device=normalized_state.device)
    glucose, ph, potassium, map_value = (
        physical[:, 0], physical[:, 1], physical[:, 4], physical[:, 5]
    )
    drive = (
        ((ph - 7.4) / 0.2).square()
        + ((potassium - 4.2) / 1.5).square()
        + ((glucose - 100) / 200).square()
        + ((map_value - 90) / 30).square()
    )
    dead = (
        (ph < 6.8) | (potassium < 2.5) | (potassium > 7.0)
        | (map_value < 40) | (glucose < 40) | (glucose > 1400)
        | (physical[:, 14] > 12.0)
    )
    return torch.where(dead, torch.full_like(drive, -100.0), 1.0 - drive)


def mpc_action(model, state_vector, horizon=6, candidates=512, gamma=0.97):
    device = next(model.parameters()).device
    grid = torch.as_tensor(np.asarray(ACT_GRID, dtype=np.float32), device=device)
    action_index = torch.randint(len(ACT_GRID), (candidates, horizon), device=device)
    action_sequence = grid[action_index] / torch.as_tensor(A_SCALE, device=device)
    initial = torch.as_tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)
    initial = initial.repeat(candidates, 1)
    predicted, risk_logits, _ = model.rollout(initial, action_sequence)
    total = torch.zeros(candidates, device=device)
    for step in range(horizon):
        death_probability = torch.sigmoid(risk_logits[:, step])
        total += (gamma ** step) * (
            reward_batch(predicted[:, step]) - 20.0 * death_probability
        )
    best = int(total.argmax())
    return ACT_GRID[int(action_index[best, 0])]


if __name__ == "__main__":
    print("Use train_intervention_jepa.py to train the multi-horizon intervention model.")
