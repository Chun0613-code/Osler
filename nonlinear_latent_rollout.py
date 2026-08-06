"""Candidate nonlinear latent state rollout for whole-body factual forecasting.

This module is deliberately separate from the production observation router.
The current router is a useful, conservative local residual model.  This
candidate tests a different hypothesis: a patient trajectory is better
represented by a recurrent latent state that evolves through several
nonlinear transitions before decoding future physiology.

Contract
--------
* Inputs are current observations, observation masks, measurement ages and a
  fixed-length history from the same ICU stay.
* Future treatment is not inferred.  This is an observation/factual model.
* Horizons are supplied as separate transition cohorts and are never exposed
  as future input features.
* Missing current values are imputed from training medians only; their masks
  and ages are explicit inputs.
* The model is a candidate until patient-heldout, hospital-heldout,
  persistence, calibration and rollout-stability gates all pass.

The model uses a bounded residual update around the current state.  This keeps
long rollouts from immediately becoming numerically explosive while allowing
nonlinear state-dependent movement when the data support it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48)


def set_seed(seed: int) -> None:
    """Make candidate experiments reproducible on CPU and CUDA."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def horizon_suffix(hours: int | float) -> str:
    value = float(hours)
    if value.is_integer():
        return f"tp{int(value)}"
    return "tp" + str(value).replace(".", "p")


def _is_current_value(column: str) -> bool:
    return column.endswith("_t") and not column.endswith("_tp")


def state_variables(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return state variables that have both current and age columns."""

    values = []
    for column in frame.columns:
        if not _is_current_value(column):
            continue
        name = column[:-2]
        if f"{name}_age_hr" in frame.columns:
            values.append(name)
    return tuple(sorted(set(values)))


def _choose_stays(frame: pd.DataFrame, max_stays: int | None, seed: int) -> pd.DataFrame:
    if not max_stays or frame["stay_id"].nunique() <= max_stays:
        return frame
    rng = np.random.default_rng(seed)
    stays = frame["stay_id"].drop_duplicates().to_numpy()
    keep = rng.choice(stays, size=max_stays, replace=False)
    return frame[frame["stay_id"].isin(set(keep))].copy()


def _time_hours(frame: pd.DataFrame) -> pd.Series:
    if "t" not in frame.columns:
        return pd.Series(np.arange(len(frame), dtype=np.float64), index=frame.index)
    parsed = pd.to_datetime(frame["t"], errors="coerce")
    numeric = parsed.astype("int64").to_numpy(dtype=np.float64)
    numeric[parsed.isna().to_numpy()] = np.nan
    if np.isfinite(numeric).any():
        origin = np.nanmin(numeric)
        return pd.Series((numeric - origin) / 3.6e12, index=frame.index)
    return pd.Series(np.arange(len(frame), dtype=np.float64), index=frame.index)


def _history_indices(frame: pd.DataFrame, history: int) -> np.ndarray:
    """Build causal history indices within each stay/horizon cohort."""

    history = max(1, int(history))
    out = np.zeros((len(frame), history), dtype=np.int64)
    times = frame["_time_hours"].to_numpy(dtype=np.float64)
    groups = frame.groupby(["_source_horizon", "stay_id"], sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[np.argsort(times[positions], kind="stable")]
        for j, position in enumerate(positions):
            selected = positions[max(0, j - history + 1): j + 1]
            if len(selected) < history:
                selected = np.concatenate([
                    np.full(history - len(selected), selected[0], dtype=np.int64),
                    selected,
                ])
            out[position] = selected
    return out


def load_multihorizon_cohort(
    data_root: str | Path,
    module: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    max_stays_per_horizon: int | None = None,
    history: int = 5,
    seed: int = 0,
    allowed_stays: set[str] | None = None,
    include_treatment_context: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load compatible transition cohorts into one horizon-labelled frame.

    Each horizon file contributes the same current observation contract but a
    single renamed ``future_<target>`` column.  The horizon label is retained
    as metadata and is converted to the number of nonlinear transition steps
    during training.
    """

    root = Path(data_root)
    parts: list[pd.DataFrame] = []
    common_vars: set[str] | None = None
    for horizon in horizons:
        path = root / f"eicu_{module}_transitions_{int(horizon)}h.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if allowed_stays is not None:
            frame = frame[frame["stay_id"].astype(str).isin(allowed_stays)].copy()
            if frame.empty:
                continue
        frame = _choose_stays(frame, max_stays_per_horizon, seed + int(horizon))
        variables = set(state_variables(frame))
        common_vars = variables if common_vars is None else common_vars & variables
        keep = [
            "subject_id", "stay_id", "hospitalid", "t", "onset_hour",
            "hours_since_onset",
        ]
        keep += [f"{v}_t" for v in sorted(variables)]
        keep += [f"{v}_age_hr" for v in sorted(variables)]
        suffix = horizon_suffix(horizon)
        keep += [f"{v}_{suffix}" for v in sorted(variables) if f"{v}_{suffix}" in frame]
        # These columns are created only by the raw-event augmentation pass.
        # They are target-specific: a single scalar next-measurement label
        # would incorrectly teach every head the schedule of one target.
        keep += [
            f"next_measurement_time_hr_{v}"
            for v in sorted(variables)
            if f"next_measurement_time_hr_{v}" in frame
        ]
        keep += [
            f"next_measurement_observed_{v}"
            for v in sorted(variables)
            if f"next_measurement_observed_{v}" in frame
        ]
        # Raw-event future-label provenance.  These fields are optional so
        # legacy cohorts remain loadable, but augmented cohorts can carry the
        # actual event time and a strict/near quality decision into training.
        for prefix in (
            "future_event_time_hr_",
            "future_delta_t_hr_",
            "future_label_quality_",
            "future_label_weight_",
            "future_label_tolerance_strict_hr_",
            "future_label_tolerance_near_hr_",
        ):
            keep += [
                f"{prefix}{v}"
                for v in sorted(variables)
                if f"{prefix}{v}" in frame
            ]
        if include_treatment_context:
            # hist_* is generated from evidence at or before this anchor.
            # act_* describes the later forecast window and remains excluded
            # from a factual context encoder.
            keep += [
                column for column in frame.columns if column.startswith("hist_")
            ]
        keep += [
            "next_measurement_time_label_source"
            for _ in [0]
            if "next_measurement_time_label_source" in frame
        ]
        keep = [column for column in keep if column in frame.columns]
        frame = frame[keep].copy()
        frame["_source_horizon"] = int(horizon)
        for variable in variables:
            future = f"{variable}_{suffix}"
            if future in frame.columns:
                frame.rename(columns={future: f"future_{variable}"}, inplace=True)
            elif f"{variable}_t" in frame.columns:
                frame[f"future_{variable}"] = np.nan
        parts.append(frame)
    if not parts:
        raise FileNotFoundError(
            f"No compatible eicu_{module}_transitions_<h>h.parquet files found in {root}"
        )

    variables = tuple(sorted(common_vars or set()))
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined["_time_hours"] = _time_hours(combined)
    combined.sort_values(["_source_horizon", "stay_id", "_time_hours"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    history_indices = _history_indices(combined, history=history)
    for position in range(history_indices.shape[1]):
        combined[f"_history_{position}"] = history_indices[:, position]
    return combined, variables


@dataclass
class StateScaler:
    """Training-only robust location/scale for state and target values."""

    variables: tuple[str, ...]
    medians: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        variables: Sequence[str],
        rows: np.ndarray,
        method: str = "standard",
    ) -> "StateScaler":
        """Fit location/scale using only the supplied training rows.

        ``robust`` uses median/MAD scaling so one hospital's extreme values
        cannot dominate the shared representation.  It is deliberately a
        global training-only transform: no hospital-specific statistics are
        carried into held-out hospitals.
        """

        if method not in {"standard", "robust"}:
            raise ValueError("method must be 'standard' or 'robust'")
        variables = tuple(variables)
        matrix = frame.iloc[rows][[f"{v}_t" for v in variables]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        medians = np.zeros(matrix.shape[1], dtype=np.float64)
        scales = np.ones(matrix.shape[1], dtype=np.float64)
        for index in range(matrix.shape[1]):
            column = matrix[:, index]
            finite = column[np.isfinite(column)]
            if len(finite):
                medians[index] = float(np.median(finite))
                if method == "robust":
                    spread = float(1.4826 * np.median(np.abs(finite - medians[index])))
                    # MAD is zero for quantized or nearly constant ICU fields;
                    # retain a standard-deviation fallback in that case.
                    if not np.isfinite(spread) or spread < 1e-6:
                        spread = float(np.std(finite))
                else:
                    spread = float(np.std(finite))
                if np.isfinite(spread) and spread >= 1e-6:
                    scales[index] = spread
        return cls(variables, medians.astype(np.float32), scales.astype(np.float32))

    def normalize_current(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = frame[[f"{v}_t" for v in self.variables]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        mask = np.isfinite(values).astype(np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        normalized = (values - self.medians) / self.scales
        ages = frame[[f"{v}_age_hr" for v in self.variables]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        ages = np.nan_to_num(ages, nan=168.0, posinf=168.0, neginf=0.0)
        ages = np.clip(ages, 0.0, 168.0) / 24.0
        return normalized.astype(np.float32), mask, ages.astype(np.float32)

    def normalize_future(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = frame[[f"future_{v}" for v in self.variables]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        mask = np.isfinite(values).astype(np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        normalized = (values - self.medians) / self.scales
        return normalized.astype(np.float32), mask

    def to_dict(self) -> dict[str, object]:
        return {
            "variables": list(self.variables),
            "medians": self.medians.tolist(),
            "scales": self.scales.tolist(),
        }


def _time_features(dt_hours: torch.Tensor, remaining_hours: torch.Tensor) -> torch.Tensor:
    dt = torch.clamp(dt_hours, min=0.05, max=168.0)
    remaining = torch.clamp(remaining_hours, min=0.05, max=168.0)
    return torch.stack([
        torch.log1p(dt) / math.log1p(48.0),
        dt / 6.0,
        torch.log1p(remaining) / math.log1p(48.0),
        torch.sin(torch.log1p(dt)),
        torch.cos(torch.log1p(dt)),
    ], dim=-1)


class NonlinearLatentRolloutNet(nn.Module):
    """GRU history encoder plus bounded nonlinear latent transition."""

    def __init__(
        self,
        n_state: int,
        input_dim: int,
        hidden: int = 96,
        latent: int = 64,
        max_step_delta: float = 0.5,
        n_regimes: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.max_step_delta = float(max_step_delta)
        self.n_regimes = max(1, int(n_regimes))
        self.history_encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden, latent),
            nn.LayerNorm(latent),
            nn.SiLU(),
        )
        transition_input = latent + n_state + 5
        self.regime_gate = nn.Sequential(
            nn.Linear(transition_input, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_regimes),
        )
        self.regime_transitions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transition_input, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent),
                nn.Tanh(),
            )
            for _ in range(self.n_regimes)
        ])
        self.regime_delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transition_input, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_state),
            )
            for _ in range(self.n_regimes)
        ])

    def encode_history(self, observations: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history_encoder(observations)
        return self.to_latent(hidden[-1])

    def rollout(
        self,
        observations: torch.Tensor,
        initial_state: torch.Tensor,
        horizon_hours: torch.Tensor,
        *,
        step_hours: float = 6.0,
        return_diagnostics: bool = False,
    ):
        """Roll latent physiology forward in bounded nonlinear steps."""

        latent = self.encode_history(observations)
        state = initial_state
        remaining = torch.clamp(horizon_hours, min=0.05, max=168.0)
        regime_history: list[torch.Tensor] = []
        max_steps = int(math.ceil(168.0 / max(step_hours, 0.05)))
        for _ in range(max_steps):
            active = remaining > 1e-4
            if not bool(active.any()):
                break
            dt = torch.minimum(remaining, torch.full_like(remaining, float(step_hours)))
            features = _time_features(dt, remaining)
            transition_input = torch.cat([latent, state, features], dim=-1)
            gate = torch.softmax(self.regime_gate(transition_input), dim=-1)
            regime_history.append(gate)
            transition_values = torch.stack(
                [block(transition_input) for block in self.regime_transitions], dim=1
            )
            latent_delta = (gate.unsqueeze(-1) * transition_values).sum(dim=1)
            latent = latent + 0.20 * latent_delta
            delta_input = torch.cat([latent, state, features], dim=-1)
            delta_values = torch.stack(
                [block(delta_input) for block in self.regime_delta_heads], dim=1
            )
            delta = (gate.unsqueeze(-1) * delta_values).sum(dim=1)
            delta = self.max_step_delta * torch.tanh(delta)
            next_state = state + delta
            state = torch.where(active.unsqueeze(-1), next_state, state)
            remaining = torch.where(active, remaining - dt, remaining)
        if not return_diagnostics:
            return state, latent
        if regime_history:
            regimes = torch.stack(regime_history, dim=1)
            entropy = -(regimes * torch.log(regimes.clamp(min=1e-8))).sum(dim=-1)
            usage = regimes.mean(dim=(0, 1))
        else:
            regimes = torch.empty(
                (len(state), 0, self.n_regimes), device=state.device
            )
            entropy = torch.empty((len(state), 0), device=state.device)
            usage = torch.zeros(self.n_regimes, device=state.device)
        return state, latent, {
            "regime_weights": regimes,
            "regime_entropy": entropy,
            "regime_usage": usage,
        }

    def forward(
        self,
        observations: torch.Tensor,
        initial_state: torch.Tensor,
        horizon_hours: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        return self.rollout(
            observations,
            initial_state,
            horizon_hours,
            return_diagnostics=return_diagnostics,
        )


def batch_inputs(
    input_matrix: np.ndarray,
    history_indices: np.ndarray,
    rows: np.ndarray,
) -> torch.Tensor:
    return torch.from_numpy(input_matrix[history_indices[rows]]).float()


def effective_rank(embeddings: np.ndarray) -> float:
    """Return entropy effective rank; low values flag representation collapse."""

    if len(embeddings) < 2:
        return 0.0
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float(singular.sum())
    if total <= 1e-8:
        return 0.0
    probabilities = singular / total
    return float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum()))


def mae_by_target(
    predicted: np.ndarray,
    truth: np.ndarray,
    current: np.ndarray,
    target_mask: np.ndarray,
    variables: Sequence[str],
    horizons: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    report: dict[str, dict[str, float | int | None]] = {}
    for index, variable in enumerate(variables):
        valid = target_mask[:, index].astype(bool) & np.isfinite(current[:, index])
        if not valid.any():
            continue
        errors = np.abs(predicted[valid, index] - truth[valid, index])
        persistence = np.abs(current[valid, index] - truth[valid, index])
        report[variable] = {
            "n": int(valid.sum()),
            "candidate_mae": float(errors.mean()),
            "persistence_mae": float(persistence.mean()),
            "delta_vs_persistence": float(errors.mean() - persistence.mean()),
            "horizon_min": float(np.min(horizons[valid])),
            "horizon_max": float(np.max(horizons[valid])),
        }
    return report


__all__ = [
    "DEFAULT_HORIZONS",
    "NonlinearLatentRolloutNet",
    "StateScaler",
    "effective_rank",
    "horizon_suffix",
    "load_multihorizon_cohort",
    "mae_by_target",
    "set_seed",
    "state_variables",
]
