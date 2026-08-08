"""Pure-JEPA-style factual body-system model with target-specific heads.

This module deliberately has no hand-built belief inputs.  It consumes only
measurement values, observation masks, measurement ages, a causal history
window, and the requested factual horizon.  A shared history encoder produces
the patient representation; every target then has an independent decoder head.

The result is a research checkpoint, not a causal or clinical policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from nonlinear_latent_rollout import (
    DEFAULT_HORIZONS,
    StateScaler,
    _history_indices,
    load_multihorizon_cohort,
    set_seed,
)


def _time_features(horizon: torch.Tensor) -> torch.Tensor:
    horizon = horizon.clamp(min=0.05, max=168.0)
    log_horizon = torch.log1p(horizon)
    return torch.stack(
        [
            log_horizon / math.log1p(48.0),
            horizon / 6.0,
            torch.sin(log_horizon),
            torch.cos(log_horizon),
        ],
        dim=-1,
    )


class PureJEPABodyModel(nn.Module):
    """Shared physiology-history encoder plus independent target heads."""

    def __init__(
        self,
        n_state: int,
        input_dim: int,
        hidden: int = 64,
        latent: int = 48,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.history_encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden + n_state, latent),
            nn.LayerNorm(latent),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        head_input = latent + 1 + 4
        self.target_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_input, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in range(n_state)
            ]
        )

    def encode(self, observations: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history_encoder(observations)
        return self.state_encoder(torch.cat([hidden[-1], current], dim=-1))

    def forward(
        self,
        observations: torch.Tensor,
        current: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encode(observations, current)
        time = _time_features(horizon)
        predictions = []
        for index, head in enumerate(self.target_heads):
            head_input = torch.cat(
                [latent, current[:, index : index + 1], time], dim=-1
            )
            # Residual form keeps a short factual forecast anchored to the
            # observed state while allowing target-specific movement.
            delta = 0.75 * torch.tanh(head(head_input))
            predictions.append(current[:, index : index + 1] + delta)
        return torch.cat(predictions, dim=-1)


def _split_subjects(frame: pd.DataFrame, seed: int, fraction: float = 0.25):
    groups = frame["subject_id"].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * fraction)))
    test_groups = set(groups[:n_test].tolist())
    test = frame["subject_id"].isin(test_groups).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _arrays(frame: pd.DataFrame, variables: Sequence[str], rows: np.ndarray):
    scaler = StateScaler.fit(frame, variables, rows)
    current, mask, ages = scaler.normalize_current(frame)
    future, future_mask = scaler.normalize_future(frame)
    onset = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    onset = np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0)
    onset = np.clip(onset, 0.0, 168.0) / 24.0
    input_matrix = np.concatenate(
        [current, mask, ages, onset[:, None]], axis=1
    ).astype(np.float32)
    history = _history_indices(frame, history=5)
    horizon = frame["_source_horizon"].to_numpy(dtype=np.float32)
    return scaler, input_matrix, current, future, future_mask, history, horizon


def _predict(model, arrays, rows, batch_size=1024):
    _, input_matrix, current, _, _, history, horizon = arrays
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            observations = torch.from_numpy(input_matrix[history[batch]]).float()
            state = torch.from_numpy(current[batch]).float()
            dt = torch.from_numpy(horizon[batch]).float()
            predictions.append(model(observations, state, dt).numpy())
    return np.concatenate(predictions, axis=0)


def _metrics(prediction, arrays, rows, variables):
    scaler, _, current, future, future_mask, _, horizon = arrays
    report = {}
    for index, target in enumerate(variables):
        valid = future_mask[rows, index].astype(bool)
        if valid.sum() < 20:
            continue
        truth = future[rows[valid], index] * scaler.scales[index] + scaler.medians[index]
        candidate = prediction[valid, index] * scaler.scales[index] + scaler.medians[index]
        persistence = current[rows[valid], index] * scaler.scales[index] + scaler.medians[index]
        candidate_error = np.abs(candidate - truth)
        persistence_error = np.abs(persistence - truth)
        report[target] = {
            "n": int(valid.sum()),
            "candidate_mae": float(candidate_error.mean()),
            "persistence_mae": float(persistence_error.mean()),
            "delta_vs_persistence": float(candidate_error.mean() - persistence_error.mean()),
            "horizon_min": float(np.min(horizon[rows[valid]])),
            "horizon_max": float(np.max(horizon[rows[valid]])),
        }
    return report


def train_module(
    data_root: str | Path,
    module: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    seed: int = 7,
    epochs: int = 5,
    batch_size: int = 512,
    hidden: int = 64,
    latent: int = 48,
    max_stays_per_horizon: int | None = 4000,
    checkpoint: str | Path | None = None,
):
    frame, variables = load_multihorizon_cohort(
        data_root,
        module,
        horizons,
        max_stays_per_horizon=max_stays_per_horizon,
        history=5,
        seed=seed,
    )
    train_rows, test_rows = _split_subjects(frame, seed)
    arrays = _arrays(frame, variables, train_rows)
    scaler, input_matrix, current, future, future_mask, history, horizon = arrays
    model = PureJEPABodyModel(
        n_state=len(variables),
        input_dim=input_matrix.shape[1],
        hidden=hidden,
        latent=latent,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    loss_history = []
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(train_rows)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            observations = torch.from_numpy(input_matrix[history[batch]]).float()
            state = torch.from_numpy(current[batch]).float()
            dt = torch.from_numpy(horizon[batch]).float()
            target = torch.from_numpy(future[batch]).float()
            target_mask = torch.from_numpy(future_mask[batch]).float()
            prediction = model(observations, state, dt)
            loss_values = torch.nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=0.5
            )
            loss = (loss_values * target_mask).sum() / target_mask.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        loss_history.append(float(np.mean(losses)) if losses else float("nan"))

    prediction = _predict(model, arrays, test_rows, batch_size=batch_size)
    report = {
        "model": "PureJEPABodyModel",
        "module": module,
        "horizons": list(horizons),
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "train_subjects": int(frame.iloc[train_rows]["subject_id"].nunique()),
        "test_subjects": int(frame.iloc[test_rows]["subject_id"].nunique()),
        "architecture": {
            "measurement_pure": True,
            "history_encoder": "GRU",
            "target_specific_heads": len(variables),
            "hand_beliefs_used": False,
            "causal_claim_allowed": False,
            "promotion_status": "candidate_only",
        },
        "loss_history": loss_history,
        "patient_heldout": _metrics(prediction, arrays, test_rows, variables),
    }
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scaler": scaler.to_dict(),
                "module": module,
                "horizons": list(horizons),
                "variables": list(variables),
                "architecture": report["architecture"],
                "history": 5,
                "input_dim": int(input_matrix.shape[1]),
                "hidden": hidden,
                "latent": latent,
            },
            checkpoint,
        )
        report["checkpoint"] = str(checkpoint.resolve())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--module", required=True)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-stays-per-horizon", type=int, default=4000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="/private/tmp/pure_jepa_body_system.json")
    args = parser.parse_args()
    set_seed(7)
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    result = train_module(
        args.data_root,
        args.module,
        horizons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_stays_per_horizon=args.max_stays_per_horizon,
        checkpoint=args.checkpoint,
    )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
