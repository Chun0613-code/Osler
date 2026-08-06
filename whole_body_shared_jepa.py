"""Shared-latent whole-body JEPA candidate.

This is the first real shared encoder: all modules use one GRU/state encoder,
with a module embedding and shared target heads. It is deliberately a
candidate until module-balanced patient/hospital/conformal gates pass. The
existing per-module validated registry remains the runtime authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from nonlinear_latent_rollout import (
    StateScaler,
    _history_indices,
    _time_hours,
    load_multihorizon_cohort,
    set_seed,
)
from sparse_aware_pure_jepa import _arrays, _masked_mean


class SharedWholeBodySparseJEPA(nn.Module):
    def __init__(self, n_state: int, input_dim: int, module_count: int, hidden: int = 64, latent: int = 48):
        super().__init__()
        self.n_state = int(n_state)
        self.input_dim = int(input_dim)
        self.module_count = int(module_count)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.module_embedding = nn.Embedding(module_count, 12)
        self.history_encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden + n_state + 12, latent), nn.LayerNorm(latent), nn.SiLU()
        )
        head_input = latent + 1 + 4
        self.value_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(head_input, hidden), nn.SiLU(), nn.Linear(hidden, 1)) for _ in range(n_state)]
        )
        self.observation_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1)) for _ in range(n_state)]
        )

    @staticmethod
    def _time_features(horizon):
        horizon = horizon.clamp(min=0.05, max=168.0)
        log_horizon = torch.log1p(horizon)
        return torch.stack([log_horizon / np.log1p(48.0), horizon / 6.0, torch.sin(log_horizon), torch.cos(log_horizon)], dim=-1)

    def forward(self, observations, current, horizon, module_id):
        _, hidden = self.history_encoder(observations)
        module = self.module_embedding(module_id.long())
        latent = self.state_encoder(torch.cat([hidden[-1], current, module], dim=-1))
        time = self._time_features(horizon)
        values, observed = [], []
        for index in range(self.n_state):
            head_input = torch.cat([latent, current[:, index:index + 1], time], dim=-1)
            values.append(current[:, index:index + 1] + 0.75 * torch.tanh(self.value_heads[index](head_input)))
            observed.append(self.observation_heads[index](head_input))
        return {"value": torch.cat(values, dim=-1), "observation_logit": torch.cat(observed, dim=-1)}


def load_shared_cohort(data_root, modules, horizons=(1, 3, 6, 12, 24, 48), max_stays_per_horizon=100, seed=2026):
    parts = []
    module_names = []
    all_variables = set()
    for module in modules:
        try:
            frame, variables = load_multihorizon_cohort(
                data_root, module, horizons, max_stays_per_horizon=max_stays_per_horizon,
                history=5, seed=seed,
            )
        except FileNotFoundError:
            continue
        module_id = len(module_names)
        module_names.append(module)
        all_variables.update(variables)
        frame = frame.copy()
        frame["_shared_module"] = module_id
        # History must remain module-local: a sepsis row must not use an AKI
        # row as its previous timestep merely because both belong to the same
        # person.  Keep those namespaced IDs for ordering/history, but retain
        # the original person for every patient-held-out validation split.
        frame["_patient_stay_id"] = frame["stay_id"].astype(str)
        frame["_patient_subject_id"] = frame["subject_id"].astype(str)
        frame["stay_id"] = module + "::" + frame["_patient_stay_id"]
        frame["subject_id"] = module + "::" + frame["_patient_subject_id"]
        parts.append(frame)
    if not parts:
        raise FileNotFoundError("No transition cohorts available for shared model")
    variables = tuple(sorted(all_variables))
    combined = pd.concat(parts, ignore_index=True, sort=False)
    for variable in variables:
        for suffix in ("_t", "_age_hr"):
            column = f"{variable}{suffix}"
            if column not in combined:
                combined[column] = np.nan
        future = f"future_{variable}"
        if future not in combined:
            combined[future] = np.nan
    combined["_time_hours"] = _time_hours(combined)
    combined.sort_values(["_source_horizon", "stay_id", "_time_hours"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    history = _history_indices(combined, history=5)
    for position in range(history.shape[1]):
        combined[f"_history_{position}"] = history[:, position]
    return combined, variables, tuple(module_names)


def _shared_arrays(frame, variables, fit_rows, history=5):
    """Build full-frame arrays with train-only normalization.

    History indices refer to the full frame.  Keeping the full frame here is
    important: slicing before building history can accidentally index a
    training row with a held-out row's position.
    """
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    scaler = StateScaler.fit(frame, variables, fit_rows)
    current, mask, ages = scaler.normalize_current(frame)
    future, future_mask = scaler.normalize_future(frame)
    onset = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    onset = np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0)
    onset = np.clip(onset, 0.0, 168.0) / 24.0
    input_matrix = np.concatenate([current, mask, ages, onset[:, None]], axis=1).astype(np.float32)
    history_indices = _history_indices(frame, history=history)
    horizon = frame["_source_horizon"].to_numpy(dtype=np.float32)
    return scaler, input_matrix, current, future, future_mask, history_indices, horizon


def fit_shared(frame, variables, module_count, rows, seed=7, epochs=1, batch_size=1024):
    set_seed(seed)
    arrays = _shared_arrays(frame, variables, rows)
    _, input_matrix, current, future, future_mask, history, horizon = arrays
    module_ids = frame["_shared_module"].to_numpy(dtype=np.int64)
    model = SharedWholeBodySparseJEPA(len(variables), input_matrix.shape[1], module_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(rows)
        epoch = []
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            output = model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
                torch.from_numpy(module_ids[batch]),
            )
            y = torch.from_numpy(future[batch]).float()
            mask = torch.from_numpy(future_mask[batch]).float()
            value_loss = _masked_mean(nn.functional.smooth_l1_loss(output["value"], y, reduction="none"), mask)
            observation_loss = nn.functional.binary_cross_entropy_with_logits(output["observation_logit"], mask)
            loss = value_loss + 0.2 * observation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch.append(float(loss.detach()))
        losses.append(float(np.mean(epoch)))
    return model, arrays, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--modules", default="aki,sepsis,respiratory,electrolyte_acid_base,cardiovascular_instability")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default="whole_body_shared_jepa_candidate.pt")
    parser.add_argument("--report", default="whole_body_shared_jepa_candidate.json")
    args = parser.parse_args()
    modules = [value.strip() for value in args.modules.split(",") if value.strip()]
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    frame, variables, module_names = load_shared_cohort(args.data_root, modules, horizons, args.max_stays)
    rows = np.arange(len(frame), dtype=np.int64)
    model, arrays, losses = fit_shared(frame, variables, len(module_names), rows, epochs=args.epochs)
    checkpoint = {
        "model_type": "SharedWholeBodySparseJEPA",
        "state_dict": model.state_dict(),
        "scaler": arrays[0].to_dict(),
        "variables": list(variables), "modules": list(module_names),
        "input_dim": model.input_dim, "hidden": model.hidden, "latent": model.latent,
        "promotion_status": "candidate_only",
        "architecture": {"shared_history_encoder": "GRU", "module_embedding": True, "measurement_pure": True, "shared_latent_validated": False, "causal_claim_allowed": False, "clinical_promotion_allowed": False},
        "training": {"rows": len(frame), "subjects": int(frame.subject_id.nunique()), "max_stays_per_horizon": args.max_stays, "loss_history": losses},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {"schema": "whole_body_shared_jepa_candidate.v1", "checkpoint": str(Path(args.output).resolve()), "modules": list(module_names), "variables": list(variables), "rows": len(frame), "loss_history": losses, "promotion_status": "candidate_only", "shared_latent_validated": False, "gate_required": ["patient_heldout", "hospital_heldout", "target_by_horizon_conformal", "module_balanced_evaluation"]}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
