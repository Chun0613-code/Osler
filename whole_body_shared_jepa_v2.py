"""Shared whole-body JEPA v2 with module-aware transfer controls.

v1 used one set of target heads for every body-system module.  That made the
+latent encoder carry the right information, but let large modules and dense
targets dominate the optimization.  v2 keeps one shared history encoder and
adds three controls against negative transfer:

* a module-specific adapter after the shared latent;
* module-specific target heads;
* balanced sampling over (module, horizon) groups and target-balanced loss.

This remains a research candidate.  The existing per-module registry and the
patient/hospital/conformal gate remain the runtime authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from nonlinear_latent_rollout import _history_indices, set_seed
from whole_body_shared_jepa import _shared_arrays, load_shared_cohort


class SharedWholeBodySparseJEPAv2(nn.Module):
    """Shared temporal representation with module-specific readouts."""

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
        adapter_input = latent + 12
        self.module_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(adapter_input, latent), nn.LayerNorm(latent), nn.SiLU()
                )
                for _ in range(module_count)
            ]
        )
        head_input = latent + 1 + 4
        self.value_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(head_input, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                        for _ in range(n_state)
                    ]
                )
                for _ in range(module_count)
            ]
        )
        self.observation_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
                        for _ in range(n_state)
                    ]
                )
                for _ in range(module_count)
            ]
        )

    @staticmethod
    def _time_features(horizon):
        horizon = horizon.clamp(min=0.05, max=168.0)
        log_horizon = torch.log1p(horizon)
        return torch.stack(
            [
                log_horizon / np.log1p(48.0),
                horizon / 6.0,
                torch.sin(log_horizon),
                torch.cos(log_horizon),
            ],
            dim=-1,
        )

    def forward(self, observations, current, horizon, module_id):
        _, hidden = self.history_encoder(observations)
        module = self.module_embedding(module_id.long())
        shared_latent = self.state_encoder(torch.cat([hidden[-1], current, module], dim=-1))
        time = self._time_features(horizon)
        values = current.clone()
        observed = current.new_zeros((current.shape[0], self.n_state))

        # Module-specific heads are selected only for the rows that belong to
        # that module.  The shared history encoder still receives every row.
        for module_index in range(self.module_count):
            selected = module_id == module_index
            if not bool(selected.any()):
                continue
            adapted = self.module_adapters[module_index](
                torch.cat([shared_latent[selected], module[selected]], dim=-1)
            )
            for target_index in range(self.n_state):
                head_input = torch.cat(
                    [
                        adapted,
                        current[selected, target_index : target_index + 1],
                        time[selected],
                    ],
                    dim=-1,
                )
                values[selected, target_index : target_index + 1] = (
                    current[selected, target_index : target_index + 1]
                    + 0.75
                    * torch.tanh(self.value_heads[module_index][target_index](head_input))
                )
                observed[selected, target_index : target_index + 1] = self.observation_heads[
                    module_index
                ][target_index](head_input)
        return {"value": values, "observation_logit": observed}


def _group_rows(frame: pd.DataFrame, rows: np.ndarray):
    grouped = {}
    modules = frame["_shared_module"].to_numpy(dtype=np.int64)
    horizons = frame["_source_horizon"].to_numpy(dtype=np.float32)
    for row in np.asarray(rows, dtype=np.int64):
        key = (int(modules[row]), float(horizons[row]))
        grouped.setdefault(key, []).append(int(row))
    return {key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()}


def _balanced_batches(frame: pd.DataFrame, rows: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Yield approximately equal module×horizon batches without dropping data."""
    groups = _group_rows(frame, rows)
    keys = sorted(groups)
    if not keys:
        return
    steps = max(1, int(np.ceil(len(rows) / float(batch_size))))
    for _ in range(steps):
        key = keys[int(rng.integers(0, len(keys)))]
        candidates = groups[key]
        replace = len(candidates) < batch_size
        yield rng.choice(candidates, size=batch_size, replace=replace)


def _target_balanced_loss(prediction, target, mask):
    element = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    count = mask.sum(dim=0)
    per_target = (element * mask).sum(dim=0) / count.clamp_min(1.0)
    active = count > 0
    return per_target[active].mean() if bool(active.any()) else element.new_zeros(())


def fit_shared_v2(frame, variables, module_count, rows, seed=7, epochs=2, batch_size=1024):
    set_seed(seed)
    arrays = _shared_arrays(frame, variables, rows)
    _, input_matrix, current, future, future_mask, history, horizon = arrays
    module_ids = frame["_shared_module"].to_numpy(dtype=np.int64)
    model = SharedWholeBodySparseJEPAv2(len(variables), input_matrix.shape[1], module_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        epoch = []
        model.train()
        for batch in _balanced_batches(frame, rows, batch_size, rng):
            output = model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
                torch.from_numpy(module_ids[batch]),
            )
            y = torch.from_numpy(future[batch]).float()
            mask = torch.from_numpy(future_mask[batch]).float()
            value_loss = _target_balanced_loss(output["value"], y, mask)
            observation_loss = nn.functional.binary_cross_entropy_with_logits(
                output["observation_logit"], mask
            )
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
    parser.add_argument(
        "--modules",
        default="dka,sepsis,aki,respiratory,integumentary_skin_wound,toxic_metabolic,electrolyte_acid_base,endocrine_stress,gi_pancreatic_nutrition,cardiac_injury,musculoskeletal_rhabdo,immune_inflammatory,cardiovascular_instability,acute_neuro,hepatic_failure,coagulopathy_heme",
    )
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", default="whole_body_shared_jepa_v2_candidate.pt")
    parser.add_argument("--report", default="whole_body_shared_jepa_v2_candidate.json")
    args = parser.parse_args()
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    frame, variables, module_names = load_shared_cohort(
        args.data_root, modules, horizons, args.max_stays
    )
    rows = np.arange(len(frame), dtype=np.int64)
    model, arrays, losses = fit_shared_v2(
        frame, variables, len(module_names), rows, epochs=args.epochs, batch_size=args.batch_size
    )
    checkpoint = {
        "model_type": "SharedWholeBodySparseJEPAv2",
        "state_dict": model.state_dict(),
        "scaler": arrays[0].to_dict(),
        "variables": list(variables),
        "modules": list(module_names),
        "input_dim": model.input_dim,
        "hidden": model.hidden,
        "latent": model.latent,
        "promotion_status": "candidate_only",
        "architecture": {
            "shared_history_encoder": "GRU",
            "module_embedding": True,
            "module_specific_adapters": True,
            "module_specific_target_heads": True,
            "module_horizon_balanced_sampling": True,
            "target_balanced_loss": True,
            "measurement_pure": True,
            "shared_latent_validated": False,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "training": {
            "rows": len(frame),
            "subjects": int(frame.subject_id.nunique()),
            "max_stays_per_horizon": args.max_stays,
            "loss_history": losses,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "whole_body_shared_jepa_candidate.v2",
        "checkpoint": str(Path(args.output).resolve()),
        "modules": list(module_names),
        "variables": list(variables),
        "rows": len(frame),
        "loss_history": losses,
        "promotion_status": "candidate_only",
        "shared_latent_validated": False,
        "gate_required": [
            "patient_heldout",
            "hospital_heldout",
            "target_by_horizon_conformal",
            "module_balanced_evaluation",
        ],
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
