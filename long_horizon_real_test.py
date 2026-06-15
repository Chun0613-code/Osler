"""Patient-level factual JEPA evaluation at horizons where persistence weakens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dka_action_contract import ACTION_KEYS
from dka_fidelity_replay import init_body
from dka_world_model import (
    A_DIM, S_MEAN, S_STD, STATE_KEYS, a2vec, load_checkpoint, s2vec,
    treatment_history_features,
)
from osler_jepa.symbolic import PHYSICAL_STABLE_THRESHOLDS
from train_intervention_jepa import choose_device


DT = 0.5
LAB_TO_INDEX = {
    "glucose": 0, "pH": 1, "HCO3": 2, "anion_gap": 3, "K": 4,
    "MAP": 5, "Na": 8, "osmolality": 9, "creatinine": 10,
    "urine_output": 11, "BHB": 12,
}


def action_sequence(trajectory, horizon):
    cells = int(round(horizon / DT))
    physical = np.zeros((cells, A_DIM), dtype=np.float32)
    for action in trajectory.get("actions", []):
        cell = int(round(float(action.get("t", 0.0)) / DT))
        if 0 <= cell < cells:
            physical[cell] = np.asarray([
                float(action.get(name, 0.0)) for name in ACTION_KEYS
            ], dtype=np.float32)
    return np.asarray([a2vec(action) for action in physical], dtype=np.float32)


def nearest_labs(trajectory, horizon, tolerance=2.0):
    selected = {}
    for lab in trajectory.get("labs", []):
        name = lab.get("var")
        if name not in LAB_TO_INDEX:
            continue
        distance = abs(float(lab["t"]) - horizon)
        if distance > tolerance:
            continue
        if name not in selected or distance < selected[name][0]:
            selected[name] = (distance, float(lab["value"]))
    return {name: value for name, (_, value) in selected.items()}


@torch.no_grad()
def evaluate(model, trajectories, horizons, device):
    report = {}
    for horizon in horizons:
        values = {
            name: {"jepa": [], "persistence": [], "jepa_changed": []}
            for name in LAB_TO_INDEX
        }
        contributing_stays = set()
        for trajectory in trajectories:
            body = init_body(trajectory["init"], trajectory.get("history_actions"))
            current = s2vec(body.observe()) * S_STD + S_MEAN
            sequence = action_sequence(trajectory, horizon)
            history_actions = [
                [float(action.get(name, 0.0)) for name in ACTION_KEYS]
                for action in trajectory.get("history_actions", [])
            ]
            history = treatment_history_features(history_actions)
            predicted, _, _ = model.rollout(
                torch.as_tensor(s2vec(body.observe()), device=device).unsqueeze(0),
                torch.as_tensor(sequence, device=device).unsqueeze(0),
                torch.as_tensor(history, device=device).unsqueeze(0),
            )
            future = predicted[0, -1].cpu().numpy() * S_STD + S_MEAN
            labs = nearest_labs(trajectory, horizon)
            for name, truth in labs.items():
                index = LAB_TO_INDEX[name]
                values[name]["jepa"].append(abs(float(future[index]) - truth))
                values[name]["persistence"].append(abs(float(current[index]) - truth))
                threshold = float(PHYSICAL_STABLE_THRESHOLDS[index])
                actual_delta = truth - float(current[index])
                if abs(actual_delta) > threshold:
                    predicted_delta = float(future[index] - current[index])
                    values[name]["jepa_changed"].append(
                        int(np.sign(predicted_delta) == np.sign(actual_delta))
                    )
                contributing_stays.add(int(trajectory["stay_id"]))
        per_state = {}
        for name, metrics in values.items():
            per_state[name] = {
                "n": len(metrics["jepa"]),
                "jepa_mae": float(np.mean(metrics["jepa"])) if metrics["jepa"] else None,
                "persistence_mae": float(np.mean(metrics["persistence"]))
                if metrics["persistence"] else None,
                "changed_n": len(metrics["jepa_changed"]),
                "jepa_changed_direction_accuracy": float(np.mean(metrics["jepa_changed"]))
                if metrics["jepa_changed"] else None,
            }
        report[f"{horizon:g}h"] = {
            "stays": len(contributing_stays),
            "per_state": per_state,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories")
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--horizons", default="6,12,24")
    parser.add_argument("--output", default="dka_long_horizon_real_test.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    trajectories = [
        json.loads(line) for line in Path(args.trajectories).read_text().splitlines()
        if line.strip()
    ]
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    horizons = [float(value) for value in args.horizons.split(",")]
    result = {
        "evaluation": "factual observed-treatment long-horizon test",
        "causal_claim_allowed": False,
        "trajectory_count": len(trajectories),
        "horizons": evaluate(model, trajectories, horizons, device),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
