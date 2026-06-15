"""Fit a small candidate residual ODE on patient-grouped MIMIC transitions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from dka_action_contract import ACTION_KEYS
from dka_fidelity_replay import init_body
from mimic_action_history import MAINTENANCE_NAMES
from osler_jepa.greybox_residual import (
    MAX_ABS_RATE, RESIDUAL_KEYS, GreyBoxResidualRuntime,
    ResidualRateNetwork, residual_features,
)


TARGET_COLUMNS = {
    "G": "glucose_tp6",
    "Ket": "anion_gap_tp6",
    "HCO3": "bicarbonate_tp6",
    "Ke": "potassium_tp6",
    "Na": "sodium_tp6",
    "Cr": "creatinine_tp6",
}
DT = 0.5


def _json_array(value, shape=None):
    if isinstance(value, str):
        value = json.loads(value)
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and (array.ndim != 2 or array.shape[1] != shape):
        return None
    return array


def _initial(row):
    values = {
        "glucose": row.get("glucose_t"),
        "HCO3": row.get("bicarbonate_t"),
        "K": row.get("potassium_t"),
        "anion_gap": row.get("anion_gap_t"),
        "Na": row.get("sodium_t"),
        "creatinine": row.get("creatinine_t"),
        "urine_output": row.get("urine_output_t"),
        "MAP": row.get("map_t"),
        "BHB": row.get("BHB_t"),
    }
    return {
        key: None if value is None or pd.isna(value) else float(value)
        for key, value in values.items()
    }


def _schedule(row):
    actions = _json_array(row.get("future_action_grid"), len(ACTION_KEYS))
    if actions is None:
        return None
    fluid_sodium = _json_array(row.get("future_fluid_sodium_grid"))
    if fluid_sodium is None or fluid_sodium.ndim != 1:
        fluid_sodium = np.full(len(actions), 140.0, dtype=np.float32)
    maintenance = _json_array(row.get("future_maintenance_grid"), len(MAINTENANCE_NAMES))
    if maintenance is None:
        maintenance = np.zeros((len(actions), len(MAINTENANCE_NAMES)), dtype=np.float32)
    result = []
    for cell, physical in enumerate(actions):
        action = dict(zip(ACTION_KEYS, physical.astype(float).tolist()))
        action["_fluid_sodium_meq_l"] = float(fluid_sodium[min(cell, len(fluid_sodium) - 1)])
        for index, name in enumerate(MAINTENANCE_NAMES):
            key = f"_{name}_ml" if name != "carbohydrate" else "_nutrition_carbohydrate_g"
            action[key] = float(maintenance[cell, index])
        result.append(action)
    return result


def _history(row):
    values = _json_array(row.get("history_action_grid"), len(ACTION_KEYS))
    if values is None:
        return None
    return [
        {"t": -len(values) * DT + index * DT,
         **dict(zip(ACTION_KEYS, action.astype(float).tolist()))}
        for index, action in enumerate(values)
    ]


def _endpoint(body):
    return np.asarray([
        body.G, body.Ket, body.HCO3, body.Ke, body.Na, body.Cr,
    ], dtype=np.float32)


def _target(row):
    values = []
    for name in RESIDUAL_KEYS:
        value = row.get(TARGET_COLUMNS[name])
        if name == "Ket" and not pd.isna(value):
            value = max(0.0, float(value) - 12.0)
        values.append(np.nan if pd.isna(value) else float(value))
    return np.asarray(values, dtype=np.float32)


def rollout(row, residual=None):
    schedule = _schedule(row)
    if not schedule:
        return None
    body = init_body(_initial(row), _history(row))
    body.residual_model = residual
    initial_observation = body.observe()
    for action in schedule:
        body.step(action, dt=DT)
    return initial_observation, _endpoint(body), schedule


def build_examples(frame):
    examples = []
    for _, row in frame.iterrows():
        baseline = rollout(row)
        if baseline is None:
            continue
        observation, mechanism, schedule = baseline
        target = _target(row)
        mask = np.isfinite(target)
        if not mask.any():
            continue
        mean_action = {
            key: float(np.mean([item.get(key, 0.0) for item in schedule]))
            for key in (*ACTION_KEYS, "_fluid_sodium_meq_l", *[
                f"_{name}_ml" if name != "carbohydrate" else "_nutrition_carbohydrate_g"
                for name in MAINTENANCE_NAMES
            ])
        }
        target_rate = np.zeros(len(RESIDUAL_KEYS), dtype=np.float32)
        target_rate[mask] = (target[mask] - mechanism[mask]) / (len(schedule) * DT)
        target_rate = np.clip(target_rate, -MAX_ABS_RATE, MAX_ABS_RATE)
        examples.append({
            "row": row,
            "stay_id": int(row["stay_id"]),
            "features": residual_features(observation, mean_action),
            "target_rate": target_rate,
            "mask": mask.astype(np.float32),
            "target": target,
            "mechanism": mechanism,
        })
    return examples


def fit_network(examples, train_indices, validation_indices=None, seed=7,
                epochs=400, bottleneck=8):
    torch.manual_seed(seed)
    x = np.stack([item["features"] for item in examples])
    y = np.stack([item["target_rate"] for item in examples])
    mask = np.stack([item["mask"] for item in examples])
    mean = x[train_indices].mean(axis=0)
    std = np.maximum(x[train_indices].std(axis=0), 1e-4)
    model = ResidualRateNetwork(input_dim=x.shape[1], bottleneck=bottleneck)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-2)
    bounds = torch.as_tensor(MAX_ABS_RATE)

    def loss_for(indices):
        features = torch.as_tensor((x[indices] - mean) / std)
        target = torch.as_tensor(y[indices])
        observed = torch.as_tensor(mask[indices])
        error = ((model(features) - target) / bounds) ** 2 * observed
        return error.sum() / observed.sum().clamp_min(1.0)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0
    for _ in range(epochs):
        model.train()
        loss = loss_for(train_indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        selected = validation_indices if validation_indices is not None else train_indices
        with torch.no_grad():
            validation = float(loss_for(selected))
        if validation < best_loss - 1e-5:
            best_loss = validation
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if validation_indices is not None and patience >= 40:
                break
    model.load_state_dict(best_state)
    return model, mean, std, best_loss


def evaluate_crossfit(examples, seed=7, bottleneck=8):
    groups = np.asarray([item["stay_id"] for item in examples])
    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    errors = {key: {"mechanism": [], "greybox": []} for key in RESIDUAL_KEYS}
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(examples)), groups=groups)):
        inner_groups = groups[train]
        inner_splitter = GroupKFold(
            n_splits=min(3, len(np.unique(inner_groups)))
        )
        inner_train_local, inner_validation_local = next(
            inner_splitter.split(train, groups=inner_groups)
        )
        inner_train = train[inner_train_local]
        inner_validation = train[inner_validation_local]
        model, mean, std, _ = fit_network(
            examples, inner_train, validation_indices=inner_validation,
            seed=seed + fold,
            bottleneck=bottleneck,
        )
        runtime = GreyBoxResidualRuntime(model, mean, std)
        for index in test:
            item = examples[index]
            _, prediction, _ = rollout(item["row"], runtime)
            for state_index, key in enumerate(RESIDUAL_KEYS):
                if not item["mask"][state_index]:
                    continue
                truth = item["target"][state_index]
                errors[key]["mechanism"].append(abs(item["mechanism"][state_index] - truth))
                errors[key]["greybox"].append(abs(prediction[state_index] - truth))
    return {
        key: {
            "n": len(values["mechanism"]),
            "mechanism_mae": float(np.mean(values["mechanism"])) if values["mechanism"] else None,
            "greybox_mae": float(np.mean(values["greybox"])) if values["greybox"] else None,
        }
        for key, values in errors.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort")
    parser.add_argument("--artifact", default="dka_greybox_residual_candidate.pt")
    parser.add_argument("--report", default="dka_greybox_residual_candidate.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bottleneck", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_parquet(args.cohort)
    examples = build_examples(frame)
    if len({item["stay_id"] for item in examples}) < 4:
        raise RuntimeError("At least four stays are required for grouped evaluation")
    crossfit = evaluate_crossfit(examples, args.seed, args.bottleneck)
    all_indices = np.arange(len(examples))
    model, mean, std, training_loss = fit_network(
        examples, all_indices, seed=args.seed, bottleneck=args.bottleneck,
    )
    metadata = {
        "source_cohort": Path(args.cohort).name,
        "source_stays": len({item["stay_id"] for item in examples}),
        "source_transitions": len(examples),
        "candidate_only": True,
        "promotion_allowed": False,
        "causal_claim_allowed": False,
    }
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "bottleneck": args.bottleneck,
        "metadata": metadata,
    }, args.artifact)
    report = {
        "model": "constrained grey-box residual ODE candidate",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "residual_keys": list(RESIDUAL_KEYS),
        "hard_owned_by_mechanism": [
            "total_body_potassium", "fluid_volume", "insulin_pk_depots",
            "dose_mass_balance", "osmotic_injury",
        ],
        "crossfit": crossfit,
        "training_loss": training_loss,
        **metadata,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved candidate artifact: {args.artifact}")


if __name__ == "__main__":
    main()
