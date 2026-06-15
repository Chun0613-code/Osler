"""Train and evaluate the action-conditioned numerical DKA JEPA.

Each simulated patient state is cloned into multiple intervention branches. The
split is performed by patient scenario before training, so no counterfactual
branch from a validation/test state can appear in the training set.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dka_body import DKABody, henderson
from dka_action_contract import ACTION_INDEX, INSULIN_KEYS, expand_action
from dka_world_model import (
    A_DIM,
    ACTION_KEYS,
    A_SCALE,
    DT,
    H_DIM,
    HISTORY_HOURS,
    MAX_OBSERVATION_AGE_HOURS,
    S_DIM,
    S_MEAN,
    S_STD,
    STATE_KEYS,
    WorldModel,
    a2vec,
    randomized_dka,
    save_checkpoint,
    s2vec,
    observation_context,
    treatment_history_features,
    vicreg,
)
from osler_jepa.curriculum import STAGES, stage_for_epoch
from osler_jepa.actions import TREATMENT_EVENT_DIM, treatment_event_features
from osler_jepa.ontology import OSLER_STATE_ONTOLOGY
from osler_jepa.symbolic import (
    RULE_IDS,
    contradiction_penalty,
    direction_targets,
    masked_binary_cross_entropy,
    masked_cross_entropy,
    rule_supervision,
    schema as symbolic_schema,
)
from osler_jepa.validator import OSLER_DKA_VALIDATOR


PROTOCOLS = (
    "no_treatment",
    "insulin",
    "rapid_sc",
    "basal_sc",
    "fluids",
    "potassium",
    "insulin_fluids",
    "full_protocol",
    "bicarbonate",
    "dextrose",
    "randomized",
)

BASE_ACTIONS = {
    "no_treatment": expand_action([0, 0, 0, 0, 0]),
    "insulin": expand_action([6, 0, 0, 0, 0]),
    "rapid_sc": expand_action({"insulin_rapid_sc": 16}),
    "basal_sc": expand_action({"insulin_basal_sc": 40}),
    "fluids": expand_action([0, 500, 0, 0, 0]),
    "potassium": expand_action([0, 0, 20, 0, 0]),
    "insulin_fluids": expand_action([6, 500, 0, 0, 0]),
    "full_protocol": expand_action([6, 500, 10, 0, 0]),
    "bicarbonate": expand_action([0, 250, 0, 50, 0]),
    "dextrose": expand_action([3, 100, 0, 0, 5]),
}


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def protocol_action(name, observation, intensity, rng, step=0):
    if name == "randomized":
        action = np.zeros(A_DIM, dtype=np.float32)
        insulin_channel = int(rng.integers(0, 5))
        if insulin_channel < 4:
            choices = (
                [0.0, 2.0, 4.0, 6.0, 10.0, 20.0],
                [0.0, 8.0, 16.0, 24.0],
                [0.0, 20.0, 40.0],
                [0.0, 20.0, 40.0, 60.0],
            )[insulin_channel]
            action[insulin_channel] = rng.choice(choices)
        action[ACTION_INDEX["fluids"]] = rng.choice([0.0, 250.0, 500.0])
        action[ACTION_INDEX["kcl"]] = rng.choice([0.0, 10.0, 20.0])
        action[ACTION_INDEX["bicarbonate"]] = rng.choice([0.0, 0.0, 25.0, 50.0])
        action[ACTION_INDEX["dextrose"]] = rng.choice([0.0, 0.0, 5.0, 10.0])
    else:
        action = BASE_ACTIONS[name] * intensity
        if name in {"rapid_sc", "basal_sc"} and step > 0:
            action = action.copy()
            action[:4] = 0.0

    jitter = rng.normal(1.0, 0.06, size=A_DIM).astype(np.float32)
    action = np.where(action > 0, action * jitter, action)

    insulin_slice = slice(0, len(INSULIN_KEYS))
    if name in {"insulin", "rapid_sc", "basal_sc", "insulin_fluids", "full_protocol"} and observation["G"] < 120:
        action[insulin_slice] *= 0.25
    if name == "full_protocol" and observation["Ke"] < 3.3:
        action[insulin_slice] = 0.0
        action[ACTION_INDEX["kcl"]] = max(action[ACTION_INDEX["kcl"]], 20.0)
    if name == "bicarbonate" and observation["pH"] > 7.15:
        action[ACTION_INDEX["bicarbonate"]] = 0.0
    if observation["G"] < 250 and observation["HCO3"] < 18 and action[insulin_slice].sum() > 0:
        action[ACTION_INDEX["dextrose"]] = max(action[ACTION_INDEX["dextrose"]], 5.0)
    if observation["G"] > 350 and name != "dextrose":
        action[ACTION_INDEX["dextrose"]] = 0.0

    upper = A_SCALE * 2.0
    return np.clip(action, 0.0, upper)


def prepare_base_body(rng):
    body = DKABody(rng=rng)
    observation = randomized_dka(body, rng)
    # Branch at different points in the treatment course, not only presentation.
    # This exposes the model to partially corrected glucose/acidosis and residual
    # treatment effects that resemble later EHR anchors.
    prior_actions = []
    prior_durations = []
    for _ in range(int(rng.integers(0, 13))):
        warmup = np.zeros(A_DIM, dtype=np.float32)
        route = int(rng.choice([0, 0, 0, 1, 2, 3]))
        warmup[route] = float(rng.choice(
            [0.0, 2.0, 4.0, 6.0, 10.0, 20.0] if route == 0 else
            [0.0, 8.0, 16.0] if route == 1 else
            [0.0, 20.0, 40.0]
        ))
        warmup[ACTION_INDEX["fluids"]] = float(rng.choice([0.0, 250.0, 500.0]))
        warmup[ACTION_INDEX["kcl"]] = float(rng.choice([0.0, 0.0, 10.0, 20.0]))
        if observation["Ke"] < 3.3:
            warmup[:4] = 0.0
            warmup[ACTION_INDEX["kcl"]] = 20.0
        if observation["G"] < 140:
            warmup[:4] *= 0.25
        if observation["G"] < 250 and observation["HCO3"] < 18 and warmup[:4].sum() > 0:
            warmup[ACTION_INDEX["dextrose"]] = float(rng.choice([5.0, 10.0]))
        observation, _, dead, _ = body.step(warmup, dt=DT)
        prior_actions.append(warmup.tolist())
        prior_durations.append(DT)
        if dead:
            observation = randomized_dka(body, rng)
            prior_actions = []
            prior_durations = []
            break
    return body, observation, prior_actions, prior_durations


def generate_branched_dataset(n_scenarios=600, seq_len=12, seed=0):
    rng = np.random.default_rng(seed)
    n_protocols = len(PROTOCOLS)
    states = np.zeros((n_scenarios, n_protocols, seq_len + 1, S_DIM), np.float32)
    actions = np.zeros((n_scenarios, n_protocols, seq_len, A_DIM), np.float32)
    action_events = np.zeros(
        (n_scenarios, n_protocols, seq_len, TREATMENT_EVENT_DIM), np.float32
    )
    histories = np.zeros((n_scenarios, n_protocols, seq_len, H_DIM), np.float32)
    valid = np.zeros((n_scenarios, n_protocols, seq_len), np.float32)
    alive = np.zeros((n_scenarios, n_protocols, seq_len), np.float32)
    time_deltas = np.zeros((n_scenarios, n_protocols, seq_len), np.float32)
    death_causes = {}
    death_causes_by_protocol = {protocol: {} for protocol in PROTOCOLS}

    for scenario in range(n_scenarios):
        base_body, base_observation, base_history, base_durations = prepare_base_body(rng)
        scenario_deltas = rng.choice(
            np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
            size=seq_len,
            p=[0.10, 0.65, 0.15, 0.10],
        )
        intensities = rng.uniform(0.75, 1.25, size=(n_protocols, A_DIM)).astype(np.float32)
        for protocol_index, protocol in enumerate(PROTOCOLS):
            body = copy.deepcopy(base_body)
            observation = dict(base_observation)
            prior_actions = list(base_history)
            prior_durations = list(base_durations)
            states[scenario, protocol_index, 0] = s2vec(observation)
            for step in range(seq_len):
                histories[scenario, protocol_index, step] = treatment_history_features(
                    prior_actions, DT, HISTORY_HOURS, prior_durations
                )
                action = protocol_action(
                    protocol, observation, intensities[protocol_index], rng, step
                )
                previous_action = prior_actions[-1] if prior_actions else None
                action_events[scenario, protocol_index, step] = (
                    treatment_event_features(
                        [action], initial_action=previous_action
                    )[0]
                )
                step_hours = float(scenario_deltas[step])
                next_observation, _, dead, info = body.step(action, dt=step_hours)
                actions[scenario, protocol_index, step] = a2vec(action)
                time_deltas[scenario, protocol_index, step] = step_hours
                states[scenario, protocol_index, step + 1] = s2vec(next_observation)
                valid[scenario, protocol_index, step] = 1.0
                alive[scenario, protocol_index, step] = 0.0 if dead else 1.0
                observation = next_observation
                prior_actions.append(action.tolist())
                prior_durations.append(step_hours)
                if dead:
                    cause = info.get("cause") or "unknown"
                    death_causes[cause] = death_causes.get(cause, 0) + 1
                    protocol_causes = death_causes_by_protocol[protocol]
                    protocol_causes[cause] = protocol_causes.get(cause, 0) + 1
                    states[scenario, protocol_index, step + 1:] = s2vec(observation)
                    break

    return {
        "states": states,
        "actions": actions,
        "action_events": action_events,
        "histories": histories,
        "time_deltas": time_deltas,
        "valid": valid,
        "alive": alive,
        "protocols": list(PROTOCOLS),
        "death_causes": death_causes,
        "death_causes_by_protocol": death_causes_by_protocol,
    }


def simulator_calibration_audit(dataset):
    """Summarize mortality and intervention-response magnitude by protocol."""
    physical = dataset["states"] * S_STD + S_MEAN
    initial = physical[:, :, 0]
    final = physical[:, :, -1]
    alive = dataset["alive"][:, :, -1]
    state_indices = {
        name: STATE_KEYS.index(name)
        for name in ("G", "HCO3", "Ke", "MAP", "V", "K_store", "osmolality")
    }
    protocol_report = {}
    for protocol_index, protocol in enumerate(PROTOCOLS):
        changes = final[:, protocol_index] - initial[:, protocol_index]
        protocol_report[protocol] = {
            "mortality_rate": round(float((1.0 - alive[:, protocol_index]).mean()), 6),
            "death_causes": dataset["death_causes_by_protocol"][protocol],
            "change_quantiles": {
                name: {
                    "p10": round(float(np.quantile(changes[:, index], 0.10)), 4),
                    "median": round(float(np.quantile(changes[:, index], 0.50)), 4),
                    "p90": round(float(np.quantile(changes[:, index], 0.90)), 4),
                }
                for name, index in state_indices.items()
            },
        }
    median = lambda protocol, state: protocol_report[protocol][
        "change_quantiles"
    ][state]["median"]
    gates = {
        "full_protocol_not_more_lethal_than_no_treatment": (
            protocol_report["full_protocol"]["mortality_rate"]
            <= protocol_report["no_treatment"]["mortality_rate"]
        ),
        "iv_insulin_reduces_glucose_vs_no_treatment": (
            median("insulin", "G") < median("no_treatment", "G")
        ),
        "fluids_raise_map_vs_no_treatment": (
            median("fluids", "MAP") > median("no_treatment", "MAP")
        ),
        "kcl_replenishes_store_vs_no_treatment": (
            median("potassium", "K_store") > median("no_treatment", "K_store")
        ),
    }
    return {
        "protocols": protocol_report,
        "mechanistic_direction_gates": gates,
        "all_direction_gates_pass": bool(all(gates.values())),
        "clinical_calibration_claim_allowed": False,
    }


def split_scenarios(n_scenarios, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_scenarios)
    train_end = int(0.70 * n_scenarios)
    validation_end = int(0.85 * n_scenarios)
    return {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "test": order[validation_end:],
    }


def masked_mean(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0) / values.shape[-1]


def _partial_observation_context(state, drop_probability):
    if drop_probability <= 0:
        return torch.ones_like(state), torch.zeros_like(state)
    mask = (torch.rand_like(state) > drop_probability).to(state.dtype)
    # Keep at least one directly observed variable in every state vector.
    empty = mask.sum(dim=-1) == 0
    if empty.any():
        mask[empty, 0] = 1.0
    measured_age = torch.rand_like(state) * 6.0
    age = torch.where(
        mask > 0,
        measured_age,
        torch.full_like(state, MAX_OBSERVATION_AGE_HOURS),
    )
    return mask, age


def _time_weighted_exposure(actions, time_deltas, horizon):
    durations = time_deltas[:, :, :horizon + 1]
    weighted = actions[:, :, :horizon + 1] * durations.unsqueeze(-1)
    return weighted.sum(dim=2) / durations.sum(dim=2).clamp_min(1e-6).unsqueeze(-1)


def batch_losses(model, states, actions, action_events, histories, time_deltas,
                 valid, alive, weights=None, observation_dropout=0.0):
    weights = weights or STAGES[-1].weights
    batch, protocols, steps, _ = actions.shape
    current = states[:, :, :-1].reshape(-1, S_DIM)
    target = states[:, :, 1:].reshape(-1, S_DIM)
    flat_actions = actions.reshape(-1, A_DIM)
    flat_action_events = action_events.reshape(-1, TREATMENT_EVENT_DIM)
    flat_histories = histories.reshape(-1, H_DIM)
    flat_deltas = time_deltas.reshape(-1)
    elapsed = torch.cumsum(time_deltas, dim=2) - time_deltas
    flat_elapsed = elapsed.reshape(-1)
    flat_valid = valid.reshape(-1)

    current_mask, current_age = _partial_observation_context(
        current, observation_dropout
    )
    latent = model.encode_state(
        current * current_mask, flat_histories, current_mask, current_age
    )
    predicted_latent = model.predict_latent(
        latent, flat_actions, flat_deltas, flat_elapsed, flat_action_events
    )
    with torch.no_grad():
        target_latent = model.Ebar(target)

    one_step_latent = masked_mean((predicted_latent - target_latent).square(), flat_valid)
    one_step_state = masked_mean((model.D(predicted_latent) - target).square(), flat_valid)
    reconstruction = masked_mean((model.D(latent) - current).square(), flat_valid)

    initial = states[:, :, 0].reshape(batch * protocols, S_DIM)
    action_sequences = actions.reshape(batch * protocols, steps, A_DIM)
    event_sequences = action_events.reshape(
        batch * protocols, steps, TREATMENT_EVENT_DIM
    )
    initial_history = histories[:, :, 0].reshape(batch * protocols, H_DIM)
    initial_mask, initial_age = _partial_observation_context(
        initial, observation_dropout
    )
    predicted, risk_logits, rollout_latents = model.rollout(
        initial * initial_mask, action_sequences, initial_history,
        observation_mask=initial_mask,
        observation_age=initial_age,
        delta_hours=time_deltas.reshape(batch * protocols, steps),
        treatment_events=event_sequences,
    )
    predicted = predicted.reshape(batch, protocols, steps, S_DIM)
    risk_logits = risk_logits.reshape(batch, protocols, steps)
    rollout_latents = rollout_latents.reshape(batch, protocols, steps, -1)

    rollout_state = masked_mean((predicted - states[:, :, 1:]).square(), valid)
    with torch.no_grad():
        rollout_targets = model.Ebar(states[:, :, 1:].reshape(-1, S_DIM)).reshape_as(
            rollout_latents
        )
    rollout_latent = masked_mean((rollout_latents - rollout_targets).square(), valid)

    death_target = 1.0 - alive
    positive = (death_target * valid).sum()
    negative = ((1.0 - death_target) * valid).sum()
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    risk = F.binary_cross_entropy_with_logits(
        risk_logits,
        death_target,
        reduction="none",
        pos_weight=positive_weight,
    )
    risk = (risk * valid).sum() / valid.sum().clamp_min(1.0)

    effect_losses = []
    osler_losses = []
    for horizon in (0, min(5, steps - 1), steps - 1):
        pair_valid = valid[:, 1:, horizon] * valid[:, :1, horizon]
        predicted_effect = predicted[:, 1:, horizon] - predicted[:, :1, horizon]
        true_effect = states[:, 1:, horizon + 1] - states[:, :1, horizon + 1]
        effect_losses.append(masked_mean((predicted_effect - true_effect).square(), pair_valid))
        action_exposure = _time_weighted_exposure(
            actions[:, 1:], time_deltas[:, 1:], horizon
        )
        control_exposure = _time_weighted_exposure(
            actions[:, :1], time_deltas[:, :1], horizon
        )
        osler_losses.append(OSLER_DKA_VALIDATOR.consistency_loss(
            predicted_effect,
            action_exposure - control_exposure,
            pair_valid,
            horizon_hours=time_deltas[:, 1:, :horizon + 1].sum(dim=2),
        ))
    effect = torch.stack(effect_losses).mean()
    osler = torch.stack(osler_losses).mean()

    initial_latent = model.encode_state(
        initial * initial_mask, initial_history, initial_mask, initial_age
    ).reshape(
        batch, protocols, -1
    )
    direction_losses = []
    status_losses = []
    proof_losses = []
    proposal_losses = []
    contradiction_losses = []
    contrastive_losses = []
    for horizon in (0, min(5, steps - 1), steps - 1):
        horizon_valid = valid[:, :, horizon]
        action_exposure = _time_weighted_exposure(
            actions, time_deltas, horizon
        )
        horizon_hours = time_deltas[:, :, :horizon + 1].sum(dim=2)
        horizon_events = action_events[:, :, :horizon + 1].amax(dim=2)
        outputs = model.symbolic_outputs(
            initial_latent.reshape(-1, initial_latent.shape[-1]),
            action_exposure.reshape(-1, A_DIM),
            delta_hours=horizon_hours.reshape(-1),
            treatment_events=horizon_events.reshape(-1, TREATMENT_EVENT_DIM),
        )
        direction_logits = outputs["direction_logits"].reshape(
            batch, protocols, S_DIM, 3
        )
        transition = states[:, :, horizon + 1] - states[:, :, 0]
        direction_label = direction_targets(transition, S_STD)
        direction_losses.append(masked_cross_entropy(
            direction_logits, direction_label, horizon_valid
        ))

        pair_valid = horizon_valid[:, 1:] * horizon_valid[:, :1]
        true_effect = (
            states[:, 1:, horizon + 1] - states[:, :1, horizon + 1]
        )
        action_difference = action_exposure[:, 1:] - action_exposure[:, :1]
        proposal_logits = outputs["proposal_logits"].reshape(
            batch, protocols, S_DIM, 3
        )[:, 1:]
        proposal_label = direction_targets(true_effect, S_STD)
        proposal_losses.append(masked_cross_entropy(
            proposal_logits, proposal_label, pair_valid
        ))

        status_label, proof_label = rule_supervision(
            true_effect, action_difference, pair_valid, horizon_hours[:, 1:]
        )
        status_logits = outputs["status_logits"].reshape(
            batch, protocols, 3
        )[:, 1:]
        proof_logits = outputs["proof_logits"].reshape(
            batch, protocols, len(RULE_IDS)
        )[:, 1:]
        status_losses.append(masked_cross_entropy(
            status_logits, status_label, pair_valid
        ))
        proof_losses.append(masked_binary_cross_entropy(
            proof_logits, proof_label, pair_valid
        ))
        contradiction_losses.append(contradiction_penalty(
            proposal_logits, action_difference, pair_valid, horizon_hours[:, 1:]
        ))

        predicted_effect = predicted[:, 1:, horizon] - predicted[:, :1, horizon]
        changed = (
            direction_targets(true_effect, S_STD) != 1
        ).any(dim=-1) & pair_valid.bool()
        predicted_magnitude = predicted_effect.abs().mean(dim=-1)
        if changed.any():
            contrastive_losses.append(
                torch.relu(predicted_magnitude.new_tensor(0.02)
                           - predicted_magnitude)[changed].mean()
            )

    direction = torch.stack(direction_losses).mean()
    symbolic_status = torch.stack(status_losses).mean()
    proof_path = torch.stack(proof_losses).mean()
    rule_proposal = torch.stack(proposal_losses).mean()
    contradiction = torch.stack(contradiction_losses).mean()
    action_contrastive = (
        torch.stack(contrastive_losses).mean()
        if contrastive_losses else effect.new_tensor(0.0)
    )

    anti_collapse = vicreg(latent[flat_valid > 0][:2048])
    total = (
        weights["one_step_latent"] * one_step_latent
        + weights["one_step_state"] * one_step_state
        + weights["rollout_state"] * rollout_state
        + weights["rollout_latent"] * rollout_latent
        + weights["effect"] * effect
        + weights["osler"] * osler
        + weights["reconstruction"] * reconstruction
        + weights["anti_collapse"] * anti_collapse
        + weights["risk"] * risk
        + weights["direction"] * direction
        + weights["symbolic_status"] * symbolic_status
        + weights["proof_path"] * proof_path
        + weights["rule_proposal"] * rule_proposal
        + weights["contradiction"] * contradiction
        + weights["action_contrastive"] * action_contrastive
    )
    return {
        "total": total,
        "one_step_latent": one_step_latent,
        "one_step_state": one_step_state,
        "rollout_state": rollout_state,
        "rollout_latent": rollout_latent,
        "effect": effect,
        "osler": osler,
        "reconstruction": reconstruction,
        "risk": risk,
        "anti_collapse": anti_collapse,
        "direction": direction,
        "symbolic_status": symbolic_status,
        "proof_path": proof_path,
        "rule_proposal": rule_proposal,
        "contradiction": contradiction,
        "action_contrastive": action_contrastive,
    }


def tensor_batch(dataset, indices, device):
    return tuple(
        torch.as_tensor(dataset[key][indices], dtype=torch.float32, device=device)
        for key in (
            "states", "actions", "action_events", "histories", "time_deltas",
            "valid", "alive"
        )
    )


@torch.no_grad()
def validation_loss(model, dataset, indices, batch_size, device, weights=None):
    model.eval()
    totals = []
    for start in range(0, len(indices), batch_size):
        selection = indices[start:start + batch_size]
        losses = batch_losses(
            model, *tensor_batch(dataset, selection, device), weights=weights
        )
        totals.append(float(losses["total"]))
    return float(np.mean(totals)) if totals else math.inf


def train_model(model, dataset, splits, epochs, batch_size, learning_rate, device):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    best_validation = math.inf
    best_state = None
    history = []
    rng = np.random.default_rng(17)

    for epoch in range(1, epochs + 1):
        stage = stage_for_epoch(epoch, epochs)
        model.train()
        order = rng.permutation(splits["train"])
        epoch_parts = []
        for start in range(0, len(order), batch_size):
            selection = order[start:start + batch_size]
            losses = batch_losses(
                model,
                *tensor_batch(dataset, selection, device),
                weights=stage.weights,
                observation_dropout=0.25,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.ema()
            epoch_parts.append({name: float(value.detach()) for name, value in losses.items()})
        scheduler.step()

        validation = validation_loss(
            model, dataset, splits["validation"], batch_size, device,
            weights=STAGES[-1].weights,
        )
        summary = {
            name: float(np.mean([part[name] for part in epoch_parts]))
            for name in epoch_parts[0]
        }
        summary["validation"] = validation
        summary["epoch"] = epoch
        summary["curriculum_stage"] = stage.name
        history.append(summary)
        if validation < best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:03d}/{epochs} train={summary['total']:.4f} "
                f"val={validation:.4f} roll={summary['rollout_state']:.4f} "
                f"effect={summary['effect']:.4f} osler={summary['osler']:.4f} "
                f"symbolic={summary['rule_proposal']:.4f} "
                f"stage={stage.name}"
            )

    model.load_state_dict(best_state)
    model.eval()
    return history, best_validation


@torch.no_grad()
def predict_split(model, dataset, indices, device, batch_size=32):
    predictions, risks = [], []
    for start in range(0, len(indices), batch_size):
        selection = indices[start:start + batch_size]
        states, actions, action_events, histories, time_deltas, _, _ = tensor_batch(
            dataset, selection, device
        )
        batch, protocols, steps, _ = actions.shape
        predicted, risk, _ = model.rollout(
            states[:, :, 0].reshape(batch * protocols, S_DIM),
            actions.reshape(batch * protocols, steps, A_DIM),
            histories[:, :, 0].reshape(batch * protocols, H_DIM),
            delta_hours=time_deltas.reshape(batch * protocols, steps),
            treatment_events=action_events.reshape(
                batch * protocols, steps, TREATMENT_EVENT_DIM
            ),
        )
        predictions.append(predicted.reshape(batch, protocols, steps, S_DIM).cpu().numpy())
        risks.append(torch.sigmoid(risk).reshape(batch, protocols, steps).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(risks)


def masked_physical_mae(prediction, truth, mask):
    physical_error = np.abs(prediction - truth) * S_STD
    output = {}
    for index, name in enumerate(STATE_KEYS):
        selected = physical_error[..., index][mask]
        output[name] = round(float(selected.mean()), 4) if len(selected) else None
    return output


def collapse_metrics(model, start_states, start_histories, device):
    with torch.no_grad():
        latent = model.encode_state(
            torch.as_tensor(start_states, dtype=torch.float32, device=device),
            torch.as_tensor(start_histories, dtype=torch.float32, device=device),
        )
    latent = latent.cpu().numpy()
    std = latent.std(axis=0)
    covariance = np.cov(latent, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    effective_rank = float(np.exp(entropy))
    active_dimensions = int((std > 0.01).sum())
    return {
        "mean_latent_std": round(float(std.mean()), 6),
        "min_latent_std": round(float(std.min()), 6),
        "effective_rank": round(effective_rank, 3),
        "active_dimensions_std_gt_0_01": active_dimensions,
        "low_rank_physiology_warning": bool(effective_rank < 8.0),
        "collapsed": bool(
            std.mean() < 0.05
            or active_dimensions < latent.shape[1] // 2
            or effective_rank < 3.0
        ),
    }


def counterfactual_metrics(prediction, states, valid):
    horizons = [0, min(5, prediction.shape[2] - 1), prediction.shape[2] - 1]
    thresholds = np.array([
        5.0, 0.005, 0.2, 0.2, 0.02, 0.5, 0.05, 0.5,
        0.2, 1.0, 0.05, 5.0, 0.1, 1.0, 0.1,
    ])
    report = {}
    for horizon in horizons:
        pair_valid = (valid[:, 1:, horizon] > 0) & (valid[:, :1, horizon] > 0)
        predicted_effect = (
            prediction[:, 1:, horizon] - prediction[:, :1, horizon]
        ) * S_STD
        true_effect = (
            states[:, 1:, horizon + 1] - states[:, :1, horizon + 1]
        ) * S_STD
        effect_error = np.abs(predicted_effect - true_effect)
        variable_mae, signs, sign_total = {}, 0, 0
        for variable, name in enumerate(STATE_KEYS):
            selected = pair_valid & (np.abs(true_effect[..., variable]) >= thresholds[variable])
            values = effect_error[..., variable][pair_valid]
            variable_mae[name] = round(float(values.mean()), 4) if len(values) else None
            signs += int((
                np.sign(predicted_effect[..., variable][selected])
                == np.sign(true_effect[..., variable][selected])
            ).sum())
            sign_total += int(selected.sum())
        report[f"{horizon + 1}_steps"] = {
            "effect_mae": variable_mae,
            "effect_sign_accuracy": round(signs / sign_total, 4) if sign_total else None,
            "sign_comparisons": sign_total,
        }
    return report


def osler_validator_audit(prediction, states, actions, time_deltas, valid):
    horizon = prediction.shape[2] - 1
    status_counts = {"verified": 0, "contradicted": 0, "unexplained": 0}
    rule_counts = {}
    flag_examples = []
    for scenario in range(prediction.shape[0]):
        baseline = dict(zip(
            STATE_KEYS,
            (prediction[scenario, 0, horizon] * S_STD + S_MEAN).tolist(),
        ))
        initial = dict(zip(
            STATE_KEYS,
            (states[scenario, 0, 0] * S_STD + S_MEAN).tolist(),
        ))
        for protocol in range(1, prediction.shape[1]):
            if not valid[scenario, protocol, horizon]:
                continue
            future = dict(zip(
                STATE_KEYS,
                (prediction[scenario, protocol, horizon] * S_STD + S_MEAN).tolist(),
            ))
            durations = time_deltas[scenario, protocol, :horizon + 1]
            action = (
                actions[scenario, protocol, :horizon + 1]
                * durations[:, None]
            ).sum(axis=0) / max(float(durations.sum()), 1e-6)
            audit = OSLER_DKA_VALIDATOR.validate(
                action, future, baseline,
                horizon_hours=float(
                    time_deltas[scenario, protocol, :horizon + 1].sum()
                ),
            )
            status_counts[audit["status"]] += 1
            for check in audit["checks"]:
                counts = rule_counts.setdefault(
                    check["rule_id"], {"verified": 0, "contradicted": 0}
                )
                counts[check["status"]] += 1
            if len(flag_examples) < 8:
                flag_examples.append({
                    "protocol": PROTOCOLS[protocol],
                    "initial_flags": OSLER_STATE_ONTOLOGY.derive_flags(initial),
                    "future_flags": OSLER_STATE_ONTOLOGY.derive_flags(future),
                    "status": audit["status"],
                })
    checked = status_counts["verified"] + status_counts["contradicted"]
    return {
        "status_counts": status_counts,
        "verified_rate_when_explainable": round(
            status_counts["verified"] / max(checked, 1), 4
        ),
        "rule_counts": rule_counts,
        "flag_transition_examples": flag_examples,
    }


@torch.no_grad()
def symbolic_head_metrics(model, dataset, indices, device):
    states, actions, action_events, histories, time_deltas, valid, _ = tensor_batch(
        dataset, indices, device
    )
    batch, protocols, steps, _ = actions.shape
    initial = states[:, :, 0]
    initial_latent = model.encode_state(
        initial.reshape(-1, S_DIM), histories[:, :, 0].reshape(-1, H_DIM)
    )
    action_exposure = _time_weighted_exposure(
        actions, time_deltas, actions.shape[2] - 1
    )
    total_hours = time_deltas.sum(dim=2)
    outputs = model.symbolic_outputs(
        initial_latent,
        action_exposure.reshape(-1, A_DIM),
        delta_hours=total_hours.reshape(-1),
        treatment_events=action_events.amax(dim=2).reshape(
            -1, TREATMENT_EVENT_DIM
        ),
    )

    direction_prediction = outputs["direction_logits"].argmax(dim=-1).reshape(
        batch, protocols, S_DIM
    )
    direction_truth = direction_targets(states[:, :, -1] - initial, S_STD)
    direction_mask = valid[:, :, -1].bool().unsqueeze(-1).expand_as(direction_truth)
    changed_mask = direction_mask & (direction_truth != 1)

    pair_valid = (valid[:, 1:, -1] * valid[:, :1, -1]).bool()
    true_effect = states[:, 1:, -1] - states[:, :1, -1]
    action_difference = action_exposure[:, 1:] - action_exposure[:, :1]
    proposal_prediction = outputs["proposal_logits"].argmax(dim=-1).reshape(
        batch, protocols, S_DIM
    )[:, 1:]
    proposal_truth = direction_targets(true_effect, S_STD)
    proposal_mask = pair_valid.unsqueeze(-1).expand_as(proposal_truth)
    proposal_changed = proposal_mask & (proposal_truth != 1)

    status_truth, proof_truth = rule_supervision(
        true_effect, action_difference, pair_valid, total_hours[:, 1:]
    )
    status_prediction = outputs["status_logits"].argmax(dim=-1).reshape(
        batch, protocols
    )[:, 1:]
    proof_prediction = (
        outputs["proof_logits"].sigmoid().reshape(
            batch, protocols, len(RULE_IDS)
        )[:, 1:] >= 0.5
    )
    proof_truth = proof_truth.bool()
    proof_mask = pair_valid.unsqueeze(-1).expand_as(proof_truth)
    true_positive = (proof_prediction & proof_truth & proof_mask).sum()
    false_positive = (proof_prediction & ~proof_truth & proof_mask).sum()
    false_negative = (~proof_prediction & proof_truth & proof_mask).sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

    status_counts = {}
    for index, name in enumerate(("verified", "contradicted", "unexplained")):
        selected = pair_valid & (status_truth == index)
        status_counts[name] = {
            "n": int(selected.sum()),
            "accuracy": round(float(
                (status_prediction[selected] == status_truth[selected]).float().mean()
            ), 4) if selected.any() else None,
        }

    def accuracy(prediction, truth, mask):
        return round(float((prediction[mask] == truth[mask]).float().mean()), 4) \
            if mask.any() else None

    per_state = {}
    for state_index, name in enumerate(STATE_KEYS):
        selected = proposal_changed[..., state_index]
        per_state[name] = {
            "n_changed": int(selected.sum()),
            "direction_accuracy": accuracy(
                proposal_prediction[..., state_index],
                proposal_truth[..., state_index],
                selected,
            ),
        }
    return {
        "future_direction_accuracy_all": accuracy(
            direction_prediction, direction_truth, direction_mask
        ),
        "future_direction_accuracy_changed_only": accuracy(
            direction_prediction, direction_truth, changed_mask
        ),
        "rule_proposal_direction_accuracy_all": accuracy(
            proposal_prediction, proposal_truth, proposal_mask
        ),
        "rule_proposal_direction_accuracy_changed_only": accuracy(
            proposal_prediction, proposal_truth, proposal_changed
        ),
        "rule_proposal_by_state": per_state,
        "status_accuracy": accuracy(
            status_prediction, status_truth, pair_valid
        ),
        "status_by_class": status_counts,
        "proof_path": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        },
    }


def evaluate_model(model, dataset, indices, device):
    prediction, risk = predict_split(model, dataset, indices, device)
    states = dataset["states"][indices]
    actions = dataset["actions"][indices]
    action_events = dataset["action_events"][indices]
    histories = dataset["histories"][indices]
    time_deltas = dataset["time_deltas"][indices]
    valid = dataset["valid"][indices]
    alive = dataset["alive"][indices]
    horizons = [0, min(5, prediction.shape[2] - 1), prediction.shape[2] - 1]
    factual = {}
    for horizon in horizons:
        mask = valid[:, :, horizon] > 0
        factual[f"{horizon + 1}_steps"] = masked_physical_mae(
            prediction[:, :, horizon], states[:, :, horizon + 1], mask
        )

    final_mask = valid[:, :, -1] > 0
    factual_mse = ((prediction[:, :, -1] - states[:, :, -1]) ** 2)[final_mask].mean()
    flat_actions = actions.reshape(-1, actions.shape[2], A_DIM).copy()
    rng = np.random.default_rng(991)
    permutation = rng.permutation(len(flat_actions))
    shuffled = flat_actions[permutation]
    shuffled_events = action_events.reshape(
        -1, action_events.shape[2], TREATMENT_EVENT_DIM
    )[permutation]
    initial = states[:, :, 0].reshape(-1, S_DIM)
    initial_history = histories[:, :, 0].reshape(-1, H_DIM)
    flat_time_deltas = time_deltas.reshape(-1, time_deltas.shape[2])
    with torch.no_grad():
        shuffled_prediction, _, _ = model.rollout(
            torch.as_tensor(initial, dtype=torch.float32, device=device),
            torch.as_tensor(shuffled, dtype=torch.float32, device=device),
            torch.as_tensor(initial_history, dtype=torch.float32, device=device),
            delta_hours=torch.as_tensor(
                flat_time_deltas, dtype=torch.float32, device=device
            ),
            treatment_events=torch.as_tensor(
                shuffled_events, dtype=torch.float32, device=device
            ),
        )
    shuffled_prediction = shuffled_prediction[:, -1].cpu().numpy().reshape(
        states.shape[0], states.shape[1], S_DIM
    )
    shuffled_mse = ((shuffled_prediction - states[:, :, -1]) ** 2)[final_mask].mean()

    risk_mask = valid > 0
    death_target = 1.0 - alive
    risk_brier = ((risk - death_target) ** 2)[risk_mask].mean()
    predicted_death = risk >= 0.5
    actual_death = death_target > 0.5
    true_positive = (predicted_death & actual_death & risk_mask).sum()
    false_negative = ((~predicted_death) & actual_death & risk_mask).sum()
    true_negative = ((~predicted_death) & (~actual_death) & risk_mask).sum()
    false_positive = (predicted_death & (~actual_death) & risk_mask).sum()
    return {
        "factual_rollout_mae": factual,
        "action_conditioning": {
            "factual_normalized_mse_6h": round(float(factual_mse), 6),
            "shuffled_action_normalized_mse_6h": round(float(shuffled_mse), 6),
            "shuffled_minus_factual": round(float(shuffled_mse - factual_mse), 6),
            "uses_interventions": bool(shuffled_mse > factual_mse),
        },
        "counterfactual": counterfactual_metrics(prediction, states, valid),
        "osler_validator": osler_validator_audit(
            prediction, states, actions, time_deltas, valid
        ),
        "symbolic_heads": symbolic_head_metrics(
            model, dataset, indices, device
        ),
        "death_risk": {
            "brier": round(float(risk_brier), 6),
            "recall": round(float(true_positive / max(true_positive + false_negative, 1)), 4),
            "specificity": round(float(true_negative / max(true_negative + false_positive, 1)), 4),
        },
        "collapse": collapse_metrics(
            model,
            states[:, :, 0].reshape(-1, S_DIM),
            histories[:, :, 0].reshape(-1, H_DIM),
            device,
        ),
    }


@torch.no_grad()
def evaluate_mimic_proxy(model, path, device):
    try:
        import pandas as pd
    except ImportError:
        return {"available": False, "reason": "pandas unavailable"}
    path = Path(path)
    if not path.exists():
        return {"available": False, "reason": f"missing {path}"}

    frame = pd.read_parquet(path)
    names = (
        "glucose", "ph", "bicarbonate", "anion_gap", "potassium", "map",
        "sodium", "osmolality", "creatinine", "urine_output", "BHB",
    )
    errors = {name: [] for name in names}
    persistence = {name: [] for name in names}
    active_errors = {name: [] for name in names}
    active_persistence = {name: [] for name in names}
    action_rates = {name: [] for name in ACTION_KEYS}
    used = 0
    active_used = 0
    target_map = {
        "glucose": ("glucose_t", "glucose_tp6", 0),
        "ph": ("ph_t", "ph_tp6", 1),
        "bicarbonate": ("bicarbonate_t", "bicarbonate_tp6", 2),
        "anion_gap": ("anion_gap_t", "anion_gap_tp6", 3),
        "potassium": ("potassium_t", "potassium_tp6", 4),
        "map": ("map_t", "map_tp6", 5),
        "sodium": ("sodium_t", "sodium_tp6", 8),
        "osmolality": ("osmolality_t", "osmolality_tp6", 9),
        "creatinine": ("creatinine_t", "creatinine_tp6", 10),
        "urine_output": ("urine_output_t", "urine_output_tp6", 11),
        "BHB": ("BHB_t", "BHB_tp6", 12),
    }
    for _, row in frame.iterrows():
        glucose = row.get("glucose_t")
        potassium = row.get("potassium_t")
        map_value = row.get("map_t")
        bicarbonate = row.get("bicarbonate_t")
        if any(np.isnan(value) for value in (glucose, potassium, map_value, bicarbonate)):
            continue
        ph = row.get("ph_t")
        ph_observed = not np.isnan(ph)
        if np.isnan(ph):
            ph = henderson(float(bicarbonate))
        anion_gap = row.get("anion_gap_t")
        anion_gap_observed = not np.isnan(anion_gap)
        if np.isnan(anion_gap):
            anion_gap = S_MEAN[3]
        volume = np.clip((float(map_value) - 30.0) / 60.0 * 15.0, 5.0, 18.0)
        state = {
            "G": glucose, "pH": ph, "HCO3": bicarbonate,
            "anion_gap": anion_gap, "Ke": potassium, "MAP": map_value,
            "V": volume, "I": 1.0,
        }
        sodium = row.get("sodium_t", 138.0)
        sodium_observed = not np.isnan(sodium)
        if np.isnan(sodium):
            sodium = 138.0
        creatinine = row.get("creatinine_t", 1.2)
        creatinine_observed = not np.isnan(creatinine)
        if np.isnan(creatinine):
            creatinine = 1.2
        osmolality = row.get("osmolality_t", np.nan)
        osmolality_observed = not np.isnan(osmolality)
        if np.isnan(osmolality):
            osmolality = row.get("osmolality_derived_t", np.nan)
        if np.isnan(osmolality):
            osmolality = 2.0 * float(sodium) + float(glucose) / 18.0
        urine_output = row.get("urine_output_t", 100.0)
        urine_output_observed = not np.isnan(urine_output)
        if np.isnan(urine_output):
            urine_output = 100.0
        bhb = row.get("BHB_t", np.nan)
        bhb_observed = not np.isnan(bhb)
        if np.isnan(bhb):
            bhb = max(0.0, float(anion_gap) - 12.0) * 0.75
        state.update({
            "Na": float(sodium),
            "osmolality": float(osmolality),
            "creatinine": float(creatinine),
            "urine_output": float(urine_output),
            "BHB": float(bhb),
            "K_store": float(np.clip(
                120.0 - 45.0 * max(0.0, 7.35 - float(ph))
                - 8.0 * max(0.0, float(creatinine) - 1.2),
                55.0, 145.0,
            )),
            "osmotic_injury": 0.0,
        })
        reported_state = {
            "G": float(glucose), "HCO3": float(bicarbonate),
            "Ke": float(potassium), "MAP": float(map_value),
        }
        for observed, name, value in (
            (ph_observed, "pH", ph),
            (anion_gap_observed, "anion_gap", anion_gap),
            (sodium_observed, "Na", sodium),
            (osmolality_observed, "osmolality", osmolality),
            (creatinine_observed, "creatinine", creatinine),
            (urine_output_observed, "urine_output", urine_output),
            (bhb_observed, "BHB", bhb),
        ):
            if observed:
                reported_state[name] = float(value)
        age_map = {
            model_name: float(row[column])
            for model_name, column in (
                ("G", "glucose_age_hr"), ("pH", "ph_age_hr"),
                ("HCO3", "bicarbonate_age_hr"),
                ("anion_gap", "anion_gap_age_hr"),
                ("Ke", "potassium_age_hr"), ("MAP", "map_age_hr"),
                ("Na", "sodium_age_hr"),
                ("osmolality", "osmolality_age_hr"),
                ("creatinine", "creatinine_age_hr"),
                ("urine_output", "urine_output_age_hr"),
                ("BHB", "BHB_age_hr"),
            )
            if column in frame and not pd.isna(row.get(column))
        }
        _, observed_mask, observed_age = observation_context(
            reported_state, age_map
        )
        detailed_actions = "future_action_grid" in frame.columns
        if detailed_actions and isinstance(row.get("future_action_grid"), str):
            physical_sequence = np.asarray(
                json.loads(row["future_action_grid"]), dtype=np.float32
            )
            for action_index, action_name in enumerate(
                ACTION_KEYS[:physical_sequence.shape[1]]
            ):
                positive = physical_sequence[:, action_index]
                action_rates[action_name].extend(
                    positive[positive > 0].astype(float).tolist()
                )
            sequence = np.asarray([a2vec(action) for action in physical_sequence])
            if isinstance(row.get("future_treatment_event_grid"), str):
                event_sequence = np.asarray(
                    json.loads(row["future_treatment_event_grid"]),
                    dtype=np.float32,
                )
            else:
                event_sequence = treatment_event_features(physical_sequence)
        else:
            action = np.array([
                4.0 if row.get("act_insulin", 0) else 0.0,
                250.0 if row.get("act_fluids", 0) else 0.0,
                10.0 if (row.get("act_kcl", 0) or row.get("act_potassium", 0)) else 0.0,
                25.0 if row.get("act_bicarbonate", 0) else 0.0,
                5.0 if row.get("act_dextrose", 0) else 0.0,
            ], dtype=np.float32)
            sequence = np.repeat(a2vec(action)[None, :], 12, axis=0)
            event_sequence = treatment_event_features(
                np.repeat(expand_action(action)[None, :], 12, axis=0)
            )
        history = None
        if "history_action_grid" in frame.columns and isinstance(
            row.get("history_action_grid"), str
        ):
            history_grid = np.asarray(
                json.loads(row["history_action_grid"]), dtype=np.float32
            )
            history = treatment_history_features(history_grid, DT, HISTORY_HOURS)
        predicted, _, _ = model.rollout(
            torch.as_tensor(s2vec(state), dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0),
            None if history is None else torch.as_tensor(
                history, dtype=torch.float32, device=device
            ).unsqueeze(0),
            observation_mask=torch.as_tensor(
                observed_mask, dtype=torch.float32, device=device
            ).unsqueeze(0),
            observation_age=torch.as_tensor(
                observed_age, dtype=torch.float32, device=device
            ).unsqueeze(0),
            treatment_events=torch.as_tensor(
                event_sequence, dtype=torch.float32, device=device
            ).unsqueeze(0),
        )
        physical = predicted[0, -1].cpu().numpy() * S_STD + S_MEAN
        row_used = False
        row_active_used = False
        active_dka = row.get("dka_active_t", np.nan)
        if pd.isna(active_dka):
            active_dka = (
                float(glucose) >= 200.0
                and (float(bicarbonate) < 18.0 or float(anion_gap) > 12.0)
            ) or float(bhb) >= 3.0
        for name, (current_column, target_column, index) in target_map.items():
            target = row.get(target_column)
            current = row.get(current_column)
            if name == "osmolality":
                if target is None or np.isnan(target):
                    target = row.get("osmolality_derived_tp6", np.nan)
                if current is None or np.isnan(current):
                    current = row.get("osmolality_derived_t", np.nan)
            if target is None or current is None:
                continue
            if not np.isnan(target) and not np.isnan(current):
                errors[name].append(abs(float(physical[index]) - float(target)))
                persistence[name].append(abs(float(current) - float(target)))
                if active_dka:
                    active_errors[name].append(
                        abs(float(physical[index]) - float(target))
                    )
                    active_persistence[name].append(
                        abs(float(current) - float(target))
                    )
                    row_active_used = True
                row_used = True
        used += int(row_used)
        active_used += int(row_active_used)

    age_columns = [
        "glucose_age_hr", "ph_age_hr", "bicarbonate_age_hr",
        "anion_gap_age_hr", "potassium_age_hr", "map_age_hr",
        "sodium_age_hr", "osmolality_age_hr", "creatinine_age_hr",
        "urine_output_age_hr", "BHB_age_hr",
    ]
    current_columns = [
        "glucose_t", "ph_t", "bicarbonate_t", "anion_gap_t",
        "potassium_t", "map_t", "sodium_t", "osmolality_t",
        "creatinine_t", "urine_output_t", "BHB_t",
    ]
    action_columns = [
        "act_insulin", "act_fluids", "act_potassium", "act_kcl",
        "act_bicarbonate", "act_dextrose",
    ]
    current_available = [column for column in current_columns if column in frame]
    core_inputs = [
        column for column in (
            "glucose_t", "potassium_t", "map_t", "bicarbonate_t",
        ) if column in frame
    ]
    detailed = "future_action_grid" in frame.columns
    lifecycle_events = "future_treatment_event_grid" in frame.columns
    stays = int(frame["stay_id"].nunique())
    return {
        "available": True,
        "rows_with_any_comparison": used,
        "rows_with_active_dka_comparison": active_used,
        "stays": stays,
        "warning": (
            f"Factual check only: {stays} stays; "
            + (
                "dose/time grids and prior history are present, but targets remain "
                "sparse and treatment selection is unadjusted."
                if detailed else
                "binary treatment flags, unknown dose/timing, stale and missing "
                "targets, no prior-treatment history, and no causal adjustment."
            )
        ),
        "data_adequacy": {
            "rows": int(len(frame)),
            "complete_core_input_rows": int(
                frame[core_inputs].notna().all(axis=1).sum()
            ),
            "fully_observed_clinical_state_rows": int(
                frame[current_available].notna().all(axis=1).sum()
            ),
            "median_measurement_age_hours": {
                column.removesuffix("_age_hr"): round(float(frame[column].median()), 3)
                for column in age_columns
                if column in frame and frame[column].notna().any()
            },
            "mean_anchor_state": {
                column.removesuffix("_t"): round(float(frame[column].mean()), 3)
                for column in current_columns
                if column in frame and frame[column].notna().any()
            },
            "positive_action_rows": {
                column.removeprefix("act_"): int(frame[column].fillna(0).sum())
                for column in action_columns if column in frame
            },
            "has_exact_dose_and_timing": detailed,
            "has_explicit_start_stop_events": lifecycle_events,
            "has_pre_anchor_treatment_history": bool("history_action_grid" in frame.columns),
            "sufficient_for_causal_validation": False,
        },
        "action_rate_support": {
            name: {
                "n_active_cells": len(values),
                "median": round(float(np.median(values)), 4) if values else None,
                "p95": round(float(np.quantile(values, 0.95)), 4) if values else None,
                "max": round(float(np.max(values)), 4) if values else None,
                "training_max": float(A_SCALE[index]),
                "fraction_above_training_max": round(
                    float(np.mean(np.asarray(values) > A_SCALE[index])), 4
                ) if values else None,
            }
            for index, (name, values) in enumerate(action_rates.items())
        },
        "mae": {
            name: {
                "n": len(values),
                "jepa": round(float(np.mean(values)), 4) if values else None,
                "persistence": round(float(np.mean(persistence[name])), 4)
                if persistence[name] else None,
            }
            for name, values in errors.items()
        },
        "mae_active_dka": {
            name: {
                "n": len(values),
                "jepa": round(float(np.mean(values)), 4) if values else None,
                "persistence": round(float(np.mean(active_persistence[name])), 4)
                if active_persistence[name] else None,
            }
            for name, values in active_errors.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="dka_intervention_jepa.pt")
    parser.add_argument("--report", default="dka_intervention_jepa_report.json")
    parser.add_argument("--mimic", default="dka_transitions_6h.parquet")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device: {device}")
    print(
        f"generating {args.scenarios} patient scenarios x {len(PROTOCOLS)} "
        f"intervention branches x {args.sequence_length} steps"
    )
    started = time.time()
    dataset = generate_branched_dataset(args.scenarios, args.sequence_length, args.seed)
    splits = split_scenarios(args.scenarios, args.seed)
    print(
        f"split by scenario: train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, test={len(splits['test'])}"
    )
    print(f"simulated deaths by cause: {dataset['death_causes']}")
    simulator_audit = simulator_calibration_audit(dataset)

    model = WorldModel()
    history, best_validation = train_model(
        model, dataset, splits, args.epochs, args.batch_size,
        args.learning_rate, device,
    )
    test_report = evaluate_model(model, dataset, splits["test"], device)
    mimic_report = evaluate_mimic_proxy(model, args.mimic, device)
    report = {
        "model": "Symbolic-grounded Osler intervention DKA JEPA",
        "trained_at_unix": int(time.time()),
        "device": str(device),
        "seed": args.seed,
        "scenario_count": args.scenarios,
        "protocols": list(PROTOCOLS),
        "state_ontology_version": OSLER_STATE_ONTOLOGY.version,
        "symbolic_interface": symbolic_schema(
            STATE_KEYS, ACTION_KEYS, OSLER_STATE_ONTOLOGY
        ),
        "simulator": {
            "version": "temporal_causal_audit_dka_v6",
            "patient_domain_randomization": [
                "weight_kg", "renal_reserve", "insulin_sensitivity",
                "counterregulatory_drive", "fluid_retention",
                "vascular_tone", "potassium_store_scale",
                "endogenous_insulin", "baseline_sodium",
                "baseline_creatinine",
            ],
            "branch_point_warmup_steps": [0, 12],
            "profile_parameters_are_research_priors": True,
            "state_variables": list(STATE_KEYS),
            "actions": [
                *ACTION_KEYS,
            ],
            "history_hours": HISTORY_HOURS,
            "hidden_belief_states": ["K_store", "osmotic_injury"],
            "observation_contract": {
                "mask": "per-state observed indicator",
                "age_hours": True,
                "training_dropout_probability": 0.25,
                "potassium_store_filter": "predict_update_gaussian_belief",
            },
            "time_intervals_hours": [0.25, 0.5, 0.75, 1.0],
            "insulin_pk": [
                "iv", "rapid_subcutaneous", "intermediate_nph", "basal",
            ],
            "calibration_audit": simulator_audit,
        },
        "curriculum": [
            {"name": stage.name, "end_fraction": stage.end_fraction,
             "weights": stage.weights}
            for stage in STAGES
        ],
        "sequence_length": args.sequence_length,
        "hours_predicted_range": [
            args.sequence_length * 0.25,
            args.sequence_length * 1.0,
        ],
        "split_by_scenario": {name: len(value) for name, value in splits.items()},
        "best_validation_loss": best_validation,
        "test": test_report,
        "mimic_external_proxy": mimic_report,
        "training_seconds": round(time.time() - started, 2),
        "last_training_epoch": history[-1],
    }
    save_checkpoint(model, args.checkpoint, metadata=report)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== held-out intervention evaluation ===")
    print(json.dumps(test_report, indent=2))
    print("\n=== MIMIC external proxy ===")
    print(json.dumps(mimic_report, indent=2))
    print(f"\nsaved checkpoint: {args.checkpoint}")
    print(f"saved report:     {args.report}")


if __name__ == "__main__":
    main()
