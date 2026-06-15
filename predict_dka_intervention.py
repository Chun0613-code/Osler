"""Predict and compare a six-hour DKA intervention with the trained JEPA.

This is a research interface, not a clinical dosing tool.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dka_body import estimate_potassium_store, henderson
from dka_osler import shield_route_aware
from dka_action_contract import ACTION_INDEX, ACTION_KEYS, expand_action
from dka_world_model import (
    A_DIM,
    DT,
    S_MEAN,
    S_STD,
    STATE_KEYS,
    A_SCALE,
    HISTORY_HOURS,
    a2vec,
    load_checkpoint,
    observation_context,
    s2vec,
)
from osler_jepa.belief import PotassiumStoreBelief
from osler_jepa.actions import (
    TREATMENT_EVENT_KEYS,
    treatment_event_features,
)
from osler_jepa.validator import OSLER_DKA_VALIDATOR
from osler_jepa.embodied_logic import OSLER_DKA_PROLOG
from osler_jepa.ontology import OSLER_STATE_ONTOLOGY
from osler_jepa.symbolic import DIRECTION_NAMES, RULE_IDS, STATUS_NAMES


def physical_state(normalized):
    values = np.asarray(normalized, dtype=np.float32) * S_STD + S_MEAN
    state = {name: float(value) for name, value in zip(STATE_KEYS, values)}
    state["G"] = max(0.0, state["G"])
    state["HCO3"] = max(0.1, state["HCO3"])
    state["anion_gap"] = max(0.0, state["anion_gap"])
    state["Ke"] = max(0.0, state["Ke"])
    state["V"] = max(1.0, state["V"])
    state["I"] = max(0.0, state["I"])
    state["Na"] = min(180.0, max(100.0, state["Na"]))
    state["osmolality"] = max(200.0, state["osmolality"])
    state["creatinine"] = max(0.1, state["creatinine"])
    state["urine_output"] = max(0.0, state["urine_output"])
    state["BHB"] = max(0.0, state["BHB"])
    state["K_store"] = max(0.0, state["K_store"])
    state["osmotic_injury"] = max(0.0, state["osmotic_injury"])
    return {name: round(value, 4) for name, value in state.items()}


@torch.no_grad()
def predict(model, state, action, hours, device, apply_osler=False, history=None,
            observation_age=None, time_deltas=None):
    if time_deltas is None:
        time_deltas = [DT] * max(1, int(round(hours / DT)))
    else:
        time_deltas = [float(value) for value in time_deltas]
        if not time_deltas or any(value <= 0 for value in time_deltas):
            raise ValueError("time_deltas must contain positive intervals")
    steps = len(time_deltas)
    history_tensor = None if history is None else torch.as_tensor(
        history, dtype=torch.float32, device=device
    ).unsqueeze(0)
    belief = PotassiumStoreBelief.from_state(state)
    model_state = dict(state)
    model_state.setdefault("K_store", belief.mean)
    initial_vector, observed_mask, observed_age = observation_context(
        state, observation_age
    )
    initial_vector = s2vec(model_state)
    latent = model.encode_state(
        torch.as_tensor(
            initial_vector, dtype=torch.float32, device=device
        ).unsqueeze(0),
        history_tensor,
        torch.as_tensor(
            observed_mask, dtype=torch.float32, device=device
        ).unsqueeze(0),
        torch.as_tensor(
            observed_age, dtype=torch.float32, device=device
        ).unsqueeze(0),
    )
    current_state = physical_state(initial_vector)
    current_state["K_store"] = round(belief.mean, 4)
    predicted, death_probability, belief_history = [], [], []
    action_schedule, applied_actions = [], []
    previous_action = None
    elapsed_hours = 0.0
    for step in range(steps):
        step_hours = time_deltas[step]
        step_action = list(action)
        trace = []
        if apply_osler:
            step_action, trace = shield_route_aware(current_state, step_action)
            step_action, prolog_trace = OSLER_DKA_PROLOG.gate(
                current_state, step_action
            )
            trace = [
                {"type": "symbolic_policy", "explanation": item}
                for item in trace
            ] + prolog_trace
        event_vector = treatment_event_features(
            [step_action], initial_action=previous_action
        )[0]
        physical_step_action = expand_action(step_action)
        latent = model.predict_latent(
            latent,
            torch.as_tensor(
                a2vec(step_action), dtype=torch.float32, device=device
            ).unsqueeze(0),
            delta_hours=step_hours,
            elapsed_hours=elapsed_hours,
            treatment_events=torch.as_tensor(
                event_vector, dtype=torch.float32, device=device
            ).unsqueeze(0),
        )
        applied_actions.append(step_action)
        normalized = model.D(latent)[0].cpu().numpy()
        current_state = physical_state(normalized)
        belief = belief.step(
            step_action,
            current_state,
            step_hours,
            model_estimate=current_state.get("K_store"),
        )
        current_state["K_store"] = round(belief.mean, 4)
        predicted.append(current_state)
        belief_history.append(belief.to_dict())
        death_probability.append(float(torch.sigmoid(model.R(latent))[0, 0]))
        action_changed = previous_action is None or not np.allclose(
            previous_action, physical_step_action
        )
        if trace or action_changed or event_vector.any():
            action_schedule.append({
                "hours": round(elapsed_hours, 2),
                "action": {
                    key: round(float(value), 4)
                    for key, value in zip(ACTION_KEYS, expand_action(step_action))
                },
                "treatment_events": [
                    name for name, active in zip(TREATMENT_EVENT_KEYS, event_vector)
                    if active > 0
                ],
                "trace": trace,
            })
        previous_action = physical_step_action
        elapsed_hours += step_hours

    sample_steps = sorted({0, min(5, steps - 1), steps - 1})
    elapsed_at_step = np.cumsum(time_deltas)
    trajectory = [{
        "hours": round(float(elapsed_at_step[step]), 2),
        "state": predicted[step],
        "death_probability": round(death_probability[step], 6),
        "belief_states": {
            "total_body_potassium_store": belief_history[step],
        },
    } for step in sample_steps]
    return trajectory, action_schedule, applied_actions


def compare(model, state, proposed_action, hours, device, input_warnings=None,
            history=None, observation_age=None, time_deltas=None):
    effective_hours = (
        float(sum(time_deltas)) if time_deltas is not None else float(hours)
    )
    intervention, action_schedule, applied_actions = predict(
        model, state, proposed_action, hours, device, apply_osler=True,
        history=history, observation_age=observation_age,
        time_deltas=time_deltas,
    )
    untreated, _, _ = predict(
        model, state, np.zeros(A_DIM, dtype=np.float32), hours, device,
        apply_osler=False, history=history, observation_age=observation_age,
        time_deltas=time_deltas,
    )
    treated_final = intervention[-1]["state"]
    untreated_final = untreated[-1]["state"]
    effect = {
        key: round(treated_final[key] - untreated_final[key], 4)
        for key in STATE_KEYS
    }
    applied_durations = np.asarray(
        time_deltas if time_deltas is not None else [DT] * len(applied_actions),
        dtype=np.float32,
    )
    mean_physical_action = np.average(
        np.asarray([expand_action(action) for action in applied_actions]),
        axis=0,
        weights=applied_durations,
    )
    mean_action = mean_physical_action / A_SCALE
    lifecycle_summary = treatment_event_features(applied_actions).max(axis=0)
    osler_validation = OSLER_DKA_VALIDATOR.validate(
        mean_action, treated_final, untreated_final,
        horizon_hours=effective_hours,
    )
    prolog_reasoning = OSLER_DKA_PROLOG.evaluate(
        state=state,
        proposed_action=proposed_action,
        applied_action=mean_physical_action,
        future=treated_final,
        baseline_future=untreated_final,
        elapsed_hours=effective_hours,
    )
    history_tensor = None if history is None else torch.as_tensor(
        history, dtype=torch.float32, device=device
    ).unsqueeze(0)
    initial_belief = PotassiumStoreBelief.from_state(state)
    symbolic_state = dict(state)
    symbolic_state.setdefault("K_store", initial_belief.mean)
    symbolic_vector, symbolic_mask, symbolic_age = observation_context(
        state, observation_age
    )
    symbolic_vector = s2vec(symbolic_state)
    with torch.no_grad():
        context_latent = model.encode_state(
            torch.as_tensor(
                symbolic_vector, dtype=torch.float32, device=device
            ).unsqueeze(0),
            history_tensor,
            torch.as_tensor(
                symbolic_mask, dtype=torch.float32, device=device
            ).unsqueeze(0),
            torch.as_tensor(
                symbolic_age, dtype=torch.float32, device=device
            ).unsqueeze(0),
        )
        symbolic = model.symbolic_outputs(
            context_latent,
            torch.as_tensor(
                mean_action, dtype=torch.float32, device=device
            ).unsqueeze(0),
            delta_hours=effective_hours,
            treatment_events=torch.as_tensor(
                lifecycle_summary, dtype=torch.float32, device=device
            ).unsqueeze(0),
        )
    direction_probability = symbolic["direction_logits"].softmax(dim=-1)[0]
    proposal_probability = symbolic["proposal_logits"].softmax(dim=-1)[0]
    status_probability = symbolic["status_logits"].softmax(dim=-1)[0]
    proof_probability = symbolic["proof_logits"].sigmoid()[0]
    status_index = int(status_probability.argmax())
    symbolic_transition = {
        "schema_version": "1.0.0",
        "predicted_osler_status": STATUS_NAMES[status_index],
        "status_confidence": round(float(status_probability[status_index]), 4),
        "proof_paths": [
            {"rule_id": rule_id, "confidence": round(float(probability), 4)}
            for rule_id, probability in zip(RULE_IDS, proof_probability)
            if float(probability) >= 0.5
        ],
        "transitions": [],
        "candidate_rule_level": "observational_association",
    }
    for index, name in enumerate(STATE_KEYS):
        future_index = int(direction_probability[index].argmax())
        proposal_index = int(proposal_probability[index].argmax())
        symbolic_transition["transitions"].append({
            "variable": OSLER_STATE_ONTOLOGY.require(name),
            "model_variable": name,
            "future_direction": DIRECTION_NAMES[future_index],
            "future_direction_confidence": round(
                float(direction_probability[index, future_index]), 4
            ),
            "intervention_effect_direction": DIRECTION_NAMES[proposal_index],
            "intervention_effect_confidence": round(
                float(proposal_probability[index, proposal_index]), 4
            ),
            "time_window_hours": effective_hours,
        })
    return {
        "research_only": True,
        "input_warnings": input_warnings or [],
        "initial_state": state,
        "proposed_action": dict(zip(
            ACTION_KEYS,
            [float(value) for value in expand_action(proposed_action)],
        )),
        "prior_treatment_history_features": (
            history.tolist() if isinstance(history, np.ndarray) else history
        ),
        "observation_contract": {
            "observed_mask": dict(zip(STATE_KEYS, symbolic_mask.astype(int).tolist())),
            "age_hours": dict(zip(STATE_KEYS, symbolic_age.tolist())),
            "potassium_store_prior": initial_belief.to_dict(),
        },
        "osler_dynamic_action_schedule": action_schedule,
        "effective_action_summary": {
            key: round(float(value), 6)
            for key, value in zip(ACTION_KEYS, mean_physical_action)
        },
        "predicted_intervention_trajectory": intervention,
        "predicted_no_treatment_trajectory": untreated,
        "predicted_effect_at_final_horizon": effect,
        "jepa_symbolic_transition": symbolic_transition,
        "osler_transition_validation": osler_validation,
        "osler_prolog_reasoning": prolog_reasoning,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--glucose", type=float, default=480.0)
    parser.add_argument("--ph", type=float)
    parser.add_argument("--hco3", type=float, default=8.0)
    parser.add_argument("--anion-gap", type=float, default=23.0)
    parser.add_argument("--potassium", type=float, default=5.6)
    parser.add_argument("--map", dest="map_value", type=float, default=78.0)
    parser.add_argument("--volume", type=float, default=12.0)
    parser.add_argument("--insulin-signal", type=float, default=1.0)
    parser.add_argument("--insulin", type=float, default=6.0)
    parser.add_argument(
        "--insulin-formulation",
        choices=("iv", "rapid_sc", "intermediate_sc", "basal_sc"),
        default="iv",
    )
    parser.add_argument("--fluids", type=float, default=500.0)
    parser.add_argument("--kcl", type=float, default=10.0)
    parser.add_argument("--bicarbonate", type=float, default=0.0)
    parser.add_argument("--dextrose", type=float, default=0.0)
    parser.add_argument("--sodium", type=float, default=138.0)
    parser.add_argument("--creatinine", type=float, default=1.2)
    parser.add_argument("--urine-output", type=float, default=100.0)
    parser.add_argument("--bhb", type=float)
    parser.add_argument("--potassium-store", type=float)
    parser.add_argument("--osmotic-injury", type=float, default=0.0)
    for name in ACTION_KEYS:
        parser.add_argument(f"--prior-{name}", type=float, default=0.0)
        parser.add_argument(f"--hours-since-{name}", type=float, default=6.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    derived_ph = henderson(args.hco3)
    ph = args.ph if args.ph is not None else derived_ph
    input_warnings = []
    if abs(ph - derived_ph) > 0.05:
        input_warnings.append(
            f"pH={ph:.3f} is inconsistent with this simulator's HCO3-derived "
            f"pH={derived_ph:.3f}; prediction may be out of distribution."
        )
    state = {
        "G": args.glucose,
        "pH": ph,
        "HCO3": args.hco3,
        "anion_gap": args.anion_gap,
        "Ke": args.potassium,
        "MAP": args.map_value,
        "V": args.volume,
        "I": args.insulin_signal,
        "Na": args.sodium,
        "osmolality": 2.0 * args.sodium + args.glucose / 18.0,
        "creatinine": args.creatinine,
        "urine_output": args.urine_output,
        "BHB": args.bhb if args.bhb is not None else max(0.0, args.anion_gap - 12.0) * 0.75,
        "K_store": args.potassium_store if args.potassium_store is not None else
        estimate_potassium_store(
            args.potassium, ph, args.creatinine, args.urine_output
        ),
        "osmotic_injury": args.osmotic_injury,
    }
    formulation_key = {
        "iv": "insulin_iv",
        "rapid_sc": "insulin_rapid_sc",
        "intermediate_sc": "insulin_intermediate_sc",
        "basal_sc": "insulin_basal_sc",
    }[args.insulin_formulation]
    action = expand_action({
        formulation_key: args.insulin,
        "fluids": args.fluids,
        "kcl": args.kcl,
        "bicarbonate": args.bicarbonate,
        "dextrose": args.dextrose,
    })
    totals = np.array([
        getattr(args, f"prior_{name}") for name in ACTION_KEYS
    ], dtype=np.float32)
    recency = np.array([
        getattr(args, f"hours_since_{name}") for name in ACTION_KEYS
    ], dtype=np.float32)
    history = np.concatenate([
        totals / (A_SCALE * HISTORY_HOURS),
        np.clip(recency / HISTORY_HOURS, 0.0, 1.0),
    ]).astype(np.float32)
    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    print(json.dumps(
        compare(
            model, state, action, args.hours, args.device, input_warnings,
            history=history,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
