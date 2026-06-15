"""Research-only DKA what-if simulator for Osler shadow mode.

This script deliberately does not compare against persistence.  It asks a
different question: given one initial DKA state, how do fixed candidate action
policies differ from a no-treatment probe inside the grey-box simulator?

The output is an explanation artifact for research and planning simulations. It
is not a factual forecast, causal estimate, dose recommendation, or live Osler
ranking signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dka_action_contract import ACTION_KEYS, expand_action
from dka_fidelity_replay import init_body
from osler_jepa.greybox_residual import GreyBoxResidualRuntime


DT = 0.5
REPORT_STATES = (
    "G", "pH", "HCO3", "anion_gap", "Ke", "MAP", "Na", "osmolality",
    "creatinine", "urine_output", "BHB", "K_store", "osmotic_injury",
)


def load_trajectory(path, index=None):
    if path is None or not Path(path).exists():
        return {
            "stay_id": "synthetic_contract_example",
            "init": {
                "glucose": 420.0, "HCO3": 10.0, "K": 4.6,
                "anion_gap": 24.0, "Na": 138.0, "creatinine": 1.2,
                "MAP": 76.0,
            },
            "history_actions": [],
            "actions": [],
        }
    trajectories = [
        json.loads(line) for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if index is not None:
        return trajectories[int(index)]
    for trajectory in trajectories:
        init = trajectory.get("init", {})
        glucose = init.get("glucose")
        bicarbonate = init.get("HCO3")
        potassium = init.get("K")
        sodium = init.get("Na")
        if all(value is not None for value in (glucose, bicarbonate, potassium, sodium)):
            if glucose >= 250.0 and bicarbonate <= 18.0 and 3.3 <= potassium <= 5.8 and sodium < 150.0:
                return trajectory
    for trajectory in trajectories:
        init = trajectory.get("init", {})
        if all(init.get(key) is not None for key in ("glucose", "HCO3", "K")):
            return trajectory
    raise ValueError("no trajectory with glucose/HCO3/K initial observations")


def observed_actions(trajectory, horizon_hours):
    cells = int(round(horizon_hours / DT))
    by_cell = {int(round(float(action.get("t", 0.0)) / DT)): action
               for action in trajectory.get("actions", [])}
    result = []
    for cell in range(cells):
        action = {name: 0.0 for name in ACTION_KEYS}
        action.update(by_cell.get(cell, {}))
        result.append(action)
    return result


def policy_action(name, observation):
    action = {key: 0.0 for key in ACTION_KEYS}
    if name == "no_treatment":
        return action, []
    if name == "fluids_only":
        action["fluids"] = 250.0
    elif name == "potassium_first":
        action["fluids"] = 100.0
        action["kcl"] = 20.0
    elif name == "osler_safety_insulin":
        action.update({"insulin_iv": 4.0, "fluids": 250.0, "kcl": 10.0})
    elif name == "insulin_fluids_dextrose":
        action.update({"insulin_iv": 4.0, "fluids": 250.0, "kcl": 10.0})
        if observation["G"] < 250.0 and observation["HCO3"] < 18.0:
            action["dextrose"] = 5.0
    else:
        raise KeyError(f"unknown protocol: {name}")

    shield = []
    if observation["Ke"] < 3.3 and action["insulin_iv"] > 0:
        action["insulin_iv"] = 0.0
        action["kcl"] = max(action["kcl"], 20.0)
        shield.append("hypokalemia_blocks_insulin_and_prioritizes_kcl")
    if observation["G"] < 120.0 and action["insulin_iv"] > 0:
        action["insulin_iv"] *= 0.25
        shield.append("low_glucose_reduces_insulin_probe")
    return action, shield


def summarize_state(observation):
    return {
        key: round(float(observation[key]), 4)
        for key in REPORT_STATES if key in observation and np.isfinite(observation[key])
    }


def simulate_policy(trajectory, protocol, horizon_hours, residual_model=None):
    body = init_body(trajectory["init"], trajectory.get("history_actions"))
    body.residual_model = residual_model
    cells = int(round(horizon_hours / DT))
    observed = observed_actions(trajectory, horizon_hours)
    curve = [{"t": 0.0, "state": summarize_state(body.observe())}]
    shield_events = []
    action_totals = {name: 0.0 for name in ACTION_KEYS}

    for cell in range(cells):
        if protocol == "observed_treatment":
            action = observed[cell]
            shield = []
        else:
            action, shield = policy_action(protocol, body.observe())
        for name in ACTION_KEYS:
            action_totals[name] += float(expand_action(action)[ACTION_KEYS.index(name)]) * DT
        if shield:
            shield_events.append({"t": round(cell * DT, 3), "events": shield})
        body.step(action, dt=DT)
        if (cell + 1) % int(round(1.0 / DT)) == 0 or cell == cells - 1:
            curve.append({
                "t": round((cell + 1) * DT, 3),
                "state": summarize_state(body.observe()),
            })
        if not body.alive:
            break
    return {
        "protocol": protocol,
        "alive": bool(body.alive),
        "death_cause": body.death_cause,
        "final_state": summarize_state(body.observe()),
        "action_totals": {
            name: round(total, 4) for name, total in action_totals.items()
            if abs(total) > 1e-9
        },
        "safety_shield_events": shield_events,
        "trajectory_hourly": curve,
    }


def effect_vs_baseline(result, baseline):
    effect = {}
    for key, value in result["final_state"].items():
        if key in baseline["final_state"]:
            effect[key] = round(float(value - baseline["final_state"][key]), 4)
    return effect


def mechanism_tags(protocol):
    tags = {
        "no_treatment": ["baseline comparator only"],
        "observed_treatment": ["replays captured EHR treatment"],
        "fluids_only": ["volume expansion", "renal perfusion support"],
        "potassium_first": ["KCl replenishes potassium store", "insulin held by design"],
        "osler_safety_insulin": [
            "insulin should lower glucose/ketogenesis",
            "KCl offsets insulin-driven extracellular potassium shift",
            "fluids support perfusion and osmolar clearance",
        ],
        "insulin_fluids_dextrose": [
            "insulin continued with glucose guard",
            "dextrose probe activates only when glucose is low during ongoing acidosis",
        ],
    }
    return tags.get(protocol, [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", default="dka_fidelity_demo_v4.jsonl")
    parser.add_argument("--trajectory-index", type=int)
    parser.add_argument("--greybox-residual", default="dka_greybox_residual_candidate_v1.pt")
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--output", default="dka_counterfactual_shadow_demo_v1.json")
    args = parser.parse_args()

    trajectory = load_trajectory(args.trajectory_jsonl, args.trajectory_index)
    residual = (
        GreyBoxResidualRuntime.load(args.greybox_residual)
        if args.greybox_residual and Path(args.greybox_residual).exists() else None
    )
    protocols = [
        "no_treatment", "observed_treatment", "fluids_only",
        "potassium_first", "osler_safety_insulin", "insulin_fluids_dextrose",
    ]
    simulations = [
        simulate_policy(trajectory, protocol, args.horizon_hours, residual)
        for protocol in protocols
    ]
    baseline = simulations[0]
    for simulation in simulations:
        simulation["effect_vs_no_treatment"] = effect_vs_baseline(simulation, baseline)
        simulation["mechanism_tags"] = mechanism_tags(simulation["protocol"])

    result = {
        "record_type": "counterfactual_shadow_demo",
        "contract_version": "1.0.0",
        "mode": "research_what_if_shadow",
        "trajectory_source": Path(args.trajectory_jsonl).name,
        "stay_id": trajectory.get("stay_id"),
        "horizon_hours": args.horizon_hours,
        "simulator": {
            "base": "DKABody",
            "greybox_residual": Path(args.greybox_residual).name if residual else None,
            "greybox_candidate_only": bool(residual),
        },
        "uses_persistence_as_judge": False,
        "decision_authority": False,
        "affects_live_recommendation": False,
        "clinical_dose_claim_allowed": False,
        "causal_claim_allowed": False,
        "initial_state": trajectory.get("init", {}),
        "baseline_protocol": "no_treatment",
        "simulations": simulations,
        "interpretation": (
            "Use these traces as a counterfactual explanation and planning-simulator "
            "artifact only. Observational MIMIC demo data cannot validate the causal "
            "effect of these probes."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
