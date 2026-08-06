"""Audit numerical floors, ceilings, and denominator guards in DKABody.

This is a simulator-integrity check, not a clinical validation. It answers:

1. Which state variables are hard-clamped?
2. Which clamped variables can influence denominators or terminal logic?
3. During real-action replay, which bounds are actually hit?

The goal is to keep numerical artifacts separate from physiology. A bound hit is
not automatically a bug; an unprotected bound used as a denominator is.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dka_action_contract import ACTION_KEYS
from dka_body import DKABody
from dka_fidelity_replay import DT, init_body


BOUND_REGISTRY = {
    "G": {
        "floor": 0.0,
        "ceiling": None,
        "uses": ["glucose kinetics", "effective osmolality", "critical burden"],
        "denominator_role": "none",
        "terminal_role": "non-terminal burden only",
        "guard": "extreme hyperglycemia excluded from terminal causes",
    },
    "I": {
        "floor": 0.0,
        "ceiling": None,
        "uses": ["insulin effect"],
        "denominator_role": "none",
        "terminal_role": "none",
        "guard": "saturating insulin_effect I/(I+I50)",
    },
    "Ket": {
        "floor": 0.0,
        "ceiling": None,
        "uses": ["anion gap", "BHB", "acid-base"],
        "denominator_role": "none",
        "terminal_role": "through acidosis burden only",
        "guard": "no direct terminal ketone switch",
    },
    "HCO3": {
        "floor": 0.1,
        "ceiling": None,
        "uses": ["pH via Henderson-Hasselbalch", "acidosis burden"],
        "denominator_role": "pH calculation floor",
        "terminal_role": "terminal if sustained pH<6.8",
        "guard": "pH uses hco3 floor and burden requires duration",
    },
    "Ke": {
        "floor": 0.0,
        "ceiling": None,
        "uses": ["potassium critical burden", "reward"],
        "denominator_role": "none",
        "terminal_role": "terminal if sustained hypo/hyperkalemia",
        "guard": "insulin shift floor and pH-corrected equilibrium",
    },
    "Ki": {
        "floor": 0.0,
        "ceiling": None,
        "uses": ["potassium store ratio", "serum K equilibrium"],
        "denominator_role": "ratio numerator only",
        "terminal_role": "none",
        "guard": "store_ratio clipped before use",
    },
    "V": {
        "floor": 1.0,
        "ceiling": None,
        "uses": ["MAP", "renal perfusion", "concentrations", "sodium balance"],
        "denominator_role": "partly protected by effective distribution volumes",
        "terminal_role": "terminal through sustained MAP<40",
        "guard": "glucose/HCO3/K use effective volumes; dNa remains clipped",
    },
    "Na": {
        "floor": 100.0,
        "ceiling": 180.0,
        "uses": ["effective osmolality", "osmotic burden"],
        "denominator_role": "none",
        "terminal_role": "reported osmotic burden only",
        "guard": "dNa hourly rate clipped; osmotic injury non-terminal",
    },
    "Cr": {
        "floor": 0.1,
        "ceiling": None,
        "uses": ["renal marker", "potassium-store prior outside live body"],
        "denominator_role": "none",
        "terminal_role": "none",
        "guard": "lagged proxy only",
    },
    "counterregulatory_stress": {
        "floor": 0.05,
        "ceiling": 2.5,
        "uses": ["hepatic glucose output", "ketogenesis"],
        "denominator_role": "none",
        "terminal_role": "none",
        "guard": "bounded hidden drive",
    },
    "renal_perfusion_state": {
        "floor": 0.0,
        "ceiling": 1.0,
        "uses": ["renal glucose/ketone/K clearance", "urine flow", "creatinine"],
        "denominator_role": "none",
        "terminal_role": "none",
        "guard": "bounded lag state",
    },
    "osmotic_injury": {
        "floor": 0.0,
        "ceiling": None,
        "uses": ["reported burden"],
        "denominator_role": "none",
        "terminal_role": "disabled",
        "guard": "OSMOTIC_INJURY_TERMINAL=False",
    },
}

DENOMINATOR_GUARDS = {
    "concentration_volume": {
        "depends_on": "V",
        "floor_formula": "max(V, 0.65 * volume_setpoint)",
        "protects": ["glucose", "dextrose", "nutrition carbohydrate", "dilution"],
        "status": "protected",
    },
    "electrolyte_distribution_volume": {
        "depends_on": "V",
        "floor_formula": "max(V, 0.55 * volume_setpoint)",
        "protects": ["serum potassium KCl effect", "renal K serum effect"],
        "status": "protected",
    },
    "bicarbonate_distribution_volume": {
        "depends_on": "V",
        "floor_formula": "max(V, 0.50 * weight_kg)",
        "protects": ["bicarbonate bolus effect"],
        "status": "protected",
    },
    "sodium_balance_volume": {
        "depends_on": "V",
        "floor_formula": "actual V, but dNa is clipped to +/-1.5 mEq/L/hr",
        "protects": ["sodium concentration update"],
        "status": "bounded_not_distribution_protected",
    },
}


def near_lower(value, floor, margin):
    return floor is not None and float(value) <= float(floor) + margin


def near_upper(value, ceiling, margin):
    return ceiling is not None and float(value) >= float(ceiling) - margin


def bound_margin(name, bound):
    if name in ("G", "HCO3", "Ke", "Cr", "counterregulatory_stress"):
        return 0.05
    if name in ("V", "Na", "Ki"):
        return 0.5
    if name == "renal_perfusion_state":
        return 0.01
    return 1e-6


def observe_bound_hits(body):
    obs = body.observe()
    values = {
        "G": obs["G"],
        "I": obs["I"],
        "Ket": obs["anion_gap"] - 12.0,
        "HCO3": obs["HCO3"],
        "Ke": obs["Ke"],
        "Ki": obs["K_store"],
        "V": obs["V"],
        "Na": obs["Na"],
        "Cr": obs["creatinine"],
        "counterregulatory_stress": obs["counterregulatory_stress"],
        "renal_perfusion_state": obs["renal_perfusion_state"],
        "osmotic_injury": obs["osmotic_injury"],
    }
    hits = []
    for name, registry in BOUND_REGISTRY.items():
        margin = bound_margin(name, registry)
        if near_lower(values[name], registry["floor"], margin):
            hits.append((name, "floor", values[name]))
        if near_upper(values[name], registry["ceiling"], margin):
            hits.append((name, "ceiling", values[name]))
    return hits


def replay_bound_hits(trajectory):
    body = init_body(trajectory["init"], trajectory.get("history_actions"))
    actions = {
        int(round(float(action.get("t", 0.0)) / DT)): action
        for action in trajectory.get("actions", [])
    }
    last_t = max(
        [float(action.get("t", 0.0)) for action in trajectory.get("actions", [])]
        + [float(lab.get("t", 0.0)) for lab in trajectory.get("labs", [])]
        + [0.0]
    )
    n = int(np.ceil(last_t / DT)) + 1
    hits = []
    for cell in range(n):
        for name, kind, value in observe_bound_hits(body):
            hits.append({
                "t": round(float(body.t), 3),
                "state": name,
                "bound": kind,
                "value": round(float(value), 5),
                "alive": bool(body.alive),
                "death_cause": body.death_cause,
            })
        if not body.alive:
            break
        action = actions.get(cell, {name: 0.0 for name in ACTION_KEYS})
        body.step(action, dt=DT)
    return hits, body


def summarize_cases(cases):
    state_hits = Counter()
    risky_hits = Counter()
    case_examples = defaultdict(list)
    deaths_with_hits = Counter()
    for case in cases:
        seen = set()
        for hit in case["bound_hits"]:
            key = (hit["state"], hit["bound"])
            state_hits[key] += 1
            seen.add(key)
            registry = BOUND_REGISTRY[hit["state"]]
            if registry["denominator_role"] != "none" or registry["terminal_role"] != "none":
                risky_hits[key] += 1
                if len(case_examples[key]) < 3:
                    case_examples[key].append({
                        "stay_id": case["stay_id"],
                        "t": hit["t"],
                        "value": hit["value"],
                        "death_cause": case["death_cause"],
                    })
        if case["death_cause"]:
            for key in seen:
                deaths_with_hits[key] += 1
    return {
        "state_bound_hit_counts": {
            f"{state}.{bound}": int(count)
            for (state, bound), count in sorted(state_hits.items())
        },
        "risky_bound_hit_counts": {
            f"{state}.{bound}": int(count)
            for (state, bound), count in sorted(risky_hits.items())
        },
        "death_case_bound_hits": {
            f"{state}.{bound}": int(count)
            for (state, bound), count in sorted(deaths_with_hits.items())
        },
        "examples": {
            f"{state}.{bound}": examples
            for (state, bound), examples in sorted(case_examples.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="?", default="/tmp/dka_maintenance_trajectories.jsonl")
    parser.add_argument("--output", default="dka_numeric_artifact_audit_v1.json")
    args = parser.parse_args()

    trajectories = [
        json.loads(line) for line in Path(args.trajectories).read_text().splitlines()
        if line.strip()
    ]
    cases = []
    for trajectory in trajectories:
        hits, body = replay_bound_hits(trajectory)
        cases.append({
            "stay_id": trajectory.get("stay_id"),
            "death_cause": body.death_cause,
            "simulated_alive": bool(body.alive),
            "bound_hit_count": len(hits),
            "bound_hits": hits[:40],
        })
    summary = summarize_cases(cases)
    result = {
        "audit": "DKABody numeric artifact audit",
        "trajectory_source": Path(args.trajectories).name,
        "bound_registry": BOUND_REGISTRY,
        "denominator_guards": DENOMINATOR_GUARDS,
        "summary": {
            "trajectory_count": len(cases),
            "death_count": sum(not case["simulated_alive"] for case in cases),
            **summary,
        },
        "cases": cases,
        "conclusion": (
            "Bounds are acceptable only when they are either reporting guards or "
            "protected before entering concentration denominators. Remaining "
            "bound hits should be interpreted separately from physiology."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "summary": result["summary"],
        "conclusion": result["conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
