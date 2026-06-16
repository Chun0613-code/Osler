"""Audit simulated deaths as hard-mechanism falsification cases.

The purpose is not to change DKABody.  It replays real trajectories, records the
first viability threshold crossings, and maps each simulated death to the hard
mechanism that grey-box residuals are intentionally not allowed to edit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dka_action_contract import ACTION_KEYS, expand_action
from dka_body import CRITICAL_LIMITS, OSM_INJURY_DEATH, OSMOTIC_INJURY_TERMINAL
from dka_body import NON_TERMINAL_CRITICAL_CAUSES
from dka_fidelity_replay import DT, VAR2ATTR, init_body


SUBSYSTEM_BY_CAUSE = {
    "cumulative hyperosmolar injury": "osmotic_injury",
    "circulatory collapse (MAP<40)": "volume_map",
    "hypokalemia (K<2.5)": "potassium_mass",
    "hyperkalemia (K>7.0)": "potassium_mass",
    "acidosis (pH<6.8)": "acid_base",
    "hypoglycemia (G<40)": "glucose_insulin",
    "extreme hyperglycemia (G>1400)": "glucose_osmotic",
}

HARD_MECHANISM_OWNERS = {
    "osmotic_injury": [
        "effective osmolality calculation",
        "osmotic injury accumulation/recovery",
        "hyperosmolar death threshold",
    ],
    "volume_map": [
        "extracellular volume balance",
        "fluid retention",
        "MAP-volume transfer",
        "renal perfusion lag",
    ],
    "potassium_mass": [
        "total-body potassium reserve",
        "renal potassium loss",
        "insulin/acidosis serum potassium shift",
        "KCl retention into total store",
    ],
    "acid_base": ["ketone/HCO3 coupling", "respiratory compensation floor"],
    "glucose_insulin": ["insulin PK", "glucose uptake"],
    "glucose_osmotic": ["glucose production/clearance", "osmotic diuresis"],
}

REPAIR_BACKLOG = {
    "osmotic_injury": [
        "Separate instantaneous osmolality from injury burden and calibrate injury half-life.",
        "Audit sodium/free-water balance before changing residual dynamics.",
        "Use real-survived high-osmolality stays as negative death labels for the OI threshold.",
    ],
    "volume_map": [
        "Fit volume-to-MAP response using observed MAP trajectory and fluid timing.",
        "Audit fluid retention and renal-perfusion lag against urine output and creatinine.",
    ],
    "potassium_mass": [
        "Trace serum K drop into insulin shift, renal K loss, and KCl replacement terms.",
        "Add a patient-level potassium-store correction using survived hypokalemia replays as constraints.",
        "Do not let grey-box residual directly write total-body K; repair the mass equation itself.",
    ],
}


def classify_death_cause(cause):
    return SUBSYSTEM_BY_CAUSE.get(cause or "survived", "none")


def threshold_excess(cause, observation):
    if cause == "cumulative hyperosmolar injury":
        return max(0.0, float(observation["osmotic_injury"]) - OSM_INJURY_DEATH)
    direction, threshold, _ = CRITICAL_LIMITS[cause]
    values = {
        "acidosis (pH<6.8)": observation["pH"],
        "hypokalemia (K<2.5)": observation["Ke"],
        "hyperkalemia (K>7.0)": observation["Ke"],
        "circulatory collapse (MAP<40)": observation["MAP"],
        "hypoglycemia (G<40)": observation["G"],
        "extreme hyperglycemia (G>1400)": observation["G"],
    }
    value = float(values[cause])
    return max(0.0, threshold - value) if direction == "below" else max(0.0, value - threshold)


def compact_observation(observation):
    keys = (
        "t", "G", "pH", "HCO3", "anion_gap", "Ke", "MAP", "V", "Na",
        "osmolality", "creatinine", "urine_output", "BHB", "K_store",
        "osmotic_injury", "critical_burden",
    )
    return {
        key: round(float(observation[key]), 5)
        for key in keys if key in observation and np.isfinite(observation[key])
    }


def action_by_cell(trajectory):
    return {
        int(round(float(action.get("t", 0.0)) / DT)): action
        for action in trajectory.get("actions", [])
    }


def observed_labs_after(trajectory, death_time):
    if death_time is None:
        return []
    labs = []
    for lab in trajectory.get("labs", []):
        if float(lab.get("t", 0.0)) > float(death_time):
            labs.append({
                "t": round(float(lab["t"]), 3),
                "var": lab.get("var"),
                "value": float(lab["value"]),
            })
    return sorted(labs, key=lambda item: item["t"])


def action_totals(actions, stop_time=None):
    totals = {name: 0.0 for name in ACTION_KEYS}
    for action in actions:
        if stop_time is not None and float(action.get("t", 0.0)) > stop_time:
            continue
        values = expand_action(action)
        for index, name in enumerate(ACTION_KEYS):
            totals[name] += float(values[index]) * DT
    return {name: round(value, 5) for name, value in totals.items() if value > 1e-8}


def audit_trajectory(trajectory):
    body = init_body(trajectory["init"], trajectory.get("history_actions"))
    actions = action_by_cell(trajectory)
    last_t = max(
        [float(action.get("t", 0.0)) for action in trajectory.get("actions", [])]
        + [float(lab.get("t", 0.0)) for lab in trajectory.get("labs", [])]
        + [0.0]
    )
    n = int(np.ceil(last_t / DT)) + 1
    first_crossings = {}
    first_osmolar_risk = None
    curve = [{"t": 0.0, "state": compact_observation(body.observe())}]

    for cell in range(n):
        action = actions.get(cell, {name: 0.0 for name in ACTION_KEYS})
        observation, _, dead, info = body.step(action, dt=DT)
        current_t = float(observation["t"])
        if first_osmolar_risk is None and observation["osmolality"] > 320.0:
            first_osmolar_risk = {
                "t": round(current_t, 3),
                "osmolality": round(float(observation["osmolality"]), 5),
            }
        for cause in CRITICAL_LIMITS:
            if cause not in first_crossings and threshold_excess(cause, observation) > 0:
                first_crossings[cause] = {
                    "t": round(current_t, 3),
                    "excess": round(float(threshold_excess(cause, observation)), 5),
                    "state": compact_observation(observation),
                }
        if (
            "cumulative hyperosmolar injury" not in first_crossings
            and observation["osmotic_injury"] > OSM_INJURY_DEATH
        ):
            first_crossings["cumulative hyperosmolar injury"] = {
                "t": round(current_t, 3),
                "excess": round(float(observation["osmotic_injury"] - OSM_INJURY_DEATH), 5),
                "state": compact_observation(observation),
            }
        curve.append({"t": round(current_t, 3), "state": compact_observation(observation)})
        if dead:
            break

    death_time = None if body.alive else float(body.t)
    future_labs = observed_labs_after(trajectory, death_time)
    death_cause = None if body.alive else body.death_cause
    subsystem = classify_death_cause(death_cause)
    last_observed_lab_time = max(
        [float(lab.get("t", 0.0)) for lab in trajectory.get("labs", [])] + [0.0]
    )
    return {
        "stay_id": trajectory.get("stay_id"),
        "simulated_alive": bool(body.alive),
        "simulated_death_cause": death_cause,
        "simulated_death_time_hours": None if death_time is None else round(death_time, 3),
        "hard_mechanism_subsystem": subsystem,
        "hard_mechanism_terms": HARD_MECHANISM_OWNERS.get(subsystem, []),
        "falsified_by_observed_later_measurements": bool(future_labs),
        "observed_hours_after_sim_death": (
            None if death_time is None else round(max(0.0, last_observed_lab_time - death_time), 3)
        ),
        "first_threshold_crossings": first_crossings,
        "first_osmolar_risk": first_osmolar_risk,
        "death_snapshot": compact_observation(body.observe()),
        "predeath_curve_tail": curve[-6:],
        "observed_labs_after_death_count": len(future_labs),
        "observed_labs_after_death_sample": future_labs[:8],
        "action_totals_to_death_or_end": action_totals(
            trajectory.get("actions", []), stop_time=death_time
        ),
        "action_coverage": trajectory.get("_cover", {}),
        "repair_hypotheses": REPAIR_BACKLOG.get(subsystem, []),
    }


def summarize(cases):
    deaths = [case for case in cases if not case["simulated_alive"]]
    cause_counts = Counter(case["simulated_death_cause"] for case in deaths)
    subsystem_counts = Counter(case["hard_mechanism_subsystem"] for case in deaths)
    first_crossing_counts = Counter()
    for case in deaths:
        if not case["first_threshold_crossings"]:
            continue
        first = min(
            case["first_threshold_crossings"].items(),
            key=lambda item: item[1]["t"],
        )[0]
        first_crossing_counts[first] += 1
    backlog = {}
    for subsystem, count in sorted(subsystem_counts.items()):
        backlog[subsystem] = {
            "death_count": int(count),
            "hard_mechanism_terms": HARD_MECHANISM_OWNERS.get(subsystem, []),
            "repair_hypotheses": REPAIR_BACKLOG.get(subsystem, []),
        }
    return {
        "trajectory_count": len(cases),
        "simulated_death_count": len(deaths),
        "falsified_death_count": sum(
            bool(case["falsified_by_observed_later_measurements"]) for case in deaths
        ),
        "death_causes": dict(cause_counts),
        "hard_mechanism_subsystems": dict(subsystem_counts),
        "first_threshold_crossing_counts": dict(first_crossing_counts),
        "repair_backlog": backlog,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="?", default="dka_fidelity_demo_v4.jsonl")
    parser.add_argument("--output", default="dka_viability_falsification_audit_v1.json")
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print the full artifact instead of the compact summary.",
    )
    args = parser.parse_args()

    trajectories = [
        json.loads(line) for line in Path(args.trajectories).read_text().splitlines()
        if line.strip()
    ]
    cases = [audit_trajectory(trajectory) for trajectory in trajectories]
    result = {
        "audit": "DKABody viability falsification from real replay trajectories",
        "trajectory_source": Path(args.trajectories).name,
        "osmotic_injury_terminal": OSMOTIC_INJURY_TERMINAL,
        "non_terminal_reported_burden_causes": sorted(NON_TERMINAL_CRITICAL_CAUSES),
        "greybox_can_edit_these": ["G", "Ket", "HCO3", "Ke", "Na", "Cr"],
        "terminal_death_drivers_are_hard_mechanisms": [
            "volume_map", "potassium_mass",
        ] + (["osmotic_injury"] if OSMOTIC_INJURY_TERMINAL else []),
        "reported_burden_only": [] if OSMOTIC_INJURY_TERMINAL else ["osmotic_injury"],
        "summary": summarize(cases),
        "cases": cases,
        "conclusion": (
            "Real trajectories that survive beyond simulated death are direct "
            "falsifications of the hard viability mechanism. Repair should target "
            "the owning equations, not widen the grey-box residual write set."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.print_full:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({
            "output": args.output,
            "summary": result["summary"],
            "conclusion": result["conclusion"],
        }, indent=2))


if __name__ == "__main__":
    main()
