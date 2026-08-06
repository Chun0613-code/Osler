"""Evaluate a persistence-anchored residual gate on patient-held-out rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dka_world_model import load_checkpoint
from osler_jepa.anchored_residual import AnchoredResidualGate
from real_world_improvement import build_examples, run_crossfit
from real_world_power_analysis import stay_level_method_deltas, summarize_delta
from train_intervention_jepa import choose_device


DENSE_CORE = ("G", "Na", "creatinine")


def _has_action_exposure(path):
    if not Path(path).exists():
        return False
    try:
        columns = pd.read_csv(path, nrows=1).columns
    except Exception:
        return False
    return "action_exposure" in columns


def load_or_build_predictions(args):
    if _has_action_exposure(args.predictions):
        return pd.read_csv(args.predictions), "loaded_existing"
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    frame = pd.read_parquet(args.mimic)
    examples = build_examples(model, frame, device)
    result = run_crossfit(examples, args.seed)
    rows = pd.DataFrame(result["numerical_rows"])
    rows.to_csv(args.predictions, index=False)
    return rows, "rebuilt_patient_oof"


def _parse_action_exposure(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def apply_gate(frame, candidate_source, gate, horizon_hours=6.0):
    output = frame.copy()
    column = f"anchored_{candidate_source}"
    details = []
    for index, row in output.iterrows():
        agreement = None
        if candidate_source == "ensemble_adapter":
            agreement = float(row["ensemble_direction_agreement"])
        decision = gate.select(
            int(row["target_index"]),
            float(row["current"]),
            float(row[candidate_source]),
            _parse_action_exposure(row["action_exposure"]),
            direction_agreement=agreement,
            horizon_hours=horizon_hours,
        )
        output.at[index, column] = decision["selected_prediction"]
        details.append(decision)
    return output, details, column


def method_mae(frame, method, active_only=False, states=None):
    selected = frame.copy()
    if active_only:
        selected = selected[selected["active_dka"].astype(bool)]
    if states is not None:
        selected = selected[selected["state"].isin(states)]
    errors = [
        abs(float(row[method]) - float(row["truth"])) / float(row["scale"])
        for _, row in selected.iterrows()
        if np.isfinite(row.get(method, np.nan)) and np.isfinite(row["truth"])
    ]
    return {
        "n": len(errors),
        "normalized_mae": round(float(np.mean(errors)), 6) if errors else None,
    }


def reason_summary(details):
    reason_counts = Counter()
    allowed_by_state = Counter()
    total_by_state = Counter()
    prolog_status = Counter()
    for item in details:
        state = item["state"]
        total_by_state[state] += 1
        if item["allowed_to_leave_persistence"]:
            allowed_by_state[state] += 1
        prolog_status[item["prolog_gate"]["status"]] += 1
        if item["abstain_reasons"]:
            reason_counts.update(item["abstain_reasons"])
        else:
            reason_counts["allowed"] += 1
    return {
        "total_predictions": int(sum(total_by_state.values())),
        "allowed_to_leave_persistence": int(sum(allowed_by_state.values())),
        "allowed_fraction": round(
            sum(allowed_by_state.values()) / max(sum(total_by_state.values()), 1), 6
        ),
        "abstain_reasons": dict(reason_counts),
        "prolog_status": dict(prolog_status),
        "allowed_by_state": {
            state: {
                "allowed": int(allowed_by_state[state]),
                "total": int(total_by_state[state]),
            }
            for state in sorted(total_by_state)
        },
    }


def summarize_candidate(frame, details, candidate_source, anchored_column):
    report = {
        "candidate_source": candidate_source,
        "anchored_column": anchored_column,
        "movement": reason_summary(details),
        "normalized_mae": {
            "all_windows": {
                "persistence": method_mae(frame, "persistence"),
                candidate_source: method_mae(frame, candidate_source),
                anchored_column: method_mae(frame, anchored_column),
            },
            "active_dka_only": {
                "persistence": method_mae(frame, "persistence", active_only=True),
                candidate_source: method_mae(frame, candidate_source, active_only=True),
                anchored_column: method_mae(frame, anchored_column, active_only=True),
            },
            "dense_core_all_windows": {
                "persistence": method_mae(frame, "persistence", states=DENSE_CORE),
                candidate_source: method_mae(frame, candidate_source, states=DENSE_CORE),
                anchored_column: method_mae(frame, anchored_column, states=DENSE_CORE),
            },
        },
        "stay_level_delta_vs_persistence": {},
        "per_state": {},
    }
    for active_only in (False, True):
        label = "active_dka_only" if active_only else "all_windows"
        deltas, stays = stay_level_method_deltas(
            frame, anchored_column, active_only=active_only
        )
        report["stay_level_delta_vs_persistence"][label] = summarize_delta(deltas, stays)
    for state in sorted(frame["state"].unique()):
        state_report = {}
        for method in ("persistence", candidate_source, anchored_column):
            state_report[method] = method_mae(frame, method, states=[state])
        deltas, stays = stay_level_method_deltas(
            frame, anchored_column, states=[state]
        )
        state_report["anchored_delta_vs_persistence"] = summarize_delta(deltas, stays)
        report["per_state"][state] = state_report
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h_demo_v4.parquet")
    parser.add_argument("--predictions", default="dka_real_world_oof_predictions_v3.csv")
    parser.add_argument("--output", default="dka_anchored_residual_hybrid_v1.json")
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows, prediction_source = load_or_build_predictions(args)
    gate = AnchoredResidualGate()
    candidate_reports = {}
    augmented = rows.copy()
    for candidate_source in ("base_jepa", "ensemble_adapter"):
        augmented, details, column = apply_gate(
            augmented, candidate_source, gate, args.horizon_hours
        )
        candidate_reports[candidate_source] = summarize_candidate(
            augmented, details, candidate_source, column
        )
    result = {
        "experiment": "persistence-anchored residual shrinkage with Prolog gate",
        "prediction_rows": Path(args.predictions).name,
        "prediction_source": prediction_source,
        "cohort": Path(args.mimic).name,
        "stay_count": int(augmented["stay_id"].nunique()),
        "horizon_hours": float(args.horizon_hours),
        "contract": {
            "anchor": "persistence/current observed value",
            "residual_source": "candidate forecast minus persistence",
            "numeric_rule": "selected = persistence + shrink * clipped_residual",
            "prolog_role": "allow/deny and expected direction only",
            "default": "persistence",
            "clinical_or_causal_claim_allowed": False,
        },
        "gate_config": gate.config.__dict__,
        "candidate_reports": candidate_reports,
        "conclusion": (
            "Anchoring converts unsafe model outputs into abstention or small, "
            "rule-consistent residuals. It reduces self-confident departures "
            "from persistence, but remains a research guard and does not create "
            "a causal or clinical promotion claim."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
