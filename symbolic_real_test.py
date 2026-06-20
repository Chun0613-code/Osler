"""Patient-held-out real-data test for symbolic-grounded DKA JEPA proposals.

Real observational cohorts can validate reproducibility, not causality.
Candidate rules therefore stay in the sandbox even when they pass the
retrospective gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dka_body import henderson
from dka_world_model import (
    ACTION_KEYS, A_DIM, DT, HISTORY_HOURS, S_MEAN, S_STD, STATE_KEYS,
    a2vec, load_checkpoint, s2vec, treatment_history_features,
)
from osler_jepa.ontology import OSLER_STATE_ONTOLOGY
from osler_jepa.rule_inducer import induce_candidates, validate_candidates
from osler_jepa.rule_sandbox import RuleSandbox
from osler_jepa.symbolic import PHYSICAL_STABLE_THRESHOLDS
from train_intervention_jepa import choose_device, evaluate_mimic_proxy


TARGET_COLUMNS = {
    0: ("glucose_t", "glucose_tp6"),
    1: ("ph_t", "ph_tp6"),
    2: ("bicarbonate_t", "bicarbonate_tp6"),
    3: ("anion_gap_t", "anion_gap_tp6"),
    4: ("potassium_t", "potassium_tp6"),
    5: ("map_t", "map_tp6"),
    8: ("sodium_t", "sodium_tp6"),
    9: ("osmolality_t", "osmolality_tp6"),
    10: ("creatinine_t", "creatinine_tp6"),
    11: ("urine_output_t", "urine_output_tp6"),
    12: ("BHB_t", "BHB_tp6"),
}


def _cohort_info(path):
    name = Path(path).name.lower()
    if "eicu" in name:
        label = "eICU demo"
        population = "eICU demo ICU DKA-like anchors"
    elif "mimiciii" in name or "mimic-iii" in name:
        label = "MIMIC-III demo"
        population = "MIMIC-III demo ICU lab-defined DKA-like anchors"
    elif "mimic" in name or "dka_transitions_6h_demo" in name:
        label = "MIMIC-IV demo"
        population = "MIMIC-IV ICU DKA-like anchors"
    else:
        label = "real-data cohort"
        population = "real-data ICU DKA-like anchors"
    return {
        "label": label,
        "population": population,
        "test_type": f"patient-held-out {label} symbolic rule test",
        "provenance_source": f"JEPA symbolic RuleProposalHead + {label}",
    }


def _value(row, name, default=np.nan):
    value = row.get(name, default)
    return default if pd.isna(value) else float(value)


def _state_from_row(row):
    glucose = _value(row, "glucose_t")
    potassium = _value(row, "potassium_t")
    map_value = _value(row, "map_t")
    bicarbonate = _value(row, "bicarbonate_t")
    if any(np.isnan(value) for value in (
        glucose, potassium, map_value, bicarbonate
    )):
        return None
    ph = _value(row, "ph_t", henderson(bicarbonate))
    anion_gap = _value(row, "anion_gap_t", float(S_MEAN[3]))
    sodium = _value(row, "sodium_t", 138.0)
    creatinine = _value(row, "creatinine_t", 1.2)
    osmolality = _value(row, "osmolality_t")
    if np.isnan(osmolality):
        osmolality = _value(row, "osmolality_derived_t")
    if np.isnan(osmolality):
        osmolality = 2.0 * sodium + glucose / 18.0
    urine_output = _value(row, "urine_output_t", 100.0)
    bhb = _value(row, "BHB_t", max(0.0, anion_gap - 12.0) * 0.75)
    return {
        "G": glucose,
        "pH": ph,
        "HCO3": bicarbonate,
        "anion_gap": anion_gap,
        "Ke": potassium,
        "MAP": map_value,
        "V": float(np.clip((map_value - 30.0) / 60.0 * 15.0, 5.0, 18.0)),
        "I": 1.0,
        "Na": sodium,
        "osmolality": osmolality,
        "creatinine": creatinine,
        "urine_output": urine_output,
        "BHB": bhb,
        "K_store": float(np.clip(
            120.0 - 45.0 * max(0.0, 7.35 - ph)
            - 8.0 * max(0.0, creatinine - 1.2), 55.0, 145.0
        )),
        "osmotic_injury": 0.0,
    }


def _action_data(row):
    if not isinstance(row.get("future_action_grid"), str):
        return None, None
    physical = np.asarray(json.loads(row["future_action_grid"]), dtype=np.float32)
    if physical.ndim != 2 or physical.shape[1] != A_DIM:
        return None, None
    normalized = np.asarray([a2vec(action) for action in physical], dtype=np.float32)
    history = None
    if isinstance(row.get("history_action_grid"), str):
        history_grid = np.asarray(
            json.loads(row["history_action_grid"]), dtype=np.float32
        )
        history = treatment_history_features(history_grid, DT, HISTORY_HOURS)
    return normalized, history


def _actual_directions(row, state):
    result = np.full(len(STATE_KEYS), -1, dtype=np.int64)
    for index, (current_column, target_column) in TARGET_COLUMNS.items():
        current = _value(row, current_column)
        target = _value(row, target_column)
        if index == 9:
            if np.isnan(current):
                current = _value(row, "osmolality_derived_t")
            if np.isnan(target):
                target = _value(row, "osmolality_derived_tp6")
        if index == 1 and np.isnan(target):
            target_bicarbonate = _value(row, "bicarbonate_tp6")
            if not np.isnan(target_bicarbonate):
                target = henderson(target_bicarbonate)
        if np.isnan(current):
            current = state[STATE_KEYS[index]]
        if np.isnan(target):
            continue
        delta = target - current
        threshold = PHYSICAL_STABLE_THRESHOLDS[index]
        result[index] = 0 if delta < -threshold else 2 if delta > threshold else 1
    return result


@torch.no_grad()
def build_records(model, frame, device):
    records = []
    for _, row in frame.iterrows():
        active = row.get("dka_active_t", True)
        if not pd.isna(active) and not bool(active):
            continue
        state = _state_from_row(row)
        sequence, history = _action_data(row)
        if state is None or sequence is None:
            continue
        exposure = sequence.mean(axis=0)
        if not np.any(exposure > 0.05):
            continue
        state_tensor = torch.as_tensor(
            s2vec(state), dtype=torch.float32, device=device
        ).unsqueeze(0)
        history_tensor = None if history is None else torch.as_tensor(
            history, dtype=torch.float32, device=device
        ).unsqueeze(0)
        latent = model.encode_state(state_tensor, history_tensor)
        outputs = model.symbolic_outputs(
            latent,
            torch.as_tensor(exposure, dtype=torch.float32, device=device).unsqueeze(0),
            delta_hours=sequence.shape[0] * DT,
        )
        records.append({
            "subject_id": int(row["subject_id"]),
            "stay_id": int(row["stay_id"]),
            "action_exposure": exposure,
            "actual_directions": _actual_directions(row, state),
            "proposal_probabilities": outputs["proposal_logits"].softmax(
                dim=-1
            )[0].cpu().numpy(),
            "status_probabilities": outputs["status_logits"].softmax(
                dim=-1
            )[0].cpu().numpy(),
            "proof_probabilities": outputs["proof_logits"].sigmoid()[0].cpu().numpy(),
        })
    return records


def external_symbolic_metrics(records):
    all_correct = 0
    all_total = 0
    changed_correct = 0
    changed_total = 0
    confidence = []
    for record in records:
        prediction = record["proposal_probabilities"].argmax(axis=-1)
        truth = record["actual_directions"]
        observed = truth >= 0
        changed = observed & (truth != 1)
        all_correct += int((prediction[observed] == truth[observed]).sum())
        all_total += int(observed.sum())
        changed_correct += int((prediction[changed] == truth[changed]).sum())
        changed_total += int(changed.sum())
        confidence.extend(record["proposal_probabilities"].max(axis=-1)[observed])
    return {
        "factual_direction_accuracy_all": round(all_correct / all_total, 4)
        if all_total else None,
        "factual_direction_accuracy_changed_only": round(
            changed_correct / changed_total, 4
        ) if changed_total else None,
        "direction_comparisons": all_total,
        "changed_direction_comparisons": changed_total,
        "mean_proposal_confidence": round(float(np.mean(confidence)), 4)
        if confidence else None,
        "interpretation": (
            "Observational factual-direction check; it is not a causal effect test."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h_demo_v4.parquet")
    parser.add_argument("--output", default="dka_symbolic_real_test_v5.json")
    parser.add_argument("--rules-root", default="rules")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    compatibility = payload.get("compatibility", {})
    symbolic_trained = not any(
        key.startswith(("DirectionHead", "StatusHead", "ProofPathHead", "RuleProposalHead"))
        for key in compatibility.get("missing_keys", [])
    )
    frame = pd.read_parquet(args.mimic)
    cohort_info = _cohort_info(args.mimic)
    stays = np.asarray(sorted(frame["stay_id"].dropna().unique()), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(stays)
    split = max(1, int(round(0.67 * len(stays))))
    split = min(split, max(1, len(stays) - 1))
    discovery_stays = set(stays[:split].tolist())
    heldout_stays = set(stays[split:].tolist())
    discovery = frame[frame["stay_id"].isin(discovery_stays)]
    heldout = frame[frame["stay_id"].isin(heldout_stays)]
    discovery_records = build_records(model, discovery, device)
    heldout_records = build_records(model, heldout, device)

    canonical = [OSLER_STATE_ONTOLOGY.require(name) for name in STATE_KEYS]
    candidates = induce_candidates(
        discovery_records, ACTION_KEYS, STATE_KEYS, canonical,
        population=cohort_info["population"],
        provenance_source=cohort_info["provenance_source"],
    )
    candidates = validate_candidates(
        candidates, heldout_records, ACTION_KEYS, STATE_KEYS
    )

    project_root = Path(__file__).resolve().parent
    validator_path = project_root / "osler_jepa" / "validator.py"
    validator_before = hashlib.sha256(validator_path.read_bytes()).hexdigest()
    sandbox = RuleSandbox(project_root / args.rules_root)
    active_before = sandbox.snapshot_active()
    run_id = f"symbolic_real_test_{int(time.time())}"
    candidate_path = sandbox.write_candidates(run_id, {
        "run_id": run_id,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_cohort": str(Path(args.mimic).resolve()),
        "discovery_stays": sorted(discovery_stays),
        "heldout_stays": sorted(heldout_stays),
        "rules": candidates,
    })
    sandbox.assert_active_immutable(active_before)
    validator_after = hashlib.sha256(validator_path.read_bytes()).hexdigest()

    validated = sum(
        rule["heldout_test"]["passes_automated_retrospective_gate"]
        for rule in candidates
    )
    conflicts = sum(bool(rule["active_rule_conflict"]) for rule in candidates)
    external = external_symbolic_metrics(heldout_records)
    issues = []
    if not symbolic_trained:
        issues.append("Checkpoint symbolic heads are untrained legacy initialization.")
    if len(heldout_stays) < 5:
        issues.append("Held-out cohort has fewer than five patient stays.")
    if not candidates:
        issues.append("No proposal met discovery support/confidence thresholds.")
    if candidates and not validated:
        issues.append("No candidate passed the small held-out retrospective gate.")
    if external["factual_direction_accuracy_changed_only"] is not None and \
            external["factual_direction_accuracy_changed_only"] < 0.5:
        issues.append("Changed-only external direction accuracy is below chance.")

    report = {
        "test_type": cohort_info["test_type"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "symbolic_heads_trained": symbolic_trained,
        "cohort": str(Path(args.mimic).resolve()),
        "cohort_label": cohort_info["label"],
        "patient_split": {
            "discovery_stays": len(discovery_stays),
            "heldout_stays": len(heldout_stays),
            "overlap": len(discovery_stays & heldout_stays),
            "discovery_records": len(discovery_records),
            "heldout_records": len(heldout_records),
        },
        "external_symbolic_metrics": external,
        "factual_numerical_benchmark": evaluate_mimic_proxy(
            model, args.mimic, torch.device(device)
        ),
        "candidate_rule_gate": {
            "proposed": len(candidates),
            "retrospectively_validated": validated,
            "active_rule_conflicts": conflicts,
            "automatically_promoted_to_active": 0,
            "candidate_file": str(candidate_path),
        },
        "safety_boundary": {
            "active_sandbox_unchanged": True,
            "validator_source_unchanged": validator_before == validator_after,
            "causal_claims_allowed": False,
            "human_review_required": True,
        },
        "issues": issues,
        "rules": candidates,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
