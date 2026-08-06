"""Aggregate symbolic JEPA transitions into observational candidate rules."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from osler_jepa.symbolic import DIRECTION_NAMES
from osler_jepa.validator import OSLER_DKA_VALIDATOR


def _active_conflict(action_index, state_index, direction_index):
    sign = -1 if direction_index == 0 else 1
    for rule in OSLER_DKA_VALIDATOR.rules:
        if rule.action_index == action_index and rule.state_index == state_index:
            if rule.expected_sign != sign:
                return rule.rule_id
    return None


def induce_candidates(records, action_keys, state_keys, canonical_states,
                      min_support=3, min_confidence=0.55,
                      population="MIMIC-IV ICU DKA-like anchors",
                      provenance_source=(
                          "JEPA symbolic RuleProposalHead + MIMIC-IV demo"
                      )):
    grouped = defaultdict(list)
    pair_totals = defaultdict(int)
    for record in records:
        for action_index, exposure in enumerate(record["action_exposure"]):
            if exposure <= 0.05:
                continue
            for state_index, actual_direction in enumerate(
                record["actual_directions"]
            ):
                if actual_direction < 0:
                    continue
                probabilities = record["proposal_probabilities"][state_index]
                predicted_direction = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_direction])
                if predicted_direction == 1 or confidence < min_confidence:
                    continue
                pair_totals[(action_index, state_index)] += 1
                grouped[(action_index, state_index, predicted_direction)].append({
                    "stay_id": record["stay_id"],
                    "confidence": confidence,
                    "agrees": int(predicted_direction == actual_direction),
                    "cointerventions": [
                        action_keys[index] for index, value in enumerate(
                            record["action_exposure"]
                        ) if index != action_index and value > 0.05
                    ],
                })

    candidates = []
    for (action_index, state_index, direction_index), evidence in grouped.items():
        stays = {item["stay_id"] for item in evidence}
        discovery_accuracy = float(np.mean([
            item["agrees"] for item in evidence
        ]))
        proposal_consistency = len(evidence) / max(
            pair_totals[(action_index, state_index)], 1
        )
        if (
            len(evidence) < min_support
            or len(stays) < 2
            or discovery_accuracy < 0.60
            or proposal_consistency < 0.65
        ):
            continue
        action = action_keys[action_index]
        state = state_keys[state_index]
        direction = DIRECTION_NAMES[direction_index]
        digest = hashlib.sha1(
            f"{action}:{state}:{direction}".encode("ascii")
        ).hexdigest()[:10]
        conflict = _active_conflict(action_index, state_index, direction_index)
        cointervention_counts = defaultdict(int)
        for item in evidence:
            for name in item["cointerventions"]:
                cointervention_counts[name] += 1
        isolated_support = sum(
            not item["cointerventions"] for item in evidence
        )
        candidates.append({
            "rule_id": f"jepa_observational_{digest}",
            "level": "observational_association",
            "context": {
                "disease": "DKA",
                "population": population,
                "cointerventions_observed": dict(sorted(cointervention_counts.items())),
            },
            "action": {"type": "intervention", "name": action},
            "predicted_transition": {
                "variable": canonical_states[state_index],
                "model_variable": state,
                "direction": direction,
                "time_window": "0-6h",
            },
            "confidence": round(float(np.mean([
                item["confidence"] for item in evidence
            ])), 4),
            "evidence": {
                "support_count": len(evidence),
                "patient_stay_count": len(stays),
                "discovery_factual_direction_accuracy": round(discovery_accuracy, 4),
                "proposal_direction_consistency": round(proposal_consistency, 4),
                "isolated_action_support_count": isolated_support,
            },
            "provenance": {
                "source": provenance_source,
                "causal_claim": False,
                "confounding_warning": True,
            },
            "status": "candidate",
            "active_rule_conflict": conflict,
        })
    return sorted(candidates, key=lambda rule: rule["rule_id"])


def validate_candidates(candidates, records, action_keys, state_keys,
                        min_support=3, min_accuracy=0.67,
                        min_patient_stays=3):
    action_index = {name: index for index, name in enumerate(action_keys)}
    state_index = {name: index for index, name in enumerate(state_keys)}
    direction_index = {name: index for index, name in enumerate(DIRECTION_NAMES)}
    for rule in candidates:
        ai = action_index[rule["action"]["name"]]
        si = state_index[rule["predicted_transition"]["model_variable"]]
        di = direction_index[rule["predicted_transition"]["direction"]]
        selected = [
            record for record in records
            if record["action_exposure"][ai] > 0.05
            and record["actual_directions"][si] >= 0
        ]
        actual = np.asarray([
            record["actual_directions"][si] for record in selected
        ], dtype=np.int64)
        accuracy = float(np.mean(actual == di)) if len(actual) else None
        persistence = float(np.mean(actual == 1)) if len(actual) else None
        isolated = [
            record for record in selected
            if int((record["action_exposure"] > 0.05).sum()) == 1
        ]
        passes = bool(
            len(actual) >= min_support
            and len({record["stay_id"] for record in selected}) >= min_patient_stays
            and accuracy is not None and accuracy >= min_accuracy
            and persistence is not None and accuracy > persistence
            and rule["evidence"]["discovery_factual_direction_accuracy"] >= 0.67
            and rule["evidence"]["proposal_direction_consistency"] >= 0.70
            and rule["evidence"]["patient_stay_count"] >= 3
            and rule["evidence"]["isolated_action_support_count"] >= 2
            and len(isolated) >= 2
            and rule["active_rule_conflict"] is None
        )
        rule["heldout_test"] = {
            "support_count": int(len(actual)),
            "patient_stay_count": len({r["stay_id"] for r in selected}),
            "direction_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "persistence_direction_accuracy": round(persistence, 4)
            if persistence is not None else None,
            "isolated_action_support_count": len(isolated),
            "action_specificity_identifiable": bool(
                rule["evidence"]["isolated_action_support_count"] >= 2
                and len(isolated) >= 2
            ),
            "passes_automated_retrospective_gate": passes,
        }
        if passes:
            rule["level"] = "retrospective_validated"
            rule["status"] = "awaiting_human_review"
        elif rule["active_rule_conflict"]:
            rule["status"] = "rejected_by_active_rule_conflict"
        else:
            rule["status"] = "insufficient_or_failed_heldout_evidence"
    return candidates
