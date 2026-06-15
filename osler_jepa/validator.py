"""Symbolic transition validation and differentiable consistency penalties."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

from dka_action_contract import ACTION_INDEX, INSULIN_KEYS
from dka_world_model_contract import STATE_KEYS
from osler_jepa.embodied_logic import parse_program


@dataclass(frozen=True)
class TransitionRule:
    rule_id: str
    action_index: int
    action_name: str
    state_index: int
    state_name: str
    expected_sign: int
    min_action: float = 0.05
    min_effect: float = 0.0
    blocker_action_index: int | None = None
    blocker_action_indices: tuple[int, ...] = ()
    blocker_max: float = 0.05
    rationale: str = ""
    provenance: str = "DKABody mechanistic equation"


class OslerTransitionValidator:
    def __init__(self, rules):
        self.rules = tuple(rules)

    def consistency_loss(self, predicted_effect, action_exposure, valid_mask=None):
        """Hinge penalty for effects that violate symbolic action directions.

        predicted_effect is normalized state difference versus no treatment.
        action_exposure is normalized cumulative/mean action difference.
        """
        losses = []
        for rule in self.rules:
            active = action_exposure[..., rule.action_index] > rule.min_action
            if rule.blocker_action_index is not None:
                active = active & (
                    action_exposure[..., rule.blocker_action_index] <= rule.blocker_max
                )
            for blocker_index in rule.blocker_action_indices:
                active = active & (
                    action_exposure[..., blocker_index] <= rule.blocker_max
                )
            if valid_mask is not None:
                active = active & valid_mask.bool()
            if not active.any():
                continue
            effect = predicted_effect[..., rule.state_index]
            signed = effect * float(rule.expected_sign)
            margin = predicted_effect.new_tensor(rule.min_effect)
            losses.append(torch.relu(margin - signed)[active].mean())
        return torch.stack(losses).mean() if losses else predicted_effect.new_tensor(0.0)

    def validate(self, action, future, baseline_future) -> dict:
        effect = {
            name: float(future[name]) - float(baseline_future[name])
            for name in future.keys() & baseline_future.keys()
        }
        checks = []
        for rule in self.rules:
            action_value = float(action[rule.action_index])
            if action_value <= rule.min_action:
                continue
            if rule.blocker_action_index is not None:
                if float(action[rule.blocker_action_index]) > rule.blocker_max:
                    continue
            if any(
                float(action[index]) > rule.blocker_max
                for index in rule.blocker_action_indices
            ):
                continue
            if rule.state_name not in effect:
                continue
            value = effect[rule.state_name]
            verified = value * rule.expected_sign >= rule.min_effect
            checks.append({
                "rule_id": rule.rule_id,
                "action": rule.action_name,
                "state": rule.state_name,
                "expected": "increase" if rule.expected_sign > 0 else "decrease",
                "observed_effect": round(value, 6),
                "status": "verified" if verified else "contradicted",
                "rationale": rule.rationale,
                "provenance": rule.provenance,
            })
        if not checks:
            status = "unexplained"
        elif any(check["status"] == "contradicted" for check in checks):
            status = "contradicted"
        else:
            status = "verified"
        return {"status": status, "checks": checks, "effect": effect}


CONSTRAINT_SCALE = 1_000_000.0
DEFAULT_RULE_PATH = (
    Path(__file__).resolve().parents[1] / "rules" / "active" / "dka_embodied.pl"
)


def compile_transition_rules(rule_path=DEFAULT_RULE_PATH):
    """Compile differentiable effect constraints from the active Prolog file."""
    path = Path(rule_path)
    clauses = parse_program(path.read_text(encoding="utf-8"))
    expected = {
        clause.head.arguments[3]: clause
        for clause in clauses
        if clause.head.predicate == "expected" and len(clause.head.arguments) == 4
    }
    declarations = {
        clause.head.arguments[0]: clause.head.arguments[1:]
        for clause in clauses
        if clause.head.predicate == "training_constraint"
        and len(clause.head.arguments) == 3
    }
    state_lookup = {name.lower(): (index, name) for index, name in enumerate(STATE_KEYS)}
    rules = []
    for rule_id, thresholds in declarations.items():
        if rule_id not in expected:
            raise ValueError(f"Training constraint has no expected/4 rule: {rule_id}")
        clause = expected[rule_id]
        action_name, state_atom, direction, _ = clause.head.arguments
        if action_name not in ACTION_INDEX:
            raise ValueError(f"Unknown action in active Prolog rule: {action_name}")
        if state_atom not in state_lookup:
            raise ValueError(f"Unknown state in active Prolog rule: {state_atom}")
        blockers = []
        for literal in clause.body:
            if not literal.negated:
                continue
            if literal.atom.predicate == "requested" and literal.atom.arguments:
                blockers.append(ACTION_INDEX[literal.atom.arguments[0]])
            elif literal.atom.predicate == "insulin_requested":
                blockers.extend(ACTION_INDEX[name] for name in INSULIN_KEYS)
        state_index, state_name = state_lookup[state_atom]
        rules.append(TransitionRule(
            rule_id=rule_id,
            action_index=ACTION_INDEX[action_name],
            action_name=action_name,
            state_index=state_index,
            state_name=state_name,
            expected_sign=1 if direction == "increase" else -1,
            min_action=float(thresholds[0]) / CONSTRAINT_SCALE,
            min_effect=float(thresholds[1]) / CONSTRAINT_SCALE,
            blocker_action_indices=tuple(sorted(set(blockers))),
            rationale=f"Compiled from active Prolog expected/4 rule {rule_id}.",
            provenance=str(path),
        ))
    return tuple(rules)


OSLER_DKA_VALIDATOR = OslerTransitionValidator(compile_transition_rules())
