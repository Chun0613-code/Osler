"""Grounded Prolog-style reasoning for Osler-controlled JEPA predictions.

The neural world model predicts continuous future states. This module owns the
discrete control contract around those predictions: action preconditions,
required co-interventions, expected effect directions, and proof traces.

The rule language intentionally implements a small, auditable ground-Prolog
subset. Active clinical rules stay human-owned in ``rules/active``; JEPA cannot
write or promote them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from dka_action_contract import ACTION_KEYS, expand_action
from osler_jepa.ontology import OSLER_STATE_ONTOLOGY
from osler_jepa.state_compiler import ground_dka_facts


_ATOM_RE = re.compile(r"^([a-z][a-z0-9_]*)\((.*)\)$")


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.arguments:
            return self.predicate
        return f"{self.predicate}({', '.join(self.arguments)})"


@dataclass(frozen=True)
class Literal:
    atom: Atom
    negated: bool = False


@dataclass(frozen=True)
class Clause:
    head: Atom
    body: tuple[Literal, ...]
    source: str


def _split_terms(text: str) -> list[str]:
    terms, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            terms.append(text[start:index].strip())
            start = index + 1
    terms.append(text[start:].strip())
    return [term for term in terms if term]


def parse_atom(text: str) -> Atom:
    text = text.strip()
    match = _ATOM_RE.match(text)
    if not match:
        if not re.match(r"^[a-z][a-z0-9_]*$", text):
            raise ValueError(f"Unsupported Prolog atom: {text!r}")
        return Atom(text)
    predicate, raw_arguments = match.groups()
    arguments = tuple(part.strip() for part in _split_terms(raw_arguments))
    if any(not re.match(r"^[a-z0-9_]+$", argument) for argument in arguments):
        raise ValueError(f"Only grounded snake_case terms are allowed: {text!r}")
    return Atom(predicate, arguments)


def parse_program(text: str) -> tuple[Clause, ...]:
    cleaned = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
    clauses = []
    for statement in cleaned.split("."):
        statement = " ".join(statement.split())
        if not statement:
            continue
        if ":-" in statement:
            head_text, body_text = statement.split(":-", 1)
            body = []
            for term in _split_terms(body_text):
                negated = term.startswith("not(") and term.endswith(")")
                atom_text = term[4:-1] if negated else term
                body.append(Literal(parse_atom(atom_text), negated))
        else:
            head_text, body = statement, []
        clauses.append(Clause(parse_atom(head_text), tuple(body), statement + "."))
    return tuple(clauses)


class GroundProlog:
    """Deterministic forward-chaining engine with stratified ground negation."""

    def __init__(self, clauses: Iterable[Clause]):
        self.clauses = tuple(clauses)

    @classmethod
    def from_file(cls, path: str | Path) -> "GroundProlog":
        return cls(parse_program(Path(path).read_text(encoding="utf-8")))

    def infer(self, initial_facts: Iterable[Atom]):
        facts = set(initial_facts)
        proofs: dict[Atom, Clause | None] = {fact: None for fact in facts}
        changed = True
        while changed:
            changed = False
            for clause in self.clauses:
                if clause.head in facts:
                    continue
                if all(
                    literal.atom not in facts if literal.negated
                    else literal.atom in facts
                    for literal in clause.body
                ):
                    facts.add(clause.head)
                    proofs[clause.head] = clause
                    changed = True
        return facts, proofs

    @staticmethod
    def proof(atom: Atom, proofs: dict[Atom, Clause | None], seen=None) -> dict:
        seen = set() if seen is None else set(seen)
        if atom in seen:
            return {"conclusion": str(atom), "cycle": True}
        seen.add(atom)
        clause = proofs.get(atom)
        if clause is None:
            return {"conclusion": str(atom), "source": "patient_or_action_fact"}
        premises = []
        for literal in clause.body:
            if literal.negated:
                premises.append({
                    "conclusion": f"not({literal.atom})",
                    "source": "closed_world_check",
                })
            else:
                premises.append(GroundProlog.proof(literal.atom, proofs, seen))
        return {
            "conclusion": str(atom),
            "rule": clause.source,
            "premises": premises,
        }


class DkaEmbodiedLogic:
    def __init__(self, rule_path: str | Path | None = None):
        self.rule_path = Path(rule_path or (
            Path(__file__).resolve().parents[1] / "rules" / "active" /
            "dka_embodied.pl"
        ))
        self.engine = GroundProlog.from_file(self.rule_path)
        self.temporal_constraints = {
            clause.head.arguments[0]: {
                "window_hours": [
                    float(clause.head.arguments[1]) / 60.0,
                    float(clause.head.arguments[2]) / 60.0,
                ],
                "confidence": float(clause.head.arguments[3]) / 1_000_000.0,
            }
            for clause in self.engine.clauses
            if clause.head.predicate == "temporal_constraint"
            and len(clause.head.arguments) == 4
        }

    @staticmethod
    def _state_facts(state: dict) -> set[Atom]:
        return {Atom(name) for name in ground_dka_facts(state)}

    @staticmethod
    def _action_facts(action) -> set[Atom]:
        values = expand_action(action)
        facts = {
            Atom("requested", (name,))
            for name, value in zip(ACTION_KEYS, values)
            if float(value) > 1e-6
        }
        if any(float(value) > 1e-6 for value in values[:4]):
            facts.add(Atom("insulin_requested"))
        return facts

    def _run(self, state: dict, action):
        return self.engine.infer(self._state_facts(state) | self._action_facts(action))

    def active_expected_rule_ids(self) -> set[str]:
        return {
            clause.head.arguments[3]
            for clause in self.engine.clauses
            if clause.head.predicate == "expected" and len(clause.head.arguments) == 4
        }

    def validator_alignment(self, validator) -> dict:
        validator_ids = {rule.rule_id for rule in validator.rules}
        prolog_ids = self.active_expected_rule_ids()
        return {
            "aligned": validator_ids <= prolog_ids,
            "missing_from_prolog": sorted(validator_ids - prolog_ids),
            "prolog_only_rules": sorted(prolog_ids - validator_ids),
        }

    def gate(self, state: dict, action):
        """Apply hard Prolog vetoes after a controller or safety policy runs."""
        values = expand_action(action)
        facts, proofs = self._run(state, values)
        blocked = sorted(atom for atom in facts if atom.predicate == "blocked")
        trace = []
        for atom in blocked:
            action_name, reason = atom.arguments
            action_index = ACTION_KEYS.index(action_name)
            if float(values[action_index]) <= 1e-6:
                continue
            values[action_index] = 0.0
            trace.append({
                "type": "prolog_veto",
                "action": action_name,
                "reason": reason,
                "proof": self.engine.proof(atom, proofs),
            })
        return values.tolist(), trace

    def evaluate(self, state: dict, proposed_action, applied_action,
                 future: dict, baseline_future: dict,
                 elapsed_hours: float | None = None) -> dict:
        proposal_facts, proposal_proofs = self._run(state, proposed_action)
        applied_facts, applied_proofs = self._run(state, applied_action)

        blocked_atoms = sorted(
            atom for atom in proposal_facts if atom.predicate == "blocked"
        )
        required_atoms = sorted(
            atom for atom in proposal_facts if atom.predicate == "required"
        )
        allowed_atoms = sorted(
            atom for atom in proposal_facts if atom.predicate == "allowed"
        )
        expected_atoms = sorted(
            atom for atom in applied_facts if atom.predicate == "expected"
        )

        effect = {
            name: float(future[name]) - float(baseline_future[name])
            for name in future.keys() & baseline_future.keys()
        }
        effect_names = {name.lower(): name for name in effect}
        checks = []
        deferred = []
        for atom in expected_atoms:
            action_name, state_name, direction, rule_id = atom.arguments
            temporal = self.temporal_constraints.get(rule_id, {
                "window_hours": [0.0, 24.0], "confidence": 1.0,
            })
            if elapsed_hours is not None and not (
                temporal["window_hours"][0]
                <= float(elapsed_hours)
                <= temporal["window_hours"][1]
            ):
                deferred.append({
                    "rule_id": rule_id,
                    "action": action_name,
                    "expected_direction": direction,
                    **temporal,
                    "status": "outside_time_window",
                })
                continue
            model_state_name = effect_names.get(state_name.lower())
            if model_state_name is None:
                continue
            observed = effect[model_state_name]
            sign = 1 if direction == "increase" else -1
            status = "verified" if observed * sign >= 0.0 else "contradicted"
            checks.append({
                "rule_id": rule_id,
                "action": action_name,
                "variable": OSLER_STATE_ONTOLOGY.require(model_state_name),
                "model_variable": model_state_name,
                "expected_direction": direction,
                "observed_effect": round(observed, 6),
                "status": status,
                **temporal,
                "proof": self.engine.proof(atom, applied_proofs),
            })

        if blocked_atoms:
            decision = "block"
        elif required_atoms:
            decision = "modify"
        else:
            decision = "allow"
        if any(check["status"] == "contradicted" for check in checks):
            prediction_status = "contradicted"
        elif checks:
            prediction_status = "verified"
        else:
            prediction_status = "unexplained"

        blocked_by_action = {}
        for atom in blocked_atoms:
            blocked_by_action.setdefault(atom.arguments[0], []).append(atom)
        required_by_action = {}
        for atom in required_atoms:
            required_by_action.setdefault(atom.arguments[0], []).append(atom)

        return {
            "engine": "ground_prolog",
            "rule_source": str(self.rule_path.relative_to(self.rule_path.parents[2])),
            "decision": decision,
            "prediction_status": prediction_status,
            "allowed_actions": [atom.arguments[0] for atom in allowed_atoms],
            "blocked_actions": [{
                "action": action_name,
                "reasons": [atom.arguments[1] for atom in atoms],
                "proofs": [
                    self.engine.proof(atom, proposal_proofs) for atom in atoms
                ],
            } for action_name, atoms in sorted(blocked_by_action.items())],
            "required_cointerventions": [{
                "action": action_name,
                "reasons": [atom.arguments[1] for atom in atoms],
                "proofs": [
                    self.engine.proof(atom, proposal_proofs) for atom in atoms
                ],
            } for action_name, atoms in sorted(required_by_action.items())],
            "effect_checks": checks,
            "deferred_effect_checks": deferred,
            "research_only": True,
        }


OSLER_DKA_PROLOG = DkaEmbodiedLogic()
