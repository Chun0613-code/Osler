"""Persistence-anchored residual gating for DKA forecasts.

The model is allowed to leave persistence only when three conditions hold:

1. the candidate residual is directionally non-trivial,
2. the candidate direction matches an active Osler/Prolog transition rule, and
3. the candidate source has enough agreement/confidence.

Even then, only a shrunk and clipped residual is applied.  Prolog owns the
permission and direction; it does not estimate the numeric magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dka_world_model_contract import STATE_KEYS
from osler_jepa.symbolic import PHYSICAL_STABLE_THRESHOLDS
from osler_jepa.validator import OSLER_DKA_VALIDATOR


STATE_STD = np.asarray(
    [200, 0.3, 10, 10, 1.5, 30, 4, 25.0, 10, 35, 1.5, 180, 4.0, 35, 5.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AnchoredResidualConfig:
    min_direction_agreement: float = 0.9
    max_shrink: float = 0.35
    max_residual_scale: float = 0.25
    supported_horizon_hours: float = 6.0
    horizon_tolerance_hours: float = 0.25
    require_prolog_direction: bool = True
    require_direction_agreement: bool = True


def direction_class(delta: float, state_index: int) -> int:
    threshold = float(PHYSICAL_STABLE_THRESHOLDS[state_index])
    if float(delta) < -threshold:
        return 0
    if float(delta) > threshold:
        return 2
    return 1


def direction_sign(delta: float, state_index: int) -> int:
    klass = direction_class(delta, state_index)
    return -1 if klass == 0 else 1 if klass == 2 else 0


def _active_rule(rule, action_exposure, horizon_hours):
    if float(action_exposure[rule.action_index]) <= rule.min_action:
        return False
    if rule.blocker_action_index is not None:
        if float(action_exposure[rule.blocker_action_index]) > rule.blocker_max:
            return False
    for blocker_index in rule.blocker_action_indices:
        if float(action_exposure[blocker_index]) > rule.blocker_max:
            return False
    if horizon_hours is not None and not (
        rule.min_hours <= float(horizon_hours) <= rule.max_hours
    ):
        return False
    return True


def prolog_direction_gate(state_index: int, action_exposure: Iterable[float],
                          horizon_hours: float | None = 6.0) -> dict:
    action_exposure = np.asarray(action_exposure, dtype=np.float32)
    checks = []
    for rule in OSLER_DKA_VALIDATOR.rules:
        if rule.state_index != state_index:
            continue
        if not _active_rule(rule, action_exposure, horizon_hours):
            continue
        checks.append({
            "rule_id": rule.rule_id,
            "action": rule.action_name,
            "state": rule.state_name,
            "expected_sign": int(rule.expected_sign),
            "expected_direction": "increase" if rule.expected_sign > 0 else "decrease",
            "confidence": float(rule.confidence),
            "window_hours": [float(rule.min_hours), float(rule.max_hours)],
            "rationale": rule.rationale,
        })
    signs = {check["expected_sign"] for check in checks}
    if not checks:
        status = "no_active_prolog_direction"
        allowed_sign = None
        confidence = 0.0
    elif len(signs) > 1:
        status = "ambiguous_prolog_direction"
        allowed_sign = None
        confidence = 0.0
    else:
        status = "matched"
        allowed_sign = signs.pop()
        confidence = max(check["confidence"] for check in checks)
    return {
        "status": status,
        "allowed_sign": allowed_sign,
        "confidence": round(float(confidence), 6),
        "checks": checks,
    }


class AnchoredResidualGate:
    def __init__(self, config: AnchoredResidualConfig | None = None):
        self.config = config or AnchoredResidualConfig()

    def select(self, state_index: int, current: float, candidate: float,
               action_exposure: Iterable[float], direction_agreement: float | None,
               horizon_hours: float = 6.0) -> dict:
        raw_residual = float(candidate) - float(current)
        candidate_sign = direction_sign(raw_residual, state_index)
        reasons = []
        supported_horizon = (
            abs(float(horizon_hours) - self.config.supported_horizon_hours)
            <= self.config.horizon_tolerance_hours
        )
        if not supported_horizon:
            reasons.append("unsupported_horizon")
        if candidate_sign == 0:
            reasons.append("candidate_residual_below_stable_threshold")

        prolog = prolog_direction_gate(state_index, action_exposure, horizon_hours)
        if self.config.require_prolog_direction and prolog["status"] != "matched":
            reasons.append(prolog["status"])
        if prolog["status"] == "matched" and candidate_sign != 0:
            if candidate_sign != int(prolog["allowed_sign"]):
                reasons.append("candidate_direction_contradicts_prolog")

        if self.config.require_direction_agreement:
            if direction_agreement is None:
                reasons.append("missing_direction_agreement")
                agreement_factor = 0.0
            else:
                agreement = float(direction_agreement)
                if agreement < self.config.min_direction_agreement:
                    reasons.append("low_direction_agreement")
                agreement_factor = np.clip(
                    (agreement - self.config.min_direction_agreement)
                    / max(1e-6, 1.0 - self.config.min_direction_agreement),
                    0.0, 1.0,
                )
        else:
            agreement_factor = 1.0

        allowed = not reasons
        max_abs_residual = self.config.max_residual_scale * float(STATE_STD[state_index])
        clipped_residual = float(np.clip(
            raw_residual, -max_abs_residual, max_abs_residual
        ))
        prolog_factor = float(prolog["confidence"]) if prolog["status"] == "matched" else 0.0
        shrink = (
            self.config.max_shrink * float(agreement_factor) * prolog_factor
            if allowed else 0.0
        )
        selected = float(current) + shrink * clipped_residual
        return {
            "state": STATE_KEYS[state_index],
            "selected_prediction": round(selected, 6),
            "selected_source": "anchored_residual" if allowed else "persistence",
            "fallback_source": "persistence",
            "candidate_prediction": round(float(candidate), 6),
            "current_prediction": round(float(current), 6),
            "raw_residual": round(raw_residual, 6),
            "clipped_residual": round(clipped_residual, 6),
            "candidate_direction": (
                "increase" if candidate_sign > 0 else
                "decrease" if candidate_sign < 0 else "stable"
            ),
            "shrink": round(float(shrink), 6),
            "direction_agreement": None if direction_agreement is None else round(float(direction_agreement), 6),
            "prolog_gate": prolog,
            "allowed_to_leave_persistence": bool(allowed),
            "abstain_reasons": reasons,
            "contract": {
                "anchor": "persistence/current observed value",
                "numeric_form": "selected = persistence + shrink * clipped(candidate - persistence)",
                "prolog_role": "permission_and_direction_only",
                "clinical_or_causal_claim_allowed": False,
            },
        }
