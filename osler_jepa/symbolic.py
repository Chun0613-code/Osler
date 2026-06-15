"""Symbolic supervision shared by the DKA JEPA and Osler validator.

The neural model may predict transitions and propose observational rules. The
active rule set remains owned by :mod:`osler_jepa.validator`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from osler_jepa.validator import OSLER_DKA_VALIDATOR


DIRECTION_NAMES = ("decrease", "stable", "increase")
STATUS_NAMES = ("verified", "contradicted", "unexplained")
RULE_IDS = tuple(rule.rule_id for rule in OSLER_DKA_VALIDATOR.rules)
PHYSICAL_STABLE_THRESHOLDS = (
    5.0, 0.005, 0.2, 0.2, 0.02, 0.5, 0.05, 0.5,
    0.2, 1.0, 0.05, 5.0, 0.1, 1.0, 0.1,
)


def direction_targets(delta, state_scale):
    """Map normalized state deltas to decrease/stable/increase classes."""
    thresholds = torch.as_tensor(
        PHYSICAL_STABLE_THRESHOLDS, dtype=delta.dtype, device=delta.device
    ) / torch.as_tensor(state_scale, dtype=delta.dtype, device=delta.device)
    targets = torch.ones_like(delta, dtype=torch.long)
    targets = torch.where(delta < -thresholds, torch.zeros_like(targets), targets)
    targets = torch.where(delta > thresholds, torch.full_like(targets, 2), targets)
    return targets


def rule_supervision(effect, action_exposure, valid_mask=None):
    """Create Osler status and proof-path labels for matched action effects."""
    shape = effect.shape[:-1]
    proof = effect.new_zeros((*shape, len(RULE_IDS)))
    contradicted = torch.zeros(shape, dtype=torch.bool, device=effect.device)
    explained = torch.zeros(shape, dtype=torch.bool, device=effect.device)

    for index, rule in enumerate(OSLER_DKA_VALIDATOR.rules):
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
        signed = effect[..., rule.state_index] * float(rule.expected_sign)
        verified = signed >= rule.min_effect
        proof[..., index] = active.to(effect.dtype)
        explained = explained | active
        contradicted = contradicted | (active & ~verified)

    status = torch.full(shape, 2, dtype=torch.long, device=effect.device)
    status = torch.where(explained, torch.zeros_like(status), status)
    status = torch.where(contradicted, torch.ones_like(status), status)
    return status, proof


def masked_cross_entropy(logits, targets, mask):
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    while mask.ndim < losses.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(losses).to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def masked_binary_cross_entropy(logits, targets, mask):
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    while mask.ndim < losses.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(losses).to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def contradiction_penalty(proposal_logits, action_exposure, valid_mask=None):
    """Penalize proposal probabilities that oppose active Osler rules."""
    probabilities = proposal_logits.softmax(dim=-1)
    penalties = []
    for rule in OSLER_DKA_VALIDATOR.rules:
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
        if active.any():
            expected_class = 2 if rule.expected_sign > 0 else 0
            expected = probabilities[..., rule.state_index, expected_class]
            penalties.append((1.0 - expected)[active].mean())
    return torch.stack(penalties).mean() if penalties else proposal_logits.new_tensor(0.0)


def schema(state_keys, action_keys, ontology):
    return {
        "version": "1.0.0",
        "ownership": {
            "active_rules": "Osler validator and human review",
            "candidate_rules": "JEPA observational proposals only",
        },
        "state_variables": [
            {"model_name": name, "canonical": ontology.require(name)}
            for name in state_keys
        ],
        "actions": list(action_keys),
        "directions": list(DIRECTION_NAMES),
        "transition_status": list(STATUS_NAMES),
        "proof_paths": list(RULE_IDS),
        "candidate_rule_level": "observational_association",
    }
