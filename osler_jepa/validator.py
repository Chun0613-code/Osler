"""Symbolic transition validation and differentiable consistency penalties."""

from __future__ import annotations

from dataclasses import dataclass
import torch


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


OSLER_DKA_VALIDATOR = OslerTransitionValidator((
    TransitionRule("iv_insulin_lowers_glucose", 0, "insulin_iv", 0, "G", -1,
                   blocker_action_index=7, min_effect=0.001,
                   rationale="Insulin increases glucose uptake and suppresses hepatic output."),
    TransitionRule("rapid_sc_insulin_lowers_glucose", 1, "insulin_rapid_sc", 0, "G", -1,
                   blocker_action_index=7, min_effect=0.0001,
                   rationale="Rapid subcutaneous insulin is absorbed from a depot and lowers glucose."),
    TransitionRule("iv_insulin_lowers_potassium_without_kcl", 0, "insulin_iv", 4, "Ke", -1,
                   blocker_action_index=5, min_effect=0.0005,
                   rationale="Insulin shifts extracellular potassium into cells."),
    TransitionRule("fluids_raise_map", 4, "fluids", 5, "MAP", 1,
                   min_effect=0.0005,
                   rationale="Volume expansion raises effective circulating volume and MAP."),
    TransitionRule("fluids_raise_volume", 4, "fluids", 6, "V", 1,
                   min_effect=0.0005,
                   rationale="Administered crystalloid increases extracellular volume."),
    TransitionRule("kcl_raises_potassium_without_iv_insulin", 5, "kcl", 4, "Ke", 1,
                   blocker_action_indices=(0, 1, 2, 3), min_effect=0.0005,
                   rationale="Potassium chloride replaces extracellular and total-body potassium."),
    TransitionRule("kcl_raises_total_body_store", 5, "kcl", 13, "K_store", 1,
                   min_effect=0.0005,
                   rationale="Potassium replacement replenishes the latent total-body reserve."),
    TransitionRule("bicarbonate_raises_hco3", 6, "bicarbonate", 2, "HCO3", 1,
                   min_effect=0.0005,
                   rationale="Administered bicarbonate directly increases bicarbonate availability."),
    TransitionRule("bicarbonate_raises_ph", 6, "bicarbonate", 1, "pH", 1,
                   min_effect=0.0001,
                   rationale="Higher bicarbonate raises pH under the simulator acid-base relation."),
    TransitionRule("dextrose_raises_glucose_without_iv_insulin", 7, "dextrose", 0, "G", 1,
                   blocker_action_indices=(0, 1, 2, 3), min_effect=0.0005,
                   rationale="Administered dextrose supplies exogenous glucose."),
))
