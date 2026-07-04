"""Grounded homeostatic objectives for action-conditioned world-model learning.

The objective rewards fidelity to observed or simulated outcomes. It does not
reward a model merely for predicting that a patient becomes healthier, and it
does not turn prediction error into a clinical reward. This keeps curiosity and
clinical correctness as separate signals.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from osler_jepa.state_compiler import DkaStateCompiler


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values).to(dtype=values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


class HomeostaticWorldModelObjective:
    """Decomposed world-model loss with explicit physiologic grounding."""

    def __init__(self, compiler: DkaStateCompiler):
        self.compiler = compiler
        self._index = {
            name: index for index, name in enumerate(compiler.state_keys)
        }

    def physiologic_burden(self, normalized_state: torch.Tensor) -> torch.Tensor:
        residual = self.compiler.tensor_residual(normalized_state)
        reference = normalized_state.new_tensor(self.compiler.reference_mask)
        range_burden = (
            residual.square() * reference
        ).sum(dim=-1) / reference.sum().clamp_min(1.0)

        mean = normalized_state.new_tensor(self.compiler.state_mean)
        std = normalized_state.new_tensor(self.compiler.state_std)
        physical = normalized_state * std + mean
        value = lambda name: physical[..., self._index[name]]
        critical = (
            F.relu((6.9 - value("pH")) / 0.1).square()
            + F.relu((3.0 - value("Ke")) / 0.5).square()
            + F.relu((value("Ke") - 6.0) / 0.5).square()
            + F.relu((55.0 - value("MAP")) / 15.0).square()
            + F.relu((70.0 - value("G")) / 30.0).square()
            + F.relu((value("osmotic_injury") - 8.0) / 4.0).square()
        )
        return range_burden + critical

    def grounded_viability_reward(
        self,
        observed_current: torch.Tensor,
        observed_future: torch.Tensor,
    ) -> torch.Tensor:
        """Positive only when the grounded future has lower physiologic burden."""
        return (
            self.physiologic_burden(observed_current)
            - self.physiologic_burden(observed_future)
        )

    def components(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        log_variance: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        error = (predicted - target).square()
        truth = _masked_mean(error, valid)

        predicted_residual = self.compiler.tensor_residual(predicted)
        target_residual = self.compiler.tensor_residual(target)
        viability = _masked_mean(
            (predicted_residual - target_residual).square(), valid
        )

        bounded_log_variance = log_variance.clamp(-6.0, 4.0)
        predicted_std = torch.exp(0.5 * bounded_log_variance)
        calibration = (
            predicted_std - torch.sqrt(error.detach() + 1e-6)
        ).square()
        uncertainty = _masked_mean(calibration, valid)

        zero = truth.new_tensor(0.0)
        intervention_sensitivity = zero
        consistency = zero
        if predicted.ndim >= 4 and predicted.shape[1] > 1:
            predicted_burden = self.physiologic_burden(predicted)
            target_burden = self.physiologic_burden(target)
            pair_valid = valid[:, 1:] * valid[:, :1]
            predicted_effect = predicted[:, 1:] - predicted[:, :1]
            target_effect = target[:, 1:] - target[:, :1]
            changed_weight = torch.sigmoid(
                30.0 * (target_effect.abs().mean(dim=-1) - 0.02)
            )
            intervention_sensitivity = _masked_mean(
                (predicted_effect - target_effect).square(),
                pair_valid * changed_weight,
            )

            predicted_burden_delta = (
                predicted_burden[:, 1:] - predicted_burden[:, :1]
            )
            target_burden_delta = target_burden[:, 1:] - target_burden[:, :1]
            ordered = target_burden_delta.abs() > 0.05
            ordering_mask = pair_valid.bool() & ordered
            if ordering_mask.any():
                direction = target_burden_delta.sign()
                consistency = F.softplus(
                    -direction * predicted_burden_delta / 0.1
                )[ordering_mask].mean()

        return {
            "world_truth": truth,
            "viability": viability,
            "intervention_sensitivity": intervention_sensitivity,
            "trajectory_consistency": consistency,
            "uncertainty_calibration": uncertainty,
        }
