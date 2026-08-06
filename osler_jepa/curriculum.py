"""Loss curriculum for Osler-guided world-model training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    end_fraction: float
    weights: dict[str, float]


STAGES = (
    CurriculumStage("state_grounding", 0.15, {
        "one_step_latent": 1.0, "one_step_state": 1.2, "rollout_state": 0.2,
        "rollout_latent": 0.1, "effect": 0.0, "osler": 0.0,
        "reconstruction": 0.4, "anti_collapse": 0.25, "risk": 0.05,
        "direction": 0.15, "symbolic_status": 0.05, "proof_path": 0.05,
        "rule_proposal": 0.0, "contradiction": 0.0,
        "action_contrastive": 0.0,
        "world_truth": 0.0, "viability": 0.0,
        "intervention_sensitivity": 0.0, "trajectory_consistency": 0.0,
        "uncertainty_calibration": 0.02,
    }),
    CurriculumStage("disease_dynamics", 0.40, {
        "one_step_latent": 0.9, "one_step_state": 1.0, "rollout_state": 1.0,
        "rollout_latent": 0.3, "effect": 0.2, "osler": 0.1,
        "reconstruction": 0.25, "anti_collapse": 0.2, "risk": 0.1,
        "direction": 0.25, "symbolic_status": 0.15, "proof_path": 0.12,
        "rule_proposal": 0.15, "contradiction": 0.05,
        "action_contrastive": 0.05,
        "world_truth": 0.0, "viability": 0.0,
        "intervention_sensitivity": 0.0, "trajectory_consistency": 0.0,
        "uncertainty_calibration": 0.03,
    }),
    CurriculumStage("treatment_effects", 0.70, {
        "one_step_latent": 0.8, "one_step_state": 1.0, "rollout_state": 1.4,
        "rollout_latent": 0.4, "effect": 1.0, "osler": 0.5,
        "reconstruction": 0.15, "anti_collapse": 0.15, "risk": 0.15,
        "direction": 0.3, "symbolic_status": 0.2, "proof_path": 0.15,
        "rule_proposal": 0.3, "contradiction": 0.12,
        "action_contrastive": 0.12,
        "world_truth": 0.0, "viability": 0.0,
        "intervention_sensitivity": 0.0, "trajectory_consistency": 0.0,
        "uncertainty_calibration": 0.04,
    }),
    CurriculumStage("counterfactual_validation", 1.0, {
        "one_step_latent": 0.8, "one_step_state": 1.0, "rollout_state": 1.6,
        "rollout_latent": 0.4, "effect": 1.2, "osler": 0.8,
        "reconstruction": 0.15, "anti_collapse": 0.15, "risk": 0.15,
        "direction": 0.35, "symbolic_status": 0.25, "proof_path": 0.2,
        "rule_proposal": 0.4, "contradiction": 0.2,
        "action_contrastive": 0.2,
        "world_truth": 0.0, "viability": 0.0,
        "intervention_sensitivity": 0.0, "trajectory_consistency": 0.0,
        "uncertainty_calibration": 0.05,
    }),
)


def stage_for_epoch(epoch: int, total_epochs: int) -> CurriculumStage:
    fraction = epoch / max(total_epochs, 1)
    for stage in STAGES:
        if fraction <= stage.end_fraction:
            return stage
    return STAGES[-1]


def weights_for_stage(stage: CurriculumStage, viability_dynamics_enabled=False):
    """Return stage weights while keeping dynamics constraints gated by evidence."""
    weights = dict(stage.weights)
    if viability_dynamics_enabled:
        weights.update({
            "world_truth": 0.15,
            "viability": 0.10,
            "intervention_sensitivity": 0.05,
            "trajectory_consistency": 0.05,
        })
    return weights
