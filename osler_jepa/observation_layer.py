"""Whole-body factual observation and prediction contracts.

This module is the non-causal observation layer for Osler-JEPA.  It collects
the currently validated factual routers, body-system edges, and multi-hop paths
into one fail-closed contract.  It is deliberately conservative: a capability
may support shadow observation or factual prediction only when an aggregate
held-out audit validated it.  Nothing here grants treatment, causal, clinical,
runtime, checkpoint-promotion, or active-rule authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


DENIED_AUTHORITIES = (
    "causal_claim",
    "counterfactual_treatment_effect",
    "clinical_decision",
    "runtime_treatment_authority",
    "checkpoint_promotion",
    "active_rule_promotion",
)

ALLOWED_OBSERVATION_USES = (
    "shadow_observation",
    "factual_prediction",
    "capability_reporting",
)


@dataclass(frozen=True)
class ObservationCapability:
    """A validated or candidate factual observation capability."""

    name: str
    status: str
    scope: str
    systems: tuple[str, ...]
    targets: tuple[str, ...]
    horizon_hours: tuple[int, ...]
    evidence: str
    allowed_uses: tuple[str, ...] = ALLOWED_OBSERVATION_USES
    denied_authorities: tuple[str, ...] = DENIED_AUTHORITIES

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def is_validated(self) -> bool:
        return self.status == "validated"


def whole_body_observation_capabilities() -> tuple[ObservationCapability, ...]:
    """Return the current whole-body observation/prediction capability map."""

    return (
        ObservationCapability(
            name="body_system_surface_factual_routers",
            status="validated",
            scope="isolated_body_system_router_surface",
            systems=(
                "cardiovascular",
                "respiratory",
                "neurologic_proxy",
                "renal_urinary",
                "electrolyte_acid_base",
                "endocrine_metabolic",
                "hepatic_gi_pancreatic_nutrition",
                "hematologic_coagulation",
                "immune_inflammatory",
                "musculoskeletal",
                "integumentary_proxy",
                "toxic_metabolic_proxy",
            ),
            targets=("dense_fast_physiologic_targets", "persistence_fallback_for_sparse_or_slow_targets"),
            horizon_hours=(6,),
            evidence="BODY_SYSTEM_COVERAGE_COMPLETE.md; 12 adult-ICU body-system routers with held-out gates",
        ),
        ObservationCapability(
            name="dka_factual_router",
            status="validated",
            scope="disease_router",
            systems=("endocrine_metabolic", "electrolyte_acid_base", "renal_perfusion"),
            targets=("glucose", "anion_gap", "potassium", "bicarbonate", "map"),
            horizon_hours=(6,),
            evidence="Nested per-target DKA factual router; MIMIC-IV robustness split gates",
        ),
        ObservationCapability(
            name="sepsis_factual_router",
            status="validated",
            scope="disease_router",
            systems=("immune_inflammatory", "cardiovascular", "renal", "respiratory"),
            targets=("map", "heart_rate", "lactate", "creatinine", "bun", "o2sat"),
            horizon_hours=(6,),
            evidence="EICU_SEPSIS_ROUTER_FINDINGS.md; full-eICU sepsis factual router",
        ),
        ObservationCapability(
            name="aki_long_horizon_factual_router",
            status="validated",
            scope="disease_router",
            systems=("renal_urinary", "fluid_electrolyte"),
            targets=("creatinine", "bun", "urine_output"),
            horizon_hours=(24, 48),
            evidence="AKI_LONG_HORIZON_FINDINGS.md; slow renal targets validate when horizon matches kinetics",
        ),
        ObservationCapability(
            name="respiratory_factual_router",
            status="validated",
            scope="disease_router",
            systems=("respiratory", "cardiovascular", "acid_base"),
            targets=("o2sat", "respiratory_rate", "map", "heart_rate", "ph", "bicarbonate"),
            horizon_hours=(6,),
            evidence="EICU_RESPIRATORY_ROUTER_FINDINGS.md; bounded full-eICU respiratory router",
        ),
        ObservationCapability(
            name="cardio_renal_long_horizon_coupling",
            status="validated",
            scope="single_hop_body_coupling",
            systems=("cardiovascular_perfusion", "renal_urinary"),
            targets=("creatinine", "bun"),
            horizon_hours=(24, 48),
            evidence="BODY_SYSTEM_FOCUSED_COUPLING_FINDINGS.md; cardio->renal passes 7/7 for creatinine/BUN",
        ),
        ObservationCapability(
            name="sepsis_map_short_horizon_coupling",
            status="validated",
            scope="single_hop_body_coupling",
            systems=("immune_inflammatory", "cardiovascular_perfusion"),
            targets=("map",),
            horizon_hours=(6,),
            evidence="BODY_SYSTEM_FOCUSED_COUPLING_FINDINGS.md; sepsis/immune->MAP passes 7/7",
        ),
        ObservationCapability(
            name="sepsis_map6_renal24_multihop",
            status="validated",
            scope="multi_hop_body_coupling",
            systems=("immune_inflammatory", "cardiovascular_perfusion", "renal_urinary"),
            targets=("creatinine", "bun"),
            horizon_hours=(24,),
            evidence="BODY_SYSTEM_MULTIHOP_COUPLING_FINDINGS.md; sepsis->MAP at 6h -> renal at 24h passes 7/7 and hospital-heldout",
        ),
        ObservationCapability(
            name="renal_belief_state_v2",
            status="validated",
            scope="predict_update_belief_state",
            systems=("renal_urinary", "cardiovascular_perfusion"),
            targets=("creatinine", "bun", "renal_reserve"),
            horizon_hours=(24, 48),
            evidence="AKI_RENAL_BELIEF_STATE_V2_FINDINGS.md; downstream observable gate beats baseline and capacity-matched placebo",
        ),
        ObservationCapability(
            name="respiratory_acid_base_observed_coupling",
            status="candidate_only",
            scope="single_hop_body_coupling",
            systems=("respiratory", "acid_base"),
            targets=("bicarbonate", "ph", "paco2", "lactate"),
            horizon_hours=(6,),
            evidence="eicu_body_system_focused_respiratory_acid_base_observed_6h_audit.json; first-class PaCO2/FiO2/PEEP/VT still fails active-window gate",
        ),
        ObservationCapability(
            name="hepato_renal_coupling",
            status="candidate_only",
            scope="single_hop_body_coupling",
            systems=("hepatic_gi", "renal_urinary"),
            targets=("creatinine", "bun"),
            horizon_hours=(6, 24, 48),
            evidence="BODY_SYSTEM_FOCUSED_COUPLING_FINDINGS.md; horizon alone does not rescue hepato-renal edge",
        ),
        ObservationCapability(
            name="renal_electrolyte_store_coupling",
            status="candidate_only",
            scope="single_hop_body_coupling",
            systems=("renal_urinary", "electrolyte_acid_base"),
            targets=("potassium", "bicarbonate", "anion_gap", "sodium", "phosphate"),
            horizon_hours=(6,),
            evidence="BODY_SYSTEM_FOCUSED_COUPLING_FINDINGS.md; explicit store features do not pass robust gate",
        ),
        ObservationCapability(
            name="sepsis_map6_renal48_multihop",
            status="candidate_only",
            scope="multi_hop_body_coupling",
            systems=("immune_inflammatory", "cardiovascular_perfusion", "renal_urinary"),
            targets=("creatinine", "bun", "urine_output"),
            horizon_hours=(48,),
            evidence="BODY_SYSTEM_MULTIHOP_COUPLING_FINDINGS.md; 48h path reaches 5/7 for creatinine/BUN and fails direct-baseline hospital-heldout",
        ),
    )


def observation_readiness() -> dict[str, object]:
    """Return a compact readiness summary for UI/API consumers."""

    capabilities = whole_body_observation_capabilities()
    validated = [capability for capability in capabilities if capability.is_validated]
    candidates = [capability for capability in capabilities if not capability.is_validated]
    return {
        "validated_count": len(validated),
        "candidate_count": len(candidates),
        "validated_capabilities": [capability.to_dict() for capability in validated],
        "candidate_or_closed_capabilities": [capability.to_dict() for capability in candidates],
        "prediction_policy": {
            "fast_horizon_hours": 6,
            "slow_renal_horizon_hours": [24, 48],
            "fallback": "persistence for unsupported, sparse, or slow targets at the wrong horizon",
            "selection_rule": "use only capabilities validated by held-out aggregate gates; otherwise abstain or fall back",
        },
        "safety_boundary": {
            "observation_and_factual_prediction_allowed": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
            "checkpoint_promotion_allowed": False,
            "active_rule_promotion_allowed": False,
        },
    }
