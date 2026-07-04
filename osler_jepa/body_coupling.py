"""Fail-closed contracts for cross-body-system coupling.

The isolated disease/body-system routers validate that Osler-JEPA can move
targets where the data support movement and fall back to persistence elsewhere.
This module describes the next layer: directed coupling edges between body
systems.  A coupling edge is usable only if an aggregate audit shows that adding
the upstream system significantly improves held-out downstream prediction beyond
both a no-upstream baseline and a capacity-matched placebo.

This is a factual predictive contract.  It does not grant causal,
counterfactual, clinical, runtime, checkpoint-promotion, or active-rule
authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class CouplingEdgeSpec:
    name: str
    source_system: str
    target_system: str
    targets: tuple[str, ...]
    horizon_hours: int
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def body_system_coupling_edges() -> tuple[CouplingEdgeSpec, ...]:
    """Return the first directed cross-system coupling hypotheses."""

    return (
        CouplingEdgeSpec(
            name="renal_to_electrolyte_acid_base",
            source_system="renal",
            target_system="electrolyte_acid_base",
            targets=("potassium", "bicarbonate", "phosphate", "sodium", "anion_gap"),
            horizon_hours=6,
            rationale="renal filtration and renal-support state should help electrolyte and acid-base trajectories",
        ),
        CouplingEdgeSpec(
            name="respiratory_to_acid_base",
            source_system="respiratory",
            target_system="acid_base",
            targets=("ph", "bicarbonate", "lactate"),
            horizon_hours=6,
            rationale="ventilation and oxygenation state should help acid-base and lactate trajectories",
        ),
        CouplingEdgeSpec(
            name="endocrine_to_electrolyte",
            source_system="endocrine_metabolic",
            target_system="electrolyte_acid_base",
            targets=("potassium", "sodium", "bicarbonate", "anion_gap"),
            horizon_hours=6,
            rationale="glycemic/osmotic state and insulin/dextrose evidence should help electrolyte trajectories",
        ),
        CouplingEdgeSpec(
            name="heme_to_perfusion",
            source_system="hematologic_coagulation",
            target_system="cardiovascular_perfusion",
            targets=("map", "lactate", "heart_rate"),
            horizon_hours=24,
            rationale="oxygen-carrying capacity, coagulation, and transfusion evidence should help perfusion targets",
        ),
        CouplingEdgeSpec(
            name="cardiovascular_to_renal",
            source_system="cardiovascular_perfusion",
            target_system="renal",
            targets=("creatinine", "urine_output"),
            horizon_hours=6,
            rationale="perfusion and hemodynamic support should help short-horizon renal output markers",
        ),
        CouplingEdgeSpec(
            name="hepatic_to_coagulation_platelets",
            source_system="hepatic",
            target_system="hematologic_coagulation",
            targets=("platelets", "bicarbonate", "creatinine"),
            horizon_hours=6,
            rationale="hepatic failure state should help platelet/coagulation-proxy and renal-acid-base targets",
        ),
        CouplingEdgeSpec(
            name="immune_to_hemodynamics",
            source_system="immune_inflammatory",
            target_system="cardiovascular_perfusion",
            targets=("map", "lactate", "platelets", "albumin"),
            horizon_hours=6,
            rationale="inflammatory state should help hemodynamic, platelet, and albumin trajectories",
        ),
    )


def _target_passes(summary: Mapping[str, object], *, required_splits: int) -> bool:
    active = summary.get("active_only", {})
    return int(active.get("candidate_passes_both_count", 0)) >= required_splits


def coupling_readiness_from_audit(
    audit_report: Mapping[str, object],
    *,
    required_splits: int = 7,
) -> dict[str, object]:
    """Summarize which coupling edges may leave fail-closed candidate status."""

    promoted_edges: list[dict[str, object]] = []
    weak_candidate_edges: list[dict[str, object]] = []
    rejected_edges: list[dict[str, object]] = []

    edge_items = audit_report.get("edges")
    if edge_items is None:
        edge_items = audit_report.get("focused_specs", [])

    for edge in edge_items:
        summary = edge.get("random_patient_splits", {}).get("summary", {})
        promoted_targets = [
            target for target in edge.get("targets", ())
            if target in summary and _target_passes(summary[target], required_splits=required_splits)
        ]
        weak_targets = [
            target for target in edge.get("targets", ())
            if (
                target in summary
                and 0 < int(summary[target].get("active_only", {}).get("candidate_passes_both_count", 0)) < required_splits
            )
        ]
        entry = {
            "name": edge.get("name"),
            "source_system": edge.get("source_system"),
            "target_system": edge.get("target_system"),
            "promoted_targets": promoted_targets,
            "weak_candidate_targets": weak_targets,
        }
        if promoted_targets:
            promoted_edges.append(entry)
        elif weak_targets:
            weak_candidate_edges.append(entry)
        else:
            rejected_edges.append(entry)

    return {
        "promoted_edges": promoted_edges,
        "weak_candidate_edges": weak_candidate_edges,
        "rejected_edges": rejected_edges,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "runtime_decision_authority": False,
        "active_rule_promotion_allowed": False,
    }
