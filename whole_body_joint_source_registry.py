"""Scientifically gated provenance groups for joint-JEPA experiments.

Entries here are factual source-attribution findings, not causal pathways and
not runtime permissions.  A group can be offered to a constrained adapter
experiment only after it has passed the patient- and hospital-held-out source
ablation audit.  The adapter itself must still pass the normal runtime gate.
"""

from __future__ import annotations


PHYSIOLOGY_GROUPS = {
    "metabolic_renal": (
        "aki",
        "electrolyte_acid_base",
        "endocrine_stress",
        "gi_pancreatic_nutrition",
        "musculoskeletal_rhabdo",
    ),
    "inflammatory_heme": (
        "sepsis",
        "immune_inflammatory",
        "integumentary_skin_wound",
        "coagulopathy_heme",
    ),
    "cardiorespiratory": (
        "cardiovascular_instability",
        "cardiac_injury",
        "respiratory",
    ),
    "neuro_hepatic_toxic": (
        "acute_neuro",
        "hepatic_failure",
        "toxic_metabolic",
    ),
}

# This is deliberately one entry.  It records the result of
# whole_body_joint_creatinine_group_attribution_5plus_10k.json:
# metabolic_renal -> creatinine@12h passed all seven patient-held-out paired
# bootstraps and the hospital-held-out paired bootstrap.  It authorizes an
# *experiment*, not an inference-time forecast override.
ATTRIBUTION_VALIDATED_GROUP_ADAPTERS = {
    "metabolic_renal->creatinine@12h": {
        "group": "metabolic_renal",
        "target": "creatinine",
        "horizons": (12.0,),
        # Keep BUN/electrolyte/other renal context while masking only the
        # current creatinine value.  This matches the attribution contract.
        "target_mask_scope": "target_only",
        # Factual administration history available at or before the anchor.
        # This is deliberately narrower than the full treatment table so the
        # adapter can test renal-relevant context without exposing every
        # target to hospital workflow proxies.
        "treatment_history_fields": (
            "hist_fluids",
            "hist_fluids_evidence_count",
            "hist_diuretics",
            "hist_diuretics_evidence_count",
            "hist_renal_replacement",
            "hist_renal_replacement_evidence_count",
            "hist_nephrotoxin",
            "hist_nephrotoxin_evidence_count",
            "hist_vasopressor",
            "hist_vasopressor_evidence_count",
            "hist_albumin",
            "hist_albumin_evidence_count",
        ),
        "evidence": "whole_body_joint_creatinine_group_attribution_5plus_10k.json",
    },
}
