"""Build the chapter A/B handoff contract after the DKA factual arc.

A = causal chapter: fail-closed RCT/IV/front-door evidence contract.
B = multi-disease chapter: reusable disease-router templates.

The output is aggregate planning metadata only. It contains no patient rows or
identifiers and makes no clinical claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osler_jepa.causal_readiness import (
    causal_source_registry,
    default_causal_specs,
    evaluate_causal_evidence_contract,
)
from osler_jepa.disease_router import disease_expansion_blueprint


def build_contract() -> dict:
    causal_specs = default_causal_specs()
    causal_readiness = {
        spec.name: evaluate_causal_evidence_contract(None, spec)
        for spec in causal_specs
    }
    return {
        "artifact": "Osler-JEPA next chapter A/B contract",
        "context": {
            "completed_arc": (
                "DKA factual advisory converged to a nested per-target router: "
                "glucose -> presentation-only, anion_gap -> real-fit grey-box "
                "residual, other targets -> persistence."
            ),
            "observational_causal_status": (
                "Full MIMIC-IV DKA diagnostics are runnable but fail causal "
                "readiness due overlap, co-treatment, and balance."
            ),
        },
        "chapter_a_causal": {
            "goal": "open causal what-if planning only with external identification evidence",
            "source_registry": causal_source_registry(),
            "readiness": causal_readiness,
            "allowed_now": [
                "prepare RCT/IV/front-door data dictionaries",
                "run fail-closed readiness checks when external evidence arrives",
                "continue observational confounding diagnostics as negative evidence",
            ],
            "forbidden_now": [
                "causal treatment-effect claims from current observational EHR",
                "clinical treatment recommendations from router deltas",
                "runtime action authority",
            ],
        },
        "chapter_b_multi_disease": disease_expansion_blueprint(),
        "shared_safety_boundary": {
            "row_level_predictions_written": False,
            "patient_identifiers_written": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
            "active_rule_promotion_allowed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="next_chapter_ab_contract.json")
    args = parser.parse_args()
    report = build_contract()
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
