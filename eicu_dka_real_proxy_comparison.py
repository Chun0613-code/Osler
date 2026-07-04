"""Build the aggregate eICU DKA real-proxy comparison report.

Inputs are the transition report plus two factual proxy evaluations. The output
contains no row-level predictions and makes no causal or clinical claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = (
    "glucose", "ph", "bicarbonate", "anion_gap", "potassium", "map",
    "sodium", "osmolality", "creatinine", "urine_output",
)


def target_block(v5, presentation, section):
    output = {}
    for target in TARGETS:
        v5_stats = v5[section].get(target, {})
        presentation_stats = presentation[section].get(target, {})
        persistence = v5_stats.get("persistence")
        v5_jepa = v5_stats.get("jepa")
        presentation_jepa = presentation_stats.get("jepa")
        output[target] = {
            "n": v5_stats.get("n"),
            "persistence": persistence,
            "v5_jepa": v5_jepa,
            "v5_beats_persistence": (
                v5_jepa is not None and persistence is not None
                and v5_jepa < persistence
            ),
            "presentation_only_jepa": presentation_jepa,
            "presentation_only_beats_persistence": (
                presentation_jepa is not None and persistence is not None
                and presentation_jepa < persistence
            ),
        }
    return output


def build_report(args):
    cohort = json.loads(Path(args.cohort_report).read_text())
    v5 = json.loads(Path(args.v5_eval).read_text())
    presentation = json.loads(Path(args.presentation_eval).read_text())
    return {
        "experiment": "eicu_demo_dka_observed_treatment_real_proxy",
        "cohort_report": str(args.cohort_report),
        "cohort": {
            "stays": cohort["stays"],
            "transitions": cohort["transitions"],
            "active_dka_transitions": cohort["active_dka_transitions"],
            "core_pair_counts": cohort["core_pair_counts"],
            "action_target_support": cohort["action_target_support"],
            "treatment_presence_evidence": cohort.get(
                "treatment_presence_evidence", {}
            ),
            "power_proxy": cohort["power_proxy"],
        },
        "models": {
            "v5": Path(args.v5_checkpoint).name,
            "presentation_only_candidate": Path(
                args.presentation_checkpoint
            ).name,
        },
        "all_windows": target_block(v5, presentation, "mae"),
        "active_dka": target_block(v5, presentation, "mae_active_dka"),
        "decision": "no_checkpoint_promotion_research_proxy_only",
        "interpretation": [
            "The numeric action grid now uses only defensible infusionDrug "
            "administrations; medication orders and coarse treatment rows are "
            "presence evidence only.",
            "The previous eICU demo power-gate claim is withdrawn because it "
            "depended on treating orders as administrations.",
            "v5 still beats active-DKA persistence for bicarbonate, potassium, "
            "and MAP, while presentation-only still beats glucose and MAP.",
            "Most active-DKA insulin information is evidence-only rather than "
            "dose-observed, so this remains underpowered for promotion.",
            "This is a factual observed-treatment proxy with treatment selection "
            "unadjusted; it cannot support causal or clinical claims.",
        ],
        "safety_boundary": {
            "factual_observed_treatment_only": True,
            "medication_orders_used_as_numeric_administrations": False,
            "unknown_dose_evidence_used_in_action_grid": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-report", default="eicu_dka_transition_report.json")
    parser.add_argument("--v5-eval", default="eicu_dka_v5_evaluation.json")
    parser.add_argument(
        "--presentation-eval",
        default="eicu_dka_physionet_presentation_only_evaluation.json",
    )
    parser.add_argument("--v5-checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument(
        "--presentation-checkpoint",
        default="dka_physionet_presentation_only_candidate.pt",
    )
    parser.add_argument("--output", default="eicu_dka_real_proxy_comparison.json")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output,
        "decision": report["decision"],
        "support_passes_reference": (
            report["cohort"]["power_proxy"]["support_passes_reference"]
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
