"""Robustness checks for the MIMIC-IV full v3.1 DKA ensemble result.

The checks intentionally write aggregate metrics only. They do not write
row-level predictions, patient identifiers, stay lists, or timestamp cutoffs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dka_world_model import load_checkpoint
from eicu_dka_per_target_ensemble import (
    build_long_predictions,
    choose_selector,
    split_stays,
    summarize_scope,
)
from train_intervention_jepa import choose_device


def _int(value):
    return int(value) if value is not None else None


def _round(value):
    return round(float(value), 6) if value is not None and np.isfinite(value) else None


def _cohort_summary(frame: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "active_dka_rows": (
            int(frame["dka_active_t"].fillna(False).sum())
            if "dka_active_t" in frame else None
        ),
        "all_icd_supported": (
            bool(frame["icd_dka_support"].fillna(False).all())
            if "icd_dka_support" in frame else None
        ),
        "has_time_column_for_order_split": "t" in frame,
        "has_careunit_or_site_for_group_split": any(
            column in frame
            for column in ("first_careunit", "last_careunit", "hospitalid", "site_id")
        ),
    }
    for target, current, future in (
        ("glucose", "glucose_t", "glucose_tp6"),
        ("potassium", "potassium_t", "potassium_tp6"),
        ("bicarbonate", "bicarbonate_t", "bicarbonate_tp6"),
        ("map", "map_t", "map_tp6"),
    ):
        if current in frame and future in frame:
            summary[f"{target}_pairs"] = int(
                (frame[current].notna() & frame[future].notna()).sum()
            )
    return summary


def _split_report(rows: pd.DataFrame, discovery_stays: set[int],
                  heldout_stays: set[int], seed: int, min_pairs: int,
                  min_stays: int, bootstrap_samples: int) -> dict:
    discovery = rows[rows["stay_id"].isin(discovery_stays)].copy()
    heldout = rows[rows["stay_id"].isin(heldout_stays)].copy()
    selector, selector_details = choose_selector(
        discovery,
        active_only=True,
        min_pairs=min_pairs,
        min_stays=min_stays,
    )
    return {
        "seed": int(seed),
        "discovery_stays": int(len(discovery_stays)),
        "heldout_stays": int(len(heldout_stays)),
        "overlap": int(len(discovery_stays & heldout_stays)),
        "selected_methods": selector,
        "discovery_selector": selector_details,
        "heldout_all_windows": summarize_scope(
            heldout, selector, False, seed + 1000, bootstrap_samples
        ),
        "heldout_active_dka_only": summarize_scope(
            heldout, selector, True, seed + 2000, bootstrap_samples
        ),
    }


def _multi_seed_summary(reports: list[dict]) -> dict:
    active = [report["heldout_active_dka_only"] for report in reports]
    all_windows = [report["heldout_all_windows"] for report in reports]

    def collect(scopes: list[dict]) -> dict:
        deltas = [
            scope["ensemble_delta_vs_persistence"]["point_delta"]
            for scope in scopes
            if scope["ensemble_delta_vs_persistence"]["point_delta"] is not None
        ]
        significant = [
            bool(scope["ensemble_delta_vs_persistence"]["significant"])
            for scope in scopes
        ]
        beats = [
            bool(scope["ensemble_delta_vs_persistence"]["beats_persistence"])
            for scope in scopes
        ]
        return {
            "splits": int(len(scopes)),
            "beats_persistence_count": int(sum(beats)),
            "significant_count": int(sum(significant)),
            "delta_min": _round(min(deltas)) if deltas else None,
            "delta_median": _round(float(np.median(deltas))) if deltas else None,
            "delta_max": _round(max(deltas)) if deltas else None,
        }

    return {
        "active_dka_only": collect(active),
        "all_windows": collect(all_windows),
    }


def _time_order_split(frame: pd.DataFrame, rows: pd.DataFrame, train_fraction: float):
    if "t" not in frame:
        return None
    temp = frame[["stay_id", "t"]].copy()
    temp["t"] = pd.to_datetime(temp["t"], errors="coerce")
    order = temp.dropna(subset=["t"]).groupby("stay_id")["t"].min().sort_values()
    stays = order.index.to_numpy(dtype=np.int64)
    if len(stays) < 3:
        return None
    split = max(1, int(round(train_fraction * len(stays))))
    split = min(split, max(1, len(stays) - 1))
    early = set(stays[:split].tolist())
    late = set(stays[split:].tolist())
    return early, late


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--v5-checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument(
        "--presentation-checkpoint",
        default="dka_physionet_presentation_only_candidate.pt",
    )
    parser.add_argument("--output", default="mimiciv_full_v31_icd_robustness.json")
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--time-train-fraction", type=float, default=0.67)
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--min-stays", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [
        int(item.strip()) for item in args.seeds.split(",")
        if item.strip()
    ]
    device = choose_device(args.device)
    v5, _ = load_checkpoint(args.v5_checkpoint, device)
    presentation, _ = load_checkpoint(args.presentation_checkpoint, device)
    frame = pd.read_parquet(args.cohort)
    rows = build_long_predictions(
        frame,
        {"v5": v5, "presentation_only": presentation},
        device,
    )

    random_split_reports = []
    for seed in seeds:
        discovery_stays, heldout_stays = split_stays(
            rows, seed, args.discovery_fraction
        )
        random_split_reports.append(_split_report(
            rows,
            discovery_stays,
            heldout_stays,
            seed,
            args.min_pairs,
            args.min_stays,
            args.bootstrap_samples,
        ))

    time_split = _time_order_split(frame, rows, args.time_train_fraction)
    if time_split is None:
        time_report = {
            "available": False,
            "reason": "cohort has no usable time column for aggregate order split",
        }
    else:
        early, late = time_split
        time_report = {
            "available": True,
            "split_type": "early-stay discovery, late-stay heldout",
            **_split_report(
                rows,
                early,
                late,
                seed=9001,
                min_pairs=args.min_pairs,
                min_stays=args.min_stays,
                bootstrap_samples=args.bootstrap_samples,
            ),
        }

    report = {
        "experiment": "MIMIC-IV full v3.1 DKA ensemble robustness",
        "cohort": str(Path(args.cohort).resolve()),
        "cohort_summary": _cohort_summary(frame),
        "prediction_rows": int(len(rows)),
        "active_prediction_rows": int(rows["active_dka"].sum()),
        "selector_gate": {
            "scope": "discovery active-DKA rows",
            "min_pairs": int(args.min_pairs),
            "min_stays": int(args.min_stays),
            "rule": "choose lowest discovery MAE per target, otherwise persistence",
        },
        "random_patient_splits": {
            "seeds": seeds,
            "summary": _multi_seed_summary(random_split_reports),
            "runs": random_split_reports,
        },
        "time_order_split": time_report,
        "diagnosis_pure_subset": {
            "available": bool("icd_dka_support" in frame),
            "already_restricted_to_icd_supported": (
                bool(frame["icd_dka_support"].fillna(False).all())
                if "icd_dka_support" in frame else None
            ),
            "interpretation": (
                "This parquet is already the ICD-supported cohort; a stricter "
                "diagnosis-only subset would require adding diagnosis-code class "
                "metadata to extraction."
            ),
        },
        "hospital_or_careunit_split": {
            "available": False,
            "reason": (
                "MIMIC-IV full v3.1 parquet does not currently include careunit, "
                "hospital, or site columns. Add aggregate-safe unit metadata to "
                "the extractor before claiming hospital/careunit robustness."
            ),
        },
        "safety_boundary": {
            "row_level_predictions_written": False,
            "patient_identifiers_written": False,
            "timestamp_cutoffs_written": False,
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
        },
        "interpretation": [
            "A robustness pass means the ensemble beats persistence across patient split seeds and time-order split.",
            "This remains an observational factual proxy and does not identify treatment effects.",
            "Hospital/careunit robustness is not tested until the extractor includes non-identifying unit metadata.",
        ],
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
