"""Promotion gate for enabling real-world-constrained JEPA dynamics losses."""

from __future__ import annotations

import json
from pathlib import Path


CORE_TARGETS = ("glucose", "potassium", "bicarbonate", "map")


def evaluate_persistence_gate(report, min_stays=30, min_pairs=20):
    """Require patient-held-out evidence that JEPA beats persistence everywhere."""
    external = report.get("mimic_external_proxy", report)
    adequacy = external.get("data_adequacy", {})
    metrics = external.get("mae_active_dka") or external.get("mae", {})
    reasons = []

    stays = int(external.get("stays", report.get("stays", 0)) or 0)
    if stays < min_stays:
        reasons.append(f"stays={stays} is below {min_stays}")

    requirements = {
        "has_exact_dose_and_timing": "exact dose/time actions are missing",
        "has_explicit_start_stop_events": "explicit treatment start/stop events are missing",
        "has_pre_anchor_treatment_history": "pre-anchor treatment history is missing",
    }
    for key, message in requirements.items():
        if not bool(adequacy.get(key, False)):
            reasons.append(message)

    comparisons = {}
    for target in CORE_TARGETS:
        item = metrics.get(target, {})
        count = int(item.get("n", 0) or 0)
        jepa = item.get("jepa")
        persistence = item.get("persistence")
        won = (
            count >= min_pairs
            and jepa is not None
            and persistence is not None
            and float(jepa) < float(persistence)
        )
        comparisons[target] = {
            "n": count,
            "jepa": jepa,
            "persistence": persistence,
            "won": won,
        }
        if count < min_pairs:
            reasons.append(f"{target} pairs={count} is below {min_pairs}")
        elif jepa is None or persistence is None:
            reasons.append(f"{target} comparison is missing")
        elif float(jepa) >= float(persistence):
            reasons.append(
                f"{target} JEPA MAE {float(jepa):.4g} did not beat "
                f"persistence {float(persistence):.4g}"
            )

    return {
        "passed": not reasons,
        "stays": stays,
        "minimum_stays": int(min_stays),
        "minimum_pairs_per_target": int(min_pairs),
        "core_targets": comparisons,
        "reasons": reasons,
        "clinical_authority": False,
    }


def load_persistence_gate(path, min_stays=30, min_pairs=20):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    result = evaluate_persistence_gate(report, min_stays=min_stays, min_pairs=min_pairs)
    result["source_report"] = str(path.resolve())
    return result
