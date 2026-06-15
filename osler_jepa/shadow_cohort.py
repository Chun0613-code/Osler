"""Patient-grouped cohort evaluation for JEPA shadow reconciliations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


CORE_DKA_STATES = ("G", "Ke", "HCO3", "MAP")


def load_reconciliations(path: str | Path) -> list[Dict[str, Any]]:
    """Load and de-duplicate the latest reconciliation per forecast/candidate."""
    ledger = Path(path)
    if not ledger.exists():
        raise FileNotFoundError(f"shadow ledger not found: {ledger}")
    latest = {}
    with ledger.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {ledger}:{line_number}: {exc.msg}"
                ) from exc
            if record.get("record_type") != "shadow_outcome_reconciliation":
                continue
            key = (record.get("forecast_event_id"), record.get("candidate"))
            latest[key] = record
    return list(latest.values())


def _patient_means(values: Iterable[tuple[str, float]]) -> Dict[str, float]:
    grouped = defaultdict(list)
    for subject, value in values:
        if subject and np.isfinite(value):
            grouped[subject].append(float(value))
    return {
        subject: float(np.mean(subject_values))
        for subject, subject_values in grouped.items()
    }


def _patient_bootstrap_interval(
    patient_values: Mapping[str, float], samples=2000, seed=23
):
    subjects = np.asarray(sorted(patient_values), dtype=object)
    if len(subjects) < 2:
        return None
    values = np.asarray([patient_values[subject] for subject in subjects], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(samples), dtype=float)
    for index in range(int(samples)):
        selected = rng.integers(0, len(values), size=len(values))
        estimates[index] = values[selected].mean()
    return [
        round(float(np.quantile(estimates, 0.025)), 6),
        round(float(np.quantile(estimates, 0.975)), 6),
    ]


def _metric_summary(values, samples, seed):
    patient_values = _patient_means(values)
    return {
        "patient_count": len(patient_values),
        "patient_mean": (
            round(float(np.mean(list(patient_values.values()))), 6)
            if patient_values else None
        ),
        "patient_cluster_bootstrap_95_ci": _patient_bootstrap_interval(
            patient_values, samples=samples, seed=seed
        ),
    }


def _checkpoint_id(record):
    checkpoint = record.get("checkpoint") or {}
    return checkpoint.get("sha256") or checkpoint.get("name") or "unknown"


def evaluate_shadow_group(
    records: Iterable[Mapping[str, Any]],
    bootstrap_samples: int = 2000,
    seed: int = 23,
    minimum_subjects: int = 30,
    minimum_episodes: int = 50,
    minimum_state_subjects: int = 20,
    minimum_changed_direction_accuracy: float = 0.60,
) -> Dict[str, Any]:
    records = list(records)
    exclusions = Counter()
    eligible = []
    for record in records:
        if record.get("status") != "scored":
            exclusions[f"status:{record.get('status')}"] += 1
        elif not record.get("subject_group_hash"):
            exclusions["missing_subject_group"] += 1
        elif not record.get("summary"):
            exclusions["missing_summary"] += 1
        else:
            eligible.append(record)

    subjects = sorted({record["subject_group_hash"] for record in eligible})
    improvement = _metric_summary([
        (
            record["subject_group_hash"],
            record["summary"]["normalized_improvement_over_persistence"],
        )
        for record in eligible
    ], bootstrap_samples, seed)
    changed_direction = _metric_summary([
        (
            record["subject_group_hash"],
            record["summary"]["jepa_changed_state_direction_accuracy"],
        )
        for record in eligible
        if record["summary"].get("jepa_changed_state_direction_accuracy") is not None
    ], bootstrap_samples, seed + 1)
    brier = _metric_summary([
        (record["subject_group_hash"], record["terminal_outcome"]["brier_score"])
        for record in eligible
        if record.get("terminal_outcome", {}).get("brier_score") is not None
    ], bootstrap_samples, seed + 2)

    state_reports = {}
    for state in sorted({
        state for record in eligible for state in (record.get("per_state") or {})
    }):
        rows = [
            (record["subject_group_hash"], record["per_state"][state])
            for record in eligible if state in (record.get("per_state") or {})
        ]
        state_reports[state] = {
            "normalized_improvement_over_persistence": _metric_summary([
                (subject, row["normalized_improvement_over_persistence"])
                for subject, row in rows
            ], bootstrap_samples, seed + 11 + len(state_reports)),
            "jepa_normalized_error": _metric_summary([
                (subject, row["jepa_normalized_error"])
                for subject, row in rows
            ], bootstrap_samples, seed + 31 + len(state_reports)),
            "persistence_normalized_error": _metric_summary([
                (subject, row["persistence_normalized_error"])
                for subject, row in rows
            ], bootstrap_samples, seed + 51 + len(state_reports)),
        }

    disagreements = sum(
        bool((record.get("forecast_provenance") or {}).get("symbolic_disagreement"))
        for record in eligible
    )
    failures = []
    if len(subjects) < minimum_subjects:
        failures.append("too_few_independent_subjects")
    if len(eligible) < minimum_episodes:
        failures.append("too_few_scored_episodes")
    improvement_ci = improvement["patient_cluster_bootstrap_95_ci"]
    if improvement_ci is None or improvement_ci[0] <= 0.0:
        failures.append("does_not_beat_persistence_with_95pct_confidence")
    direction_mean = changed_direction["patient_mean"]
    if direction_mean is None or direction_mean < minimum_changed_direction_accuracy:
        failures.append("changed_state_direction_accuracy_below_gate")
    for state in CORE_DKA_STATES:
        state_improvement = state_reports.get(state, {}).get(
            "normalized_improvement_over_persistence", {}
        )
        state_count = state_improvement.get("patient_count", 0)
        if state_count < minimum_state_subjects:
            failures.append(f"insufficient_subject_coverage:{state}")
        state_ci = state_improvement.get("patient_cluster_bootstrap_95_ci")
        if state_ci is None or state_ci[0] <= 0.0:
            failures.append(f"core_state_not_better_than_persistence:{state}")
    if disagreements:
        failures.append("symbolic_disagreement_present")

    return {
        "support": {
            "input_records": len(records),
            "scored_episodes": len(eligible),
            "independent_subjects": len(subjects),
            "excluded_records": dict(sorted(exclusions.items())),
            "duplicate_patient_weighting": "one patient mean before bootstrap",
        },
        "overall": {
            "normalized_improvement_over_persistence": improvement,
            "changed_state_direction_accuracy": changed_direction,
            "terminal_risk_brier": brier,
            "symbolic_disagreement_count": disagreements,
        },
        "states": state_reports,
        "automated_retrospective_gate": {
            "passes": not failures,
            "failures": failures,
            "thresholds": {
                "minimum_independent_subjects": minimum_subjects,
                "minimum_scored_episodes": minimum_episodes,
                "minimum_subjects_per_core_state": minimum_state_subjects,
                "minimum_changed_state_direction_accuracy": (
                    minimum_changed_direction_accuracy
                ),
                "improvement_ci_lower_bound_must_exceed": 0.0,
                "core_state_ci_lower_bound_must_exceed": 0.0,
            },
        },
        "clinical_promotion_allowed": False,
        "causal_claim_allowed": False,
        "online_weight_update_allowed": False,
        "automatic_rule_promotion_allowed": False,
        "limitations": [
            "Shadow outcomes are observational factual forecasts, not causal effects.",
            "Bootstrap intervals quantify sampling variability, not hidden bias.",
            "Passing the automated gate requires external validation and human review.",
        ],
    }


def evaluate_shadow_cohort(
    records: Iterable[Mapping[str, Any]],
    checkpoint: Optional[str] = None,
    candidate: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    selected = list(records)
    if checkpoint is not None:
        selected = [record for record in selected if _checkpoint_id(record) == checkpoint]
    if candidate is not None:
        selected = [record for record in selected if record.get("candidate") == candidate]
    groups = defaultdict(list)
    for record in selected:
        key = f"{_checkpoint_id(record)}::{record.get('candidate') or 'unknown'}"
        groups[key].append(record)
    reports = {}
    for key, group in sorted(groups.items()):
        report = evaluate_shadow_group(group, **kwargs)
        report["identity"] = {
            "checkpoint": group[0].get("checkpoint"),
            "candidate": group[0].get("candidate"),
        }
        reports[key] = report
    return {
        "schema_version": "1.0.0",
        "evaluation": "patient-grouped JEPA shadow forecast audit",
        "groups": reports,
        "clinical_promotion_allowed": False,
        "causal_claim_allowed": False,
        "online_weight_update_allowed": False,
    }
