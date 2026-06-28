"""Fail-closed causal evidence gates for Osler-JEPA.

Observational EHR target-trial diagnostics can show why a causal claim is not
ready. This module describes what would be required to open the next causal
chapter: randomized, instrumental-variable, or front-door evidence with explicit
assignment, timing, outcomes, and audit metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


SUPPORTED_IDENTIFICATION_DESIGNS = (
    "randomized_trial",
    "instrumental_variable",
    "front_door",
)


@dataclass(frozen=True)
class CausalEvidenceSpec:
    name: str
    disease: str
    identification_design: str
    treatment_strategy: str
    comparator_strategy: str
    outcome: str
    time_zero: str
    assignment_column: str
    treatment_column: str
    outcome_column: str
    patient_id_column: str
    baseline_covariates: tuple[str, ...] = field(default_factory=tuple)
    source_registry: tuple[str, ...] = field(default_factory=tuple)

    def required_columns(self) -> set[str]:
        return {
            self.assignment_column,
            self.treatment_column,
            self.outcome_column,
            self.patient_id_column,
            *self.baseline_covariates,
        }


def _smd(values: np.ndarray, assignment: np.ndarray) -> float:
    left = values[assignment == 0]
    right = values[assignment == 1]
    if len(left) == 0 or len(right) == 0:
        return np.nan
    pooled = np.sqrt((np.nanvar(left) + np.nanvar(right)) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-8:
        pooled = 1.0
    return float((np.nanmean(right) - np.nanmean(left)) / pooled)


def evaluate_causal_evidence_contract(
    frame: pd.DataFrame | None,
    spec: CausalEvidenceSpec,
    *,
    min_subjects: int = 100,
    max_assignment_imbalance: float = 0.25,
    max_outcome_missing_fraction: float = 0.20,
    max_abs_baseline_smd: float = 0.10,
    min_adherence_fraction: float = 0.80,
) -> dict:
    """Evaluate whether an external evidence table can support causal claims.

    A passing result allows a research causal-effect estimate to be produced.
    It never grants clinical or runtime treatment authority.
    """

    failures: list[str] = []
    warnings: list[str] = []
    if spec.identification_design not in SUPPORTED_IDENTIFICATION_DESIGNS:
        failures.append("unsupported_identification_design")

    if frame is None:
        failures.append("evidence_table_not_present")
        return {
            "spec": asdict(spec),
            "support": {
                "rows": 0,
                "subjects": 0,
                "assignment_arms": [],
                "outcome_missing_fraction": None,
                "adherence_fraction": None,
                "max_abs_baseline_smd": None,
            },
            "readiness": {
                "passes": False,
                "failures": failures,
                "warnings": warnings,
            },
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        }

    missing = sorted(spec.required_columns() - set(frame.columns))
    if missing:
        failures.append("missing_required_columns")

    if missing:
        assignment = np.asarray([], dtype=int)
        support = {
            "rows": int(len(frame)),
            "subjects": 0,
            "assignment_arms": [],
            "outcome_missing_fraction": None,
            "adherence_fraction": None,
            "max_abs_baseline_smd": None,
            "missing_columns": missing,
        }
    else:
        selected = frame[list(spec.required_columns())].copy()
        subjects = int(selected[spec.patient_id_column].nunique())
        assignment_raw = selected[spec.assignment_column].dropna().unique().tolist()
        assignment_values = sorted(assignment_raw)
        assignment_map = {value: index for index, value in enumerate(assignment_values)}
        assignment = selected[spec.assignment_column].map(assignment_map).to_numpy()
        if len(assignment_values) != 2:
            failures.append("assignment_is_not_binary_two_arm")
        rows = int(len(selected))
        if subjects < min_subjects:
            failures.append("too_few_subjects")
        if rows == 0:
            failures.append("empty_evidence_table")
        arm_fraction = (
            float(np.nanmean(assignment == 1)) if len(assignment) else np.nan
        )
        if np.isfinite(arm_fraction):
            imbalance = abs(arm_fraction - 0.5)
            if imbalance > max_assignment_imbalance:
                failures.append("assignment_imbalance_too_large")
        outcome_missing = float(selected[spec.outcome_column].isna().mean())
        if outcome_missing > max_outcome_missing_fraction:
            failures.append("outcome_missingness_too_high")
        treatment = selected[spec.treatment_column].fillna(0).to_numpy()
        expected_treated = assignment == 1
        adherence = float(np.mean((treatment > 0) == expected_treated))
        if adherence < min_adherence_fraction:
            failures.append("treatment_adherence_too_low")
        smds = []
        for covariate in spec.baseline_covariates:
            values = selected[covariate].to_numpy(dtype=float)
            smds.append(abs(_smd(values, assignment)))
        max_smd = float(np.nanmax(smds)) if smds else 0.0
        if max_smd > max_abs_baseline_smd:
            failures.append("baseline_covariate_imbalance")
        if spec.identification_design != "randomized_trial":
            warnings.append(
                "non-randomized identification requires design-specific external review"
            )
        support = {
            "rows": rows,
            "subjects": subjects,
            "assignment_arms": [str(value) for value in assignment_values],
            "assignment_fraction_arm_1": (
                round(arm_fraction, 6) if np.isfinite(arm_fraction) else None
            ),
            "outcome_missing_fraction": round(outcome_missing, 6),
            "adherence_fraction": round(adherence, 6),
            "max_abs_baseline_smd": round(max_smd, 6),
        }

    passes = not failures
    return {
        "spec": asdict(spec),
        "support": support,
        "readiness": {
            "passes": passes,
            "failures": failures,
            "warnings": warnings,
            "gate": {
                "min_subjects": int(min_subjects),
                "max_assignment_imbalance": float(max_assignment_imbalance),
                "max_outcome_missing_fraction": float(max_outcome_missing_fraction),
                "max_abs_baseline_smd": float(max_abs_baseline_smd),
                "min_adherence_fraction": float(min_adherence_fraction),
            },
        },
        "causal_claim_allowed": bool(passes),
        "clinical_claim_allowed": False,
        "runtime_decision_authority": False,
    }


def causal_source_registry() -> list[Mapping[str, object]]:
    """Return the fail-closed source registry for chapter A."""

    return [
        {
            "source": "BioLINCC",
            "role": "candidate NHLBI randomized-trial data source",
            "status": "not_in_workspace",
            "allowed_use_when_present": "research causal-effect estimation only",
        },
        {
            "source": "Vivli",
            "role": "candidate randomized-trial data source",
            "status": "not_in_workspace",
            "allowed_use_when_present": "research causal-effect estimation only",
        },
        {
            "source": "YODA",
            "role": "candidate randomized-trial data source",
            "status": "not_in_workspace",
            "allowed_use_when_present": "research causal-effect estimation only",
        },
        {
            "source": "observational EHR",
            "role": "factual forecasting and confounding diagnostics",
            "status": "available",
            "allowed_use_when_present": "no causal claim without external design",
        },
    ]


def default_causal_specs() -> tuple[CausalEvidenceSpec, ...]:
    return (
        CausalEvidenceSpec(
            name="dka_insulin_strategy_rct",
            disease="DKA",
            identification_design="randomized_trial",
            treatment_strategy="protocolized insulin strategy",
            comparator_strategy="alternative protocolized insulin strategy",
            outcome="six-hour glucose or anion-gap change",
            time_zero="randomization",
            assignment_column="assigned_arm",
            treatment_column="received_protocol_treatment",
            outcome_column="outcome_delta",
            patient_id_column="participant_key",
            baseline_covariates=("baseline_glucose", "baseline_anion_gap"),
            source_registry=("BioLINCC", "Vivli", "YODA", "trial sponsor repository"),
        ),
        CausalEvidenceSpec(
            name="sepsis_fluid_strategy_rct",
            disease="sepsis",
            identification_design="randomized_trial",
            treatment_strategy="protocolized early fluid strategy",
            comparator_strategy="restricted or alternative protocolized fluid strategy",
            outcome="twenty-four-hour MAP/lactate response or vasopressor-free days",
            time_zero="randomization",
            assignment_column="assigned_arm",
            treatment_column="received_protocol_treatment",
            outcome_column="outcome_delta",
            patient_id_column="participant_key",
            baseline_covariates=("baseline_map", "baseline_lactate", "baseline_vasopressor"),
            source_registry=("BioLINCC", "Vivli", "YODA", "trial sponsor repository"),
        ),
        CausalEvidenceSpec(
            name="aki_rrt_timing_strategy_rct",
            disease="AKI",
            identification_design="randomized_trial",
            treatment_strategy="early renal replacement or renal-support strategy",
            comparator_strategy="delayed or standard renal-support strategy",
            outcome="twenty-four-to-forty-eight-hour creatinine/BUN/urine trajectory or RRT-free survival",
            time_zero="randomization",
            assignment_column="assigned_arm",
            treatment_column="received_protocol_treatment",
            outcome_column="outcome_delta",
            patient_id_column="participant_key",
            baseline_covariates=("baseline_creatinine", "baseline_bun", "baseline_urine_output"),
            source_registry=("BioLINCC", "Vivli", "YODA", "trial sponsor repository"),
        ),
        CausalEvidenceSpec(
            name="respiratory_oxygen_ventilation_strategy_rct",
            disease="respiratory_failure",
            identification_design="randomized_trial",
            treatment_strategy="protocolized oxygenation or ventilation strategy",
            comparator_strategy="alternative oxygenation or ventilation strategy",
            outcome="six-to-twenty-four-hour oxygenation response or ventilator-free days",
            time_zero="randomization",
            assignment_column="assigned_arm",
            treatment_column="received_protocol_treatment",
            outcome_column="outcome_delta",
            patient_id_column="participant_key",
            baseline_covariates=("baseline_o2sat", "baseline_respiratory_rate", "baseline_ventilation"),
            source_registry=("BioLINCC", "Vivli", "YODA", "trial sponsor repository"),
        ),
    )
