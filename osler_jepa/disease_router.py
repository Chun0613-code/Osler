"""Disease-agnostic target-router contracts.

The DKA router result showed the stable shape: each target should route to the
source that passed its own held-out gate, while unsupported targets fall back to
persistence. This module makes that pattern reusable for the next disease
modules without importing DKA-specific state names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SourceMetric:
    rows: int
    stays: int
    mae: float | None
    point_delta_vs_persistence: float | None
    ci_low: float | None
    ci_high: float | None

    @property
    def significant_win(self) -> bool:
        return (
            self.mae is not None
            and self.point_delta_vs_persistence is not None
            and self.point_delta_vs_persistence < 0
            and self.ci_high is not None
            and self.ci_high < 0
        )


@dataclass(frozen=True)
class DiseaseModuleSpec:
    disease: str
    cohort_definition: str
    targets: tuple[str, ...]
    action_channels: tuple[str, ...]
    candidate_sources: tuple[str, ...] = (
        "persistence",
        "mechanistic_world_model",
        "real_fit_residual",
        "pretrained_presentation_prior",
    )
    safety_boundary: str = (
        "research factual advisory only until patient-held-out, symbolic, "
        "causal-readiness, and clinical-review gates pass"
    )
    notes: tuple[str, ...] = field(default_factory=tuple)


def choose_target_routes(
    discovery_metrics: Mapping[str, Mapping[str, SourceMetric]],
    *,
    min_rows: int,
    min_stays: int,
) -> dict[str, dict[str, object]]:
    """Choose a fail-closed route for each target.

    Only non-persistence sources with enough support and a significant
    stay-level win over persistence can be selected.
    """

    routes: dict[str, dict[str, object]] = {}
    for target, source_metrics in discovery_metrics.items():
        candidates = {}
        details = {}
        for source, metric in source_metrics.items():
            details[source] = asdict(metric) | {
                "significant_win": metric.significant_win,
            }
            if source == "persistence":
                continue
            if metric.rows < min_rows or metric.stays < min_stays:
                continue
            if not metric.significant_win:
                continue
            candidates[source] = metric.mae
        if candidates:
            selected = min(candidates, key=lambda key: candidates[key])
            reason = "lowest_mae_among_significant_supported_sources"
        else:
            selected = "persistence"
            reason = "no_supported_source_significantly_beat_persistence"
        routes[target] = {
            "selected_source": selected,
            "reason": reason,
            "method_stats": details,
        }
    return routes


def disease_expansion_templates() -> tuple[DiseaseModuleSpec, ...]:
    """Return chapter-B starter modules in the validated router style."""

    return (
        DiseaseModuleSpec(
            disease="sepsis",
            cohort_definition=(
                "suspected infection plus organ dysfunction or SepsisLabel, "
                "with hourly vitals/labs and treatment timing"
            ),
            targets=(
                "MAP",
                "lactate",
                "vasopressor_requirement",
                "creatinine",
                "urine_output",
                "oxygenation",
            ),
            action_channels=(
                "fluids",
                "vasopressors",
                "antibiotics",
                "oxygen_or_ventilation",
                "renal_replacement_context",
            ),
            notes=(
                "Start as factual router using PhysioNet/eICU/MIMIC trajectories.",
                "Do not claim antibiotic or fluid causal effects from observational EHR.",
            ),
        ),
        DiseaseModuleSpec(
            disease="acute_kidney_injury",
            cohort_definition=(
                "KDIGO-like creatinine or urine-output change with medication "
                "and hemodynamic context"
            ),
            targets=(
                "creatinine",
                "urine_output",
                "potassium",
                "bicarbonate",
                "MAP",
                "fluid_balance",
            ),
            action_channels=(
                "fluids",
                "vasopressors",
                "diuretics",
                "nephrotoxin_exposure",
                "renal_replacement_therapy",
            ),
            notes=(
                "Use belief states for renal reserve and sodium-water balance.",
                "Persistence will be strong for creatinine; router must fall back.",
            ),
        ),
        DiseaseModuleSpec(
            disease="respiratory_failure",
            cohort_definition=(
                "ARDS / ventilator-coded respiratory failure or severe measured "
                "hypoxemia with respiratory vitals, oxygenation, acid-base, "
                "and observed respiratory treatment evidence"
            ),
            targets=(
                "respiratory_rate",
                "oxygen_saturation",
                "heart_rate",
                "MAP",
                "bicarbonate",
                "pH",
            ),
            action_channels=(
                "oxygen_or_ventilation",
                "bronchodilators",
                "systemic_steroids",
                "antibiotics",
                "fluids",
                "vasopressors",
            ),
            notes=(
                "Start as factual router; do not claim oxygen or ventilation causal effects from observational EHR.",
                "Asthma-specific peak-flow/work-of-breathing targets remain a future non-ICU data problem.",
            ),
        ),
        DiseaseModuleSpec(
            disease="cardiovascular_instability",
            cohort_definition=(
                "cardiogenic shock / heart failure / acute coronary / arrhythmia "
                "diagnosis support with hemodynamic, lactate, renal, and electrolyte targets"
            ),
            targets=(
                "MAP",
                "heart_rate",
                "lactate",
                "creatinine",
                "urine_output",
                "potassium",
                "bicarbonate",
            ),
            action_channels=(
                "vasopressors",
                "inotropes",
                "fluids",
                "diuretics",
                "antiarrhythmics",
                "ventilation",
            ),
            notes=(
                "Use as hemodynamic factual router only.",
                "Vasopressor/inotrope treatment effects stay Chapter-A closed.",
            ),
        ),
        DiseaseModuleSpec(
            disease="acute_neuro",
            cohort_definition=(
                "stroke / intracranial hemorrhage / seizure / coma / encephalopathy "
                "diagnosis support with physiologic proxy targets"
            ),
            targets=(
                "MAP",
                "heart_rate",
                "respiratory_rate",
                "oxygen_saturation",
                "glucose",
                "sodium",
                "pH",
            ),
            action_channels=(
                "ventilation",
                "vasopressors",
                "antiepileptics",
                "osmotherapy",
                "fluids",
            ),
            notes=(
                "ICU EHR does not reliably expose detailed neurologic exam targets.",
                "Treat this as a physiologic proxy router, not a neurologic outcome model.",
            ),
        ),
        DiseaseModuleSpec(
            disease="hepatic_failure",
            cohort_definition=(
                "hepatic failure / cirrhosis / hepatic encephalopathy diagnosis support "
                "with bilirubin, platelet, lactate, renal, and perfusion targets"
            ),
            targets=(
                "bilirubin",
                "direct_bilirubin",
                "platelets",
                "lactate",
                "bicarbonate",
                "creatinine",
                "MAP",
            ),
            action_channels=(
                "fluids",
                "vasopressors",
                "antibiotics",
                "renal_replacement",
                "hepatic_encephalopathy_treatment",
            ),
            notes=(
                "Coagulation and encephalopathy labels are incomplete in shared ICU state maps.",
                "Use as bounded factual proxy router until richer hepatic data are mapped.",
            ),
        ),
        DiseaseModuleSpec(
            disease="coagulopathy_heme",
            cohort_definition=(
                "coagulopathy / thrombocytopenia / anemia / bleeding diagnosis support "
                "with platelet, inflammatory, perfusion, acid-base, and renal targets"
            ),
            targets=(
                "platelets",
                "WBC",
                "lactate",
                "MAP",
                "bicarbonate",
                "pH",
                "creatinine",
            ),
            action_channels=(
                "transfusion",
                "anticoagulants",
                "antiplatelets",
                "fluids",
                "vasopressors",
                "antibiotics",
                "nephrotoxin_context",
            ),
            notes=(
                "Hemoglobin and INR are not yet first-class shared state variables.",
                "Transfusion and anticoagulation effects stay Chapter-A closed.",
            ),
        ),
    )


def disease_expansion_blueprint() -> dict[str, object]:
    return {
        "architecture": "per-disease, per-target gated factual router",
        "do_not_mix_latent_spaces_initially": True,
        "promotion_sequence": [
            "define disease cohort and target/action contract",
            "build observed-treatment transitions with masks and ages",
            "train or evaluate candidate sources",
            "select sources on discovery patients only",
            "evaluate on held-out patients and time/care-unit splits",
            "keep causal claims closed unless causal_readiness gates pass",
        ],
        "templates": [asdict(spec) for spec in disease_expansion_templates()],
    }
