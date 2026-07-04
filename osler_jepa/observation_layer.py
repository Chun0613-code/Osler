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


PREDICTION_HORIZONS_HOURS = (1, 3, 6, 12, 24, 48)

DERIVED_COMPLETION_RULES: dict[str, dict[str, object]] = {
    "map": {
        "source": "derived_formula_completion",
        "inputs": ("sbp", "dbp"),
        "formula": "dbp + (sbp - dbp) / 3",
        "confidence": "high_when_inputs_observed",
    },
    "anion_gap": {
        "source": "derived_formula_completion",
        "inputs": ("sodium", "chloride", "bicarbonate"),
        "formula": "sodium - chloride - bicarbonate",
        "confidence": "high_when_inputs_observed",
    },
    "serum_osmolality": {
        "source": "derived_formula_completion",
        "inputs": ("sodium", "glucose", "bun"),
        "formula": "2*sodium + glucose/18 + bun/2.8",
        "confidence": "moderate_formula_estimate",
    },
}

SAME_GROUP_CALIBRATED_COMPLETION_GROUPS: dict[str, dict[str, object]] = {
    "red_cell_indices": {
        "source": "same_group_calibrated_completion",
        "targets": ("hemoglobin", "hematocrit", "rbc"),
        "rationale": "near-linear red-cell measurement family; not cross-system inference",
    },
    "body_size": {
        "source": "same_group_calibrated_completion",
        "targets": ("bmi", "weight", "waist"),
        "rationale": "body-size measurement family with deterministic or near-deterministic relationships",
    },
}

VALIDATED_NOWCAST_MODULE_TARGETS: dict[str, tuple[str, ...]] = {
    "sepsis": (
        "bicarbonate",
        "bilirubin_direct",
        "bun",
        "creatinine",
        "heart_rate",
        "map",
        "ph",
        "platelets",
        "potassium",
        "respiratory_rate",
        "sodium",
        "temperature",
        "vasopressor_requirement",
        "wbc",
    ),
    "aki": (
        "bicarbonate",
        "bun",
        "creatinine",
        "heart_rate",
        "map",
        "potassium",
        "respiratory_rate",
        "sodium",
        "temperature",
    ),
    "respiratory": ("bicarbonate", "bun", "creatinine", "map", "potassium", "respiratory_rate"),
    "integumentary_skin_wound": ("albumin", "bun", "hematocrit", "hemoglobin", "potassium"),
    "toxic_metabolic": ("anion_gap", "bicarbonate", "bun", "chloride", "sodium"),
    "electrolyte_acid_base": (
        "anion_gap",
        "bicarbonate",
        "bun",
        "calcium",
        "chloride",
        "creatinine",
        "phosphate",
        "potassium",
        "sodium",
    ),
    "endocrine_stress": ("anion_gap", "bicarbonate", "bun", "potassium"),
    "gi_pancreatic_nutrition": ("albumin", "bicarbonate", "calcium", "map", "total_protein"),
    "cardiac_injury": ("bun", "potassium"),
    "hepatic_failure": ("bicarbonate", "bilirubin_direct", "bun", "creatinine", "ph", "platelets", "potassium", "wbc"),
    "coagulopathy_heme": ("bun", "hematocrit", "hemoglobin", "potassium"),
}

VALIDATED_FULL_VARIABLE_FORECAST_6H_CELLS: dict[str, tuple[str, ...]] = {
    "sepsis": (
        "bicarbonate",
        "glucose",
        "heart_rate",
        "lactate",
        "map",
        "o2sat",
        "ph",
        "platelets",
        "potassium",
        "respiratory_rate",
        "sodium",
        "temperature",
        "wbc",
    ),
    "aki": (
        "bicarbonate",
        "glucose",
        "heart_rate",
        "lactate",
        "map",
        "o2sat",
        "potassium",
        "respiratory_rate",
        "sodium",
        "temperature",
        "wbc",
    ),
    "respiratory": ("glucose", "heart_rate", "map", "o2sat", "potassium", "respiratory_rate"),
    "integumentary_skin_wound": ("map", "potassium"),
    "toxic_metabolic": ("heart_rate", "map", "potassium", "respiratory_rate"),
    "electrolyte_acid_base": ("glucose", "potassium"),
    "endocrine_stress": ("glucose", "map"),
    "gi_pancreatic_nutrition": ("glucose", "map"),
    "cardiac_injury": ("map", "o2sat", "respiratory_rate"),
    "musculoskeletal_rhabdo": ("potassium",),
    "cardiovascular_instability": ("heart_rate", "map", "o2sat", "potassium", "respiratory_rate"),
    "acute_neuro": ("map", "potassium", "respiratory_rate"),
    "hepatic_failure": ("map", "potassium"),
    "coagulopathy_heme": ("hematocrit", "hemoglobin", "map"),
}

VALIDATED_FULL_VARIABLE_INTERVAL_6H_CELLS: dict[str, tuple[str, ...]] = {
    "sepsis": (
        "bicarbonate",
        "glucose",
        "heart_rate",
        "lactate",
        "map",
        "o2sat",
        "ph",
        "platelets",
        "potassium",
        "respiratory_rate",
        "sodium",
        "temperature",
        "wbc",
    ),
    "aki": (
        "bicarbonate",
        "glucose",
        "heart_rate",
        "lactate",
        "map",
        "o2sat",
        "potassium",
        "respiratory_rate",
        "sodium",
        "temperature",
        "wbc",
    ),
    "respiratory": ("glucose", "heart_rate", "map", "o2sat", "potassium", "respiratory_rate"),
    "integumentary_skin_wound": ("map",),
    "toxic_metabolic": ("heart_rate", "map", "potassium", "respiratory_rate"),
    "electrolyte_acid_base": ("glucose", "potassium"),
    "endocrine_stress": ("map",),
    "gi_pancreatic_nutrition": ("glucose", "map"),
    "cardiac_injury": ("map", "respiratory_rate"),
    "cardiovascular_instability": ("heart_rate", "map", "respiratory_rate"),
    "acute_neuro": ("map", "respiratory_rate"),
    "hepatic_failure": ("map", "potassium"),
    "coagulopathy_heme": ("hemoglobin", "map"),
}

MIMICIV_EXTERNAL_NOWCAST_TARGETS = (
    "albumin",
    "anion_gap",
    "bicarbonate",
    "bun",
    "calcium",
    "chloride",
    "fibrinogen",
    "heart_rate",
    "hematocrit",
    "hemoglobin",
    "ionized_calcium",
    "magnesium",
    "map",
    "minute_volume",
    "paco2",
    "pao2",
    "ph",
    "phosphate",
    "platelets",
    "potassium",
    "respiratory_rate",
    "serum_osmolality",
    "sodium",
)

MIMICIV_EXTERNAL_FORECAST_6H_TARGETS = (
    "anion_gap",
    "bicarbonate",
    "calcium",
    "chloride",
    "heart_rate",
    "hemoglobin",
    "magnesium",
    "map",
    "minute_volume",
    "paco2",
    "pao2",
    "ph",
    "phosphate",
    "potassium",
    "respiratory_rate",
    "temperature",
    "urine_output",
    "wbc",
)

MIMICIV_EXTERNAL_INTERVAL_6H_TARGETS = (
    "anion_gap",
    "bicarbonate",
    "calcium",
    "heart_rate",
    "magnesium",
    "map",
    "minute_volume",
    "paco2",
    "pao2",
    "ph",
    "phosphate",
    "potassium",
    "respiratory_rate",
    "temperature",
)

MIMICIV_RADIOLOGY_NOTE_TARGETS = (
    "rad_pulmonary_edema",
    "rad_pleural_effusion",
    "rad_consolidation",
    "rad_atelectasis",
    "rad_pneumothorax",
    "rad_cardiomegaly",
)

MIMICIV_RADIOLOGY_NOTE_VALIDATED_NOWCAST_TARGETS: tuple[str, ...] = ()
MIMICIV_RADIOLOGY_NOTE_VALIDATED_FORECAST_TARGETS: tuple[str, ...] = ()
MIMICIV_RADIOLOGY_NOTE_VALIDATED_INTERVAL_TARGETS: tuple[str, ...] = ()

EICU_NEURO_NOTE_TARGETS = (
    "neuro_gcs",
    "neuro_sedation_score",
    "neuro_delirium_present",
    "neuro_mental_abnormal",
    "neuro_pupils_abnormal",
    "neuro_motor_abnormal",
)

EICU_NEURO_NOTE_VALIDATED_NOWCAST_TARGETS: tuple[str, ...] = ()
EICU_NEURO_NOTE_VALIDATED_FORECAST_TARGETS: tuple[str, ...] = ()
EICU_NEURO_NOTE_VALIDATED_INTERVAL_TARGETS: tuple[str, ...] = ()

MIMICIV_ED_FORECAST_TARGETS: dict[int, tuple[str, ...]] = {
    1: ("map", "sbp"),
    3: ("dbp", "map", "respiratory_rate", "sbp", "temperature"),
    6: ("dbp", "heart_rate", "map", "respiratory_rate", "sbp", "temperature"),
}

MIMICIV_ED_NOWCAST_TARGETS = ("acuity", "dbp", "map", "pain", "sbp")

VALIDATED_INTERMEDIATE_MOVE_CELLS: dict[str, dict[int, tuple[str, ...]]] = {
    "acute_neuro": {
        1: ("map", "respiratory_rate"),
        3: ("map", "respiratory_rate"),
        12: ("heart_rate", "map", "o2sat", "respiratory_rate"),
    },
    "cardiac_injury": {
        1: ("map",),
        3: ("map",),
        12: ("heart_rate", "map", "potassium"),
    },
    "cardiovascular_instability": {
        1: ("map",),
        3: ("heart_rate", "map"),
        12: ("heart_rate", "map"),
    },
    "coagulopathy_heme": {
        3: ("hemoglobin", "map"),
        12: ("map",),
    },
    "electrolyte_acid_base": {
        1: ("map",),
        3: ("map", "potassium"),
        12: ("anion_gap", "map", "potassium", "sodium"),
    },
    "endocrine_stress": {
        3: ("glucose", "map"),
        12: ("glucose", "map"),
    },
    "gi_pancreatic_nutrition": {
        3: ("glucose", "map"),
        12: ("glucose", "map"),
    },
    "hepatic_failure": {
        3: ("map",),
        12: ("map",),
    },
    "immune_inflammatory": {
        12: ("map",),
    },
    "integumentary_skin_wound": {
        1: ("map",),
        3: ("glucose", "hematocrit", "map"),
        12: ("glucose", "heart_rate", "map"),
    },
    "musculoskeletal_rhabdo": {
        1: ("map",),
    },
    "respiratory": {
        1: ("map", "o2sat", "respiratory_rate"),
        3: ("bicarbonate", "heart_rate", "map", "o2sat", "paco2", "ph", "respiratory_rate"),
        12: ("bicarbonate", "heart_rate", "map", "o2sat", "paco2", "ph", "respiratory_rate"),
    },
    "toxic_metabolic": {
        1: ("map",),
        3: ("anion_gap", "heart_rate", "map", "o2sat", "potassium", "respiratory_rate"),
        12: ("anion_gap", "bicarbonate", "glucose", "heart_rate", "map", "o2sat", "potassium", "respiratory_rate"),
    },
}

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


@dataclass(frozen=True)
class TrajectoryGridCell:
    """One target-horizon cell in the factual rollout map."""

    target: str
    horizon_hours: int
    status: str
    source: str
    context: str
    can_move: bool
    reason: str
    interval_status: str = "needs_calibration_audit"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WholeBodyStateForecastCell:
    """One current-state or future-state cell in the unified output object."""

    target: str
    module: str
    horizon_hours: int
    time_axis: str
    status: str
    source: str
    point_estimate: float | None
    lower: float | None
    upper: float | None
    interval_level: float | None
    interval_status: str
    can_estimate: bool
    can_move: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
            name="whole_body_same_time_nowcasting",
            status="validated",
            scope="same_time_observation_completion",
            systems=("whole_body",),
            targets=(
                "map",
                "creatinine",
                "heart_rate",
                "ph",
                "platelets",
                "respiratory_rate",
                "vasopressor_requirement",
                "potassium",
                "bicarbonate",
                "bun",
                "sodium",
                "temperature",
                "wbc",
                "hemoglobin",
                "hematocrit",
                "albumin",
                "anion_gap",
                "chloride",
                "calcium",
                "phosphate",
                "total_protein",
                "bilirubin_direct",
            ),
            horizon_hours=(0,),
            evidence="EICU_FULL_VARIABLE_COVERAGE_FINDINGS.md; full numeric-variable sweep validates 71 module-target same-time nowcasts",
            allowed_uses=("shadow_observation", "same_time_state_completion", "capability_reporting"),
        ),
        ObservationCapability(
            name="eicu_full_variable_6h_forecast_coverage",
            status="validated",
            scope="full_numeric_variable_forecast_sweep",
            systems=("whole_body",),
            targets=("59_module_target_forecast_cells", "51_calibrated_interval_cells"),
            horizon_hours=(6,),
            evidence="EICU_FULL_VARIABLE_COVERAGE_FINDINGS.md; 242 eligible numeric targets swept through forecast and interval gates",
        ),
        ObservationCapability(
            name="mimiciv_cross_database_observation_validation",
            status="validated",
            scope="external_database_observation_validation",
            systems=("whole_body",),
            targets=(
                "23_target_nowcast_cells",
                "18_target_6h_forecast_cells",
                "14_target_calibrated_interval_cells",
            ),
            horizon_hours=(0, 6),
            evidence=(
                "MIMICIV_CROSS_DATABASE_COVERAGE_FINDINGS.md; full MIMIC-IV "
                "v3.1 target-level audit passes patient, careunit, and time "
                "held-out gates for a bounded target subset"
            ),
        ),
        ObservationCapability(
            name="mimiciv_ed_scene_observation_validation",
            status="validated",
            scope="ed_scene_observation_validation",
            systems=("emergency_department", "pre_icu_acuity"),
            targets=(
                "5_target_nowcast_cells",
                "1h_2_target_forecast_interval_cells",
                "3h_5_target_forecast_interval_cells",
                "6h_6_target_forecast_interval_cells",
            ),
            horizon_hours=(0, 1, 3, 6),
            evidence=(
                "MIMICIV_ED_OBSERVATION_FINDINGS.md; ED-scene table-based "
                "vitals audit validates dense pre-ICU physiology across "
                "patient, arrival-transport, and chronological held-out gates"
            ),
        ),
        ObservationCapability(
            name="mimiciv_ed_to_icu_baseline_candidate",
            status="candidate_only",
            scope="ed_to_early_icu_baseline",
            systems=("emergency_department", "intensive_care_unit", "transition_of_care"),
            targets=(
                "0_validated_ed_increment_targets",
                "heart_rate_map_3h_6h_near_miss",
            ),
            horizon_hours=(1, 3, 6),
            evidence=(
                "MIMICIV_ED_TO_ICU_BASELINE_FINDINGS.md; prior ED trajectory "
                "does not robustly improve early ICU prediction beyond ICU "
                "state, although heart_rate/MAP show candidate-only near-miss "
                "signal at 3h-6h"
            ),
            allowed_uses=("capability_reporting",),
        ),
        ObservationCapability(
            name="nhanes_healthy_population_nowcast_validation",
            status="validated",
            scope="external_population_nowcast_validation",
            systems=("healthy_population", "non_icu", "whole_body"),
            targets=(
                "29_cross_sectional_nowcast_targets",
                "glucose_fallback",
                "alk_phos_fallback",
                "hs_crp_fallback",
            ),
            horizon_hours=(0,),
            evidence=(
                "NHANES_NOWCAST_FINDINGS.md; NHANES 2017-2018 public "
                "cross-sectional audit validates 29/32 targets with "
                "participant-heldout median, capacity-matched placebo, and "
                "expanded sibling-variable leakage guards; "
                "no forecast, treatment, or causal claim is supported"
            ),
            allowed_uses=(
                "shadow_observation",
                "same_time_state_completion",
                "capability_reporting",
            ),
        ),
        ObservationCapability(
            name="mimiciv_radiology_note_structured_observation_candidate",
            status="candidate_only",
            scope="note_backed_measurement_depth",
            systems=("radiology_notes", "respiratory", "cardiovascular"),
            targets=MIMICIV_RADIOLOGY_NOTE_TARGETS,
            horizon_hours=(0, 6),
            evidence=(
                "MIMICIV_RADIOLOGY_NOTE_OBSERVATION_FINDINGS.md; "
                "317,371 timestamped chest radiology reports extracted into "
                "six structured findings, but 0/6 nowcast, 0/6 forecast, and "
                "0/6 interval targets pass robust held-out gates"
            ),
            allowed_uses=(
                "shadow_observation",
                "structured_note_observation",
                "capability_reporting",
            ),
        ),
        ObservationCapability(
            name="eicu_neuro_note_structured_observation_candidate",
            status="candidate_only",
            scope="note_backed_measurement_depth",
            systems=("neurologic", "nursing_flowsheet", "progress_note_structured_exam"),
            targets=EICU_NEURO_NOTE_TARGETS,
            horizon_hours=(0, 6),
            evidence=(
                "EICU_NEURO_NOTE_OBSERVATION_FINDINGS.md and "
                "EICU_NEURO_NOTE_ALL_ICU_FINDINGS.md; bounded and all-ICU "
                "audits extract timestamped GCS/delirium evidence, but 0 "
                "note-derived neuro targets pass robust nowcast, forecast, or "
                "interval gates even after scaling to 100,862 subjects"
            ),
            allowed_uses=(
                "shadow_observation",
                "structured_note_observation",
                "capability_reporting",
            ),
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
            name="cardiovascular_heart_rate_belief_state",
            status="validated",
            scope="predict_update_belief_state",
            systems=("cardiovascular_perfusion",),
            targets=("heart_rate", "patient_specific_perfusion_shock_state"),
            horizon_hours=(6,),
            evidence=(
                "CARDIOVASCULAR_BELIEF_FINDINGS.md; full eICU cardiovascular "
                "cohort validates heart-rate prediction with predict-update "
                "perfusion belief across 7/7 patient splits and "
                "hospital-heldout, beating baseline and capacity-matched "
                "placebo; no direct hidden-state accuracy claim"
            ),
        ),
        ObservationCapability(
            name="cardiovascular_belief_remaining_targets_candidate",
            status="candidate_only",
            scope="predict_update_belief_state",
            systems=("cardiovascular_perfusion",),
            targets=("map", "lactate", "o2sat", "respiratory_rate"),
            horizon_hours=(6,),
            evidence=(
                "CARDIOVASCULAR_BELIEF_FINDINGS.md; bounded cohort failed "
                "robust downstream observable gates for MAP, lactate, O2 "
                "saturation, and respiratory rate; only heart_rate is "
                "validated on the full cohort"
            ),
            allowed_uses=("capability_reporting",),
        ),
        ObservationCapability(
            name="electrolyte_acid_base_belief_state",
            status="validated",
            scope="predict_update_belief_state",
            systems=("electrolyte_acid_base", "renal_urinary", "osmotic_fluid"),
            targets=(
                "potassium",
                "bicarbonate",
                "anion_gap",
                "creatinine",
                "patient_specific_electrolyte_acid_base_state",
            ),
            horizon_hours=(6,),
            evidence=(
                "ELECTROLYTE_BELIEF_FINDINGS.md; full eICU electrolyte "
                "cohort validates predict-update belief for potassium, "
                "bicarbonate, anion_gap, and creatinine across 7/7 patient "
                "splits and hospital-heldout, beating baseline and "
                "capacity-matched placebo; no direct hidden-state accuracy "
                "claim"
            ),
        ),
        ObservationCapability(
            name="electrolyte_belief_remaining_targets_candidate",
            status="candidate_only",
            scope="predict_update_belief_state",
            systems=("electrolyte_acid_base", "osmotic_fluid"),
            targets=(
                "sodium_near_miss",
                "chloride_near_miss",
                "magnesium_hospital_fail",
                "calcium",
                "phosphate",
                "ionized_calcium",
                "serum_osmolality",
                "map",
            ),
            horizon_hours=(6,),
            evidence=(
                "ELECTROLYTE_BELIEF_FINDINGS.md; sodium/chloride are "
                "near-miss, magnesium fails hospital-heldout despite 7/7 "
                "patient splits, and remaining sparse/deep targets do not "
                "pass the robust downstream observable gate"
            ),
            allowed_uses=("capability_reporting",),
        ),
        ObservationCapability(
            name="respiratory_gas_exchange_belief_state",
            status="validated",
            scope="predict_update_belief_state",
            systems=("respiratory", "gas_exchange", "acid_base"),
            targets=(
                "o2sat",
                "respiratory_rate",
                "heart_rate",
                "bicarbonate",
                "patient_specific_respiratory_gas_exchange_state",
            ),
            horizon_hours=(6,),
            evidence=(
                "RESPIRATORY_BELIEF_FINDINGS.md; full eICU respiratory "
                "cohort validates predict-update belief for O2 saturation, "
                "respiratory rate, heart rate, and bicarbonate across 7/7 "
                "patient splits and hospital-heldout, beating baseline and "
                "capacity-matched placebo; no direct hidden-state accuracy "
                "claim"
            ),
        ),
        ObservationCapability(
            name="respiratory_belief_remaining_targets_candidate",
            status="candidate_only",
            scope="predict_update_belief_state",
            systems=("respiratory", "gas_exchange", "acid_base"),
            targets=(
                "map_near_miss",
                "paco2",
                "ph",
            ),
            horizon_hours=(6,),
            evidence=(
                "RESPIRATORY_BELIEF_FINDINGS.md; MAP is a near-miss that "
                "fails hospital-heldout, while PaCO2 and pH show partial "
                "signal but fail the strict 7/7 patient-split gate"
            ),
            allowed_uses=("capability_reporting",),
        ),
        ObservationCapability(
            name="endocrine_glycemic_stress_belief_state",
            status="validated",
            scope="predict_update_belief_state",
            systems=("endocrine_metabolic", "glycemic_stress", "osmotic_fluid", "acid_base"),
            targets=(
                "glucose",
                "anion_gap",
                "bicarbonate",
                "sodium",
                "potassium",
                "map",
                "patient_specific_endocrine_glycemic_stress_state",
            ),
            horizon_hours=(6,),
            evidence=(
                "ENDOCRINE_BELIEF_FINDINGS.md; full eICU endocrine "
                "cohort validates predict-update belief for glucose, "
                "anion gap, bicarbonate, sodium, potassium, and MAP across "
                "7/7 patient splits and hospital-heldout, beating baseline "
                "and capacity-matched placebo; no direct hidden-state "
                "accuracy claim"
            ),
        ),
        ObservationCapability(
            name="endocrine_belief_remaining_targets_candidate",
            status="candidate_only",
            scope="predict_update_belief_state",
            systems=("endocrine_metabolic", "glycemic_stress", "hormonal_axis"),
            targets=(
                "serum_ketones_sparse",
                "serum_osmolality",
                "temperature",
                "tsh_sparse",
                "free_t4_sparse",
                "cortisol_sparse",
            ),
            horizon_hours=(6,),
            evidence=(
                "ENDOCRINE_BELIEF_FINDINGS.md; sparse endocrine hormone, "
                "ketone, osmolality, and temperature targets do not pass "
                "the strict downstream observable gate"
            ),
            allowed_uses=("capability_reporting",),
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
            "nowcasting_rule": "same-time state completion is allowed only for validated module-target pairs and does not authorize future movement",
            "completion_source_hierarchy": (
                "observed > derived_formula_completion > same_group_calibrated_completion "
                "> same_time_nowcast_ridge > missing"
            ),
            "external_population_nowcast_rule": (
                "cross-sectional healthy-population audits may validate "
                "same-time state completion, but they do not validate future "
                "forecasting or ICU trajectory behavior"
            ),
            "note_observation_rule": (
                "timestamp-valid note findings may be displayed as observed evidence, "
                "but candidate-only note targets may not be imputed or forecast"
            ),
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


def _cell(
    target: str,
    horizon_hours: int,
    status: str,
    source: str,
    context: str,
    can_move: bool,
    reason: str,
    interval_status: str = "needs_calibration_audit",
) -> TrajectoryGridCell:
    return TrajectoryGridCell(
        target=target,
        horizon_hours=horizon_hours,
        status=status,
        source=source,
        context=context,
        can_move=can_move,
        reason=reason,
        interval_status=interval_status,
    )


def trajectory_prediction_grid() -> tuple[TrajectoryGridCell, ...]:
    """Return the current target x horizon factual rollout map.

    Unlisted target-horizon pairs are intentionally unsupported and should use
    the same fallback policy as explicit unsupported cells.  This map is a
    capability contract, not a generated patient trajectory.
    """

    fast_validated = {
        "glucose": ("dka_or_body_surface_router", "validated 6h dense/fast target"),
        "anion_gap": ("dka_realfit_router", "validated 6h DKA anion-gap recovery target"),
        "potassium": ("dka_or_body_surface_router", "validated 6h electrolyte target"),
        "bicarbonate": ("dka_or_respiratory_surface_router", "validated 6h acid-base target in bounded contexts"),
        "ph": ("respiratory_surface_router", "validated 6h respiratory acid-base surface target"),
        "map": ("body_surface_or_sepsis_map_router", "validated 6h perfusion target"),
        "heart_rate": ("body_surface_router", "validated 6h dense vital-sign target"),
        "oxygen_saturation": ("respiratory_or_sepsis_router", "validated 6h oxygenation target"),
        "respiratory_rate": ("respiratory_or_sepsis_router", "validated 6h respiratory target"),
        "lactate": ("sepsis_or_body_surface_router", "validated 6h perfusion/metabolic target"),
    }
    slow_renal_validated = {
        "creatinine": ("aki_long_horizon_router_or_renal_belief_v2", "validated 24-48h slow renal target"),
        "bun": ("aki_long_horizon_router_or_renal_belief_v2", "validated 24-48h slow renal target"),
        "urine_output": ("aki_long_horizon_router", "validated long-horizon AKI router target"),
    }

    cells: list[TrajectoryGridCell] = []
    for target, (source, reason) in fast_validated.items():
        for horizon in PREDICTION_HORIZONS_HOURS:
            if horizon == 6:
                cells.append(
                    _cell(
                        target,
                        horizon,
                        "validated",
                        source,
                        "fast_physiology",
                        True,
                        reason,
                    )
                )
            else:
                cells.append(
                    _cell(
                        target,
                        horizon,
                        "fallback_pending_validation",
                        "persistence",
                        "fast_physiology",
                        False,
                        "no aggregate held-out gate for this target at this horizon",
                    )
                )

    for target, (source, reason) in slow_renal_validated.items():
        for horizon in PREDICTION_HORIZONS_HOURS:
            if horizon in (24, 48):
                cells.append(
                    _cell(
                        target,
                        horizon,
                        "validated",
                        source,
                        "slow_renal_accumulation",
                        True,
                        reason,
                    )
                )
            else:
                cells.append(
                    _cell(
                        target,
                        horizon,
                        "fallback_wrong_horizon",
                        "persistence",
                        "slow_renal_accumulation",
                        False,
                        "slow renal targets do not reliably move at this horizon",
                    )
                )

    for module, by_horizon in VALIDATED_INTERMEDIATE_MOVE_CELLS.items():
        for horizon, targets in by_horizon.items():
            for target in targets:
                cells.append(
                    _cell(
                        target,
                        horizon,
                        "validated",
                        f"{module}_intermediate_router",
                        module,
                        True,
                        "validated all-module intermediate-horizon move cell",
                    )
                )

    for module, targets in VALIDATED_FULL_VARIABLE_FORECAST_6H_CELLS.items():
        calibrated_targets = set(VALIDATED_FULL_VARIABLE_INTERVAL_6H_CELLS.get(module, ()))
        for target in targets:
            cells.append(
                _cell(
                    target,
                    6,
                    "validated",
                    f"{module}_full_variable_router",
                    module,
                    True,
                    "validated full-variable 6h forecast cell",
                    interval_status="calibrated" if target in calibrated_targets else "needs_calibration_audit",
                )
            )

    for target in ("bilirubin", "inr", "ptt", "fibrinogen", "paco2"):
        for horizon in PREDICTION_HORIZONS_HOURS:
            cells.append(
                _cell(
                    target,
                    horizon,
                    "fallback_observability_limited",
                    "persistence_or_missing",
                    "sparse_or_observability_limited",
                    False,
                    "current audits do not validate forward movement for this sparse target",
                )
            )

    return tuple(cells)


def nowcast_state_grid() -> tuple[WholeBodyStateForecastCell, ...]:
    """Return validated same-time state-completion cells.

    These cells represent current-state estimates only.  They do not authorize
    future movement, even when the same target also appears in a forecast grid.
    """

    cells: list[WholeBodyStateForecastCell] = []
    for module, targets in VALIDATED_NOWCAST_MODULE_TARGETS.items():
        for target in targets:
            cells.append(
                WholeBodyStateForecastCell(
                    target=target,
                    module=module,
                    horizon_hours=0,
                    time_axis="current",
                    status="validated_nowcast_available",
                    source="same_time_nowcast_ridge",
                    point_estimate=None,
                    lower=None,
                    upper=None,
                    interval_level=None,
                    interval_status="not_a_future_interval",
                    can_estimate=True,
                    can_move=False,
                    reason=(
                        "validated same-time imputation; does not imply "
                        "validated future forecasting"
                    ),
                )
            )
    return tuple(cells)


def _trajectory_state_cells() -> tuple[WholeBodyStateForecastCell, ...]:
    cells: list[WholeBodyStateForecastCell] = []
    for cell in trajectory_prediction_grid():
        cells.append(
            WholeBodyStateForecastCell(
                target=cell.target,
                module=cell.context,
                horizon_hours=cell.horizon_hours,
                time_axis="future",
                status=cell.status,
                source=cell.source,
                point_estimate=None,
                lower=None,
                upper=None,
                interval_level=None,
                interval_status=cell.interval_status,
                can_estimate=cell.status == "validated",
                can_move=cell.can_move,
                reason=cell.reason,
            )
        )
    return tuple(cells)


def uncertainty_calibration_policy() -> dict[str, object]:
    """Return the interval-calibration contract for future prediction outputs."""

    return {
        "point_estimate_allowed": "validated target-horizon cells only",
        "interval_method": "split_conformal_residual_interval_per_target_horizon_source",
        "coverage_targets": [0.5, 0.8, 0.9],
        "primary_coverage_target": 0.9,
        "coverage_gate": {
            "required": True,
            "minimum_patient_heldout_splits": 7,
            "hospital_heldout_required": True,
            "acceptable_90pct_coverage_range": [0.87, 0.93],
        },
        "missing_calibration_policy": (
            "emit point estimate only when the target-horizon is validated; mark "
            "interval_status as needs_calibration_audit and do not show a numeric "
            "confidence interval"
        ),
        "fallback_interval_policy": (
            "for persistence fallback cells, interval calibration must be learned "
            "from persistence residuals at the same target and horizon before any "
            "numeric interval is shown"
        ),
        "output_schema": {
            "target": "physiology variable name",
            "horizon_hours": "prediction horizon",
            "point_estimate": "numeric value or null when missing",
            "lower": "numeric lower interval bound or null",
            "upper": "numeric upper interval bound or null",
            "interval_level": "coverage level, e.g. 0.9, or null",
            "source": "validated router, belief state, multihop path, or persistence",
            "status": "validated, fallback, candidate_only, or missing",
            "can_move": "whether model is allowed to move the target away from persistence",
        },
    }


def whole_body_state_forecast_schema() -> dict[str, object]:
    """Return the unified object schema for current and future body state."""

    return {
        "object_name": "whole_body_state_forecast",
        "patient_time": {
            "patient_key": "runtime-local identifier, never committed in aggregate artifacts",
            "anchor_time": "runtime timestamp or encounter-relative anchor",
            "timezone": "runtime-local if displayed",
        },
        "current_state_cell": {
            "target": "physiology variable name",
            "module": "validated module context",
            "horizon_hours": 0,
            "time_axis": "current",
            "point_estimate": "observed value or validated nowcast value",
            "lower": None,
            "upper": None,
            "interval_level": None,
            "interval_status": "not_a_future_interval",
            "source": (
                "observed, derived_formula_completion, "
                "same_group_calibrated_completion, same_time_nowcast_ridge, or missing"
            ),
            "status": (
                "observed, derived_formula_available, same_group_calibrated_available, "
                "validated_nowcast_available, unsupported, or missing"
            ),
            "can_estimate": "true only for observed or validated nowcast cells",
            "can_move": False,
        },
        "future_state_cell": {
            "target": "physiology variable name",
            "module": "validated forecast context",
            "horizon_hours": "one of PREDICTION_HORIZONS_HOURS",
            "time_axis": "future",
            "point_estimate": "validated forecast value or runtime persistence value",
            "lower": "numeric lower interval bound only when conformal calibration passes",
            "upper": "numeric upper interval bound only when conformal calibration passes",
            "interval_level": "coverage level, usually 0.9, or null",
            "interval_status": "calibrated or needs_calibration_audit",
            "source": "validated router, belief state, multihop path, persistence, or missing",
            "status": "validated, fallback, candidate_only, or missing",
            "can_estimate": "true only when point estimate is allowed",
            "can_move": "true only when the target-horizon cell is validated",
        },
    }


def whole_body_state_forecast_template() -> dict[str, object]:
    """Return a complete fail-closed template for the unified output object.

    This is a contract-level template, not a row-level patient forecast. Runtime
    code may fill numeric values only when the corresponding cell is validated
    or directly observed.
    """

    nowcast_cells = nowcast_state_grid()
    trajectory_cells = _trajectory_state_cells()
    validated_future = [cell for cell in trajectory_cells if cell.status == "validated"]
    fallback_future = [cell for cell in trajectory_cells if cell.status != "validated"]
    return {
        "object_name": "whole_body_state_forecast",
        "artifact_status": "schema_and_capability_template",
        "runtime_values_included": False,
        "schema": whole_body_state_forecast_schema(),
        "patient_time": {
            "patient_key": None,
            "anchor_time": None,
            "status": "runtime_required",
        },
        "current_state": {
            "role": "present-tense body-state completion",
            "completion_source_hierarchy": [
                "observed",
                "derived_formula_completion",
                "same_group_calibrated_completion",
                "same_time_nowcast_ridge",
                "missing",
            ],
            "derived_formula_rules": DERIVED_COMPLETION_RULES,
            "same_group_calibrated_completion_groups": SAME_GROUP_CALIBRATED_COMPLETION_GROUPS,
            "validated_nowcast_module_target_cells": sum(
                len(targets) for targets in VALIDATED_NOWCAST_MODULE_TARGETS.values()
            ),
            "cells": [cell.to_dict() for cell in nowcast_cells],
            "policy": {
                "observed_values_preferred": True,
                "derived_formula_before_ridge_nowcast": True,
                "same_group_calibration_not_counted_as_cross_system_nowcast": True,
                "nowcast_only_when_target_missing": True,
                "validated_module_target_required": True,
                "future_claim_from_nowcast_allowed": False,
            },
        },
        "future_trajectory": {
            "role": "validated factual physiology rollout",
            "horizons_hours": list(PREDICTION_HORIZONS_HOURS),
            "validated_cell_count": len(validated_future),
            "fallback_or_closed_cell_count": len(fallback_future),
            "cells": [cell.to_dict() for cell in trajectory_cells],
            "policy": {
                "move_rule": "only validated future cells may move away from persistence",
                "fallback_rule": "unsupported target-horizon pairs stay at persistence or missing",
                "no_recursive_unvalidated_rollout": True,
            },
        },
        "uncertainty": uncertainty_calibration_policy(),
        "source_artifacts": [
            "WHOLE_BODY_NOWCASTING_FINDINGS.md",
            "whole_body_nowcasting_audit.json",
            "WHOLE_BODY_TRAJECTORY_UNCERTAINTY_LAYER.md",
            "whole_body_rollout_uncertainty_contract.json",
            "whole_body_all_modules_intermediate_horizon_move_audit.json",
            "whole_body_all_modules_conformal_coverage_audit.json",
            "MIMICIV_CROSS_DATABASE_COVERAGE_FINDINGS.md",
            "mimiciv_cross_database_coverage_audit.json",
            "MIMICIV_ED_OBSERVATION_FINDINGS.md",
            "mimiciv_ed_observation_coverage_audit_1h.json",
            "mimiciv_ed_observation_coverage_audit_3h.json",
            "mimiciv_ed_observation_coverage_audit_6h.json",
            "MIMICIV_ED_TO_ICU_BASELINE_FINDINGS.md",
            "mimiciv_ed_to_icu_baseline_audit_1h.json",
            "mimiciv_ed_to_icu_baseline_audit_3h.json",
            "mimiciv_ed_to_icu_baseline_audit_6h.json",
            "NHANES_NOWCAST_FINDINGS.md",
            "nhanes_2017_2018_nowcast_audit.json",
            "MIMICIV_RADIOLOGY_NOTE_OBSERVATION_FINDINGS.md",
            "mimiciv_radiology_note_coverage_audit.json",
            "EICU_NEURO_NOTE_OBSERVATION_FINDINGS.md",
            "eicu_neuro_note_coverage_audit.json",
            "EICU_NEURO_NOTE_ALL_ICU_FINDINGS.md",
            "eicu_neuro_note_all_icu_coverage_audit.json",
            "CARDIOVASCULAR_BELIEF_FINDINGS.md",
            "eicu_cardiovascular_belief_audit.json",
            "eicu_cardiovascular_belief_full_heart_rate_audit.json",
            "eicu_cardiovascular_instability_full_transition_report.json",
            "ELECTROLYTE_BELIEF_FINDINGS.md",
            "eicu_electrolyte_belief_full_audit.json",
            "eicu_electrolyte_acid_base_full_transition_report.json",
            "RESPIRATORY_BELIEF_FINDINGS.md",
            "eicu_respiratory_belief_full_audit.json",
            "eicu_respiratory_full_transition_report.json",
            "ENDOCRINE_BELIEF_FINDINGS.md",
            "eicu_endocrine_belief_full_audit.json",
            "eicu_endocrine_stress_full_transition_report.json",
        ],
        "safety_boundary": {
            **observation_readiness()["safety_boundary"],
            "same_time_state_completion_allowed": True,
            "complete_human_simulation_claim_allowed": False,
        },
    }


def rollout_readiness() -> dict[str, object]:
    """Return readiness for a whole-body factual trajectory object."""

    grid = trajectory_prediction_grid()
    validated = [cell for cell in grid if cell.status == "validated"]
    fallback = [cell for cell in grid if cell.status != "validated"]
    return {
        "horizons_hours": list(PREDICTION_HORIZONS_HOURS),
        "validated_cell_count": len(validated),
        "fallback_or_closed_cell_count": len(fallback),
        "validated_cells": [cell.to_dict() for cell in validated],
        "fallback_or_closed_cells": [cell.to_dict() for cell in fallback],
        "trajectory_policy": {
            "rollout_shape": "target x horizon factual state grid",
            "move_rule": "only validated cells may move away from persistence",
            "fallback_rule": "unsupported target-horizon pairs stay at persistence or missing",
            "no_recursive_unvalidated_rollout": True,
        },
        "uncertainty_policy": uncertainty_calibration_policy(),
        "safety_boundary": observation_readiness()["safety_boundary"],
    }
