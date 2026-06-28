"""Body-system disease contracts for eICU factual routers.

These contracts extend Chapter B beyond single diseases into broad organ-system
coverage.  Each module is still disease-specific at runtime: it has its own
cohort definition, target/action contract, active-window rule, and held-out
router gate.  Nothing here grants shared latent-space, causal, clinical, or
runtime action authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from eicu_aki_transition_extract import (
    DIURETIC_TERMS,
    NEPHROTOXIN_TERMS,
    RENAL_REPLACEMENT_TERMS,
)
from eicu_respiratory_transition_extract import (
    BRONCHODILATOR_TERMS,
    SYSTEMIC_STEROID_TERMS,
)
from eicu_sepsis_transition_extract import (
    ANTIBIOTIC_TERMS,
    FLUID_TERMS,
    VASOPRESSOR_TERMS,
    VENTILATION_TERMS,
)


INOTROPE_TERMS = (
    "dobutamine",
    "milrinone",
    "inamrinone",
    "isoproterenol",
)

ANTIARRHYTHMIC_TERMS = (
    "amiodarone",
    "lidocaine",
    "procainamide",
    "diltiazem",
    "verapamil",
    "metoprolol",
    "esmolol",
)

ANTICOAGULANT_TERMS = (
    "heparin",
    "enoxaparin",
    "lovenox",
    "warfarin",
    "argatroban",
    "bivalirudin",
)

ANTIPLATELET_TERMS = (
    "aspirin",
    "clopidogrel",
    "plavix",
    "ticagrelor",
    "prasugrel",
)

ANTIEPILEPTIC_TERMS = (
    "levetiracetam",
    "keppra",
    "phenytoin",
    "fosphenytoin",
    "valproate",
    "valproic",
    "lacosamide",
    "lorazepam",
    "midazolam",
)

OSMOTHERAPY_TERMS = (
    "mannitol",
    "hypertonic saline",
    "3% saline",
    "sodium chloride 3",
    "23.4% sodium chloride",
)

HEPATIC_ENCEPHALOPATHY_TERMS = (
    "lactulose",
    "rifaximin",
)

TRANSFUSION_TERMS = (
    "packed red blood",
    "packed red blood cells",
    "prbc",
    "prbcs",
    "rbc transfusion",
    "red blood cell",
    "red blood cells",
    "platelet transfusion",
    "platelet concentrate",
    "fresh frozen plasma",
    "ffp",
    "cryoprecipitate",
    "blood product",
    "blood products",
    "transfusion",
    "transfuse",
)


@dataclass(frozen=True)
class BodySystemDiseaseConfig:
    name: str
    display_name: str
    body_system: str
    diagnosis_patterns: tuple[str, ...]
    targets: tuple[str, ...]
    state_vars: tuple[str, ...]
    action_terms: dict[str, tuple[str, ...]]
    active_low: dict[str, float] = field(default_factory=dict)
    active_high: dict[str, float] = field(default_factory=dict)
    active_actions: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(self.action_terms.keys())

    def as_report_dict(self) -> dict[str, object]:
        return asdict(self) | {"action_keys": self.action_keys}


BODY_SYSTEM_CONFIGS: dict[str, BodySystemDiseaseConfig] = {
    "cardiovascular_instability": BodySystemDiseaseConfig(
        name="cardiovascular_instability",
        display_name="Cardiovascular instability / shock / heart failure",
        body_system="cardiovascular",
        diagnosis_patterns=(
            "cardiogenic shock",
            "shock, cardiogenic",
            "heart failure",
            "congestive heart failure",
            "acute coronary",
            "myocardial infarction",
            "cardiac arrest",
            "arrhythmia",
            "atrial fibrillation",
            "428",
            "410",
            "i50",
            "i21",
            "i46",
            "i48",
        ),
        targets=(
            "map",
            "heart_rate",
            "lactate",
            "creatinine",
            "urine_output",
            "potassium",
            "bicarbonate",
        ),
        state_vars=(
            "map",
            "heart_rate",
            "lactate",
            "creatinine",
            "urine_output",
            "potassium",
            "bicarbonate",
            "ph",
            "sodium",
            "o2sat",
            "respiratory_rate",
            "temperature",
            "glucose",
        ),
        action_terms={
            "vasopressor": VASOPRESSOR_TERMS,
            "inotrope": INOTROPE_TERMS,
            "fluids": FLUID_TERMS,
            "diuretics": DIURETIC_TERMS,
            "antiarrhythmic": ANTIARRHYTHMIC_TERMS,
            "ventilation": VENTILATION_TERMS,
        },
        active_low={"map": 65.0, "urine_output": 30.0},
        active_high={"lactate": 2.0, "heart_rate": 120.0, "creatinine": 2.0},
        active_actions=("hist_vasopressor", "hist_inotrope"),
        notes=(
            "Covers hemodynamic factual forecasting, not treatment-effect estimation.",
            "Vasopressor/inotrope claims remain observational and fail-closed for causality.",
        ),
    ),
    "acute_neuro": BodySystemDiseaseConfig(
        name="acute_neuro",
        display_name="Acute neurologic injury / seizure / coma",
        body_system="nervous_system",
        diagnosis_patterns=(
            "stroke",
            "cerebrovascular accident",
            "intracranial hemorrhage",
            "subarachnoid",
            "subdural",
            "seizure",
            "status epilepticus",
            "coma",
            "encephalopathy",
            "traumatic brain",
            "i63",
            "i61",
            "i60",
            "r56",
            "g40",
        ),
        targets=(
            "map",
            "heart_rate",
            "respiratory_rate",
            "o2sat",
            "glucose",
            "sodium",
            "ph",
        ),
        state_vars=(
            "map",
            "heart_rate",
            "respiratory_rate",
            "o2sat",
            "glucose",
            "sodium",
            "ph",
            "bicarbonate",
            "potassium",
            "creatinine",
            "temperature",
            "wbc",
        ),
        action_terms={
            "ventilation": VENTILATION_TERMS,
            "vasopressor": VASOPRESSOR_TERMS,
            "antiepileptic": ANTIEPILEPTIC_TERMS,
            "osmotherapy": OSMOTHERAPY_TERMS,
            "fluids": FLUID_TERMS,
        },
        active_low={"o2sat": 92.0, "map": 65.0, "glucose": 70.0, "ph": 7.30},
        active_high={"respiratory_rate": 24.0, "glucose": 250.0, "sodium": 150.0},
        active_actions=("hist_ventilation", "hist_antiepileptic", "hist_osmotherapy"),
        notes=(
            "ICU EHR does not expose neurologic exam granularity reliably; targets are physiologic proxies.",
            "No neurologic outcome or treatment-effect claim is allowed.",
        ),
    ),
    "hepatic_failure": BodySystemDiseaseConfig(
        name="hepatic_failure",
        display_name="Hepatic failure / cirrhosis / hepatic encephalopathy",
        body_system="hepatic_gastrointestinal",
        diagnosis_patterns=(
            "liver failure",
            "hepatic failure",
            "cirrhosis",
            "hepatic encephalopathy",
            "acute liver",
            "alcoholic hepatitis",
            "shock liver",
            "liver disease",
            "k72",
            "k74",
            "570",
            "571",
        ),
        targets=(
            "bilirubin",
            "bilirubin_direct",
            "platelets",
            "lactate",
            "bicarbonate",
            "creatinine",
            "map",
        ),
        state_vars=(
            "bilirubin",
            "bilirubin_direct",
            "platelets",
            "lactate",
            "bicarbonate",
            "creatinine",
            "map",
            "bun",
            "sodium",
            "potassium",
            "ph",
            "wbc",
            "glucose",
            "heart_rate",
        ),
        action_terms={
            "fluids": FLUID_TERMS,
            "vasopressor": VASOPRESSOR_TERMS,
            "antibiotics": ANTIBIOTIC_TERMS,
            "renal_replacement": RENAL_REPLACEMENT_TERMS,
            "hepatic_encephalopathy_tx": HEPATIC_ENCEPHALOPATHY_TERMS,
        },
        active_low={"platelets": 100.0, "map": 65.0},
        active_high={"bilirubin": 2.0, "lactate": 2.0, "creatinine": 2.0},
        active_actions=("hist_vasopressor", "hist_renal_replacement"),
        notes=(
            "Coagulation labs are incomplete in the current eICU state map; bilirubin/platelets are proxies.",
            "No transplant, bleeding, or encephalopathy treatment-effect claim is allowed.",
        ),
    ),
    "coagulopathy_heme": BodySystemDiseaseConfig(
        name="coagulopathy_heme",
        display_name="Coagulopathy / thrombocytopenia / hematologic instability",
        body_system="hematologic",
        diagnosis_patterns=(
            "thrombocytopenia",
            "coagulopathy",
            "disseminated intravascular",
            "dic",
            "anemia",
            "hemorrhage",
            "gastrointestinal bleeding",
            "gi bleed",
            "bleeding",
            "d65",
            "d69",
            "d62",
            "285",
            "286",
        ),
        targets=(
            "platelets",
            "hemoglobin",
            "hematocrit",
            "inr",
            "ptt",
            "fibrinogen",
            "wbc",
            "lactate",
            "map",
            "bicarbonate",
            "ph",
            "creatinine",
        ),
        state_vars=(
            "platelets",
            "hemoglobin",
            "hematocrit",
            "inr",
            "ptt",
            "fibrinogen",
            "wbc",
            "lactate",
            "map",
            "bicarbonate",
            "ph",
            "creatinine",
            "bun",
            "potassium",
            "sodium",
            "heart_rate",
            "temperature",
            "glucose",
        ),
        action_terms={
            "transfusion": TRANSFUSION_TERMS,
            "anticoagulant": ANTICOAGULANT_TERMS,
            "antiplatelet": ANTIPLATELET_TERMS,
            "fluids": FLUID_TERMS,
            "vasopressor": VASOPRESSOR_TERMS,
            "antibiotics": ANTIBIOTIC_TERMS,
            "nephrotoxin": NEPHROTOXIN_TERMS,
        },
        active_low={"platelets": 100.0, "hemoglobin": 8.0, "map": 65.0, "ph": 7.30},
        active_high={"inr": 1.5, "ptt": 45.0, "lactate": 2.0, "wbc": 12.0, "creatinine": 2.0},
        active_actions=("hist_transfusion", "hist_vasopressor"),
        notes=(
            "Hemoglobin, hematocrit, INR, PTT, fibrinogen, and transfusion evidence are first-class inputs.",
            "Transfusion and anticoagulation effects remain causal-closed.",
        ),
    ),
}


def get_body_system_config(name: str) -> BodySystemDiseaseConfig:
    try:
        return BODY_SYSTEM_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BODY_SYSTEM_CONFIGS))
        raise ValueError(f"unknown body-system disease '{name}'. Available: {available}") from exc
