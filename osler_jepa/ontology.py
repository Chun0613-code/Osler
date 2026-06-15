"""Versioned state vocabulary shared by symbolic Osler and neural world models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class StateSpec:
    canonical: str
    organ: str
    unit: str
    kind: str = "continuous"
    aliases: tuple[str, ...] = ()
    normal_low: float | None = None
    normal_high: float | None = None
    provenance: str = "Osler internal research ontology"


class StateOntology:
    def __init__(self, version: str, specs: Iterable[StateSpec]):
        self.version = version
        self.specs = {spec.canonical: spec for spec in specs}
        self._aliases = {}
        for spec in self.specs.values():
            for name in (spec.canonical, *spec.aliases):
                alias = self._normalize(name)
                existing = self._aliases.get(alias)
                if existing is not None and existing != spec.canonical:
                    raise ValueError(
                        f"Ontology alias {name!r} maps to both {existing} and "
                        f"{spec.canonical}"
                    )
                self._aliases[alias] = spec.canonical

    @staticmethod
    def _normalize(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def canonicalize(self, name: str) -> str | None:
        return self._aliases.get(self._normalize(name))

    def require(self, name: str) -> str:
        canonical = self.canonicalize(name)
        if canonical is None:
            raise KeyError(f"Unknown Osler state: {name}")
        return canonical

    def canonicalize_mapping(self, values: Mapping[str, float]) -> dict[str, float]:
        output = {}
        for name, value in values.items():
            canonical = self.canonicalize(name)
            if canonical is not None:
                output[canonical] = float(value)
        return output

    def derive_flags(self, values: Mapping[str, float]) -> list[str]:
        state = self.canonicalize_mapping(values)
        flags = []
        glucose = state.get("metabolic.glucose")
        ph = state.get("metabolic.ph")
        potassium = state.get("electrolyte.potassium")
        map_value = state.get("cardiovascular.map")
        osmolality = state.get("metabolic.effective_osmolality")
        creatinine = state.get("renal.creatinine")
        urine_output = state.get("renal.urine_output")
        bhb = state.get("metabolic.beta_hydroxybutyrate")
        potassium_store = state.get("electrolyte.total_body_potassium_store")
        osmotic_injury = state.get("neurologic.hyperosmolar_injury")
        if glucose is not None and glucose > 250:
            flags.append("hyperglycemia")
        if glucose is not None and glucose < 70:
            flags.append("hypoglycemia")
        if ph is not None and ph < 7.0:
            flags.append("severe_acidosis")
        if potassium is not None and potassium > 5.5:
            flags.append("hyperkalemia")
        if potassium is not None and potassium < 3.0:
            flags.append("hypokalemia")
        if map_value is not None and map_value < 65:
            flags.append("hypotension")
        if osmolality is not None and osmolality > 320:
            flags.append("hyperosmolarity")
        if creatinine is not None and creatinine > 2.0:
            flags.append("renal_dysfunction")
        if urine_output is not None and urine_output < 30:
            flags.append("oliguria")
        if bhb is not None and bhb >= 3.0:
            flags.append("ketonemia")
        if potassium_store is not None and potassium_store < 80:
            flags.append("total_body_potassium_depletion")
        if osmotic_injury is not None and osmotic_injury >= 8.0:
            flags.append("high_hyperosmolar_injury_burden")
        return flags

    def schema(self) -> dict:
        return {
            "version": self.version,
            "states": [
                {
                    "canonical": spec.canonical,
                    "organ": spec.organ,
                    "unit": spec.unit,
                    "kind": spec.kind,
                    "aliases": list(spec.aliases),
                    "normal_range": [spec.normal_low, spec.normal_high],
                    "provenance": spec.provenance,
                }
                for spec in self.specs.values()
            ],
        }


OSLER_STATE_ONTOLOGY = StateOntology(
    version="1.1.0",
    specs=(
        StateSpec("metabolic.glucose", "pancreas", "mg/dL", aliases=(
            "G", "glucose", "blood_glucose", "blood_glucose_elevation",
        ), normal_low=70, normal_high=140),
        StateSpec("metabolic.ph", "blood", "pH", aliases=("pH", "blood_ph"),
                  normal_low=7.35, normal_high=7.45),
        StateSpec("metabolic.bicarbonate", "blood", "mEq/L", aliases=(
            "HCO3", "bicarbonate",
        ), normal_low=22, normal_high=28),
        StateSpec("metabolic.anion_gap", "blood", "mEq/L", aliases=(
            "anion_gap",
        ), normal_low=8, normal_high=16),
        StateSpec("metabolic.ketogenesis", "liver", "normalized", aliases=(
            "ketogenesis",
        )),
        StateSpec("metabolic.beta_hydroxybutyrate", "blood", "mmol/L", aliases=(
            "BHB", "beta_hydroxybutyrate", "beta-hydroxybutyrate",
        ), normal_low=0.0, normal_high=0.6),
        StateSpec("metabolic.effective_osmolality", "blood", "mOsm/kg", aliases=(
            "osmolality", "effective_osmolality", "serum_osmolality",
        ), normal_low=275, normal_high=295),
        StateSpec("electrolyte.potassium", "blood", "mEq/L", aliases=(
            "Ke", "potassium", "K",
        ), normal_low=3.5, normal_high=5.0),
        StateSpec("electrolyte.sodium", "blood", "mEq/L", aliases=(
            "Na", "sodium",
        ), normal_low=135, normal_high=145),
        StateSpec("electrolyte.total_body_potassium_store", "whole_body",
                  "mEq-equivalent", aliases=("K_store", "Ki", "potassium_store")),
        StateSpec("neurologic.hyperosmolar_injury", "brain", "burden", aliases=(
            "osmotic_injury", "hyperosmolar_injury",
        ), normal_low=0.0, normal_high=2.0),
        StateSpec("cardiovascular.map", "vessel", "mmHg", aliases=(
            "MAP", "map", "systemic_BP", "systemic_bp",
        ), normal_low=65, normal_high=105),
        StateSpec("cardiovascular.perfusion", "vessel", "normalized", aliases=(
            "perfusion",
        )),
        StateSpec("cardiovascular.sbp", "vessel", "mmHg", aliases=("sbp",)),
        StateSpec("cardiovascular.heart_rate", "heart", "bpm", aliases=(
            "heart_rate", "HR",
        ), normal_low=60, normal_high=100),
        StateSpec("respiratory.spo2", "lung", "%", aliases=("spo2",),
                  normal_low=92, normal_high=100),
        StateSpec("respiratory.hypoxia", "lung", "normalized", aliases=("hypoxia",)),
        StateSpec("fluid.extracellular_volume", "vessel", "L", aliases=(
            "V", "volume",
        )),
        StateSpec("fluid.hypovolemia", "vessel", "normalized", aliases=("hypovolemia",)),
        StateSpec("fluid.overload", "vessel", "normalized", aliases=("fluid_overload",)),
        StateSpec("endocrine.insulin_signal", "pancreas", "mU/L", aliases=(
            "I", "insulin_signal",
        )),
        StateSpec("renal.creatinine", "kidney", "mg/dL", aliases=(
            "creatinine", "Cr",
        )),
        StateSpec("renal.urine_output", "kidney", "mL/hr", aliases=(
            "urine_output", "urine_output_ml_hr",
        )),
        StateSpec("cardiovascular.preload", "heart", "normalized", aliases=("preload",)),
        StateSpec("cardiovascular.afterload", "heart", "normalized", aliases=("afterload",)),
        StateSpec("respiratory.airflow_resistance", "lung", "normalized", aliases=(
            "airflow_resistance",
        )),
        StateSpec("respiratory.bronchospasm", "lung", "normalized", aliases=(
            "bronchospasm",
        )),
        StateSpec("hemostasis.platelet_aggregation", "blood", "normalized", aliases=(
            "platelet_aggregation",
        )),
        StateSpec("hemostasis.clot_propagation_risk", "blood", "normalized", aliases=(
            "clot_propagation_risk",
        )),
        StateSpec("hemostasis.coagulation_activity", "blood", "normalized", aliases=(
            "coagulation_cascade",
        )),
        StateSpec("infection.load", "immune", "normalized", aliases=(
            "infection_load",
        )),
        StateSpec("infection.pathogen_burden", "immune", "normalized", aliases=(
            "bacterial_pathogen",
        )),
    ),
)


DKA_MODEL_TO_CANONICAL = {
    "G": "metabolic.glucose",
    "pH": "metabolic.ph",
    "HCO3": "metabolic.bicarbonate",
    "anion_gap": "metabolic.anion_gap",
    "Ke": "electrolyte.potassium",
    "MAP": "cardiovascular.map",
    "V": "fluid.extracellular_volume",
    "I": "endocrine.insulin_signal",
    "Na": "electrolyte.sodium",
    "osmolality": "metabolic.effective_osmolality",
    "creatinine": "renal.creatinine",
    "urine_output": "renal.urine_output",
    "BHB": "metabolic.beta_hydroxybutyrate",
    "K_store": "electrolyte.total_body_potassium_store",
    "osmotic_injury": "neurologic.hyperosmolar_injury",
}
