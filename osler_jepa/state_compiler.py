"""Compile DKA observations into shared symbolic and numerical state features.

This module is the single source of truth for the grounded patient-state facts
consumed by both Prolog and JEPA. The neural model receives the resulting fact
vector plus signed residuals from physiologic reference ranges; missingness is
kept explicit and never silently converted into a positive fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from osler_jepa.ontology import OSLER_STATE_ONTOLOGY, StateOntology


DKA_SYMBOLIC_FACT_KEYS = (
    "hyperglycemia",
    "hypoglycemia",
    "low_potassium",
    "critical_hypokalemia",
    "hyperkalemia",
    "potassium_not_low",
    "severe_acidosis",
    "ketoacidosis_unresolved",
    "severe_hypotension",
    "hypovolemia",
    "renal_dysfunction",
    "oliguria",
    "total_body_potassium_depletion",
    "high_hyperosmolar_injury_burden",
)


def ground_dka_facts(state: Mapping[str, float]) -> frozenset[str]:
    """Ground the state-only facts used by the active DKA Prolog program."""
    get = lambda name, default: float(state.get(name, default))
    facts = {"patient_state_available"}
    glucose = get("G", 140.0)
    potassium = get("Ke", 4.0)
    if glucose >= 250:
        facts.add("hyperglycemia")
    if glucose < 70:
        facts.add("hypoglycemia")
    if potassium < 3.3:
        facts.add("low_potassium")
    if potassium < 3.0:
        facts.add("critical_hypokalemia")
    if potassium > 5.5:
        facts.add("hyperkalemia")
    if potassium >= 4.5:
        facts.add("potassium_not_low")
    if get("pH", 7.4) < 7.0:
        facts.add("severe_acidosis")
    if (
        get("HCO3", 24.0) < 18.0
        or get("anion_gap", 12.0) > 12.0
        or get("BHB", 0.0) >= 1.0
    ):
        facts.add("ketoacidosis_unresolved")
    if get("MAP", 75.0) < 55.0:
        facts.add("severe_hypotension")
    if get("V", 14.0) < 12.0:
        facts.add("hypovolemia")
    if get("creatinine", 1.0) > 2.5:
        facts.add("renal_dysfunction")
    if get("urine_output", 100.0) < 30.0:
        facts.add("oliguria")
    if get("K_store", 120.0) < 80.0:
        facts.add("total_body_potassium_depletion")
    if get("osmotic_injury", 0.0) >= 8.0:
        facts.add("high_hyperosmolar_injury_burden")
    return frozenset(facts)


@dataclass(frozen=True)
class CompiledDkaState:
    facts: tuple[str, ...]
    symbolic_vector: np.ndarray
    residual_vector: np.ndarray
    observed_mask: np.ndarray
    provenance: dict[str, str]

    @property
    def feature_vector(self) -> np.ndarray:
        return np.concatenate([self.symbolic_vector, self.residual_vector])


class DkaStateCompiler:
    """Shared symbolic-state compiler for the numerical DKA world model."""

    def __init__(
        self,
        state_keys: Sequence[str],
        state_mean: Sequence[float],
        state_std: Sequence[float],
        ontology: StateOntology = OSLER_STATE_ONTOLOGY,
    ):
        self.state_keys = tuple(state_keys)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.ontology = ontology
        if not (
            len(self.state_keys) == len(self.state_mean) == len(self.state_std)
        ):
            raise ValueError("State keys, means, and standard deviations must align")
        self._index = {name: index for index, name in enumerate(self.state_keys)}
        lows, highs = [], []
        for name in self.state_keys:
            spec = ontology.specs[ontology.require(name)]
            lows.append(np.nan if spec.normal_low is None else spec.normal_low)
            highs.append(np.nan if spec.normal_high is None else spec.normal_high)
        self.normal_low = np.asarray(lows, dtype=np.float32)
        self.normal_high = np.asarray(highs, dtype=np.float32)
        self.reference_mask = (
            np.isfinite(self.normal_low) | np.isfinite(self.normal_high)
        ).astype(np.float32)

    @property
    def symbolic_dim(self) -> int:
        return len(DKA_SYMBOLIC_FACT_KEYS)

    @property
    def context_dim(self) -> int:
        return self.symbolic_dim + len(self.state_keys)

    def compile_mapping(
        self,
        state: Mapping[str, float],
        observed: Sequence[str] | None = None,
    ) -> CompiledDkaState:
        observed_names = set(state if observed is None else observed)
        facts = ground_dka_facts({
            name: value for name, value in state.items() if name in observed_names
        })
        symbolic = np.asarray(
            [float(name in facts) for name in DKA_SYMBOLIC_FACT_KEYS],
            dtype=np.float32,
        )
        physical = self.state_mean.copy()
        mask = np.zeros(len(self.state_keys), dtype=np.float32)
        provenance = {}
        for name, value in state.items():
            if name not in self._index or name not in observed_names or value is None:
                continue
            index = self._index[name]
            physical[index] = float(value)
            mask[index] = 1.0
            provenance[name] = "observed"
        residual = self._numpy_residual(physical) * mask
        return CompiledDkaState(
            facts=tuple(sorted(facts)),
            symbolic_vector=symbolic,
            residual_vector=residual,
            observed_mask=mask,
            provenance=provenance,
        )

    def tensor_context(
        self,
        normalized_state: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fact flags and range residuals for a normalized state tensor."""
        if observed_mask is None:
            observed_mask = torch.ones_like(normalized_state)
        mask = observed_mask.to(dtype=normalized_state.dtype)
        mean = normalized_state.new_tensor(self.state_mean)
        std = normalized_state.new_tensor(self.state_std)
        physical = normalized_state * std + mean

        def value(name):
            return physical[..., self._index[name]]

        def known(name):
            return mask[..., self._index[name]] > 0

        glucose = value("G")
        potassium = value("Ke")
        flags = (
            known("G") & (glucose >= 250),
            known("G") & (glucose < 70),
            known("Ke") & (potassium < 3.3),
            known("Ke") & (potassium < 3.0),
            known("Ke") & (potassium > 5.5),
            known("Ke") & (potassium >= 4.5),
            known("pH") & (value("pH") < 7.0),
            (
                (known("HCO3") & (value("HCO3") < 18.0))
                | (known("anion_gap") & (value("anion_gap") > 12.0))
                | (known("BHB") & (value("BHB") >= 1.0))
            ),
            known("MAP") & (value("MAP") < 55.0),
            known("V") & (value("V") < 12.0),
            known("creatinine") & (value("creatinine") > 2.5),
            known("urine_output") & (value("urine_output") < 30.0),
            known("K_store") & (value("K_store") < 80.0),
            known("osmotic_injury") & (value("osmotic_injury") >= 8.0),
        )
        symbolic = torch.stack(flags, dim=-1).to(dtype=normalized_state.dtype)
        residual = self.tensor_residual(normalized_state) * mask
        return torch.cat([symbolic, residual], dim=-1)

    def tensor_residual(self, normalized_state: torch.Tensor) -> torch.Tensor:
        mean = normalized_state.new_tensor(self.state_mean)
        std = normalized_state.new_tensor(self.state_std)
        physical = normalized_state * std + mean
        low = normalized_state.new_tensor(self.normal_low)
        high = normalized_state.new_tensor(self.normal_high)
        has_low = torch.isfinite(low)
        has_high = torch.isfinite(high)
        below = torch.where(
            has_low,
            torch.minimum(physical - low, physical.new_zeros(())),
            physical.new_zeros(()),
        )
        above = torch.where(
            has_high,
            torch.maximum(physical - high, physical.new_zeros(())),
            physical.new_zeros(()),
        )
        bounded = (below + above) / std.clamp_min(1e-6)
        has_reference = has_low | has_high
        return torch.where(has_reference, bounded, normalized_state)

    def _numpy_residual(self, physical: np.ndarray) -> np.ndarray:
        below = np.where(
            np.isfinite(self.normal_low),
            np.minimum(physical - self.normal_low, 0.0),
            0.0,
        )
        above = np.where(
            np.isfinite(self.normal_high),
            np.maximum(physical - self.normal_high, 0.0),
            0.0,
        )
        bounded = (below + above) / np.maximum(self.state_std, 1e-6)
        normalized = (physical - self.state_mean) / np.maximum(self.state_std, 1e-6)
        return np.where(
            np.isfinite(self.normal_low) | np.isfinite(self.normal_high),
            bounded,
            normalized,
        ).astype(np.float32)

    def schema(self) -> dict:
        return {
            "type": "prolog_facts_plus_numeric_reference_residuals",
            "symbolic_facts": list(DKA_SYMBOLIC_FACT_KEYS),
            "state_variables": list(self.state_keys),
            "context_dim": self.context_dim,
            "missingness": "facts_and_residuals_require_observed_input",
            "rule_authority": "human_owned_active_prolog",
            "symbolic_head_context": True,
            "continuous_dynamics_authority": False,
        }
