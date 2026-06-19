"""Interpretable belief states used by the Osler-JEPA runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from dka_action_contract import ACTION_INDEX, expand_action
from dka_body import estimate_potassium_store


@dataclass(frozen=True)
class PotassiumStoreBelief:
    """Gaussian belief over latent total-body potassium reserve.

    The prediction step accounts for documented replacement and estimated renal
    loss. The update step fuses the mechanistic serum/pH proxy and, when present,
    the JEPA latent-state estimate. This is a research belief, not a measured lab.
    """

    mean: float
    variance: float
    source: str = "mechanistic_prior"

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(self.variance, 1e-6))

    @classmethod
    def from_state(cls, state: dict) -> "PotassiumStoreBelief":
        provided = state.get("K_store")
        if provided is not None and np.isfinite(provided):
            return cls(
                mean=float(np.clip(provided, 40.0, 170.0)),
                variance=16.0,
                source="provided_belief",
            )
        mean = estimate_potassium_store(
            state.get("Ke", 4.0),
            state.get("pH", 7.35),
            state.get("creatinine", 1.2),
            state.get("urine_output", 100.0),
        )
        return cls(mean=mean, variance=100.0)

    def predict(self, action, state: dict, delta_hours: float):
        values = expand_action(action)
        hours = max(0.0, float(delta_hours))
        kcl_rate = float(values[ACTION_INDEX["kcl"]])
        urine_output = max(0.0, float(state.get("urine_output", 100.0)))
        glucose = max(0.0, float(state.get("G", 140.0)))
        creatinine = max(0.1, float(state.get("creatinine", 1.2)))

        replacement = 0.45 * kcl_rate * hours
        osmotic_drive = max(0.0, (glucose - 180.0) / 300.0)
        urine_drive = max(0.0, (urine_output - 30.0) / 120.0)
        renal_clearance = float(np.clip(1.4 / creatinine, 0.25, 1.5))
        estimated_loss = 1.8 * osmotic_drive * urine_drive * renal_clearance * hours
        mean = float(np.clip(
            self.mean + replacement - estimated_loss, 40.0, 170.0
        ))
        process_variance = (2.0 + 2.0 * osmotic_drive + 0.5 * kcl_rate / 10.0) * hours
        return PotassiumStoreBelief(
            mean=mean,
            variance=float(min(400.0, self.variance + process_variance)),
            source="predicted_from_actions_and_renal_loss",
        )

    def update(self, state: dict, model_estimate: float | None = None):
        proxy = estimate_potassium_store(
            state.get("Ke", 4.0),
            state.get("pH", 7.35),
            state.get("creatinine", 1.2),
            state.get("urine_output", 100.0),
        )
        # Serum potassium is an indirect and acid-base-dependent observation of
        # total-body reserve, so its measurement variance remains deliberately high.
        posterior_mean, posterior_variance = _kalman_update(
            self.mean, self.variance, proxy, 64.0
        )
        source = "serum_proxy_update"
        if model_estimate is not None and np.isfinite(model_estimate):
            posterior_mean, posterior_variance = _kalman_update(
                posterior_mean, posterior_variance,
                float(np.clip(model_estimate, 40.0, 170.0)), 36.0,
            )
            source = "serum_proxy_plus_jepa_update"
        return PotassiumStoreBelief(
            mean=float(np.clip(posterior_mean, 40.0, 170.0)),
            variance=float(max(posterior_variance, 1.0)),
            source=source,
        )

    def step(self, action, state: dict, delta_hours: float,
             model_estimate: float | None = None):
        return self.predict(action, state, delta_hours).update(
            state, model_estimate=model_estimate
        )

    def to_dict(self) -> dict:
        return {
            "name": "potassium_store",
            "mean": round(self.mean, 4),
            "standard_deviation": round(self.standard_deviation, 4),
            "source": self.source,
            "measured": False,
            "validation_gate": downstream_gate_spec(
                ("Ke", "K_store", "urine_output")
            ),
        }


@dataclass(frozen=True)
class HiddenStateBelief:
    """Generic scalar belief for unmeasured physiologic state.

    These beliefs are deliberately useful only through downstream observable
    prediction improvements. They should never be treated as measured labs.
    """

    name: str
    mean: float
    variance: float
    source: str
    downstream_targets: tuple[str, ...]
    measured: bool = False

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(self.variance, 1e-6))

    def update(self, observation, observation_variance, source):
        mean, variance = _kalman_update(
            self.mean, self.variance, observation, observation_variance
        )
        return HiddenStateBelief(
            name=self.name,
            mean=float(mean),
            variance=float(max(variance, 1.0e-6)),
            source=source,
            downstream_targets=self.downstream_targets,
            measured=False,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": round(float(self.mean), 4),
            "standard_deviation": round(self.standard_deviation, 4),
            "source": self.source,
            "measured": False,
            "validation_gate": downstream_gate_spec(self.downstream_targets),
        }


def downstream_gate_spec(targets):
    return {
        "type": "downstream_observable_prediction_improvement",
        "targets": list(targets),
        "direct_hidden_state_accuracy_claim_allowed": False,
        "requires_capacity_matched_placebo": True,
        "promotion_rule": (
            "Keep this belief only if adding it improves held-out prediction of "
            "its observable downstream targets more than a capacity-matched "
            "placebo belief, without worsening collapse, counterfactual, or "
            "calibration gates."
        ),
    }


def downstream_observable_gate(
    belief_name,
    baseline_mae,
    candidate_mae,
    targets,
    placebo_mae=None,
    min_relative_improvement=0.02,
    min_placebo_margin=0.01,
    max_relative_worsening=0.005,
):
    """Gate an unmeasured belief by observable downstream prediction value."""

    placebo_mae = placebo_mae or {}
    target_reports = {}
    usable_improvements = []
    usable_placebo_margins = []
    for target in targets:
        base = baseline_mae.get(target)
        candidate = candidate_mae.get(target)
        if base is None or candidate is None or not np.isfinite(base) or base <= 0:
            target_reports[target] = {
                "baseline_mae": base,
                "candidate_mae": candidate,
                "placebo_mae": placebo_mae.get(target),
                "relative_improvement": None,
                "placebo_margin": None,
                "usable": False,
            }
            continue
        improvement = (float(base) - float(candidate)) / float(base)
        usable_improvements.append(improvement)
        placebo = placebo_mae.get(target)
        placebo_margin = None
        if placebo is not None and np.isfinite(placebo) and float(placebo) > 0:
            placebo_margin = (float(placebo) - float(candidate)) / float(placebo)
            usable_placebo_margins.append(placebo_margin)
        target_reports[target] = {
            "baseline_mae": round(float(base), 6),
            "candidate_mae": round(float(candidate), 6),
            "placebo_mae": (
                round(float(placebo), 6)
                if placebo is not None and np.isfinite(placebo) else None
            ),
            "relative_improvement": round(float(improvement), 6),
            "placebo_margin": (
                round(float(placebo_margin), 6)
                if placebo_margin is not None else None
            ),
            "usable": True,
        }
    mean_improvement = (
        float(np.mean(usable_improvements)) if usable_improvements else None
    )
    worst = float(np.min(usable_improvements)) if usable_improvements else None
    mean_placebo_margin = (
        float(np.mean(usable_placebo_margins))
        if usable_placebo_margins else None
    )
    worst_placebo_margin = (
        float(np.min(usable_placebo_margins))
        if usable_placebo_margins else None
    )
    passed = bool(
        usable_improvements
        and mean_improvement >= float(min_relative_improvement)
        and worst >= -float(max_relative_worsening)
        and usable_placebo_margins
        and mean_placebo_margin >= float(min_placebo_margin)
        and worst_placebo_margin >= -float(max_relative_worsening)
    )
    return {
        "belief": belief_name,
        "passed": passed,
        "mean_relative_improvement": (
            round(mean_improvement, 6) if mean_improvement is not None else None
        ),
        "worst_relative_improvement": (
            round(worst, 6) if worst is not None else None
        ),
        "mean_placebo_margin": (
            round(mean_placebo_margin, 6)
            if mean_placebo_margin is not None else None
        ),
        "worst_placebo_margin": (
            round(worst_placebo_margin, 6)
            if worst_placebo_margin is not None else None
        ),
        "targets": target_reports,
        "requires_capacity_matched_placebo": True,
        "placebo_control_present": bool(usable_placebo_margins),
        "direct_hidden_state_accuracy_claim_allowed": False,
        "clinical_claim_allowed": False,
    }


def acid_base_buffer_belief_from_state(state):
    hco3 = float(state.get("HCO3", 15.0))
    ph = float(state.get("pH", 7.25))
    anion_gap = float(state.get("anion_gap", 18.0))
    hco3_component = np.clip((hco3 - 6.0) / 18.0, 0.0, 1.0)
    ph_component = np.clip((ph - 6.8) / 0.6, 0.0, 1.0)
    gap_penalty = np.clip((anion_gap - 12.0) / 20.0, 0.0, 1.0)
    reserve = 0.55 * hco3_component + 0.35 * ph_component + 0.10 * (1.0 - gap_penalty)
    return HiddenStateBelief(
        name="acid_base_buffer_reserve",
        mean=float(np.clip(reserve, 0.0, 1.0)),
        variance=0.05,
        source="hco3_ph_anion_gap_proxy",
        downstream_targets=("HCO3", "pH", "anion_gap"),
    )


def insulin_sensitivity_belief_from_state(state):
    glucose = float(state.get("G", 250.0))
    insulin_signal = max(0.0, float(state.get("I", 1.0)))
    stress = max(0.1, float(state.get("counterregulatory_stress", 1.0)))
    if insulin_signal <= 1.5:
        estimate = 1.0
        variance = 0.18
        source = "weakly_identified_without_insulin_exposure"
    else:
        hyperglycemia = np.clip((glucose - 180.0) / 420.0, 0.0, 1.0)
        estimate = np.clip(1.35 - 0.55 * hyperglycemia / stress, 0.45, 1.70)
        variance = 0.10
        source = "glucose_insulin_stress_proxy"
    return HiddenStateBelief(
        name="insulin_sensitivity",
        mean=float(estimate),
        variance=variance,
        source=source,
        downstream_targets=("G", "BHB", "HCO3"),
    )


def sodium_water_balance_belief_from_state(state):
    sodium = float(state.get("Na", 138.0))
    osmolality = float(state.get("osmolality", 2.0 * sodium + float(state.get("G", 140.0)) / 18.0))
    map_value = float(state.get("MAP", 80.0))
    urine = float(state.get("urine_output", 100.0))
    osm_component = np.clip((310.0 - osmolality) / 35.0, -1.0, 1.0)
    perfusion_component = np.clip((map_value - 65.0) / 35.0, -1.0, 1.0)
    urine_component = np.clip((urine - 30.0) / 170.0, 0.0, 1.0)
    balance = 0.45 * osm_component + 0.35 * perfusion_component + 0.20 * urine_component
    return HiddenStateBelief(
        name="sodium_water_balance",
        mean=float(np.clip(balance, -1.0, 1.0)),
        variance=0.08,
        source="sodium_osmolality_map_urine_proxy",
        downstream_targets=("Na", "osmolality", "MAP", "urine_output"),
    )


def renal_reserve_belief_from_state(state):
    creatinine = max(0.1, float(state.get("creatinine", 1.2)))
    urine = max(0.0, float(state.get("urine_output", 100.0)))
    map_value = float(state.get("MAP", 80.0))
    creatinine_component = np.clip(1.2 / creatinine, 0.25, 1.5)
    urine_component = np.clip(urine / 120.0, 0.0, 1.5)
    perfusion_component = np.clip((map_value - 40.0) / 50.0, 0.0, 1.2)
    reserve = (
        0.55 * creatinine_component
        + 0.25 * urine_component
        + 0.20 * perfusion_component
    )
    return HiddenStateBelief(
        name="renal_reserve",
        mean=float(np.clip(reserve, 0.20, 1.60)),
        variance=0.12,
        source="creatinine_urine_map_proxy",
        downstream_targets=("creatinine", "urine_output", "Ke", "HCO3"),
    )


def infer_hidden_beliefs(state):
    """Return all current DKA hidden-state beliefs with provenance."""

    return {
        "potassium_store": PotassiumStoreBelief.from_state(state),
        "acid_base_buffer_reserve": acid_base_buffer_belief_from_state(state),
        "insulin_sensitivity": insulin_sensitivity_belief_from_state(state),
        "sodium_water_balance": sodium_water_balance_belief_from_state(state),
        "renal_reserve": renal_reserve_belief_from_state(state),
    }


def _kalman_update(prior_mean, prior_variance, observation, observation_variance):
    gain = prior_variance / max(prior_variance + observation_variance, 1e-6)
    mean = prior_mean + gain * (observation - prior_mean)
    variance = (1.0 - gain) * prior_variance
    return mean, variance
