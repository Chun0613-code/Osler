"""PhysioNet-derived calibration priors for DKABody.

These priors are aggregate statistics from PhysioNet/CinC Challenge 2019.  They
can make simulated DKA presentations and observation masks look more like real
ICU time-series, but they cannot identify treatment effects because the dataset
does not contain explicit DKA action channels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dka_body import DKAPatientProfile, estimate_potassium_store
from dka_world_model_contract import STATE_KEYS


DEFAULT_CALIBRATION_PATH = Path("physionet2019_dkabody_calibration.json")


def _quantile_range(section: dict[str, object], key: str, fallback: tuple[float, float]):
    values = section.get(key, {}).get("quantiles", {})
    lo = values.get("p05", fallback[0])
    hi = values.get("p95", fallback[1])
    if lo is None or hi is None or float(lo) >= float(hi):
        return fallback
    return float(lo), float(hi)


class PhysioNetDkaCalibration:
    """Aggregate PhysioNet calibration used as a safe simulator prior."""

    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.presentation = payload["presentation_model"]
        self.profile = payload["patient_variability_model"]
        self.measurement = payload["measurement_model"]

    @classmethod
    def load(cls, path: str | Path | None):
        if path is None:
            return None
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def sample_profile(self, rng: np.random.Generator) -> DKAPatientProfile:
        observed = self.profile.get("observed_proxy_ranges", {})
        unobserved = self.profile.get("unobserved_defaults", {})
        renal_lo, renal_hi = _quantile_range(observed, "renal_reserve", (0.55, 1.10))
        stress_lo, stress_hi = _quantile_range(
            observed, "counterregulatory_drive", (0.75, 1.55)
        )
        vascular_lo, vascular_hi = _quantile_range(
            observed, "vascular_tone", (0.82, 1.15)
        )
        fluid_lo, fluid_hi = _quantile_range(
            observed, "fluid_retention", (0.65, 1.00)
        )
        k_lo, k_hi = _quantile_range(
            observed, "potassium_store_scale", (0.65, 1.05)
        )
        na_lo, na_hi = _quantile_range(observed, "baseline_sodium", (132.0, 145.0))
        cr_lo, cr_hi = _quantile_range(
            observed, "baseline_creatinine", (0.6, 1.4)
        )
        insulin_lo, insulin_hi = unobserved.get(
            "insulin_sensitivity", {"range": [0.50, 1.70]}
        )["range"]
        weight_mu, weight_sd, weight_lo, weight_hi = unobserved.get(
            "weight_kg", {"normal_clip": [78.0, 18.0, 45.0, 135.0]}
        )["normal_clip"]
        endogenous_lo, endogenous_hi = unobserved.get(
            "endogenous_insulin", {"range": [0.5, 3.0]}
        )["range"]
        return DKAPatientProfile(
            weight_kg=float(np.clip(rng.normal(weight_mu, weight_sd), weight_lo, weight_hi)),
            renal_reserve=float(rng.uniform(renal_lo, renal_hi)),
            insulin_sensitivity=float(rng.uniform(insulin_lo, insulin_hi)),
            counterregulatory_drive=float(rng.uniform(stress_lo, stress_hi)),
            fluid_retention=float(rng.uniform(fluid_lo, fluid_hi)),
            vascular_tone=float(rng.uniform(vascular_lo, vascular_hi)),
            potassium_store_scale=float(rng.uniform(k_lo, k_hi)),
            endogenous_insulin=float(rng.uniform(endogenous_lo, endogenous_hi)),
            baseline_sodium=float(rng.uniform(na_lo, na_hi)),
            baseline_creatinine=float(rng.uniform(cr_lo, cr_hi)),
        )

    def _sample_raw_presentation(self, rng: np.random.Generator) -> dict[str, float]:
        features = self.presentation["transformed_feature_order"]
        mean = np.asarray(self.presentation["mean"], dtype=np.float64)
        covariance = np.asarray(self.presentation["covariance"], dtype=np.float64)
        sample = rng.multivariate_normal(mean, covariance)
        raw = {}
        for index, feature in enumerate(features):
            value = float(sample[index])
            if feature.startswith("log_"):
                key = feature.removeprefix("log_")
                value = float(np.exp(value))
            else:
                key = feature
            clips = self.presentation["clip_quantiles"].get(key, {})
            lo = clips.get("p01", clips.get("p05"))
            hi = clips.get("p99", clips.get("p95"))
            if lo is not None and hi is not None:
                value = float(np.clip(value, float(lo), float(hi)))
            raw[key] = value
        return raw

    def apply_presentation(self, body, rng: np.random.Generator):
        """Set a DKABody to a PhysioNet-calibrated DKA-like presentation."""
        raw = self._sample_raw_presentation(rng)
        body.G = float(raw["glucose"])
        body.HCO3 = float(raw["hco3"])
        body.Ke = float(raw["potassium"])
        body.Na = float(raw.get("sodium", body.profile.baseline_sodium))
        body.Cr = float(raw.get("creatinine", body.profile.baseline_creatinine))
        # MAP is derived from V and vascular tone.  Invert the simplified MAP
        # equation to get an ECF volume consistent with the sampled MAP.
        target_map = float(raw["map"])
        filling = np.clip((target_map - 30.0) / 60.0, 0.25, 1.2)
        body.V = float(np.clip(
            filling * body.volume_setpoint / max(body.profile.vascular_tone, 1e-6),
            0.45 * body.volume_setpoint,
            1.05 * body.volume_setpoint,
        ))
        # No direct anion-gap/BHB channel exists in PhysioNet 2019; derive a
        # ketoacid burden from bicarbonate deficit as a physiology prior.
        body.Ket = float(np.clip(24.0 - body.HCO3 + 0.015 * max(0.0, body.G - 200.0), 4.0, 22.0))
        body.Ki = float(estimate_potassium_store(
            body.Ke, body.pH, body.Cr, body.urine_output_ml_hr
        ))
        body.I = body.profile.endogenous_insulin
        body.counterregulatory_stress = float(np.clip(
            body.profile.counterregulatory_drive
            * (0.65 + max(0.0, body.G - 200.0) / 550.0
               + max(0.0, 18.0 - body.HCO3) / 24.0),
            0.35,
            2.2,
        ))
        body.renal_perfusion_state = body._instantaneous_renal_perfusion()
        body.osmotic_injury = 0.0
        body.critical_burdens = {cause: 0.0 for cause in body.critical_burdens}
        body.alive = True
        body.death_cause = None
        body.t = 0.0
        return body.observe()

    def observation_model_for_state_keys(self) -> dict[str, list[float]]:
        mapping = {
            "G": "Glucose",
            "pH": "pH",
            "HCO3": "HCO3",
            "Ke": "Potassium",
            "MAP": "MAP",
            "Na": "Sodium",
            "creatinine": "Creatinine",
        }
        features = self.measurement.get("features", {})
        probabilities = []
        age_p90 = []
        age_median = []
        for state_key in STATE_KEYS:
            phys_key = mapping.get(state_key)
            stats = features.get(phys_key, {}) if phys_key else {}
            has_observed_source = bool(stats)
            probabilities.append(float(
                stats.get("observation_rate", 0.35 if not has_observed_source else 1.0)
            ))
            interval = stats.get("interval_hours", {})
            age_median.append(float(interval.get("median", 6.0 if phys_key else 12.0)))
            age_p90.append(float(interval.get("p90", 18.0 if phys_key else 24.0)))
        return {
            "probabilities": probabilities,
            "age_median_hours": age_median,
            "age_p90_hours": age_p90,
        }
