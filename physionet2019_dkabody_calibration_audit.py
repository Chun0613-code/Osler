"""Audit calibrated DKABody against PhysioNet-derived aggregate priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dka_body import DKABody
from dka_physionet_calibration import PhysioNetDkaCalibration


VARIABLES = {
    "Glucose": "G",
    "HCO3": "HCO3",
    "pH": "pH",
    "Potassium": "Ke",
    "MAP": "MAP",
    "Creatinine": "creatinine",
}


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"p10": None, "p50": None, "p90": None}
    return {
        "p10": round(float(np.quantile(values, 0.10)), 6),
        "p50": round(float(np.quantile(values, 0.50)), 6),
        "p90": round(float(np.quantile(values, 0.90)), 6),
    }


def simulate_no_action(calibration, samples, horizons, seed):
    rng = np.random.default_rng(seed)
    deltas = {
        str(horizon): {name: [] for name in VARIABLES}
        for horizon in horizons
    }
    presentations = {name: [] for name in VARIABLES}
    deaths = {}
    for _ in range(samples):
        body = DKABody(rng=rng, profile=calibration.sample_profile(rng))
        start = calibration.apply_presentation(body, rng)
        for name, state_key in VARIABLES.items():
            presentations[name].append(start[state_key])
        previous_hour = 0
        previous = start
        for horizon in horizons:
            step = horizon - previous_hour
            obs, _, dead, info = body.step({}, dt=float(step))
            for name, state_key in VARIABLES.items():
                deltas[str(horizon)][name].append(
                    (obs[state_key] - start[state_key]) / float(horizon)
                )
            previous_hour = horizon
            previous = obs
            if dead:
                cause = info.get("cause") or "unknown"
                deaths[cause] = deaths.get(cause, 0) + 1
                break
    return {
        "presentation_samples": {
            name: {"n": len(values), "quantiles": quantiles(values)}
            for name, values in presentations.items()
        },
        "no_action_delta_per_hour": {
            horizon: {
                name: {
                    "n": len(values),
                    "mean": round(float(np.mean(values)), 6) if values else None,
                    "median": round(float(np.median(values)), 6) if values else None,
                    "quantiles": quantiles(values),
                }
                for name, values in feature_map.items()
            }
            for horizon, feature_map in deltas.items()
        },
        "deaths": deaths,
    }


def compare_to_physionet(calibration_payload, simulation):
    phys = calibration_payload["action_unobserved_drift_model"]["delta_per_hour"]
    comparison = {}
    for horizon, feature_map in simulation["no_action_delta_per_hour"].items():
        comparison[horizon] = {}
        for feature, sim_stats in feature_map.items():
            phys_stats = phys[horizon].get(feature, {})
            sim_median = sim_stats["median"]
            phys_median = phys_stats.get("median")
            comparison[horizon][feature] = {
                "dkabody_no_action_median_delta_per_hour": sim_median,
                "physionet_action_unobserved_median_delta_per_hour": phys_median,
                "median_gap": (
                    round(float(sim_median - phys_median), 6)
                    if sim_median is not None and phys_median is not None else None
                ),
                "physionet_n": phys_stats.get("n"),
                "dkabody_n": sim_stats.get("n"),
            }
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default="physionet2019_dkabody_calibration.json")
    parser.add_argument("--output", default="physionet2019_dkabody_calibration_audit.json")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    calibration = PhysioNetDkaCalibration.load(args.calibration)
    horizons = tuple(
        int(value)
        for value in calibration.payload["action_unobserved_drift_model"]["horizons_hours"]
    )
    simulation = simulate_no_action(calibration, args.samples, horizons, args.seed)
    report = {
        "experiment": "physionet2019_calibrated_dkabody_audit",
        "calibration": args.calibration,
        "samples": args.samples,
        "simulation": simulation,
        "drift_comparison": compare_to_physionet(calibration.payload, simulation),
        "interpretation": (
            "PhysioNet drift is action-unobserved factual drift, not clean "
            "no-treatment causal drift. Large gaps should guide simulator review "
            "but must not be fit as treatment-effect truth."
        ),
        "safety_boundary": {
            "clinical_claim_allowed": False,
            "causal_no_treatment_claim_allowed": False,
            "treatment_effect_claim_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "samples": args.samples,
        "deaths": simulation["deaths"],
        "glucose_6h": report["drift_comparison"]["6"]["Glucose"],
    }, indent=2))


if __name__ == "__main__":
    main()
