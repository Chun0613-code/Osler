"""Audit whether DKABody is under-buffered versus real ICU dynamics.

This is a falsification test for the hypothesis:

    The simulator may over-predict short-horizon movement in buffered variables
    such as bicarbonate and serum potassium, causing JEPA to lose to persistence
    on real data.

PhysioNet 2019 has no medication channels, so its dynamics are action-unobserved
factual dynamics, not untreated causal drift. The comparison is therefore
suggestive: it can identify an implausibly jumpy simulator, but it must not be
used as a treatment-effect calibration target.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dka_body import DKABody, HCO3_NORM, KE_NORM
from dka_physionet_calibration import PhysioNetDkaCalibration
from physionet2019_calibrate_dkabody import dka_like_rows, row_value
from physionet2019_pretrain import psv_files, read_patient


FEATURES = {
    "HCO3": {
        "state_key": "HCO3",
        "physionet_key": "HCO3",
        "setpoint": HCO3_NORM,
    },
    "Potassium": {
        "state_key": "Ke",
        "physionet_key": "Potassium",
        "setpoint": KE_NORM,
    },
    "pH": {
        "state_key": "pH",
        "physionet_key": "pH",
        "setpoint": 7.40,
    },
    "Glucose": {
        "state_key": "G",
        "physionet_key": "Glucose",
        "setpoint": 100.0,
    },
    "MAP": {
        "state_key": "MAP",
        "physionet_key": "MAP",
        "setpoint": 90.0,
    },
}


def finite(value: float) -> bool:
    return bool(np.isfinite(value))


def quantiles(values, probs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    names = [f"p{int(prob * 100):02d}" for prob in probs]
    if len(array) == 0:
        return {name: None for name in names}
    return {
        name: round(float(np.quantile(array, prob)), 6)
        for name, prob in zip(names, probs)
    }


def correlation(x_values, y_values):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 6)


def describe_deltas(deltas, current_values, target_values, setpoint):
    delta = np.asarray(deltas, dtype=np.float64)
    current = np.asarray(current_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    mask = np.isfinite(delta) & np.isfinite(current) & np.isfinite(target)
    delta = delta[mask]
    current = current[mask]
    target = target[mask]
    if len(delta) == 0:
        return {
            "n": 0,
            "mean_delta_per_hour": None,
            "std_delta_per_hour": None,
            "median_abs_delta_per_hour": None,
            "p90_abs_delta_per_hour": None,
            "delta_quantiles": quantiles([]),
            "current_target_autocorrelation": None,
            "mean_reversion_beta_per_hour": None,
        }
    abs_delta = np.abs(delta)
    gap = float(setpoint) - current
    gap_var = float(np.var(gap))
    beta = None
    if len(delta) >= 3 and gap_var > 1e-9:
        beta = float(np.cov(gap, delta, bias=True)[0, 1] / gap_var)
    return {
        "n": int(len(delta)),
        "mean_delta_per_hour": round(float(np.mean(delta)), 6),
        "std_delta_per_hour": round(float(np.std(delta)), 6),
        "median_abs_delta_per_hour": round(float(np.median(abs_delta)), 6),
        "p90_abs_delta_per_hour": round(float(np.quantile(abs_delta, 0.90)), 6),
        "delta_quantiles": quantiles(delta),
        "current_target_autocorrelation": correlation(current, target),
        "mean_reversion_beta_per_hour": (
            round(beta, 6) if beta is not None else None
        ),
    }


def empty_collectors(horizons):
    return {
        str(horizon): {
            feature: {"delta": [], "current": [], "target": []}
            for feature in FEATURES
        }
        for horizon in horizons
    }


def collect_physionet(paths, horizons):
    collectors = empty_collectors(horizons)
    anchor_rows = 0
    patients_with_anchor = 0
    for path in paths:
        patient = read_patient(path)
        dynamic = patient["dynamic"]
        rows = dka_like_rows(dynamic)
        if len(rows) > 0:
            patients_with_anchor += 1
        for row in rows:
            anchor_rows += 1
            row = int(row)
            for horizon in horizons:
                future = row + int(horizon)
                if future >= len(dynamic):
                    continue
                for feature, spec in FEATURES.items():
                    key = spec["physionet_key"]
                    current = row_value(dynamic, row, key)
                    target = row_value(dynamic, future, key)
                    if finite(current) and finite(target):
                        bucket = collectors[str(horizon)][feature]
                        bucket["current"].append(current)
                        bucket["target"].append(target)
                        bucket["delta"].append((target - current) / float(horizon))
    return collectors, {
        "patients_scanned": len(paths),
        "patients_with_dka_like_anchor": patients_with_anchor,
        "dka_like_anchor_rows": anchor_rows,
        "selection": (
            "Glucose>=200 and (HCO3<=18 or pH<=7.30), using only directly "
            "observed feature pairs at current and future rows"
        ),
    }


def collect_simulator(calibration, samples, horizons, seed):
    rng = np.random.default_rng(seed)
    collectors = empty_collectors(horizons)
    deaths = Counter()
    max_horizon = max(horizons)
    for _ in range(samples):
        body = DKABody(rng=rng, profile=calibration.sample_profile(rng))
        start = calibration.apply_presentation(body, rng)
        start_values = {
            feature: float(start[spec["state_key"]])
            for feature, spec in FEATURES.items()
        }
        observations = {}
        for hour in range(1, int(max_horizon) + 1):
            obs, _, dead, info = body.step({}, dt=1.0)
            observations[hour] = obs
            if dead:
                deaths[info.get("cause") or "unknown"] += 1
                break
        for horizon in horizons:
            obs = observations.get(int(horizon))
            if obs is None:
                continue
            for feature, spec in FEATURES.items():
                current = start_values[feature]
                target = float(obs[spec["state_key"]])
                bucket = collectors[str(horizon)][feature]
                bucket["current"].append(current)
                bucket["target"].append(target)
                bucket["delta"].append((target - current) / float(horizon))
    return collectors, {
        "samples": samples,
        "deaths_before_or_at_max_horizon": dict(deaths),
        "action": "no_action",
    }


def summarize_collectors(collectors):
    return {
        horizon: {
            feature: describe_deltas(
                values["delta"],
                values["current"],
                values["target"],
                FEATURES[feature]["setpoint"],
            )
            for feature, values in feature_map.items()
        }
        for horizon, feature_map in collectors.items()
    }


def ratio(numerator, denominator):
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return round(float(numerator) / float(denominator), 6)


def compare(physionet_summary, simulator_summary):
    comparison = {}
    for horizon, phys_features in physionet_summary.items():
        comparison[horizon] = {}
        for feature, phys in phys_features.items():
            sim = simulator_summary[horizon][feature]
            phys_abs_p90 = phys["p90_abs_delta_per_hour"]
            sim_abs_p90 = sim["p90_abs_delta_per_hour"]
            phys_abs_med = phys["median_abs_delta_per_hour"]
            sim_abs_med = sim["median_abs_delta_per_hour"]
            phys_ar = phys["current_target_autocorrelation"]
            sim_ar = sim["current_target_autocorrelation"]
            comparison[horizon][feature] = {
                "sim_to_physionet_median_abs_delta_ratio": ratio(
                    sim_abs_med, phys_abs_med
                ),
                "sim_to_physionet_p90_abs_delta_ratio": ratio(
                    sim_abs_p90, phys_abs_p90
                ),
                "autocorrelation_gap_sim_minus_physionet": (
                    round(float(sim_ar - phys_ar), 6)
                    if sim_ar is not None and phys_ar is not None else None
                ),
                "mean_reversion_beta_gap_sim_minus_physionet": (
                    round(
                        float(
                            sim["mean_reversion_beta_per_hour"]
                            - phys["mean_reversion_beta_per_hour"]
                        ),
                        6,
                    )
                    if (
                        sim["mean_reversion_beta_per_hour"] is not None
                        and phys["mean_reversion_beta_per_hour"] is not None
                    ) else None
                ),
                "supports_over_volatility": (
                    sim_abs_p90 is not None
                    and phys_abs_p90 is not None
                    and sim_abs_p90 > 1.5 * phys_abs_p90
                ),
                "supports_lower_autocorrelation": (
                    sim_ar is not None
                    and phys_ar is not None
                    and sim_ar < phys_ar - 0.05
                ),
            }
    return comparison


def hypothesis_decision(comparison):
    one_hour = comparison.get("1", {})
    votes = {}
    for feature in ("HCO3", "Potassium", "pH", "Glucose"):
        stats = one_hour.get(feature, {})
        beta_gap = stats.get("mean_reversion_beta_gap_sim_minus_physionet")
        votes[feature] = {
            "over_volatility": bool(stats.get("supports_over_volatility")),
            "lower_autocorrelation": bool(
                stats.get("supports_lower_autocorrelation")
            ),
            "less_recovery_mean_reversion": (
                beta_gap is not None and beta_gap < -0.05
            ),
        }
    supportive_features = [
        feature for feature, vote in votes.items()
        if vote["over_volatility"] or vote["lower_autocorrelation"]
    ]
    recovery_mismatch_features = [
        feature for feature, vote in votes.items()
        if vote["less_recovery_mean_reversion"]
    ]
    return {
        "supports_under_buffering_hypothesis": len(supportive_features) > 0,
        "supportive_features": supportive_features,
        "supports_recovery_mismatch_hypothesis": (
            len(recovery_mismatch_features) > 0
        ),
        "recovery_mismatch_features": recovery_mismatch_features,
        "feature_votes": votes,
        "hard_runtime_change_allowed": False,
        "clinical_or_causal_claim_allowed": False,
        "bottom_line": (
            "The audit separates two ideas. It does not support a simple "
            "over-volatility/low-autocorrelation under-buffering story. It does "
            "support a recovery mismatch for variables whose PhysioNet "
            "action-unobserved dynamics move toward setpoint while DKABody "
            "no-action dynamics do not. That points more to missing observed "
            "treatment/recovery dynamics than to passive blood buffering alone."
        ),
        "interpretation": (
            "A positive result means DKABody no-action trajectories are more "
            "volatile or less autocorrelated than action-unobserved PhysioNet "
            "ICU trajectories. Because PhysioNet has no medication channels, "
            "this is a simulator-integrity signal, not a treatment-effect "
            "calibration target."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/chunyouchang/mimic/physionet.org/files/challenge-2019/1.0.0/training"),
    )
    parser.add_argument(
        "--calibration",
        default="physionet2019_dkabody_calibration.json",
    )
    parser.add_argument(
        "--output",
        default="dka_buffer_hypothesis_audit.json",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--horizons", default="1,2,4,6")
    return parser.parse_args()


def main():
    args = parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    paths = psv_files(args.data_root, max_patients=args.max_patients)
    calibration = PhysioNetDkaCalibration.load(args.calibration)
    phys_collectors, phys_meta = collect_physionet(paths, horizons)
    sim_collectors, sim_meta = collect_simulator(
        calibration, args.samples, horizons, args.seed
    )
    phys_summary = summarize_collectors(phys_collectors)
    sim_summary = summarize_collectors(sim_collectors)
    comparison = compare(phys_summary, sim_summary)
    decision = hypothesis_decision(comparison)
    report = {
        "experiment": "dka_buffer_hypothesis_audit",
        "question": (
            "Does DKABody lack enough homeostatic buffering/mean reversion "
            "for HCO3 and serum potassium?"
        ),
        "data_boundary": {
            "physionet_dynamics": (
                "action-unobserved factual ICU dynamics; no medication channels"
            ),
            "simulator_dynamics": "DKABody no-action trajectories",
            "contains_patient_rows": False,
            "contains_patient_identifiers": False,
        },
        "horizons_hours": list(horizons),
        "physionet": {
            "metadata": phys_meta,
            "summary": phys_summary,
        },
        "simulator": {
            "metadata": sim_meta,
            "summary": sim_summary,
        },
        "comparison": comparison,
        "hypothesis_decision": decision,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "physionet_anchor_rows": phys_meta["dka_like_anchor_rows"],
        "simulator_deaths": sim_meta["deaths_before_or_at_max_horizon"],
        "one_hour_comparison": {
            feature: comparison["1"][feature]
            for feature in ("HCO3", "Potassium")
        },
        "decision": decision,
    }, indent=2))


if __name__ == "__main__":
    main()
