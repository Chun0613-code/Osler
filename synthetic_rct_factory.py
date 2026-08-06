"""
Synthetic RCT factory for method validation only.

This creates fictional, randomized DKA-like longitudinal data.  It is useful
for testing estimators, power calculations, data contracts, and causal audit
code because the data-generating truth is known.  It is not real patient data,
does not establish a treatment effect, and must never be used as clinical
evidence or to promote a checkpoint.

The generator has four safeguards that the earlier teaching demo did not have:

* baseline moments and missingness are calibrated from a local real-data cohort;
* treatment assignment is balanced within severity strata;
* physiology is generated from a constrained, coupled response model;
* the output carries oracle effects and a randomization audit so recovery can be
  checked without pretending that synthetic recovery is external validation.

Example:
    python synthetic_rct_factory.py \
      --n 100000 \
      --calibration-cohort dka_transitions_6h_mimiciv_full_v31_icd.parquet \
      --output /private/tmp/osler_synthetic_rct_dka_100k.parquet \
      --audit /private/tmp/osler_synthetic_rct_dka_100k_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HORIZONS_H = (1, 3, 6, 12, 24, 48)
ARMS = (
    "standard_protocol",
    "insulin_intensified",
    "insulin_conservative",
    "fluid_intensified",
    "potassium_repletion",
)

# These are simulation knobs, not patient-specific clinical estimates.  The
# calibration cohort controls the starting distribution; these coefficients
# only define a transparent, bounded synthetic response surface.
TARGETS = (
    "glucose",
    "potassium",
    "bicarbonate",
    "anion_gap",
    "pH",
    "sodium",
    "creatinine",
    "MAP",
    "heart_rate",
    "lactate",
    "urine_output",
)

CALIBRATION_COLUMNS = {
    "glucose": "glucose_t",
    "potassium": "potassium_t",
    "bicarbonate": "bicarbonate_t",
    "anion_gap": "anion_gap_t",
    "pH": "ph_t",
    "sodium": "sodium_t",
    "creatinine": "creatinine_t",
    "MAP": "map_t",
    "heart_rate": "heart_rate_t",
    "lactate": "lactate_t",
    "urine_output": "urine_output_t",
}

# Physiologic direction of a latent acute-severity factor.  It creates
# correlated presentations without copying any real patient's row.
SEVERITY_LOADING = {
    "glucose": 0.85,
    "potassium": 0.12,
    "bicarbonate": -0.70,
    "anion_gap": 0.75,
    "pH": -0.65,
    "sodium": 0.15,
    "creatinine": 0.45,
    "MAP": -0.40,
    "heart_rate": 0.55,
    "lactate": 0.55,
    "urine_output": -0.35,
}

CLIP_RANGES = {
    "glucose": (40.0, 1400.0),
    "potassium": (1.8, 8.0),
    "bicarbonate": (2.0, 45.0),
    "anion_gap": (5.0, 55.0),
    "pH": (6.75, 7.65),
    "sodium": (110.0, 175.0),
    "creatinine": (0.2, 12.0),
    "MAP": (35.0, 150.0),
    "heart_rate": (35.0, 220.0),
    "lactate": (0.3, 18.0),
    "urine_output": (0.0, 800.0),
}

# Relative protocol intensities.  Assignment remains randomized; the values
# are deliberately exposed in the output instead of hidden in a black box.
ARM_PROTOCOL = {
    "standard_protocol": {"insulin": 1.00, "fluid": 1.00, "kcl": 1.00},
    "insulin_intensified": {"insulin": 1.25, "fluid": 1.00, "kcl": 1.05},
    "insulin_conservative": {"insulin": 0.75, "fluid": 1.00, "kcl": 0.95},
    "fluid_intensified": {"insulin": 1.00, "fluid": 1.25, "kcl": 1.00},
    "potassium_repletion": {"insulin": 1.00, "fluid": 1.00, "kcl": 1.35},
}


def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def load_calibration(path: str | None, max_rows: int = 250_000) -> dict:
    """Return aggregate moments only; no source identifiers leave this function."""
    defaults = {
        name: {"mean": value, "std": spread, "q01": low, "q99": high, "observed_rate": 1.0}
        for name, (value, spread, low, high) in {
            "glucose": (480.0, 130.0, 100.0, 900.0),
            "potassium": (4.8, 0.9, 2.8, 7.0),
            "bicarbonate": (12.0, 6.0, 3.0, 30.0),
            "anion_gap": (24.0, 9.0, 8.0, 48.0),
            "pH": (7.20, 0.16, 6.85, 7.48),
            "sodium": (137.0, 8.0, 120.0, 155.0),
            "creatinine": (1.5, 1.0, 0.4, 6.0),
            "MAP": (75.0, 18.0, 42.0, 120.0),
            "heart_rate": (105.0, 24.0, 55.0, 180.0),
            "lactate": (2.5, 2.0, 0.5, 10.0),
            "urine_output": (150.0, 120.0, 0.0, 500.0),
        }.items()
    }
    if not path:
        return {"source": "built_in_simulation_priors", "targets": defaults}

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"calibration cohort not found: {source}")
    usecols = [column for column in CALIBRATION_COLUMNS.values() if column]
    frame = pd.read_parquet(source, columns=usecols)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=0)

    stats = {}
    for target, column in CALIBRATION_COLUMNS.items():
        values = _finite_values(frame, column)
        if values.size < 20:
            stats[target] = defaults[target]
            continue
        q01, q99 = np.quantile(values, [0.01, 0.99])
        stats[target] = {
            "mean": float(np.mean(values)),
            "std": float(max(np.std(values), defaults[target]["std"] * 0.25)),
            "q01": float(q01),
            "q99": float(q99),
            "observed_rate": float(values.size / len(frame)),
        }
    return {"source": str(source), "rows_used": int(len(frame)), "targets": stats}


def _balanced_arms(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign arms in a balanced way inside each severity stratum."""
    arms = np.empty(len(strata), dtype=object)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        repeated = np.resize(np.asarray(ARMS, dtype=object), len(indices))
        arms[indices[rng.permutation(len(indices))]] = repeated
    return arms


def _sample_baseline(n: int, calibration: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    names = list(TARGETS)
    stats = calibration["targets"]
    means = np.array([stats[name]["mean"] for name in names], dtype=float)
    scales = np.array([max(stats[name]["std"], 1e-6) for name in names], dtype=float)

    # A one-factor construction is stable when some clinical labs are sparse,
    # while retaining clinically meaningful cross-variable correlation.
    severity = rng.normal(0.0, 1.0, n)
    z = np.column_stack([
        SEVERITY_LOADING[name] * severity
        + np.sqrt(max(0.0, 1.0 - SEVERITY_LOADING[name] ** 2)) * rng.normal(0.0, 1.0, n)
        for name in names
    ])
    values = means + z * scales
    out = {}
    for i, name in enumerate(names):
        low, high = CLIP_RANGES[name]
        cal_low = float(stats[name].get("q01", low))
        cal_high = float(stats[name].get("q99", high))
        out[name] = np.clip(values[:, i], max(low, cal_low), min(high, cal_high))

    # Keep the acid-base panel internally coherent in the synthetic truth.
    out["anion_gap"] = np.clip(
        12.0 + 0.55 * np.maximum(out["glucose"] - 180.0, 0.0) / 50.0
        + 0.55 * np.maximum(24.0 - out["bicarbonate"], 0.0)
        + rng.normal(0.0, 2.0, n),
        *CLIP_RANGES["anion_gap"],
    )
    out["pH"] = np.clip(
        6.10 + np.log10(np.maximum(out["bicarbonate"], 1.0) / (0.03 * 25.0))
        + rng.normal(0.0, 0.015, n),
        *CLIP_RANGES["pH"],
    )
    out["urine_output"] = np.maximum(out["urine_output"], 0.0)
    baseline = pd.DataFrame(out)
    severity_score = (
        0.40 * (baseline["glucose"] - means[names.index("glucose")]) / scales[names.index("glucose")]
        + 0.30 * (baseline["anion_gap"] - means[names.index("anion_gap")]) / scales[names.index("anion_gap")]
        - 0.25 * (baseline["bicarbonate"] - means[names.index("bicarbonate")]) / scales[names.index("bicarbonate")]
        + 0.20 * (baseline["creatinine"] - means[names.index("creatinine")]) / scales[names.index("creatinine")]
    ).to_numpy()
    return baseline, severity_score


def _protocol_effect(
    arm: str,
    target: str,
    h: float,
    severity: np.ndarray,
    patient_insulin_sensitivity: np.ndarray,
    patient_renal_reserve: np.ndarray,
) -> np.ndarray:
    p = ARM_PROTOCOL[arm]
    intensity = p["insulin"]
    fluid = p["fluid"]
    kcl = p["kcl"]
    recovery = 1.0 - np.exp(-h / 8.0)
    fast = 1.0 - np.exp(-h / 4.0)
    slow = 1.0 - np.exp(-h / 18.0)
    untreated = np.clip(severity, -1.5, 2.5)

    if target == "glucose":
        return -115.0 * intensity * patient_insulin_sensitivity * recovery + 12.0 * untreated * slow
    if target == "anion_gap":
        return -9.0 * intensity * patient_insulin_sensitivity * recovery - 1.5 * fluid * slow + 1.6 * untreated * slow
    if target == "bicarbonate":
        return 8.0 * intensity * patient_insulin_sensitivity * recovery + 1.2 * fluid * slow - 1.8 * untreated * slow
    if target == "pH":
        return 0.11 * intensity * patient_insulin_sensitivity * recovery + 0.02 * fluid * slow - 0.025 * untreated * slow
    if target == "potassium":
        return -0.42 * intensity * patient_insulin_sensitivity * fast + 0.75 * (kcl - 1.0) * fast + 0.05 * untreated * slow
    if target == "sodium":
        return 0.8 * (fluid - 1.0) * slow + 0.10 * untreated * slow
    if target == "creatinine":
        return -0.16 * (fluid - 1.0) * patient_renal_reserve * slow + 0.10 * untreated * slow
    if target == "MAP":
        return 8.0 * (fluid - 1.0) * fast - 2.2 * untreated * slow
    if target == "heart_rate":
        return -4.0 * (fluid - 1.0) * fast + 4.0 * untreated * slow
    if target == "lactate":
        return -0.55 * (fluid - 1.0) * patient_renal_reserve * slow + 0.55 * untreated * slow
    if target == "urine_output":
        return 42.0 * (fluid - 1.0) * patient_renal_reserve * fast - 15.0 * untreated * slow
    raise KeyError(target)


def _measurement_noise(target: str, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    relative = {"glucose": 0.035, "potassium": 0.025, "bicarbonate": 0.05, "anion_gap": 0.06,
                "pH": 0.004, "sodium": 0.012, "creatinine": 0.06, "MAP": 0.04,
                "heart_rate": 0.04, "lactate": 0.08, "urine_output": 0.10}[target]
    absolute = {"pH": 0.015, "potassium": 0.06, "MAP": 2.0, "heart_rate": 3.0}
    return rng.normal(0.0, np.maximum(np.abs(truth) * relative, absolute.get(target, 0.0)))


def _clip_target(target: str, values: np.ndarray) -> np.ndarray:
    return np.clip(values, *CLIP_RANGES[target])


def generate(n: int, calibration: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    if n < len(ARMS) * 10:
        raise ValueError(f"n must be at least {len(ARMS) * 10}")
    rng = np.random.default_rng(seed)
    baseline, severity_score = _sample_baseline(n, calibration, rng)
    # These latent traits are fixed per fictional patient and shared across all
    # targets/horizons, so cross-organ response patterns remain coherent.
    patient_insulin_sensitivity = np.exp(rng.normal(0.0, 0.18, n))
    patient_renal_reserve = np.clip(
        1.0 - 0.18 * severity_score + rng.normal(0.0, 0.08, n), 0.45, 1.25
    )
    strata = pd.qcut(severity_score, q=min(5, n // len(ARMS)), labels=False, duplicates="drop")
    arms = _balanced_arms(np.asarray(strata), rng)
    patient_ids = np.array([f"synthetic_rct_{i:08d}" for i in range(n)], dtype=object)

    rows = []
    oracle_effects = {}
    for h in HORIZONS_H:
        for arm in ARMS:
            mask = arms == arm
            count = int(mask.sum())
            if count == 0:
                continue
            local_severity = severity_score[mask]
            for target in TARGETS:
                base = baseline.loc[mask, target].to_numpy(dtype=float)
                effect = _protocol_effect(
                    arm,
                    target,
                    h,
                    local_severity,
                    patient_insulin_sensitivity[mask],
                    patient_renal_reserve[mask],
                )
                truth = _clip_target(target, base + effect)
                # Effects are deliberately target-specific and stochastic at
                # the patient level; oracle columns expose the exact simulated
                # response for validation without hiding it in the model.
                oracle_effects[(h, arm, target)] = effect
                measured = truth + _measurement_noise(target, truth, rng)
                stats = calibration["targets"][target]
                observed_rate = float(np.clip(stats.get("observed_rate", 0.8), 0.35, 1.0))
                observed = rng.random(count) < observed_rate
                measured = np.where(observed, measured, np.nan)
                if target == "pH":
                    measured = np.where(np.isfinite(measured), np.clip(measured, *CLIP_RANGES[target]), np.nan)
                target_frame = pd.DataFrame({
                    "synthetic_patient_id": patient_ids[mask],
                    "randomization_stratum": np.asarray(strata)[mask].astype(int),
                    "arm": arm,
                    "horizon_h": h,
                    "randomization_probability": 1.0 / len(ARMS),
                    "baseline_severity_score": local_severity,
                    "target": target,
                    "baseline_value": base,
                    "true_value": truth,
                    "observed_value": measured,
                    "observed_mask": observed,
                    "oracle_effect_vs_no_protocol": effect,
                })
                rows.append(target_frame)
    data = pd.concat(rows, ignore_index=True)

    # Expose protocol inputs once per row so this is directly usable by a
    # factual transition contract without pretending an action was observed.
    for key in ("insulin", "fluid", "kcl"):
        data[f"protocol_{key}_multiplier"] = data["arm"].map(lambda arm: ARM_PROTOCOL[arm][key]).astype(float)
    data["act_insulin_iv_rate_u_hr"] = 6.0 * data["protocol_insulin_multiplier"]
    data["act_fluids_rate_ml_hr"] = 150.0 * data["protocol_fluid_multiplier"]
    data["act_kcl_rate_meq_hr"] = 10.0 * data["protocol_kcl_multiplier"]
    data["synthetic_rct"] = True
    data["fictional_data"] = True
    data["not_real_patient_data"] = True
    data["causal_claim_allowed"] = False
    data["simulator_version"] = "osler_synthetic_rct_factory_v1"
    data["seed"] = int(seed)

    metadata = {
        "synthetic_rct": True,
        "fictional_data": True,
        "not_real_patient_data": True,
        "causal_claim_allowed": False,
        "clinical_use_allowed": False,
        "purpose": ["estimator validation", "power analysis", "data-contract testing"],
        "arms": list(ARMS),
        "horizons_h": list(HORIZONS_H),
        "targets": list(TARGETS),
        "n_patients": int(n),
        "n_rows": int(len(data)),
        "seed": int(seed),
        "calibration": calibration,
        "oracle_is_simulator_truth": True,
        "warning": "Synthetic recovery cannot validate a clinical treatment effect or replace randomized human data.",
    }
    return data, metadata


def _standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0)
    return float((np.mean(left) - np.mean(right)) / pooled) if pooled > 1e-12 else 0.0


def audit(data: pd.DataFrame, metadata: dict) -> dict:
    baseline = data[data["horizon_h"] == HORIZONS_H[0]]
    arm_counts = data.groupby("arm")["synthetic_patient_id"].nunique().to_dict()
    balance = {}
    for target in TARGETS:
        values = baseline.loc[baseline["target"] == target]
        if values.empty:
            continue
        standard = values.loc[values["arm"] == "standard_protocol", "baseline_value"].to_numpy(float)
        balance[target] = {
            arm: _standardized_mean_difference(
                values.loc[values["arm"] == arm, "baseline_value"].to_numpy(float), standard
            )
            for arm in ARMS
            if arm != "standard_protocol"
        }
    max_abs_smd = max(
        (abs(value) for target in balance.values() for value in target.values() if np.isfinite(value)),
        default=float("nan"),
    )

    recovery = {}
    for target in TARGETS:
        frame = data[(data["horizon_h"] == 6) & (data["target"] == target)]
        standard = frame.loc[frame["arm"] == "standard_protocol", "true_value"].mean()
        target_report = {}
        for arm in ARMS:
            if arm == "standard_protocol":
                continue
            arm_frame = frame[frame["arm"] == arm]
            estimated = float(arm_frame["true_value"].mean() - standard)
            oracle = float(arm_frame["oracle_effect_vs_no_protocol"].mean()
                           - frame.loc[frame["arm"] == "standard_protocol", "oracle_effect_vs_no_protocol"].mean())
            target_report[arm] = {
                "estimated_mean_difference_vs_standard": estimated,
                "oracle_mean_difference_vs_standard": oracle,
                "absolute_recovery_error": abs(estimated - oracle),
            }
        recovery[target] = target_report

    pass_balance = bool(np.isfinite(max_abs_smd) and max_abs_smd < 0.10)
    return {
        "metadata": metadata,
        "arm_counts": {str(k): int(v) for k, v in arm_counts.items()},
        "balance": balance,
        "max_absolute_baseline_smd": None if not np.isfinite(max_abs_smd) else float(max_abs_smd),
        "randomization_balance_pass": pass_balance,
        "known_effect_recovery_at_6h": recovery,
        "audit_boundary": {
            "synthetic_only": True,
            "causal_claim_allowed": False,
            "clinical_use_allowed": False,
            "interpretation": "Recovery checks whether estimators can recover simulator truth; it is not external validation.",
        },
    }


def write_frame(data: pd.DataFrame, output: str) -> str:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        data.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        data.to_csv(path, index=False)
    else:
        raise ValueError("output must end in .parquet or .csv")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="number of fictional patients")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--calibration-cohort", default=None, help="local parquet used only for aggregate baseline moments")
    parser.add_argument("--output", default="/private/tmp/osler_synthetic_rct_dka.parquet")
    parser.add_argument("--audit", default="/private/tmp/osler_synthetic_rct_dka_audit.json")
    args = parser.parse_args()

    calibration = load_calibration(args.calibration_cohort)
    data, metadata = generate(args.n, calibration, args.seed)
    output = write_frame(data, args.output)
    report = audit(data, metadata)
    report["output"] = output
    report["audit_path"] = str(Path(args.audit))
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "output": output,
        "audit": args.audit,
        "n_patients": args.n,
        "n_rows": len(data),
        "randomization_balance_pass": report["randomization_balance_pass"],
        "max_absolute_baseline_smd": report["max_absolute_baseline_smd"],
        "causal_claim_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
