"""Build aggregate PhysioNet 2019 calibration priors for DKABody.

The output is an aggregate JSON artifact.  It contains no patient rows and no
stay identifiers.  Because PhysioNet Challenge 2019 has no medication channels,
the drift section is explicitly action-unobserved and must not be interpreted as
a clean no-treatment causal estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dka_body import henderson, estimate_potassium_store
from physionet2019_pretrain import DYNAMIC_KEYS, psv_files, read_patient


INDEX = {name: index for index, name in enumerate(DYNAMIC_KEYS)}
REQUIRED_PRESENTATION = ("Glucose", "HCO3", "Potassium", "MAP")
PRESENTATION_FEATURES = (
    "log_glucose", "hco3", "ph", "potassium", "map", "log_creatinine",
)
MEASUREMENT_FEATURES = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "HCO3", "pH",
    "Creatinine", "Glucose", "Lactate", "Potassium", "WBC", "Platelets",
)
DRIFT_FEATURES = ("Glucose", "HCO3", "pH", "Potassium", "MAP", "Creatinine")


def finite(value: float) -> bool:
    return bool(np.isfinite(value))


def row_value(dynamic: np.ndarray, row: int, key: str) -> float:
    return float(dynamic[row, INDEX[key]])


def quantiles(values, digits: int = 6) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {key: None for key in ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")}
    probs = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    names = ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
    return {
        name: round(float(np.quantile(array, prob)), digits)
        for name, prob in zip(names, probs)
    }


def dka_like_rows(dynamic: np.ndarray) -> np.ndarray:
    glucose = dynamic[:, INDEX["Glucose"]]
    hco3 = dynamic[:, INDEX["HCO3"]]
    ph = dynamic[:, INDEX["pH"]]
    acidotic = (np.isfinite(hco3) & (hco3 <= 18.0)) | (
        np.isfinite(ph) & (ph <= 7.30)
    )
    return np.flatnonzero(np.isfinite(glucose) & (glucose >= 200.0) & acidotic)


def presentation_record(dynamic: np.ndarray, row: int) -> dict[str, float] | None:
    values = {}
    for key in REQUIRED_PRESENTATION:
        value = row_value(dynamic, row, key)
        if not finite(value):
            return None
        values[key] = value
    ph = row_value(dynamic, row, "pH")
    values["pH"] = ph if finite(ph) else henderson(values["HCO3"])
    creatinine = row_value(dynamic, row, "Creatinine")
    values["Creatinine"] = creatinine if finite(creatinine) else math.nan
    lactate = row_value(dynamic, row, "Lactate")
    values["Lactate"] = lactate if finite(lactate) else math.nan
    return {
        "glucose": values["Glucose"],
        "hco3": values["HCO3"],
        "ph": values["pH"],
        "potassium": values["Potassium"],
        "map": values["MAP"],
        "creatinine": values["Creatinine"],
        "lactate": values["Lactate"],
    }


def collect_presentations(paths: list[Path]) -> list[dict[str, float]]:
    records = []
    for path in paths:
        patient = read_patient(path)
        dynamic = patient["dynamic"]
        rows = dka_like_rows(dynamic)
        for row in rows:
            record = presentation_record(dynamic, int(row))
            if record is not None:
                records.append(record)
                break
    return records


def presentation_model(records: list[dict[str, float]]) -> dict[str, object]:
    complete = [
        record for record in records
        if finite(record["creatinine"]) and record["creatinine"] > 0
    ]
    if len(complete) < 25:
        raise ValueError("Too few complete DKA-like presentation records")
    matrix = []
    for record in complete:
        matrix.append([
            math.log(max(record["glucose"], 1e-3)),
            record["hco3"],
            record["ph"],
            record["potassium"],
            record["map"],
            math.log(max(record["creatinine"], 0.05)),
        ])
    values = np.asarray(matrix, dtype=np.float64)
    covariance = np.cov(values, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * 1e-6
    raw = defaultdict(list)
    for record in complete:
        for key, value in record.items():
            if finite(value):
                raw[key].append(value)
    return {
        "selection": {
            "criteria": "first per-patient row with Glucose>=200 and (HCO3<=18 or pH<=7.30), requiring Glucose/HCO3/Potassium/MAP",
            "complete_records_used": len(complete),
            "all_presentation_records": len(records),
        },
        "transformed_feature_order": list(PRESENTATION_FEATURES),
        "mean": np.mean(values, axis=0).round(8).tolist(),
        "covariance": covariance.round(8).tolist(),
        "clip_quantiles": {
            key: quantiles(vals) for key, vals in raw.items()
        },
        "clinical_claim_allowed": False,
    }


def patient_variability_model(records: list[dict[str, float]]) -> dict[str, object]:
    proxies = defaultdict(list)
    for record in records:
        glucose = record["glucose"]
        hco3 = record["hco3"]
        ph = record["ph"]
        potassium = record["potassium"]
        map_value = record["map"]
        creatinine = record["creatinine"]
        lactate = record["lactate"]
        if finite(creatinine) and creatinine > 0:
            proxies["baseline_creatinine"].append(creatinine)
            proxies["renal_reserve"].append(float(np.clip(0.9 / creatinine, 0.35, 1.20)))
        proxies["vascular_tone"].append(float(np.clip((map_value - 30.0) / (60.0 * 0.80), 0.60, 1.30)))
        proxies["fluid_retention"].append(float(np.clip((map_value - 30.0) / 60.0, 0.55, 1.05)))
        stress = (
            0.65
            + max(0.0, glucose - 200.0) / 550.0
            + max(0.0, 18.0 - hco3) / 24.0
            + (0.10 * max(0.0, lactate - 2.0) if finite(lactate) else 0.0)
        )
        proxies["counterregulatory_drive"].append(float(np.clip(stress, 0.55, 1.90)))
        store = estimate_potassium_store(
            potassium, ph,
            creatinine if finite(creatinine) else 1.2,
            urine_output=100.0,
        )
        proxies["potassium_store_scale"].append(float(np.clip(store / 120.0, 0.40, 1.25)))
    return {
        "observed_proxy_ranges": {
            key: {
                "source": "aggregate PhysioNet DKA-like presentation proxy",
                "quantiles": quantiles(values),
            }
            for key, values in sorted(proxies.items())
        },
        "unobserved_defaults": {
            "weight_kg": {
                "normal_clip": [78.0, 18.0, 45.0, 135.0],
                "reason": "Challenge 2019 does not include weight.",
            },
            "insulin_sensitivity": {
                "range": [0.50, 1.70],
                "reason": "Not identifiable without explicit insulin exposure.",
            },
            "endogenous_insulin": {
                "range": [0.50, 3.00],
                "reason": "Not measured in Challenge 2019.",
            },
            "baseline_sodium": {
                "range": [132.0, 145.0],
                "reason": "Challenge 2019 does not include sodium.",
            },
        },
        "causal_claim_allowed": False,
    }


def collect_action_unobserved_drift(paths: list[Path], horizons: tuple[int, ...]):
    deltas = {
        str(horizon): {feature: [] for feature in DRIFT_FEATURES}
        for horizon in horizons
    }
    anchors = 0
    for path in paths:
        patient = read_patient(path)
        dynamic = patient["dynamic"]
        rows = dka_like_rows(dynamic)
        for row in rows:
            anchors += 1
            for horizon in horizons:
                future = int(row) + horizon
                if future >= len(dynamic):
                    continue
                for feature in DRIFT_FEATURES:
                    current = row_value(dynamic, int(row), feature)
                    target = row_value(dynamic, future, feature)
                    if finite(current) and finite(target):
                        deltas[str(horizon)][feature].append((target - current) / horizon)
    return {
        "selection": {
            "anchor_criteria": "all DKA-like rows, Glucose>=200 and (HCO3<=18 or pH<=7.30)",
            "anchor_rows": anchors,
            "warning": (
                "Action channels are absent; these are action-unobserved factual "
                "drifts, not clean no-treatment causal drifts."
            ),
        },
        "horizons_hours": list(horizons),
        "delta_per_hour": {
            horizon: {
                feature: {
                    "n": len(values),
                    "mean": round(float(np.mean(values)), 6) if values else None,
                    "median": round(float(np.median(values)), 6) if values else None,
                    "quantiles": quantiles(values),
                }
                for feature, values in feature_map.items()
            }
            for horizon, feature_map in deltas.items()
        },
        "causal_no_treatment_claim_allowed": False,
    }


def measurement_model(paths: list[Path]) -> dict[str, object]:
    rows_seen = 0
    observed = {feature: 0 for feature in MEASUREMENT_FEATURES}
    intervals = {feature: [] for feature in MEASUREMENT_FEATURES}
    consecutive_deltas = {feature: [] for feature in MEASUREMENT_FEATURES}
    for path in paths:
        patient = read_patient(path)
        dynamic = patient["dynamic"]
        rows_seen += dynamic.shape[0]
        for feature in MEASUREMENT_FEATURES:
            series = dynamic[:, INDEX[feature]]
            finite_rows = np.flatnonzero(np.isfinite(series))
            observed[feature] += int(len(finite_rows))
            if len(finite_rows) > 1:
                intervals[feature].extend(np.diff(finite_rows).astype(float).tolist())
                consecutive = finite_rows[np.diff(np.r_[-99, finite_rows]) == 1]
                if len(consecutive) > 1:
                    prev = consecutive - 1
                    mask = np.isfinite(series[prev])
                    consecutive_deltas[feature].extend(
                        (series[consecutive[mask]] - series[prev[mask]]).astype(float).tolist()
                    )
    features = {}
    for feature in MEASUREMENT_FEATURES:
        deltas = np.asarray(consecutive_deltas[feature], dtype=np.float64)
        features[feature] = {
            "observation_rate": round(observed[feature] / max(rows_seen, 1), 6),
            "observed_cells": observed[feature],
            "interval_hours": {
                "n": len(intervals[feature]),
                **quantiles(intervals[feature], digits=3),
            },
            "consecutive_noise_proxy": {
                "n": int(len(deltas)),
                "mad": round(float(np.median(np.abs(deltas - np.median(deltas)))), 6)
                if len(deltas) else None,
                "delta_quantiles": quantiles(deltas),
            },
        }
    return {
        "rows_seen": rows_seen,
        "features": features,
        "intended_use": "observation mask, measurement age, and noise realism",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/chunyouchang/mimic/physionet.org/files/challenge-2019/1.0.0/training"),
    )
    parser.add_argument("--output", type=Path, default=Path("physionet2019_dkabody_calibration.json"))
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    paths = psv_files(args.data_root, max_patients=args.max_patients)
    presentations = collect_presentations(paths)
    payload = {
        "artifact": "physionet2019_dkabody_calibration",
        "version": 1,
        "data_root": str(args.data_root),
        "patients_scanned": len(paths),
        "seed": args.seed,
        "built_at_unix": int(time.time()),
        "presentation_model": presentation_model(presentations),
        "patient_variability_model": patient_variability_model(presentations),
        "action_unobserved_drift_model": collect_action_unobserved_drift(
            paths, horizons=(1, 2, 4, 6)
        ),
        "measurement_model": measurement_model(paths),
        "safety_boundary": {
            "contains_patient_rows": False,
            "contains_patient_identifiers": False,
            "treatment_effect_claim_allowed": False,
            "causal_no_treatment_claim_allowed": False,
            "clinical_claim_allowed": False,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "patients_scanned": payload["patients_scanned"],
        "presentation_records": payload["presentation_model"]["selection"],
        "measurement_rows": payload["measurement_model"]["rows_seen"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
