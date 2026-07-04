"""Pretrain the ICU JEPA on the PhysioNet eICU demo CSV release.

The eICU demo is table based, not PhysioNet Challenge 2019 PSV.  This adapter
maps eICU vital/lab rows into the existing ICU JEPA feature contract, builds
hourly patient-level transitions, and trains the same generic state-dynamics
model used by ``physionet2019_pretrain.py``.

The auxiliary binary head uses hospital/unit discharge mortality as a demo
proxy because the eICU demo does not include the Challenge 2019 sepsis label.
That head is not a clinical mortality model; the intended signal here is still
general ICU state-dynamics pretraining.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physionet2019_pretrain import (
    DYNAMIC_KEYS,
    STATIC_KEYS,
    MAX_OBSERVATION_AGE_HOURS,
    Normalizer,
    PhysioNetICUJepa,
    TransitionDataset,
    carry_forward,
    choose_device,
    evaluate_model,
    save_checkpoint,
    train_model,
)


ID = "patientunitstayid"
HOUR = "hour"
RISK_KEY = "MortalityProxy"

VITAL_PERIODIC_MAP = {
    "heartrate": "HR",
    "sao2": "O2Sat",
    "temperature": "Temp",
    "respiration": "Resp",
    "etco2": "EtCO2",
    "systemicsystolic": "SBP",
    "systemicdiastolic": "DBP",
    "systemicmean": "MAP",
}

VITAL_APERIODIC_MAP = {
    "noninvasivesystolic": "SBP",
    "noninvasivediastolic": "DBP",
    "noninvasivemean": "MAP",
}

LAB_MAP = {
    "Base Excess": "BaseExcess",
    "HCO3": "HCO3",
    "bicarbonate": "HCO3",
    "FiO2": "FiO2",
    "pH": "pH",
    "paCO2": "PaCO2",
    "O2 Sat (%)": "SaO2",
    "AST (SGOT)": "AST",
    "BUN": "BUN",
    "alkaline phos.": "Alkalinephos",
    "calcium": "Calcium",
    "ionized calcium": "Calcium",
    "chloride": "Chloride",
    "creatinine": "Creatinine",
    "direct bilirubin": "Bilirubin_direct",
    "bedside glucose": "Glucose",
    "glucose": "Glucose",
    "lactate": "Lactate",
    "magnesium": "Magnesium",
    "phosphate": "Phosphate",
    "potassium": "Potassium",
    "total bilirubin": "Bilirubin_total",
    "troponin - I": "TroponinI",
    "Hct": "Hct",
    "Hgb": "Hgb",
    "PTT": "PTT",
    "WBC x 1000": "WBC",
    "fibrinogen": "Fibrinogen",
    "platelets x 1000": "Platelets",
}

PLAUSIBLE_RANGES = {
    "HR": (20.0, 250.0),
    "O2Sat": (40.0, 100.0),
    "Temp": (25.0, 45.0),
    "SBP": (30.0, 300.0),
    "MAP": (20.0, 250.0),
    "DBP": (10.0, 200.0),
    "Resp": (1.0, 80.0),
    "EtCO2": (5.0, 100.0),
    "BaseExcess": (-40.0, 40.0),
    "HCO3": (2.0, 60.0),
    "FiO2": (0.0, 100.0),
    "pH": (6.5, 8.1),
    "PaCO2": (5.0, 200.0),
    "SaO2": (40.0, 100.0),
    "AST": (0.0, 20000.0),
    "BUN": (1.0, 300.0),
    "Alkalinephos": (0.0, 5000.0),
    "Calcium": (2.0, 20.0),
    "Chloride": (50.0, 150.0),
    "Creatinine": (0.05, 30.0),
    "Bilirubin_direct": (0.0, 80.0),
    "Glucose": (10.0, 2000.0),
    "Lactate": (0.1, 30.0),
    "Magnesium": (0.2, 10.0),
    "Phosphate": (0.2, 20.0),
    "Potassium": (1.0, 10.0),
    "Bilirubin_total": (0.0, 80.0),
    "TroponinI": (0.0, 500.0),
    "Hct": (5.0, 75.0),
    "Hgb": (2.0, 25.0),
    "PTT": (5.0, 250.0),
    "WBC": (0.1, 300.0),
    "Fibrinogen": (20.0, 1500.0),
    "Platelets": (1.0, 2000.0),
}


@dataclass
class EicuSequence:
    dynamic: np.ndarray
    static: np.ndarray
    risk: np.ndarray
    time: np.ndarray


def parse_age(value: object) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value).strip()
    if text == "> 89":
        return 90.0
    try:
        return float(text)
    except ValueError:
        return math.nan


def clean_feature_values(feature: str, values: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce").astype("float32")
    if feature == "Temp":
        fahrenheit = cleaned > 60.0
        cleaned.loc[fahrenheit] = (cleaned.loc[fahrenheit] - 32.0) * (5.0 / 9.0)
    if feature == "FiO2":
        fractional = (cleaned > 0.0) & (cleaned <= 1.0)
        cleaned.loc[fractional] = cleaned.loc[fractional] * 100.0
    lower, upper = PLAUSIBLE_RANGES[feature]
    cleaned = cleaned.where((cleaned >= lower) & (cleaned <= upper))
    return cleaned


def add_hour_column(frame: pd.DataFrame, offset_column: str, max_hours_per_stay: int) -> pd.DataFrame:
    frame = frame.copy()
    frame[HOUR] = np.floor(pd.to_numeric(frame[offset_column], errors="coerce") / 60.0)
    frame = frame[np.isfinite(frame[HOUR])]
    frame[HOUR] = frame[HOUR].astype("int32")
    frame = frame[(frame[HOUR] >= 0) & (frame[HOUR] <= max_hours_per_stay)]
    return frame


def read_vital_periodic(data_root: Path, max_hours_per_stay: int) -> pd.DataFrame:
    columns = [ID, "observationoffset", *VITAL_PERIODIC_MAP.keys()]
    frame = pd.read_csv(data_root / "vitalPeriodic.csv.gz", usecols=columns)
    frame = add_hour_column(frame, "observationoffset", max_hours_per_stay)
    for source, feature in VITAL_PERIODIC_MAP.items():
        frame[source] = clean_feature_values(feature, frame[source])
    grouped = frame.groupby([ID, HOUR], sort=False)[list(VITAL_PERIODIC_MAP)].median()
    grouped = grouped.rename(columns=VITAL_PERIODIC_MAP)
    return grouped


def read_vital_aperiodic(data_root: Path, max_hours_per_stay: int) -> pd.DataFrame:
    columns = [ID, "observationoffset", *VITAL_APERIODIC_MAP.keys()]
    frame = pd.read_csv(data_root / "vitalAperiodic.csv.gz", usecols=columns)
    frame = add_hour_column(frame, "observationoffset", max_hours_per_stay)
    for source, feature in VITAL_APERIODIC_MAP.items():
        frame[source] = clean_feature_values(feature, frame[source])
    grouped = frame.groupby([ID, HOUR], sort=False)[list(VITAL_APERIODIC_MAP)].median()
    grouped = grouped.rename(columns=VITAL_APERIODIC_MAP)
    return grouped


def read_labs(data_root: Path, max_hours_per_stay: int) -> pd.DataFrame:
    columns = [ID, "labresultoffset", "labname", "labresult"]
    frame = pd.read_csv(data_root / "lab.csv.gz", usecols=columns)
    frame = frame[frame["labname"].isin(LAB_MAP)]
    frame = add_hour_column(frame, "labresultoffset", max_hours_per_stay)
    frame["feature"] = frame["labname"].map(LAB_MAP)
    frame["value"] = pd.to_numeric(frame["labresult"], errors="coerce").astype("float32")
    for feature in sorted(set(LAB_MAP.values())):
        mask = frame["feature"] == feature
        if mask.any():
            frame.loc[mask, "value"] = clean_feature_values(feature, frame.loc[mask, "value"])
    frame = frame.dropna(subset=["value"])
    grouped = frame.groupby([ID, HOUR, "feature"], sort=False)["value"].median()
    return grouped.unstack("feature")


def static_features(patient: pd.Series) -> np.ndarray:
    gender = math.nan
    if patient["gender"] == "Male":
        gender = 1.0
    elif patient["gender"] == "Female":
        gender = 0.0

    unit_type = str(patient["unittype"]) if not pd.isna(patient["unittype"]) else ""
    unit1 = 1.0 if "MICU" in unit_type or "Med-Surg" in unit_type else 0.0
    unit2 = 1.0 if "SICU" in unit_type or "CTICU" in unit_type else 0.0
    hospital_admit_hours = pd.to_numeric(patient["hospitaladmitoffset"], errors="coerce") / 60.0
    return np.asarray(
        [parse_age(patient["age"]), gender, unit1, unit2, hospital_admit_hours],
        dtype=np.float32,
    )


def mortality_proxy(patient: pd.Series) -> float:
    unit_expired = patient.get("unitdischargestatus") == "Expired"
    hospital_expired = patient.get("hospitaldischargestatus") == "Expired"
    return float(unit_expired or hospital_expired)


def split_patient_ids(patient_ids: list[int], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patient_ids))
    train_end = int(0.70 * len(order))
    validation_end = int(0.85 * len(order))
    return {
        "train": [patient_ids[int(i)] for i in order[:train_end]],
        "validation": [patient_ids[int(i)] for i in order[train_end:validation_end]],
        "test": [patient_ids[int(i)] for i in order[validation_end:]],
    }


def load_patient_table(data_root: Path, max_stays: int | None) -> pd.DataFrame:
    columns = [
        ID,
        "gender",
        "age",
        "hospitaladmitoffset",
        "hospitaldischargestatus",
        "unitdischargestatus",
        "unittype",
    ]
    patients = pd.read_csv(data_root / "patient.csv.gz", usecols=columns)
    patients = patients.sort_values(ID).drop_duplicates(ID)
    if max_stays is not None:
        patients = patients.head(max_stays)
    return patients.set_index(ID, drop=False)


def build_sequences(
    data_root: Path,
    max_stays: int | None,
    max_hours_per_stay: int,
) -> tuple[dict[int, EicuSequence], dict[str, object]]:
    patients = load_patient_table(data_root, max_stays)
    patient_ids = set(int(value) for value in patients.index)

    periodic = read_vital_periodic(data_root, max_hours_per_stay)
    aperiodic = read_vital_aperiodic(data_root, max_hours_per_stay)
    labs = read_labs(data_root, max_hours_per_stay)
    observations = periodic.combine_first(aperiodic).combine_first(labs)
    observations = observations[[key for key in DYNAMIC_KEYS if key in observations.columns]]
    observations = observations[observations.index.get_level_values(ID).isin(patient_ids)]

    sequences: dict[int, EicuSequence] = {}
    used_hours = []
    for patient_id, group in observations.groupby(level=ID, sort=False):
        patient_id = int(patient_id)
        if patient_id not in patients.index:
            continue
        hourly = group.droplevel(ID).sort_index()
        max_hour = int(hourly.index.max())
        if max_hour <= 0:
            continue
        hours = np.arange(max_hour + 1, dtype=np.float32)
        hourly = hourly.reindex(range(max_hour + 1))
        dynamic = np.full((len(hourly), len(DYNAMIC_KEYS)), np.nan, dtype=np.float32)
        for index, key in enumerate(DYNAMIC_KEYS):
            if key in hourly:
                dynamic[:, index] = hourly[key].to_numpy(dtype=np.float32)
        patient = patients.loc[patient_id]
        risk = np.full(len(hourly), mortality_proxy(patient), dtype=np.float32)
        sequences[patient_id] = EicuSequence(
            dynamic=dynamic,
            static=static_features(patient),
            risk=risk,
            time=hours,
        )
        used_hours.append(max_hour + 1)

    mapped_target_counts = {
        key: int(np.isfinite(np.concatenate([seq.dynamic[:, i] for seq in sequences.values()])).sum())
        for i, key in enumerate(DYNAMIC_KEYS)
    }
    summary = {
        "stays_in_patient_table": int(len(patients)),
        "stays_with_observations": int(len(sequences)),
        "max_hours_per_stay": max_hours_per_stay,
        "median_hours_per_used_stay": round(float(np.median(used_hours)), 3) if used_hours else 0.0,
        "risk_target": RISK_KEY,
        "risk_positive_rate_by_stay": round(
            float(np.mean([seq.risk[0] for seq in sequences.values()])), 6
        ) if sequences else 0.0,
        "dynamic_observed_values": mapped_target_counts,
        "dynamic_features_with_any_data": [
            key for key, count in mapped_target_counts.items() if count > 0
        ],
    }
    return sequences, summary


def fit_sequence_normalizer(sequences: dict[int, EicuSequence], patient_ids: list[int]) -> Normalizer:
    dyn_sum = np.zeros(len(DYNAMIC_KEYS), dtype=np.float64)
    dyn_sum_sq = np.zeros(len(DYNAMIC_KEYS), dtype=np.float64)
    dyn_count = np.zeros(len(DYNAMIC_KEYS), dtype=np.float64)
    static_sum = np.zeros(len(STATIC_KEYS), dtype=np.float64)
    static_sum_sq = np.zeros(len(STATIC_KEYS), dtype=np.float64)
    static_count = np.zeros(len(STATIC_KEYS), dtype=np.float64)

    for patient_id in patient_ids:
        seq = sequences[patient_id]
        dynamic = seq.dynamic
        mask = np.isfinite(dynamic)
        dyn_sum += np.nan_to_num(dynamic, nan=0.0).sum(axis=0)
        dyn_sum_sq += np.nan_to_num(dynamic * dynamic, nan=0.0).sum(axis=0)
        dyn_count += mask.sum(axis=0)

        static = seq.static
        static_mask = np.isfinite(static)
        static_sum += np.nan_to_num(static, nan=0.0)
        static_sum_sq += np.nan_to_num(static * static, nan=0.0)
        static_count += static_mask

    dyn_count = np.maximum(dyn_count, 1.0)
    static_count = np.maximum(static_count, 1.0)
    dyn_mean = dyn_sum / dyn_count
    static_mean = static_sum / static_count
    dyn_var = np.maximum(dyn_sum_sq / dyn_count - dyn_mean * dyn_mean, 1e-6)
    static_var = np.maximum(static_sum_sq / static_count - static_mean * static_mean, 1e-6)
    return Normalizer(
        dyn_mean.astype(np.float32),
        np.sqrt(dyn_var).astype(np.float32),
        static_mean.astype(np.float32),
        np.sqrt(static_var).astype(np.float32),
    )


def sequence_transitions(
    seq: EicuSequence,
    normalizer: Normalizer,
    horizon_hours: int,
) -> dict[str, np.ndarray] | None:
    if len(seq.time) <= horizon_hours:
        return None
    values, mask, age = carry_forward(seq.dynamic, normalizer.dynamic_mean)
    static_values = seq.static.copy()
    static_mask = np.isfinite(static_values).astype(np.float32)
    static_values = np.where(np.isfinite(static_values), static_values, normalizer.static_mean)

    x_values, y_values = values[:-horizon_hours], values[horizon_hours:]
    x_mask, y_mask = mask[:-horizon_hours], mask[horizon_hours:]
    x_age, y_age = age[:-horizon_hours], age[horizon_hours:]
    x_time, y_time = seq.time[:-horizon_hours], seq.time[horizon_hours:]
    delta = np.clip(y_time - x_time, 1.0, float(horizon_hours)).astype(np.float32)
    static = np.repeat(static_values[None, :], len(x_values), axis=0)
    static_observed = np.repeat(static_mask[None, :], len(x_values), axis=0)
    risk = seq.risk[horizon_hours:].astype(np.float32)

    return {
        "x": ((x_values - normalizer.dynamic_mean) / normalizer.dynamic_std).astype(np.float32),
        "x_mask": x_mask.astype(np.float32),
        "x_age": (x_age / MAX_OBSERVATION_AGE_HOURS).astype(np.float32),
        "y": ((y_values - normalizer.dynamic_mean) / normalizer.dynamic_std).astype(np.float32),
        "y_mask": y_mask.astype(np.float32),
        "y_age": (y_age / MAX_OBSERVATION_AGE_HOURS).astype(np.float32),
        "static": ((static - normalizer.static_mean) / normalizer.static_std).astype(np.float32),
        "static_mask": static_observed.astype(np.float32),
        "delta": (delta / max(float(horizon_hours), 1.0)).reshape(-1, 1).astype(np.float32),
        "sepsis": risk.reshape(-1, 1),
    }


def build_split_arrays(
    sequences: dict[int, EicuSequence],
    patient_ids: list[int],
    normalizer: Normalizer,
    horizon_hours: int,
    max_transitions: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "x": [], "x_mask": [], "x_age": [], "y": [], "y_mask": [], "y_age": [],
        "static": [], "static_mask": [], "delta": [], "sepsis": [],
    }
    for patient_id in patient_ids:
        transition = sequence_transitions(sequences[patient_id], normalizer, horizon_hours)
        if transition is None:
            continue
        for key in parts:
            parts[key].append(transition[key])

    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items() if values}
    if not arrays:
        raise ValueError("No transitions were built; lower --horizon-hours or check the eICU data.")
    if max_transitions is not None and arrays["x"].shape[0] > max_transitions:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(arrays["x"].shape[0], size=max_transitions, replace=False))
        arrays = {key: value[selected] for key, value in arrays.items()}
    return arrays


def split_summary(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    target_counts = arrays["y_mask"].sum(axis=0).astype(int)
    return {
        "transitions": int(arrays["x"].shape[0]),
        "direct_observation_density": round(float(arrays["x_mask"].mean()), 6),
        "future_target_density": round(float(arrays["y_mask"].mean()), 6),
        "mortality_proxy_positive_rate": round(float(arrays["sepsis"].mean()), 6),
        "features_with_future_targets": int((target_counts > 0).sum()),
        "future_target_counts": {
            key: int(target_counts[index])
            for index, key in enumerate(DYNAMIC_KEYS)
            if int(target_counts[index]) > 0
        },
    }


def build_cache(
    data_root: Path,
    cache_path: Path,
    max_stays: int | None,
    max_hours_per_stay: int,
    max_transitions_per_split: int | None,
    horizon_hours: int,
    seed: int,
) -> dict[str, object]:
    sequences, data_summary = build_sequences(data_root, max_stays, max_hours_per_stay)
    patient_ids = sorted(sequences)
    splits = split_patient_ids(patient_ids, seed)
    normalizer = fit_sequence_normalizer(sequences, splits["train"])

    arrays = {}
    split_summaries = {}
    for split, split_ids in splits.items():
        arrays[split] = build_split_arrays(
            sequences,
            split_ids,
            normalizer,
            horizon_hours,
            max_transitions_per_split,
            seed + len(split),
        )
        split_summaries[split] = {
            "stays": int(len(split_ids)),
            **split_summary(arrays[split]),
        }

    payload = {
        "normalizer_json": json.dumps(normalizer.to_json()),
        "eicu_summary_json": json.dumps(data_summary),
        "horizon_hours": np.array([horizon_hours], dtype=np.float32),
    }
    for split, split_arrays in arrays.items():
        for key, value in split_arrays.items():
            payload[f"{split}_{key}"] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return cache_summary(cache_path)


def cache_summary(cache_path: Path) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as data:
        normalizer = json.loads(str(data["normalizer_json"]))
        data_summary = json.loads(str(data["eicu_summary_json"]))
        splits = {}
        for split in ("train", "validation", "test"):
            arrays = {
                key: data[f"{split}_{key}"]
                for key in (
                    "x", "x_mask", "y_mask", "sepsis",
                )
            }
            target_counts = arrays["y_mask"].sum(axis=0).astype(int)
            splits[split] = {
                "transitions": int(arrays["x"].shape[0]),
                "direct_observation_density": round(float(arrays["x_mask"].mean()), 6),
                "future_target_density": round(float(arrays["y_mask"].mean()), 6),
                "mortality_proxy_positive_rate": round(float(arrays["sepsis"].mean()), 6),
                "features_with_future_targets": int((target_counts > 0).sum()),
            }
    return {
        "cache": str(cache_path),
        "source": "PhysioNet eICU Collaborative Research Database Demo 2.0.1",
        "dynamic_features": normalizer["dynamic_keys"],
        "static_features": normalizer["static_keys"],
        "eicu_adapter": data_summary,
        "splits": splits,
        "auxiliary_risk_target": RISK_KEY,
        "intervention_claim_allowed": False,
        "counterfactual_claim_allowed": False,
        "intended_use": "general ICU state-dynamics pretraining only",
    }


@torch.no_grad()
def evaluate_risk_head(
    model: PhysioNetICUJepa,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    logits = []
    targets = []
    model.eval()
    for batch in loader:
        x, x_mask, x_age, _y, _y_mask, _y_age, static, static_mask, delta, risk = (
            tensor.to(device) for tensor in batch
        )
        _pred, risk_logit, _latent, _future = model(x, x_mask, x_age, static, static_mask, delta)
        logits.append(risk_logit.detach().cpu().numpy().reshape(-1))
        targets.append(risk.detach().cpu().numpy().reshape(-1))

    y_score = np.concatenate(logits)
    y_true = np.concatenate(targets)
    positive = int(y_true.sum())
    negative = int(len(y_true) - positive)
    probs = 1.0 / (1.0 + np.exp(-np.clip(y_score, -60.0, 60.0)))
    eps = 1e-7
    bce = -np.mean(y_true * np.log(probs + eps) + (1.0 - y_true) * np.log(1.0 - probs + eps))
    auc = None
    if positive > 0 and negative > 0:
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
        auc = (ranks[y_true == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative)
    return {
        "target": RISK_KEY,
        "positive_rate": round(float(y_true.mean()), 6),
        "binary_cross_entropy": round(float(bce), 6),
        "auc_rank_estimate": None if auc is None else round(float(auc), 6),
        "positive_transitions": positive,
        "negative_transitions": negative,
        "clinical_claim_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("physionet.org/files/eicu-crd-demo/2.0.1"),
    )
    parser.add_argument("--cache", type=Path, default=Path("eicu_demo_cache.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("eicu_demo_icu_jepa.pt"))
    parser.add_argument("--report", type=Path, default=Path("eicu_demo_icu_jepa_report.json"))
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--max-hours-per-stay", type=int, default=168)
    parser.add_argument("--max-transitions-per-split", type=int, default=None)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.rebuild_cache or not args.cache.exists():
        summary = build_cache(
            args.data_root,
            args.cache,
            args.max_stays,
            args.max_hours_per_stay,
            args.max_transitions_per_split,
            args.horizon_hours,
            args.seed,
        )
    else:
        summary = cache_summary(args.cache)
    if args.profile_only:
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    cache = np.load(args.cache, allow_pickle=False)
    normalizer = json.loads(str(cache["normalizer_json"]))
    train_set = TransitionDataset(cache, "train")
    validation_set = TransitionDataset(cache, "validation")
    test_set = TransitionDataset(cache, "test")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False
    )
    device = choose_device(args.device)
    model = PhysioNetICUJepa(len(DYNAMIC_KEYS), len(STATIC_KEYS))
    history, _ = train_model(
        model,
        train_loader,
        validation_loader,
        args.epochs,
        args.learning_rate,
        device,
    )
    validation_metrics = evaluate_model(model, validation_loader, normalizer, device)
    test_metrics = evaluate_model(model, test_loader, normalizer, device)
    validation_metrics["auxiliary_risk"] = evaluate_risk_head(model, validation_loader, device)
    test_metrics["auxiliary_risk"] = evaluate_risk_head(model, test_loader, device)
    metadata = {
        "data_root": str(args.data_root),
        "cache": str(args.cache),
        "stays_seen_cap": args.max_stays,
        "max_hours_per_stay": args.max_hours_per_stay,
        "max_transitions_per_split": args.max_transitions_per_split,
        "horizon_hours": args.horizon_hours,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start, 3),
        "auxiliary_risk_target": RISK_KEY,
    }
    save_checkpoint(model, args.checkpoint, normalizer, metadata)
    report = {
        "dataset": summary,
        "metadata": metadata,
        "training_history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "promotion_boundary": {
            "may_initialize_or_pretrain_generic_encoder": True,
            "may_replace_dka_intervention_checkpoint": False,
            "reason": (
                "This generic pretraining run does not consume eICU treatment "
                "tables and the demo has no Challenge 2019 sepsis label; it "
                "only checks ICU state dynamics."
            ),
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "report": str(args.report),
        "test_masked_state_mse": report["test"]["masked_state_mse"],
        "collapsed": report["test"]["latent_collapse"]["collapsed"],
        "risk_target": RISK_KEY,
        "risk_auc": report["test"]["auxiliary_risk"]["auc_rank_estimate"],
    }, indent=2))


if __name__ == "__main__":
    main()
