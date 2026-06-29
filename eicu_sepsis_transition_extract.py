"""Build a full-eICU sepsis transition cohort with observed treatment evidence.

This is the Chapter-B sepsis starter adapter.  It intentionally mirrors the
DKA transition contract, but it does not use a mechanistic sepsis simulator and
does not estimate treatment effects.  The output is a factual observed-treatment
cohort:

    state_t + observed treatment evidence in [t, t+6h] + state_t+6h

Medication orders and coarse treatment rows are evidence only.  They are never
promoted to causal treatment effects or clinical recommendations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_demo_pretrain import (
    VITAL_APERIODIC_MAP,
    VITAL_PERIODIC_MAP,
    clean_feature_values,
)


ID = "patientunitstayid"
BASE_TIME = pd.Timestamp("2100-01-01")
DELTA_H = 6.0
ANCHOR_STEP_H = 4.0
LOOKBACK_H = 6.0
TARGET_TOL_H = 2.0
EPISODE_MAX_H = 96.0

SEPSIS_PATTERNS = (
    "sepsis",
    "septic",
    "septicemia",
    "septic shock",
    "severe sepsis",
    "995.91",
    "995.92",
    "785.52",
    "r65.2",
    "a41",
)

LAB_TO_STATE = {
    "lactate": "lactate",
    "creatinine": "creatinine",
    "WBC x 1000": "wbc",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "total bilirubin": "bilirubin",
    "direct bilirubin": "bilirubin_direct",
    "platelets x 1000": "platelets",
    "PT - INR": "inr",
    "PTT": "ptt",
    "fibrinogen": "fibrinogen",
    "HCO3": "bicarbonate",
    "bicarbonate": "bicarbonate",
    "pH": "ph",
    "anion gap": "anion_gap",
    "chloride": "chloride",
    "potassium": "potassium",
    "sodium": "sodium",
    "calcium": "calcium",
    "ionized calcium": "ionized_calcium",
    "magnesium": "magnesium",
    "phosphate": "phosphate",
    "BUN": "bun",
    "bedside glucose": "glucose",
    "glucose": "glucose",
    "albumin": "albumin",
    "prealbumin": "prealbumin",
    "total protein": "total_protein",
    "AST (SGOT)": "ast",
    "ALT (SGPT)": "alt",
    "alkaline phos.": "alkaline_phos",
    "lipase": "lipase",
    "amylase": "amylase",
    "triglycerides": "triglycerides",
    "troponin - I": "troponin_i",
    "troponin - T": "troponin_t",
    "BNP": "bnp",
    "CPK": "cpk",
    "CPK-MB": "ck_mb",
    "CPK-MB INDEX": "ck_mb_index",
    "LDH": "ldh",
    "myoglobin": "myoglobin",
    "serum osmolality": "serum_osmolality",
    "serum ketones": "serum_ketones",
    "TSH": "tsh",
    "free T4": "free_t4",
    "cortisol": "cortisol",
    "CRP": "crp",
    "CRP-hs": "crp_hs",
    "ESR": "esr",
    "Ferritin": "ferritin",
    "uric acid": "uric_acid",
}

VITAL_TO_STATE = {
    "heartrate": "heart_rate",
    "respiration": "respiratory_rate",
    "sao2": "o2sat",
    "temperature": "temperature",
    "systemicmean": "map",
    "noninvasivemean": "map",
}

PLAUSIBLE = {
    "map": (20.0, 250.0),
    "lactate": (0.1, 40.0),
    "creatinine": (0.05, 30.0),
    "urine_output": (0.0, 2000.0),
    "o2sat": (40.0, 100.0),
    "heart_rate": (20.0, 250.0),
    "respiratory_rate": (1.0, 80.0),
    "temperature": (25.0, 45.0),
    "wbc": (0.1, 300.0),
    "hemoglobin": (1.0, 25.0),
    "hematocrit": (3.0, 75.0),
    "bilirubin": (0.0, 80.0),
    "bilirubin_direct": (0.0, 80.0),
    "platelets": (1.0, 2000.0),
    "inr": (0.4, 20.0),
    "ptt": (5.0, 250.0),
    "fibrinogen": (20.0, 1500.0),
    "bicarbonate": (1.0, 60.0),
    "ph": (6.5, 8.1),
    "anion_gap": (0.0, 80.0),
    "chloride": (60.0, 160.0),
    "sodium": (100.0, 180.0),
    "potassium": (1.0, 10.0),
    "calcium": (2.0, 20.0),
    "ionized_calcium": (0.2, 4.0),
    "magnesium": (0.2, 12.0),
    "phosphate": (0.1, 20.0),
    "bun": (1.0, 300.0),
    "glucose": (10.0, 2000.0),
    "albumin": (0.1, 8.0),
    "prealbumin": (0.1, 100.0),
    "total_protein": (0.1, 15.0),
    "ast": (0.0, 20000.0),
    "alt": (0.0, 20000.0),
    "alkaline_phos": (0.0, 5000.0),
    "lipase": (0.0, 50000.0),
    "amylase": (0.0, 50000.0),
    "triglycerides": (1.0, 10000.0),
    "troponin_i": (0.0, 500.0),
    "troponin_t": (0.0, 500.0),
    "bnp": (0.0, 100000.0),
    "cpk": (0.0, 200000.0),
    "ck_mb": (0.0, 5000.0),
    "ck_mb_index": (0.0, 100.0),
    "ldh": (0.0, 50000.0),
    "myoglobin": (0.0, 50000.0),
    "serum_osmolality": (150.0, 500.0),
    "serum_ketones": (0.0, 20.0),
    "tsh": (0.0, 500.0),
    "free_t4": (0.0, 10.0),
    "cortisol": (0.0, 200.0),
    "crp": (0.0, 1000.0),
    "crp_hs": (0.0, 1000.0),
    "esr": (0.0, 200.0),
    "ferritin": (0.0, 100000.0),
    "uric_acid": (0.0, 50.0),
}

FEATURE_FOR_CLEANING = {
    "map": "MAP",
    "lactate": "Lactate",
    "creatinine": "Creatinine",
    "o2sat": "O2Sat",
    "heart_rate": "HR",
    "respiratory_rate": "Resp",
    "temperature": "Temp",
    "bicarbonate": "HCO3",
    "ph": "pH",
    "potassium": "Potassium",
    "glucose": "Glucose",
}

STATE_VARS = [
    "map",
    "lactate",
    "creatinine",
    "urine_output",
    "o2sat",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "wbc",
    "bilirubin",
    "bilirubin_direct",
    "platelets",
    "bicarbonate",
    "ph",
    "sodium",
    "potassium",
    "bun",
    "glucose",
]

TARGET_VARS = [
    "map",
    "lactate",
    "creatinine",
    "urine_output",
    "o2sat",
    "heart_rate",
    "respiratory_rate",
    "vasopressor_requirement",
]

ACTION_KEYS = (
    "vasopressor",
    "fluids",
    "antibiotics",
    "ventilation",
    "renal_replacement",
)

ACTION_EVIDENCE_COLUMNS = [
    "stay_id",
    "starttime",
    "endtime",
    "action",
    "source",
    "evidence_kind",
    "dose_observed",
    "original_label",
]

VASOPRESSOR_TERMS = (
    "norepinephrine",
    "levophed",
    "epinephrine",
    "vasopressin",
    "phenylephrine",
    "neosynephrine",
    "dopamine",
)

FLUID_TERMS = (
    "normal saline",
    "sodium chloride",
    "nacl",
    "lactated",
    "ringer",
    "crystalloid",
    "fluid bolus",
    "iv fluid",
    "saline bolus",
)

ANTIBIOTIC_TERMS = (
    "vancomycin",
    "zosyn",
    "piperacillin",
    "tazobactam",
    "meropenem",
    "imipenem",
    "ertapenem",
    "cefepime",
    "ceftriaxone",
    "ceftazidime",
    "cefazolin",
    "cefuroxime",
    "ciprofloxacin",
    "levofloxacin",
    "metronidazole",
    "gentamicin",
    "tobramycin",
    "linezolid",
    "daptomycin",
    "clindamycin",
    "aztreonam",
    "ampicillin",
    "amoxicillin",
    "penicillin",
    "nafcillin",
    "oxacillin",
)

VENTILATION_TERMS = (
    "ventilator",
    "mechanical ventilation",
    "intub",
    "ett",
    "bipap",
    "cpap",
)

RENAL_REPLACEMENT_TERMS = (
    "dialysis",
    "crrt",
    "cvvh",
    "hemofiltration",
    "renal replacement",
)


def pseudo_time(minutes_or_hours: float, unit: str = "minutes") -> pd.Timestamp:
    if unit == "hours":
        return BASE_TIME + pd.Timedelta(hours=float(minutes_or_hours))
    return BASE_TIME + pd.Timedelta(minutes=float(minutes_or_hours))


def horizon_suffix(horizon_hours: float) -> str:
    value = float(horizon_hours)
    if value.is_integer():
        return f"tp{int(value)}"
    return "tp" + str(value).replace(".", "p")


def future_column(var: str, horizon_hours: float) -> str:
    return f"{var}_{horizon_suffix(horizon_hours)}"


def _finite_number(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else math.nan


def _contains_any(value: object, patterns: tuple[str, ...]) -> bool:
    text = str(value or "").lower()
    return any(pattern in text for pattern in patterns)


def _text_series_contains_any(series: pd.Series, patterns: tuple[str, ...]) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask |= text.str.contains(pattern, regex=False)
    return mask


def _read_csv_chunks(path: Path, usecols: list[str], chunksize: int):
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)


def clean_state_values(var: str, values: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce").astype("float32")
    feature = FEATURE_FOR_CLEANING.get(var)
    if feature in set(VITAL_PERIODIC_MAP.values()) | set(VITAL_APERIODIC_MAP.values()):
        cleaned = clean_feature_values(feature, cleaned)
    lower, upper = PLAUSIBLE[var]
    return cleaned.where((cleaned >= lower) & (cleaned <= upper))


def empty_measurements() -> pd.DataFrame:
    return pd.DataFrame(columns=["stay_id", "offset", "var", "valuenum"])


def read_patient_meta(data_root: Path) -> pd.DataFrame:
    columns = [
        ID,
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "wardid",
        "apacheadmissiondx",
        "unittype",
        "unitdischargeoffset",
        "unitdischargestatus",
        "hospitaldischargestatus",
    ]
    frame = pd.read_csv(data_root / "patient.csv.gz", usecols=columns, low_memory=False)
    frame = frame.rename(columns={ID: "stay_id"})
    frame["stay_id"] = frame["stay_id"].astype("int64")
    frame["subject_id"] = frame["uniquepid"].fillna(frame["stay_id"]).astype(str)
    frame["unitdischarge_hr"] = pd.to_numeric(
        frame["unitdischargeoffset"], errors="coerce"
    ) / 60.0
    return frame.drop_duplicates("stay_id").set_index("stay_id", drop=False)


def read_diagnosis_sepsis_stays(data_root: Path, meta: pd.DataFrame) -> dict[int, dict[str, object]]:
    hits: dict[int, dict[str, object]] = {}
    specs = [
        (
            "diagnosis.csv.gz",
            [ID, "diagnosisoffset", "diagnosisstring", "icd9code"],
            "diagnosisoffset",
        ),
        (
            "admissionDx.csv.gz",
            [ID, "admitdxenteredoffset", "admitdxpath", "admitdxname", "admitdxtext"],
            "admitdxenteredoffset",
        ),
    ]
    for filename, columns, offset_column in specs:
        path = data_root / filename
        if not path.exists():
            continue
        text_columns = [column for column in columns if column not in (ID, offset_column)]
        for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
            text = chunk[text_columns].fillna("").agg(" ".join, axis=1)
            matched = chunk[_text_series_contains_any(text, SEPSIS_PATTERNS)]
            for record in matched.to_dict("records"):
                stay_id = int(record[ID])
                offset = _finite_number(record.get(offset_column))
                anchor = max(0.0, offset / 60.0) if np.isfinite(offset) else 0.0
                current = hits.get(stay_id)
                if current is None or anchor < float(current["anchor_hour"]):
                    hits[stay_id] = {
                        "anchor_hour": float(anchor),
                        "criteria": f"diagnosis_text_or_code:{filename}",
                    }

    if "apacheadmissiondx" in meta:
        matched = meta[
            meta["apacheadmissiondx"].map(lambda value: _contains_any(value, SEPSIS_PATTERNS))
        ]
        for stay_id in matched["stay_id"].astype(int):
            hits.setdefault(int(stay_id), {
                "anchor_hour": 0.0,
                "criteria": "patient_apacheadmissiondx",
            })
    return hits


def read_labs(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    columns = [ID, "labresultoffset", "labname", "labresult"]
    rows = []
    for chunk in _read_csv_chunks(data_root / "lab.csv.gz", columns, chunksize=750_000):
        chunk = chunk[chunk[ID].isin(stay_ids) & chunk["labname"].isin(LAB_TO_STATE)].copy()
        if chunk.empty:
            continue
        chunk["var"] = chunk["labname"].map(LAB_TO_STATE)
        chunk["valuenum"] = pd.to_numeric(chunk["labresult"], errors="coerce").astype("float32")
        for var in sorted(set(LAB_TO_STATE.values())):
            mask = chunk["var"] == var
            if mask.any():
                chunk.loc[mask, "valuenum"] = clean_state_values(var, chunk.loc[mask, "valuenum"])
        chunk = chunk.dropna(subset=["valuenum"])
        if chunk.empty:
            continue
        rows.append(chunk.rename(columns={ID: "stay_id", "labresultoffset": "offset"})[
            ["stay_id", "offset", "var", "valuenum"]
        ])
    return pd.concat(rows, ignore_index=True) if rows else empty_measurements()


def read_vitals(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    rows = []
    periodic_cols = [ID, "observationoffset", *VITAL_PERIODIC_MAP.keys()]
    for chunk in _read_csv_chunks(data_root / "vitalPeriodic.csv.gz", periodic_cols, chunksize=750_000):
        chunk = chunk[chunk[ID].isin(stay_ids)].copy()
        if chunk.empty:
            continue
        for source, state_var in VITAL_TO_STATE.items():
            if source not in chunk:
                continue
            values = clean_state_values(state_var, chunk[source])
            rows.append(pd.DataFrame({
                "stay_id": chunk[ID].astype("int64"),
                "offset": chunk["observationoffset"],
                "var": state_var,
                "valuenum": values,
            }))

    aperiodic_cols = [ID, "observationoffset", *VITAL_APERIODIC_MAP.keys()]
    for chunk in _read_csv_chunks(data_root / "vitalAperiodic.csv.gz", aperiodic_cols, chunksize=750_000):
        chunk = chunk[chunk[ID].isin(stay_ids)].copy()
        if chunk.empty:
            continue
        for source, state_var in VITAL_TO_STATE.items():
            if source not in chunk:
                continue
            values = clean_state_values(state_var, chunk[source])
            rows.append(pd.DataFrame({
                "stay_id": chunk[ID].astype("int64"),
                "offset": chunk["observationoffset"],
                "var": state_var,
                "valuenum": values,
            }))
    if not rows:
        return empty_measurements()
    return pd.concat(rows, ignore_index=True).dropna(subset=["valuenum"])


def read_urine_output(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "intakeOutput.csv.gz"
    if not path.exists():
        return empty_measurements()
    columns = [ID, "intakeoutputoffset", "celllabel", "cellvaluenumeric"]
    rows = []
    for chunk in _read_csv_chunks(path, columns, chunksize=750_000):
        chunk = chunk[chunk[ID].isin(stay_ids)].copy()
        if chunk.empty:
            continue
        labels = chunk["celllabel"].fillna("").astype(str).str.lower()
        urine = labels.str.contains("urine|foley|urinary|void", regex=True)
        chunk = chunk[urine].copy()
        if chunk.empty:
            continue
        chunk["valuenum"] = clean_state_values("urine_output", chunk["cellvaluenumeric"])
        chunk = chunk.dropna(subset=["valuenum"])
        if chunk.empty:
            continue
        chunk["offset"] = pd.to_numeric(chunk["intakeoutputoffset"], errors="coerce")
        chunk = chunk[np.isfinite(chunk["offset"])]
        chunk["hour"] = np.floor(chunk["offset"] / 60.0).astype("int32")
        grouped = chunk.groupby([ID, "hour"], sort=False)["valuenum"].sum().reset_index()
        grouped["offset"] = grouped["hour"] * 60.0
        grouped["var"] = "urine_output"
        rows.append(grouped.rename(columns={ID: "stay_id"})[
            ["stay_id", "offset", "var", "valuenum"]
        ])
    return pd.concat(rows, ignore_index=True) if rows else empty_measurements()


def read_measurements(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    measurements = pd.concat(
        [
            read_labs(data_root, stay_ids),
            read_vitals(data_root, stay_ids),
            read_urine_output(data_root, stay_ids),
        ],
        ignore_index=True,
    )
    if measurements.empty:
        return measurements
    measurements["stay_id"] = measurements["stay_id"].astype("int64")
    measurements["time_hr"] = pd.to_numeric(measurements["offset"], errors="coerce") / 60.0
    measurements = measurements[np.isfinite(measurements["time_hr"])]
    return measurements.sort_values(["stay_id", "time_hr", "var"]).reset_index(drop=True)


def classify_sepsis_action(label: object) -> tuple[str, ...]:
    text = str(label or "").lower()
    compact = re.sub(r"\s+", "", text)
    actions = []
    if any(term in text for term in VASOPRESSOR_TERMS):
        actions.append("vasopressor")
    if any(term in text for term in FLUID_TERMS) or compact in {"ns", "lr"}:
        actions.append("fluids")
    if any(term in text for term in ANTIBIOTIC_TERMS):
        actions.append("antibiotics")
    if any(term in text for term in VENTILATION_TERMS):
        actions.append("ventilation")
    if any(term in text for term in RENAL_REPLACEMENT_TERMS):
        actions.append("renal_replacement")
    return tuple(dict.fromkeys(actions))


def action_terms_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for terms in (
        VASOPRESSOR_TERMS,
        FLUID_TERMS,
        ANTIBIOTIC_TERMS,
        VENTILATION_TERMS,
        RENAL_REPLACEMENT_TERMS,
    ):
        for term in terms:
            mask |= text.str.contains(term, regex=False)
    return mask


def empty_action_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_EVIDENCE_COLUMNS)


def medication_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "medication.csv.gz"
    columns = [
        ID,
        "drugstartoffset",
        "drugstopoffset",
        "drugname",
        "routeadmin",
        "drugordercancelled",
    ]
    rows = []
    for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
        chunk = chunk[
            chunk[ID].isin(stay_ids)
            & (chunk["drugordercancelled"].fillna("No").astype(str).str.lower() != "yes")
            & action_terms_mask(chunk["drugname"])
        ]
        for record in chunk.to_dict("records"):
            start = _finite_number(record.get("drugstartoffset"))
            if not np.isfinite(start):
                continue
            stop = _finite_number(record.get("drugstopoffset"))
            if not np.isfinite(stop) or stop <= start:
                stop = start + 30.0
            label = record.get("drugname")
            for action in classify_sepsis_action(label):
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(start),
                    "endtime": pseudo_time(stop),
                    "action": action,
                    "source": "eicu_medication",
                    "evidence_kind": "active_medication_order",
                    "dose_observed": False,
                    "original_label": str(label or ""),
                })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def infusion_evidence(data_root: Path, stay_ids: set[int], max_snapshot_hours: float = 4.0) -> pd.DataFrame:
    path = data_root / "infusionDrug.csv.gz"
    columns = [
        ID,
        "infusionoffset",
        "drugname",
        "drugrate",
        "infusionrate",
        "drugamount",
        "volumeoffluid",
    ]
    frames = []
    for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
        chunk = chunk[chunk[ID].isin(stay_ids) & action_terms_mask(chunk["drugname"])].copy()
        if chunk.empty:
            continue
        chunk["offset"] = pd.to_numeric(chunk["infusionoffset"], errors="coerce")
        chunk = chunk[np.isfinite(chunk["offset"])]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return empty_action_evidence()
    frame = pd.concat(frames, ignore_index=True).sort_values([ID, "drugname", "offset"])
    rows = []
    for (_stay, _label), group in frame.groupby([ID, "drugname"], sort=False):
        offsets = group["offset"].to_numpy(dtype=np.float64)
        next_offsets = np.r_[offsets[1:], np.nan]
        for record, next_offset in zip(group.to_dict("records"), next_offsets):
            start = float(record["offset"])
            if np.isfinite(next_offset) and next_offset > start:
                duration = min(float(next_offset - start), max_snapshot_hours * 60.0)
            else:
                duration = 60.0
            rate = pd.to_numeric(record.get("drugrate"), errors="coerce")
            if not np.isfinite(rate):
                rate = pd.to_numeric(record.get("infusionrate"), errors="coerce")
            amount = pd.to_numeric(record.get("drugamount"), errors="coerce")
            volume = pd.to_numeric(record.get("volumeoffluid"), errors="coerce")
            dose_observed = bool(
                (np.isfinite(rate) and float(rate) > 0.0)
                or (np.isfinite(amount) and float(amount) > 0.0)
                or (np.isfinite(volume) and float(volume) > 0.0)
            )
            label = record.get("drugname")
            for action in classify_sepsis_action(label):
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(start),
                    "endtime": pseudo_time(start + duration),
                    "action": action,
                    "source": "eicu_infusiondrug",
                    "evidence_kind": "infusion_presence",
                    "dose_observed": dose_observed,
                    "original_label": str(label or ""),
                })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def treatment_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "treatment.csv.gz"
    columns = [ID, "treatmentoffset", "treatmentstring"]
    rows = []
    for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
        chunk = chunk[chunk[ID].isin(stay_ids) & action_terms_mask(chunk["treatmentstring"])]
        for record in chunk.to_dict("records"):
            offset = _finite_number(record.get("treatmentoffset"))
            if not np.isfinite(offset):
                continue
            label = record.get("treatmentstring")
            for action in classify_sepsis_action(label):
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(offset),
                    "endtime": pseudo_time(offset + 30.0),
                    "action": action,
                    "source": "eicu_treatment",
                    "evidence_kind": "coarse_treatment_presence",
                    "dose_observed": False,
                    "original_label": str(label or ""),
                })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def respiratory_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    rows = []
    care_path = data_root / "respiratoryCare.csv.gz"
    if care_path.exists():
        columns = [ID, "respcarestatusoffset", "ventstartoffset", "ventendoffset", "airwaytype"]
        for chunk in _read_csv_chunks(care_path, columns, chunksize=500_000):
            chunk = chunk[chunk[ID].isin(stay_ids)].copy()
            if chunk.empty:
                continue
            for record in chunk.to_dict("records"):
                start = _finite_number(record.get("ventstartoffset"))
                status = _finite_number(record.get("respcarestatusoffset"))
                if not np.isfinite(start):
                    airway = str(record.get("airwaytype") or "").lower()
                    if not airway or airway in {"none", "nan"}:
                        continue
                    start = status if np.isfinite(status) else math.nan
                if not np.isfinite(start):
                    continue
                stop = _finite_number(record.get("ventendoffset"))
                if not np.isfinite(stop) or stop <= start:
                    stop = start + 6.0 * 60.0
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(start),
                    "endtime": pseudo_time(stop),
                    "action": "ventilation",
                    "source": "eicu_respiratorycare",
                    "evidence_kind": "ventilation_interval_or_status",
                    "dose_observed": False,
                    "original_label": str(record.get("airwaytype") or "ventilation"),
                })
    chart_path = data_root / "respiratoryCharting.csv.gz"
    if chart_path.exists():
        columns = [ID, "respchartoffset", "respchartvaluelabel", "respchartvalue"]
        for chunk in _read_csv_chunks(chart_path, columns, chunksize=500_000):
            text = (
                chunk["respchartvaluelabel"].fillna("").astype(str)
                + " "
                + chunk["respchartvalue"].fillna("").astype(str)
            )
            chunk = chunk[chunk[ID].isin(stay_ids) & _text_series_contains_any(text, VENTILATION_TERMS)]
            for record in chunk.to_dict("records"):
                offset = _finite_number(record.get("respchartoffset"))
                if not np.isfinite(offset):
                    continue
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(offset),
                    "endtime": pseudo_time(offset + 60.0),
                    "action": "ventilation",
                    "source": "eicu_respiratorycharting",
                    "evidence_kind": "ventilation_chart_presence",
                    "dose_observed": False,
                    "original_label": " ".join(
                        str(record.get(column) or "")
                        for column in ("respchartvaluelabel", "respchartvalue")
                    ),
                })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def read_action_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    frames = [
        medication_evidence(data_root, stay_ids),
        infusion_evidence(data_root, stay_ids),
        treatment_evidence(data_root, stay_ids),
        respiratory_evidence(data_root, stay_ids),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_action_evidence()
    evidence = pd.concat(frames, ignore_index=True)
    return evidence.sort_values(["stay_id", "starttime", "source", "action"]).reset_index(drop=True)


def measurement_lookup(measurements: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    lookup = {}
    for (stay_id, var), group in measurements.groupby(["stay_id", "var"], sort=False):
        ordered = group.sort_values("time_hr")
        lookup[(int(stay_id), str(var))] = (
            ordered["time_hr"].to_numpy(dtype=np.float64),
            ordered["valuenum"].to_numpy(dtype=np.float64),
        )
    return lookup


def backward_value(times: np.ndarray, values: np.ndarray, t: float, lookback: float) -> tuple[float, float]:
    index = np.searchsorted(times, t, side="right") - 1
    if index < 0:
        return np.nan, np.nan
    age = t - float(times[index])
    if age < -1e-6 or age > lookback:
        return np.nan, np.nan
    return float(values[index]), float(age)


def nearest_value(times: np.ndarray, values: np.ndarray, t: float, tolerance: float) -> float:
    index = np.searchsorted(times, t)
    candidates = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return np.nan
    best = min(candidates, key=lambda item: abs(float(times[item]) - t))
    if abs(float(times[best]) - t) > tolerance:
        return np.nan
    return float(values[best])


def build_anchors(
    sepsis_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    meta: pd.DataFrame,
    horizon_hours: float = DELTA_H,
) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame()
    max_observed = measurements.groupby("stay_id")["time_hr"].max().to_dict()
    rows = []
    for stay_id, payload in sepsis_stays.items():
        if stay_id not in max_observed:
            continue
        anchor = max(0.0, float(payload["anchor_hour"]))
        discharge = meta.loc[stay_id, "unitdischarge_hr"] if stay_id in meta.index else np.nan
        observed_last = float(max_observed[stay_id])
        last = min(anchor + EPISODE_MAX_H, observed_last - horizon_hours)
        if np.isfinite(discharge):
            last = min(last, float(discharge) - horizon_hours)
        t = anchor
        while t <= last:
            subject = str(meta.loc[stay_id, "subject_id"]) if stay_id in meta.index else str(stay_id)
            hospital = int(meta.loc[stay_id, "hospitalid"]) if stay_id in meta.index and pd.notna(meta.loc[stay_id, "hospitalid"]) else -1
            unit = str(meta.loc[stay_id, "unittype"]) if stay_id in meta.index else ""
            rows.append({
                "subject_id": subject,
                "stay_id": int(stay_id),
                "hospitalid": hospital,
                "unittype": unit,
                "onset": pseudo_time(anchor, unit="hours"),
                "onset_hour": anchor,
                "onset_criteria": payload["criteria"],
                "hours_since_onset": float(t - anchor),
                "t": pseudo_time(t, unit="hours"),
                "t_hour": float(t),
                "t_plus": pseudo_time(t + horizon_hours, unit="hours"),
                "t_plus_hour": float(t + horizon_hours),
            })
            t += ANCHOR_STEP_H
    return pd.DataFrame(rows)


def assemble_states(
    anchors: pd.DataFrame,
    measurements: pd.DataFrame,
    horizon_hours: float = DELTA_H,
) -> pd.DataFrame:
    lookup = measurement_lookup(measurements)
    rows = []
    for anchor in anchors.itertuples(index=False):
        row = {}
        for var in STATE_VARS:
            times, values = lookup.get((int(anchor.stay_id), var), (np.asarray([]), np.asarray([])))
            if len(times):
                current, age = backward_value(times, values, float(anchor.t_hour), LOOKBACK_H)
                future = nearest_value(times, values, float(anchor.t_plus_hour), TARGET_TOL_H)
            else:
                current, age, future = np.nan, np.nan, np.nan
            row[f"{var}_t"] = current
            row[f"{var}_age_hr"] = age
            row[future_column(var, horizon_hours)] = future
        rows.append(row)
    return pd.DataFrame(rows, index=anchors.index)


def _hours_since_base(series: pd.Series) -> np.ndarray:
    return (
        (pd.to_datetime(series, errors="coerce") - BASE_TIME)
        / pd.Timedelta(hours=1)
    ).to_numpy(dtype=np.float64)


def prepare_evidence_lookup(evidence: pd.DataFrame) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if evidence.empty:
        return {}
    frame = evidence.copy()
    frame["start_hr"] = _hours_since_base(frame["starttime"])
    frame["end_hr"] = _hours_since_base(frame["endtime"])
    frame = frame[np.isfinite(frame["start_hr"]) & np.isfinite(frame["end_hr"])]
    output: dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for (stay_id, action), group in frame.groupby(["stay_id", "action"], sort=False):
        output.setdefault(int(stay_id), {})[str(action)] = (
            group["start_hr"].to_numpy(dtype=np.float64),
            group["end_hr"].to_numpy(dtype=np.float64),
            group["dose_observed"].fillna(False).to_numpy(dtype=bool),
        )
    return output


def evidence_window_summary(
    action_lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    anchor_hour: float,
    history_hours: float = LOOKBACK_H,
    future_hours: float = DELTA_H,
) -> dict[str, object]:
    history_start = float(anchor_hour - history_hours)
    future_end = float(anchor_hour + future_hours)
    output = {}
    for action in ACTION_KEYS:
        starts, ends, dose_observed = action_lookup.get(
            action,
            (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)),
        )
        history_mask = (ends >= history_start) & (starts < anchor_hour)
        future_mask = (ends >= anchor_hour) & (starts < future_end)
        output[f"hist_{action}_evidence_count"] = int(history_mask.sum())
        output[f"act_{action}_evidence_count"] = int(future_mask.sum())
        output[f"hist_{action}"] = int(history_mask.any())
        output[f"act_{action}"] = int(future_mask.any())
        output[f"act_{action}_dose_observed"] = int(
            dose_observed[future_mask].any() if future_mask.any() else False
        )
    output["vasopressor_requirement_t"] = output["hist_vasopressor"]
    output[f"vasopressor_requirement_{horizon_suffix(future_hours)}"] = output["act_vasopressor"]
    return output


def assemble_actions(
    anchors: pd.DataFrame,
    evidence: pd.DataFrame,
    horizon_hours: float = DELTA_H,
) -> pd.DataFrame:
    lookup = prepare_evidence_lookup(evidence)
    rows = []
    for anchor in anchors.itertuples(index=False):
        rows.append(evidence_window_summary(
            lookup.get(int(anchor.stay_id), {}),
            float(anchor.t_hour),
            future_hours=horizon_hours,
        ))
    return pd.DataFrame(rows, index=anchors.index)


def assemble_outcomes(anchors: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for anchor in anchors.itertuples(index=False):
        status = ""
        discharge_hr = np.nan
        if int(anchor.stay_id) in meta.index:
            record = meta.loc[int(anchor.stay_id)]
            status = " ".join(str(record.get(column, "")) for column in (
                "unitdischargestatus",
                "hospitaldischargestatus",
            )).lower()
            discharge_hr = float(record.get("unitdischarge_hr", np.nan))
        died = "expired" in status and np.isfinite(discharge_hr) and discharge_hr > float(anchor.t_plus_hour)
        rows.append({
            "died_after_window": int(died),
            "hrs_to_death_from_cut": (
                float(discharge_hr - float(anchor.t_plus_hour)) if died else np.nan
            ),
        })
    return pd.DataFrame(rows, index=anchors.index)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sepsis_active_t"] = (
        (frame["map_t"] < 65.0)
        | (frame["lactate_t"] >= 2.0)
        | (frame["vasopressor_requirement_t"] > 0)
        | (frame["creatinine_t"] >= 2.0)
        | (frame["urine_output_t"] < 30.0)
        | (frame["o2sat_t"] < 92.0)
        | (frame["wbc_t"] >= 12.0)
        | (frame["wbc_t"] <= 4.0)
    )
    return frame


def filter_evaluable(frame: pd.DataFrame, horizon_hours: float = DELTA_H) -> pd.DataFrame:
    has_pair = np.zeros(len(frame), dtype=bool)
    for var in TARGET_VARS:
        has_pair |= frame[f"{var}_t"].notna().values & frame[future_column(var, horizon_hours)].notna().values
    return frame[has_pair].reset_index(drop=True)


def target_pair_counts(frame: pd.DataFrame, horizon_hours: float = DELTA_H) -> dict[str, int]:
    return {
        var: int((frame[f"{var}_t"].notna() & frame[future_column(var, horizon_hours)].notna()).sum())
        for var in TARGET_VARS
    }


def action_support_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output = {}
    for action in ACTION_KEYS:
        mask = frame[f"act_{action}"].fillna(0).astype(bool)
        active = frame["sepsis_active_t"].fillna(False).astype(bool)
        output[action] = {
            "windows": int(mask.sum()),
            "stays": int(frame.loc[mask, "stay_id"].nunique()),
            "active_windows": int((mask & active).sum()),
            "active_stays": int(frame.loc[mask & active, "stay_id"].nunique()),
            "dose_observed_windows": int(frame[f"act_{action}_dose_observed"].fillna(0).sum()),
        }
    return output


def cohort_report(
    frame: pd.DataFrame,
    data_root: Path,
    output: Path,
    sepsis_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    evidence: pd.DataFrame,
    horizon_hours: float = DELTA_H,
) -> dict[str, object]:
    return {
        "dataset": "PhysioNet eICU Collaborative Research Database 2.0",
        "data_root": data_root.name,
        "output": str(output),
        "sepsis_like_stay_count_before_evaluable_filter": int(len(sepsis_stays)),
        "stays": int(frame["stay_id"].nunique()) if len(frame) else 0,
        "subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if len(frame) else 0,
        "transitions": int(len(frame)),
        "active_sepsis_transitions": int(frame["sepsis_active_t"].fillna(False).sum()) if len(frame) else 0,
        "measurements": {
            "rows": int(len(measurements)),
            "stays": int(measurements["stay_id"].nunique()) if len(measurements) else 0,
            "by_var": (
                measurements["var"].value_counts().astype(int).to_dict()
                if len(measurements) else {}
            ),
        },
        "target_pair_counts": target_pair_counts(frame, horizon_hours) if len(frame) else {},
        "action_support": action_support_counts(frame) if len(frame) else {},
        "action_evidence": {
            "rows": int(len(evidence)),
            "sources": (
                evidence["source"].value_counts().astype(int).to_dict()
                if len(evidence) else {}
            ),
            "by_action": (
                evidence["action"].value_counts().astype(int).to_dict()
                if len(evidence) else {}
            ),
            "dose_observed_fraction": (
                round(float(evidence["dose_observed"].fillna(False).mean()), 6)
                if len(evidence) else 0.0
            ),
        },
        "schema": {
            "targets": TARGET_VARS,
            "state_vars": STATE_VARS,
            "action_channels": ACTION_KEYS,
            "horizon_hours": horizon_hours,
            "future_suffix": horizon_suffix(horizon_hours),
            "anchor_step_hours": ANCHOR_STEP_H,
            "lookback_hours": LOOKBACK_H,
            "target_tolerance_hours": TARGET_TOL_H,
        },
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included_in_report": False,
            "factual_observed_treatment_only": True,
            "medication_orders_used_as_numeric_administrations": False,
            "coarse_treatment_rows_used_as_numeric_administrations": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
        },
    }


def build_transitions(
    data_root: Path,
    output: Path,
    horizon_hours: float = DELTA_H,
) -> dict[str, object]:
    meta = read_patient_meta(data_root)
    sepsis_stays = read_diagnosis_sepsis_stays(data_root, meta)
    stay_ids = set(sepsis_stays)
    measurements = read_measurements(data_root, stay_ids)
    anchors = build_anchors(sepsis_stays, measurements, meta, horizon_hours=horizon_hours)
    evidence = read_action_evidence(data_root, stay_ids)
    if anchors.empty:
        frame = pd.DataFrame()
    else:
        states = assemble_states(anchors, measurements, horizon_hours=horizon_hours)
        actions = assemble_actions(anchors, evidence, horizon_hours=horizon_hours)
        outcomes = assemble_outcomes(anchors, meta)
        frame = pd.concat(
            [
                anchors[[
                    "subject_id",
                    "stay_id",
                    "hospitalid",
                    "unittype",
                    "onset",
                    "onset_hour",
                    "onset_criteria",
                    "hours_since_onset",
                    "t",
                    "t_plus",
                ]],
                states,
                actions,
                outcomes,
            ],
            axis=1,
        )
        frame = filter_evaluable(add_derived_columns(frame), horizon_hours=horizon_hours)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return cohort_report(frame, data_root, output, sepsis_stays, measurements, evidence, horizon_hours=horizon_hours)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("eicu-collaborative-research-database-2.0"),
    )
    parser.add_argument("--output", type=Path, default=Path("eicu_sepsis_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("eicu_sepsis_transition_report.json"))
    parser.add_argument("--horizon-hours", type=float, default=DELTA_H)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_transitions(args.data_root, args.output, horizon_hours=args.horizon_hours)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "stays": report["stays"],
        "subjects": report["subjects"],
        "transitions": report["transitions"],
        "active_sepsis_transitions": report["active_sepsis_transitions"],
        "target_pair_counts": report["target_pair_counts"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
