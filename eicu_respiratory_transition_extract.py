"""Build a full-eICU respiratory-failure transition cohort.

This is the fourth Chapter-B disease module.  It intentionally mirrors the
sepsis/AKI observed-treatment transition contract:

    state_t + observed treatment evidence in [t, t+h] + state_t+h

The cohort is respiratory-failure / hypoxemia-like, defined by diagnosis text
support or objective oxygenation/respiratory-rate evidence.  It is a factual
forecasting adapter only; it does not estimate causal treatment effects of
oxygen, ventilation, bronchodilators, steroids, antibiotics, fluids, or
vasopressors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_transition_extract import (
    ANTIBIOTIC_TERMS,
    BASE_TIME,
    FLUID_TERMS,
    ID,
    LOOKBACK_H,
    TARGET_TOL_H,
    VASOPRESSOR_TERMS,
    VENTILATION_TERMS,
    _finite_number,
    _read_csv_chunks,
    _text_series_contains_any,
    backward_value,
    clean_state_values,
    measurement_lookup,
    nearest_value,
    prepare_evidence_lookup,
    pseudo_time,
    read_measurements,
    read_patient_meta,
)


DELTA_H = 6.0
ANCHOR_STEP_H = 4.0
EPISODE_MAX_H = 96.0

RESPIRATORY_PATTERNS = (
    "acute respiratory distress",
    "ards",
    "mechanical ventilation",
    "ventilator",
    "j80",
)

TARGET_VARS = (
    "o2sat",
    "respiratory_rate",
    "heart_rate",
    "map",
    "bicarbonate",
    "ph",
)

STATE_VARS = (
    "o2sat",
    "respiratory_rate",
    "heart_rate",
    "map",
    "temperature",
    "wbc",
    "lactate",
    "bicarbonate",
    "ph",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "glucose",
    "urine_output",
)

ACTION_KEYS = (
    "ventilation",
    "bronchodilator",
    "systemic_steroid",
    "antibiotics",
    "fluids",
    "vasopressor",
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

BRONCHODILATOR_TERMS = (
    "albuterol",
    "salbutamol",
    "levalbuterol",
    "xopenex",
    "ipratropium",
    "atrovent",
    "duoneb",
    "combivent",
    "nebulizer",
)

SYSTEMIC_STEROID_TERMS = (
    "methylprednisolone",
    "solumedrol",
    "solu-medrol",
    "prednisone",
    "prednisolone",
    "hydrocortisone",
    "dexamethasone",
    "decadron",
)


def _contains_any(value: object, patterns: tuple[str, ...]) -> bool:
    text = str(value or "").lower()
    return any(pattern in text for pattern in patterns)


def horizon_suffix(horizon_hours: float) -> str:
    value = float(horizon_hours)
    if value.is_integer():
        return f"tp{int(value)}"
    return "tp" + str(value).replace(".", "p")


def future_column(var: str, horizon_hours: float) -> str:
    return f"{var}_{horizon_suffix(horizon_hours)}"


def classify_respiratory_action(label: object) -> tuple[str, ...]:
    text = str(label or "").lower()
    compact = text.replace(" ", "")
    actions = []
    if any(term in text for term in VENTILATION_TERMS):
        actions.append("ventilation")
    if any(term in text for term in BRONCHODILATOR_TERMS):
        actions.append("bronchodilator")
    if any(term in text for term in SYSTEMIC_STEROID_TERMS):
        actions.append("systemic_steroid")
    if any(term in text for term in ANTIBIOTIC_TERMS):
        actions.append("antibiotics")
    if any(term in text for term in FLUID_TERMS) or compact in {"ns", "lr"}:
        actions.append("fluids")
    if any(term in text for term in VASOPRESSOR_TERMS):
        actions.append("vasopressor")
    return tuple(dict.fromkeys(actions))


def action_terms_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for terms in (
        VENTILATION_TERMS,
        BRONCHODILATOR_TERMS,
        SYSTEMIC_STEROID_TERMS,
        ANTIBIOTIC_TERMS,
        FLUID_TERMS,
        VASOPRESSOR_TERMS,
    ):
        for term in terms:
            mask |= text.str.contains(term, regex=False)
    return mask


def empty_action_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_EVIDENCE_COLUMNS)


def read_diagnosis_respiratory_stays(data_root: Path, meta: pd.DataFrame) -> dict[int, dict[str, object]]:
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
            matched = chunk[_text_series_contains_any(text, RESPIRATORY_PATTERNS)]
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
            meta["apacheadmissiondx"].map(lambda value: _contains_any(value, RESPIRATORY_PATTERNS))
        ]
        for stay_id in matched["stay_id"].astype(int):
            hits.setdefault(int(stay_id), {
                "anchor_hour": 0.0,
                "criteria": "patient_apacheadmissiondx",
            })
    return hits


def read_measurement_respiratory_stays(data_root: Path) -> dict[int, dict[str, object]]:
    hits: dict[int, dict[str, object]] = {}
    # Only vitalPeriodic carries oxygen saturation in eICU v2.0.  Common
    # hypoxemia thresholds are too broad in ICU data, so this objective entry is
    # deliberately limited to severe measured hypoxemia.
    for filename in ("vitalPeriodic.csv.gz",):
        path = data_root / filename
        if not path.exists():
            continue
        columns = [ID, "observationoffset", "sao2", "respiration"]
        for chunk in _read_csv_chunks(path, columns, chunksize=750_000):
            offset = pd.to_numeric(chunk["observationoffset"], errors="coerce") / 60.0
            o2sat = clean_state_values("o2sat", chunk["sao2"])
            mask = o2sat <= 60.0
            matched = chunk[mask.fillna(False)].copy()
            if matched.empty:
                continue
            matched["anchor_hour"] = offset[matched.index].clip(lower=0.0)
            matched["criteria"] = "objective_severe_hypoxemia_o2sat_le_60"
            for record in matched[[ID, "anchor_hour", "criteria"]].to_dict("records"):
                stay_id = int(record[ID])
                anchor = float(record["anchor_hour"])
                current = hits.get(stay_id)
                if current is None or anchor < float(current["anchor_hour"]):
                    hits[stay_id] = {
                        "anchor_hour": anchor,
                        "criteria": str(record["criteria"]),
                    }
    return hits


def merge_respiratory_like_stays(
    data_root: Path,
    meta: pd.DataFrame,
    max_stays: int | None = None,
) -> dict[int, dict[str, object]]:
    diagnosis = read_diagnosis_respiratory_stays(data_root, meta)
    measurement = read_measurement_respiratory_stays(data_root)
    merged = dict(diagnosis)
    for stay_id, payload in measurement.items():
        if stay_id in merged:
            merged[stay_id] = {
                "anchor_hour": min(float(merged[stay_id]["anchor_hour"]), float(payload["anchor_hour"])),
                "criteria": "diagnosis_and_objective_respiratory_evidence",
            }
        else:
            merged[stay_id] = payload
    if max_stays is not None and len(merged) > max_stays:
        keep = sorted(merged)[:max_stays]
        merged = {stay_id: merged[stay_id] for stay_id in keep}
    return merged


def medication_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "medication.csv.gz"
    columns = [ID, "drugstartoffset", "drugstopoffset", "drugname", "drugordercancelled"]
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
            for action in classify_respiratory_action(record.get("drugname")):
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(start),
                    "endtime": pseudo_time(stop),
                    "action": action,
                    "source": "eicu_medication",
                    "evidence_kind": "active_medication_order",
                    "dose_observed": False,
                    "original_label": str(record.get("drugname") or ""),
                })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def infusion_evidence(data_root: Path, stay_ids: set[int], max_snapshot_hours: float = 4.0) -> pd.DataFrame:
    path = data_root / "infusionDrug.csv.gz"
    columns = [ID, "infusionoffset", "drugname", "drugrate", "infusionrate", "drugamount", "volumeoffluid"]
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
            for action in classify_respiratory_action(record.get("drugname")):
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(start),
                    "endtime": pseudo_time(start + duration),
                    "action": action,
                    "source": "eicu_infusiondrug",
                    "evidence_kind": "infusion_presence",
                    "dose_observed": dose_observed,
                    "original_label": str(record.get("drugname") or ""),
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
            for action in classify_respiratory_action(record.get("treatmentstring")):
                rows.append({
                    "stay_id": int(record[ID]),
                    "starttime": pseudo_time(offset),
                    "endtime": pseudo_time(offset + 30.0),
                    "action": action,
                    "source": "eicu_treatment",
                    "evidence_kind": "coarse_treatment_presence",
                    "dose_observed": False,
                    "original_label": str(record.get("treatmentstring") or ""),
                })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def respiratory_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    rows = []
    care_path = data_root / "respiratoryCare.csv.gz"
    if care_path.exists():
        columns = [ID, "respcarestatusoffset", "ventstartoffset", "ventendoffset", "airwaytype"]
        for chunk in _read_csv_chunks(care_path, columns, chunksize=500_000):
            chunk = chunk[chunk[ID].isin(stay_ids)].copy()
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
            chunk = chunk[chunk[ID].isin(stay_ids)].copy()
            if chunk.empty:
                continue
            text = chunk["respchartvaluelabel"].fillna("").astype(str) + " " + chunk["respchartvalue"].fillna("").astype(str)
            chunk = chunk[_text_series_contains_any(text, VENTILATION_TERMS)]
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


def build_anchors(
    respiratory_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    meta: pd.DataFrame,
    horizon_hours: float = DELTA_H,
) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame()
    max_observed = measurements.groupby("stay_id")["time_hr"].max().to_dict()
    rows = []
    for stay_id, payload in respiratory_stays.items():
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


def evidence_window_summary(
    action_lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    anchor_hour: float,
    horizon_hours: float = DELTA_H,
) -> dict[str, object]:
    history_start = float(anchor_hour - LOOKBACK_H)
    future_end = float(anchor_hour + horizon_hours)
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
            horizon_hours,
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
            "hrs_to_death_from_cut": float(discharge_hr - float(anchor.t_plus_hour)) if died else np.nan,
        })
    return pd.DataFrame(rows, index=anchors.index)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["respiratory_active_t"] = (
        (frame["o2sat_t"] <= 92.0)
        | (frame["respiratory_rate_t"] >= 24.0)
        | (frame["hist_ventilation"] > 0)
        | (frame["ph_t"] <= 7.30)
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
    active = frame["respiratory_active_t"].fillna(False).astype(bool)
    for action in ACTION_KEYS:
        mask = frame[f"act_{action}"].fillna(0).astype(bool)
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
    respiratory_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    evidence: pd.DataFrame,
    horizon_hours: float = DELTA_H,
) -> dict[str, object]:
    criteria_counts = pd.Series(
        [payload["criteria"] for payload in respiratory_stays.values()],
        dtype="object",
    ).value_counts().astype(int).to_dict()
    return {
        "dataset": "PhysioNet eICU Collaborative Research Database 2.0",
        "data_root": data_root.name,
        "output": output.name,
        "respiratory_like_stay_count_before_evaluable_filter": int(len(respiratory_stays)),
        "respiratory_criteria_counts_before_filter": criteria_counts,
        "stays": int(frame["stay_id"].nunique()) if len(frame) else 0,
        "subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if len(frame) else 0,
        "transitions": int(len(frame)),
        "active_respiratory_transitions": int(frame["respiratory_active_t"].fillna(False).sum()) if len(frame) else 0,
        "measurements": {
            "rows": int(len(measurements)),
            "stays": int(measurements["stay_id"].nunique()) if len(measurements) else 0,
            "by_var": measurements["var"].value_counts().astype(int).to_dict() if len(measurements) else {},
        },
        "target_pair_counts": target_pair_counts(frame, horizon_hours) if len(frame) else {},
        "action_support": action_support_counts(frame) if len(frame) else {},
        "action_evidence": {
            "rows": int(len(evidence)),
            "sources": evidence["source"].value_counts().astype(int).to_dict() if len(evidence) else {},
            "by_action": evidence["action"].value_counts().astype(int).to_dict() if len(evidence) else {},
            "dose_observed_fraction": round(float(evidence["dose_observed"].fillna(False).mean()), 6) if len(evidence) else 0.0,
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
    max_stays: int | None = None,
    horizon_hours: float = DELTA_H,
) -> dict[str, object]:
    meta = read_patient_meta(data_root)
    respiratory_stays = merge_respiratory_like_stays(data_root, meta, max_stays=max_stays)
    stay_ids = set(respiratory_stays)
    measurements = read_measurements(data_root, stay_ids)
    anchors = build_anchors(respiratory_stays, measurements, meta, horizon_hours=horizon_hours)
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
    return cohort_report(frame, data_root, output, respiratory_stays, measurements, evidence, horizon_hours=horizon_hours)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("eicu-collaborative-research-database-2.0"))
    parser.add_argument("--output", type=Path, default=Path("eicu_respiratory_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("eicu_respiratory_transition_report.json"))
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--horizon-hours", type=float, default=DELTA_H)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_transitions(
        args.data_root,
        args.output,
        max_stays=args.max_stays,
        horizon_hours=args.horizon_hours,
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "stays": report["stays"],
        "transitions": report["transitions"],
        "active_respiratory_transitions": report["active_respiratory_transitions"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
