"""Build a full-eICU AKI transition cohort with observed treatment evidence.

This is the third disease module in Chapter B after DKA and sepsis.  The AKI
cohort is defined by either diagnosis text/code support or KDIGO-like
creatinine rise:

* creatinine increase >= 0.3 mg/dL within 48 hours; or
* creatinine ratio >= 1.5 versus the previous 7-day minimum.

The output is a factual observed-treatment transition table.  It does not infer
causal effects of fluids, diuretics, vasopressors, renal replacement therapy, or
nephrotoxic medication exposure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_transition_extract import (
    BASE_TIME,
    ID,
    LOOKBACK_H,
    TARGET_TOL_H,
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
EPISODE_MAX_H = 120.0

AKI_PATTERNS = (
    "acute kidney injury",
    "acute renal failure",
    "acute tubular necrosis",
    "aki",
    "arf",
    "renal failure, acute",
    "kidney failure, acute",
    "584",
    "n17",
)

TARGET_VARS = (
    "creatinine",
    "urine_output",
    "potassium",
    "bicarbonate",
    "map",
    "bun",
    "sodium",
)

STATE_VARS = (
    "creatinine",
    "urine_output",
    "potassium",
    "bicarbonate",
    "map",
    "bun",
    "sodium",
    "heart_rate",
    "respiratory_rate",
    "o2sat",
    "temperature",
    "wbc",
    "lactate",
    "glucose",
)

ACTION_KEYS = (
    "fluids",
    "vasopressor",
    "diuretics",
    "renal_replacement",
    "nephrotoxin",
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

VASOPRESSOR_TERMS = (
    "norepinephrine",
    "levophed",
    "epinephrine",
    "vasopressin",
    "phenylephrine",
    "neosynephrine",
    "dopamine",
)

DIURETIC_TERMS = (
    "furosemide",
    "lasix",
    "bumetanide",
    "bumex",
    "torsemide",
    "chlorothiazide",
    "metolazone",
    "hydrochlorothiazide",
    "hctz",
)

RENAL_REPLACEMENT_TERMS = (
    "dialysis",
    "crrt",
    "cvvh",
    "cvvhd",
    "hemofiltration",
    "renal replacement",
)

NEPHROTOXIN_TERMS = (
    "vancomycin",
    "gentamicin",
    "tobramycin",
    "amikacin",
    "amphotericin",
    "acyclovir",
    "ketorolac",
    "ibuprofen",
    "naproxen",
    "tacrolimus",
    "cyclosporine",
    "cisplatin",
)


def _contains_any(value: object, patterns: tuple[str, ...]) -> bool:
    text = str(value or "").lower()
    return any(pattern in text for pattern in patterns)


def classify_aki_action(label: object) -> tuple[str, ...]:
    text = str(label or "").lower()
    compact = text.replace(" ", "")
    actions = []
    if any(term in text for term in FLUID_TERMS) or compact in {"ns", "lr"}:
        actions.append("fluids")
    if any(term in text for term in VASOPRESSOR_TERMS):
        actions.append("vasopressor")
    if any(term in text for term in DIURETIC_TERMS):
        actions.append("diuretics")
    if any(term in text for term in RENAL_REPLACEMENT_TERMS):
        actions.append("renal_replacement")
    if any(term in text for term in NEPHROTOXIN_TERMS):
        actions.append("nephrotoxin")
    return tuple(dict.fromkeys(actions))


def action_terms_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for terms in (
        FLUID_TERMS,
        VASOPRESSOR_TERMS,
        DIURETIC_TERMS,
        RENAL_REPLACEMENT_TERMS,
        NEPHROTOXIN_TERMS,
    ):
        for term in terms:
            mask |= text.str.contains(term, regex=False)
    return mask


def empty_action_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_EVIDENCE_COLUMNS)


def read_diagnosis_aki_stays(data_root: Path, meta: pd.DataFrame) -> dict[int, dict[str, object]]:
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
            matched = chunk[_text_series_contains_any(text, AKI_PATTERNS)]
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
            meta["apacheadmissiondx"].map(lambda value: _contains_any(value, AKI_PATTERNS))
        ]
        for stay_id in matched["stay_id"].astype(int):
            hits.setdefault(int(stay_id), {
                "anchor_hour": 0.0,
                "criteria": "patient_apacheadmissiondx",
            })
    return hits


def read_creatinine_aki_stays(data_root: Path) -> dict[int, dict[str, object]]:
    columns = [ID, "labresultoffset", "labname", "labresult"]
    frames = []
    for chunk in _read_csv_chunks(data_root / "lab.csv.gz", columns, chunksize=750_000):
        chunk = chunk[chunk["labname"] == "creatinine"].copy()
        if chunk.empty:
            continue
        chunk["time_hr"] = pd.to_numeric(chunk["labresultoffset"], errors="coerce") / 60.0
        chunk["creatinine"] = clean_state_values("creatinine", chunk["labresult"])
        chunk = chunk.dropna(subset=["time_hr", "creatinine"])
        if not chunk.empty:
            frames.append(chunk[[ID, "time_hr", "creatinine"]])
    if not frames:
        return {}
    frame = pd.concat(frames, ignore_index=True)
    hits: dict[int, dict[str, object]] = {}
    for stay_id, group in frame.groupby(ID, sort=False):
        ordered = group.sort_values("time_hr")
        times = ordered["time_hr"].to_numpy(dtype=np.float64)
        values = ordered["creatinine"].to_numpy(dtype=np.float64)
        for index, (time_hr, value) in enumerate(zip(times, values)):
            previous_times = times[:index]
            previous_values = values[:index]
            if len(previous_values) == 0:
                continue
            prior48 = previous_values[previous_times >= time_hr - 48.0]
            prior168 = previous_values[previous_times >= time_hr - 168.0]
            meets_absolute = len(prior48) and value - float(np.nanmin(prior48)) >= 0.3
            baseline = float(np.nanmin(prior168)) if len(prior168) else math.nan
            meets_ratio = np.isfinite(baseline) and baseline > 0.0 and value / baseline >= 1.5
            if meets_absolute or meets_ratio:
                hits[int(stay_id)] = {
                    "anchor_hour": float(max(0.0, time_hr)),
                    "criteria": (
                        "kdigo_like_creatinine_rise_absolute"
                        if meets_absolute else
                        "kdigo_like_creatinine_rise_ratio"
                    ),
                }
                break
    return hits


def merge_aki_like_stays(data_root: Path, meta: pd.DataFrame, max_stays: int | None = None) -> dict[int, dict[str, object]]:
    diagnosis = read_diagnosis_aki_stays(data_root, meta)
    creatinine = read_creatinine_aki_stays(data_root)
    merged = dict(diagnosis)
    for stay_id, payload in creatinine.items():
        if stay_id in merged:
            merged[stay_id] = {
                "anchor_hour": min(float(merged[stay_id]["anchor_hour"]), float(payload["anchor_hour"])),
                "criteria": "diagnosis_and_kdigo_like_creatinine",
            }
        else:
            merged[stay_id] = payload
    if max_stays is not None and len(merged) > max_stays:
        keep = sorted(merged)[:max_stays]
        merged = {stay_id: merged[stay_id] for stay_id in keep}
    return merged


def medication_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "medication.csv.gz"
    columns = [
        ID,
        "drugstartoffset",
        "drugstopoffset",
        "drugname",
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
            for action in classify_aki_action(label):
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
            for action in classify_aki_action(label):
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
            for action in classify_aki_action(label):
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


def read_action_evidence(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    frames = [
        medication_evidence(data_root, stay_ids),
        infusion_evidence(data_root, stay_ids),
        treatment_evidence(data_root, stay_ids),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_action_evidence()
    evidence = pd.concat(frames, ignore_index=True)
    return evidence.sort_values(["stay_id", "starttime", "source", "action"]).reset_index(drop=True)


def build_anchors(
    aki_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame()
    max_observed = measurements.groupby("stay_id")["time_hr"].max().to_dict()
    rows = []
    for stay_id, payload in aki_stays.items():
        if stay_id not in max_observed:
            continue
        anchor = max(0.0, float(payload["anchor_hour"]))
        discharge = meta.loc[stay_id, "unitdischarge_hr"] if stay_id in meta.index else np.nan
        observed_last = float(max_observed[stay_id])
        last = min(anchor + EPISODE_MAX_H, observed_last - DELTA_H)
        if np.isfinite(discharge):
            last = min(last, float(discharge) - DELTA_H)
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
                "t_plus": pseudo_time(t + DELTA_H, unit="hours"),
                "t_plus_hour": float(t + DELTA_H),
            })
            t += ANCHOR_STEP_H
    return pd.DataFrame(rows)


def assemble_states(anchors: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
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
            row[f"{var}_tp6"] = future
        rows.append(row)
    return pd.DataFrame(rows, index=anchors.index)


def evidence_window_summary(
    action_lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    anchor_hour: float,
) -> dict[str, object]:
    history_start = float(anchor_hour - LOOKBACK_H)
    future_end = float(anchor_hour + DELTA_H)
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


def assemble_actions(anchors: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    lookup = prepare_evidence_lookup(evidence)
    rows = []
    for anchor in anchors.itertuples(index=False):
        rows.append(evidence_window_summary(
            lookup.get(int(anchor.stay_id), {}),
            float(anchor.t_hour),
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
    frame["aki_active_t"] = (
        (frame["creatinine_t"] >= 1.5)
        | (frame["urine_output_t"] < 30.0)
        | (frame["potassium_t"] >= 5.5)
        | (frame["bun_t"] >= 40.0)
        | (frame["hist_renal_replacement"] > 0)
    )
    return frame


def filter_evaluable(frame: pd.DataFrame) -> pd.DataFrame:
    has_pair = np.zeros(len(frame), dtype=bool)
    for var in TARGET_VARS:
        has_pair |= frame[f"{var}_t"].notna().values & frame[f"{var}_tp6"].notna().values
    return frame[has_pair].reset_index(drop=True)


def target_pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        var: int((frame[f"{var}_t"].notna() & frame[f"{var}_tp6"].notna()).sum())
        for var in TARGET_VARS
    }


def action_support_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output = {}
    for action in ACTION_KEYS:
        mask = frame[f"act_{action}"].fillna(0).astype(bool)
        active = frame["aki_active_t"].fillna(False).astype(bool)
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
    aki_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    evidence: pd.DataFrame,
) -> dict[str, object]:
    criteria_counts = pd.Series(
        [payload["criteria"] for payload in aki_stays.values()],
        dtype="object",
    ).value_counts().astype(int).to_dict()
    return {
        "dataset": "PhysioNet eICU Collaborative Research Database 2.0",
        "data_root": data_root.name,
        "output": output.name,
        "aki_like_stay_count_before_evaluable_filter": int(len(aki_stays)),
        "aki_criteria_counts_before_filter": criteria_counts,
        "stays": int(frame["stay_id"].nunique()) if len(frame) else 0,
        "subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if len(frame) else 0,
        "transitions": int(len(frame)),
        "active_aki_transitions": int(frame["aki_active_t"].fillna(False).sum()) if len(frame) else 0,
        "measurements": {
            "rows": int(len(measurements)),
            "stays": int(measurements["stay_id"].nunique()) if len(measurements) else 0,
            "by_var": (
                measurements["var"].value_counts().astype(int).to_dict()
                if len(measurements) else {}
            ),
        },
        "target_pair_counts": target_pair_counts(frame) if len(frame) else {},
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
            "horizon_hours": DELTA_H,
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


def build_transitions(data_root: Path, output: Path, max_stays: int | None = None) -> dict[str, object]:
    meta = read_patient_meta(data_root)
    aki_stays = merge_aki_like_stays(data_root, meta, max_stays=max_stays)
    stay_ids = set(aki_stays)
    measurements = read_measurements(data_root, stay_ids)
    anchors = build_anchors(aki_stays, measurements, meta)
    evidence = read_action_evidence(data_root, stay_ids)
    if anchors.empty:
        frame = pd.DataFrame()
    else:
        states = assemble_states(anchors, measurements)
        actions = assemble_actions(anchors, evidence)
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
        frame = filter_evaluable(add_derived_columns(frame))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return cohort_report(frame, data_root, output, aki_stays, measurements, evidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("eicu-collaborative-research-database-2.0"),
    )
    parser.add_argument("--output", type=Path, default=Path("eicu_aki_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("eicu_aki_transition_report.json"))
    parser.add_argument("--max-stays", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_transitions(args.data_root, args.output, max_stays=args.max_stays)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "stays": report["stays"],
        "subjects": report["subjects"],
        "transitions": report["transitions"],
        "active_aki_transitions": report["active_aki_transitions"],
        "target_pair_counts": report["target_pair_counts"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
