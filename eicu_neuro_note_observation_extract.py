"""Extract timestamped eICU neurologic note/flowsheet observations.

This adapter deepens the acute-neuro observation layer with structured
neurologic exam signals from eICU nursing and progress-note tables.  It is a
measurement-depth adapter, not a treatment or causal module.

Leakage policy:
* current ``neuro_*_t`` observations use only note/flowsheet rows at or before
  the anchor, with a bounded lookback;
* future ``neuro_*_tp6`` labels use rows after the anchor and nearest to the
  forecast target time;
* row-level enhanced cohorts are local-only and must not be committed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_transition_extract import BASE_TIME


TARGETS = (
    "neuro_gcs",
    "neuro_sedation_score",
    "neuro_delirium_present",
    "neuro_mental_abnormal",
    "neuro_pupils_abnormal",
    "neuro_motor_abnormal",
)

ABNORMAL_MENTAL = (
    "lethargic",
    "sedated",
    "somnolent",
    "obtunded",
    "comatose",
    "unresponsive",
    "garbled",
    "slurred",
    "aphasic",
    "dysphasic",
    "inappropriate",
    "not oriented",
    "partially oriented",
    "depressed",
    "delusional",
    "agitated",
    "combative",
    "withdrawn",
)
NORMAL_MENTAL = (
    "normal loc",
    "oriented x3",
    "calm/appropriate",
    "clear",
    "normal",
    "spontaneous",
)
ABNORMAL_PUPILS = (
    "sluggish",
    "non-reactive",
    "non reactive",
    "do not react",
    "does not react",
    "unequal",
    "fixed",
    "asym",
)
NORMAL_PUPILS = (
    "pupils equal, react to light",
    "react to light",
    "brisk",
    "equal",
)
ABNORMAL_MOTOR = (
    "+3-gravity",
    "+2-no gravity",
    "+1-flicker",
    "0-none",
    "decreased",
    "absent",
    "extension",
    "flexion",
    "decorticate",
    "decerebrate",
    "flaccid",
    "no response",
    "withdraws",
    "localizes pain",
)
NORMAL_MOTOR = (
    "+5-normal",
    "+4-resists",
    "normal",
    "obeys commands",
)


def _finite_float(value: object) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def pseudo_time_from_offset(offset_minutes: object) -> pd.Timestamp | pd.NaT:
    value = _finite_float(offset_minutes)
    if not np.isfinite(value):
        return pd.NaT
    return BASE_TIME + pd.Timedelta(minutes=float(value))


def hours_since_base(series: pd.Series) -> np.ndarray:
    return (
        (pd.to_datetime(series, errors="coerce") - BASE_TIME)
        .dt.total_seconds()
        .to_numpy(dtype=np.float64)
        / 3600.0
    )


def text_has(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def binary_from_terms(value: object, positive: tuple[str, ...], negative: tuple[str, ...]) -> float:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return np.nan
    if text_has(text, positive):
        return 1.0
    if text_has(text, negative):
        return 0.0
    return np.nan


def read_nurse_charting(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "nurseCharting.csv.gz"
    columns = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypecat",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ]
    rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=500_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        label = chunk["nursingchartcelltypevallabel"].fillna("").astype(str).str.lower()
        name = chunk["nursingchartcelltypevalname"].fillna("").astype(str).str.lower()
        value = chunk["nursingchartvalue"].fillna("").astype(str).str.lower()
        relevant = (
            label.str.contains("glasgow|gcs|sedation|rass|delirium|mental|neurological", regex=True)
            | name.str.contains("gcs|eyes|motor|verbal|sedation|delirium", regex=True)
            | value.str.contains("cam-icu|rass|delirium|obeys|withdraws|flaccid", regex=True)
        )
        for record in chunk[relevant].to_dict("records"):
            stay_id = int(record["patientunitstayid"])
            note_time = pseudo_time_from_offset(record.get("nursingchartoffset"))
            if pd.isna(note_time):
                continue
            row_label = str(record.get("nursingchartcelltypevallabel") or "").lower()
            row_name = str(record.get("nursingchartcelltypevalname") or "").lower()
            row_value = str(record.get("nursingchartvalue") or "").strip().lower()
            numeric = _finite_float(row_value)

            if "glasgow" in row_label and row_name == "gcs total" and 3 <= numeric <= 15:
                rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_gcs", "value": numeric})
            if (("rass" in row_label) or ("sedation" in row_label and row_name == "sedation score")) and -5 <= numeric <= 4:
                rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_sedation_score", "value": numeric})
            if "delirium" in row_label:
                if row_value in {"yes", "y", "positive", "present", "1"}:
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_delirium_present", "value": 1.0})
                elif row_value in {"no", "n", "negative", "absent", "0"}:
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_delirium_present", "value": 0.0})

            mental = binary_from_terms(row_value, ABNORMAL_MENTAL, NORMAL_MENTAL)
            if np.isfinite(mental):
                rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_mental_abnormal", "value": mental})
            motor = binary_from_terms(row_value, ABNORMAL_MOTOR, NORMAL_MOTOR)
            if np.isfinite(motor) and ("motor" in row_label or "motor" in row_name or "gcs" in row_name):
                rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_motor_abnormal", "value": motor})
    return pd.DataFrame(rows, columns=["stay_id", "note_time", "target", "value"])


def read_nurse_assessment(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "nurseAssessment.csv.gz"
    columns = [
        "patientunitstayid",
        "nurseassessoffset",
        "cellattributepath",
        "celllabel",
        "cellattribute",
        "cellattributevalue",
    ]
    rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=500_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        context = (
            chunk["cellattributepath"].fillna("").astype(str)
            + " "
            + chunk["celllabel"].fillna("").astype(str)
            + " "
            + chunk["cellattribute"].fillna("").astype(str)
        ).str.lower()
        relevant = context.str.contains("neurologic|pupil|motor|mental|orientation|loc|speech|sensation", regex=True)
        for record in chunk[relevant].to_dict("records"):
            stay_id = int(record["patientunitstayid"])
            note_time = pseudo_time_from_offset(record.get("nurseassessoffset"))
            if pd.isna(note_time):
                continue
            context_text = " ".join(
                str(record.get(key) or "").lower()
                for key in ("cellattributepath", "celllabel", "cellattribute")
            )
            value = str(record.get("cellattributevalue") or "").strip().lower()
            if "pupil" in context_text:
                out = binary_from_terms(value, ABNORMAL_PUPILS, NORMAL_PUPILS)
                if np.isfinite(out):
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_pupils_abnormal", "value": out})
            if "motor" in context_text or "strength" in context_text:
                out = binary_from_terms(value, ABNORMAL_MOTOR, NORMAL_MOTOR)
                if np.isfinite(out):
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_motor_abnormal", "value": out})
            if any(term in context_text for term in ("mental", "orientation", "loc", "speech")):
                out = binary_from_terms(value, ABNORMAL_MENTAL, NORMAL_MENTAL)
                if np.isfinite(out):
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_mental_abnormal", "value": out})
    return pd.DataFrame(rows, columns=["stay_id", "note_time", "target", "value"])


GCS_TOTAL_RE = re.compile(r"/neurologic/gcs/([3-9]|1[0-5])$")


def read_physical_exam(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    path = data_root / "physicalExam.csv.gz"
    columns = [
        "patientunitstayid",
        "physicalexamoffset",
        "physicalexampath",
        "physicalexamvalue",
        "physicalexamtext",
    ]
    rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=500_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        context = (
            chunk["physicalexampath"].fillna("").astype(str)
            + " "
            + chunk["physicalexamvalue"].fillna("").astype(str)
            + " "
            + chunk["physicalexamtext"].fillna("").astype(str)
        ).str.lower()
        relevant = context.str.contains("neurologic|gcs|mental|orientation|pupil|motor|conscious|delir|agitat", regex=True)
        for record in chunk[relevant].to_dict("records"):
            stay_id = int(record["patientunitstayid"])
            note_time = pseudo_time_from_offset(record.get("physicalexamoffset"))
            if pd.isna(note_time):
                continue
            path_text = str(record.get("physicalexampath") or "").strip().lower()
            value_text = str(record.get("physicalexamvalue") or record.get("physicalexamtext") or "").strip().lower()
            match = GCS_TOTAL_RE.search(path_text)
            if match:
                rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_gcs", "value": float(match.group(1))})
            if "pupil" in path_text:
                out = binary_from_terms(value_text, ABNORMAL_PUPILS, NORMAL_PUPILS)
                if np.isfinite(out):
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_pupils_abnormal", "value": out})
            if "motor" in path_text:
                out = binary_from_terms(value_text, ABNORMAL_MOTOR, NORMAL_MOTOR)
                if np.isfinite(out):
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_motor_abnormal", "value": out})
            if any(term in path_text for term in ("mental status", "orientation", "level of consciousness")):
                out = binary_from_terms(value_text, ABNORMAL_MENTAL, NORMAL_MENTAL)
                if np.isfinite(out):
                    rows.append({"stay_id": stay_id, "note_time": note_time, "target": "neuro_mental_abnormal", "value": out})
    return pd.DataFrame(rows, columns=["stay_id", "note_time", "target", "value"])


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    events = events.dropna(subset=["stay_id", "note_time", "target", "value"]).copy()
    events["stay_id"] = events["stay_id"].astype("int64")
    events["value"] = pd.to_numeric(events["value"], errors="coerce")
    events = events[np.isfinite(events["value"])].copy()
    # Multiple rows can be entered at the same offset.  A mean is appropriate
    # for scores and for binary indicators when repeated assessments disagree.
    return (
        events.groupby(["stay_id", "note_time", "target"], as_index=False)["value"]
        .mean()
        .sort_values(["stay_id", "target", "note_time"])
        .reset_index(drop=True)
    )


def attach_target(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    target: str,
    horizon_hours: float,
    lookback_hours: float,
    future_tolerance_hours: float,
) -> None:
    frame[f"{target}_t"] = np.nan
    frame[f"{target}_age_hr"] = np.nan
    frame[f"{target}_tp{int(horizon_hours)}"] = np.nan
    target_events = events[events["target"] == target].copy()
    if target_events.empty:
        return
    target_events["event_hour"] = hours_since_base(target_events["note_time"])
    frame_hours = hours_since_base(frame["t"])
    future_hours = hours_since_base(frame["t_plus"])
    frame["_anchor_hour_tmp"] = frame_hours
    frame["_future_hour_tmp"] = future_hours

    for stay_id, idx in frame.groupby("stay_id").groups.items():
        stay_events = target_events[target_events["stay_id"] == int(stay_id)]
        if stay_events.empty:
            continue
        stay_events = stay_events.sort_values("event_hour")
        event_hours = stay_events["event_hour"].to_numpy(dtype=np.float64)
        values = stay_events["value"].to_numpy(dtype=np.float64)
        row_index = np.asarray(list(idx))
        anchors = frame.loc[row_index, "_anchor_hour_tmp"].to_numpy(dtype=np.float64)
        futures = frame.loc[row_index, "_future_hour_tmp"].to_numpy(dtype=np.float64)

        current_idx = np.searchsorted(event_hours, anchors, side="right") - 1
        valid_current = current_idx >= 0
        current_age = np.full(len(row_index), np.nan, dtype=np.float64)
        current_values = np.full(len(row_index), np.nan, dtype=np.float64)
        if valid_current.any():
            ages = anchors[valid_current] - event_hours[current_idx[valid_current]]
            keep = ages <= float(lookback_hours)
            selected_positions = np.where(valid_current)[0][keep]
            current_values[selected_positions] = values[current_idx[selected_positions]]
            current_age[selected_positions] = ages[keep]
        frame.loc[row_index, f"{target}_t"] = current_values
        frame.loc[row_index, f"{target}_age_hr"] = current_age

        future_values = np.full(len(row_index), np.nan, dtype=np.float64)
        for i, (anchor, future) in enumerate(zip(anchors, futures, strict=False)):
            left = max(anchor + 1e-9, future - float(future_tolerance_hours))
            right = future + float(future_tolerance_hours)
            start = int(np.searchsorted(event_hours, left, side="left"))
            stop = int(np.searchsorted(event_hours, right, side="right"))
            if start >= stop:
                continue
            local = np.arange(start, stop)
            best = local[np.argmin(np.abs(event_hours[local] - future))]
            future_values[i] = values[best]
        frame.loc[row_index, f"{target}_tp{int(horizon_hours)}"] = future_values

    frame.drop(columns=["_anchor_hour_tmp", "_future_hour_tmp"], inplace=True, errors="ignore")


def extract_events(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    pieces = [
        read_nurse_charting(data_root, stay_ids),
        read_nurse_assessment(data_root, stay_ids),
        read_physical_exam(data_root, stay_ids),
    ]
    return aggregate_events(pd.concat(pieces, ignore_index=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("eicu_acute_neuro_transitions_6h.parquet"))
    parser.add_argument("--data-root", type=Path, default=Path("/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0"))
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/eicu_acute_neuro_note_observation_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("eicu_neuro_note_observation_report.json"))
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--future-tolerance-hours", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    stay_ids = {int(value) for value in pd.to_numeric(frame["stay_id"], errors="coerce").dropna().unique()}
    print(f"Anchor rows: {len(frame):,}; stays: {len(stay_ids):,}", flush=True)
    events = extract_events(args.data_root, stay_ids)
    print(f"Neuro note events: {len(events):,}; stays: {events['stay_id'].nunique() if len(events) else 0:,}", flush=True)
    enhanced = frame.copy()
    for target in TARGETS:
        attach_target(
            enhanced,
            events,
            target,
            horizon_hours=args.horizon_hours,
            lookback_hours=args.lookback_hours,
            future_tolerance_hours=args.future_tolerance_hours,
        )
    enhanced.to_parquet(args.output, index=False)
    suffix = f"tp{int(args.horizon_hours)}"
    current_support = {target: int(enhanced[f"{target}_t"].notna().sum()) for target in TARGETS}
    future_support = {target: int(enhanced[f"{target}_{suffix}"].notna().sum()) for target in TARGETS}
    pair_counts = {
        target: int(enhanced[f"{target}_t"].notna().sum() if f"{target}_{suffix}" not in enhanced else (
            enhanced[f"{target}_t"].notna() & enhanced[f"{target}_{suffix}"].notna()
        ).sum())
        for target in TARGETS
    }
    report = {
        "artifact": "eICU neuro-note structured observation extraction",
        "source_cohort": args.cohort.name,
        "output": "local-only row-level neuro-note transition parquet",
        "row_level_outputs_committed": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "rows": int(len(enhanced)),
        "subjects": int(enhanced["subject_id"].nunique()),
        "stays": int(enhanced["stay_id"].nunique()),
        "hospitals": int(enhanced["hospitalid"].nunique()) if "hospitalid" in enhanced else None,
        "events": int(len(events)),
        "event_stays": int(events["stay_id"].nunique()) if len(events) else 0,
        "targets": list(TARGETS),
        "current_support": current_support,
        "future_support": future_support,
        "pair_counts": pair_counts,
        "leakage_guard": {
            "current_note_time_rule": f"note_time <= anchor t, with {args.lookback_hours:g}h lookback",
            "future_note_time_rule": f"note_time > anchor t and nearest to t+{args.horizon_hours:g}h within +/-{args.future_tolerance_hours:g}h",
            "same_source_nowcast_rule": "all neuro_* features must be excluded from same-time nowcast features",
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "events": report["events"],
        "pair_counts": pair_counts,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
