"""Build an all-ICU eICU neuro-note observation cohort.

The bounded acute-neuro note audit found near-miss signals for GCS and
delirium/CAM.  This extractor scales that specific question to all eICU stays
with timestamped neuro-note observations.  It intentionally restricts the
target set to the two supported targets:

* neuro_gcs
* neuro_delirium_present

Row-level outputs are local-only and must not be committed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_neuro_note_observation_extract import attach_target, hours_since_base, pseudo_time_from_offset
from eicu_sepsis_transition_extract import (
    BASE_TIME,
    STATE_VARS,
    assemble_states,
    read_measurements,
    read_patient_meta,
)


TARGETS = ("neuro_gcs", "neuro_delirium_present")
ANCHOR_STEP_H = 4.0
DEFAULT_MAX_ANCHORS_PER_STAY = 16


GCS_TOTAL_RE = re.compile(r"/neurologic/gcs/([3-9]|1[0-5])$")


def _finite_float(value: object) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def read_gcs_delirium_events(data_root: Path) -> pd.DataFrame:
    """Read only the supported neuro-note targets from eICU note-like tables."""

    rows: list[pd.DataFrame] = []

    chart_path = data_root / "nurseCharting.csv.gz"
    chart_cols = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ]
    for chunk in pd.read_csv(chart_path, usecols=chart_cols, chunksize=750_000):
        label = chunk["nursingchartcelltypevallabel"].fillna("").astype(str).str.lower()
        name = chunk["nursingchartcelltypevalname"].fillna("").astype(str).str.lower()
        value = chunk["nursingchartvalue"].fillna("").astype(str).str.strip().str.lower()

        gcs_mask = label.str.contains("glasgow", regex=False) & (name == "gcs total")
        if gcs_mask.any():
            gcs = chunk.loc[gcs_mask, ["patientunitstayid", "nursingchartoffset", "nursingchartvalue"]].copy()
            gcs["value"] = pd.to_numeric(gcs["nursingchartvalue"], errors="coerce")
            gcs = gcs[(gcs["value"] >= 3) & (gcs["value"] <= 15)]
            if not gcs.empty:
                rows.append(pd.DataFrame({
                    "stay_id": gcs["patientunitstayid"].astype("int64"),
                    "note_time": gcs["nursingchartoffset"].map(pseudo_time_from_offset),
                    "target": "neuro_gcs",
                    "value": gcs["value"].astype("float64"),
                }))

        delirium_mask = label.str.contains("delirium", regex=False)
        if delirium_mask.any():
            delirium = chunk.loc[delirium_mask, ["patientunitstayid", "nursingchartoffset"]].copy()
            delirium_value = value.loc[delirium_mask]
            mapped = delirium_value.map({
                "yes": 1.0,
                "y": 1.0,
                "positive": 1.0,
                "present": 1.0,
                "1": 1.0,
                "no": 0.0,
                "n": 0.0,
                "negative": 0.0,
                "absent": 0.0,
                "0": 0.0,
            })
            delirium["value"] = pd.to_numeric(mapped, errors="coerce")
            delirium = delirium.dropna(subset=["value"])
            if not delirium.empty:
                rows.append(pd.DataFrame({
                    "stay_id": delirium["patientunitstayid"].astype("int64"),
                    "note_time": delirium["nursingchartoffset"].map(pseudo_time_from_offset),
                    "target": "neuro_delirium_present",
                    "value": delirium["value"].astype("float64"),
                }))

    exam_path = data_root / "physicalExam.csv.gz"
    exam_cols = [
        "patientunitstayid",
        "physicalexamoffset",
        "physicalexampath",
    ]
    for chunk in pd.read_csv(exam_path, usecols=exam_cols, chunksize=750_000):
        path_text = chunk["physicalexampath"].fillna("").astype(str).str.lower()
        extracted = path_text.str.extract(GCS_TOTAL_RE, expand=False)
        mask = extracted.notna()
        if mask.any():
            rows.append(pd.DataFrame({
                "stay_id": chunk.loc[mask, "patientunitstayid"].astype("int64"),
                "note_time": chunk.loc[mask, "physicalexamoffset"].map(pseudo_time_from_offset),
                "target": "neuro_gcs",
                "value": pd.to_numeric(extracted.loc[mask], errors="coerce").astype("float64"),
            }))

    if not rows:
        return pd.DataFrame(columns=["stay_id", "note_time", "target", "value"])
    events = pd.concat(rows, ignore_index=True).dropna(subset=["stay_id", "note_time", "target", "value"])
    events["stay_id"] = events["stay_id"].astype("int64")
    events["value"] = pd.to_numeric(events["value"], errors="coerce")
    events = events[np.isfinite(events["value"])].copy()
    return (
        events.groupby(["stay_id", "note_time", "target"], as_index=False)["value"]
        .mean()
        .sort_values(["stay_id", "target", "note_time"])
        .reset_index(drop=True)
    )


def build_event_anchors(
    events: pd.DataFrame,
    meta: pd.DataFrame,
    horizon_hours: float,
    max_anchors_per_stay: int,
) -> pd.DataFrame:
    rows = []
    if events.empty:
        return pd.DataFrame()
    events = events.copy()
    events["event_hour"] = hours_since_base(events["note_time"])
    events = events[np.isfinite(events["event_hour"]) & (events["event_hour"] >= 0.0)]
    for stay_id, group in events.groupby("stay_id", sort=False):
        if int(stay_id) not in meta.index:
            continue
        discharge = meta.loc[int(stay_id), "unitdischarge_hr"]
        max_hour = float(discharge) - float(horizon_hours) if np.isfinite(discharge) else np.inf
        group = group[group["event_hour"] <= max_hour]
        if group.empty:
            continue
        # Keep at most one anchor per coarse time bucket, using the latest
        # neuro observation in that bucket so the current target is known.
        group = group.sort_values("event_hour").copy()
        group["bucket"] = np.floor(group["event_hour"] / ANCHOR_STEP_H).astype("int64")
        anchors = group.groupby("bucket", as_index=False).tail(1)
        if max_anchors_per_stay > 0 and len(anchors) > max_anchors_per_stay:
            positions = np.linspace(0, len(anchors) - 1, max_anchors_per_stay).round().astype(int)
            anchors = anchors.iloc[sorted(set(int(pos) for pos in positions))]
        subject = str(meta.loc[int(stay_id), "subject_id"])
        hospital = int(meta.loc[int(stay_id), "hospitalid"]) if pd.notna(meta.loc[int(stay_id), "hospitalid"]) else -1
        unit = str(meta.loc[int(stay_id), "unittype"])
        for anchor_hour in anchors["event_hour"].to_numpy(dtype=np.float64):
            rows.append({
                "subject_id": subject,
                "stay_id": int(stay_id),
                "hospitalid": hospital,
                "unittype": unit,
                "onset": BASE_TIME,
                "onset_hour": 0.0,
                "onset_criteria": "all_icu_neuro_note_observed",
                "hours_since_onset": float(anchor_hour),
                "t": BASE_TIME + pd.Timedelta(hours=float(anchor_hour)),
                "t_hour": float(anchor_hour),
                "t_plus": BASE_TIME + pd.Timedelta(hours=float(anchor_hour + horizon_hours)),
                "t_plus_hour": float(anchor_hour + horizon_hours),
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("eicu-collaborative-research-database-2.0"))
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/eicu_neuro_note_all_icu_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("/private/tmp/eicu_neuro_note_all_icu_observation_report.json"))
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--future-tolerance-hours", type=float, default=2.0)
    parser.add_argument("--max-anchors-per-stay", type=int, default=DEFAULT_MAX_ANCHORS_PER_STAY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = read_patient_meta(args.data_root)
    print(f"eICU stays: {len(meta):,}", flush=True)
    events = read_gcs_delirium_events(args.data_root)
    stay_ids = {int(value) for value in events["stay_id"].dropna().unique()}
    print(f"Neuro events: {len(events):,}; stays: {len(stay_ids):,}", flush=True)
    measurements = read_measurements(args.data_root, stay_ids)
    print(f"Measurements: {len(measurements):,}; stays: {measurements['stay_id'].nunique() if len(measurements) else 0:,}", flush=True)
    anchors = build_event_anchors(events, meta, args.horizon_hours, args.max_anchors_per_stay)
    if measurements.empty or anchors.empty:
        raise SystemExit("no anchors or measurements available")
    measured_stays = set(measurements["stay_id"].astype("int64").unique())
    anchors = anchors[anchors["stay_id"].isin(measured_stays)].reset_index(drop=True)
    print(f"Anchors: {len(anchors):,}; stays: {anchors['stay_id'].nunique():,}", flush=True)
    states = assemble_states(anchors, measurements, horizon_hours=args.horizon_hours)
    frame = pd.concat([anchors.reset_index(drop=True), states.reset_index(drop=True)], axis=1)
    for target in TARGETS:
        attach_target(
            frame,
            events,
            target,
            horizon_hours=args.horizon_hours,
            lookback_hours=args.lookback_hours,
            future_tolerance_hours=args.future_tolerance_hours,
        )
    frame.to_parquet(args.output, index=False)
    suffix = f"tp{int(args.horizon_hours)}"
    current_support = {target: int(frame[f"{target}_t"].notna().sum()) for target in TARGETS}
    future_support = {target: int(frame[f"{target}_{suffix}"].notna().sum()) for target in TARGETS}
    pair_counts = {
        target: int((frame[f"{target}_t"].notna() & frame[f"{target}_{suffix}"].notna()).sum())
        for target in TARGETS
    }
    report = {
        "artifact": "eICU all-ICU neuro-note structured observation extraction",
        "output": "local-only all-ICU neuro-note transition parquet",
        "row_level_outputs_committed": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "events": int(len(events)),
        "event_stays": int(len(stay_ids)),
        "measurement_rows": int(len(measurements)),
        "targets": list(TARGETS),
        "current_support": current_support,
        "future_support": future_support,
        "pair_counts": pair_counts,
        "anchor_policy": {
            "source": "timestamped GCS/delirium observations",
            "step_hours": ANCHOR_STEP_H,
            "max_anchors_per_stay": int(args.max_anchors_per_stay),
        },
        "leakage_guard": {
            "current_note_time_rule": f"note_time <= anchor t, with {args.lookback_hours:g}h lookback",
            "future_note_time_rule": f"note_time > anchor t and nearest to t+{args.horizon_hours:g}h within +/-{args.future_tolerance_hours:g}h",
            "same_source_nowcast_rule": "all neuro_* features must be excluded from same-time nowcast features",
        },
        "state_variables": list(STATE_VARS),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "rows": report["rows"],
        "subjects": report["subjects"],
        "pair_counts": pair_counts,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
