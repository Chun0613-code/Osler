"""Build MIMIC-IV-ED observation transition cohorts.

This adapter treats the emergency department as a new observation scene rather
than a treatment or causal module.  It maps ED triage and vital-sign tables into
the same ``*_t`` / ``*_tp{h}`` factual observation contract used by the ICU
whole-body audits.

Row-level outputs are local-only.  Committed reports must remain aggregate-only:
no patient identifiers, no row-level values, and no local filesystem paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ED_DIR = Path(os.environ.get("MIMIC_ED_DIR", "mimic-iv-ed-2.2"))
STATE_VARS = (
    "acuity",
    "dbp",
    "heart_rate",
    "map",
    "o2sat",
    "pain",
    "respiratory_rate",
    "sbp",
    "temperature",
)

PLAUSIBLE = {
    "acuity": (1.0, 5.0),
    "dbp": (10.0, 200.0),
    "heart_rate": (20.0, 250.0),
    "map": (20.0, 250.0),
    "o2sat": (40.0, 100.0),
    "pain": (0.0, 10.0),
    "respiratory_rate": (1.0, 80.0),
    "sbp": (40.0, 300.0),
    "temperature": (25.0, 45.0),
}


def ed_path(ed_dir: Path, name: str) -> Path:
    base = ed_dir / "ed"
    return base / f"{name}.csv.gz"


def clean_temperature(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fahrenheit = numeric > 60.0
    numeric = numeric.where(~fahrenheit, (numeric - 32.0) * (5.0 / 9.0))
    return numeric


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def normalize_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["subject_id", "stay_id", "charttime", "var", "valuenum"])
    frame = frame.copy()
    frame["charttime"] = pd.to_datetime(frame["charttime"], errors="coerce")
    frame = frame.dropna(subset=["charttime"])
    rows: list[pd.DataFrame] = []
    mapping = {
        "acuity": "acuity",
        "temperature": "temperature",
        "heartrate": "heart_rate",
        "resprate": "respiratory_rate",
        "o2sat": "o2sat",
        "sbp": "sbp",
        "dbp": "dbp",
        "pain": "pain",
    }
    for source, target in mapping.items():
        if source not in frame:
            continue
        values = clean_temperature(frame[source]) if target == "temperature" else _numeric(frame, source)
        rows.append(pd.DataFrame({
            "subject_id": frame["subject_id"].astype("int64"),
            "stay_id": frame["stay_id"].astype("int64"),
            "charttime": frame["charttime"],
            "var": target,
            "valuenum": values,
        }))
    sbp = _numeric(frame, "sbp")
    dbp = _numeric(frame, "dbp")
    calculated_map = dbp + (sbp - dbp) / 3.0
    rows.append(pd.DataFrame({
        "subject_id": frame["subject_id"].astype("int64"),
        "stay_id": frame["stay_id"].astype("int64"),
        "charttime": frame["charttime"],
        "var": "map",
        "valuenum": calculated_map,
    }))
    out = pd.concat(rows, ignore_index=True)
    out["valuenum"] = pd.to_numeric(out["valuenum"], errors="coerce").astype("float64")
    out = out.dropna(subset=["valuenum"])
    for variable, (lo, hi) in PLAUSIBLE.items():
        mask = out["var"] == variable
        if mask.any():
            out.loc[mask, "valuenum"] = out.loc[mask, "valuenum"].where(
                (out.loc[mask, "valuenum"] >= lo) & (out.loc[mask, "valuenum"] <= hi)
            )
    out = out.dropna(subset=["valuenum"])
    return (
        out.groupby(["subject_id", "stay_id", "charttime", "var"], as_index=False)["valuenum"]
        .mean()
        .sort_values(["stay_id", "charttime", "var"])
        .reset_index(drop=True)
    )


def read_stays(ed_dir: Path) -> pd.DataFrame:
    stays = pd.read_csv(
        ed_path(ed_dir, "edstays"),
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "gender",
            "race",
            "arrival_transport",
            "disposition",
        ],
    )
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    stays = stays.dropna(subset=["intime", "outtime"])
    stays = stays[stays["outtime"] > stays["intime"]].copy()
    stays["ed_los_hours"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
    stays["subject_id"] = stays["subject_id"].astype("int64")
    stays["stay_id"] = stays["stay_id"].astype("int64")
    stays["arrival_transport"] = stays["arrival_transport"].fillna("UNKNOWN").astype(str)
    stays["disposition"] = stays["disposition"].fillna("UNKNOWN").astype(str)
    return stays


def read_measurements(ed_dir: Path, stays: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    vitals = pd.read_csv(ed_path(ed_dir, "vitalsign"))
    vitals = normalize_measurements(vitals)
    triage = pd.read_csv(ed_path(ed_dir, "triage"))
    triage = triage.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
    triage["charttime"] = triage["intime"]
    triage = normalize_measurements(triage)
    measurements = pd.concat([triage, vitals], ignore_index=True)
    measurements = measurements.merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    measurements = measurements[
        (measurements["charttime"] >= measurements["intime"])
        & (measurements["charttime"] <= measurements["outtime"])
    ].copy()
    measurements = measurements.drop(columns=["intime", "outtime"])
    return measurements, int(len(vitals)), int(len(triage))


def build_anchors(
    stays: pd.DataFrame,
    measurements: pd.DataFrame,
    horizon_hours: float,
    anchor_step_hours: float,
    max_anchors_per_stay: int,
) -> pd.DataFrame:
    vital_times = (
        measurements[measurements["var"] != "acuity"][["stay_id", "charttime"]]
        .drop_duplicates()
        .merge(stays[["stay_id", "subject_id", "intime", "outtime", "arrival_transport", "disposition", "ed_los_hours"]], on="stay_id", how="inner")
    )
    vital_times["hour"] = (vital_times["charttime"] - vital_times["intime"]).dt.total_seconds() / 3600.0
    vital_times = vital_times[
        (vital_times["hour"] >= 0.0)
        & (vital_times["charttime"] + pd.Timedelta(hours=float(horizon_hours)) <= vital_times["outtime"])
    ].copy()
    if vital_times.empty:
        return pd.DataFrame()
    vital_times["bucket"] = np.floor(vital_times["hour"] / float(anchor_step_hours)).astype("int64")
    rows = []
    for stay_id, group in vital_times.sort_values("charttime").groupby("stay_id", sort=False):
        anchors = group.groupby("bucket", as_index=False).tail(1)
        if max_anchors_per_stay > 0 and len(anchors) > max_anchors_per_stay:
            positions = np.linspace(0, len(anchors) - 1, max_anchors_per_stay).round().astype(int)
            anchors = anchors.iloc[sorted(set(int(pos) for pos in positions))]
        for record in anchors.itertuples(index=False):
            rows.append({
                "subject_id": str(record.subject_id),
                "stay_id": int(stay_id),
                "first_careunit": f"ED:{record.arrival_transport}",
                "last_careunit": f"ED:{record.disposition}",
                "arrival_transport": str(record.arrival_transport),
                "disposition": str(record.disposition),
                "t": record.charttime,
                "t_plus": record.charttime + pd.Timedelta(hours=float(horizon_hours)),
                "hours_since_onset": float(record.hour),
                "ed_los_hours": float(record.ed_los_hours),
            })
    return pd.DataFrame(rows)


def assemble_state(
    anchors: pd.DataFrame,
    measurements: pd.DataFrame,
    suffix: str,
    time_column: str,
    direction: str,
    tolerance_hours: float,
) -> pd.DataFrame:
    pieces: dict[str, object] = {
        "stay_id": anchors["stay_id"].to_numpy(),
        time_column: anchors[time_column].to_numpy(),
    }
    left = anchors[["stay_id", time_column]].copy()
    left[time_column] = pd.to_datetime(left[time_column]).astype("datetime64[ns]")
    left_sorted = left.sort_values(time_column)
    tolerance = pd.Timedelta(hours=float(tolerance_hours))
    for variable in STATE_VARS:
        values = measurements.loc[
            measurements["var"] == variable,
            ["stay_id", "charttime", "valuenum"],
        ].copy()
        if values.empty:
            pieces[f"{variable}{suffix}"] = np.full(len(anchors), np.nan, dtype=np.float64)
            if suffix == "_t":
                pieces[f"{variable}_age_hr"] = np.full(len(anchors), np.nan, dtype=np.float64)
            continue
        values["charttime"] = pd.to_datetime(values["charttime"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            left_sorted,
            values.sort_values("charttime"),
            left_on=time_column,
            right_on="charttime",
            by="stay_id",
            direction=direction,
            tolerance=tolerance,
        )
        merged = merged.set_index(left_sorted.index).reindex(left.index)
        pieces[f"{variable}{suffix}"] = merged["valuenum"].to_numpy(dtype=np.float64)
        if suffix == "_t":
            age = (merged[time_column] - merged["charttime"]).dt.total_seconds() / 3600.0
            pieces[f"{variable}_age_hr"] = age.to_numpy(dtype=np.float64)
    return pd.DataFrame(pieces, index=anchors.index)


def pair_counts(frame: pd.DataFrame, future_suffix: str) -> dict[str, int]:
    counts = {}
    for variable in STATE_VARS:
        current = f"{variable}_t"
        future = f"{variable}_{future_suffix}"
        if current in frame and future in frame:
            counts[variable] = int((frame[current].notna() & frame[future].notna()).sum())
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ed-dir", type=Path, default=DEFAULT_ED_DIR)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/mimiciv_ed_observation_transitions_1h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("mimiciv_ed_observation_transition_report_1h.json"))
    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--lookback-hours", type=float, default=2.0)
    parser.add_argument("--target-tolerance-hours", type=float, default=0.75)
    parser.add_argument("--anchor-step-hours", type=float, default=1.0)
    parser.add_argument("--max-anchors-per-stay", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stays = read_stays(args.ed_dir)
    measurements, vital_rows, triage_rows = read_measurements(args.ed_dir, stays)
    anchors = build_anchors(
        stays,
        measurements,
        args.horizon_hours,
        args.anchor_step_hours,
        args.max_anchors_per_stay,
    )
    if anchors.empty:
        raise SystemExit("no ED anchors available")
    state_t = assemble_state(anchors, measurements, "_t", "t", "backward", args.lookback_hours)
    future_suffix = f"tp{int(args.horizon_hours)}" if float(args.horizon_hours).is_integer() else f"tp{args.horizon_hours:g}"
    state_f = assemble_state(anchors, measurements, f"_{future_suffix}", "t_plus", "nearest", args.target_tolerance_hours)
    frame = pd.concat(
        [
            anchors[[
                "subject_id",
                "stay_id",
                "first_careunit",
                "last_careunit",
                "arrival_transport",
                "disposition",
                "t",
                "t_plus",
                "hours_since_onset",
                "ed_los_hours",
            ]].reset_index(drop=True),
            state_t.drop(columns=["stay_id", "t"]).reset_index(drop=True),
            state_f.drop(columns=["stay_id", "t_plus"]).reset_index(drop=True),
        ],
        axis=1,
    )
    has_pair = np.zeros(len(frame), dtype=bool)
    for variable in STATE_VARS:
        has_pair |= frame[f"{variable}_t"].notna().to_numpy() & frame[f"{variable}_{future_suffix}"].notna().to_numpy()
    frame = frame.loc[has_pair].reset_index(drop=True)
    frame.to_parquet(args.output, index=False)
    counts = pair_counts(frame, future_suffix)
    report = {
        "artifact": "MIMIC-IV-ED v2.2 ED-scene observation transition cohort",
        "output": "local-only ED observation transition parquet",
        "horizon_hours": float(args.horizon_hours),
        "future_suffix": future_suffix,
        "anchor_step_hours": float(args.anchor_step_hours),
        "lookback_hours": float(args.lookback_hours),
        "target_tolerance_hours": float(args.target_tolerance_hours),
        "max_anchors_per_stay": int(args.max_anchors_per_stay),
        "selected_ed_stays": int(len(stays)),
        "vitalsign_measurement_rows": int(vital_rows),
        "triage_measurement_rows": int(triage_rows),
        "aggregate_measurement_rows": int(len(measurements)),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "arrival_transport_groups": int(frame["arrival_transport"].nunique()),
        "disposition_groups": int(frame["disposition"].nunique()),
        "state_variables": list(STATE_VARS),
        "pair_counts": counts,
        "eligible_pair_targets": int(sum(count >= 1 for count in counts.values())),
        "patient_ids_included_in_report": False,
        "row_level_outputs_committed": False,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "rows": report["rows"],
        "subjects": report["subjects"],
        "future_suffix": future_suffix,
        "eligible_pair_targets": report["eligible_pair_targets"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
