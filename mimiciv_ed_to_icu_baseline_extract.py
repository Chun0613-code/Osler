"""Build ED-to-early-ICU baseline transition cohorts.

This adapter tests whether a patient's ED trajectory improves early ICU factual
prediction.  It is deliberately conservative:

* ED stays must belong to the same admission as the ICU stay;
* the ED stay must end before ICU `intime`;
* only aggregate reports are committed;
* row-level parquet outputs remain local-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mimiciv_ed_observation_extract import STATE_VARS as ED_STATE_VARS
from mimiciv_ed_observation_extract import read_measurements as read_ed_measurements
from mimiciv_ed_observation_extract import read_stays as read_ed_stays
from mimiciv_observation_transition_extract import (
    MIMIC_DIR,
    STATE_VARS as ICU_STATE_VARS,
    assemble_state,
    con,
    csv_expr,
    icu_path,
    pull_measurements,
    register_stays,
)


DEFAULT_ED_DIR = Path(os.environ.get("MIMIC_ED_DIR", "mimic-iv-ed-2.2"))
DEFAULT_MAX_ED_TO_ICU_GAP_H = 24.0
DEFAULT_ANCHOR_OFFSET_H = 1.0


def pull_icu_meta(c, mimic_dir: Path, horizon_hours: float) -> pd.DataFrame:
    frame = c.execute(f"""
        SELECT subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime
        FROM {csv_expr(icu_path(mimic_dir, 'icustays'))}
        WHERE outtime > intime + INTERVAL '{float(DEFAULT_ANCHOR_OFFSET_H + horizon_hours):g} hours'
        ORDER BY stay_id
    """).df()
    for column in ("intime", "outtime"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    frame["subject_id"] = frame["subject_id"].astype("int64")
    frame["hadm_id"] = frame["hadm_id"].astype("int64")
    frame["stay_id"] = frame["stay_id"].astype("int64")
    return frame


def link_prior_ed_to_icu(
    icu: pd.DataFrame,
    ed: pd.DataFrame,
    max_gap_hours: float,
) -> pd.DataFrame:
    ed = ed.dropna(subset=["hadm_id"]).copy()
    ed["hadm_id"] = ed["hadm_id"].astype("int64")
    ed = ed.rename(columns={
        "stay_id": "ed_stay_id",
        "intime": "ed_intime",
        "outtime": "ed_outtime",
        "ed_los_hours": "ed_los_hours",
    })
    linked = icu.merge(
        ed[[
            "subject_id",
            "hadm_id",
            "ed_stay_id",
            "ed_intime",
            "ed_outtime",
            "arrival_transport",
            "disposition",
            "ed_los_hours",
        ]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    linked["ed_to_icu_gap_hr"] = (
        linked["intime"] - linked["ed_outtime"]
    ).dt.total_seconds() / 3600.0
    linked = linked[
        (linked["ed_to_icu_gap_hr"] >= 0.0)
        & (linked["ed_to_icu_gap_hr"] <= float(max_gap_hours))
    ].copy()
    if linked.empty:
        return linked
    linked = (
        linked.sort_values(["stay_id", "ed_to_icu_gap_hr"])
        .groupby("stay_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return linked


def summarize_ed_trajectory(ed_measurements: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    ed_ids = set(linked["ed_stay_id"].astype("int64").unique())
    measures = ed_measurements[ed_measurements["stay_id"].isin(ed_ids)].copy()
    measures = measures.rename(columns={"stay_id": "ed_stay_id"})
    if measures.empty:
        return linked.copy()

    wide_parts = []
    for variable in ED_STATE_VARS:
        part = measures[measures["var"] == variable].sort_values(["ed_stay_id", "charttime"])
        if part.empty:
            continue
        first = part.groupby("ed_stay_id", as_index=False).head(1)[["ed_stay_id", "valuenum"]]
        first = first.rename(columns={"valuenum": f"ed_{variable}_first"})
        last = part.groupby("ed_stay_id", as_index=False).tail(1)[["ed_stay_id", "charttime", "valuenum"]]
        last = last.rename(columns={"charttime": f"ed_{variable}_last_time", "valuenum": f"ed_{variable}_last"})
        mean = part.groupby("ed_stay_id", as_index=False)["valuenum"].mean().rename(columns={"valuenum": f"ed_{variable}_mean"})
        merged = first.merge(last, on="ed_stay_id", how="outer").merge(mean, on="ed_stay_id", how="outer")
        merged[f"ed_{variable}_delta"] = merged[f"ed_{variable}_last"] - merged[f"ed_{variable}_first"]
        wide_parts.append(merged)

    out = linked.copy()
    for part in wide_parts:
        out = out.merge(part, on="ed_stay_id", how="left")

    for variable in ED_STATE_VARS:
        time_col = f"ed_{variable}_last_time"
        if time_col in out:
            age = (out["intime"] - pd.to_datetime(out[time_col], errors="coerce")).dt.total_seconds() / 3600.0
            out[f"ed_{variable}_age_to_icu_hr"] = age
            out = out.drop(columns=[time_col])
    return out


def build_anchors(linked: pd.DataFrame, anchor_offset_hours: float, horizon_hours: float) -> pd.DataFrame:
    anchors = linked.copy()
    anchors["t"] = anchors["intime"] + pd.Timedelta(hours=float(anchor_offset_hours))
    anchors["t_plus"] = anchors["t"] + pd.Timedelta(hours=float(horizon_hours))
    anchors = anchors[anchors["t_plus"] <= anchors["outtime"]].copy()
    anchors["hours_since_onset"] = float(anchor_offset_hours)
    anchors["subject_id"] = anchors["subject_id"].astype(str)
    return anchors.reset_index(drop=True)


def pair_counts(frame: pd.DataFrame, future_suffix: str, targets: tuple[str, ...]) -> dict[str, int]:
    counts = {}
    for variable in targets:
        current = f"{variable}_t"
        future = f"{variable}_{future_suffix}"
        if current in frame and future in frame:
            counts[variable] = int((frame[current].notna() & frame[future].notna()).sum())
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--ed-dir", type=Path, default=DEFAULT_ED_DIR)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/mimiciv_ed_to_icu_baseline_transitions_1h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("/private/tmp/mimiciv_ed_to_icu_baseline_transition_report_1h.json"))
    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--anchor-offset-hours", type=float, default=DEFAULT_ANCHOR_OFFSET_H)
    parser.add_argument("--lookback-hours", type=float, default=2.0)
    parser.add_argument("--target-tolerance-hours", type=float, default=0.75)
    parser.add_argument("--max-ed-to-icu-gap-hours", type=float, default=DEFAULT_MAX_ED_TO_ICU_GAP_H)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    future_suffix = f"tp{int(args.horizon_hours)}" if float(args.horizon_hours).is_integer() else f"tp{args.horizon_hours:g}"
    c = con()
    icu = pull_icu_meta(c, args.mimic_dir, args.horizon_hours)
    ed = read_ed_stays(args.ed_dir)
    linked = link_prior_ed_to_icu(icu, ed, args.max_ed_to_icu_gap_hours)
    if linked.empty:
        raise SystemExit("no prior ED-to-ICU links available")
    ed_measurements, ed_vital_rows, ed_triage_rows = read_ed_measurements(args.ed_dir, ed)
    linked = summarize_ed_trajectory(ed_measurements, linked)
    anchors = build_anchors(linked, args.anchor_offset_hours, args.horizon_hours)
    register_stays(c, anchors[["stay_id"]].drop_duplicates())
    icu_measurements = pull_measurements(c, args.mimic_dir)
    state_t = assemble_state(anchors, icu_measurements, "_t", "t", "backward", args.lookback_hours)
    state_f = assemble_state(anchors, icu_measurements, f"_{future_suffix}", "t_plus", "nearest", args.target_tolerance_hours)
    id_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "ed_stay_id",
        "first_careunit",
        "last_careunit",
        "arrival_transport",
        "disposition",
        "intime",
        "outtime",
        "ed_intime",
        "ed_outtime",
        "t",
        "t_plus",
        "hours_since_onset",
    ]
    ed_feature_cols = [column for column in anchors.columns if column.startswith("ed_") and column not in {"ed_stay_id", "ed_intime", "ed_outtime"}]
    frame = pd.concat(
        [
            anchors[id_cols + ed_feature_cols].reset_index(drop=True),
            state_t.drop(columns=["stay_id", "t"]).reset_index(drop=True),
            state_f.drop(columns=["stay_id", "t_plus"]).reset_index(drop=True),
        ],
        axis=1,
    )
    targets = tuple(variable for variable in ICU_STATE_VARS if f"{variable}_t" in frame and f"{variable}_{future_suffix}" in frame)
    has_pair = np.zeros(len(frame), dtype=bool)
    for variable in targets:
        has_pair |= frame[f"{variable}_t"].notna().to_numpy() & frame[f"{variable}_{future_suffix}"].notna().to_numpy()
    frame = frame.loc[has_pair].reset_index(drop=True)
    frame.to_parquet(args.output, index=False)
    counts = pair_counts(frame, future_suffix, targets)
    report = {
        "artifact": "MIMIC-IV-ED to early-ICU baseline transition cohort",
        "output": "local-only ED-to-ICU transition parquet",
        "horizon_hours": float(args.horizon_hours),
        "future_suffix": future_suffix,
        "anchor_offset_hours": float(args.anchor_offset_hours),
        "lookback_hours": float(args.lookback_hours),
        "target_tolerance_hours": float(args.target_tolerance_hours),
        "max_ed_to_icu_gap_hours": float(args.max_ed_to_icu_gap_hours),
        "icu_stays_scanned": int(len(icu)),
        "linked_prior_ed_icu_stays": int(len(linked)),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "icu_stays": int(frame["stay_id"].nunique()),
        "ed_stays": int(frame["ed_stay_id"].nunique()),
        "first_careunits": int(frame["first_careunit"].nunique()),
        "ed_vitalsign_measurement_rows": int(ed_vital_rows),
        "ed_triage_measurement_rows": int(ed_triage_rows),
        "icu_measurement_rows": int(len(icu_measurements)),
        "ed_feature_count": int(len(ed_feature_cols)),
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
