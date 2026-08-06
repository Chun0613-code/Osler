"""Audit whether MIMIC-IV Waveform features improve factual forecasts.

Baseline is the current table-based transition cohort, including already
observed treatment context when present.  Candidate adds bounded waveform
features observed before the anchor.  Placebo adds the same number of random
features.  This is a factual accuracy audit only: no causal, counterfactual, or
clinical authority is granted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_full_variable_coverage_audit import forecast_feature_columns, forecast_rows
from eicu_nowcasting_audit import DEFAULT_SEEDS
from eicu_sepsis_target_router import split_subjects
from mimiciv_observation_transition_extract import MIMIC_DIR, csv_expr, icu_path
from mimiciv_treatment_context_accuracy_audit import audit_split, split_passed, summarize_random
from mimiciv_cross_database_coverage_audit import add_time_holdout_column
from osler_jepa.waveform import parse_wfdb_header_text
from osler_jepa.waveform_features import read_waveform_feature_window


DEFAULT_TARGETS = ("heart_rate", "map", "o2sat", "respiratory_rate")


def parse_master_header(master_path: Path) -> dict[str, object] | None:
    """Parse a MIMIC4WDB multi-segment master header into segment rows."""

    lines = [line.strip() for line in master_path.read_text(errors="replace").splitlines() if line.strip()]
    header_line = next((line for line in lines if not line.startswith("#")), None)
    if header_line is None or "/" not in header_line.split()[0]:
        return None
    header = header_line.split()
    record_name = header[0].split("/", 1)[0]
    fs = float(header[2].split("/", 1)[0])
    sample_count = int(float(header[3]))
    base_datetime = pd.to_datetime(f"{header[5]} {header[4]}", dayfirst=True, errors="coerce")
    comments = {match.group(1): match.group(2) for line in lines for match in [re.match(r"#\s*(subject_id|hadm_id)\s+(\d+)", line)] if match}
    if pd.isna(base_datetime) or "subject_id" not in comments:
        return None
    segment_lines = []
    saw_header = False
    for line in lines:
        if line.startswith("#"):
            continue
        if not saw_header:
            saw_header = True
            continue
        parts = line.split()
        if len(parts) >= 2:
            segment_lines.append((parts[0], int(float(parts[1]))))
    return {
        "record_name": record_name,
        "subject_id": str(comments["subject_id"]),
        "hadm_id": str(comments.get("hadm_id")) if comments.get("hadm_id") else None,
        "base_datetime": base_datetime,
        "fs": fs,
        "sample_count": sample_count,
        "segments": segment_lines,
        "master_path": str(master_path),
    }


def build_segment_index(waveform_root: Path, mimic_dir: Path) -> pd.DataFrame:
    """Build segment/stay alignment metadata for local MIMIC4WDB files."""

    import duckdb

    rows = []
    for master_path in sorted(waveform_root.glob("waves/*/*/*/*.hea")):
        if "_" in master_path.stem:
            continue
        parsed = parse_master_header(master_path)
        if not parsed:
            continue
        offset = 0
        for segment_name, length in parsed["segments"]:
            segment_path = master_path.parent / segment_name
            header_path = segment_path.with_suffix(".hea")
            if length <= 0 or not header_path.exists():
                offset += max(length, 0)
                continue
            try:
                segment_header = parse_wfdb_header_text(header_path.read_text(errors="replace"))
            except ValueError:
                offset += max(length, 0)
                continue
            if segment_header.header_kind != "signal_segment" or not segment_header.signal_groups:
                offset += max(length, 0)
                continue
            start = parsed["base_datetime"] + pd.to_timedelta(offset / parsed["fs"], unit="s")
            end = parsed["base_datetime"] + pd.to_timedelta((offset + length) / parsed["fs"], unit="s")
            rows.append({
                "subject_id": str(parsed["subject_id"]),
                "hadm_id": str(parsed["hadm_id"]) if parsed["hadm_id"] else None,
                "waveform_record": str(parsed["record_name"]),
                "segment_name": segment_name,
                "segment_path": str(segment_path),
                "segment_start": start,
                "segment_end": end,
                "segment_offset_samples": int(offset),
                "segment_samples": int(length),
                "fs": float(parsed["fs"]),
                "signal_groups": "|".join(segment_header.signal_groups),
            })
            offset += max(length, 0)
    segments = pd.DataFrame(rows)
    if segments.empty:
        return segments

    con = duckdb.connect()
    icustays = con.execute(f"""
        SELECT subject_id::VARCHAR AS subject_id,
               hadm_id::VARCHAR AS hadm_id,
               stay_id,
               intime,
               outtime
        FROM {csv_expr(icu_path(mimic_dir, 'icustays'))}
    """).df()
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    merged = segments.merge(icustays, on=["subject_id", "hadm_id"], how="left")
    merged = merged[
        (merged["stay_id"].notna())
        & (merged["segment_end"] >= merged["intime"])
        & (merged["segment_start"] <= merged["outtime"])
    ].copy()
    if not merged.empty:
        merged["stay_id"] = merged["stay_id"].astype("int64")
    return merged.reset_index(drop=True)


def candidate_anchor_rows(
    transitions: pd.DataFrame,
    segments: pd.DataFrame,
    seconds: float,
    max_anchors: int,
    seed: int,
) -> pd.DataFrame:
    """Return transition rows with a usable waveform segment before anchor."""

    transitions = transitions.copy()
    transitions["subject_id"] = transitions["subject_id"].astype(str)
    merged = transitions.merge(
        segments,
        on=["subject_id", "stay_id"],
        how="inner",
    )
    earliest = merged["segment_start"] + pd.to_timedelta(seconds, unit="s")
    eligible = merged[(merged["t"] >= earliest) & (merged["t"] <= merged["segment_end"])].copy()
    if eligible.empty:
        return eligible
    eligible["distance_to_end_s"] = (eligible["segment_end"] - eligible["t"]).dt.total_seconds().abs()
    eligible = eligible.sort_values(["subject_id", "stay_id", "t", "distance_to_end_s"])
    eligible = eligible.drop_duplicates(["subject_id", "stay_id", "t"], keep="first")
    if len(eligible) > int(max_anchors):
        rng = np.random.default_rng(seed)
        subjects = eligible["subject_id"].drop_duplicates().to_numpy(dtype=object)
        rng.shuffle(subjects)
        keep_subjects = []
        total = 0
        counts = eligible.groupby("subject_id").size()
        for subject in subjects:
            keep_subjects.append(subject)
            total += int(counts.loc[subject])
            if total >= int(max_anchors):
                break
        eligible = eligible[eligible["subject_id"].isin(set(keep_subjects))].copy()
    return eligible.reset_index(drop=True)


def attach_waveform_features(rows: pd.DataFrame, seconds: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read waveform windows before anchors and append feature columns."""

    output_rows = []
    failures = 0
    for _, row in rows.iterrows():
        fs = float(row["fs"])
        sampfrom = int(round((row["t"] - row["segment_start"]).total_seconds() * fs - seconds * fs))
        try:
            record = read_waveform_feature_window(row["segment_path"], sampfrom=sampfrom, seconds=seconds)
        except Exception:
            failures += 1
            continue
        payload = row.to_dict()
        payload.update(record.features)
        output_rows.append(payload)
    frame = pd.DataFrame(output_rows)
    wave_features = [column for column in frame.columns if column.startswith("wave_")] if not frame.empty else []
    summary = {
        "candidate_anchor_rows": int(len(rows)),
        "feature_rows": int(len(frame)),
        "read_failures": int(failures),
        "wave_feature_count": int(len(wave_features)),
        "wave_features": sorted(wave_features),
        "signal_group_counts": (
            frame["signal_groups"].str.get_dummies(sep="|").sum().astype(int).to_dict()
            if not frame.empty and "signal_groups" in frame
            else {}
        ),
    }
    return frame, summary


def wave_feature_sets(frame: pd.DataFrame, target: str) -> dict[str, list[str]]:
    table = forecast_feature_columns(frame, target)
    wave = [
        column for column in frame.columns
        if column.startswith("wave_")
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    candidate = sorted(set(table + wave))
    wave = [column for column in candidate if column.startswith("wave_")]
    baseline = [column for column in candidate if column not in wave]
    return {"baseline": baseline, "candidate": candidate, "wave": wave}


def audit_target_waveform(
    frame: pd.DataFrame,
    target: str,
    future_suffix: str,
    seeds: tuple[int, ...],
    discovery_fraction: float,
    alpha: float,
    inner_folds: int,
    bootstrap_samples: int,
    min_pairs: int,
    min_subjects: int,
) -> dict[str, object]:
    rows = forecast_rows(frame, target, future_suffix)
    sets = wave_feature_sets(frame, target)
    wave_count = len(sets["wave"])
    random_reports = []
    for seed in seeds:
        discovery_groups, heldout_groups = split_subjects(frame, seed, discovery_fraction)
        random_reports.append(audit_split(
            rows,
            target,
            sets["baseline"],
            sets["candidate"],
            wave_count,
            discovery_groups,
            heldout_groups,
            "subject_id",
            seed,
            alpha,
            inner_folds,
            bootstrap_samples,
            min_pairs,
            min_subjects,
        ))
    careunit = {"available": False}
    if "first_careunit" in rows and rows["first_careunit"].nunique(dropna=True) >= 2:
        discovery, heldout = split_subjects(frame, 9001, discovery_fraction, group_column="first_careunit")
        careunit = {
            "available": True,
            **audit_split(
                rows,
                target,
                sets["baseline"],
                sets["candidate"],
                wave_count,
                discovery,
                heldout,
                "first_careunit",
                9001,
                alpha,
                inner_folds,
                bootstrap_samples,
                min_pairs,
                min_subjects,
            ),
        }
    time = {"available": False}
    if "time_holdout_group" in rows and rows["time_holdout_group"].nunique(dropna=True) >= 2:
        time = {
            "available": True,
            **audit_split(
                rows,
                target,
                sets["baseline"],
                sets["candidate"],
                wave_count,
                {"early"},
                {"late"},
                "time_holdout_group",
                9901,
                alpha,
                inner_folds,
                bootstrap_samples,
                min_pairs,
                min_subjects,
            ),
        }
    summary = summarize_random(random_reports)
    split_count = len(seeds)
    validated = bool(
        wave_count > 0
        and summary["evaluated_splits"] == split_count
        and summary["selected_candidate_splits"] == split_count
        and summary["heldout_beats_baseline_splits"] == split_count
        and summary["heldout_beats_placebo_splits"] == split_count
        and split_passed(careunit)
        and split_passed(time)
    )
    return {
        "support": {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
            "baseline_feature_count": int(len(sets["baseline"])),
            "candidate_feature_count": int(len(sets["candidate"])),
            "wave_feature_count": int(wave_count),
            "wave_features": sets["wave"],
        },
        "random_patient_splits": summary,
        "careunit_holdout": {
            "available": bool(careunit.get("available")),
            "status": careunit.get("status"),
            "gate_passed": split_passed(careunit),
            "heldout": careunit.get("heldout") if careunit.get("available") else None,
        },
        "time_holdout": {
            "available": bool(time.get("available")),
            "status": time.get("status"),
            "gate_passed": split_passed(time),
            "heldout": time.get("heldout") if time.get("available") else None,
        },
        "validated": validated,
    }


def run_waveform_precision_audit(args) -> dict[str, object]:
    segments = build_segment_index(args.waveform_root, args.mimic_dir)
    columns = None
    transitions = pd.read_parquet(args.cohort, columns=columns).reset_index(drop=True)
    transitions = add_time_holdout_column(transitions, args.discovery_fraction)
    anchors = candidate_anchor_rows(
        transitions,
        segments,
        seconds=args.seconds,
        max_anchors=args.max_anchors,
        seed=args.seed,
    )
    feature_frame, feature_summary = attach_waveform_features(anchors, seconds=args.seconds)
    target_reports = {}
    if not feature_frame.empty:
        for target in args.targets:
            print(f"Auditing waveform target: {target}", flush=True)
            target_reports[target] = audit_target_waveform(
                feature_frame,
                target,
                args.future_suffix,
                tuple(args.seeds),
                args.discovery_fraction,
                args.alpha,
                args.inner_folds,
                args.bootstrap_samples,
                args.min_pairs,
                args.min_subjects,
            )
    return {
        "artifact": "MIMIC-IV Waveform factual precision audit",
        "definition": "baseline table+treatment-context forecast vs candidate plus observed pre-anchor waveform features vs capacity-matched placebo",
        "boundary": {
            "factual_observed_inputs_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "row_level_features_committed": False,
        },
        "segment_index": {
            "segments": int(len(segments)),
            "subjects": int(segments["subject_id"].nunique()) if not segments.empty else 0,
            "stays": int(segments["stay_id"].nunique()) if not segments.empty and "stay_id" in segments else 0,
        },
        "feature_extraction": feature_summary,
        "targets": target_reports,
        "validated_targets": sorted([target for target, report in target_reports.items() if report.get("validated")]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--max-anchors", type=int, default=500)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--future-suffix", default="tp6")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--seed", type=int, default=7105)
    parser.add_argument("--discovery-fraction", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--min-pairs", type=int, default=80)
    parser.add_argument("--min-subjects", type=int, default=20)
    args = parser.parse_args()

    report = run_waveform_precision_audit(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({
        "validated_targets": report["validated_targets"],
        "segment_index": report["segment_index"],
        "feature_extraction": report["feature_extraction"],
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
