"""Extract timestamped MIMIC-IV radiology-note findings into observation targets.

This is the first note-backed depth adapter for the whole-body observation
layer.  It converts chest radiology reports into conservative binary findings
and attaches them to a local MIMIC-IV observation transition cohort.

Leakage guard:
    Current ``rad_*_t`` observations use only reports whose authored timestamp
    is <= the anchor.  Future ``rad_*_tp6`` labels use reports authored after
    the anchor and near ``t + 6h``.

The row-level enhanced parquet is local-only and ignored by git.  The committed
JSON report is aggregate-only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIMIC_DIR = Path("/Users/chunyouchang/Downloads/mimic-iv-3.1")
DEFAULT_NOTE_DIR = Path("/Users/chunyouchang/Downloads/mimic-iv-note-deidentified-free-text-clinical-notes-2.2 (2)/note")

LOOKBACK_H = 72.0
TARGET_H = 6.0
TARGET_TOL_H = 2.0


FINDING_PATTERNS: dict[str, tuple[str, ...]] = {
    "pulmonary_edema": (
        r"pulmonary edema",
        r"interstitial edema",
        r"alveolar edema",
        r"vascular congestion",
        r"pulmonary vascular congestion",
        r"volume overload",
    ),
    "pleural_effusion": (
        r"pleural effusion",
        r"\beffusion\b",
        r"\beffusions\b",
    ),
    "consolidation": (
        r"\bconsolidation\b",
        r"\bconsolidations\b",
        r"airspace opacity",
        r"airspace opacities",
        r"airspace disease",
        r"\binfiltrate\b",
        r"\binfiltrates\b",
        r"\bpneumonia\b",
    ),
    "atelectasis": (
        r"\batelectasis\b",
        r"volume loss",
    ),
    "pneumothorax": (
        r"\bpneumothorax\b",
    ),
    "cardiomegaly": (
        r"\bcardiomegaly\b",
        r"enlarged cardiomediastinal",
        r"enlarged cardiac",
        r"cardiac silhouette is enlarged",
        r"enlargement of the cardiac silhouette",
    ),
}

RAD_TARGETS = tuple(f"rad_{name}" for name in FINDING_PATTERNS)

NEGATION_RE = re.compile(
    r"\b(no|without|absent|absence of|negative for|free of|clear of|not|resolved|resolution of|resolved|cleared)\b"
)
NO_ACUTE_RE = re.compile(
    r"no acute (cardiopulmonary|pulmonary|thoracic|chest) (process|abnormality|disease|finding)"
)
SPLIT_RE = re.compile(r"[\n.;:]|(?<=\))\s+")


def chest_note_ids(note_dir: Path) -> set[str]:
    path = note_dir / "radiology_detail.csv.gz"
    keep: set[str] = set()
    usecols = ["note_id", "field_name", "field_value"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False):
        names = chunk[chunk["field_name"].eq("exam_name")].copy()
        if names.empty:
            continue
        text = names["field_value"].fillna("").astype(str).str.lower()
        mask = text.str.contains("chest|portable chest|cxr|thorax|rib", regex=True)
        keep.update(names.loc[mask, "note_id"].astype(str).tolist())
    return keep


def sentence_negates(sentence: str, start: int) -> bool:
    prefix = sentence[max(0, start - 90):start]
    return bool(NEGATION_RE.search(prefix))


def extract_findings(text: object) -> dict[str, float]:
    raw = str(text or "").lower()
    sentences = [part.strip() for part in SPLIT_RE.split(raw) if part.strip()]
    result = {f"rad_{name}": np.nan for name in FINDING_PATTERNS}
    positive = {name: False for name in FINDING_PATTERNS}
    negative = {name: False for name in FINDING_PATTERNS}

    no_acute = bool(NO_ACUTE_RE.search(raw))
    for sentence in sentences:
        for finding, patterns in FINDING_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, sentence):
                    if sentence_negates(sentence, match.start()):
                        negative[finding] = True
                    else:
                        positive[finding] = True

    for finding in FINDING_PATTERNS:
        key = f"rad_{finding}"
        if positive[finding]:
            result[key] = 1.0
        elif negative[finding] or no_acute:
            result[key] = 0.0
    return result


def read_chest_radiology_notes(note_dir: Path, hadm_ids: set[int]) -> pd.DataFrame:
    note_ids = chest_note_ids(note_dir)
    path = note_dir / "radiology.csv.gz"
    rows = []
    usecols = ["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000, low_memory=False):
        chunk = chunk[
            chunk["note_id"].astype(str).isin(note_ids)
            & chunk["hadm_id"].isin(hadm_ids)
        ].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["note_time"] = chunk["storetime"].where(chunk["storetime"].notna(), chunk["charttime"])
        chunk = chunk[chunk["note_time"].notna()].copy()
        findings = pd.DataFrame([extract_findings(text) for text in chunk["text"]], index=chunk.index)
        part = pd.concat(
            [
                chunk[["note_id", "subject_id", "hadm_id", "charttime", "storetime", "note_time"]],
                findings,
            ],
            axis=1,
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["note_id", "subject_id", "hadm_id", "note_time", *RAD_TARGETS])
    notes = pd.concat(rows, ignore_index=True)
    notes["hadm_id"] = pd.to_numeric(notes["hadm_id"], errors="coerce").astype("int64")
    return notes.sort_values(["hadm_id", "note_time", "note_id"]).reset_index(drop=True)


def read_stay_hadm_map(mimic_dir: Path) -> pd.DataFrame:
    path = mimic_dir / "icu" / "icustays.csv.gz"
    return pd.read_csv(path, usecols=["stay_id", "hadm_id"])


def merge_current_note_features(anchors: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    left = anchors[["hadm_id", "t"]].copy()
    left["t"] = pd.to_datetime(left["t"])
    left_sorted = left.sort_values("t")
    right = notes[["hadm_id", "note_time", "note_id", *RAD_TARGETS]].copy()
    right = right.sort_values("note_time")
    merged = pd.merge_asof(
        left_sorted,
        right,
        left_on="t",
        right_on="note_time",
        by="hadm_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=LOOKBACK_H),
    )
    merged = merged.set_index(left_sorted.index).reindex(left.index)
    out = pd.DataFrame(index=anchors.index)
    age = (left["t"] - merged["note_time"]).dt.total_seconds() / 3600.0
    out["rad_chest_report_t"] = merged["note_id"].notna().astype(float)
    out["rad_chest_report_age_hr"] = age.to_numpy(dtype=np.float64)
    for target in RAD_TARGETS:
        out[f"{target}_t"] = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=np.float64)
        out[f"{target}_age_hr"] = age.to_numpy(dtype=np.float64)
    return out


def merge_future_note_features(anchors: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    left = anchors[["hadm_id", "t", "t_plus"]].copy()
    left["t"] = pd.to_datetime(left["t"])
    left["t_plus"] = pd.to_datetime(left["t_plus"])
    left_sorted = left.sort_values("t_plus")
    right = notes[["hadm_id", "note_time", "note_id", *RAD_TARGETS]].copy()
    right = right.sort_values("note_time")
    merged = pd.merge_asof(
        left_sorted,
        right,
        left_on="t_plus",
        right_on="note_time",
        by="hadm_id",
        direction="nearest",
        tolerance=pd.Timedelta(hours=TARGET_TOL_H),
    )
    merged = merged.set_index(left_sorted.index).reindex(left.index)
    authored_after_anchor = merged["note_time"] > left["t"]
    out = pd.DataFrame(index=anchors.index)
    out["rad_chest_report_tp6"] = (merged["note_id"].notna() & authored_after_anchor).astype(float)
    for target in RAD_TARGETS:
        values = pd.to_numeric(merged[target], errors="coerce")
        out[f"{target}_tp6"] = values.where(authored_after_anchor).to_numpy(dtype=np.float64)
    return out


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for target in RAD_TARGETS:
        current = f"{target}_t"
        future = f"{target}_tp6"
        counts[target] = int((frame[current].notna() & frame[future].notna()).sum())
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_6h.parquet"))
    parser.add_argument("--mimic-dir", type=Path, default=DEFAULT_MIMIC_DIR)
    parser.add_argument("--note-dir", type=Path, default=DEFAULT_NOTE_DIR)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/mimiciv_observation_radiology_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("mimiciv_radiology_note_observation_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    stay_map = read_stay_hadm_map(args.mimic_dir)
    frame = frame.merge(stay_map, on="stay_id", how="left")
    frame = frame[frame["hadm_id"].notna()].copy()
    frame["hadm_id"] = frame["hadm_id"].astype("int64")
    hadm_ids = set(frame["hadm_id"].dropna().astype(int).unique())

    print(f"Anchor rows: {len(frame):,}; hadm_ids: {len(hadm_ids):,}", flush=True)
    notes = read_chest_radiology_notes(args.note_dir, hadm_ids)
    print(f"Chest radiology notes: {len(notes):,}; admissions: {notes['hadm_id'].nunique():,}", flush=True)

    current = merge_current_note_features(frame, notes)
    future = merge_future_note_features(frame, notes)
    enhanced = pd.concat([frame, current, future], axis=1)
    enhanced.to_parquet(args.output, index=False)

    current_support = {
        target: int(enhanced[f"{target}_t"].notna().sum())
        for target in RAD_TARGETS
    }
    future_support = {
        target: int(enhanced[f"{target}_tp6"].notna().sum())
        for target in RAD_TARGETS
    }
    prevalence = {
        target: {
            "current_positive_rate": (
                float(enhanced[f"{target}_t"].mean(skipna=True))
                if enhanced[f"{target}_t"].notna().any() else None
            ),
            "future_positive_rate": (
                float(enhanced[f"{target}_tp6"].mean(skipna=True))
                if enhanced[f"{target}_tp6"].notna().any() else None
            ),
        }
        for target in RAD_TARGETS
    }
    report = {
        "artifact": "MIMIC-IV radiology-note structured observation extraction",
        "source_cohort": str(args.cohort),
        "output": str(args.output),
        "radiology_note_dir": str(args.note_dir),
        "leakage_guard": {
            "current_note_time_rule": "note_time <= anchor t, with 72h lookback",
            "future_note_time_rule": "note_time > anchor t and nearest to t+6h within +/-2h",
            "note_time_definition": "storetime when available, otherwise charttime",
        },
        "rows": int(len(enhanced)),
        "subjects": int(enhanced["subject_id"].nunique()),
        "stays": int(enhanced["stay_id"].nunique()),
        "admissions_in_source_cohort": int(enhanced["hadm_id"].nunique()),
        "chest_radiology_notes": int(len(notes)),
        "chest_radiology_admissions": int(notes["hadm_id"].nunique()) if len(notes) else 0,
        "current_chest_report_rows": int(enhanced["rad_chest_report_t"].sum()),
        "future_chest_report_rows": int(enhanced["rad_chest_report_tp6"].sum()),
        "targets": list(RAD_TARGETS),
        "current_support": current_support,
        "future_support": future_support,
        "pair_counts": pair_counts(enhanced),
        "prevalence": prevalence,
        "row_level_outputs_committed": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "rows": report["rows"],
        "current_chest_report_rows": report["current_chest_report_rows"],
        "future_chest_report_rows": report["future_chest_report_rows"],
        "pair_counts": report["pair_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
