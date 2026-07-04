"""Audit eICU demo treatment tables for DKA action-channel coverage.

This script does not train a model and does not estimate treatment effects. It
checks whether the eICU demo has enough time-stamped treatment evidence to build
a future action-conditioned DKA cohort. The output is aggregate-only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dka_action_contract import ACTION_KEYS
from eicu_demo_pretrain import LAB_MAP, clean_feature_values
from mimic_action_history import classify_actions, insulin_channel


ID = "patientunitstayid"
HOUR = "hour"
DKA_GLUCOSE_MIN = 200.0
DKA_HCO3_MAX = 18.0
DKA_PH_MAX = 7.30
DKA_COOCCUR_HOURS = 4

DKA_DIAGNOSIS_PATTERNS = (
    "diabetic ketoacidosis",
    "ketoacidosis",
    "dka",
    "250.1",
    "e10.1",
    "e11.1",
)

ACTION_COLUMNS = [
    ID, HOUR, "action", "source", "has_dose_text", "has_rate",
    "has_start_stop", "duration_hours",
]


def add_hour(frame: pd.DataFrame, offset_column: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[HOUR] = np.floor(pd.to_numeric(frame[offset_column], errors="coerce") / 60.0)
    frame = frame[np.isfinite(frame[HOUR])]
    frame[HOUR] = frame[HOUR].astype("int32")
    return frame


def text_contains_any(value: object, patterns: tuple[str, ...]) -> bool:
    text = str(value or "").lower()
    return any(pattern in text for pattern in patterns)


def classify_eicu_action(label: object, route: object = "", source: str = "") -> tuple[str, ...]:
    """Map an eICU medication/treatment label into the DKA action contract."""

    actions = []
    for action in classify_actions(label):
        if action == "insulin":
            actions.append(insulin_channel(label, route=route, source=source))
        else:
            actions.append(action)
    text = str(label or "").lower().replace(" ", "")
    if re.search(r"\bd(?:5|10|20|50)\b", str(label or "").lower()) or any(
        token in text for token in ("d5ns", "d10ns", "d5nacl", "d10nacl", "d5lr")
    ):
        actions.append("dextrose")
    return tuple(action for action in dict.fromkeys(actions) if action in ACTION_KEYS)


def read_lab_dka_like_stays(data_root: Path) -> dict[int, dict[str, object]]:
    columns = [ID, "labresultoffset", "labname", "labresult"]
    frame = pd.read_csv(data_root / "lab.csv.gz", usecols=columns)
    frame = frame[frame["labname"].isin(LAB_MAP)]
    frame = add_hour(frame, "labresultoffset")
    frame["feature"] = frame["labname"].map(LAB_MAP)
    frame = frame[frame["feature"].isin(("Glucose", "HCO3", "pH"))]
    frame["value"] = pd.to_numeric(frame["labresult"], errors="coerce").astype("float32")
    for feature in ("Glucose", "HCO3", "pH"):
        mask = frame["feature"] == feature
        if mask.any():
            frame.loc[mask, "value"] = clean_feature_values(feature, frame.loc[mask, "value"])
    frame = frame.dropna(subset=["value"])

    by_patient: dict[int, dict[str, object]] = {}
    for patient_id, group in frame.groupby(ID, sort=False):
        glucose_hours = group[
            (group["feature"] == "Glucose") & (group["value"] >= DKA_GLUCOSE_MIN)
        ][HOUR].to_numpy(dtype=np.int32)
        if len(glucose_hours) == 0:
            continue
        acidotic = group[
            ((group["feature"] == "HCO3") & (group["value"] <= DKA_HCO3_MAX))
            | ((group["feature"] == "pH") & (group["value"] <= DKA_PH_MAX))
        ]
        if acidotic.empty:
            continue
        acid_hours = acidotic[HOUR].to_numpy(dtype=np.int32)
        best_anchor = None
        for hour in glucose_hours:
            close = acid_hours[np.abs(acid_hours - hour) <= DKA_COOCCUR_HOURS]
            if len(close):
                best_anchor = int(min(hour, int(close[0])))
                break
        if best_anchor is None:
            continue
        by_patient[int(patient_id)] = {
            "anchor_hour": best_anchor,
            "criteria": "glucose_and_acidemia_or_low_hco3_within_4h",
        }
    return by_patient


def read_diagnosis_dka_stays(data_root: Path) -> dict[int, dict[str, object]]:
    hits: dict[int, dict[str, object]] = {}
    specs = [
        ("diagnosis.csv.gz", [ID, "diagnosisoffset", "diagnosisstring", "icd9code"]),
        ("admissionDx.csv.gz", [ID, "admitdxenteredoffset", "admitdxpath", "admitdxname", "admitdxtext"]),
    ]
    for filename, columns in specs:
        path = data_root / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=columns)
        text_columns = [column for column in columns if column != ID and not column.endswith("offset")]
        text = frame[text_columns].fillna("").agg(" ".join, axis=1)
        matched = frame[text.map(lambda value: text_contains_any(value, DKA_DIAGNOSIS_PATTERNS))]
        for patient_id in matched[ID].dropna().astype(int).unique():
            hits[int(patient_id)] = {
                "anchor_hour": 0,
                "criteria": f"diagnosis_text_or_code:{filename}",
            }
    return hits


def merge_dka_like_stays(data_root: Path) -> dict[int, dict[str, object]]:
    lab_hits = read_lab_dka_like_stays(data_root)
    diagnosis_hits = read_diagnosis_dka_stays(data_root)
    merged = dict(diagnosis_hits)
    for patient_id, payload in lab_hits.items():
        if patient_id in merged:
            merged[patient_id] = {
                "anchor_hour": min(int(merged[patient_id]["anchor_hour"]), int(payload["anchor_hour"])),
                "criteria": "diagnosis_and_lab_dka_like",
            }
        else:
            merged[patient_id] = payload
    return merged


def empty_action_frame() -> pd.DataFrame:
    return finalize_action_frame(pd.DataFrame(columns=ACTION_COLUMNS))


def finalize_action_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reindex(columns=ACTION_COLUMNS).copy()
    if len(frame):
        frame[ID] = pd.to_numeric(frame[ID], errors="coerce").astype("int64")
        frame[HOUR] = pd.to_numeric(frame[HOUR], errors="coerce").astype("int32")
    else:
        frame[ID] = frame[ID].astype("int64")
        frame[HOUR] = frame[HOUR].astype("int32")
    frame["action"] = frame["action"].astype("object")
    frame["source"] = frame["source"].astype("object")
    for column in ("has_dose_text", "has_rate", "has_start_stop"):
        frame[column] = frame[column].fillna(False).astype("bool")
    frame["duration_hours"] = pd.to_numeric(frame["duration_hours"], errors="coerce").astype("float64")
    return frame


def medication_actions(data_root: Path) -> pd.DataFrame:
    path = data_root / "medication.csv.gz"
    if not path.exists():
        return empty_action_frame()
    columns = [ID, "drugstartoffset", "drugstopoffset", "drugname", "dosage", "routeadmin"]
    frame = pd.read_csv(path, usecols=columns)
    frame = add_hour(frame, "drugstartoffset")
    rows = []
    for row in frame.itertuples(index=False):
        route = getattr(row, "routeadmin", "")
        for action in classify_eicu_action(getattr(row, "drugname", ""), route=route, source="medication"):
            stop = pd.to_numeric(getattr(row, "drugstopoffset", math.nan), errors="coerce")
            start = pd.to_numeric(getattr(row, "drugstartoffset", math.nan), errors="coerce")
            duration = None
            if np.isfinite(start) and np.isfinite(stop) and stop > start:
                duration = float((stop - start) / 60.0)
            rows.append({
                ID: int(getattr(row, ID)),
                HOUR: int(getattr(row, HOUR)),
                "action": action,
                "source": "medication",
                "has_dose_text": pd.notna(getattr(row, "dosage", None)),
                "has_rate": False,
                "has_start_stop": duration is not None,
                "duration_hours": duration,
            })
    return finalize_action_frame(pd.DataFrame(rows)) if rows else empty_action_frame()


def infusion_actions(data_root: Path) -> pd.DataFrame:
    path = data_root / "infusiondrug.csv.gz"
    if not path.exists():
        return empty_action_frame()
    columns = [ID, "infusionoffset", "drugname", "drugrate", "infusionrate", "drugamount", "volumeoffluid"]
    frame = pd.read_csv(path, usecols=columns)
    frame = add_hour(frame, "infusionoffset")
    rows = []
    for row in frame.itertuples(index=False):
        for action in classify_eicu_action(getattr(row, "drugname", ""), route="iv", source="infusiondrug"):
            has_rate = pd.notna(getattr(row, "drugrate", None)) or pd.notna(getattr(row, "infusionrate", None))
            has_amount = pd.notna(getattr(row, "drugamount", None)) or pd.notna(getattr(row, "volumeoffluid", None))
            rows.append({
                ID: int(getattr(row, ID)),
                HOUR: int(getattr(row, HOUR)),
                "action": action,
                "source": "infusiondrug",
                "has_dose_text": bool(has_amount),
                "has_rate": bool(has_rate),
                "has_start_stop": False,
                "duration_hours": None,
            })
    return finalize_action_frame(pd.DataFrame(rows)) if rows else empty_action_frame()


def treatment_actions(data_root: Path) -> pd.DataFrame:
    path = data_root / "treatment.csv.gz"
    if not path.exists():
        return empty_action_frame()
    columns = [ID, "treatmentoffset", "treatmentstring", "activeupondischarge"]
    frame = pd.read_csv(path, usecols=columns)
    frame = add_hour(frame, "treatmentoffset")
    rows = []
    for row in frame.itertuples(index=False):
        for action in classify_eicu_action(getattr(row, "treatmentstring", ""), source="treatment"):
            rows.append({
                ID: int(getattr(row, ID)),
                HOUR: int(getattr(row, HOUR)),
                "action": action,
                "source": "treatment",
                "has_dose_text": False,
                "has_rate": False,
                "has_start_stop": False,
                "duration_hours": None,
            })
    return finalize_action_frame(pd.DataFrame(rows)) if rows else empty_action_frame()


def build_action_table(data_root: Path) -> pd.DataFrame:
    frames = [
        medication_actions(data_root),
        infusion_actions(data_root),
        treatment_actions(data_root),
    ]
    frames = [finalize_action_frame(frame) for frame in frames if not frame.empty]
    if not frames:
        return empty_action_frame()
    actions = pd.concat(frames, ignore_index=True)
    actions = actions[actions["action"].isin(ACTION_KEYS)]
    return actions.sort_values([ID, HOUR, "source", "action"]).reset_index(drop=True)


def summarize_actions(actions: pd.DataFrame, patient_ids: set[int] | None = None) -> dict[str, object]:
    if patient_ids is not None:
        actions = actions[actions[ID].isin(patient_ids)]
    if actions.empty:
        return {"rows": 0, "stays": 0, "by_action": {}, "by_source": {}}
    by_action = {}
    for action, group in actions.groupby("action"):
        by_action[action] = {
            "rows": int(len(group)),
            "stays": int(group[ID].nunique()),
            "has_rate_rows": int(group["has_rate"].sum()),
            "has_dose_text_rows": int(group["has_dose_text"].sum()),
            "has_start_stop_rows": int(group["has_start_stop"].sum()),
        }
    by_source = {
        source: {"rows": int(len(group)), "stays": int(group[ID].nunique())}
        for source, group in actions.groupby("source")
    }
    return {
        "rows": int(len(actions)),
        "stays": int(actions[ID].nunique()),
        "by_action": by_action,
        "by_source": by_source,
    }


def dka_window_action_summary(actions: pd.DataFrame, dka_stays: dict[int, dict[str, object]],
                              before_hours: int, after_hours: int) -> dict[str, object]:
    if actions.empty or not dka_stays:
        return {"rows": 0, "stays": 0, "by_action": {}, "by_source": {}}
    frames = []
    for patient_id, payload in dka_stays.items():
        anchor = int(payload["anchor_hour"])
        stay_actions = actions[
            (actions[ID] == patient_id)
            & (actions[HOUR] >= anchor - before_hours)
            & (actions[HOUR] <= anchor + after_hours)
        ].copy()
        if not stay_actions.empty:
            stay_actions["relative_hour"] = stay_actions[HOUR] - anchor
            frames.append(stay_actions)
    window = pd.concat(frames, ignore_index=True) if frames else empty_action_frame()
    return summarize_actions(window)


def build_report(data_root: Path, before_hours: int, after_hours: int) -> dict[str, object]:
    patients = pd.read_csv(data_root / "patient.csv.gz", usecols=[ID])
    dka_stays = merge_dka_like_stays(data_root)
    actions = build_action_table(data_root)
    dka_ids = set(dka_stays)
    criteria_counts = {}
    for payload in dka_stays.values():
        criteria = str(payload["criteria"])
        criteria_counts[criteria] = criteria_counts.get(criteria, 0) + 1
    return {
        "dataset": "PhysioNet eICU Collaborative Research Database Demo 2.0.1",
        "data_root": str(data_root),
        "patient_table_stays": int(patients[ID].nunique()),
        "dka_like_stays": {
            "count": int(len(dka_stays)),
            "criteria_counts": criteria_counts,
            "definition": {
                "lab": (
                    f"Glucose >= {DKA_GLUCOSE_MIN} and "
                    f"(HCO3 <= {DKA_HCO3_MAX} or pH <= {DKA_PH_MAX}) "
                    f"within {DKA_COOCCUR_HOURS}h"
                ),
                "diagnosis_patterns": list(DKA_DIAGNOSIS_PATTERNS),
            },
        },
        "action_contract": {
            "channels": list(ACTION_KEYS),
            "sources": ["medication", "infusiondrug", "treatment"],
            "dose_time_resolution_note": (
                "medication has start/stop offsets and dosage text; infusiondrug "
                "has time-stamped rates/amounts but no explicit stop rows; "
                "treatment is coarse presence/procedure text."
            ),
        },
        "all_stay_action_coverage": summarize_actions(actions),
        "dka_like_anytime_action_coverage": summarize_actions(actions, dka_ids),
        "dka_like_window_action_coverage": dka_window_action_summary(
            actions, dka_stays, before_hours, after_hours
        ),
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included": False,
            "action_conditioned_training_ready": bool(len(dka_stays) and not actions.empty),
            "causal_treatment_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("physionet.org/files/eicu-crd-demo/2.0.1"),
    )
    parser.add_argument("--output", type=Path, default=Path("eicu_demo_action_audit.json"))
    parser.add_argument("--before-hours", type=int, default=6)
    parser.add_argument("--after-hours", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root, args.before_hours, args.after_hours)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "dka_like_stays": report["dka_like_stays"]["count"],
        "all_action_rows": report["all_stay_action_coverage"]["rows"],
        "dka_window_action_rows": report["dka_like_window_action_coverage"]["rows"],
        "action_conditioned_training_ready": report["safety_boundary"]["action_conditioned_training_ready"],
    }, indent=2))


if __name__ == "__main__":
    main()
