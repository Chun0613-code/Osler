"""Build bounded eICU body-system transition cohorts.

This generic adapter extends Chapter B without copying a new extractor for
every organ system.  Each disease is still isolated by its own contract from
``eicu_body_system_configs.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS, BodySystemDiseaseConfig, get_body_system_config
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
EPISODE_MAX_H = 96.0
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


def horizon_suffix(horizon_hours: float) -> str:
    value = float(horizon_hours)
    if value.is_integer():
        return f"tp{int(value)}"
    return "tp" + str(value).replace(".", "p")


def future_column(var: str, horizon_hours: float) -> str:
    return f"{var}_{horizon_suffix(horizon_hours)}"


def empty_action_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_EVIDENCE_COLUMNS)


def classify_action(config: BodySystemDiseaseConfig, label: object) -> tuple[str, ...]:
    text = str(label or "").lower()
    compact = text.replace(" ", "")
    actions = []
    for action, terms in config.action_terms.items():
        if any(term in text for term in terms):
            actions.append(action)
        elif action == "fluids" and compact in {"ns", "lr"}:
            actions.append(action)
    return tuple(dict.fromkeys(actions))


def action_terms_mask(config: BodySystemDiseaseConfig, series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for terms in config.action_terms.values():
        for term in terms:
            mask |= text.str.contains(term, regex=False)
    return mask


def read_diagnosis_stays(
    data_root: Path,
    meta: pd.DataFrame,
    config: BodySystemDiseaseConfig,
) -> dict[int, dict[str, object]]:
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
            matched = chunk[_text_series_contains_any(text, config.diagnosis_patterns)]
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
            meta["apacheadmissiondx"].map(
                lambda value: any(pattern in str(value or "").lower() for pattern in config.diagnosis_patterns)
            )
        ]
        for stay_id in matched["stay_id"].astype(int):
            hits.setdefault(int(stay_id), {
                "anchor_hour": 0.0,
                "criteria": "patient_apacheadmissiondx",
            })
    return hits


def bounded_stays(
    stays: dict[int, dict[str, object]],
    max_stays: int | None,
) -> dict[int, dict[str, object]]:
    if max_stays is None or len(stays) <= max_stays:
        return stays
    keep = sorted(stays)[:max_stays]
    return {stay_id: stays[stay_id] for stay_id in keep}


def medication_evidence(
    data_root: Path,
    stay_ids: set[int],
    config: BodySystemDiseaseConfig,
) -> pd.DataFrame:
    path = data_root / "medication.csv.gz"
    columns = [ID, "drugstartoffset", "drugstopoffset", "drugname", "drugordercancelled"]
    rows = []
    for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
        chunk = chunk[
            chunk[ID].isin(stay_ids)
            & (chunk["drugordercancelled"].fillna("No").astype(str).str.lower() != "yes")
            & action_terms_mask(config, chunk["drugname"])
        ]
        for record in chunk.to_dict("records"):
            start = _finite_number(record.get("drugstartoffset"))
            if not np.isfinite(start):
                continue
            stop = _finite_number(record.get("drugstopoffset"))
            if not np.isfinite(stop) or stop <= start:
                stop = start + 30.0
            label = record.get("drugname")
            for action in classify_action(config, label):
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


def infusion_evidence(
    data_root: Path,
    stay_ids: set[int],
    config: BodySystemDiseaseConfig,
    max_snapshot_hours: float = 4.0,
) -> pd.DataFrame:
    path = data_root / "infusionDrug.csv.gz"
    columns = [ID, "infusionoffset", "drugname", "drugrate", "infusionrate", "drugamount", "volumeoffluid"]
    frames = []
    for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
        chunk = chunk[chunk[ID].isin(stay_ids) & action_terms_mask(config, chunk["drugname"])].copy()
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
            for action in classify_action(config, label):
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


def treatment_evidence(
    data_root: Path,
    stay_ids: set[int],
    config: BodySystemDiseaseConfig,
) -> pd.DataFrame:
    path = data_root / "treatment.csv.gz"
    columns = [ID, "treatmentoffset", "treatmentstring"]
    rows = []
    for chunk in _read_csv_chunks(path, columns, chunksize=500_000):
        chunk = chunk[chunk[ID].isin(stay_ids) & action_terms_mask(config, chunk["treatmentstring"])]
        for record in chunk.to_dict("records"):
            offset = _finite_number(record.get("treatmentoffset"))
            if not np.isfinite(offset):
                continue
            label = record.get("treatmentstring")
            for action in classify_action(config, label):
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


def respiratory_evidence(data_root: Path, stay_ids: set[int], config: BodySystemDiseaseConfig) -> pd.DataFrame:
    if "ventilation" not in config.action_keys:
        return empty_action_evidence()
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
                    start = status if np.isfinite(status) else np.nan
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
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_action_evidence()


def read_action_evidence(data_root: Path, stay_ids: set[int], config: BodySystemDiseaseConfig) -> pd.DataFrame:
    frames = [
        medication_evidence(data_root, stay_ids, config),
        infusion_evidence(data_root, stay_ids, config),
        treatment_evidence(data_root, stay_ids, config),
        respiratory_evidence(data_root, stay_ids, config),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_action_evidence()
    evidence = pd.concat(frames, ignore_index=True)
    return evidence.sort_values(["stay_id", "starttime", "source", "action"]).reset_index(drop=True)


def build_anchors(
    disease_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    meta: pd.DataFrame,
    horizon_hours: float,
) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame()
    max_observed = measurements.groupby("stay_id")["time_hr"].max().to_dict()
    rows = []
    for stay_id, payload in disease_stays.items():
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
            rows.append({
                "subject_id": str(meta.loc[stay_id, "subject_id"]) if stay_id in meta.index else str(stay_id),
                "stay_id": int(stay_id),
                "hospitalid": int(meta.loc[stay_id, "hospitalid"]) if stay_id in meta.index and pd.notna(meta.loc[stay_id, "hospitalid"]) else -1,
                "unittype": str(meta.loc[stay_id, "unittype"]) if stay_id in meta.index else "",
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
    config: BodySystemDiseaseConfig,
    horizon_hours: float,
) -> pd.DataFrame:
    lookup = measurement_lookup(measurements)
    rows = []
    for anchor in anchors.itertuples(index=False):
        row = {}
        for var in config.state_vars:
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
    config: BodySystemDiseaseConfig,
    anchor_hour: float,
    horizon_hours: float,
) -> dict[str, object]:
    history_start = float(anchor_hour - LOOKBACK_H)
    future_end = float(anchor_hour + horizon_hours)
    output = {}
    for action in config.action_keys:
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
        output[f"act_{action}_dose_observed"] = int(dose_observed[future_mask].any() if future_mask.any() else False)
    return output


def assemble_actions(
    anchors: pd.DataFrame,
    evidence: pd.DataFrame,
    config: BodySystemDiseaseConfig,
    horizon_hours: float,
) -> pd.DataFrame:
    lookup = prepare_evidence_lookup(evidence)
    rows = []
    for anchor in anchors.itertuples(index=False):
        rows.append(evidence_window_summary(
            lookup.get(int(anchor.stay_id), {}),
            config,
            float(anchor.t_hour),
            horizon_hours,
        ))
    return pd.DataFrame(rows, index=anchors.index)


def add_active_column(frame: pd.DataFrame, config: BodySystemDiseaseConfig) -> pd.DataFrame:
    frame = frame.copy()
    active = np.zeros(len(frame), dtype=bool)
    for var, threshold in config.active_low.items():
        column = f"{var}_t"
        if column in frame:
            active |= pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64) <= float(threshold)
    for var, threshold in config.active_high.items():
        column = f"{var}_t"
        if column in frame:
            active |= pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64) >= float(threshold)
    for action_column in config.active_actions:
        if action_column in frame:
            active |= frame[action_column].fillna(0).astype(bool).to_numpy()
    frame[f"{config.name}_active_t"] = active
    return frame


def filter_evaluable(frame: pd.DataFrame, config: BodySystemDiseaseConfig, horizon_hours: float) -> pd.DataFrame:
    has_pair = np.zeros(len(frame), dtype=bool)
    for var in config.targets:
        current = f"{var}_t"
        future = future_column(var, horizon_hours)
        if current in frame and future in frame:
            has_pair |= frame[current].notna().values & frame[future].notna().values
    return frame[has_pair].reset_index(drop=True)


def target_pair_counts(frame: pd.DataFrame, config: BodySystemDiseaseConfig, horizon_hours: float) -> dict[str, int]:
    output = {}
    for var in config.targets:
        current = f"{var}_t"
        future = future_column(var, horizon_hours)
        if current in frame and future in frame:
            output[var] = int((frame[current].notna() & frame[future].notna()).sum())
    return output


def action_support_counts(frame: pd.DataFrame, config: BodySystemDiseaseConfig) -> dict[str, dict[str, int]]:
    output = {}
    active = frame[f"{config.name}_active_t"].fillna(False).astype(bool)
    for action in config.action_keys:
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
    disease_stays: dict[int, dict[str, object]],
    measurements: pd.DataFrame,
    evidence: pd.DataFrame,
    config: BodySystemDiseaseConfig,
    horizon_hours: float,
    max_stays: int | None,
) -> dict[str, object]:
    criteria_counts = pd.Series(
        [payload["criteria"] for payload in disease_stays.values()],
        dtype="object",
    ).value_counts().astype(int).to_dict()
    active_column = f"{config.name}_active_t"
    return {
        "dataset": "PhysioNet eICU Collaborative Research Database 2.0",
        "data_root": data_root.name,
        "disease": config.name,
        "display_name": config.display_name,
        "body_system": config.body_system,
        "output": output.name,
        "bounded_max_stays": max_stays,
        "candidate_stay_count_before_bound": int(len(disease_stays)),
        "criteria_counts_before_filter": criteria_counts,
        "stays": int(frame["stay_id"].nunique()) if len(frame) else 0,
        "subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if len(frame) else 0,
        "transitions": int(len(frame)),
        "active_transitions": int(frame[active_column].fillna(False).sum()) if len(frame) else 0,
        "measurements": {
            "rows": int(len(measurements)),
            "stays": int(measurements["stay_id"].nunique()) if len(measurements) else 0,
            "by_var": measurements["var"].value_counts().astype(int).to_dict() if len(measurements) else {},
        },
        "target_pair_counts": target_pair_counts(frame, config, horizon_hours) if len(frame) else {},
        "action_support": action_support_counts(frame, config) if len(frame) else {},
        "action_evidence": {
            "rows": int(len(evidence)),
            "sources": evidence["source"].value_counts().astype(int).to_dict() if len(evidence) else {},
            "by_action": evidence["action"].value_counts().astype(int).to_dict() if len(evidence) else {},
            "dose_observed_fraction": round(float(evidence["dose_observed"].fillna(False).mean()), 6) if len(evidence) else 0.0,
        },
        "schema": {
            "config": config.as_report_dict(),
            "targets": config.targets,
            "state_vars": config.state_vars,
            "action_channels": config.action_keys,
            "active_column": active_column,
            "horizon_hours": horizon_hours,
            "future_suffix": horizon_suffix(horizon_hours),
            "anchor_step_hours": ANCHOR_STEP_H,
            "lookback_hours": LOOKBACK_H,
            "target_tolerance_hours": TARGET_TOL_H,
        },
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included_in_report": False,
            "bounded_engineering_cohort": max_stays is not None,
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
    config: BodySystemDiseaseConfig,
    max_stays: int | None,
    horizon_hours: float,
) -> dict[str, object]:
    meta = read_patient_meta(data_root)
    all_stays = read_diagnosis_stays(data_root, meta, config)
    disease_stays = bounded_stays(all_stays, max_stays=max_stays)
    stay_ids = set(disease_stays)
    measurements = read_measurements(data_root, stay_ids)
    anchors = build_anchors(disease_stays, measurements, meta, horizon_hours=horizon_hours)
    evidence = read_action_evidence(data_root, stay_ids, config)
    if anchors.empty:
        frame = pd.DataFrame()
    else:
        states = assemble_states(anchors, measurements, config, horizon_hours=horizon_hours)
        actions = assemble_actions(anchors, evidence, config, horizon_hours=horizon_hours)
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
            ],
            axis=1,
        )
        frame = filter_evaluable(add_active_column(frame, config), config, horizon_hours=horizon_hours)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    report = cohort_report(
        frame,
        data_root,
        output,
        disease_stays,
        measurements,
        evidence,
        config,
        horizon_hours,
        max_stays,
    )
    report["candidate_stay_count_before_bound"] = int(len(all_stays))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("eicu-collaborative-research-database-2.0"))
    parser.add_argument("--disease", choices=sorted(BODY_SYSTEM_CONFIGS), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-stays", type=int, default=2500)
    parser.add_argument("--horizon-hours", type=float, default=DELTA_H)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_body_system_config(args.disease)
    output = args.output or Path(f"eicu_{config.name}_transitions_6h.parquet")
    report_path = args.report or Path(f"eicu_{config.name}_transition_report.json")
    report = build_transitions(
        args.data_root,
        output,
        config,
        max_stays=args.max_stays,
        horizon_hours=args.horizon_hours,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "disease": config.name,
        "output": str(output),
        "report": str(report_path),
        "stays": report["stays"],
        "transitions": report["transitions"],
        "active_transitions": report["active_transitions"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
