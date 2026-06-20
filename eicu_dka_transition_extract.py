"""Build an eICU-demo DKA transition cohort with observed treatment grids.

The output mirrors ``dka_transition_extract.py`` closely enough to feed the
existing factual JEPA proxy evaluator:

    state_t + history_action_grid + future_action_grid + state_t+6h

This is an observed-treatment factual cohort. It is not a causal estimator and
does not infer missing drug doses from future outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dka_action_contract import ACTION_KEYS, INSULIN_KEYS
from eicu_demo_action_audit import (
    DKA_COOCCUR_HOURS,
    DKA_GLUCOSE_MIN,
    DKA_HCO3_MAX,
    DKA_PH_MAX,
    ID as EICU_ID,
    add_hour,
    merge_dka_like_stays,
)
from eicu_demo_pretrain import (
    LAB_MAP,
    VITAL_APERIODIC_MAP,
    VITAL_PERIODIC_MAP,
    clean_feature_values,
)
from mimic_action_history import (
    action_window_summary,
    deduplicate_events,
    normalize_events,
)


BASE_TIME = pd.Timestamp("2100-01-01")
DELTA_H = 6.0
ANCHOR_STEP_H = 4.0
LOOKBACK_H = 6.0
TARGET_TOL_H = 2.0
EPISODE_MAX_H = 72.0

LAB_TO_STATE = {
    **{name: "glucose" for name in ("bedside glucose", "glucose")},
    "pH": "ph",
    "HCO3": "bicarbonate",
    "bicarbonate": "bicarbonate",
    "anion gap": "anion_gap",
    "potassium": "potassium",
    "lactate": "lactate",
    "creatinine": "creatinine",
    "sodium": "sodium",
    "serum osmolality": "osmolality",
    "serum ketones": "BHB",
}
FEATURE_FOR_CLEANING = {
    "glucose": "Glucose",
    "ph": "pH",
    "bicarbonate": "HCO3",
    "anion_gap": "anion_gap",
    "potassium": "Potassium",
    "lactate": "Lactate",
    "creatinine": "Creatinine",
    "sodium": "sodium",
    "osmolality": "serum_osmolality",
    "BHB": "serum_ketones",
    "heart_rate": "HR",
    "sbp": "SBP",
    "dbp": "DBP",
    "map": "MAP",
    "urine_output": "urine_output",
}
PLAUSIBLE = {
    "glucose": (10.0, 2000.0),
    "ph": (6.5, 8.1),
    "bicarbonate": (1.0, 60.0),
    "anion_gap": (0.0, 80.0),
    "potassium": (1.0, 10.0),
    "lactate": (0.0, 40.0),
    "creatinine": (0.05, 30.0),
    "sodium": (100.0, 180.0),
    "osmolality": (200.0, 450.0),
    "BHB": (0.0, 20.0),
    "heart_rate": (10.0, 300.0),
    "sbp": (30.0, 300.0),
    "dbp": (10.0, 200.0),
    "map": (20.0, 250.0),
    "urine_output": (0.0, 1000.0),
}
STATE_VARS = [
    "glucose", "ph", "bicarbonate", "anion_gap", "potassium", "lactate",
    "creatinine", "sodium", "osmolality", "BHB", "heart_rate", "sbp",
    "dbp", "map", "urine_output",
]
TARGET_VARS = [
    "glucose", "ph", "bicarbonate", "anion_gap", "potassium", "map",
    "sodium", "osmolality", "creatinine", "urine_output", "BHB",
]
VITAL_TO_STATE = {
    "heartrate": "heart_rate",
    "systemicsystolic": "sbp",
    "systemicdiastolic": "dbp",
    "systemicmean": "map",
    "noninvasivesystolic": "sbp",
    "noninvasivediastolic": "dbp",
    "noninvasivemean": "map",
}
ACTION_SEARCH_TERMS = (
    "insulin", "saline", "sodium chloride", "nacl", "lactated", "ringer",
    "potassium chloride", "kcl", "bicarbonate", "dextrose", "d5", "d10",
    "d20", "d50",
)


def pseudo_time(minutes_or_hours: float, unit: str = "minutes") -> pd.Timestamp:
    if unit == "hours":
        return BASE_TIME + pd.Timedelta(hours=float(minutes_or_hours))
    return BASE_TIME + pd.Timedelta(minutes=float(minutes_or_hours))


def clean_state_values(var: str, values: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce").astype("float32")
    feature = FEATURE_FOR_CLEANING.get(var)
    if feature in set(LAB_MAP.values()) | set(VITAL_PERIODIC_MAP.values()) | set(VITAL_APERIODIC_MAP.values()):
        cleaned = clean_feature_values(feature, cleaned)
    lower, upper = PLAUSIBLE[var]
    return cleaned.where((cleaned >= lower) & (cleaned <= upper))


def read_labs(data_root: Path) -> pd.DataFrame:
    columns = [EICU_ID, "labresultoffset", "labname", "labresult"]
    frame = pd.read_csv(data_root / "lab.csv.gz", usecols=columns)
    frame = frame[frame["labname"].isin(LAB_TO_STATE)]
    frame = add_hour(frame, "labresultoffset")
    frame["var"] = frame["labname"].map(LAB_TO_STATE)
    frame["valuenum"] = pd.to_numeric(frame["labresult"], errors="coerce").astype("float32")
    for var in sorted(set(LAB_TO_STATE.values())):
        mask = frame["var"] == var
        if mask.any():
            frame.loc[mask, "valuenum"] = clean_state_values(var, frame.loc[mask, "valuenum"])
    frame = frame.dropna(subset=["valuenum"])
    return frame.rename(columns={EICU_ID: "stay_id", "labresultoffset": "offset"})[
        ["stay_id", "offset", "var", "valuenum"]
    ]


def read_vitals(data_root: Path) -> pd.DataFrame:
    parts = []
    periodic_cols = [EICU_ID, "observationoffset", *VITAL_PERIODIC_MAP.keys()]
    periodic = pd.read_csv(data_root / "vitalPeriodic.csv.gz", usecols=periodic_cols)
    periodic = add_hour(periodic, "observationoffset")
    for source, state_var in VITAL_TO_STATE.items():
        if source not in periodic:
            continue
        values = clean_state_values(state_var, periodic[source])
        part = pd.DataFrame({
            "stay_id": periodic[EICU_ID],
            "offset": periodic["observationoffset"],
            "var": state_var,
            "valuenum": values,
        })
        parts.append(part)

    aperiodic_cols = [EICU_ID, "observationoffset", *VITAL_APERIODIC_MAP.keys()]
    aperiodic = pd.read_csv(data_root / "vitalAperiodic.csv.gz", usecols=aperiodic_cols)
    aperiodic = add_hour(aperiodic, "observationoffset")
    for source, state_var in VITAL_TO_STATE.items():
        if source not in aperiodic:
            continue
        values = clean_state_values(state_var, aperiodic[source])
        parts.append(pd.DataFrame({
            "stay_id": aperiodic[EICU_ID],
            "offset": aperiodic["observationoffset"],
            "var": state_var,
            "valuenum": values,
        }))
    if not parts:
        return empty_measurements()
    return pd.concat(parts, ignore_index=True).dropna(subset=["valuenum"])


def read_urine_output(data_root: Path) -> pd.DataFrame:
    path = data_root / "intakeOutput.csv.gz"
    if not path.exists():
        return empty_measurements()
    columns = [EICU_ID, "intakeoutputoffset", "celllabel", "cellvaluenumeric"]
    frame = pd.read_csv(path, usecols=columns)
    labels = frame["celllabel"].fillna("").astype(str).str.lower()
    urine = labels.str.contains("urine|foley|urinary|void", regex=True)
    frame = frame[urine].copy()
    frame["valuenum"] = clean_state_values("urine_output", frame["cellvaluenumeric"])
    frame = frame.dropna(subset=["valuenum"])
    if frame.empty:
        return empty_measurements()
    frame["offset"] = pd.to_numeric(frame["intakeoutputoffset"], errors="coerce")
    frame = frame[np.isfinite(frame["offset"])]
    frame["hour"] = np.floor(frame["offset"] / 60.0).astype("int32")
    grouped = (
        frame.groupby([EICU_ID, "hour"], sort=False)["valuenum"].sum().reset_index()
    )
    grouped["offset"] = grouped["hour"] * 60.0
    grouped["valuenum"] = grouped["valuenum"] / 4.0
    grouped["var"] = "urine_output"
    return grouped.rename(columns={EICU_ID: "stay_id"})[
        ["stay_id", "offset", "var", "valuenum"]
    ]


def empty_measurements() -> pd.DataFrame:
    return pd.DataFrame(columns=["stay_id", "offset", "var", "valuenum"])


def read_measurements(data_root: Path) -> pd.DataFrame:
    measurements = pd.concat(
        [read_labs(data_root), read_vitals(data_root), read_urine_output(data_root)],
        ignore_index=True,
    )
    measurements["stay_id"] = measurements["stay_id"].astype("int64")
    measurements["time_hr"] = pd.to_numeric(measurements["offset"], errors="coerce") / 60.0
    measurements = measurements[np.isfinite(measurements["time_hr"])]
    return measurements.sort_values(["stay_id", "time_hr", "var"]).reset_index(drop=True)


def action_terms_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for term in ACTION_SEARCH_TERMS:
        mask |= text.str.contains(term, regex=False)
    return mask


def eicu_medication_raw(data_root: Path) -> pd.DataFrame:
    path = data_root / "medication.csv.gz"
    columns = [
        EICU_ID, "medicationid", "drugstartoffset", "drugstopoffset", "drugname",
        "dosage", "routeadmin", "drugordercancelled",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[
        (frame["drugordercancelled"].fillna("No").astype(str).str.lower() != "yes")
        & action_terms_mask(frame["drugname"])
    ].copy()
    rows = []
    for row in frame.itertuples(index=False):
        start = pd.to_numeric(getattr(row, "drugstartoffset"), errors="coerce")
        if not np.isfinite(start):
            continue
        stop = pd.to_numeric(getattr(row, "drugstopoffset"), errors="coerce")
        end = stop if np.isfinite(stop) and stop > start else start
        rows.append({
            "subject_id": int(getattr(row, EICU_ID)),
            "stay_id": int(getattr(row, EICU_ID)),
            "starttime": pseudo_time(start),
            "endtime": pseudo_time(end),
            "label": getattr(row, "drugname"),
            "amount": getattr(row, "dosage"),
            "uom": "",
            "rate": np.nan,
            "rate_uom": "",
            "route": getattr(row, "routeadmin", ""),
            "product_description": "",
            "itemid": np.nan,
            "orderid": getattr(row, "medicationid"),
            "source": "eicu_medication",
        })
    return pd.DataFrame(rows)


def infer_rate_unit(label: object) -> str:
    text = str(label or "").lower()
    if "unit" in text:
        return "unit/hr"
    if "meq" in text or "mmol" in text:
        return "meq/hr"
    if "gram" in text or "gm" in text:
        return "g/hr"
    return "ml/hr"


def eicu_infusion_raw(data_root: Path, max_snapshot_hours: float = 4.0) -> pd.DataFrame:
    path = data_root / "infusiondrug.csv.gz"
    columns = [
        EICU_ID, "infusiondrugid", "infusionoffset", "drugname", "drugrate",
        "infusionrate", "drugamount", "volumeoffluid",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[action_terms_mask(frame["drugname"])].copy()
    frame["offset"] = pd.to_numeric(frame["infusionoffset"], errors="coerce")
    frame = frame[np.isfinite(frame["offset"])].sort_values([EICU_ID, "drugname", "offset"])
    rows = []
    for (_stay, _label), group in frame.groupby([EICU_ID, "drugname"], sort=False):
        offsets = group["offset"].to_numpy(dtype=np.float64)
        next_offsets = np.r_[offsets[1:], np.nan]
        for record, next_offset in zip(group.to_dict("records"), next_offsets):
            start = float(record["offset"])
            if np.isfinite(next_offset) and next_offset > start:
                duration_minutes = min(float(next_offset - start), max_snapshot_hours * 60.0)
            else:
                duration_minutes = 60.0
            rate = record.get("drugrate")
            if pd.isna(rate):
                rate = record.get("infusionrate")
            rows.append({
                "subject_id": int(record[EICU_ID]),
                "stay_id": int(record[EICU_ID]),
                "starttime": pseudo_time(start),
                "endtime": pseudo_time(start + duration_minutes),
                "label": record.get("drugname"),
                "amount": np.nan,
                "uom": "",
                "rate": rate,
                "rate_uom": infer_rate_unit(record.get("drugname")),
                "route": "IV",
                "product_description": "",
                "itemid": np.nan,
                "orderid": record.get("infusiondrugid"),
                "source": "eicu_infusiondrug",
            })
    return pd.DataFrame(rows)


def read_actions(data_root: Path) -> pd.DataFrame:
    raw = pd.concat(
        [eicu_medication_raw(data_root), eicu_infusion_raw(data_root)],
        ignore_index=True,
    )
    if raw.empty:
        return normalize_events(raw)
    events = normalize_events(raw)
    events = events[events["amount"].notna() & (events["amount"] > 0)]
    return deduplicate_events(events).sort_values(["stay_id", "starttime"]).reset_index(drop=True)


def read_patient_meta(data_root: Path) -> pd.DataFrame:
    columns = [
        EICU_ID, "unitdischargeoffset", "unitdischargestatus",
        "hospitaldischargestatus",
    ]
    frame = pd.read_csv(data_root / "patient.csv.gz", usecols=columns)
    frame = frame.rename(columns={EICU_ID: "stay_id"})
    frame["stay_id"] = frame["stay_id"].astype("int64")
    frame["unitdischarge_hr"] = pd.to_numeric(
        frame["unitdischargeoffset"], errors="coerce"
    ) / 60.0
    return frame.drop_duplicates("stay_id").set_index("stay_id", drop=False)


def build_anchors(data_root: Path, measurements: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    dka_stays = merge_dka_like_stays(data_root)
    max_observed = measurements.groupby("stay_id")["time_hr"].max().to_dict()
    rows = []
    for stay_id, payload in dka_stays.items():
        if stay_id not in max_observed:
            continue
        anchor = float(payload["anchor_hour"])
        discharge = meta.loc[stay_id, "unitdischarge_hr"] if stay_id in meta.index else np.nan
        observed_last = float(max_observed[stay_id])
        last = min(anchor + EPISODE_MAX_H, observed_last - DELTA_H)
        if np.isfinite(discharge):
            last = min(last, float(discharge) - DELTA_H)
        t = anchor
        while t <= last:
            rows.append({
                "subject_id": int(stay_id),
                "stay_id": int(stay_id),
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
    return pd.DataFrame(rows), dka_stays


def measurement_lookup(measurements: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    lookup = {}
    for (stay_id, var), group in measurements.groupby(["stay_id", "var"], sort=False):
        ordered = group.sort_values("time_hr")
        lookup[(int(stay_id), str(var))] = (
            ordered["time_hr"].to_numpy(dtype=np.float64),
            ordered["valuenum"].to_numpy(dtype=np.float64),
        )
    return lookup


def backward_value(times: np.ndarray, values: np.ndarray, t: float, lookback: float) -> tuple[float, float]:
    index = np.searchsorted(times, t, side="right") - 1
    if index < 0:
        return np.nan, np.nan
    age = t - float(times[index])
    if age < -1e-6 or age > lookback:
        return np.nan, np.nan
    return float(values[index]), float(age)


def nearest_value(times: np.ndarray, values: np.ndarray, t: float, tolerance: float) -> float:
    index = np.searchsorted(times, t)
    candidates = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return np.nan
    best = min(candidates, key=lambda item: abs(float(times[item]) - t))
    if abs(float(times[best]) - t) > tolerance:
        return np.nan
    return float(values[best])


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


def assemble_actions(anchors: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = {stay: frame for stay, frame in actions.groupby("stay_id")}
    empty = actions.iloc[0:0]
    for anchor in anchors.itertuples(index=False):
        events = grouped.get(int(anchor.stay_id), empty)
        summary = action_window_summary(events, pd.Timestamp(anchor.t), LOOKBACK_H, DELTA_H)
        row = {
            key: json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in summary.items()
        }
        for action in ACTION_KEYS:
            row[f"act_{action}"] = int(row[f"act_{action}_total"] > 0)
            row[f"act_{action}_rate_mean"] = row[f"act_{action}_total"] / DELTA_H
        row["act_insulin"] = int(row["act_insulin_total"] > 0)
        row["act_insulin_rate_mean"] = row["act_insulin_total"] / DELTA_H
        rows.append(row)
    return pd.DataFrame(rows, index=anchors.index)


def assemble_outcomes(anchors: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for anchor in anchors.itertuples(index=False):
        status = ""
        discharge_hr = np.nan
        if int(anchor.stay_id) in meta.index:
            record = meta.loc[int(anchor.stay_id)]
            status = " ".join(str(record.get(column, "")) for column in (
                "unitdischargestatus", "hospitaldischargestatus",
            )).lower()
            discharge_hr = float(record.get("unitdischarge_hr", np.nan))
        died = "expired" in status and np.isfinite(discharge_hr) and discharge_hr > float(anchor.t_plus_hour)
        rows.append({
            "died_after_window": int(died),
            "hrs_to_death_from_cut": (
                float(discharge_hr - float(anchor.t_plus_hour)) if died else np.nan
            ),
            "vaso_onset_after_window": 0,
        })
    return pd.DataFrame(rows, index=anchors.index)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "sodium_t" in frame and "glucose_t" in frame:
        frame["osmolality_derived_t"] = 2.0 * frame["sodium_t"] + frame["glucose_t"] / 18.0
    if "sodium_tp6" in frame and "glucose_tp6" in frame:
        frame["osmolality_derived_tp6"] = 2.0 * frame["sodium_tp6"] + frame["glucose_tp6"] / 18.0
    frame["dka_active_t"] = (
        (frame["glucose_t"] >= 200.0)
        & (
            (frame["bicarbonate_t"] < 18.0)
            | (frame["anion_gap_t"] > 12.0)
            | (frame["ph_t"] < 7.30)
        )
    ) | (frame["BHB_t"] >= 3.0)
    return frame


def filter_evaluable(frame: pd.DataFrame) -> pd.DataFrame:
    has_pair = np.zeros(len(frame), dtype=bool)
    for var in TARGET_VARS:
        has_pair |= frame[f"{var}_t"].notna().values & frame[f"{var}_tp6"].notna().values
    return frame[has_pair].reset_index(drop=True)


def support_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    specs = {
        "insulin_any_to_glucose": ("act_insulin_total", "glucose"),
        "insulin_iv_to_glucose": ("act_insulin_iv_total", "glucose"),
        "rapid_sc_insulin_to_glucose": ("act_insulin_rapid_sc_total", "glucose"),
        "fluids_to_map": ("act_fluids_total", "map"),
        "kcl_to_potassium": ("act_kcl_total", "potassium"),
        "bicarbonate_to_hco3": ("act_bicarbonate_total", "bicarbonate"),
        "dextrose_to_glucose": ("act_dextrose_total", "glucose"),
    }
    output = {}
    for name, (action_col, target) in specs.items():
        if action_col not in frame:
            continue
        mask = (
            (frame[action_col].fillna(0.0) > 0.0)
            & frame[f"{target}_t"].notna()
            & frame[f"{target}_tp6"].notna()
        )
        output[name] = {
            "transitions": int(mask.sum()),
            "stays": int(frame.loc[mask, "stay_id"].nunique()),
            "active_dka_transitions": int((mask & frame["dka_active_t"].fillna(False)).sum()),
            "active_dka_stays": int(frame.loc[mask & frame["dka_active_t"].fillna(False), "stay_id"].nunique()),
        }
    return output


def cohort_report(frame: pd.DataFrame, data_root: Path, output: Path,
                  dka_like_stays: dict[int, dict[str, object]],
                  actions: pd.DataFrame) -> dict[str, object]:
    core_pair_counts = {
        var: int((frame[f"{var}_t"].notna() & frame[f"{var}_tp6"].notna()).sum())
        for var in ("glucose", "potassium", "bicarbonate", "ph", "map")
    }
    action_support = support_counts(frame)
    power_proxy = {
        "map_gate_reference_stays": 42,
        "glucose_gate_reference_stays": 77,
        "note": (
            "Reference thresholds come from prior MIMIC-demo power analysis; "
            "eICU effect sizes may differ."
        ),
        "support_passes_reference": {
            "fluids_to_map": action_support.get("fluids_to_map", {}).get("stays", 0) >= 42,
            "insulin_any_to_glucose": action_support.get("insulin_any_to_glucose", {}).get("stays", 0) >= 77,
            "kcl_to_potassium": action_support.get("kcl_to_potassium", {}).get("stays", 0) >= 77,
        },
    }
    return {
        "dataset": "PhysioNet eICU Collaborative Research Database Demo 2.0.1",
        "data_root": str(data_root),
        "output": str(output),
        "stays": int(frame["stay_id"].nunique()),
        "transitions": int(len(frame)),
        "active_dka_transitions": int(frame["dka_active_t"].fillna(False).sum()),
        "dka_like_stay_count_before_evaluable_filter": int(len(dka_like_stays)),
        "core_pair_counts": core_pair_counts,
        "action_target_support": action_support,
        "power_proxy": power_proxy,
        "action_quality": {
            "normalized_event_rows": int(len(actions)),
            "sources": (
                actions["source"].value_counts().astype(int).to_dict()
                if not actions.empty else {}
            ),
            "exact_timing_fraction": (
                round(float((actions["timing_source"] == "recorded_interval").mean()), 6)
                if not actions.empty else 0.0
            ),
            "high_confidence_dose_fraction": (
                round(float((actions["dose_confidence"] >= 0.75).mean()), 6)
                if not actions.empty else 0.0
            ),
        },
        "schema": {
            "compatible_with_evaluate_mimic_proxy": True,
            "has_exact_action_grids": "future_action_grid" in frame,
            "has_treatment_lifecycle": "future_treatment_event_grid" in frame,
            "has_treatment_history": "history_action_grid" in frame,
        },
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included_in_report": False,
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
        },
    }


def build_transitions(data_root: Path, output: Path) -> dict[str, object]:
    measurements = read_measurements(data_root)
    meta = read_patient_meta(data_root)
    anchors, dka_like_stays = build_anchors(data_root, measurements, meta)
    actions = read_actions(data_root)
    states = assemble_states(anchors, measurements)
    action_summary = assemble_actions(anchors, actions)
    outcomes = assemble_outcomes(anchors, meta)
    frame = pd.concat(
        [
            anchors[[
                "subject_id", "stay_id", "onset", "onset_hour", "onset_criteria",
                "hours_since_onset", "t", "t_plus",
            ]],
            states,
            action_summary,
            outcomes,
        ],
        axis=1,
    )
    frame = filter_evaluable(add_derived_columns(frame))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return cohort_report(frame, data_root, output, dka_like_stays, actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("physionet.org/files/eicu-crd-demo/2.0.1"),
    )
    parser.add_argument("--output", type=Path, default=Path("eicu_dka_transitions_6h_demo.parquet"))
    parser.add_argument("--report", type=Path, default=Path("eicu_dka_transition_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_transitions(args.data_root, args.output)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "stays": report["stays"],
        "transitions": report["transitions"],
        "action_target_support": report["action_target_support"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2))


if __name__ == "__main__":
    main()
