"""Build a MIMIC-III demo DKA-like observed-treatment transition cohort.

MIMIC-III demo v1.4 is small and contains no ICD-coded DKA admissions in the
local demo files. This adapter therefore treats it as a lab-defined DKA-like
schema/power audit, not as a diagnosis-confirmed DKA cohort.

The output mirrors the existing factual proxy contract:

    state_t + history_action_grid + future_action_grid + state_t+6h

No hidden treatment is inferred from outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_dka_transition_extract import (
    ANCHOR_STEP_H,
    DELTA_H,
    DKA_GLUCOSE_MIN,
    DKA_HCO3_MAX,
    DKA_PH_MAX,
    EPISODE_MAX_H,
    LOOKBACK_H,
    PLAUSIBLE,
    STATE_VARS,
    TARGET_TOL_H,
    action_terms_mask,
    assemble_actions,
    assemble_states,
    filter_evaluable,
    support_counts,
)
from mimic_action_history import deduplicate_events, normalize_events


DKA_COOCCUR_H = 4.0
LAB_ITEMIDS = {
    "glucose": [50809, 50931],
    "ph": [50820],
    "bicarbonate": [50803, 50882],
    "anion_gap": [50868],
    "potassium": [50822, 50971],
    "lactate": [50813],
    "creatinine": [50912],
    "sodium": [50824, 50983],
    "osmolality": [50964],
    "BHB": [],
}
CHART_ITEMIDS = {
    "glucose": [
        1455, 807, 811, 1310, 3744, 3745, 1529, 1812, 2338,
        220621, 225664, 226537, 228388,
    ],
    "ph": [780, 1126, 3839, 4202, 4753, 223830],
    "heart_rate": [211, 220045],
    "sbp": [6, 51, 442, 455, 3313, 220050, 220179, 224167, 225309, 227243],
    "dbp": [8364, 8368, 8440, 8441, 8502, 220051, 220180, 224643, 225310, 227242],
    "map": [52, 456, 438, 1321, 5680, 6399, 220052, 220181, 225312],
}
DKA_ICD9_CODES = {"24910", "24911", "25010", "25011", "25012", "25013"}
ACTION_TERMS = (
    "insulin", "saline", "sodium chloride", "nacl", "lactated", "ringer",
    "potassium chloride", "kcl", "bicarbonate", "dextrose", "d5", "d10",
    "d20", "d50",
)


def read_icu(data_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        data_root / "ICUSTAYS.csv",
        parse_dates=["intime", "outtime"],
    )
    frame = frame.rename(columns={"icustay_id": "stay_id"})
    return frame[[
        "subject_id", "hadm_id", "stay_id", "intime", "outtime", "los",
    ]].drop_duplicates("stay_id")


def _clip(var: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float32")
    lower, upper = PLAUSIBLE[var]
    return numeric.where((numeric >= lower) & (numeric <= upper))


def read_lab_measurements(data_root: Path, icu: pd.DataFrame) -> pd.DataFrame:
    ids = sorted({itemid for values in LAB_ITEMIDS.values() for itemid in values})
    frame = pd.read_csv(
        data_root / "LABEVENTS.csv",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    frame = frame[frame["itemid"].isin(ids)].copy()
    item_to_var = {
        itemid: var for var, values in LAB_ITEMIDS.items() for itemid in values
    }
    frame["var"] = frame["itemid"].map(item_to_var)
    frame["valuenum"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    frame = frame.dropna(subset=["var", "valuenum"])
    merged = frame.merge(
        icu[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    merged = merged[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] <= merged["outtime"])
    ].copy()
    for var in sorted(set(LAB_ITEMIDS) & set(PLAUSIBLE)):
        mask = merged["var"] == var
        if mask.any():
            merged.loc[mask, "valuenum"] = _clip(var, merged.loc[mask, "valuenum"])
    merged = merged.dropna(subset=["valuenum"])
    return merged[[
        "subject_id", "hadm_id", "stay_id", "charttime", "var", "valuenum",
    ]]


def read_chart_measurements(data_root: Path, icu: pd.DataFrame) -> pd.DataFrame:
    ids = sorted({itemid for values in CHART_ITEMIDS.values() for itemid in values})
    frame = pd.read_csv(
        data_root / "CHARTEVENTS.csv",
        usecols=[
            "subject_id", "hadm_id", "icustay_id", "itemid", "charttime",
            "valuenum",
        ],
        parse_dates=["charttime"],
        low_memory=False,
    )
    frame = frame[frame["itemid"].isin(ids)].copy()
    item_to_var = {
        itemid: var for var, values in CHART_ITEMIDS.items() for itemid in values
    }
    frame["var"] = frame["itemid"].map(item_to_var)
    frame["valuenum"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    frame = frame.dropna(subset=["var", "valuenum"]).rename(
        columns={"icustay_id": "stay_id"}
    )
    frame = frame.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    frame = frame[
        (frame["charttime"] >= frame["intime"])
        & (frame["charttime"] <= frame["outtime"])
    ].copy()
    for var in sorted(set(CHART_ITEMIDS) & set(PLAUSIBLE)):
        mask = frame["var"] == var
        if mask.any():
            frame.loc[mask, "valuenum"] = _clip(var, frame.loc[mask, "valuenum"])
    frame = frame.dropna(subset=["valuenum"])
    return frame[[
        "subject_id", "hadm_id", "stay_id", "charttime", "var", "valuenum",
    ]]


def read_urine_output(data_root: Path, icu: pd.DataFrame) -> pd.DataFrame:
    items = pd.read_csv(data_root / "D_ITEMS.csv", usecols=["itemid", "label", "linksto"])
    labels = items["label"].fillna("").astype(str).str.lower()
    output_items = items[
        items["linksto"].fillna("").str.lower().eq("outputevents")
        & labels.str.contains("urine|foley|void", regex=True)
        & ~labels.str.contains("irrig", regex=True)
    ]["itemid"].astype(int).tolist()
    columns = ["subject_id", "hadm_id", "stay_id", "charttime", "var", "valuenum"]
    if not output_items:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(
        data_root / "OUTPUTEVENTS.csv",
        usecols=["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "value"],
        parse_dates=["charttime"],
    )
    frame = frame[frame["itemid"].isin(output_items)].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).rename(columns={"icustay_id": "stay_id"})
    frame = frame[frame["value"] >= 0]
    frame = frame.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    frame = frame[
        (frame["charttime"] >= frame["intime"])
        & (frame["charttime"] <= frame["outtime"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for stay_id, group in frame.groupby("stay_id", sort=False):
        group = group.sort_values("charttime").set_index("charttime")
        hourly = group["value"].rolling("4h", closed="both").sum() / 4.0
        part = hourly.rename("valuenum").reset_index()
        part["subject_id"] = int(group["subject_id"].iloc[0])
        part["hadm_id"] = int(group["hadm_id"].iloc[0])
        part["stay_id"] = int(stay_id)
        part["var"] = "urine_output"
        rows.append(part[columns])
    output = pd.concat(rows, ignore_index=True)
    output["valuenum"] = _clip("urine_output", output["valuenum"])
    return output.dropna(subset=["valuenum"])


def read_measurements(data_root: Path, icu: pd.DataFrame) -> pd.DataFrame:
    measurements = pd.concat(
        [
            read_lab_measurements(data_root, icu),
            read_chart_measurements(data_root, icu),
            read_urine_output(data_root, icu),
        ],
        ignore_index=True,
    )
    measurements = measurements.merge(icu[["stay_id", "intime"]], on="stay_id", how="left")
    measurements["time_hr"] = (
        measurements["charttime"] - measurements["intime"]
    ).dt.total_seconds() / 3600.0
    return measurements[np.isfinite(measurements["time_hr"])].sort_values(
        ["stay_id", "time_hr", "var"]
    ).reset_index(drop=True)


def dka_diagnosis_stays(data_root: Path, icu: pd.DataFrame) -> set[int]:
    diag = pd.read_csv(data_root / "DIAGNOSES_ICD.csv", dtype={"icd9_code": str})
    selected = diag[diag["icd9_code"].fillna("").isin(DKA_ICD9_CODES)]
    if selected.empty:
        return set()
    stays = selected.merge(icu[["hadm_id", "stay_id"]], on="hadm_id", how="inner")
    return set(stays["stay_id"].astype(int).tolist())


def lab_dka_like_stays(measurements: pd.DataFrame) -> dict[int, dict[str, object]]:
    output = {}
    for stay_id, group in measurements.groupby("stay_id", sort=False):
        glucose = group[
            (group["var"] == "glucose")
            & (group["valuenum"] >= DKA_GLUCOSE_MIN)
        ]
        acidosis = group[
            ((group["var"] == "bicarbonate") & (group["valuenum"] < DKA_HCO3_MAX))
            | ((group["var"] == "ph") & (group["valuenum"] < DKA_PH_MAX))
            | ((group["var"] == "anion_gap") & (group["valuenum"] > 12.0))
        ]
        if glucose.empty or acidosis.empty:
            continue
        best_time = None
        criteria = set()
        for glucose_row in glucose.itertuples(index=False):
            delta = (
                acidosis["charttime"] - pd.Timestamp(glucose_row.charttime)
            ).abs().dt.total_seconds() / 3600.0
            close = acidosis[delta <= DKA_COOCCUR_H]
            if close.empty:
                continue
            anchor = min(pd.Timestamp(glucose_row.charttime), close["charttime"].min())
            best_time = anchor if best_time is None or anchor < best_time else best_time
            criteria.add("hyperglycemia")
            criteria.update(str(var) for var in close["var"].unique())
        if best_time is not None:
            first = group.iloc[0]
            output[int(stay_id)] = {
                "subject_id": int(first["subject_id"]),
                "hadm_id": int(first["hadm_id"]),
                "anchor_time": best_time,
                "criteria": sorted(criteria),
            }
    return output


def mimiciii_inputevents_raw(data_root: Path) -> pd.DataFrame:
    items = pd.read_csv(data_root / "D_ITEMS.csv", usecols=["itemid", "label"])
    item_labels = items.set_index("itemid")["label"].to_dict()

    cv = pd.read_csv(
        data_root / "INPUTEVENTS_CV.csv",
        usecols=[
            "subject_id", "hadm_id", "icustay_id", "charttime", "itemid",
            "amount", "amountuom", "rate", "rateuom", "originalroute", "orderid",
        ],
        parse_dates=["charttime"],
    )
    cv["label"] = cv["itemid"].map(item_labels)
    cv = cv[action_terms_mask(cv["label"])].copy()
    cv_raw = pd.DataFrame({
        "subject_id": cv["subject_id"],
        "stay_id": cv["icustay_id"],
        "starttime": cv["charttime"],
        "endtime": cv["charttime"],
        "label": cv["label"],
        "amount": cv["amount"],
        "uom": cv["amountuom"],
        "rate": cv["rate"],
        "rate_uom": cv["rateuom"],
        "route": cv["originalroute"],
        "product_description": "mimiciii_cv",
        "itemid": cv["itemid"],
        "orderid": cv["orderid"],
        "source": "inputevents",
    })

    mv = pd.read_csv(
        data_root / "INPUTEVENTS_MV.csv",
        usecols=[
            "subject_id", "hadm_id", "icustay_id", "starttime", "endtime",
            "itemid", "amount", "amountuom", "rate", "rateuom", "orderid",
        ],
        parse_dates=["starttime", "endtime"],
    )
    mv["label"] = mv["itemid"].map(item_labels)
    mv = mv[action_terms_mask(mv["label"])].copy()
    mv_raw = pd.DataFrame({
        "subject_id": mv["subject_id"],
        "stay_id": mv["icustay_id"],
        "starttime": mv["starttime"],
        "endtime": mv["endtime"],
        "label": mv["label"],
        "amount": mv["amount"],
        "uom": mv["amountuom"],
        "rate": mv["rate"],
        "rate_uom": mv["rateuom"],
        "route": "IV",
        "product_description": "mimiciii_mv",
        "itemid": mv["itemid"],
        "orderid": mv["orderid"],
        "source": "inputevents",
    })
    return pd.concat([cv_raw, mv_raw], ignore_index=True)


def read_actions(data_root: Path) -> pd.DataFrame:
    raw = mimiciii_inputevents_raw(data_root)
    if raw.empty:
        return normalize_events(raw)
    events = normalize_events(raw)
    events = events[events["amount"].notna() & (events["amount"] > 0)]
    return deduplicate_events(events).sort_values(["stay_id", "starttime"]).reset_index(drop=True)


def prescription_order_matches(data_root: Path) -> int:
    frame = pd.read_csv(
        data_root / "PRESCRIPTIONS.csv",
        usecols=["drug", "drug_name_generic", "route", "dose_val_rx", "dose_unit_rx"],
    )
    text = frame.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(False, index=frame.index)
    for term in ACTION_TERMS:
        mask |= text.str.contains(term, regex=False)
    return int(mask.sum())


def build_anchors(icu: pd.DataFrame, measurements: pd.DataFrame,
                  dka_like: dict[int, dict[str, object]]) -> pd.DataFrame:
    max_observed = measurements.groupby("stay_id")["charttime"].max().to_dict()
    icu_index = icu.set_index("stay_id", drop=False)
    rows = []
    for stay_id, payload in dka_like.items():
        if stay_id not in icu_index.index or stay_id not in max_observed:
            continue
        stay = icu_index.loc[stay_id]
        anchor_time = pd.Timestamp(payload["anchor_time"])
        last = min(
            anchor_time + pd.Timedelta(hours=EPISODE_MAX_H),
            pd.Timestamp(stay["outtime"]) - pd.Timedelta(hours=DELTA_H),
            pd.Timestamp(max_observed[stay_id]) - pd.Timedelta(hours=DELTA_H),
        )
        current = anchor_time
        while current <= last:
            rows.append({
                "subject_id": int(stay["subject_id"]),
                "stay_id": int(stay_id),
                "onset": anchor_time,
                "onset_hour": (
                    anchor_time - pd.Timestamp(stay["intime"])
                ).total_seconds() / 3600.0,
                "onset_criteria": payload["criteria"],
                "hours_since_onset": (
                    current - anchor_time
                ).total_seconds() / 3600.0,
                "t": current,
                "t_hour": (
                    current - pd.Timestamp(stay["intime"])
                ).total_seconds() / 3600.0,
                "t_plus": current + pd.Timedelta(hours=DELTA_H),
                "t_plus_hour": (
                    current + pd.Timedelta(hours=DELTA_H)
                    - pd.Timestamp(stay["intime"])
                ).total_seconds() / 3600.0,
            })
            current += pd.Timedelta(hours=ANCHOR_STEP_H)
    return pd.DataFrame(rows)


def assemble_outcomes(anchors: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    admissions = pd.read_csv(
        data_root / "ADMISSIONS.csv",
        usecols=["subject_id", "deathtime"],
        parse_dates=["deathtime"],
    )
    death_by_subject = admissions.groupby("subject_id")["deathtime"].min().to_dict()
    rows = []
    for anchor in anchors.itertuples(index=False):
        death_time = death_by_subject.get(int(anchor.subject_id), pd.NaT)
        died = pd.notna(death_time) and pd.Timestamp(death_time) > pd.Timestamp(anchor.t_plus)
        rows.append({
            "died_after_window": int(died),
            "hrs_to_death_from_cut": (
                (pd.Timestamp(death_time) - pd.Timestamp(anchor.t_plus)).total_seconds() / 3600.0
                if died else np.nan
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


def cohort_report(frame: pd.DataFrame, data_root: Path, output: Path,
                  diag_stays: set[int], lab_stays: dict[int, dict[str, object]],
                  actions: pd.DataFrame, prescription_matches: int) -> dict[str, object]:
    core_pair_counts = {
        var: int((frame[f"{var}_t"].notna() & frame[f"{var}_tp6"].notna()).sum())
        for var in ("glucose", "potassium", "bicarbonate", "ph", "map")
        if f"{var}_t" in frame and f"{var}_tp6" in frame
    } if not frame.empty else {}
    action_support = support_counts(frame) if not frame.empty else {}
    return {
        "dataset": "PhysioNet MIMIC-III Clinical Database Demo 1.4",
        "data_root": str(data_root),
        "output": str(output),
        "stays": int(frame["stay_id"].nunique()) if not frame.empty else 0,
        "transitions": int(len(frame)),
        "active_dka_transitions": int(frame["dka_active_t"].fillna(False).sum())
        if not frame.empty else 0,
        "diagnosis_confirmed_dka_stays": int(len(diag_stays)),
        "lab_dka_like_stays_before_evaluable_filter": int(len(lab_stays)),
        "dka_like_stay_count_before_evaluable_filter": int(
            len(set(diag_stays) | set(lab_stays))
        ),
        "core_pair_counts": core_pair_counts,
        "action_target_support": action_support,
        "power_proxy": {
            "map_gate_reference_stays": 42,
            "glucose_gate_reference_stays": 77,
            "note": (
                "Reference thresholds come from prior MIMIC-demo power analysis; "
                "MIMIC-III demo is too small and has no ICD-confirmed DKA."
            ),
            "support_passes_reference": {
                "fluids_to_map": action_support.get("fluids_to_map", {}).get("stays", 0) >= 42,
                "insulin_any_to_glucose": action_support.get("insulin_any_to_glucose", {}).get("stays", 0) >= 77,
                "kcl_to_potassium": action_support.get("kcl_to_potassium", {}).get("stays", 0) >= 77,
            },
        },
        "action_quality": {
            "normalized_event_rows": int(len(actions)),
            "sources": actions["source"].value_counts().astype(int).to_dict()
            if not actions.empty else {},
            "exact_timing_fraction": round(
                float((actions["timing_source"] == "recorded_interval").mean()), 6
            ) if not actions.empty else 0.0,
            "high_confidence_dose_fraction": round(
                float((actions["dose_confidence"] >= 0.75).mean()), 6
            ) if not actions.empty else 0.0,
            "prescription_order_matches_not_used_as_administrations": int(
                prescription_matches
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
            "diagnosis_pure_dka": False,
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
        },
    }


def build_transitions(data_root: Path, output: Path) -> dict[str, object]:
    icu = read_icu(data_root)
    measurements = read_measurements(data_root, icu)
    diag_stays = dka_diagnosis_stays(data_root, icu)
    lab_stays = lab_dka_like_stays(measurements)
    anchors = build_anchors(icu, measurements, lab_stays)
    actions = read_actions(data_root)
    prescription_matches = prescription_order_matches(data_root)
    if anchors.empty:
        frame = pd.DataFrame()
    else:
        states = assemble_states(anchors, measurements)
        action_summary = assemble_actions(anchors, actions)
        outcomes = assemble_outcomes(anchors, data_root)
        frame = pd.concat(
            [
                anchors[[
                    "subject_id", "stay_id", "onset", "onset_hour",
                    "onset_criteria", "hours_since_onset", "t", "t_plus",
                ]],
                states,
                action_summary,
                outcomes,
            ],
            axis=1,
        )
        frame = filter_evaluable(add_derived_columns(frame))
    if not frame.empty:
        frame.to_parquet(output, index=False)
    return cohort_report(
        frame, data_root, output, diag_stays, lab_stays, actions,
        prescription_matches,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("physionet.org/files/mimiciii-demo/1.4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mimiciii_dka_transitions_6h_demo.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("mimiciii_dka_transition_report.json"),
    )
    args = parser.parse_args()

    report = build_transitions(args.data_root, args.output)
    args.report.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.report}")


if __name__ == "__main__":
    main()
