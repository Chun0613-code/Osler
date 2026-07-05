"""Build MIMIC-IV whole-ICU observation transition cohorts.

This adapter is deliberately observational-only.  It maps credentialed
MIMIC-IV v3.1 labs, vitals, and urine output into the same ``*_t`` /
``*_tp6`` contract used by the eICU whole-body observation audits.  With
``--include-treatment-context`` it also adds factual observed-treatment
``hist_*`` and ``act_*`` context from inputevents, emar, and procedureevents.

The output parquet is local-only and ignored by git.  Reports are aggregate
only: no row-level predictions, patient identifiers, raw values, treatment
effects, or causal claims are meant to be committed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS
from eicu_sepsis_transition_extract import PLAUSIBLE


MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", "/Users/chunyouchang/Downloads/mimic-iv-3.1"))
DELTA_H = 6.0
ANCHOR_STEP_H = 6.0
LOOKBACK_H = 6.0
TARGET_TOL_H = 2.0
EPISODE_MAX_H = 72.0


LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "glucose": (50809, 50931, 52027, 52569),
    "ph": (50820,),
    "paco2": (50818,),
    "pao2": (50821,),
    "bicarbonate": (50803, 50882),
    "anion_gap": (50868, 52500),
    "chloride": (50806, 50902, 52434, 52535),
    "potassium": (50822, 50971, 52452, 52610),
    "sodium": (50824, 50983, 52455, 52623),
    "calcium": (50893, 52034, 52035),
    "ionized_calcium": (50808, 51624),
    "magnesium": (50960,),
    "phosphate": (50970,),
    "lactate": (50813, 52442, 53154),
    "creatinine": (50912, 52024, 52546),
    "bun": (51006, 52647),
    "wbc": (51301, 51755, 51756),
    "hemoglobin": (50811, 51222, 51640),
    "hematocrit": (50810, 51221, 51638, 51639, 52028),
    "bilirubin": (50885, 53089),
    "bilirubin_direct": (50883,),
    "platelets": (51265, 53189),
    "inr": (51237, 51675),
    "ptt": (51275, 52923),
    "fibrinogen": (51214, 51623, 52116),
    "albumin": (50862, 53085),
    "total_protein": (50976, 53096),
    "ast": (50878,),
    "alt": (50861,),
    "alkaline_phos": (50863, 53086),
    "lipase": (50956,),
    "amylase": (50867, 53087),
    "triglycerides": (51000,),
    "troponin_i": (51002, 52642),
    "troponin_t": (51003,),
    "bnp": (50963,),
    "cpk": (50910,),
    "ck_mb": (50911, 51580),
    "ldh": (50954,),
    "myoglobin": (51695,),
    "serum_osmolality": (50964, 51701, 52031),
    "serum_ketones": (51567,),
    "tsh": (50993,),
    "cortisol": (50909,),
    "crp": (50889,),
    "esr": (51288,),
    "ferritin": (50924,),
    "uric_acid": (51007,),
}


VITAL_ITEMIDS: dict[str, tuple[int, ...]] = {
    "heart_rate": (220045,),
    "respiratory_rate": (220210, 224689, 224690),
    "o2sat": (220277,),
    "temperature_f": (223761,),
    "temperature": (223762, 226329),
    "map": (220052, 220181),
    "fio2": (223835, 229841),
    "peep": (220339, 224699, 224700),
    "minute_volume": (224687,),
}


EXTRA_PLAUSIBLE = {
    "paco2": (5.0, 200.0),
    "pao2": (10.0, 700.0),
    "fio2": (20.0, 100.0),
    "peep": (0.0, 40.0),
    "minute_volume": (0.1, 40.0),
}


STATE_VARS = tuple(sorted(set(LAB_ITEMIDS) | (set(VITAL_ITEMIDS) - {"temperature_f"}) | {"urine_output"}))


def merged_action_terms() -> dict[str, tuple[str, ...]]:
    terms: dict[str, set[str]] = {}
    for config in BODY_SYSTEM_CONFIGS.values():
        for action, values in config.action_terms.items():
            terms.setdefault(action, set()).update(str(value).lower() for value in values)
    return {action: tuple(sorted(values)) for action, values in sorted(terms.items())}


MIMIC_ACTION_TERMS = merged_action_terms()
ACTION_KEYS = tuple(sorted(MIMIC_ACTION_TERMS))
ACTION_EVIDENCE_COLUMNS = (
    "stay_id",
    "starttime",
    "endtime",
    "action",
    "source",
    "dose_observed",
    "amount_like",
    "rate_like",
    "original_label",
)


def con():
    import duckdb

    c = duckdb.connect()
    c.execute("PRAGMA threads=4")
    return c


def csv_expr(path: Path) -> str:
    return f"read_csv_auto('{path}', all_varchar=false)"


def hosp_path(mimic_dir: Path, name: str) -> Path:
    return mimic_dir / "hosp" / f"{name}.csv.gz"


def icu_path(mimic_dir: Path, name: str) -> Path:
    return mimic_dir / "icu" / f"{name}.csv.gz"


def sql_case(mapping: dict[str, tuple[int, ...]], alias: str) -> str:
    parts = []
    for variable, itemids in mapping.items():
        ids = ",".join(str(itemid) for itemid in itemids)
        parts.append(f"WHEN {alias}.itemid IN ({ids}) THEN '{variable}'")
    return " ".join(parts)


def sql_text_match_expr(columns: tuple[str, ...], terms: tuple[str, ...]) -> str:
    text_expr = " || ' ' || ".join(f"lower(coalesce({column}, ''))" for column in columns)
    clauses = []
    for term in sorted(set(terms)):
        escaped = str(term).lower().replace("'", "''")
        clauses.append(f"{text_expr} LIKE '%{escaped}%'")
    return "(" + " OR ".join(clauses) + ")" if clauses else "FALSE"


def classify_treatment_label(label: object) -> tuple[str, ...]:
    text = str(label or "").lower()
    compact = text.replace(" ", "").replace("-", "")
    actions = []
    for action, terms in MIMIC_ACTION_TERMS.items():
        if any(term in text for term in terms):
            actions.append(action)
        elif action == "fluids" and compact in {"ns", "lr", "d5w", "d10w", "d50w"}:
            actions.append(action)
    return tuple(dict.fromkeys(actions))


def empty_treatment_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_EVIDENCE_COLUMNS)


def clean_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["charttime"] = pd.to_datetime(frame["charttime"])
    frame["valuenum"] = pd.to_numeric(frame["valuenum"], errors="coerce").astype("float64")
    temp_f = frame["var"] == "temperature_f"
    if temp_f.any():
        frame.loc[temp_f, "valuenum"] = (frame.loc[temp_f, "valuenum"] - 32.0) * (5.0 / 9.0)
        frame.loc[temp_f, "var"] = "temperature"
    for var, bounds in {**PLAUSIBLE, **EXTRA_PLAUSIBLE}.items():
        lo, hi = bounds
        mask = frame["var"] == var
        if mask.any():
            frame.loc[mask, "valuenum"] = frame.loc[mask, "valuenum"].where(
                (frame.loc[mask, "valuenum"] >= lo) & (frame.loc[mask, "valuenum"] <= hi)
            )
    return frame.dropna(subset=["valuenum"]).sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def pull_stay_meta(c, mimic_dir: Path, max_stays: int | None) -> pd.DataFrame:
    limit = "" if max_stays is None else f"LIMIT {int(max_stays)}"
    frame = c.execute(f"""
        SELECT subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime
        FROM {csv_expr(icu_path(mimic_dir, 'icustays'))}
        WHERE outtime > intime + INTERVAL '{int(DELTA_H)} hours'
        ORDER BY stay_id
        {limit}
    """).df()
    for column in ("intime", "outtime"):
        frame[column] = pd.to_datetime(frame[column])
    frame["los_hours"] = (frame["outtime"] - frame["intime"]).dt.total_seconds() / 3600.0
    return frame


def register_stays(c, meta: pd.DataFrame) -> None:
    c.register("_mimic_obs_stays", meta[["stay_id"]])
    c.execute("CREATE OR REPLACE TEMP TABLE target_stays AS SELECT stay_id FROM _mimic_obs_stays")


def pull_measurements(c, mimic_dir: Path) -> pd.DataFrame:
    lab_ids = sorted({itemid for itemids in LAB_ITEMIDS.values() for itemid in itemids})
    vital_ids = sorted({itemid for itemids in VITAL_ITEMIDS.values() for itemid in itemids})
    lab_case = sql_case(LAB_ITEMIDS, "le")
    vital_case = sql_case(VITAL_ITEMIDS, "ce")
    labs = c.execute(f"""
        SELECT ie.subject_id, ie.stay_id, le.charttime,
               CASE {lab_case} END AS var,
               le.valuenum
        FROM {csv_expr(hosp_path(mimic_dir, 'labevents'))} le
        JOIN {csv_expr(icu_path(mimic_dir, 'icustays'))} ie
          ON le.hadm_id = ie.hadm_id
         AND le.charttime BETWEEN ie.intime AND ie.outtime
        JOIN target_stays ts ON ie.stay_id = ts.stay_id
        WHERE le.itemid IN ({",".join(str(itemid) for itemid in lab_ids)})
          AND le.valuenum IS NOT NULL
    """).df()
    vitals = c.execute(f"""
        SELECT ce.subject_id, ce.stay_id, ce.charttime,
               CASE {vital_case} END AS var,
               ce.valuenum
        FROM {csv_expr(icu_path(mimic_dir, 'chartevents'))} ce
        JOIN target_stays ts ON ce.stay_id = ts.stay_id
        WHERE ce.itemid IN ({",".join(str(itemid) for itemid in vital_ids)})
          AND ce.valuenum IS NOT NULL
    """).df()
    urine = c.execute(f"""
        SELECT oe.subject_id, oe.stay_id, oe.charttime, SUM(oe.value) AS valuenum
        FROM {csv_expr(icu_path(mimic_dir, 'outputevents'))} oe
        JOIN {csv_expr(icu_path(mimic_dir, 'd_items'))} di ON oe.itemid = di.itemid
        JOIN target_stays ts ON oe.stay_id = ts.stay_id
        WHERE (
                lower(di.label) IN ('foley', 'void', 'condom cath', 'straight cath')
             OR lower(di.label) LIKE '%urine%'
        )
          AND lower(di.label) NOT LIKE '%irrigant%'
          AND lower(coalesce(oe.valueuom, 'ml')) LIKE '%ml%'
          AND oe.value >= 0
        GROUP BY 1,2,3
    """).df()
    if not urine.empty:
        urine["var"] = "urine_output"
    frame = pd.concat(
        [
            labs[["subject_id", "stay_id", "charttime", "var", "valuenum"]],
            vitals[["subject_id", "stay_id", "charttime", "var", "valuenum"]],
            urine[["subject_id", "stay_id", "charttime", "var", "valuenum"]],
        ],
        ignore_index=True,
    )
    return clean_measurements(frame)


def _finite_numeric(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _evidence_rows_from_records(records: list[dict[str, object]], source: str) -> pd.DataFrame:
    rows = []
    for record in records:
        label = str(record.get("label") or "")
        actions = classify_treatment_label(label)
        if not actions:
            continue
        start = pd.to_datetime(record.get("starttime"), errors="coerce")
        if pd.isna(start):
            continue
        end = pd.to_datetime(record.get("endtime"), errors="coerce")
        if pd.isna(end) or end <= start:
            end = start + pd.Timedelta(hours=1)
        amount = _finite_numeric(record.get("amount_like"))
        rate = _finite_numeric(record.get("rate_like"))
        dose_observed = bool((np.isfinite(amount) and amount > 0.0) or (np.isfinite(rate) and rate > 0.0))
        for action in actions:
            rows.append({
                "stay_id": int(record["stay_id"]),
                "starttime": start,
                "endtime": end,
                "action": action,
                "source": source,
                "dose_observed": dose_observed,
                "amount_like": amount,
                "rate_like": rate,
                "original_label": label,
            })
    return pd.DataFrame(rows, columns=ACTION_EVIDENCE_COLUMNS) if rows else empty_treatment_evidence()


def pull_inputevent_evidence(c, mimic_dir: Path) -> pd.DataFrame:
    terms = tuple(term for values in MIMIC_ACTION_TERMS.values() for term in values)
    text_filter = sql_text_match_expr(
        (
            "di.label",
            "ie.ordercategoryname",
            "ie.secondaryordercategoryname",
            "ie.ordercomponenttypedescription",
            "ie.ordercategorydescription",
        ),
        terms,
    )
    frame = c.execute(f"""
        SELECT ie.stay_id,
               ie.starttime,
               ie.endtime,
               concat_ws(' ', di.label, ie.ordercategoryname, ie.secondaryordercategoryname,
                         ie.ordercomponenttypedescription, ie.ordercategorydescription) AS label,
               ie.amount AS amount_like,
               ie.rate AS rate_like
        FROM {csv_expr(icu_path(mimic_dir, 'inputevents'))} ie
        JOIN {csv_expr(icu_path(mimic_dir, 'd_items'))} di ON ie.itemid = di.itemid
        JOIN target_stays ts ON ie.stay_id = ts.stay_id
        WHERE ie.starttime IS NOT NULL
          AND {text_filter}
    """).df()
    return _evidence_rows_from_records(frame.to_dict("records"), "mimic_inputevents")


def pull_emar_evidence(c, mimic_dir: Path) -> pd.DataFrame:
    terms = tuple(term for values in MIMIC_ACTION_TERMS.values() for term in values)
    text_filter = sql_text_match_expr(("emar.medication", "emar.event_txt"), terms)
    frame = c.execute(f"""
        SELECT ie.stay_id,
               emar.charttime AS starttime,
               emar.charttime + INTERVAL '1 hour' AS endtime,
               concat_ws(' ', emar.medication, emar.event_txt) AS label,
               NULL AS amount_like,
               NULL AS rate_like,
               lower(coalesce(emar.event_txt, '')) AS event_text
        FROM {csv_expr(hosp_path(mimic_dir, 'emar'))} emar
        JOIN {csv_expr(icu_path(mimic_dir, 'icustays'))} ie
          ON emar.hadm_id = ie.hadm_id
         AND emar.charttime BETWEEN ie.intime AND ie.outtime
        JOIN target_stays ts ON ie.stay_id = ts.stay_id
        WHERE emar.charttime IS NOT NULL
          AND {text_filter}
    """).df()
    if not frame.empty:
        blocked = (
            frame["event_text"].str.contains("not given", regex=False, na=False)
            | frame["event_text"].str.contains("held", regex=False, na=False)
            | frame["event_text"].str.contains("missed", regex=False, na=False)
            | frame["event_text"].str.contains("refused", regex=False, na=False)
            | frame["event_text"].str.contains("canceled", regex=False, na=False)
            | frame["event_text"].str.contains("cancelled", regex=False, na=False)
        )
        frame = frame.loc[~blocked].copy()
    return _evidence_rows_from_records(frame.to_dict("records"), "mimic_emar")


def pull_procedure_evidence(c, mimic_dir: Path) -> pd.DataFrame:
    terms = tuple(term for values in MIMIC_ACTION_TERMS.values() for term in values)
    text_filter = sql_text_match_expr(("di.label", "pe.ordercategoryname"), terms)
    frame = c.execute(f"""
        SELECT pe.stay_id,
               pe.starttime,
               pe.endtime,
               concat_ws(' ', di.label, pe.ordercategoryname) AS label,
               pe.value AS amount_like,
               NULL AS rate_like
        FROM {csv_expr(icu_path(mimic_dir, 'procedureevents'))} pe
        JOIN {csv_expr(icu_path(mimic_dir, 'd_items'))} di ON pe.itemid = di.itemid
        JOIN target_stays ts ON pe.stay_id = ts.stay_id
        WHERE pe.starttime IS NOT NULL
          AND {text_filter}
    """).df()
    return _evidence_rows_from_records(frame.to_dict("records"), "mimic_procedureevents")


def pull_treatment_evidence(c, mimic_dir: Path) -> pd.DataFrame:
    frames = [
        pull_inputevent_evidence(c, mimic_dir),
        pull_emar_evidence(c, mimic_dir),
        pull_procedure_evidence(c, mimic_dir),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_treatment_evidence()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["stay_id", "starttime", "source", "action"])
        .reset_index(drop=True)
    )


def build_anchors(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in meta.itertuples(index=False):
        stop = min(
            record.outtime - pd.Timedelta(hours=DELTA_H),
            record.intime + pd.Timedelta(hours=EPISODE_MAX_H),
        )
        t = record.intime + pd.Timedelta(hours=LOOKBACK_H)
        while t <= stop:
            rows.append((record.subject_id, record.stay_id, record.first_careunit, record.last_careunit, t))
            t += pd.Timedelta(hours=ANCHOR_STEP_H)
    anchors = pd.DataFrame(rows, columns=["subject_id", "stay_id", "first_careunit", "last_careunit", "t"])
    anchors["subject_id"] = anchors["subject_id"].astype(str)
    anchors["t_plus"] = anchors["t"] + pd.Timedelta(hours=DELTA_H)
    return anchors


def assemble_state(
    anchors: pd.DataFrame,
    measurements: pd.DataFrame,
    suffix: str,
    when_col: str,
    direction: str,
    tolerance_h: float,
) -> pd.DataFrame:
    pieces: dict[str, object] = {
        "stay_id": anchors["stay_id"].to_numpy(),
        when_col: anchors[when_col].to_numpy(),
    }
    left_base = anchors[["stay_id", when_col]].copy()
    left_base[when_col] = pd.to_datetime(left_base[when_col]).astype("datetime64[ns]")
    left_sorted = left_base.sort_values(when_col)
    tolerance = pd.Timedelta(hours=float(tolerance_h))
    for var in STATE_VARS:
        values = measurements.loc[
            measurements["var"] == var,
            ["stay_id", "charttime", "valuenum"],
        ].copy()
        if values.empty:
            pieces[f"{var}{suffix}"] = np.full(len(anchors), np.nan, dtype=np.float64)
            if suffix == "_t":
                pieces[f"{var}_age_hr"] = np.full(len(anchors), np.nan, dtype=np.float64)
            continue
        values["charttime"] = pd.to_datetime(values["charttime"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            left_sorted,
            values.sort_values("charttime"),
            left_on=when_col,
            right_on="charttime",
            by="stay_id",
            direction=direction,
            tolerance=tolerance,
        )
        merged = merged.set_index(left_sorted.index).reindex(left_base.index)
        pieces[f"{var}{suffix}"] = merged["valuenum"].to_numpy(dtype=np.float64)
        if suffix == "_t":
            age = (merged[when_col] - merged["charttime"]).dt.total_seconds() / 3600.0
            pieces[f"{var}_age_hr"] = age.to_numpy(dtype=np.float64)
    return pd.DataFrame(pieces, index=anchors.index)


def _timestamp_hours(series: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(series, errors="coerce")
    raw = timestamps.astype("int64").to_numpy(dtype=np.float64) / 3.6e12
    return np.where(timestamps.notna().to_numpy(), raw, np.nan)


def prepare_treatment_lookup(evidence: pd.DataFrame) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    if evidence.empty:
        return {}
    frame = evidence.copy()
    frame["start_hr"] = _timestamp_hours(frame["starttime"])
    frame["end_hr"] = _timestamp_hours(frame["endtime"])
    frame = frame[np.isfinite(frame["start_hr"]) & np.isfinite(frame["end_hr"])]
    lookup: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for (stay_id, action), group in frame.groupby(["stay_id", "action"], sort=False):
        lookup.setdefault(int(stay_id), {})[str(action)] = {
            "starts": group["start_hr"].to_numpy(dtype=np.float64),
            "ends": group["end_hr"].to_numpy(dtype=np.float64),
            "dose_observed": group["dose_observed"].fillna(False).to_numpy(dtype=bool),
            "amount_like": pd.to_numeric(group["amount_like"], errors="coerce").to_numpy(dtype=np.float64),
            "rate_like": pd.to_numeric(group["rate_like"], errors="coerce").to_numpy(dtype=np.float64),
        }
    return lookup


def _positive_sum(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0.0)
    return float(values[mask].sum()) if mask.any() else 0.0


def _positive_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0.0)
    return float(values[mask].mean()) if mask.any() else 0.0


def treatment_window_summary(
    action_lookup: dict[str, dict[str, np.ndarray]],
    anchor_hour: float,
    horizon_hours: float,
) -> dict[str, object]:
    output: dict[str, object] = {}
    history_start = float(anchor_hour - LOOKBACK_H)
    future_end = float(anchor_hour + horizon_hours)
    for action in ACTION_KEYS:
        payload = action_lookup.get(action)
        if payload is None:
            output[f"hist_{action}"] = 0
            output[f"act_{action}"] = 0
            output[f"hist_{action}_evidence_count"] = 0
            output[f"act_{action}_evidence_count"] = 0
            output[f"hist_{action}_dose_observed"] = 0
            output[f"act_{action}_dose_observed"] = 0
            output[f"hist_{action}_amount_like_sum"] = 0.0
            output[f"act_{action}_amount_like_sum"] = 0.0
            output[f"hist_{action}_rate_like_mean"] = 0.0
            output[f"act_{action}_rate_like_mean"] = 0.0
            continue
        starts = payload["starts"]
        ends = payload["ends"]
        dose_observed = payload["dose_observed"]
        amount = payload["amount_like"]
        rate = payload["rate_like"]
        hist_mask = (ends >= history_start) & (starts < anchor_hour)
        act_mask = (ends >= anchor_hour) & (starts < future_end)
        for prefix, mask in (("hist", hist_mask), ("act", act_mask)):
            output[f"{prefix}_{action}"] = int(mask.any())
            output[f"{prefix}_{action}_evidence_count"] = int(mask.sum())
            output[f"{prefix}_{action}_dose_observed"] = int(dose_observed[mask].any() if mask.any() else False)
            output[f"{prefix}_{action}_amount_like_sum"] = round(_positive_sum(amount[mask]), 6)
            output[f"{prefix}_{action}_rate_like_mean"] = round(_positive_mean(rate[mask]), 6)
    return output


def assemble_treatment_context(
    anchors: pd.DataFrame,
    evidence: pd.DataFrame,
    horizon_hours: float,
) -> pd.DataFrame:
    lookup = prepare_treatment_lookup(evidence)
    anchor_hours = _timestamp_hours(anchors["t"])
    rows = []
    for anchor, hour in zip(anchors.itertuples(index=False), anchor_hours):
        rows.append(treatment_window_summary(
            lookup.get(int(anchor.stay_id), {}),
            float(hour),
            horizon_hours,
        ))
    return pd.DataFrame(rows, index=anchors.index)


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for var in STATE_VARS:
        current = f"{var}_t"
        future = f"{var}_tp6"
        if current in frame and future in frame:
            counts[var] = int((frame[current].notna() & frame[future].notna()).sum())
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def action_support_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output = {}
    for action in ACTION_KEYS:
        act_column = f"act_{action}"
        hist_column = f"hist_{action}"
        if act_column not in frame:
            continue
        act_mask = frame[act_column].fillna(0).astype(bool)
        hist_mask = frame.get(hist_column, pd.Series(False, index=frame.index)).fillna(0).astype(bool)
        output[action] = {
            "history_windows": int(hist_mask.sum()),
            "future_windows": int(act_mask.sum()),
            "future_stays": int(frame.loc[act_mask, "stay_id"].nunique()),
            "dose_observed_windows": int(frame.get(f"act_{action}_dose_observed", pd.Series(0, index=frame.index)).fillna(0).sum()),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--output", type=Path, default=Path("mimiciv_observation_transitions_6h.parquet"))
    parser.add_argument("--report", type=Path, default=Path("mimiciv_observation_transition_report.json"))
    parser.add_argument("--max-stays", type=int, default=-1)
    parser.add_argument(
        "--include-treatment-context",
        action="store_true",
        help=(
            "Add observed factual treatment context from inputevents, emar, and "
            "procedureevents as hist_* and act_* features. This does not grant "
            "causal or treatment-effect authority."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_stays = None if args.max_stays < 0 else int(args.max_stays)
    c = con()
    meta = pull_stay_meta(c, args.mimic_dir, max_stays)
    register_stays(c, meta)
    print(f"Selected ICU stays: {len(meta):,}")
    measurements = pull_measurements(c, args.mimic_dir)
    print(f"Observation rows: {len(measurements):,}; stays with observations: {measurements['stay_id'].nunique():,}")
    anchors = build_anchors(meta)
    print(f"Candidate anchors: {len(anchors):,}")
    state_t = assemble_state(anchors, measurements, "_t", "t", "backward", LOOKBACK_H)
    state_f = assemble_state(anchors, measurements, "_tp6", "t_plus", "nearest", TARGET_TOL_H)
    treatment_evidence = empty_treatment_evidence()
    treatment_context = pd.DataFrame(index=anchors.index)
    if args.include_treatment_context:
        treatment_evidence = pull_treatment_evidence(c, args.mimic_dir)
        print(
            "Treatment evidence rows: "
            f"{len(treatment_evidence):,}; stays with treatment evidence: "
            f"{treatment_evidence['stay_id'].nunique() if len(treatment_evidence) else 0:,}"
        )
        treatment_context = assemble_treatment_context(anchors, treatment_evidence, DELTA_H)
    frame = pd.concat(
        [
            anchors[["subject_id", "stay_id", "first_careunit", "last_careunit", "t", "t_plus"]],
            state_t.drop(columns=["stay_id", "t"]),
            state_f.drop(columns=["stay_id", "t_plus"]),
            treatment_context,
        ],
        axis=1,
    )
    has_pair = np.zeros(len(frame), dtype=bool)
    for var in STATE_VARS:
        has_pair |= frame[f"{var}_t"].notna().to_numpy() & frame[f"{var}_tp6"].notna().to_numpy()
    frame = frame.loc[has_pair].reset_index(drop=True)
    frame.to_parquet(args.output, index=False)
    counts = pair_counts(frame)
    report = {
        "artifact": "MIMIC-IV v3.1 whole-ICU observation transition cohort",
        "mimic_dir": str(args.mimic_dir),
        "output": str(args.output.resolve()),
        "delta_h": DELTA_H,
        "anchor_step_h": ANCHOR_STEP_H,
        "lookback_h": LOOKBACK_H,
        "target_tolerance_h": TARGET_TOL_H,
        "episode_max_h": EPISODE_MAX_H,
        "selected_stays": int(len(meta)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "rows": int(len(frame)),
        "careunits": int(frame["first_careunit"].nunique()),
        "pair_counts": counts,
        "eligible_pair_targets": int(sum(count >= 1 for count in counts.values())),
        "treatment_context": {
            "included": bool(args.include_treatment_context),
            "sources": treatment_evidence["source"].value_counts().astype(int).to_dict() if len(treatment_evidence) else {},
            "evidence_rows": int(len(treatment_evidence)),
            "evidence_stays": int(treatment_evidence["stay_id"].nunique()) if len(treatment_evidence) else 0,
            "action_keys": list(ACTION_KEYS) if args.include_treatment_context else [],
            "action_support": action_support_counts(frame) if args.include_treatment_context else {},
            "factual_observed_treatment_only": bool(args.include_treatment_context),
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
        },
        "patient_ids_included_in_report": False,
        "row_level_outputs_committed": False,
        "causal_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "rows": report["rows"],
        "subjects": report["subjects"],
        "eligible_pair_targets": report["eligible_pair_targets"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
