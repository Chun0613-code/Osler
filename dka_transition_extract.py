"""
dka_transition_extract.py
==========================
Build the (state_t, action_t, state_{t+6h}, outcome_after_t) transition table for a
lab-defined DKA mini-body cohort from MIMIC-IV.

Current scope:
  - Delta = 6h, with 30-minute action grids
  - Cohort = lab-based DKA only (no ICD)
  - State includes sodium/osmolality, creatinine, urine output, and BHB
  - Actions combine inputevents and eMAR administration records for insulin,
    fluids, KCl, bicarbonate, and dextrose
  - Every anchor retains the preceding six hours of action exposure and recency

This script does NOT run anywhere but your machine -- it needs your MIMIC-IV files.
Run order:
  1. python dka_transition_extract.py --discover        # verify itemids first!
  2. python dka_transition_extract.py                   # build the transition table

Data access assumption: DuckDB reading the MIMIC-IV csv.gz files directly.
If you are on PostgreSQL, the SQL in pull_measurements()/pull_actions() is standard
and ports over by swapping the read_csv_auto(...) source for the table name.
"""

import argparse
import json
import os
import duckdb
import numpy as np
import pandas as pd

from mimic_action_history import (
    ACTION_NAMES, action_window_summary, deduplicate_events, normalize_events,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MIMIC_DIR = os.environ.get("MIMIC_DIR", "/path/to/mimic-iv")  # <-- set this
HOSP = os.path.join(MIMIC_DIR, "hosp")
ICU = os.path.join(MIMIC_DIR, "icu")
OUT_PATH = os.environ.get("OUT_PATH", "dka_transitions_6h.parquet")

DELTA_H = 6.0           # prediction horizon (hours)
ANCHOR_STEP_H = 4.0     # spacing between anchor times within a DKA episode
LOOKBACK_H = 6.0        # state_t = most recent measurement within this lookback
TARGET_TOL_H = 2.0      # state_{t+6h} = nearest measurement within t+6h +/- this
EPISODE_MAX_H = 72.0    # build anchors from DKA onset up to onset + this

# Lab-based DKA criteria (co-occurring within DKA_COOCCUR_H)
DKA_GLUCOSE_MIN = 250.0
DKA_HCO3_MAX = 18.0
DKA_ANIONGAP_MIN = 12.0
DKA_COOCCUR_H = 12.0

# --- itemids: VERIFY against YOUR d_labitems / d_items (run --discover first) ---
# labevents (hosp). Lists allowed: serum + blood-gas variants where relevant.
LAB_ITEMIDS = {
    "glucose":    [50931, 50809],
    "ph":         [50820],
    "bicarbonate":[50882, 50803],
    "anion_gap":  [50868],
    "potassium":  [50971, 50822],
    "lactate":    [50813],
    "creatinine": [50912],
    "sodium":     [50983, 50824],
    "osmolality": [50964],
    "BHB":        [],
}
# chartevents (icu). Vitals.
VITAL_ITEMIDS = {
    "heart_rate": [220045],
    "sbp":        [220179, 220050],   # NBP sys, ABP sys
    "dbp":        [220180, 220051],
    "map":        [220181, 220052],
}
# inputevents (icu). Actions -- THESE ARE THE LEAST RELIABLE, VERIFY CAREFULLY.
ACTION_ITEMIDS = {
    "insulin":     [223257, 223258, 223259, 223260, 223261, 223262],
    "fluids":      [225158, 225828, 225823, 220949, 225159],  # NaCl/LR/D5/etc
    "potassium":   [225166],   # KCl
    "bicarbonate": [227533],   # Na bicarb 8.4%  (uncertain -- verify)
}
# vasopressors for the deterioration/viability outcome
VASO_ITEMIDS = [221906, 221289, 221662, 221653, 229617]  # norepi/epi/dopa/dobut/phenyl (verify)

PLAUSIBLE = {  # crude physiologic bounds; values outside -> set NaN
    "glucose": (10, 2000), "ph": (6.5, 7.9), "bicarbonate": (1, 60),
    "anion_gap": (0, 60), "potassium": (1.0, 9.5), "lactate": (0, 40),
    "creatinine": (0.1, 25), "heart_rate": (10, 300),
    "sodium": (100, 180), "osmolality": (200, 450), "BHB": (0, 20),
    "urine_output": (0, 1000),
    "sbp": (30, 300), "dbp": (10, 200), "map": (20, 250),
}
STATE_VARS = list(LAB_ITEMIDS) + list(VITAL_ITEMIDS) + ["urine_output"]
TARGET_VARS = [
    "glucose", "ph", "bicarbonate", "anion_gap", "potassium", "map",
    "sodium", "osmolality", "creatinine", "urine_output", "BHB",
]


# ----------------------------------------------------------------------------
def con():
    c = duckdb.connect()
    c.execute("PRAGMA threads=4;")
    return c


def f(module, name):
    return f"read_csv_auto('{os.path.join(module, name)}.csv.gz')"


def discover(c):
    """Print itemid candidates so you can confirm the maps above."""
    print("=== d_labitems (search labs) ===")
    for kw in [
        "glucose", "ph", "bicarbonate", "anion", "potassium", "lactate",
        "creatinine", "sodium", "osmolality", "hydroxybutyrate", "ketone",
    ]:
        df = c.execute(
            f"SELECT itemid, label, fluid, category FROM {f(HOSP,'d_labitems')} "
            f"WHERE lower(label) LIKE '%{kw}%'"
        ).df()
        print(f"\n[{kw}]\n", df.to_string(index=False))
    print("\n=== d_items (search vitals + actions) ===")
    for kw in ["heart rate", "blood pressure", "non invasive", "arterial blood pressure",
               "insulin", "sodium chloride", "lactated", "dextrose", "potassium chloride",
               "sodium bicarbonate", "norepinephrine", "epinephrine", "vasopressin",
               "phenylephrine", "dopamine", "dobutamine"]:
        df = c.execute(
            f"SELECT itemid, label, linksto, category FROM {f(ICU,'d_items')} "
            f"WHERE lower(label) LIKE '%{kw}%'"
        ).df()
        print(f"\n[{kw}]\n", df.to_string(index=False))


def resolved_lab_itemids(c):
    mapping = {name: list(ids) for name, ids in LAB_ITEMIDS.items()}
    try:
        dynamic = c.execute(f"""
            SELECT itemid, lower(label) AS label
            FROM {f(HOSP,'d_labitems')}
            WHERE lower(label) LIKE '%hydroxybutyrate%'
               OR lower(label) LIKE '%beta-hydroxy%'
               OR lower(label) LIKE '%ketone bod%'
        """).df()
        mapping["BHB"].extend(dynamic["itemid"].astype(int).tolist())
    except Exception:
        pass
    return {name: sorted(set(ids)) for name, ids in mapping.items()}


def pull_measurements(c):
    """Long table: subject_id, stay_id, charttime, var, valuenum (labs + vitals)."""
    lab_itemids = resolved_lab_itemids(c)
    lab_ids = sorted({i for v in lab_itemids.values() for i in v})
    vit_ids = sorted({i for v in VITAL_ITEMIDS.values() for i in v})
    lab_case = " ".join(
        f"WHEN le.itemid IN ({','.join(map(str,v))}) THEN '{k}'"
        for k, v in lab_itemids.items() if v
    )
    vit_case = " ".join(
        f"WHEN ce.itemid IN ({','.join(map(str,v))}) THEN '{k}'" for k, v in VITAL_ITEMIDS.items()
    )
    labs = c.execute(f"""
        SELECT ie.subject_id, ie.stay_id, le.charttime,
               CASE {lab_case} END AS var, le.valuenum
        FROM {f(HOSP,'labevents')} le
        JOIN {f(ICU,'icustays')} ie
          ON le.hadm_id = ie.hadm_id
         AND le.charttime BETWEEN ie.intime AND ie.outtime
        WHERE le.itemid IN ({','.join(map(str,lab_ids))})
          AND le.valuenum IS NOT NULL
    """).df()
    vits = c.execute(f"""
        SELECT ce.subject_id, ce.stay_id, ce.charttime,
               CASE {vit_case} END AS var, ce.valuenum
        FROM {f(ICU,'chartevents')} ce
        WHERE ce.itemid IN ({','.join(map(str,vit_ids))})
          AND ce.valuenum IS NOT NULL
    """).df()
    try:
        urine = c.execute(f"""
            SELECT oe.subject_id, oe.stay_id, oe.charttime, SUM(oe.value) AS value
            FROM {f(ICU,'outputevents')} oe
            JOIN {f(ICU,'d_items')} di ON oe.itemid = di.itemid
            WHERE lower(di.linksto) = 'outputevents'
              AND (
                    lower(di.label) IN ('foley', 'void', 'condom cath', 'straight cath')
                 OR lower(di.label) LIKE '%urine%'
              )
              AND lower(di.label) NOT LIKE '%irrigant%'
              AND lower(coalesce(oe.valueuom, 'ml')) LIKE '%ml%'
              AND oe.value >= 0
            GROUP BY 1,2,3
        """).df()
        urine["charttime"] = pd.to_datetime(urine["charttime"])
        rolling = []
        for (_, stay_id), group in urine.groupby(["subject_id", "stay_id"]):
            group = group.sort_values("charttime").set_index("charttime")
            hourly = group["value"].rolling("4h", closed="both").sum() / 4.0
            part = hourly.rename("valuenum").reset_index()
            part["subject_id"] = group["subject_id"].iloc[0] if "subject_id" in group else _
            part["stay_id"] = stay_id
            part["var"] = "urine_output"
            rolling.append(part[["subject_id", "stay_id", "charttime", "var", "valuenum"]])
        urine = pd.concat(rolling, ignore_index=True) if rolling else pd.DataFrame(columns=labs.columns)
    except Exception:
        urine = pd.DataFrame(columns=labs.columns)
    m = pd.concat([labs, vits, urine], ignore_index=True)
    m["charttime"] = pd.to_datetime(m["charttime"])
    # plausibility clip -> NaN, then drop
    for var, (lo, hi) in PLAUSIBLE.items():
        bad = (m["var"] == var) & ((m["valuenum"] < lo) | (m["valuenum"] > hi))
        m = m[~bad]
    return m.dropna(subset=["valuenum"]).sort_values(["stay_id", "charttime"])


def pull_actions(c):
    """Dose/time-resolved administrations from inputevents plus eMAR."""
    label_filter = " OR ".join(
        f"lower(di.label) LIKE '%{term}%'" for term in (
            "insulin", "saline", "sodium chloride", "nacl", "lactated ringer",
            "plasmalyte", "plasma-lyte", "potassium chloride", "bicarbonate",
            "kcl", "dextrose", "d5", "d10", "d20", "d50",
        )
    )
    known_action_ids = sorted({item for ids in ACTION_ITEMIDS.values() for item in ids})
    raw_input = c.execute(f"""
        SELECT ie.subject_id, ie.stay_id, ie.starttime, ie.endtime,
               di.label, ie.amount, lower(ie.amountuom) AS uom,
               ie.rate, lower(ie.rateuom) AS rate_uom,
               NULL AS route, NULL AS product_description, 'inputevents' AS source
        FROM {f(ICU,'inputevents')} ie
        JOIN {f(ICU,'d_items')} di ON ie.itemid = di.itemid
        WHERE ie.itemid IN ({','.join(map(str, known_action_ids))}) OR {label_filter}
    """).df()

    try:
        med_filter = " OR ".join(
            f"lower(e.medication) LIKE '%{term}%'" for term in (
                "insulin", "saline", "sodium chloride", "lactated ringer",
                "potassium chloride", "kcl", "bicarbonate", "dextrose",
                "glucose", "d5", "d10", "d50",
            )
        )
        raw_emar = c.execute(f"""
            SELECT e.subject_id, i.stay_id, e.charttime AS starttime,
                   e.charttime AS endtime, e.medication AS label,
                   d.dose_given AS amount, lower(d.dose_given_unit) AS uom,
                   d.infusion_rate AS rate,
                   lower(d.infusion_rate_unit) AS rate_uom,
                   d.route, d.product_description, 'emar' AS source
            FROM {f(HOSP,'emar')} e
            JOIN {f(ICU,'icustays')} i
              ON e.hadm_id = i.hadm_id
             AND e.charttime BETWEEN i.intime AND i.outtime
            LEFT JOIN {f(HOSP,'emar_detail')} d
              ON e.subject_id = d.subject_id
             AND e.emar_id = d.emar_id
             AND e.emar_seq = d.emar_seq
            WHERE ({med_filter})
              AND lower(coalesce(e.event_txt, 'administered')) NOT LIKE '%not given%'
        """).df()
    except Exception as error:
        print(f"WARNING: eMAR unavailable, using inputevents only: {str(error)[:160]}")
        raw_emar = pd.DataFrame(columns=raw_input.columns)

    actions = deduplicate_events(
        normalize_events(pd.concat([raw_input, raw_emar], ignore_index=True))
    )

    vaso_ids = ",".join(map(str, VASO_ITEMIDS))
    vaso = c.execute(f"""
        SELECT subject_id, stay_id, starttime, endtime, 'vasopressor' AS action,
               amount, lower(amountuom) AS uom, 'inputevents' AS source
        FROM {f(ICU,'inputevents')}
        WHERE itemid IN ({vaso_ids})
    """).df()
    for column in ("starttime", "endtime"):
        vaso[column] = pd.to_datetime(vaso[column])
    return actions.sort_values(["stay_id", "starttime"]), vaso


def pull_stay_meta(c):
    df = c.execute(f"""
        SELECT i.subject_id, i.stay_id, i.intime, i.outtime, a.deathtime
        FROM {f(ICU,'icustays')} i
        JOIN {f(HOSP,'admissions')} a ON i.hadm_id = a.hadm_id
    """).df()
    for col in ["intime", "outtime", "deathtime"]:
        df[col] = pd.to_datetime(df[col])
    return df


def find_dka_onset(meas):
    """Return DataFrame[stay_id, onset] for stays meeting co-occurring lab DKA criteria."""
    g = meas[(meas["var"] == "glucose") & (meas["valuenum"] > DKA_GLUCOSE_MIN)][["stay_id", "charttime"]]
    h = meas[(meas["var"] == "bicarbonate") & (meas["valuenum"] < DKA_HCO3_MAX)][["stay_id", "charttime"]]
    a = meas[(meas["var"] == "anion_gap") & (meas["valuenum"] > DKA_ANIONGAP_MIN)][["stay_id", "charttime"]]
    tol = pd.Timedelta(hours=DKA_COOCCUR_H)
    onsets = []
    for sid, gg in g.groupby("stay_id"):
        hh = h[h["stay_id"] == sid]["charttime"].values
        aa = a[a["stay_id"] == sid]["charttime"].values
        if len(hh) == 0 or len(aa) == 0:
            continue
        for t in gg["charttime"]:
            tv = np.datetime64(t)
            if (np.abs(hh - tv) <= tol).any() and (np.abs(aa - tv) <= tol).any():
                onsets.append((sid, t))
                break
    return pd.DataFrame(onsets, columns=["stay_id", "onset"])


def build_anchors(onsets, meta):
    rows = []
    meta_i = meta.set_index("stay_id")
    for sid, onset in onsets.itertuples(index=False):
        outtime = meta_i.loc[sid, "outtime"]
        last = min(onset + pd.Timedelta(hours=EPISODE_MAX_H), outtime - pd.Timedelta(hours=DELTA_H))
        t = onset
        while t <= last:
            rows.append((sid, t))
            t += pd.Timedelta(hours=ANCHOR_STEP_H)
    return pd.DataFrame(rows, columns=["stay_id", "t"])


def assemble_state(anchors, meas, suffix, when_col, direction, tol_h):
    """merge_asof each var onto anchors; return value + age (hours) columns."""
    out = anchors[["stay_id", when_col]].copy()
    tol = pd.Timedelta(hours=tol_h)
    for var in STATE_VARS:
        mv = meas[meas["var"] == var][["stay_id", "charttime", "valuenum"]].sort_values("charttime")
        if mv.empty:
            out[f"{var}{suffix}"] = np.nan
            if suffix == "_t":
                out[f"{var}_age_hr"] = np.nan
            continue
        left = anchors[["stay_id", when_col]].sort_values(when_col)
        merged = pd.merge_asof(
            left, mv, left_on=when_col, right_on="charttime",
            by="stay_id", direction=direction, tolerance=tol,
        )
        merged = merged.set_index(left.index)
        out[f"{var}{suffix}"] = merged["valuenum"].values
        if suffix == "_t":
            age = (merged[when_col] - merged["charttime"]).dt.total_seconds() / 3600.0
            out[f"{var}_age_hr"] = age.values
    return out


def assemble_actions(anchors, actions):
    """Exact future action grids and six-hour pre-anchor treatment history."""
    rows = []
    grouped = {stay: frame for stay, frame in actions.groupby("stay_id")}
    empty = actions.iloc[0:0]
    for anchor in anchors.itertuples():
        summary = action_window_summary(grouped.get(anchor.stay_id, empty), anchor.t)
        row = {
            key: json.dumps(value) if key.endswith("_grid") else value
            for key, value in summary.items()
        }
        for action in ACTION_NAMES:
            row[f"act_{action}"] = int(row[f"act_{action}_total"] > 0)
            row[f"act_{action}_rate_mean"] = row[f"act_{action}_total"] / DELTA_H
        row["act_insulin"] = int(row["act_insulin_total"] > 0)
        row["act_insulin_rate_mean"] = row["act_insulin_total"] / DELTA_H
        rows.append(row)
    return pd.DataFrame(rows, index=anchors.index)


def assemble_outcomes(anchors, vasopressors, meta):
    """No-leakage: events strictly AFTER t+DELTA_H."""
    meta_i = meta.set_index("stay_id")
    cut = anchors["t"] + pd.Timedelta(hours=DELTA_H)
    death = meta_i.reindex(anchors["stay_id"].values)["deathtime"].values
    death = pd.to_datetime(death)
    died_after = (death > cut.values)
    hrs_to_death = (death - cut.values) / np.timedelta64(1, "h")
    out = pd.DataFrame(index=anchors.index)
    out["died_after_window"] = np.where(np.isnat(death), 0, died_after.astype(int))
    out["hrs_to_death_from_cut"] = np.where(np.isnat(death), np.nan, hrs_to_death)
    # vasopressor onset after t+DELTA_H
    vaso = vasopressors[["stay_id", "starttime"]]
    merged = anchors.reset_index().merge(vaso, on="stay_id", how="left")
    after = merged["starttime"] > (merged["t"] + pd.Timedelta(hours=DELTA_H))
    hit = merged.loc[after, "index"].unique()
    flag = np.zeros(len(anchors), dtype=int)
    flag[anchors.index.get_indexer(hit)] = 1
    out["vaso_onset_after_window"] = flag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    args = ap.parse_args()
    c = con()

    if args.discover:
        discover(c)
        return

    print("Pulling measurements...")
    meas = pull_measurements(c)
    print(f"  {len(meas):,} measurement rows, {meas['stay_id'].nunique():,} stays")

    print("Finding DKA onsets...")
    onsets = find_dka_onset(meas)
    print(f"  DKA stays: {len(onsets):,}")
    if len(onsets) == 0:
        print("  No DKA stays found -- check thresholds and itemids (run --discover).")
        return

    meta = pull_stay_meta(c)
    actions, vasopressors = pull_actions(c)

    print("Building anchors...")
    anchors = build_anchors(onsets, meta).reset_index(drop=True)
    anchors["onset"] = anchors["stay_id"].map(onsets.set_index("stay_id")["onset"])
    anchors["hours_since_onset"] = (
        anchors["t"] - anchors["onset"]
    ).dt.total_seconds() / 3600.0
    anchors["subject_id"] = anchors["stay_id"].map(
        meta.set_index("stay_id")["subject_id"])
    anchors["t_plus"] = anchors["t"] + pd.Timedelta(hours=DELTA_H)
    print(f"  anchors (transitions): {len(anchors):,}")

    print("Assembling state_t / state_t+6h / actions / outcomes...")
    st = assemble_state(anchors, meas, "_t", "t", "backward", LOOKBACK_H)
    sf = assemble_state(anchors.rename(columns={"t_plus": "t_plus"}), meas,
                        "_tp6", "t_plus", "nearest", TARGET_TOL_H)
    ac = assemble_actions(anchors, actions)
    oc = assemble_outcomes(anchors, vasopressors, meta)

    df = pd.concat([
        anchors[["subject_id", "stay_id", "onset", "hours_since_onset", "t", "t_plus"]],
        st.drop(columns=["stay_id", "t"]),
        sf.drop(columns=["stay_id", "t_plus"]),
        ac, oc,
    ], axis=1)

    if "sodium_t" in df and "glucose_t" in df:
        df["osmolality_derived_t"] = 2.0 * df["sodium_t"] + df["glucose_t"] / 18.0
    if "sodium_tp6" in df and "glucose_tp6" in df:
        df["osmolality_derived_tp6"] = 2.0 * df["sodium_tp6"] + df["glucose_tp6"] / 18.0
    df["dka_active_t"] = (
        (df["glucose_t"] >= 200.0)
        & ((df["bicarbonate_t"] < 18.0) | (df["anion_gap_t"] > 12.0))
    ) | (df["BHB_t"] >= 3.0)

    # require at least the forecast targets present at BOTH ends for >=1 target
    has_pair = np.zeros(len(df), dtype=bool)
    for v in TARGET_VARS:
        has_pair |= df[f"{v}_t"].notna().values & df[f"{v}_tp6"].notna().values
    df = df[has_pair].reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} transitions -> {OUT_PATH}")
    print("Action prevalence:\n", df[[c for c in df if c.startswith('act_')]].mean())
    print("Death-after-window rate:", round(df["died_after_window"].mean(), 4))
    print("Vaso-onset-after-window rate:", round(df["vaso_onset_after_window"].mean(), 4))


if __name__ == "__main__":
    main()
