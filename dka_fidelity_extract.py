"""
dka_fidelity_extract.py
=======================
Extract per-stay continuous DKA trajectories from MIMIC for the fidelity exam.
Unlike the 6h-transition extractor, this keeps the FULL time course of a stay:
  - initial observed state at DKA onset
  - the real five-action time-series as per-hour rates on a 30-minute grid
  - six hours of pre-onset treatment history
  - the observed 13-state trajectory, including sodium/osmolality, creatinine,
    urine output, and BHB when measured

Output: trajectories.jsonl (one JSON object per stay), consumed by
dka_fidelity_replay.py.

Action rates are normalized from both inputevents and eMAR/eMAR-detail amounts and
rates. Each event total is spread across the grid cells its administration interval
covers. Dextrose-containing fluids contribute both carrier volume and dextrose grams.

Run:  MIMIC_DIR=/path python dka_fidelity_extract.py
"""

import os
import json
import duckdb
import numpy as np
import pandas as pd

from dka_transition_extract import (
    find_dka_onset as find_dka_onset_v2,
    pull_actions as pull_actions_v2,
    pull_measurements as pull_measurements_v2,
    pull_stay_meta as pull_stay_meta_v2,
)
from mimic_action_history import (
    ACTION_NAMES,
    action_rate_grid,
    treatment_event_grid,
    treatment_event_records,
)

MIMIC_DIR = os.environ.get("MIMIC_DIR", "/path/to/mimic-iv")
HOSP, ICU = os.path.join(MIMIC_DIR, "hosp"), os.path.join(MIMIC_DIR, "icu")
OUT_PATH = os.environ.get("OUT_PATH", "trajectories.jsonl")

DT = 0.5            # must match the replay grid
WINDOW_H = 24.0
INIT_LOOKBACK_H = 12.0
HISTORY_H = 6.0

DKA_GLUCOSE_MIN, DKA_HCO3_MAX, DKA_AG_MIN, DKA_COOCCUR_H = 250.0, 18.0, 12.0, 12.0

# verify against your d_labitems / d_items (you discovered most of these already)
LAB_ITEMIDS = {
    "glucose": [50931, 50809], "ph": [50820],
    "bicarbonate": [50882, 50803], "anion_gap": [50868], "potassium": [50971, 50822],
}
# inputevents (icu). amount-based.
ACTION_ITEMIDS = {
    "insulin": [223257, 223258, 223259, 223260, 223261, 223262],
    "fluids":  [225158, 225828, 225823, 220949, 225159],
    "kcl":     [225166],
}
# target units (amount converted to these); warn on anything unexpected
AMOUNT_TARGET = {"insulin": "unit", "fluids": "ml", "kcl": "meq"}
LAB2SCHEMA = {"glucose": "glucose", "ph": "pH", "bicarbonate": "HCO3", "potassium": "K"}
LAB2SCHEMA.update({
    "anion_gap": "anion_gap", "sodium": "Na", "osmolality": "osmolality",
    "creatinine": "creatinine", "urine_output": "urine_output", "BHB": "BHB",
})


def con():
    c = duckdb.connect(); c.execute("PRAGMA threads=4;"); return c


def f(mod, name):
    return f"read_csv_auto('{os.path.join(mod, name)}.csv.gz')"


def pull_labs(c):
    case = " ".join(f"WHEN le.itemid IN ({','.join(map(str,v))}) THEN '{k}'"
                    for k, v in LAB_ITEMIDS.items())
    ids = sorted({i for v in LAB_ITEMIDS.values() for i in v})
    df = c.execute(f"""
        SELECT ie.stay_id, le.charttime, CASE {case} END AS var, le.valuenum
        FROM {f(HOSP,'labevents')} le
        JOIN {f(ICU,'icustays')} ie
          ON le.hadm_id = ie.hadm_id AND le.charttime BETWEEN ie.intime AND ie.outtime
        WHERE le.itemid IN ({','.join(map(str,ids))}) AND le.valuenum IS NOT NULL
    """).df()
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df.sort_values(["stay_id", "charttime"])


def pull_actions(c):
    case = " ".join(f"WHEN itemid IN ({','.join(map(str,v))}) THEN '{k}'"
                    for k, v in ACTION_ITEMIDS.items())
    ids = sorted({i for v in ACTION_ITEMIDS.values() for i in v})
    df = c.execute(f"""
        SELECT stay_id, starttime, endtime, CASE {case} END AS action,
               amount, lower(amountuom) AS uom
        FROM {f(ICU,'inputevents')}
        WHERE itemid IN ({','.join(map(str,ids))}) AND amount IS NOT NULL
    """).df()
    for col in ("starttime", "endtime"):
        df[col] = pd.to_datetime(df[col])
    return df.dropna(subset=["action"])


def pull_meta(c):
    df = c.execute(f"SELECT stay_id, intime, outtime FROM {f(ICU,'icustays')}").df()
    for col in ("intime", "outtime"):
        df[col] = pd.to_datetime(df[col])
    return df.set_index("stay_id")


def find_dka_onset(labs):
    g = labs[(labs["var"] == "glucose") & (labs["valuenum"] > DKA_GLUCOSE_MIN)]
    h = labs[(labs["var"] == "bicarbonate") & (labs["valuenum"] < DKA_HCO3_MAX)]
    a = labs[(labs["var"] == "anion_gap") & (labs["valuenum"] > DKA_AG_MIN)]
    tol = pd.Timedelta(hours=DKA_COOCCUR_H)
    out = []
    for sid, gg in g.groupby("stay_id"):
        hv = h[h["stay_id"] == sid]["charttime"].values
        av = a[a["stay_id"] == sid]["charttime"].values
        if len(hv) == 0 or len(av) == 0:
            continue
        for t in gg["charttime"]:
            tv = np.datetime64(t)
            if (np.abs(hv - tv) <= tol).any() and (np.abs(av - tv) <= tol).any():
                out.append((sid, t)); break
    return dict(out)


def convert_amount(action, amt, uom, warn):
    tgt = AMOUNT_TARGET[action]
    if uom is None:
        return amt
    if tgt == "ml" and uom in ("l",):
        return amt * 1000.0
    if uom in (tgt, tgt + "s", "units" if tgt == "unit" else tgt):
        return amt
    # unrecognized unit: pass through but record it
    warn.add(f"{action}:{uom}")
    return amt


def build_action_grid(events, onset, n_cells, warn):
    """events: rows for ONE action in the window. Return per-cell rate array."""
    rate = np.zeros(n_cells)
    for _, e in events.iterrows():
        s = (e["starttime"] - onset).total_seconds() / 3600.0
        en = (e["endtime"] - onset).total_seconds() / 3600.0
        amt = convert_amount(e["action"], float(e["amount"]), e["uom"], warn)
        if en <= s:  # instantaneous bolus
            k = int(np.clip(s // DT, 0, n_cells - 1))
            rate[k] += amt / DT
            continue
        dur = en - s
        for k in range(n_cells):
            lo, hi = k * DT, (k + 1) * DT
            ov = max(0.0, min(en, hi) - max(s, lo))
            if ov > 0:
                rate[k] += (amt * ov / dur) / DT
    return rate


def main():
    c = con()
    print("Pulling labs / actions / meta...")
    labs = pull_measurements_v2(c).rename(columns={"valuenum": "valuenum"})
    actions, _ = pull_actions_v2(c)
    meta_frame = pull_stay_meta_v2(c)
    meta = meta_frame.set_index("stay_id")
    onset_frame = find_dka_onset_v2(labs)
    onsets = dict(onset_frame[["stay_id", "onset"]].itertuples(index=False, name=None))
    print(f"DKA stays: {len(onsets)}")
    if not onsets:
        print("none found -- check itemids/thresholds"); return

    warn = set()
    out = []
    for sid, onset in onsets.items():
        if sid not in meta.index:
            continue
        end = min(onset + pd.Timedelta(hours=WINDOW_H), meta.loc[sid, "outtime"])
        win_h = (end - onset).total_seconds() / 3600.0
        if win_h < 2:
            continue
        n_cells = int(np.ceil(win_h / DT))

        # initial state: last value at/before onset within lookback
        lo = onset - pd.Timedelta(hours=INIT_LOOKBACK_H)
        init = {}
        slab = labs[labs["stay_id"] == sid]
        for var, key in [
            ("glucose", "glucose"), ("bicarbonate", "HCO3"),
            ("potassium", "K"), ("anion_gap", "anion_gap"),
            ("sodium", "Na"), ("osmolality", "osmolality"),
            ("creatinine", "creatinine"), ("urine_output", "urine_output"),
            ("BHB", "BHB"), ("map", "MAP"),
        ]:
            cand = slab[(slab["var"] == var) & (slab["charttime"] <= onset)
                        & (slab["charttime"] >= lo)]
            init[key] = float(cand.iloc[-1]["valuenum"]) if len(cand) else None
        if init["glucose"] is None or init["HCO3"] is None:
            continue

        # Action grids include six hours before onset so downstream models can
        # distinguish residual treatment effects from untreated presentation.
        sact = actions[(actions["stay_id"] == sid)
                       & (actions["endtime"] >= onset - pd.Timedelta(hours=HISTORY_H))
                       & (actions["starttime"] <= end)]
        future = sact[(sact["endtime"] >= onset) & (sact["starttime"] <= end)]
        history = sact[(sact["endtime"] >= onset - pd.Timedelta(hours=HISTORY_H))
                       & (sact["starttime"] < onset)]
        future_grid = action_rate_grid(future, onset, n_cells * DT, DT)
        future_event_grid = treatment_event_grid(
            future, onset, n_cells * DT, DT
        )
        history_grid = action_rate_grid(
            history, onset - pd.Timedelta(hours=HISTORY_H), HISTORY_H, DT
        )
        history_event_grid = treatment_event_grid(
            history, onset - pd.Timedelta(hours=HISTORY_H), HISTORY_H, DT
        )
        cover = {}
        for a in ACTION_NAMES:
            ev = sact[sact["action"] == a]
            cover[a] = len(ev)
        actions_list = []
        for k in range(n_cells):
            row = {"t": round(k * DT, 3)}
            any_nz = False
            for action_index, a in enumerate(ACTION_NAMES):
                v = round(float(future_grid[k, action_index]), 3)
                row[a] = v
                any_nz = any_nz or v > 0
            if any_nz:
                actions_list.append(row)
        history_list = []
        for k in range(len(history_grid)):
            row = {"t": round(-HISTORY_H + k * DT, 3)}
            for action_index, action in enumerate(ACTION_NAMES):
                row[action] = round(float(history_grid[k, action_index]), 3)
            history_list.append(row)

        # observed lab trajectory in window
        labs_list = []
        for _, r in slab[(slab["charttime"] >= onset) & (slab["charttime"] <= end)].iterrows():
            if r["var"] in LAB2SCHEMA:
                labs_list.append({
                    "t": round((r["charttime"] - onset).total_seconds() / 3600.0, 3),
                    "var": LAB2SCHEMA[r["var"]], "value": float(r["valuenum"])})

        out.append({
            "stay_id": int(sid), "init": init,
            "history_actions": history_list, "actions": actions_list,
            "history_treatment_event_grid": history_event_grid.tolist(),
            "treatment_event_grid": future_event_grid.tolist(),
            "history_treatment_events": treatment_event_records(
                history, onset - pd.Timedelta(hours=HISTORY_H), HISTORY_H
            ),
            "treatment_events": treatment_event_records(
                future, onset, n_cells * DT
            ),
            "labs": labs_list, "_cover": cover,
            "_provenance": {
                "action_sources": sorted(sact["source"].dropna().unique().tolist()),
                "history_hours": HISTORY_H,
                "grid_hours": DT,
            },
        })

    with open(OUT_PATH, "w") as fh:
        for tr in out:
            fh.write(json.dumps(tr) + "\n")
    print(f"\nWrote {len(out)} trajectories -> {OUT_PATH}")
    print("action coverage (events per stay, mean):")
    for a in ACTION_NAMES:
        m = np.mean([tr["_cover"][a] for tr in out]) if out else 0
        nz = sum(tr["_cover"][a] > 0 for tr in out)
        print(f"  {a:<9} mean={m:4.1f} events   stays-with-any={nz}/{len(out)}")
    print("\nActions combine inputevents infusions and eMAR administrations when available.")


if __name__ == "__main__":
    main()
