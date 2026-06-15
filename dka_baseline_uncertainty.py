"""
dka_baseline_uncertainty.py
===========================
Same persistence-vs-GBM delta-MAE as dka_baseline.py, but with the uncertainty
attached so you can SEE whether the sample can support a conclusion at all.

For each target x {ALL, action_present}:
  - point delta-MAE (persist - gbm; >0 means gbm beats persistence)
  - subject-level bootstrap 95% CI on that delta (resample SUBJECTS w/ replacement)
  - leave-one-subject-out range: drop each single subject, how far does delta move?

Read it like this:
  - CI crosses 0            -> sample cannot tell you the model beats persistence.
  - LOSO range is huge      -> the result is hostage to one or two patients.
Either of those = demo cannot answer "does the method work". That is the point.

Usage:  python dka_baseline_uncertainty.py dka_transitions_6h.parquet
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

TARGET_VARS = ["glucose", "ph", "anion_gap", "potassium", "sbp"]
TARGET_DRIVER = {
    "glucose": "act_insulin", "ph": "act_insulin", "anion_gap": "act_insulin",
    "potassium": "act_potassium", "sbp": "act_fluids",
}
N_SPLITS = 5
N_BOOT = 2000
RNG = np.random.default_rng(0)


def feature_cols(df):
    state = [c for c in df if c.endswith("_t") and not c.endswith("_age_hr")
             and c != "t" and not c.startswith("act_")]
    age = [c for c in df if c.endswith("_age_hr")]
    act = [c for c in df if c.startswith("act_")]
    return state + age + act


def cv_errors(df, base, feats):
    """Return per-row frame with subject, action flag, persist_err, gbm_err."""
    tgt, persist = f"{base}_tp6", f"{base}_t"          # <- corrected naming
    sub = df[df[tgt].notna() & df[persist].notna()].copy()
    if len(sub) < 20 or sub["subject_id"].nunique() < 3:
        return None
    X = sub[feats].to_numpy(float)
    y = sub[tgt].to_numpy(float)
    groups = sub["subject_id"].to_numpy()
    preds = np.full(len(sub), np.nan)
    gkf = GroupKFold(n_splits=min(N_SPLITS, sub["subject_id"].nunique()))
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          l2_regularization=1.0)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    out = pd.DataFrame({
        "subject_id": groups,
        "driver": sub[TARGET_DRIVER[base]].to_numpy(int),
        "persist_err": np.abs(y - sub[persist].to_numpy(float)),
        "gbm_err": np.abs(y - preds),
    })
    return out


def delta(frame):
    return frame["persist_err"].mean() - frame["gbm_err"].mean()


def boot_ci(frame):
    subs = frame["subject_id"].unique()
    by = {s: frame[frame["subject_id"] == s] for s in subs}
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.choice(subs, size=len(subs), replace=True)
        deltas[b] = delta(pd.concat([by[s] for s in pick], ignore_index=True))
    return np.percentile(deltas, [2.5, 50, 97.5])


def loso_range(frame):
    subs = frame["subject_id"].unique()
    ds = [delta(frame[frame["subject_id"] != s]) for s in subs]
    return min(ds), max(ds)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dka_transitions_6h.parquet"
    df = pd.read_parquet(path)
    feats = feature_cols(df)
    print(f"Loaded {len(df):,} transitions, {df['subject_id'].nunique()} subjects\n")
    print(f"{'target':<11}{'split':<16}{'n':>5}{'nsubj':>6}"
          f"{'delta':>9}{'95% CI':>20}{'LOSO range':>20}{'crosses0':>10}")
    for v in TARGET_VARS:
        err = cv_errors(df, v, feats)
        if err is None:
            print(f"{v:<11}{'(too few)':<16}")
            continue
        for label, mask in [("ALL", np.ones(len(err), bool)),
                            ("action_present", err["driver"] == 1)]:
            e = err[mask]
            if len(e) < 20 or e["subject_id"].nunique() < 3:
                print(f"{v:<11}{label:<16}{len(e):>5}{e['subject_id'].nunique():>6}"
                      f"{'(too few)':>9}")
                continue
            d = delta(e)
            lo, md, hi = boot_ci(e)
            l, h = loso_range(e)
            crosses = "YES" if lo <= 0 <= hi else "no"
            print(f"{v:<11}{label:<16}{len(e):>5}{e['subject_id'].nunique():>6}"
                  f"{d:>9.2f}{f'[{lo:.1f}, {hi:.1f}]':>20}"
                  f"{f'[{l:.1f}, {h:.1f}]':>20}{crosses:>10}")
    print("\n  crosses0=YES  -> sample can't show gbm beats persistence.")
    print("  wide LOSO     -> result depends on one/two patients.")


if __name__ == "__main__":
    main()
