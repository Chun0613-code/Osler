"""
dka_baseline.py
===============
Baselines that any world model MUST beat before it is worth building.

For each forecast target X (X_{t+6h}):
  - PERSISTENCE: predict X_{t+6h} = X_t.  This is the honest floor.
  - GBM:         HistGradientBoostingRegressor on (state_t values + ages + action flags),
                 subject-level GroupKFold (no patient leakage), native NaN handling.

Reported separately for ACTION-PRESENT vs ACTION-ABSENT windows, because persistence is
trivially strong when nothing was done. The number that matters is delta-MAE over
persistence on the action-present windows -- that is the only place a dynamics model
can earn its keep.

Also: outcome AUROC (death-after-window, vaso-onset-after-window) via GBM classifier.

Usage:  python dka_baseline.py dka_transitions_6h.parquet
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

TARGET_VARS = ["glucose", "ph", "anion_gap", "potassium", "sbp"]
# which action's presence to split each target on
TARGET_DRIVER = {
    "glucose": "act_insulin", "ph": "act_insulin", "anion_gap": "act_insulin",
    "potassium": "act_potassium", "sbp": "act_fluids",
}
N_SPLITS = 5


def feature_cols(df):
    state = [c for c in df if (c.endswith("_t") and not c.endswith("_age_hr"))
             and c not in ("t",) and not c.startswith("act_")]
    age = [c for c in df if c.endswith("_age_hr")]
    act = [c for c in df if c.startswith("act_")]
    return state + age + act


def gbm_cv_mae(df, target, feats):
    y_col = f"{target}_tp6"
    x_col = f"{target}_t"

    sub = df[df[y_col].notna() & df[x_col].notna()].copy()
    X = sub[feats].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    groups = sub["subject_id"].to_numpy()
    preds = np.full(len(sub), np.nan)
    gkf = GroupKFold(n_splits=min(N_SPLITS, sub["subject_id"].nunique()))
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_depth=None, l2_regularization=1.0)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    sub["gbm_pred"] = preds
    sub["persist_pred"] = sub[x_col]
    return sub


def report_target(df, target):
    feats = feature_cols(df)
    sub = gbm_cv_mae(df, target, feats)
    sub = sub.rename(columns={f"{target}_tp6": "y"})
    driver = TARGET_DRIVER[target]
    rows = []
    for label, mask in [("ALL", np.ones(len(sub), bool)),
                        ("action_present", sub[driver] == 1),
                        ("action_absent", sub[driver] == 0)]:
        s = sub[mask]
        if len(s) < 20:
            rows.append((target, label, len(s), np.nan, np.nan, np.nan))
            continue
        pe = (s["y"] - s["persist_pred"]).abs().mean()
        ge = (s["y"] - s["gbm_pred"]).abs().mean()
        rows.append((target, label, len(s), round(pe, 3), round(ge, 3),
                     round(pe - ge, 3)))
    return rows


def report_outcome(df, target):
    feats = feature_cols(df)
    sub = df[df[target].notna()].copy()
    if sub[target].nunique() < 2 or sub[target].sum() < 10:
        return (target, sub[target].sum(), np.nan)
    X = sub[feats].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=int)
    groups = sub["subject_id"].to_numpy()
    preds = np.full(len(sub), np.nan)
    gkf = GroupKFold(n_splits=min(N_SPLITS, sub["subject_id"].nunique()))
    for tr, te in gkf.split(X, y, groups):
        if y[tr].sum() == 0:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           l2_regularization=1.0)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(preds)
    auc = roc_auc_score(y[ok], preds[ok]) if y[ok].sum() > 0 else np.nan
    return (target, int(y.sum()), round(auc, 3))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dka_transitions_6h.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} transitions, {df['subject_id'].nunique():,} subjects\n")

    print("=== FUTURE-LAB FORECAST: MAE (lower=better), delta = persist - gbm ===")
    print(f"{'target':<11}{'split':<16}{'n':>7}{'persist':>9}{'gbm':>8}{'delta':>8}")
    for v in TARGET_VARS:
        for r in report_target(df, v):
            t, lbl, n, pe, ge, d = r
            print(f"{t:<11}{lbl:<16}{n:>7}{str(pe):>9}{str(ge):>8}{str(d):>8}")
    print("\n  -> If delta<=0 on action_present, the model is just doing persistence.")

    print("\n=== OUTCOME AUROC (subject-grouped CV) ===")
    print(f"{'outcome':<28}{'n_pos':>7}{'auroc':>8}")
    for o in ["died_after_window", "vaso_onset_after_window"]:
        t, npos, auc = report_outcome(df, o)
        print(f"{t:<28}{npos:>7}{str(auc):>8}")


if __name__ == "__main__":
    main()
