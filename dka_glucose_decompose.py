"""
dka_glucose_decompose.py
========================
The only target with signal on demo is glucose. Before believing it means the
model "understands insulin", separate two explanations:

  (R) mean reversion: glucose is extreme by cohort definition and reverts; a model
      using only the glucose LEVEL beats persistence trivially.
  (A) action information: the insulin/fluids flags add predictive power BEYOND the
      glucose level and the rest of the state.

Nested models, evaluated on ACTION-PRESENT windows (where actions could matter),
subject-grouped CV, with subject-level bootstrap CI on the incremental gains:

  persist          : glucose_{t+6h} = glucose_t            (no model)
  glucose_only     : GBM on [glucose_t, glucose_age]       <- captures (R)
  state_only       : GBM on all state_t, NO actions
  state_plus_action: GBM on all state_t + action flags     <- adds (A)

Key numbers:
  gain_state_over_glucose   = MAE(glucose_only)  - MAE(state_only)
  gain_action_over_state    = MAE(state_only)    - MAE(state_plus_action)
If gain_action_over_state CI crosses 0 -> actions add nothing beyond state here,
i.e. the glucose win is reversion/level, not action understanding.

Usage:  python dka_glucose_decompose.py dka_transitions_6h.parquet
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

N_SPLITS = 5
N_BOOT = 2000
RNG = np.random.default_rng(0)


def all_state_feats(df):
    return [c for c in df if (c.endswith("_t") and not c.endswith("_age_hr") and c != "t"
                              and not c.startswith("act_"))
            or c.endswith("_age_hr")]


def cv_pred(sub, feats):
    X = sub[feats].to_numpy(float)
    y = sub["glucose_tp6"].to_numpy(float)
    g = sub["subject_id"].to_numpy()
    pred = np.full(len(sub), np.nan)
    gkf = GroupKFold(n_splits=min(N_SPLITS, sub["subject_id"].nunique()))
    for tr, te in gkf.split(X, y, g):
        keep = []
        for column in range(X.shape[1]):
            values = X[tr, column]
            finite = values[np.isfinite(values)]
            if len(finite) and len(np.unique(finite)) > 1:
                keep.append(column)
        if not keep:
            pred[te] = np.mean(y[tr])
            continue
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          l2_regularization=1.0)
        m.fit(X[tr][:, keep], y[tr])
        pred[te] = m.predict(X[te][:, keep])
    return pred


def boot_ci_diff(subject, err_a, err_b):
    """CI on mean(err_a) - mean(err_b), resampling subjects."""
    d = pd.DataFrame({"s": subject, "a": err_a, "b": err_b})
    subs = d["s"].unique()
    by = {s: d[d["s"] == s] for s in subs}
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(subs, size=len(subs), replace=True)
        c = pd.concat([by[s] for s in pick], ignore_index=True)
        vals[i] = c["a"].mean() - c["b"].mean()
    return np.percentile(vals, [2.5, 50, 97.5])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dka_transitions_6h.parquet"
    df = pd.read_parquet(path)
    sub = df[df["glucose_tp6"].notna() & df["glucose_t"].notna()].copy()
    sub = sub[sub["act_insulin"] == 1].copy()   # action-present windows
    print(f"action-present glucose windows: n={len(sub)}, "
          f"subjects={sub['subject_id'].nunique()}\n")
    if len(sub) < 20:
        print("too few to decompose."); return

    state = all_state_feats(sub)
    act = [c for c in sub if c.startswith("act_")]
    glu = [c for c in ["glucose_t", "glucose_age_hr"] if c in sub]

    y = sub["glucose_tp6"].to_numpy(float)
    err = {}
    err["persist"]      = np.abs(y - sub["glucose_t"].to_numpy(float))
    err["glucose_only"] = np.abs(y - cv_pred(sub, glu))
    err["state_only"]   = np.abs(y - cv_pred(sub, state))
    err["state+action"] = np.abs(y - cv_pred(sub, state + act))

    print(f"{'model':<16}{'MAE':>9}")
    for k in ["persist", "glucose_only", "state_only", "state+action"]:
        print(f"{k:<16}{err[k].mean():>9.2f}")

    s = sub["subject_id"].to_numpy()
    print("\nincremental gains (positive = the added info helps), 95% CI:")
    lo, md, hi = boot_ci_diff(s, err["persist"], err["glucose_only"])
    print(f"  glucose_only over persist     {md:6.2f}  [{lo:.2f}, {hi:.2f}]  "
          f"{'crosses0' if lo<=0<=hi else ''}")
    lo, md, hi = boot_ci_diff(s, err["glucose_only"], err["state_only"])
    print(f"  state over glucose-level      {md:6.2f}  [{lo:.2f}, {hi:.2f}]  "
          f"{'crosses0' if lo<=0<=hi else ''}")
    lo, md, hi = boot_ci_diff(s, err["state_only"], err["state+action"])
    print(f"  ACTION over state             {md:6.2f}  [{lo:.2f}, {hi:.2f}]  "
          f"{'crosses0' if lo<=0<=hi else ''}")
    print("\n  If 'ACTION over state' crosses 0: the glucose win is level/reversion,")
    print("  not action understanding -- even on the one target that has signal.")


if __name__ == "__main__":
    main()
