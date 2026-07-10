"""
rd_insulin_glucose.py

Regression-discontinuity probe of the insulin -> glucose control loop.

ICU insulin sliding scales dose insulin off a glucose THRESHOLD. Just above vs
just below the threshold, patients are near-identical but one is (more likely)
insulin-treated -- a quasi-random perturbation. This directly attacks the
time-varying confounding (dose driven by current glucose) that defeated the
naive per-patient response belief.

Fuzzy RD:
  running var  = current glucose (glucose_t)
  treatment    = insulin given (act_insulin dose / P(insulin>0))
  outcome      = glucose change over 6h (glucose_tp6 - glucose_t)
  effect       = jump_in_outcome_at_cutoff / jump_in_treatment_at_cutoff  (Wald)

Reports: (a) whether a treatment-probability discontinuity even EXISTS at the
threshold (first stage), (b) the RD insulin->glucose effect vs the naive
(confounded) OLS estimate. Subject-clustered bootstrap CIs.

This is a quasi-experiment: cleaner than naive observation but weaker than an
RCT, and its causal validity rests on the (untestable) RD continuity assumption.
"""
import argparse
import json

import numpy as np
import pandas as pd


def local_linear_jump(x, y, cutoff, h):
    m = np.abs(x - cutoff) <= h
    xr = x[m] - cutoff
    above = (xr >= 0).astype(float)
    X = np.column_stack([np.ones(m.sum()), above, xr, above * xr])
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return beta[1], int(m.sum())  # coefficient on `above` = jump at cutoff


def subject_boot_rd(x, tr, dg, subj, cutoff, h, n=300, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    idx_by = {s: np.where(subj == s)[0] for s in subs}
    ests = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs), replace=True)
        ii = np.concatenate([idx_by[s] for s in pick])
        fs, _ = local_linear_jump(x[ii], tr[ii], cutoff, h)
        rf, _ = local_linear_jump(x[ii], dg[ii], cutoff, h)
        if abs(fs) > 1e-6:
            ests.append(rf / fs)
    ests = np.array(ests)
    return (float(np.nanpercentile(ests, 2.5)), float(np.nanpercentile(ests, 97.5))) if len(ests) else (np.nan, np.nan)


def run(cohort, id_col, bandwidth, cutoffs):
    df = pd.read_parquet(cohort)
    g = pd.to_numeric(df["glucose_t"], errors="coerce").to_numpy()
    gp = pd.to_numeric(df["glucose_tp6"], errors="coerce").to_numpy()
    ins = pd.to_numeric(df["act_insulin_amount_like_sum"], errors="coerce").fillna(0).to_numpy()
    subj = df[id_col].to_numpy()
    ok = np.isfinite(g) & np.isfinite(gp)
    g, gp, ins, subj = g[ok], gp[ok], ins[ok], subj[ok]
    dg = gp - g
    tr_bin = (ins > 0).astype(float)
    print(f"rows with glucose pair: {len(g)}   insulin-treated fraction: {tr_bin.mean():.3f}")

    print("\nP(insulin>0) by glucose bin (look for a jump = a threshold rule):")
    bins = np.arange(80, 401, 20)
    lab = pd.cut(g, bins)
    tab = pd.Series(tr_bin).groupby(lab, observed=True).mean()
    for interval, p in tab.items():
        print(f"  glucose {int(interval.left):3d}-{int(interval.right):3d}:  P(insulin)={p:.3f}")

    print("\nfirst-stage treatment-probability jump at candidate cutoffs:")
    best = None
    for c in cutoffs:
        fs, npts = local_linear_jump(g, tr_bin, c, bandwidth)
        print(f"  cutoff {c}:  jump in P(insulin)={fs:+.3f}   (n={npts})")
        if best is None or abs(fs) > abs(best[1]):
            best = (c, fs, npts)
    cutoff, fs_best, npts = best
    print(f"\nstrongest discontinuity at glucose = {cutoff}  (first-stage jump {fs_best:+.3f})")

    rf, _ = local_linear_jump(g, dg, cutoff, bandwidth)
    rd = rf / fs_best if abs(fs_best) > 1e-6 else np.nan
    lo, hi = subject_boot_rd(g, tr_bin, dg, subj, cutoff, bandwidth)

    m = np.abs(g - cutoff) <= bandwidth
    Xn = np.column_stack([np.ones(m.sum()), ins[m], g[m]])
    beta_n, *_ = np.linalg.lstsq(Xn, dg[m], rcond=None)
    naive = beta_n[1]

    print(f"\n{'estimate of insulin -> 6h glucose change':45s}")
    print(f"  naive OLS (confounded, within band): {naive:+.4f}  mg/dL per unit insulin")
    print(f"  fuzzy RD (quasi-random at threshold): {rd:+.4f}  per unit insulin   95% CI [{lo:+.3f}, {hi:+.3f}]")
    clean = (hi < 0)
    print(f"\nfirst-stage strong (|jump|>0.05): {abs(fs_best) > 0.05}")
    print(f"RD shows insulin significantly LOWERS glucose (CI<0): {clean}")
    return {"cutoff": cutoff, "first_stage_jump": round(float(fs_best), 4),
            "naive_ols": round(float(naive), 4), "rd_estimate": round(float(rd), 4),
            "rd_ci": [round(lo, 3), round(hi, 3)],
            "first_stage_strong": bool(abs(fs_best) > 0.05),
            "rd_insulin_lowers_glucose": bool(clean)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--bandwidth", type=float, default=30.0)
    ap.add_argument("--cutoffs", default="140,150,160,180,200")
    ap.add_argument("--output", default="/tmp/rd_insulin_glucose.json")
    a = ap.parse_args()
    cutoffs = [float(x) for x in a.cutoffs.split(",")]
    out = run(a.cohort, a.id_col, a.bandwidth, cutoffs)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.output}")
