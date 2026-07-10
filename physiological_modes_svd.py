"""
physiological_modes_svd.py

Turn the learned linear model into interpretable physiology.

We fit the LINEARIZED DYNAMICS matrix A:

    delta_state (6h)  ~=  A @ current_state          (both standardized)

Predicting the CHANGE (not the level) removes the trivial persistence identity,
so A captures the actual coupling / dynamics around the homeostatic operating
point -- an empirical estimate of the system Jacobian.

Then:
  SVD  A = U S V^T  -> every linear map is rotate -> stretch -> rotate.
      right vectors (V) = combinations of CURRENT variables that drive motion
      left  vectors (U) = combinations of variables that MOVE in response
      singular values   = strength of each coupling mode; effective rank = how
                          many independent directions physiology actually uses

  eig(A)              -> relaxation modes of the homeostatic loops.
      eigenvalue real part < 0  => that mode decays back toward setpoint
      |real part| ~ how fast (per 6h step)
      eigenvector = the combination of variables that relaxes together
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DENSE = ["glucose", "potassium", "sodium", "chloride", "bicarbonate", "anion_gap",
         "bun", "creatinine", "calcium", "magnesium", "phosphate", "heart_rate",
         "map", "o2sat", "respiratory_rate", "temperature", "lactate",
         "hemoglobin", "hematocrit", "wbc", "platelets"]


def run(cohort, future_suffix, alpha, min_rows, top_modes, only_vars=None):
    df = pd.read_parquet(cohort)
    pool = only_vars if only_vars else DENSE
    vs = [v for v in pool if f"{v}_t" in df.columns and f"{v}{future_suffix}" in df.columns]
    if only_vars:
        missing = [v for v in only_vars if v not in vs]
        if missing:
            print(f"warning: requested vars missing from cohort: {missing}")
    cur = df[[f"{v}_t" for v in vs]].apply(pd.to_numeric, errors="coerce")
    fut = df[[f"{v}{future_suffix}" for v in vs]].apply(pd.to_numeric, errors="coerce")
    delta = fut.to_numpy() - cur.to_numpy()

    med = cur.median()
    X = cur.fillna(med).to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"rows={len(df)}  state variables={len(vs)}")

    A = np.zeros((len(vs), len(vs)))
    for j, v in enumerate(vs):
        y = delta[:, j]
        ok = np.isfinite(y)
        if ok.sum() < min_rows:
            print(f"  skip {v}: only {int(ok.sum())} rows")
            continue
        ys = (y[ok] - y[ok].mean()) / (y[ok].std() + 1e-9)
        A[j] = Ridge(alpha).fit(X[ok], ys).coef_

    U, S, Vt = np.linalg.svd(A)
    p = S / S.sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    print(f"\nsingular values: {np.round(S, 3)}")
    print(f"effective rank of physiology dynamics: {erank:.2f} / {len(vs)}")
    print("  (how many INDEPENDENT coupling directions the body actually uses)")

    print(f"\n=== top {top_modes} coupling modes (rotate -> stretch -> rotate) ===")
    for k in range(top_modes):
        drivers = sorted(zip(vs, Vt[k]), key=lambda t: -abs(t[1]))[:4]
        movers = sorted(zip(vs, U[:, k]), key=lambda t: -abs(t[1]))[:4]
        print(f"\nmode {k+1}   strength (singular value) = {S[k]:.3f}")
        print("  driven by :", ", ".join(f"{n}{'+' if w>0 else '-'}{abs(w):.2f}" for n, w in drivers))
        print("  moves     :", ", ".join(f"{n}{'+' if w>0 else '-'}{abs(w):.2f}" for n, w in movers))

    w, V = np.linalg.eig(A)
    order = np.argsort(-np.abs(w.real))
    print(f"\n=== relaxation modes (eigenvalues of the dynamics) ===")
    print("  real<0 = mean-reverting toward setpoint; |real| = speed per 6h")
    stable = int((w.real < 0).sum())
    print(f"  mean-reverting modes: {stable}/{len(w)}")
    for k in order[:top_modes]:
        vec = sorted(zip(vs, np.abs(V[:, k])), key=lambda t: -t[1])[:4]
        lam = w[k]
        print(f"\n  eigenvalue {lam.real:+.3f}{lam.imag:+.3f}j   "
              f"({'decays' if lam.real < 0 else 'grows'} per 6h)")
        print("    variables moving together:", ", ".join(f"{n}({m:.2f})" for n, m in vec))

    return {"variables": vs, "singular_values": [round(float(s), 4) for s in S],
            "effective_rank": round(erank, 3),
            "mean_reverting_modes": f"{stable}/{len(w)}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--future-suffix", default="_tp6")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min-rows", type=int, default=2000)
    ap.add_argument("--top-modes", type=int, default=4)
    ap.add_argument("--vars", default="", help="comma-separated variable subset (for cross-database comparison)")
    ap.add_argument("--output", default="/tmp/physiological_modes_svd.json")
    a = ap.parse_args()
    only = [v for v in a.vars.split(",") if v] or None
    out = run(a.cohort, a.future_suffix, a.alpha, a.min_rows, a.top_modes, only)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.output}")
