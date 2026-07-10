"""
synthetic_dose_response_validation.py

Validates YOUR specific machinery (fuzzy RD + naive dose-response) on synthetic
ICU-like data where the insulin->glucose effect is KNOWN, with MIMIC-like
confounding (insulin dose driven by current glucose) PLUS a threshold rule (so
an RD discontinuity exists).

Demonstrates three things at once:
  1. naive OLS is BIASED by the confounding (recovers the wrong effect).
  2. fuzzy RD at the threshold RECOVERS the true effect.
  3. therefore MIMIC's RD failure was because MIMIC has NO discontinuity to
     exploit (P(insulin) rose smoothly) -- not because RD is wrong.

No download needed; ground truth is set by us.
"""
import argparse
import json

import numpy as np


def local_linear_jump(x, y, cutoff, h):
    m = np.abs(x - cutoff) <= h
    xr = x[m] - cutoff
    above = (xr >= 0).astype(float)
    X = np.column_stack([np.ones(m.sum()), above, xr, above * xr])
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return beta[1]


def simulate(n, true_effect, cutoff, jump, confound, seed, with_jump=True):
    rng = np.random.default_rng(seed)
    glucose = rng.normal(170, 45, n).clip(60, 400)
    # UNOBSERVED confounder U (e.g. illness severity): raises insulin dose AND
    # independently pushes glucose up -- NOT available to the naive regression.
    # This is the time-varying confounding that biases naive estimates in MIMIC.
    U = rng.normal(0, 1, n)
    base = confound * (glucose - 120) + 8.0 * U
    thr = jump * (glucose >= cutoff) if with_jump else 0.0
    insulin = (base + thr + rng.normal(0, 3, n)).clip(0, None)
    setpoint = 110
    dg = 0.15 * (setpoint - glucose) + true_effect * insulin + 12.0 * U + rng.normal(0, 15, n)
    return glucose, insulin, dg


def run(n, true_effect, cutoff, jump, confound, bandwidth, seed):
    print(f"ground truth insulin effect: {true_effect:+.3f} mg/dL per unit insulin\n")

    for label, with_jump in [("WITH threshold discontinuity (like a clean protocol)", True),
                             ("NO discontinuity (like real MIMIC)", False)]:
        g, ins, dg = simulate(n, true_effect, cutoff, jump, confound, seed, with_jump)
        # naive OLS: dg ~ insulin + glucose
        X = np.column_stack([np.ones(n), ins, g])
        beta, *_ = np.linalg.lstsq(X, dg, rcond=None)
        naive = beta[1]
        # fuzzy RD at cutoff
        fs = local_linear_jump(g, ins.astype(float), cutoff, bandwidth)
        rf = local_linear_jump(g, dg, cutoff, bandwidth)
        rd = rf / fs if abs(fs) > 1e-6 else np.nan
        print(f"--- {label} ---")
        print(f"  first-stage jump in insulin at cutoff: {fs:+.3f}  (strong if |.|>0.05)")
        print(f"  naive OLS effect : {naive:+.4f}   (truth {true_effect:+.3f})")
        print(f"  fuzzy RD effect  : {rd:+.4f}   (truth {true_effect:+.3f})")
        print()

    return {"true_effect": true_effect}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--true-effect", type=float, default=-0.20)
    ap.add_argument("--cutoff", type=float, default=180.0)
    ap.add_argument("--jump", type=float, default=6.0, help="insulin-dose jump at the threshold")
    ap.add_argument("--confound", type=float, default=0.08, help="how strongly glucose drives dose")
    ap.add_argument("--bandwidth", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="/tmp/synthetic_dose_response_validation.json")
    a = ap.parse_args()
    out = run(a.n, a.true_effect, a.cutoff, a.jump, a.confound, a.bandwidth, a.seed)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.output}")
