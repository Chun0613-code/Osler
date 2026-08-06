"""
dka_calibrate.py  (L0 calibrator, k-fold)
=========================================
Fit the sim's constants to real DKA trajectories, WITH cross-validation so the result
is trustworthy and you can see the overfit gap.

  - in-sample : fit on all stays, report on all stays  (OPTIMISTIC / overfit)
  - CV (k-fold, pooled out-of-fold): each stay is scored under constants fitted WITHOUT
    it -> honest generalization estimate. THIS is the number to trust.

On the 12-stay demo the folds are tiny and noisy (the point is the machinery + showing
the overfit gap). On full MIMIC (hundreds of DKA stays) the CV number is the real,
non-overfit L0 calibration.

Params fit: glucose (K_HEP, K_UPTAKE_INS) + K+ transcellular shift, buffer,
and bounded urinary concentration.
Acid-base held (pH/HCO3 already faithful).

Run:  python dka_calibrate.py trajectories.jsonl [k]
"""

import sys
import json
import numpy as np
from scipy.optimize import minimize
import dka_body
from dka_fidelity_replay import replay

PARAMS = [
    "K_HEP", "K_UPTAKE_INS", "K_K_INS", "K_K_BUF",
    "K_URINE_K_BASE", "K_URINE_K_OSM",
]
BOUNDS = {
    "K_HEP": (20, 60), "K_UPTAKE_INS": (0.05, 0.30),
    "K_K_INS": (0.05, 0.40), "K_K_BUF": (0.10, 0.60),
    "K_URINE_K_BASE": (2.0, 20.0), "K_URINE_K_OSM": (2.0, 25.0),
}
VARSCALE = {"glucose": 50.0, "HCO3": 3.0, "K": 0.5, "pH": 0.1}
DEATH_PENALTY = 3.0
X0 = np.array([40.0, 0.13, 0.15, 0.25, 8.0, 12.0])


def set_params(x):
    for name, v in zip(PARAMS, x):
        lo, hi = BOUNDS[name]
        setattr(dka_body, name, float(np.clip(v, lo, hi)))


def collect_err(trajs):
    """replay each stay; return list of (var,|err|) and #deaths under current params."""
    pairs, deaths = [], 0
    for tr in trajs:
        paired, _, alive, _ = replay(tr)
        if not alive:
            deaths += 1
        for var, t, real, sim in paired:
            pairs.append((var, abs(real - sim)))
    return pairs, deaths


def mae_of(pairs, deaths, n):
    d = {}
    for var, e in pairs:
        d.setdefault(var, []).append(e)
    return {v: float(np.mean(e)) for v, e in d.items()}, deaths / n


def objective(x, trajs):
    set_params(x)
    pairs, deaths = collect_err(trajs)
    mae, dr = mae_of(pairs, deaths, len(trajs))
    return sum(mae.get(v, 0) / VARSCALE[v] for v in VARSCALE) + DEATH_PENALTY * dr


def fit_on(train):
    res = minimize(objective, X0, args=(train,), method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-2, "fatol": 1e-3})
    return res.x


def report(tag, pairs, deaths, n):
    mae, dr = mae_of(pairs, deaths, n)
    print(f"  {tag:<22}" + "  ".join(f"{k}={mae.get(k,0):6.2f}" for k in VARSCALE)
          + f"   deaths={dr*100:3.0f}%")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "trajectories.jsonl"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    trajs = [json.loads(l) for l in open(path) if l.strip()]
    n = len(trajs)
    print(f"L0 calibration on {n} DKA stays, {k}-fold CV\n")

    # baseline: sim defaults, no fitting
    set_params(X0)
    p, d = collect_err(trajs); report("defaults (no fit)", p, d, n)

    # in-sample (overfit): fit on all, report on all
    x_all = fit_on(trajs); set_params(x_all)
    p, d = collect_err(trajs); report("in-sample (overfit)", p, d, n)

    # k-fold pooled out-of-fold (honest)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = [idx[i::k] for i in range(k)]
    pooled, pooled_deaths = [], 0
    for fi in range(k):
        test = [trajs[i] for i in folds[fi]]
        train = [trajs[i] for i in range(n) if i not in set(folds[fi])]
        if not test or not train:
            continue
        set_params(fit_on(train))
        pr, de = collect_err(test)
        pooled += pr; pooled_deaths += de
    report("CV out-of-fold (HONEST)", pooled, pooled_deaths, n)

    print("\n  in-sample vs CV gap = overfit optimism. On 12 stays it's large/noisy;")
    print("  on full MIMIC the CV row is the trustworthy L0 calibration.")
    clipped = {p_: round(float(np.clip(v, *BOUNDS[p_])), 3) for p_, v in zip(PARAMS, x_all)}
    print(f"\n  (deployable params, all-data fit, clipped to bounds: {clipped})")
    at_bound = [p_ for p_, v in zip(PARAMS, x_all) if v <= BOUNDS[p_][0] or v >= BOUNDS[p_][1]]
    if at_bound:
        print(f"  NOTE: {at_bound} hit a bound -> structural mismatch worth investigating, "
              f"not just a constant to nudge.")


if __name__ == "__main__":
    main()
