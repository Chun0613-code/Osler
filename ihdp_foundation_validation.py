"""
ihdp_foundation_validation.py

Foundation check: does the causal-ADJUSTMENT machinery recover a KNOWN effect
under confounding? Uses the semi-synthetic IHDP benchmark (ground truth known).

Download the IHDP-1000 csvs first (no account), e.g.:
  github.com/claudiashi57/dragonnet  ->  dat/ihdp/csv/ihdp_npci_*.csv
then:
  python3 ihdp_foundation_validation.py --csv-glob 'dat/ihdp/csv/ihdp_npci_*.csv'

Each csv has: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25.
True ATE = mean(mu1 - mu0). We compare:
  naive ATE   = E[y|t=1] - E[y|t=0]                 (confounded, should be biased)
  adjusted ATE = T-learner outcome regression        (should recover true ATE)

If adjusted-error << naive-error, the confounding-adjustment machinery is sound.
This validates the FOUNDATION (recovering a known effect under confounding), NOT
the RD / dose-response specifics (IHDP is binary-treatment, no running variable).
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def load(path):
    # standard IHDP-npci format is HEADERLESS, positional:
    # treatment, y_factual, y_cfactual, mu0, mu1, x1..x25
    df = pd.read_csv(path, header=None)
    t = df[0].to_numpy().astype(float)
    y = df[1].to_numpy().astype(float)
    mu0 = df[3].to_numpy().astype(float)
    mu1 = df[4].to_numpy().astype(float)
    X = df.iloc[:, 5:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    return t, y, mu0, mu1, X


def estimates(t, y, X):
    naive = y[t == 1].mean() - y[t == 0].mean()
    m1 = Ridge(1.0).fit(X[t == 1], y[t == 1])
    m0 = Ridge(1.0).fit(X[t == 0], y[t == 0])
    adj = float((m1.predict(X) - m0.predict(X)).mean())
    return naive, adj


def run(csv_glob):
    files = sorted(glob.glob(csv_glob))
    if not files:
        print(f"no files matched {csv_glob} -- download the IHDP-1000 csvs first")
        return {"error": "no_files"}
    print(f"replications: {len(files)}")
    ne, ae, trues = [], [], []
    for f in files:
        t, y, mu0, mu1, X = load(f)
        true = float((mu1 - mu0).mean())
        naive, adj = estimates(t, y, X)
        ne.append(abs(naive - true)); ae.append(abs(adj - true)); trues.append(true)
    ne, ae = np.array(ne), np.array(ae)
    print(f"\nmean true ATE            : {np.mean(trues):.3f}")
    print(f"naive ATE  |error|  mean : {ne.mean():.3f}")
    print(f"adjusted ATE |error| mean: {ae.mean():.3f}")
    print(f"adjusted better than naive in: {int((ae < ne).sum())}/{len(files)} replications")
    ok = ae.mean() < 0.5 * ne.mean()
    print(f"\nFOUNDATION VALIDATED (adjustment recovers truth, naive biased): {ok}")
    return {"replications": len(files), "naive_error": round(float(ne.mean()), 4),
            "adjusted_error": round(float(ae.mean()), 4),
            "adjusted_beats_naive": f"{int((ae < ne).sum())}/{len(files)}",
            "foundation_validated": bool(ok)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-glob", required=True, help="glob for ihdp_npci_*.csv")
    ap.add_argument("--output", default="/tmp/ihdp_foundation_validation.json")
    a = ap.parse_args()
    out = run(a.csv_glob)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.output}")
