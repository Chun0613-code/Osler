"""Run matched-control and AIPW diagnostics on extracted DKA transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from osler_jepa.causal_evaluation import evaluate_target_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="dka_transitions_6h.parquet")
    parser.add_argument("--output", default="dka_causal_evaluation.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    report = evaluate_target_trials(
        frame, bootstrap_samples=args.bootstrap_samples
    )
    report["cohort"] = {
        "path": str(Path(args.input)),
        "rows": int(len(frame)),
        "stays": int(frame["stay_id"].nunique()),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
