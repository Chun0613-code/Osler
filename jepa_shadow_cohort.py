"""Aggregate JEPA shadow reconciliations into patient-grouped evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osler_jepa.shadow_cohort import evaluate_shadow_cohort, load_reconciliations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", default="jepa_shadow_cohort_report.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--candidate")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    report = evaluate_shadow_cohort(
        load_reconciliations(args.ledger),
        checkpoint=args.checkpoint,
        candidate=args.candidate,
        bootstrap_samples=args.bootstrap_samples,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
