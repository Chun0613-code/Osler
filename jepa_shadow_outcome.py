"""Score a stored JEPA shadow forecast against later measured physiology."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from osler_jepa.shadow_outcomes import reconcile_from_ledger


def _read_json(value: str):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--future-json", required=True)
    parser.add_argument("--actual-action-json", required=True)
    parser.add_argument("--elapsed-hours", required=True, type=float)
    parser.add_argument("--candidate")
    parser.add_argument("--output")
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument(
        "--subject-key-env", default="OSLER_JEPA_SUBJECT_KEY",
        help="Environment variable containing a local subject key; never stored.",
    )
    args = parser.parse_args()

    record = reconcile_from_ledger(
        args.ledger,
        args.event_id,
        _read_json(args.future_json),
        _read_json(args.actual_action_json),
        args.elapsed_hours,
        candidate=args.candidate,
        append=not args.no_append,
        subject_key=os.environ.get(args.subject_key_env),
    )
    rendered = json.dumps(record, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
