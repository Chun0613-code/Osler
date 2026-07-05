"""Extract bounded candidate features from local MIMIC-IV Waveform records.

This is a feature-engineering candidate step, not a validated forecasting
pipeline.  Row-level feature files should be written outside the repository
(for example under /tmp) and must pass a separate held-out accuracy audit before
any waveform feature is used for factual prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from osler_jepa.waveform import parse_wfdb_header_text
from osler_jepa.waveform_features import (
    aggregate_feature_summary,
    read_waveform_feature_record,
)


def iter_signal_segment_records(root: Path):
    """Yield record paths for WFDB headers that contain signal labels."""

    for header_path in sorted(root.rglob("*.hea")):
        try:
            parsed = parse_wfdb_header_text(header_path.read_text(errors="replace"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if parsed.header_kind != "signal_segment" or not parsed.signal_groups:
            continue
        yield header_path.with_suffix("")


def extract_waveform_features(
    root: Path,
    seconds: float,
    max_records: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract candidate waveform features and return rows plus summary."""

    feature_records = []
    failures = 0
    for record_path in iter_signal_segment_records(root):
        if max_records is not None and len(feature_records) >= max_records:
            break
        try:
            feature_records.append(read_waveform_feature_record(record_path, seconds=seconds))
        except Exception:
            failures += 1

    rows: list[dict[str, object]] = []
    feature_names = sorted({name for record in feature_records for name in record.features})
    for record in feature_records:
        row: dict[str, object] = {
            "record_name": record.record_name,
            "source_path": record.source_path,
            "sampling_frequency_hz": record.sampling_frequency_hz,
            "seconds_read": record.seconds_read,
            "signals": "|".join(record.signals),
            "signal_groups": "|".join(record.signal_groups),
        }
        for name in feature_names:
            row[name] = record.features.get(name)
        rows.append(row)

    summary = aggregate_feature_summary(feature_records)
    summary.update(
        {
            "artifact": "MIMIC-IV Waveform bounded feature extraction smoke",
            "seconds_requested": seconds,
            "max_records": max_records,
            "read_failures": failures,
            "authority": {
                "forecast_feature_authority": False,
                "accuracy_audit_required_before_use": True,
            },
        }
    )
    return rows, summary


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    """Write row-level feature candidates to CSV."""

    if not rows:
        output.write_text("")
        return
    fieldnames = sorted({name for row in rows for name in row})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-root", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--features-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    rows, summary = extract_waveform_features(
        args.waveform_root,
        seconds=args.seconds,
        max_records=args.max_records,
    )
    write_csv(rows, args.features_output)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
