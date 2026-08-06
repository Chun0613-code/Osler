"""Build an aggregate MIMIC-IV Waveform manifest.

This script scans WFDB ``.hea`` headers and reports only aggregate signal
coverage.  It does not read waveform samples, does not emit row-level records,
and does not authorize any forecast feature.  The output is the first gate for
deciding whether the heavier waveform feature-extraction pipeline is worth
running.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osler_jepa.waveform import (
    parse_wfdb_header_text,
    summarize_waveform_headers,
    waveform_readiness,
)


def iter_header_paths(root: Path):
    """Yield WFDB header paths under ``root``."""

    yield from sorted(root.rglob("*.hea"))


def scan_waveform_headers(root: Path, limit: int | None = None) -> dict[str, object]:
    """Scan waveform headers and return an aggregate-only manifest."""

    headers = []
    parse_errors = 0
    for path in iter_header_paths(root):
        if limit is not None and len(headers) >= limit:
            break
        try:
            headers.append(parse_wfdb_header_text(path.read_text(errors="replace")))
        except (OSError, UnicodeDecodeError, ValueError):
            parse_errors += 1

    summary = summarize_waveform_headers(headers)
    summary.update(
        {
            "artifact": "MIMIC-IV Waveform aggregate manifest",
            "waveform_root_name": root.name,
            "headers_scanned": int(summary["record_count"]),
            "parse_errors": int(parse_errors),
            "readiness": waveform_readiness(summary),
            "boundary": {
                "raw_samples_read": False,
                "row_level_manifest_emitted": False,
                "forecast_feature_authority": False,
                "accuracy_audit_required_before_use": True,
            },
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = scan_waveform_headers(args.waveform_root, limit=args.limit)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest["readiness"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
