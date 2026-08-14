"""Build the fail-closed registry for renal patient-specific precision cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVALUATED_CELLS = tuple(
    (target, horizon)
    for target in ("creatinine", "bun", "urine_output")
    for horizon in (1, 3, 6, 12, 24, 48)
) + (("map", 3), ("map", 6))
REPORT_OVERRIDES = {
    ("creatinine", 3): "patient_state_precision_creatinine3_full_20260814.json",
    ("creatinine", 12): "patient_state_precision_creatinine12_full_20260814.json",
    ("creatinine", 24): "patient_state_precision_creatinine24_full_20260814.json",
    ("bun", 6): "patient_state_precision_bun6_full_20260814.json",
    ("bun", 24): "patient_state_precision_bun24_full_20260814.json",
    ("urine_output", 6): "patient_state_precision_urine_output6_full_20260814.json",
    ("urine_output", 12): "patient_state_precision_urine_output12_full_20260814.json",
    ("map", 3): "patient_state_precision_map3_full_20260814.json",
    ("map", 6): "patient_state_precision_map6_full_20260814.json",
}


def _artifact_name(target: str, horizon: int) -> str:
    return f"patient_state_precision_{target}{horizon}_v1"


def build_registry(root: Path) -> dict[str, object]:
    cells: dict[str, object] = {}
    promoted: list[str] = []
    for target, horizon in EVALUATED_CELLS:
        key = f"{target}@{horizon}h"
        report_name = REPORT_OVERRIDES.get(
            (target, horizon),
            f"patient_state_interval_width_{target}{horizon}_full_20260813.json",
        )
        report_path = root / report_name
        if not report_path.exists():
            raise FileNotFoundError(f"missing renal precision report: {report_name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        point_pass = report["promotion"]["status"] == "promoted"
        interval_pass = bool(report["interval_narrowing"]["claim_allowed"])
        precision_pass = bool(point_pass and interval_pass)
        patient = report["patient_heldout"]["summary"]
        external = report["external_heldout"]["summary"]
        artifact = None
        if precision_pass:
            artifact_name = _artifact_name(target, horizon)
            artifact_path = root / artifact_name
            metadata_path = artifact_path / "metadata.json"
            manifest_path = artifact_path / "MANIFEST.sha256"
            if not metadata_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(
                    f"precision-promoted cell lacks complete artifact: {key}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("target") != target
                or metadata.get("horizon_hours") != horizon
                or metadata.get("validation_report") != report_name
            ):
                raise ValueError(f"artifact/report mismatch for {key}")
            artifact = artifact_name
            promoted.append(key)
        cells[key] = {
            "target": target,
            "horizon_hours": horizon,
            "adapter_organ": "kidneys",
            "target_system": (
                "cardiovascular_perfusion" if target == "map" else "renal"
            ),
            "point_personalization_pass": point_pass,
            "matched_interval_narrowing_pass": interval_pass,
            "precision_promoted": precision_pass,
            "patient_narrowing_gates": (
                f"{patient['matched_interval_narrowing_passes']}/7"
            ),
            "external_narrowing_gates": (
                f"{external['matched_interval_narrowing_passes']}/3"
            ),
            "patient_median_delta_half_width": patient[
                "median_patient_equalized_delta_half_width"
            ],
            "external_median_delta_half_width": external[
                "median_patient_equalized_delta_half_width"
            ],
            "validation_report": report_name,
            "artifact": artifact,
            "fallback": None if precision_pass else "validated_population_source",
        }
    return {
        "schema": "renal_patient_state_precision_registry.v1",
        "coverage_target": 0.90,
        "required_coverage_range": [0.87, 0.93],
        "required_gates": [
            "7_seed_patient_heldout",
            "hospital_heldout",
            "care_unit_heldout",
            "late_stage_heldout",
            "population_anchor",
            "capacity_matched_placebo",
            "patient_paired_bootstrap",
            "matched_coverage_interval_narrowing",
        ],
        "evaluated_cells": len(cells),
        "precision_promoted_cells": promoted,
        "precision_promoted_count": len(promoted),
        "cells": cells,
        "boundary": {
            "factual_only": True,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "unsupported_cells_keep_validated_population_source": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("renal_patient_state_precision_registry_20260813.json"),
    )
    args = parser.parse_args()
    registry = build_registry(args.root.resolve())
    args.output.write_text(
        json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "evaluated_cells": registry["evaluated_cells"],
                "precision_promoted_cells": registry["precision_promoted_cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
