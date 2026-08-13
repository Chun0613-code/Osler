"""Convert public eICU CRD Demo 2.0.1 tables into forecast demo cases.

Only observations at or before the selected anchor are used. Source tables are
read locally; this adapter does not download data or persist patient records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from osler_jepa.urine_output import derive_interval_urine_rate


SOURCE = {
    "dataset": "eICU Collaborative Research Database Demo",
    "version": "2.0.1",
    "url": "https://physionet.org/content/eicu-crd-demo/2.0.1/",
    "license": "Open Data Commons Open Database License v1.0",
    "deidentification": "PhysioNet demo data; HIPAA safe-harbor deidentified",
}

LAB_MAP = {
    "BUN": "bun",
    "HCO3": "bicarbonate",
    "anion gap": "anion_gap",
    "creatinine": "creatinine",
    "glucose": "glucose",
    "potassium": "potassium",
    "sodium": "sodium",
}

VITAL_MAP = {
    "heartrate": "heart_rate",
    "respiration": "respiratory_rate",
    "sao2": "o2sat",
    "systemicmean": "map",
    "temperature": "temperature",
}


def build_monitoring_case(
    data_dir: str | Path,
    patient_unit_stay_id: int,
    *,
    anchor_offset_minutes: int | None = None,
    max_points: int = 6,
) -> dict[str, Any]:
    """Build one causally ordered monitoring case from local demo CSV files."""

    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    data_dir = Path(data_dir)
    stay_id = int(patient_unit_stay_id)
    patient = _read(data_dir, "patient", [
        "patientunitstayid", "gender", "age", "unittype", "apacheadmissiondx"
    ])
    patient = patient[patient["patientunitstayid"].eq(stay_id)]
    if len(patient) != 1:
        raise ValueError(f"expected one patient row for stay {stay_id}")

    labs = _read(data_dir, "lab", [
        "patientunitstayid", "labresultoffset", "labname", "labresult"
    ])
    labs = labs[
        labs["patientunitstayid"].eq(stay_id)
        & labs["labname"].isin(LAB_MAP)
        & labs["labresultoffset"].ge(0)
    ].copy()
    labs["labresult"] = pd.to_numeric(labs["labresult"], errors="coerce")
    labs = labs[np.isfinite(labs["labresult"])]
    core_offsets = sorted(
        labs.loc[labs["labname"].isin(("BUN", "creatinine")), "labresultoffset"]
        .astype(int)
        .unique()
        .tolist()
    )
    if anchor_offset_minutes is None:
        if not core_offsets:
            raise ValueError("stay has no non-future BUN/creatinine anchor")
        anchor_offset_minutes = core_offsets[-1]
    anchor = int(anchor_offset_minutes)
    if anchor < 0:
        raise ValueError("anchor_offset_minutes must be non-negative")
    labs = labs[labs["labresultoffset"].le(anchor)]
    core_offsets = [offset for offset in core_offsets if offset <= anchor]
    if len(core_offsets) < 2:
        raise ValueError("stay needs at least two BUN/creatinine observation times")
    offsets = _evenly_spaced(core_offsets, max_points)

    vitals = _read(data_dir, "vitalPeriodic", [
        "patientunitstayid", "observationoffset", *VITAL_MAP.keys()
    ])
    vitals = vitals[
        vitals["patientunitstayid"].eq(stay_id)
        & vitals["observationoffset"].between(0, anchor)
    ].sort_values("observationoffset")

    intake = _read(data_dir, "intakeOutput", [
        "patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel",
        "cellvaluenumeric"
    ])
    intake = intake[
        intake["patientunitstayid"].eq(stay_id)
        & intake["intakeoutputoffset"].between(0, anchor)
    ]
    urine = derive_interval_urine_rate(intake, id_column="patientunitstayid")

    trajectory = [
        _snapshot(offset, labs=labs, vitals=vitals, urine=urine)
        for offset in offsets
    ]
    row = patient.iloc[0]
    return {
        "id": f"eicu-demo-{stay_id}",
        "title": "Deidentified post-operative ICU monitoring trajectory",
        "case_source": "eicu_crd_demo_2.0.1",
        "source_metadata": {
            **SOURCE,
            "patient_unit_stay_id": stay_id,
            "source_table_offsets_are_minutes_from_icu_admission": True,
        },
        "case_context": {
            "age": _json_scalar(row.get("age")),
            "gender": _json_scalar(row.get("gender")),
            "unit_type": _json_scalar(row.get("unittype")),
            "source_admission_label": _json_scalar(row.get("apacheadmissiondx")),
            "diagnostic_claim_allowed": False,
        },
        "forecast_payload": {
            "patient_id": f"eicu-demo-{stay_id}",
            "case_source": "eicu_crd_demo_2.0.1",
            "anchor_hour": anchor / 60.0,
            "trajectory": trajectory,
        },
        "safety_boundary": {
            "factual_research_only": True,
            "diagnostic_claim_allowed": False,
            "causal_claim_allowed": False,
            "treatment_recommendation_allowed": False,
            "automated_prescribing_allowed": False,
        },
    }


def _read(data_dir: Path, table: str, columns: list[str]) -> pd.DataFrame:
    for suffix in (".csv.gz", ".csv"):
        path = data_dir / f"{table}{suffix}"
        if path.is_file():
            return pd.read_csv(path, usecols=columns)
    raise ValueError(f"missing eICU demo table: {table}.csv.gz")


def _evenly_spaced(offsets: list[int], max_points: int) -> list[int]:
    if len(offsets) <= max_points:
        return offsets
    indices = np.linspace(0, len(offsets) - 1, max_points).round().astype(int)
    return [offsets[index] for index in sorted(set(indices.tolist()))]


def _snapshot(
    offset: int,
    *,
    labs: pd.DataFrame,
    vitals: pd.DataFrame,
    urine: pd.DataFrame,
) -> dict[str, float]:
    result: dict[str, float] = {"hours_since_onset": offset / 60.0}
    for source, target in LAB_MAP.items():
        observed = labs[
            labs["labname"].eq(source) & labs["labresultoffset"].le(offset)
        ].sort_values("labresultoffset")
        if observed.empty:
            continue
        latest = observed.iloc[-1]
        result[target] = float(latest["labresult"])
        result[f"{target}_age_hr"] = (
            offset - float(latest["labresultoffset"])
        ) / 60.0
    for source, target in VITAL_MAP.items():
        observed = vitals[
            vitals["observationoffset"].le(offset) & vitals[source].notna()
        ]
        if observed.empty:
            continue
        latest = observed.iloc[-1]
        age_minutes = offset - float(latest["observationoffset"])
        if age_minutes <= 240:
            result[target] = float(latest[source])
            result[f"{target}_age_hr"] = age_minutes / 60.0
    observed_urine = urine[urine["offset"].le(offset)].sort_values("offset")
    if not observed_urine.empty:
        latest = observed_urine.iloc[-1]
        result["urine_output"] = float(latest["valuenum"])
        result["urine_output_age_hr"] = (
            offset - float(latest["offset"])
        ) / 60.0
    return result


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stay-id", type=int, required=True)
    parser.add_argument("--anchor-offset-minutes", type=int)
    parser.add_argument("--max-points", type=int, default=6)
    args = parser.parse_args()
    case = build_monitoring_case(
        args.data_dir,
        args.stay_id,
        anchor_offset_minutes=args.anchor_offset_minutes,
        max_points=args.max_points,
    )
    print(json.dumps(case, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
