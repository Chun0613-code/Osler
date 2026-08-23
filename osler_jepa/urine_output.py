"""Canonical interval-aware urine-output normalization.

eICU stores a urine volume at the end of an irregular collection interval.
Treating that volume as an instantaneous value, or dividing every value by a
fixed number of hours, encodes local charting practice instead of physiology.
This module converts observed volumes to an auditable mL/hour rate using the
patient's actual preceding collection interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


URINE_LABEL_PATTERN = r"urine|foley|urinary|void"
NON_VOLUME_LABEL_PATTERN = r"count|occurrence|unmeasured|mixed"


def derive_timestamped_urine_rate(
    frame: pd.DataFrame,
    *,
    id_column: str,
    time_column: str,
    value_column: str,
    maximum_interval_hours: float = 24.0,
) -> pd.DataFrame:
    """Normalize timestamped urine volumes to an interval rate in mL/hour.

    MIMIC outputevents records a volume at chart time, not an instantaneous
    rate. The preceding timestamp is therefore part of the measurement. The
    first event is excluded because its collection start is unknown.
    """

    output_columns = (
        id_column,
        time_column,
        "valuenum",
        "collection_interval_hr",
        "var",
        "quality",
    )
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    if maximum_interval_hours <= 0:
        raise ValueError("maximum_interval_hours must be positive")

    selected = frame[[id_column, time_column, value_column]].copy()
    selected[time_column] = pd.to_datetime(selected[time_column], errors="coerce")
    selected["volume_ml"] = pd.to_numeric(
        selected[value_column], errors="coerce"
    )
    selected = selected[
        selected[time_column].notna()
        & np.isfinite(selected["volume_ml"])
        & selected["volume_ml"].ge(0.0)
    ]
    if selected.empty:
        return pd.DataFrame(columns=output_columns)

    grouped = (
        selected.groupby([id_column, time_column], sort=False)["volume_ml"]
        .sum()
        .reset_index()
        .sort_values([id_column, time_column], kind="stable")
    )
    grouped["collection_interval_hr"] = (
        grouped.groupby(id_column, sort=False)[time_column]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )
    interval = grouped["collection_interval_hr"]
    grouped = grouped[
        interval.gt(0.0) & interval.le(float(maximum_interval_hours))
    ].copy()
    if grouped.empty:
        return pd.DataFrame(columns=output_columns)

    grouped["valuenum"] = grouped["volume_ml"] / grouped["collection_interval_hr"]
    grouped = grouped[
        np.isfinite(grouped["valuenum"])
        & grouped["valuenum"].between(0.0, 2000.0, inclusive="both")
    ].copy()
    grouped["var"] = "urine_output"
    grouped["quality"] = "derived_interval_rate"
    return grouped[list(output_columns)].reset_index(drop=True)


def derive_interval_urine_rate(
    frame: pd.DataFrame,
    *,
    id_column: str,
    offset_column: str = "intakeoutputoffset",
    label_column: str = "celllabel",
    value_column: str = "cellvaluenumeric",
    maximum_interval_hours: float = 24.0,
) -> pd.DataFrame:
    """Return interval-normalized urine output in mL/hour.

    The first volume in an encounter is excluded because its collection start
    is unknown. Intervals longer than ``maximum_interval_hours`` are excluded
    because they are usually retrospective summaries rather than comparable
    physiological measurements. Rows that count voids or combine urine and
    stool are not volumes and are also excluded.
    """

    output_columns = (
        "stay_id",
        "offset",
        "var",
        "valuenum",
        "collection_interval_hr",
        "quality",
    )
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    if maximum_interval_hours <= 0:
        raise ValueError("maximum_interval_hours must be positive")

    labels = frame[label_column].fillna("").astype(str).str.lower()
    paths = frame.get(
        "cellpath", pd.Series("", index=frame.index, dtype="object")
    ).fillna("").astype(str).str.lower()
    is_urine = labels.str.contains(URINE_LABEL_PATTERN, regex=True)
    is_non_volume = labels.str.contains(NON_VOLUME_LABEL_PATTERN, regex=True)
    is_output_volume = paths.eq("") | paths.str.contains(
        r"\|output \(ml\)\|", regex=True
    )

    selected = frame.loc[is_urine & ~is_non_volume & is_output_volume].copy()
    if selected.empty:
        return pd.DataFrame(columns=output_columns)
    selected["offset"] = pd.to_numeric(
        selected[offset_column], errors="coerce"
    )
    selected["volume_ml"] = pd.to_numeric(
        selected[value_column], errors="coerce"
    )
    selected = selected[
        np.isfinite(selected["offset"])
        & np.isfinite(selected["volume_ml"])
        & (selected["volume_ml"] >= 0.0)
    ]
    if selected.empty:
        return pd.DataFrame(columns=output_columns)

    # Multiple catheter rows can share a chart time. Sum them before deriving
    # the interval so duplicated rows do not create a zero-duration rate.
    grouped = (
        selected.groupby([id_column, "offset"], sort=False)["volume_ml"]
        .sum()
        .reset_index()
        .sort_values([id_column, "offset"], kind="stable")
    )
    grouped["collection_interval_hr"] = (
        grouped.groupby(id_column, sort=False)["offset"].diff() / 60.0
    )
    interval = grouped["collection_interval_hr"]
    grouped = grouped[
        interval.gt(0.0) & interval.le(float(maximum_interval_hours))
    ].copy()
    if grouped.empty:
        return pd.DataFrame(columns=output_columns)

    grouped["valuenum"] = (
        grouped["volume_ml"] / grouped["collection_interval_hr"]
    )
    grouped = grouped[
        np.isfinite(grouped["valuenum"])
        & grouped["valuenum"].between(0.0, 2000.0, inclusive="both")
    ].copy()
    grouped["var"] = "urine_output"
    grouped["quality"] = "derived"
    grouped = grouped.rename(columns={id_column: "stay_id"})
    return grouped[list(output_columns)].reset_index(drop=True)
