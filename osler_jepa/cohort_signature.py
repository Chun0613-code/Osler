"""De-identified fingerprints for matching model-evaluation cohorts."""

from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd


def cohort_signature(
    frame: pd.DataFrame,
    variables: Sequence[str],
) -> str:
    """Hash anchor identity and supervision without exporting row data."""

    preferred = (
        "subject_id",
        "stay_id",
        "encounter_id",
        "t",
        "anchor_time",
        "_source_horizon",
        "horizon_hours",
    )
    columns = [column for column in preferred if column in frame]
    for variable in variables:
        for pattern in (
            f"{variable}_t",
            f"future_{variable}",
            f"{variable}_future",
            f"future_delta_t_hr_{variable}",
            f"future_label_weight_{variable}",
        ):
            if pattern in frame and pattern not in columns:
                columns.append(pattern)
    view = frame[columns].copy()
    sort_columns = [
        column
        for column in ("subject_id", "stay_id", "encounter_id", "t", "anchor_time", "_source_horizon", "horizon_hours")
        if column in view
    ]
    if sort_columns:
        view = view.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    # Pandas' typed hashing is deterministic for the same values/dtypes. The
    # SHA-256 wrapper keeps the artifact compact and non-row-level.
    hashed = pd.util.hash_pandas_object(view, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


__all__ = ["cohort_signature"]
