"""Joint whole-body JEPA with one patient state and cross-system interactions.

The shared-module candidates share parameters, but their training rows are
still module-specific.  This module builds the next level: rows from different
body-system tables are aligned on the same ``stay_id + anchor time + horizon``
and merged into one masked body state.  A variable-token mixer lets observed
cardiac, respiratory, renal, metabolic, and hematologic values interact before
the future heads predict any target.

This is factual observation/forecasting only.  It does not infer treatment
effects and it remains candidate-only until the independent joint gate passes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from nonlinear_latent_rollout import (
    StateScaler,
    _history_indices,
    _time_hours,
    load_multihorizon_cohort,
    set_seed,
)
from whole_body_joint_source_registry import ATTRIBUTION_VALIDATED_GROUP_ADAPTERS, PHYSIOLOGY_GROUPS


def _event_module_variable_contracts(
    modules,
    available_variables,
    *,
    contract_mode="overlap",
):
    """Resolve organ contracts without reading legacy transition labels."""

    from eicu_aki_transition_extract import STATE_VARS as AKI_STATE_VARS
    from eicu_body_system_configs import BODY_SYSTEM_CONFIGS
    from eicu_dka_transition_extract import STATE_VARS as DKA_STATE_VARS
    from eicu_respiratory_transition_extract import (
        STATE_VARS as RESPIRATORY_STATE_VARS,
    )
    from eicu_sepsis_transition_extract import STATE_VARS as SEPSIS_STATE_VARS
    from osler_jepa.organ_schema import (
        assert_disjoint_contracts,
        canonical_organ_contracts,
    )

    if contract_mode == "canonical_organs":
        output = canonical_organ_contracts(available_variables)
        assert_disjoint_contracts(output)
        return output
    if contract_mode != "overlap":
        raise ValueError(
            "contract_mode must be 'overlap' or 'canonical_organs'"
        )

    special = {
        "aki": tuple(AKI_STATE_VARS),
        "dka": tuple(DKA_STATE_VARS),
        "respiratory": tuple(RESPIRATORY_STATE_VARS),
        "sepsis": tuple(SEPSIS_STATE_VARS),
    }
    available = set(available_variables)
    output = {}
    for module in modules:
        if module in BODY_SYSTEM_CONFIGS:
            candidates = BODY_SYSTEM_CONFIGS[module].state_vars
        else:
            candidates = special.get(module, ())
        selected = tuple(sorted(available.intersection(candidates)))
        if selected:
            output[module] = selected
    return output


def _multiscale_history_indices(
    frame,
    lags_hours=(48.0, 24.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0),
):
    """Return causal, fixed-width history views spanning fast and slow time."""

    lags = np.asarray(tuple(float(value) for value in lags_hours), dtype=np.float64)
    if len(lags) == 0 or np.any(lags < 0.0):
        raise ValueError("history lags must contain non-negative hours")
    # Oldest-to-newest is the natural order for the recurrent encoder.
    lags = np.sort(np.unique(lags))[::-1]
    output = np.zeros((len(frame), len(lags)), dtype=np.int64)
    times = frame["_time_hours"].to_numpy(dtype=np.float64)
    groups = frame.groupby(["_source_horizon", "stay_id"], sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        order = np.argsort(times[positions], kind="stable")
        positions = positions[order]
        group_times = times[positions]
        for local_index, position in enumerate(positions):
            anchor_time = group_times[local_index]
            selected = []
            for lag in lags:
                cutoff = anchor_time - lag
                candidate = int(
                    np.searchsorted(
                        group_times[: local_index + 1],
                        cutoff,
                        side="right",
                    )
                    - 1
                )
                selected.append(positions[max(candidate, 0)])
            output[position] = selected
    return output


def _precomputed_or_recent_history_indices(frame, history=5):
    """Use the loader's causal history view instead of silently rebuilding it."""

    history_columns = [
        column
        for column in frame.columns
        if column.startswith("_history_")
        and column[len("_history_") :].isdigit()
    ]
    history_columns.sort(key=lambda column: int(column[len("_history_") :]))
    if not history_columns:
        return _history_indices(frame, history=history)

    expected = [f"_history_{index}" for index in range(len(history_columns))]
    if history_columns != expected:
        raise ValueError(
            "Precomputed history columns must be contiguous from _history_0"
        )
    raw = frame[history_columns].apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any():
        raise ValueError("Precomputed history contains missing or non-numeric indices")
    history_indices = raw.to_numpy(dtype=np.int64)
    if not np.array_equal(
        history_indices.astype(np.float64),
        raw.to_numpy(dtype=np.float64),
    ):
        raise ValueError("Precomputed history indices must be integers")
    if (
        np.any(history_indices < 0)
        or np.any(history_indices >= len(frame))
    ):
        raise ValueError("Precomputed history index is outside the current frame")

    selected = history_indices.reshape(-1)
    repeated_rows = np.repeat(
        np.arange(len(frame), dtype=np.int64),
        history_indices.shape[1],
    )
    stay = frame["stay_id"].astype(str).to_numpy()
    horizon = pd.to_numeric(
        frame["_source_horizon"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    time = pd.to_numeric(
        frame["_time_hours"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if np.any(stay[selected] != stay[repeated_rows]):
        raise ValueError("Precomputed history crosses patient stays")
    if np.any(horizon[selected] != horizon[repeated_rows]):
        raise ValueError("Precomputed history crosses forecast horizons")
    if np.any(time[selected] > time[repeated_rows] + 1e-8):
        raise ValueError("Precomputed history contains future observations")
    return history_indices


def load_joint_event_examples(
    path,
    modules,
    horizons=(1, 3, 6, 12, 24, 48),
    max_stays=1000,
    min_modules=2,
    seed=2026,
    organ_contract_mode="overlap",
    history_mode="recent",
    history_steps=5,
    history_lags_hours=(48.0, 24.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0),
):
    """Load canonical-ledger examples as a whole-body joint training cohort."""

    frame = pd.read_parquet(Path(path))
    if frame.empty:
        raise ValueError("Canonical event examples are empty")
    if "t" not in frame and "anchor_time" in frame:
        frame["t"] = pd.to_datetime(frame["anchor_time"], errors="coerce")
    else:
        frame["t"] = pd.to_datetime(frame["t"], errors="coerce")
    if "_source_horizon" not in frame and "horizon_hours" in frame:
        frame["_source_horizon"] = pd.to_numeric(
            frame["horizon_hours"], errors="coerce"
        )
    frame = frame[
        pd.to_numeric(frame["_source_horizon"], errors="coerce").isin(
            [float(value) for value in horizons]
        )
    ].copy()
    if frame.empty:
        raise ValueError("No event examples match the requested horizons")
    if "subject_id" not in frame:
        frame["subject_id"] = frame.get("patient_id", "").astype(str)
    if "stay_id" not in frame:
        frame["stay_id"] = frame["encounter_id"].astype(str)
    if "hospitalid" not in frame:
        frame["hospitalid"] = frame.get("hospital_id", "__unknown_hospital__")
    if "careunit" not in frame:
        frame["careunit"] = frame.get("care_unit", "__unknown_careunit__")
    if "database" not in frame:
        frame["database"] = "__unknown_database__"

    variables = tuple(
        sorted(
            column[len("future_") :]
            for column in frame.columns
            if column.startswith("future_")
            and f"{column[len('future_') :]}_t" in frame.columns
            and not column.startswith(
                (
                    "future_event_time_",
                    "future_delta_t_hr_",
                    "future_label_quality_",
                    "future_label_weight_",
                )
            )
        )
    )
    module_variables = _event_module_variable_contracts(
        modules,
        variables,
        contract_mode=organ_contract_mode,
    )
    module_names = tuple(module_variables)
    if not module_names:
        raise ValueError("No requested module has variables in the event examples")

    stays = frame["stay_id"].astype(str).drop_duplicates().to_numpy()
    if max_stays is not None and len(stays) > int(max_stays):
        rng = np.random.default_rng(seed)
        stays = rng.choice(
            stays, size=int(max_stays), replace=False
        )
        frame = frame[frame["stay_id"].astype(str).isin(set(stays))].copy()

    presence_columns = []
    for module_index, module in enumerate(module_names):
        observed = np.zeros(len(frame), dtype=bool)
        for variable in module_variables[module]:
            observed |= pd.to_numeric(
                frame.get(
                    f"_observed_{variable}",
                    frame[f"{variable}_t"].notna().astype(float),
                ),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=np.float64) > 0.5
        column = f"_module_present_{module_index}"
        frame[column] = observed.astype(np.float32)
        frame[f"present__{module}"] = frame[column]
        presence_columns.append(column)
    frame["_module_count"] = frame[presence_columns].sum(axis=1).astype(np.int16)
    frame = frame[frame["_module_count"] >= int(min_modules)].copy()
    if frame.empty:
        raise ValueError("No event example contains enough observed organ modules")

    frame["hours_since_onset"] = (
        frame.groupby("stay_id")["t"].transform(
            lambda values: (
                values - values.min()
            ).dt.total_seconds() / 3600.0
        )
    )
    frame["onset_hour"] = 0.0
    frame["_time_hours"] = _time_hours(frame)
    frame.sort_values(
        ["_source_horizon", "stay_id", "_time_hours"],
        inplace=True,
    )
    frame.reset_index(drop=True, inplace=True)
    frame = _canonicalize_audit_provenance(frame)
    frame.attrs["module_variables"] = {
        module: list(module_variables[module]) for module in module_names
    }
    frame.attrs["treatment_history_columns"] = sorted(
        column for column in frame.columns if column.startswith("hist_")
    )
    frame.attrs["source_contract"] = "canonical_event_ledger"
    frame.attrs["organ_contract_mode"] = organ_contract_mode
    if history_mode == "multiscale":
        history = _multiscale_history_indices(
            frame,
            lags_hours=history_lags_hours,
        )
    elif history_mode == "recent":
        history = _history_indices(frame, history=history_steps)
    else:
        raise ValueError("history_mode must be 'recent' or 'multiscale'")
    frame.attrs["history_mode"] = history_mode
    frame.attrs["history_lags_hours"] = (
        [float(value) for value in history_lags_hours]
        if history_mode == "multiscale"
        else None
    )
    for position in range(history.shape[1]):
        frame[f"_history_{position}"] = history[:, position]
    return frame, variables, module_names


def load_joint_cohort(
    data_root,
    modules,
    horizons=(1, 3, 6, 12, 24, 48),
    max_stays_per_horizon=1000,
    min_modules=2,
    seed=2026,
    alignment="asof",
    asof_max_age_hours=6.0,
    future_label_tolerance_hours=0.0,
    asof_age_policy="horizon_scaled",
    variable_age_limits: Mapping[str, float] | None = None,
    include_treatment_context=False,
):
    """Align module rows into a masked patient-time body state.

    ``exact`` preserves the original conservative contract: only observations
    with the same timestamp are merged.  ``asof`` is the production research
    path: at each anchor it uses the latest observation from each module at or
    before that anchor, with an explicit age limit.  Future labels are still
    accepted only from a transition whose own anchor is exactly the shared
    anchor.  This separates valid asynchronous *inputs* from invalid shifted
    future labels.
    """
    root = Path(data_root)
    module_stays = {}
    for module in modules:
        stays = set()
        for horizon in horizons:
            path = root / f"eicu_{module}_transitions_{int(horizon)}h.parquet"
            if path.exists():
                stays.update(pd.read_parquet(path, columns=["stay_id"])["stay_id"].astype(str).tolist())
        if stays:
            module_stays[module] = stays
    stay_counts = {}
    for stays in module_stays.values():
        for stay in stays:
            stay_counts[stay] = stay_counts.get(stay, 0) + 1
    eligible_stays = sorted(stay for stay, count in stay_counts.items() if count >= int(min_modules))
    if not eligible_stays:
        raise FileNotFoundError("No patient stays are represented in enough modules for joint alignment")
    if max_stays_per_horizon is not None and len(eligible_stays) > int(max_stays_per_horizon):
        rng = np.random.default_rng(seed)
        eligible_stays = sorted(
            rng.choice(np.asarray(eligible_stays, dtype=object), size=int(max_stays_per_horizon), replace=False).tolist()
        )
    allowed_stays = set(eligible_stays)
    parts = []
    module_names = []
    module_variables = {}
    all_variables = set()
    for module in modules:
        try:
            frame, variables = load_multihorizon_cohort(
                data_root,
                module,
                horizons,
                max_stays_per_horizon=None,
                history=5,
                seed=seed,
                allowed_stays=allowed_stays,
                include_treatment_context=include_treatment_context,
            )
        except FileNotFoundError:
            continue
        module_index = len(module_names)
        module_names.append(module)
        module_variables[module] = tuple(sorted(variables))
        all_variables.update(variables)
        presence = f"_module_present_{module_index}"
        frame = frame.copy()
        frame[presence] = 1.0
        keys = [
            "subject_id",
            "stay_id",
            "hospitalid",
            "database",
            "unittype",
            "careunit",
            "care_unit",
            "first_careunit",
            "last_careunit",
            "t",
            "onset_hour",
            "hours_since_onset",
            "_source_horizon",
            "_time_hours",
            presence,
        ]
        for variable in sorted(variables):
            keys.extend(
                [
                    f"{variable}_t",
                    f"{variable}_age_hr",
                    f"future_{variable}",
                    f"future_event_time_hr_{variable}",
                    f"future_delta_t_hr_{variable}",
                    f"future_label_quality_{variable}",
                    f"future_label_weight_{variable}",
                    f"next_measurement_time_hr_{variable}",
                ]
            )
        if include_treatment_context:
            # Only past/anchor-time evidence is eligible as an input.  The
            # corresponding act_* columns describe the future forecast window
            # and must never cross the context/target boundary.
            keys.extend(
                column for column in frame.columns if column.startswith("hist_")
            )
        keys = [column for column in dict.fromkeys(keys) if column in frame.columns]
        selected = frame[keys].copy()
        selected["_row_time_hr"] = _row_timestamp_hours(selected)
        parts.append(selected)
    if not parts:
        raise FileNotFoundError("No module transition cohorts available")

    variables = tuple(sorted(all_variables))
    if alignment == "asof":
        combined = _asof_align_parts(
            parts,
            module_names,
            module_variables,
            variables,
            max_age_hours=float(asof_max_age_hours),
            future_label_tolerance_hours=float(future_label_tolerance_hours),
            age_policy=str(asof_age_policy),
            variable_age_limits=variable_age_limits,
            min_modules=int(min_modules),
        )
        combined.sort_values(["_source_horizon", "stay_id", "_time_hours"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined.attrs["module_variables"] = {
            name: list(module_variables.get(name, ())) for name in module_names
        }
        combined.attrs["treatment_history_columns"] = sorted(
            column for column in combined.columns if column.startswith("hist_")
        )
        combined = _canonicalize_audit_provenance(combined)
        history = _history_indices(combined, history=5)
        for position in range(history.shape[1]):
            combined[f"_history_{position}"] = history[:, position]
        return combined, variables, tuple(module_names)
    if alignment != "exact":
        raise ValueError(f"Unsupported joint alignment mode: {alignment}")

    # Defragment before the wide groupby; each module contributes a different
    # subset of state/future columns.
    combined = pd.concat(parts, ignore_index=True, sort=False).copy()
    presence_columns = [f"_module_present_{index}" for index in range(len(module_names))]
    group_keys = ["stay_id", "_source_horizon", "t"]
    # first() takes the first non-null observation when the same patient/time
    # appears in multiple module tables.  This preserves measurement masks and
    # avoids inventing a value by averaging across extractors.
    combined = combined.groupby(group_keys, sort=False, dropna=False).first().reset_index()
    for column in ["subject_id", "hospitalid", "onset_hour", "hours_since_onset"] + presence_columns:
        if column not in combined.columns:
            combined[column] = 0.0 if column in presence_columns else np.nan
    combined[presence_columns] = combined[presence_columns].fillna(0.0).astype(np.float32)
    combined["_module_count"] = combined[presence_columns].sum(axis=1).astype(np.int16)
    combined = combined[combined["_module_count"] >= int(min_modules)].copy()

    for variable in variables:
        for suffix in ("_t", "_age_hr"):
            column = f"{variable}{suffix}"
            if column not in combined.columns:
                combined[column] = np.nan
        next_measurement = f"next_measurement_time_hr_{variable}"
        if next_measurement not in combined.columns:
            combined[next_measurement] = np.nan
        future = f"future_{variable}"
        if future not in combined.columns:
            combined[future] = np.nan
        for prefix, default in (
            ("future_delta_t_hr_", np.nan),
            ("future_label_weight_", np.nan),
            ("future_event_time_hr_", np.nan),
        ):
            column = f"{prefix}{variable}"
            if column not in combined.columns:
                combined[column] = default
    combined["_time_hours"] = _time_hours(combined)
    combined.sort_values(["_source_horizon", "stay_id", "_time_hours"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.attrs["module_variables"] = {
        name: list(module_variables.get(name, ())) for name in module_names
    }
    combined.attrs["treatment_history_columns"] = sorted(
        column for column in combined.columns if column.startswith("hist_")
    )
    combined = _canonicalize_audit_provenance(combined)
    history = _history_indices(combined, history=5)
    for position in range(history.shape[1]):
        combined[f"_history_{position}"] = history[:, position]
    return combined, variables, tuple(module_names)


def _canonicalize_audit_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve database/care-unit metadata without making them model inputs."""

    frame = frame.copy()

    def coalesce(columns, default):
        output = pd.Series(pd.NA, index=frame.index, dtype="string")
        for column in columns:
            if column not in frame:
                continue
            values = frame[column].astype("string").str.strip()
            usable = values.notna() & (values != "") & (values.str.lower() != "nan")
            output = output.where(output.notna(), values.where(usable))
        return output.fillna(default).astype(str)

    frame["careunit"] = coalesce(
        ("careunit", "care_unit", "first_careunit", "unittype", "last_careunit"),
        "__unknown_careunit__",
    )
    frame["database"] = coalesce(
        ("database",),
        "__unknown_database__",
    )
    frame.attrs["audit_provenance_columns"] = [
        "database",
        "hospitalid",
        "careunit",
    ]
    frame.attrs["physiology_encoder_excludes"] = [
        "database",
        "hospitalid",
        "careunit",
    ]
    return frame


def _row_timestamp_hours(frame: pd.DataFrame) -> np.ndarray:
    """Return absolute timestamps in hours for safe as-of joins."""

    parsed = pd.to_datetime(frame["t"], errors="coerce")
    numeric = parsed.astype("int64", copy=False).to_numpy(dtype=np.float64)
    numeric[parsed.isna().to_numpy()] = np.nan
    return numeric / 3.6e12


def _asof_align_parts(
    parts,
    module_names,
    module_variables,
    variables,
    max_age_hours,
    future_label_tolerance_hours,
    min_modules,
    age_policy="horizon_scaled",
    variable_age_limits=None,
):
    """Build a causal asynchronous whole-body state.

    The row anchors are the union of observed transition anchors.  For each
    module, ``merge_asof(..., direction='backward')`` adds only the most recent
    row at or before the anchor.  The original measurement age is increased by
    the time from that row to the anchor. Future labels use a separate nearest
    join so a bounded horizon window can be evaluated without contaminating
    the input state with a future observation.
    """

    treatment_columns = tuple(
        sorted(
            {
                column
                for part in parts
                for column in part.columns
                if str(column).startswith("hist_")
            }
        )
    )

    anchor_parts = []
    for part in parts:
        columns = [
            column
            for column in [
                "subject_id",
                "stay_id",
                "hospitalid",
                "database",
                "unittype",
                "careunit",
                "care_unit",
                "first_careunit",
                "last_careunit",
                "t",
                "onset_hour",
                "hours_since_onset",
                "_source_horizon",
                "_row_time_hr",
            ]
            if column in part.columns
        ]
        anchor_parts.append(part[columns].copy())
    anchors = pd.concat(anchor_parts, ignore_index=True, sort=False)
    anchors = anchors[np.isfinite(anchors["_row_time_hr"].to_numpy(dtype=np.float64))].copy()
    anchors.sort_values(["_source_horizon", "stay_id", "_row_time_hr"], inplace=True)
    anchors = anchors.drop_duplicates(
        ["stay_id", "_source_horizon", "_row_time_hr"], keep="first"
    ).reset_index(drop=True)
    anchors["_anchor_id"] = np.arange(len(anchors), dtype=np.int64)
    anchors["_anchor_time_hr"] = anchors["_row_time_hr"].astype(np.float64)

    initial_columns = {}
    for variable in variables:
        initial_columns[f"{variable}_t"] = np.full(len(anchors), np.nan, dtype=np.float64)
        initial_columns[f"{variable}_age_hr"] = np.full(len(anchors), np.nan, dtype=np.float64)
        # The same laboratory/vital can occur in several module tables.  Keep
        # the source of the freshest causal observation so source ablations do
        # not mistake duplicated columns for an independent organ signal.
        initial_columns[f"_source_module_{variable}"] = np.full(
            len(anchors), -1, dtype=np.int16
        )
        initial_columns[f"future_{variable}"] = np.full(len(anchors), np.nan, dtype=np.float64)
        initial_columns[f"future_delta_t_hr_{variable}"] = np.full(len(anchors), np.nan, dtype=np.float64)
        initial_columns[f"future_event_time_hr_{variable}"] = np.full(len(anchors), np.nan, dtype=np.float64)
        initial_columns[f"future_label_weight_{variable}"] = np.full(len(anchors), np.nan, dtype=np.float64)
        initial_columns[f"next_measurement_time_hr_{variable}"] = np.full(
            len(anchors), np.nan, dtype=np.float64
        )
    for module_index in range(len(module_names)):
        initial_columns[f"_module_present_{module_index}"] = np.zeros(
            len(anchors), dtype=np.float32
        )
    for column in treatment_columns:
        initial_columns[column] = np.zeros(len(anchors), dtype=np.float32)
        initial_columns[f"_treatment_observed_{column}"] = np.zeros(
            len(anchors), dtype=np.float32
        )
        initial_columns[f"_treatment_age_hr_{column}"] = np.full(
            len(anchors), np.nan, dtype=np.float32
        )
    initial_columns["_module_count"] = np.zeros(len(anchors), dtype=np.int16)
    combined = pd.concat(
        [anchors.reset_index(drop=True), pd.DataFrame(initial_columns)], axis=1
    )
    for column in ["subject_id", "hospitalid", "onset_hour", "hours_since_onset"]:
        if column not in combined.columns:
            combined[column] = np.nan

    for module_index, (module, part) in enumerate(zip(module_names, parts)):
        module_vars = module_variables[module]
        part = part[np.isfinite(part["_row_time_hr"].to_numpy(dtype=np.float64))].copy()
        right_columns = ["stay_id", "_source_horizon", "_row_time_hr"]
        right_columns += [f"{variable}_t" for variable in module_vars]
        right_columns += [f"{variable}_age_hr" for variable in module_vars]
        right_columns += [f"future_{variable}" for variable in module_vars]
        right_columns += [f"future_delta_t_hr_{variable}" for variable in module_vars]
        right_columns += [f"future_event_time_hr_{variable}" for variable in module_vars]
        right_columns += [f"future_label_weight_{variable}" for variable in module_vars]
        right_columns += [f"next_measurement_time_hr_{variable}" for variable in module_vars]
        right_columns += [column for column in treatment_columns if column in part.columns]
        right_columns = [column for column in right_columns if column in part.columns]
        right = part[right_columns].copy()
        right.sort_values(["_row_time_hr", "_source_horizon", "stay_id"], inplace=True)
        left_columns = [
            "_anchor_id",
            "_anchor_time_hr",
            "stay_id",
            "_source_horizon",
        ]
        left = anchors[left_columns].copy()
        left.sort_values(["_anchor_time_hr", "_source_horizon", "stay_id"], inplace=True)
        matched = pd.merge_asof(
            left,
            right,
            left_on="_anchor_time_hr",
            right_on="_row_time_hr",
            by=["_source_horizon", "stay_id"],
            direction="backward",
            tolerance=float(max_age_hours),
        )
        matched.sort_values("_anchor_id", inplace=True)
        matched.reset_index(drop=True, inplace=True)
        label_left = anchors[left_columns].copy()
        label_left.sort_values(
            ["_anchor_time_hr", "_source_horizon", "stay_id"], inplace=True
        )
        label_match = pd.merge_asof(
            label_left,
            right,
            left_on="_anchor_time_hr",
            right_on="_row_time_hr",
            by=["_source_horizon", "stay_id"],
            direction="nearest",
            tolerance=float(future_label_tolerance_hours),
        )
        label_match.sort_values("_anchor_id", inplace=True)
        label_match.reset_index(drop=True, inplace=True)
        anchor_time = anchors["_anchor_time_hr"].to_numpy(dtype=np.float64)
        source_time = pd.to_numeric(matched["_row_time_hr"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        present = np.isfinite(source_time)
        delta = np.where(present, np.maximum(anchor_time - source_time, 0.0), np.nan)
        horizon_values = pd.to_numeric(
            anchors["_source_horizon"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        combined[f"_module_present_{module_index}"] = np.zeros(
            len(anchors), dtype=np.float32
        )
        label_time = pd.to_numeric(label_match["_row_time_hr"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        label_allowed = np.isfinite(label_time) & (
            np.abs(anchor_time - label_time)
            <= float(future_label_tolerance_hours) + 1e-6
        )

        module_usable = np.zeros(len(anchors), dtype=bool)
        for variable in module_vars:
            source = pd.to_numeric(
                matched.get(f"{variable}_t", pd.Series(np.nan, index=matched.index)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            source_age = pd.to_numeric(
                matched.get(f"{variable}_age_hr", pd.Series(np.nan, index=matched.index)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            age = np.where(np.isfinite(source), np.nan_to_num(source_age, nan=0.0) + delta, np.nan)
            if variable_age_limits and variable in variable_age_limits:
                allowed_age = np.full(
                    len(anchors), float(variable_age_limits[variable]), dtype=np.float64
                )
            elif str(age_policy) == "fixed":
                allowed_age = np.full(len(anchors), float(max_age_hours), dtype=np.float64)
            elif str(age_policy) == "horizon_scaled":
                # A 1h target must not be driven by a 6h-old measurement.
                # Longer horizons may use older observations, but never beyond
                # the configured cap.
                allowed_age = np.minimum(
                    float(max_age_hours), np.maximum(0.5, horizon_values / 2.0)
                )
            else:
                raise ValueError(f"Unsupported as-of age policy: {age_policy}")
            age_valid = np.isfinite(age) & (age <= allowed_age + 1e-6)
            source = np.where(age_valid, source, np.nan)
            age = np.where(age_valid, age, np.nan)
            module_usable |= np.isfinite(source)
            existing = combined[f"{variable}_t"].to_numpy(dtype=np.float64).copy()
            existing_age = combined[f"{variable}_age_hr"].to_numpy(dtype=np.float64).copy()
            existing_source_module = combined[
                f"_source_module_{variable}"
            ].to_numpy(dtype=np.int16).copy()
            replace = np.isfinite(source) & (
                ~np.isfinite(existing) | ~np.isfinite(existing_age) | (age < existing_age)
            )
            if replace.any():
                existing[replace] = source[replace]
                existing_age[replace] = age[replace]
                existing_source_module[replace] = np.int16(module_index)
                combined[f"{variable}_t"] = existing
                combined[f"{variable}_age_hr"] = existing_age
                combined[f"_source_module_{variable}"] = existing_source_module

            # The raw label is relative to the matched source row.  When an
            # older as-of observation is carried forward, rebase its absolute
            # event time to the current anchor; never copy the stale relative
            # delay unchanged.
            source_next_time = pd.to_numeric(
                matched.get(
                    f"next_measurement_time_hr_{variable}",
                    pd.Series(np.nan, index=matched.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            rebased_next_time = source_time + source_next_time - anchor_time
            rebased_next_time = np.where(
                present
                & np.isfinite(source_next_time)
                & (source_next_time >= 0.0)
                & (rebased_next_time >= 0.0),
                rebased_next_time,
                np.nan,
            )
            existing_next_time = combined[
                f"next_measurement_time_hr_{variable}"
            ].to_numpy(dtype=np.float64).copy()
            replace_next_time = np.isfinite(rebased_next_time) & (
                ~np.isfinite(existing_next_time)
                | (rebased_next_time < existing_next_time)
            )
            if replace_next_time.any():
                existing_next_time[replace_next_time] = rebased_next_time[
                    replace_next_time
                ]
                combined[f"next_measurement_time_hr_{variable}"] = existing_next_time

            future = pd.to_numeric(
                label_match.get(
                    f"future_{variable}", pd.Series(np.nan, index=label_match.index)
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            future = np.where(label_allowed, future, np.nan)
            existing_future = combined[f"future_{variable}"].to_numpy(dtype=np.float64).copy()
            replace_future = np.isfinite(future) & ~np.isfinite(existing_future)
            if replace_future.any():
                existing_future[replace_future] = future[replace_future]
                combined[f"future_{variable}"] = existing_future

            actual_delta = pd.to_numeric(
                label_match.get(
                    f"future_delta_t_hr_{variable}",
                    pd.Series(np.nan, index=label_match.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            actual_time = pd.to_numeric(
                label_match.get(
                    f"future_event_time_hr_{variable}",
                    pd.Series(np.nan, index=label_match.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            label_weight = pd.to_numeric(
                label_match.get(
                    f"future_label_weight_{variable}",
                    pd.Series(np.nan, index=label_match.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            actual_delta = np.where(label_allowed, actual_delta, np.nan)
            actual_time = np.where(label_allowed, actual_time, np.nan)
            label_weight = np.where(label_allowed, label_weight, np.nan)
            existing_delta = combined[f"future_delta_t_hr_{variable}"].to_numpy(dtype=np.float64).copy()
            existing_time = combined[f"future_event_time_hr_{variable}"].to_numpy(dtype=np.float64).copy()
            existing_weight = combined[f"future_label_weight_{variable}"].to_numpy(dtype=np.float64).copy()
            replace_meta = np.isfinite(future) & ~np.isfinite(existing_future)
            # If a module has already supplied the label, keep that raw event
            # metadata; otherwise copy the matched module row as a unit.
            replace_meta |= np.isfinite(future) & np.isfinite(existing_future) & ~np.isfinite(existing_delta)
            if replace_meta.any():
                existing_delta[replace_meta] = actual_delta[replace_meta]
                existing_time[replace_meta] = actual_time[replace_meta]
                existing_weight[replace_meta] = label_weight[replace_meta]
                combined[f"future_delta_t_hr_{variable}"] = existing_delta
                combined[f"future_event_time_hr_{variable}"] = existing_time
                combined[f"future_label_weight_{variable}"] = existing_weight

        # History treatment features are joined as-of like other context, but
        # are never read from act_* future-window fields. Several disease
        # extractors can report the same treatment; retain the strongest
        # currently observed history evidence rather than allowing a missing
        # extractor's zero to erase another module's positive record.
        treatment_valid = present & (delta <= float(max_age_hours) + 1e-6)
        for column in treatment_columns:
            source = pd.to_numeric(
                matched.get(column, pd.Series(np.nan, index=matched.index)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            valid = treatment_valid & np.isfinite(source)
            if not valid.any():
                continue
            existing = combined[column].to_numpy(dtype=np.float32).copy()
            observed_column = f"_treatment_observed_{column}"
            existing_observed = combined[observed_column].to_numpy(dtype=np.float32).copy()
            age_column = f"_treatment_age_hr_{column}"
            existing_age = combined[age_column].to_numpy(dtype=np.float32).copy()
            # Treatment evidence is binary or a count. Both have a natural
            # monotone union: retain the greatest causal observation.
            replace = valid & (
                (existing_observed <= 0.0)
                | (source > existing)
                | ((source == existing) & (delta < existing_age))
            )
            existing[replace] = source[replace].astype(np.float32)
            existing_observed[valid] = 1.0
            existing_age[replace] = delta[replace].astype(np.float32)
            combined[column] = existing
            combined[observed_column] = existing_observed
            combined[age_column] = existing_age

        combined[f"_module_present_{module_index}"] = module_usable.astype(np.float32)

    presence_columns = [f"_module_present_{index}" for index in range(len(module_names))]
    combined[presence_columns] = combined[presence_columns].fillna(0.0).astype(np.float32)
    combined["_module_count"] = combined[presence_columns].sum(axis=1).astype(np.int16)
    combined = combined[combined["_module_count"] >= int(min_modules)].copy()
    combined["_time_hours"] = _time_hours(combined)
    combined.drop(columns=["_anchor_id", "_anchor_time_hr", "_row_time_hr"], inplace=True)
    combined.attrs["treatment_history_columns"] = list(treatment_columns)
    return combined


def _joint_arrays(
    frame,
    variables,
    module_names,
    fit_rows,
    history=5,
    measurement_process_mode="observed",
    state_normalization="standard",
    include_treatment_context=False,
):
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    scaler = StateScaler.fit(frame, variables, fit_rows, method=state_normalization)
    current, mask, ages = scaler.normalize_current(frame)
    future, future_mask = scaler.normalize_future(frame)
    source_horizon = pd.to_numeric(
        frame.get("_source_horizon"), errors="coerce"
    ).to_numpy(dtype=np.float32)
    future_delta_t = np.repeat(source_horizon[:, None], len(variables), axis=1)
    future_label_weight = np.ones_like(future_delta_t, dtype=np.float32)
    for index, variable in enumerate(variables):
        delta_column = f"future_delta_t_hr_{variable}"
        weight_column = f"future_label_weight_{variable}"
        if delta_column in frame.columns:
            delta = pd.to_numeric(frame[delta_column], errors="coerce").to_numpy(dtype=np.float32)
            future_delta_t[:, index] = np.where(
                np.isfinite(delta) & (delta > 0.0), delta, future_delta_t[:, index]
            )
        if weight_column in frame.columns:
            weight = pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(dtype=np.float32)
            # Missing provenance on legacy rows means “retain legacy label”;
            # explicit zero means the raw event label failed the strict gate.
            future_label_weight[:, index] = np.where(
                np.isfinite(weight), np.clip(weight, 0.0, 1.0), 1.0
            )
    # Exclude near/missing raw labels from the supervised target while keeping
    # their coverage available to the audit.  This makes the future task a
    # consistent physiological time series instead of a mixed-quality target.
    # A future value is a supervised delta only when the same target was also
    # observed at the anchor.  Otherwise the normalized median would hide an
    # unpaired endpoint and turn an unknown change into a valid-looking label.
    future_mask = future_mask * mask * future_label_weight
    window_horizons = np.asarray(
        sorted(set(pd.to_numeric(frame["_source_horizon"], errors="coerce").dropna().tolist())),
        dtype=np.float32,
    )
    future_window, future_window_mask, future_window_delta_t = _future_window_arrays(
        frame,
        future,
        future_mask,
        future_delta_t,
        window_horizons,
    )
    onset = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=np.float32)
    onset = np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0)
    onset = np.clip(onset, 0.0, 168.0)[:, None] / 24.0
    presence = frame[[f"_module_present_{i}" for i in range(len(module_names))]].to_numpy(dtype=np.float32)
    # A delta target is only identifiable when both ends of the delta were
    # actually measured.  StateScaler replaces missing values with the fit
    # median (zero in normalized space), so using future_mask alone would turn
    # "anchor missing, future observed" into a false zero-change label.
    future_window_mask = future_window_mask * mask[:, None, :]
    if measurement_process_mode == "neutralized":
        # Keep the physiological values while removing the hospital/ward
        # measurement schedule from the model input.  Missing normalized
        # values are replaced by the fit-set median (zero in normalized space).
        current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.ones_like(mask, dtype=np.float32)
        ages = np.zeros_like(ages, dtype=np.float32)
        presence = np.ones_like(presence, dtype=np.float32)
    elif measurement_process_mode != "observed":
        raise ValueError(
            "measurement_process_mode must be 'observed' or 'neutralized'"
        )
    row_time = pd.to_numeric(frame["_time_hours"], errors="coerce").to_numpy(dtype=np.float32)
    row_time = np.nan_to_num(row_time, nan=0.0, posinf=0.0, neginf=0.0)
    time_origin = frame.groupby("stay_id")["_time_hours"].transform("min").to_numpy(dtype=np.float32)
    time_feature = np.clip(row_time - time_origin, 0.0, 168.0)[:, None] / 168.0
    # Measurement-process labels are kept separate from physiological future
    # labels.  They are used only by the optional next-measurement-time head.
    next_measurement_time = np.full(
        (len(frame), len(variables)), np.nan, dtype=np.float32
    )
    for index, variable in enumerate(variables):
        column = f"next_measurement_time_hr_{variable}"
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
            next_measurement_time[:, index] = np.where(
                np.isfinite(values) & (values >= 0.0),
                np.clip(values, 0.0, 168.0),
                np.nan,
            )
    next_measurement_mask = np.isfinite(next_measurement_time).astype(np.float32)
    input_parts = [current, mask, ages, onset, presence, time_feature]
    if include_treatment_context:
        treatment_columns = tuple(
            frame.attrs.get(
                "treatment_history_columns",
                sorted(column for column in frame.columns if column.startswith("hist_")),
            )
        )
        if treatment_columns:
            raw_treatment = np.column_stack(
                [
                    pd.to_numeric(
                        frame.get(column, pd.Series(0.0, index=frame.index)),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                    for column in treatment_columns
                ]
            )
            # Binary history flags and evidence counts share this compact
            # representation. Counts are compressed before fit-only
            # standardization so a few repeated chart entries cannot dominate
            # the physiological state encoder.
            treatment = np.log1p(np.clip(raw_treatment, 0.0, None))
            fit_treatment = treatment[fit_rows]
            center = np.median(fit_treatment, axis=0, keepdims=True)
            scale = np.std(fit_treatment, axis=0, keepdims=True)
            treatment = (treatment - center) / np.maximum(scale, 1e-3)
            treatment_observed = np.column_stack(
                [
                    pd.to_numeric(
                        frame.get(
                            f"_treatment_observed_{column}",
                            pd.Series(0.0, index=frame.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                    for column in treatment_columns
                ]
            )
            treatment_age = np.column_stack(
                [
                    pd.to_numeric(
                        frame.get(
                            f"_treatment_age_hr_{column}",
                            pd.Series(np.nan, index=frame.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(72.0)
                    .clip(0.0, 72.0)
                    .to_numpy(dtype=np.float32)
                    / 72.0
                    for column in treatment_columns
                ]
            )
            input_parts.extend([treatment, treatment_observed, treatment_age])
    input_matrix = np.concatenate(input_parts, axis=1).astype(np.float32)
    history_indices = _precomputed_or_recent_history_indices(
        frame,
        history=history,
    )
    horizon = frame["_source_horizon"].to_numpy(dtype=np.float32)
    return (
        scaler,
        input_matrix,
        current,
        mask,
        ages,
        future,
        future_mask,
        history_indices,
        horizon,
        presence,
        future_window,
        future_window_mask,
        future_delta_t,
        future_window_delta_t,
        window_horizons,
        next_measurement_time,
        next_measurement_mask,
    )


class _GradientReversal(torch.autograd.Function):
    """Reverse encoder gradients while training a domain classifier."""

    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value, coefficient=1.0):
    return _GradientReversal.apply(value, coefficient)


def _future_window_arrays(frame, future, future_mask, future_delta_t, window_horizons):
    """Assemble causal future-label prefixes for the JEPA target encoder.

    Rows from separate horizon transition files are joined only when they
    share the same stay and anchor timestamp.  For a row at horizon ``h``,
    labels beyond ``h`` are masked out, so the target encoder never receives a
    future label from a later horizon than the task being evaluated.
    """

    window_horizons = np.asarray(window_horizons, dtype=np.float32)
    count = len(frame)
    width = len(window_horizons)
    n_state = future.shape[1]
    values = np.zeros((count, width, n_state), dtype=np.float32)
    masks = np.zeros((count, width, n_state), dtype=np.float32)
    actual_deltas = np.zeros((count, width, n_state), dtype=np.float32)
    times = pd.to_numeric(frame.get("_time_hours"), errors="coerce").to_numpy(dtype=np.float64)
    source_horizons = pd.to_numeric(
        frame.get("_source_horizon"), errors="coerce"
    ).to_numpy(dtype=np.float64)
    groups = {}
    for row, (stay, time, source_horizon) in enumerate(
        zip(frame["stay_id"].astype(str), times, source_horizons)
    ):
        if not np.isfinite(time) or not np.isfinite(source_horizon):
            continue
        groups.setdefault((stay, round(float(time), 6)), {})[
            int(round(float(source_horizon)))
        ] = row
    for row, (stay, time, source_horizon) in enumerate(
        zip(frame["stay_id"].astype(str), times, source_horizons)
    ):
        if not np.isfinite(time) or not np.isfinite(source_horizon):
            continue
        matches = groups.get((stay, round(float(time), 6)), {})
        for position, target_horizon in enumerate(window_horizons):
            if float(target_horizon) > float(source_horizon) + 1e-6:
                continue
            matched_row = matches.get(int(round(float(target_horizon))))
            if matched_row is None:
                continue
            values[row, position] = future[matched_row]
            masks[row, position] = future_mask[matched_row]
            actual_deltas[row, position] = future_delta_t[matched_row]
            fallback = float(target_horizon)
            actual_deltas[row, position] = np.where(
                np.isfinite(actual_deltas[row, position])
                & (actual_deltas[row, position] > 0.0),
                actual_deltas[row, position],
                fallback,
            )
    return values, masks, actual_deltas


class NeuralBeliefFilter(nn.Module):
    """A small predict-update filter for patient-specific history state."""

    def __init__(self, input_dim, hidden):
        super().__init__()
        self.hidden = int(hidden)
        self.observation_projection = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.innovation_projection = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.update_gate = nn.Linear(2 * hidden, hidden)
        self.decay_rate = nn.Parameter(torch.full((hidden,), -1.0))
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.SiLU(),
            nn.Linear(max(8, hidden // 2), hidden),
        )

    def forward(self, observations):
        batch_size, steps, _ = observations.shape
        time = observations[:, :, -1:].detach()
        belief = observations.new_zeros((batch_size, self.hidden))
        previous_time = time[:, 0]
        for step in range(steps):
            current_time = time[:, step]
            delta = (current_time - previous_time).abs().clamp(0.0, 1.0)
            decay = torch.exp(-nn.functional.softplus(self.decay_rate) * delta)
            predicted = belief * decay
            observation = self.observation_projection(observations[:, step])
            combined = torch.cat([predicted, observation], dim=-1)
            innovation = torch.tanh(self.innovation_projection(combined))
            gate = torch.sigmoid(self.update_gate(combined))
            belief = predicted + gate * (innovation - predicted)
            previous_time = current_time
        uncertainty = nn.functional.softplus(self.uncertainty_head(belief)) + 1e-4
        return belief, uncertainty


class JointWholeBodyJEPA(nn.Module):
    """One body latent with explicit cross-variable token interaction."""

    def __init__(
        self,
        n_state,
        input_dim,
        module_count,
        hidden=96,
        latent=64,
        token_dim=32,
        variable_module_ids=None,
        variable_module_membership=None,
        horizon_values=None,
        regime_count=3,
        target_horizon_regime_adapter=False,
        patient_residual_adapter=False,
        uncertainty_gate=False,
        validated_edge_adapters=False,
        validated_edge_specs=None,
        adapter_only_treatment_context=False,
        measurement_time_head=False,
        belief_input_mode="observed",
    ):
        super().__init__()
        self.n_state = int(n_state)
        self.input_dim = int(input_dim)
        self.module_count = int(module_count)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.token_dim = int(token_dim)
        self.regime_count = max(2, int(regime_count))
        if horizon_values is None:
            horizon_values = (1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
        horizon_values = np.asarray(horizon_values, dtype=np.float32).reshape(-1)
        if horizon_values.size == 0 or not np.isfinite(horizon_values).all():
            raise ValueError("horizon_values must contain finite values")
        self.target_horizon_regime_adapter_enabled = bool(
            target_horizon_regime_adapter
        )
        self.patient_residual_adapter_enabled = bool(patient_residual_adapter)
        self.uncertainty_gate_enabled = bool(uncertainty_gate)
        self.validated_edge_adapters_enabled = bool(validated_edge_adapters)
        self.validated_edge_specs = tuple(validated_edge_specs or ())
        self.adapter_only_treatment_context = bool(adapter_only_treatment_context)
        adapter_treatment_indices = sorted(
            {
                int(index)
                for spec in self.validated_edge_specs
                for index in spec.get("treatment_context_indices", ())
            }
        )
        if any(index < 0 or index >= self.input_dim for index in adapter_treatment_indices):
            raise ValueError("treatment context index is outside the model input")
        self.register_buffer(
            "adapter_treatment_context_indices",
            torch.as_tensor(adapter_treatment_indices, dtype=torch.long),
            persistent=False,
        )
        self.measurement_time_head = bool(measurement_time_head)
        if belief_input_mode not in {
            "observed",
            "current_only",
            "placebo_time_only",
        }:
            raise ValueError(
                "belief_input_mode must be observed, current_only, or "
                "placebo_time_only"
            )
        self.belief_input_mode = str(belief_input_mode)
        self.register_buffer(
            "horizon_bucket_values",
            torch.as_tensor(horizon_values, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "regime_cutpoints",
            torch.zeros((n_state, self.regime_count - 1), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "regime_cutpoint_valid",
            torch.zeros((n_state, self.regime_count - 1), dtype=torch.float32),
            persistent=True,
        )
        self.belief_filter = NeuralBeliefFilter(input_dim, hidden)
        self.variable_embedding = nn.Parameter(torch.randn(n_state, token_dim) * 0.02)
        if variable_module_ids is None:
            variable_module_ids = np.zeros(n_state, dtype=np.int64)
        self.register_buffer(
            "variable_module_ids",
            torch.as_tensor(variable_module_ids, dtype=torch.long),
            persistent=True,
        )
        if variable_module_membership is None:
            variable_module_membership = np.zeros((module_count, n_state), dtype=np.float32)
            for index, module_id in enumerate(variable_module_ids):
                variable_module_membership[int(module_id), index] = 1.0
        membership = np.asarray(variable_module_membership, dtype=np.float32)
        if membership.shape != (module_count, n_state):
            raise ValueError(
                "variable_module_membership must have shape "
                f"({module_count}, {n_state}), got {membership.shape}"
            )
        self.register_buffer(
            "variable_module_membership",
            torch.as_tensor(membership, dtype=torch.float32),
            persistent=True,
        )
        self.module_token_embedding = nn.Embedding(max(1, module_count), token_dim)
        self.token_projection = nn.Sequential(nn.Linear(3, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        variable_encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=token_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.variable_mixer = nn.TransformerEncoder(variable_encoder_layer, num_layers=2)
        module_encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=token_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.module_mixer = nn.TransformerEncoder(module_encoder_layer, num_layers=2)
        self.presence_projection = nn.Sequential(
            nn.Linear(module_count, 16), nn.LayerNorm(16), nn.SiLU()
        )
        self.body_pool_projection = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU()
        )
        state_input = hidden + token_dim + (3 * n_state) + 16
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input, latent), nn.LayerNorm(latent), nn.SiLU()
        )
        head_input = latent + 1 + 4
        adapter_input = latent + 1 + 4
        self.target_temporal_adapters = nn.ModuleList()
        for _ in range(n_state):
            adapter = nn.Sequential(
                nn.Linear(adapter_input, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, latent),
            )
            # Start as an exact residual identity.  The adapter must earn a
            # target-specific departure from the shared body state.
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            self.target_temporal_adapters.append(adapter)
        self.target_temporal_gates = nn.Parameter(torch.full((n_state,), -2.0))
        if self.patient_residual_adapter_enabled:
            self.patient_residual_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(latent, hidden // 2),
                        nn.LayerNorm(hidden // 2),
                        nn.SiLU(),
                        nn.Linear(hidden // 2, 1),
                    )
                    for _ in range(n_state)
                ]
            )
            for head in self.patient_residual_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            self.patient_residual_gates = nn.Parameter(torch.full((n_state,), -3.0))
        if self.target_horizon_regime_adapter_enabled:
            self.target_horizon_regime_table = nn.Parameter(
                torch.zeros(
                    (n_state, len(horizon_values), self.regime_count, latent),
                    dtype=torch.float32,
                )
            )
            self.target_horizon_regime_gates = nn.Parameter(
                torch.full(
                    (n_state, len(horizon_values), self.regime_count),
                    -3.0,
                    dtype=torch.float32,
                )
            )
            # A small direct residual is supervised by factual value loss as
            # well as JEPA loss.  It is initialized to zero and remains
            # fail-closed until the target/horizon/regime cell earns signal.
            self.target_horizon_regime_value_bias = nn.Parameter(
                torch.zeros(
                    (n_state, len(horizon_values), self.regime_count),
                    dtype=torch.float32,
                )
            )
            self.target_horizon_regime_value_gates = nn.Parameter(
                torch.full(
                    (n_state, len(horizon_values), self.regime_count),
                    -3.0,
                    dtype=torch.float32,
                )
            )
        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(head_input, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                for _ in range(n_state)
            ]
        )
        point_model_rng_state = torch.random.get_rng_state()
        self.value_scale_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_input, hidden // 2),
                    nn.SiLU(),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in range(n_state)
            ]
        )
        for head in self.value_scale_heads:
            # One normalized unit is a conservative starting width. The scale
            # head is trained on detached point errors so uncertainty
            # calibration cannot move the physiological forecast itself.
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 0.5)
        # The detached scale branch is an audit/calibration add-on. Preserve
        # the RNG stream so adding it cannot silently change initialization of
        # the physiological value, belief, or target-encoder branches.
        torch.random.set_rng_state(point_model_rng_state)
        self.observation_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
                for _ in range(n_state)
            ]
        )
        self.reconstruction_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(latent, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
                for _ in range(n_state)
            ]
        )
        if self.uncertainty_gate_enabled:
            self.uncertainty_gate_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(latent + hidden, hidden // 2),
                        nn.LayerNorm(hidden // 2),
                        nn.SiLU(),
                        nn.Linear(hidden // 2, 1),
                    )
                    for _ in range(n_state)
                ]
            )
            for head in self.uncertainty_gate_heads:
                # Start close to the persistence anchor.  A target must earn
                # permission to move as the belief becomes more certain.
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, -1.5)
        if self.validated_edge_adapters_enabled:
            self.edge_value_heads = nn.ModuleDict()
            self.edge_gates = nn.ParameterDict()
            self.edge_source_context_encoders = nn.ModuleDict()
            self.edge_treatment_context_encoders = nn.ModuleDict()
            self.edge_contextual_gates = nn.ModuleDict()
            for spec in self.validated_edge_specs:
                key = str(spec["key"])
                source_indices = tuple(
                    int(value) for value in spec.get("source_variable_indices", ())
                )
                contextual_regime_gate = bool(
                    spec.get("contextual_regime_gate", False)
                )
                source_context_dim = hidden // 2 if (
                    contextual_regime_gate and source_indices
                ) else 0
                treatment_context_indices = tuple(
                    int(value) for value in spec.get("treatment_context_indices", ())
                )
                treatment_context_dim = hidden // 4 if (
                    contextual_regime_gate and treatment_context_indices
                ) else 0
                if source_context_dim:
                    self.edge_source_context_encoders[key] = nn.Sequential(
                        nn.Linear(3 * len(source_indices), hidden // 2),
                        nn.LayerNorm(hidden // 2),
                        nn.SiLU(),
                    )
                if treatment_context_dim:
                    self.edge_treatment_context_encoders[key] = nn.Sequential(
                        nn.Linear(len(treatment_context_indices), treatment_context_dim),
                        nn.LayerNorm(treatment_context_dim),
                        nn.SiLU(),
                    )
                self.edge_value_heads[key] = nn.Sequential(
                    nn.Linear(
                        latent + 1 + 4 + 1 + source_context_dim + treatment_context_dim,
                        hidden // 2,
                    ),
                    nn.LayerNorm(hidden // 2),
                    nn.SiLU(),
                    nn.Linear(hidden // 2, 1),
                )
                nn.init.zeros_(self.edge_value_heads[key][-1].weight)
                nn.init.zeros_(self.edge_value_heads[key][-1].bias)
                self.edge_gates[key] = nn.Parameter(torch.tensor(-2.0))
                if source_context_dim:
                    self.edge_contextual_gates[key] = nn.Sequential(
                        nn.Linear(
                            latent + 1 + 4 + 1 + source_context_dim + treatment_context_dim,
                            hidden // 2,
                        ),
                        nn.LayerNorm(hidden // 2),
                        nn.SiLU(),
                        nn.Linear(hidden // 2, 1),
                    )
                    # Start near the persistence anchor. The gate must learn
                    # a supported patient-state regime before it can move.
                    nn.init.zeros_(self.edge_contextual_gates[key][-1].weight)
                    nn.init.zeros_(self.edge_contextual_gates[key][-1].bias)
        # Construct this optional branch last so enabling it does not change
        # initialization of the physiological forecast heads.  That keeps the
        # baseline/candidate comparison apples-to-apples.
        self.measurement_time_heads = nn.ModuleList()
        if self.measurement_time_head:
            for _ in range(n_state):
                self.measurement_time_heads.append(
                    nn.Sequential(
                        nn.Linear(head_input, hidden // 2),
                        nn.LayerNorm(hidden // 2),
                        nn.SiLU(),
                        nn.Linear(hidden // 2, 1),
                    )
                )

    @torch.no_grad()
    def configure_supervision_strata(self, regime_metadata):
        """Install training-only regime cutpoints for inference-time routing.

        Cutpoints are estimated from the training rows only.  The model then
        derives the anchor regime from the current observed state, so the
        regime adapter never consumes a future label.
        """

        cutpoints = regime_metadata.get("target_cutpoints_normalized", {})
        self.regime_cutpoints.zero_()
        self.regime_cutpoint_valid.zero_()
        for index in range(self.n_state):
            values = cutpoints.get(str(index), [])
            if not values:
                continue
            count = min(len(values), self.regime_count - 1)
            tensor = torch.as_tensor(values[:count], device=self.regime_cutpoints.device)
            self.regime_cutpoints[index, :count].copy_(tensor)
            self.regime_cutpoint_valid[index, :count] = 1.0

    def _infer_regime_ids(self, current, mask=None):
        """Infer target-specific anchor regimes from observed normalized values."""

        regime_ids = torch.zeros(
            current.shape[0], self.n_state, dtype=torch.long, device=current.device
        )
        neutral = int(self.regime_count // 2)
        for index in range(self.n_state):
            valid = self.regime_cutpoint_valid[index] > 0.0
            if bool(valid.any()):
                cuts = self.regime_cutpoints[index][valid]
                regime_ids[:, index] = torch.bucketize(current[:, index], cuts)
            if mask is not None:
                regime_ids[:, index] = torch.where(
                    mask[:, index] > 0.0,
                    regime_ids[:, index],
                    torch.full_like(regime_ids[:, index], neutral),
                )
        return regime_ids.clamp(max=self.regime_count - 1)

    def _horizon_bucket_ids(self, horizons):
        distances = (horizons.unsqueeze(-1) - self.horizon_bucket_values).abs()
        return distances.argmin(dim=-1)

    @staticmethod
    def _time_features(horizon):
        horizon = horizon.clamp(min=0.05, max=168.0)
        log_horizon = torch.log1p(horizon)
        return torch.stack(
            [
                log_horizon / np.log1p(48.0),
                horizon / 6.0,
                torch.sin(log_horizon),
                torch.cos(log_horizon),
            ],
            dim=-1,
        )

    def _encode_body_latent(
        self,
        hidden_state,
        current,
        mask,
        ages,
        presence,
        *,
        target=False,
        return_organ_latents=False,
    ):
        if target:
            variable_embedding = self.target_variable_embedding
            module_token_embedding = self.target_module_token_embedding
            token_projection = self.target_token_projection
            variable_mixer = self.target_variable_mixer
            module_mixer = self.target_module_mixer
            presence_projection = self.target_presence_projection
            body_pool_projection = self.target_body_pool_projection
            state_encoder = self.target_state_encoder
        else:
            variable_embedding = self.variable_embedding
            module_token_embedding = self.module_token_embedding
            token_projection = self.token_projection
            variable_mixer = self.variable_mixer
            module_mixer = self.module_mixer
            presence_projection = self.presence_projection
            body_pool_projection = self.body_pool_projection
            state_encoder = self.state_encoder
        hidden = hidden_state
        tokens = token_projection(torch.stack([current, mask, ages], dim=-1))
        tokens = tokens + variable_embedding.unsqueeze(0)
        tokens = tokens + module_token_embedding(self.variable_module_ids).unsqueeze(0)
        # The context path keeps the full cross-variable Transformer.  The
        # future target path is stop-gradient and is evaluated repeatedly for
        # every horizon/target, so the fast target view intentionally keeps
        # token identity and pooling but skips the repeated mixer launches.
        target_encoder_mode = getattr(self, "target_encoder_mode", "full")
        if not (target and target_encoder_mode == "fast"):
            tokens = variable_mixer(tokens)

        # Build one representation per organ before fusing the body. A flat
        # mean lets large modules dominate and lets missingness masquerade as
        # a cross-organ signal.
        membership = self.variable_module_membership.unsqueeze(0)
        observed_weight = membership * mask.unsqueeze(1)
        denominator = observed_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        module_tokens = torch.einsum("bnd,bmn->bmd", tokens, observed_weight) / denominator
        module_ids = torch.arange(self.module_count, device=tokens.device, dtype=torch.long)
        module_tokens = module_tokens + module_token_embedding(module_ids).unsqueeze(0)
        if not (target and target_encoder_mode == "fast"):
            module_tokens = module_mixer(module_tokens)
        module_weights = presence.unsqueeze(-1)
        module_pooled = (module_tokens * module_weights).sum(dim=1) / module_weights.sum(
            dim=1
        ).clamp_min(1.0)
        variable_pooled = tokens.mean(dim=1)
        pooled = body_pool_projection(
            torch.cat([variable_pooled, module_pooled], dim=-1)
        )
        presence_latent = presence_projection(presence)
        body_latent = state_encoder(
            torch.cat([hidden, pooled, current, mask, ages, presence_latent], dim=-1)
        )
        if return_organ_latents:
            # These tokens are learned inside the same body encoder and are
            # therefore a factual, model-derived representation rather than a
            # hand-written organ belief.  Presence remains explicit so a
            # missing organ view can never look like a measured zero state.
            return body_latent, module_tokens, presence
        return body_latent

    def encode_context_components(
        self, observations, current, mask, ages, presence
    ):
        """Return body, organ and patient-specific predict-update states."""

        belief_observations = observations
        if self.belief_input_mode == "current_only":
            belief_observations = observations[:, -1:, :].expand_as(observations)
        elif self.belief_input_mode == "placebo_time_only":
            # Keep sequence timing and identical parameter capacity, while
            # removing every patient-specific physiological input.
            belief_observations = torch.zeros_like(observations)
            belief_observations[..., -1] = observations[..., -1].detach()
        if (
            self.adapter_only_treatment_context
            and self.adapter_treatment_context_indices.numel() > 0
        ):
            belief_observations = observations.clone()
            belief_observations[
                ..., self.adapter_treatment_context_indices
            ] = 0.0
        belief_state, belief_uncertainty = self.belief_filter(belief_observations)
        latent, organ_latents, organ_presence = self._encode_body_latent(
            belief_state,
            current,
            mask,
            ages,
            presence,
            target=False,
            return_organ_latents=True,
        )
        return (
            latent,
            belief_state,
            belief_uncertainty,
            organ_latents,
            organ_presence,
        )

    def encode_context(self, observations, current, mask, ages, presence):
        """Infer a patient belief state, then encode the anchor body state."""
        return self.encode_context_components(
            observations, current, mask, ages, presence
        )[:3]

    def encode_latent(self, observations, current, mask, ages, presence):
        """Encode only anchor-time information into the context latent."""

        return self.encode_context(observations, current, mask, ages, presence)[0]

    def _decode_latent(
        self,
        latent,
        current,
        horizon,
        ages=None,
        target_latents=None,
        target_horizons=None,
        current_mask=None,
        belief_uncertainty=None,
        module_presence=None,
        treatment_observations=None,
    ):
        time = self._time_features(horizon)
        if target_horizons is None:
            target_horizons = horizon.unsqueeze(-1).expand(-1, self.n_state)
        values, value_scales = [], []
        observed, reconstructed, next_measurement_times = [], [], []
        temporal_latents, temporal_gates, persistence_gates = [], [], []
        inferred_regime_ids = (
            self._infer_regime_ids(current, current_mask)
            if self.target_horizon_regime_adapter_enabled
            else None
        )
        for index in range(self.n_state):
            target_base = latent if target_latents is None else target_latents[:, index]
            target_time = self._time_features(target_horizons[:, index])
            target_context = torch.cat(
                [target_base, current[:, index : index + 1], target_time], dim=-1
            )
            gate = torch.sigmoid(self.target_temporal_gates[index])
            adapted = target_base + 0.25 * gate * torch.tanh(
                self.target_temporal_adapters[index](target_context)
            )
            head_input = torch.cat(
                [adapted, current[:, index : index + 1], target_time], dim=-1
            )
            value_residual = 0.0
            if inferred_regime_ids is not None:
                horizon_ids = self._horizon_bucket_ids(target_horizons[:, index])
                regime_ids = inferred_regime_ids[:, index]
                value_residual = (
                    0.25
                    * torch.sigmoid(
                        self.target_horizon_regime_value_gates[index][
                            horizon_ids, regime_ids
                        ]
                    )
                    * self.target_horizon_regime_value_bias[index][
                        horizon_ids, regime_ids
                    ]
                ).unsqueeze(-1)
            if self.patient_residual_adapter_enabled:
                patient_residual = (
                    0.25
                    * torch.sigmoid(self.patient_residual_gates[index])
                    * torch.tanh(self.patient_residual_heads[index](latent))
                )
                value_residual = value_residual + patient_residual
            if (
                self.validated_edge_adapters_enabled
                and module_presence is not None
            ):
                for spec in self.validated_edge_specs:
                    if int(spec["target_index"]) != index:
                        continue
                    source_values = spec.get("source_module_indices")
                    if source_values is None:
                        source_values = (int(spec["source_module_index"]),)
                    source_indices = tuple(int(value) for value in source_values)
                    # A source group is represented by its observed fraction;
                    # a single-source edge remains exactly 0/1 as before.
                    source_present = module_presence[:, list(source_indices)].mean(
                        dim=1, keepdim=True
                    )
                    allowed_horizons = tuple(
                        float(value) for value in spec.get("allowed_horizons", ())
                    )
                    if allowed_horizons:
                        allowed = torch.zeros_like(source_present)
                        for allowed_horizon in allowed_horizons:
                            allowed = torch.maximum(
                                allowed,
                                torch.isclose(
                                    horizon[:, None],
                                    torch.full_like(horizon[:, None], allowed_horizon),
                                    atol=1e-4,
                                ).float(),
                            )
                        source_present = source_present * allowed
                    source_context = None
                    source_variable_indices = tuple(
                        int(value) for value in spec.get("source_variable_indices", ())
                    )
                    if (
                        source_variable_indices
                        and str(spec["key"]) in self.edge_source_context_encoders
                    ):
                        source_values = current[:, list(source_variable_indices)]
                        source_mask = current_mask[:, list(source_variable_indices)]
                        source_age_values = (
                            ages
                            if ages is not None
                            else torch.zeros_like(current)
                        )
                        source_ages = source_age_values[:, list(source_variable_indices)].clamp(
                            min=0.0, max=72.0
                        ) / 72.0
                        # Values are already training-normalized.  Keep mask
                        # and measurement age explicit so the gate cannot
                        # mistake missing or stale evidence for physiology.
                        source_context = self.edge_source_context_encoders[
                            str(spec["key"])
                        ](
                            torch.cat(
                                [
                                    source_values * source_mask,
                                    source_mask,
                                    source_ages,
                                ],
                                dim=-1,
                            )
                        )
                    treatment_context = None
                    treatment_context_indices = tuple(
                        int(value)
                        for value in spec.get("treatment_context_indices", ())
                    )
                    if (
                        treatment_context_indices
                        and treatment_observations is not None
                        and str(spec["key"]) in self.edge_treatment_context_encoders
                    ):
                        # The final causal history step is the anchor. Its
                        # selected values, observation flags, and ages came
                        # only from pre-anchor hist_* fields.
                        treatment_values = treatment_observations[
                            :, -1, list(treatment_context_indices)
                        ]
                        treatment_context = self.edge_treatment_context_encoders[
                            str(spec["key"])
                        ](treatment_values)
                    edge_input = torch.cat(
                        [
                            latent,
                            current[:, index : index + 1],
                            target_time,
                            source_present,
                        ]
                        + ([source_context] if source_context is not None else [])
                        + ([treatment_context] if treatment_context is not None else []),
                        dim=-1,
                    )
                    edge_delta = self.edge_value_heads[str(spec["key"])](edge_input)
                    edge_gate = self.edge_gates[str(spec["key"])]
                    if str(spec["key"]) in self.edge_contextual_gates:
                        edge_gate = edge_gate + self.edge_contextual_gates[
                            str(spec["key"])
                        ](edge_input)
                    value_residual = value_residual + (
                        0.25
                        * source_present
                        * torch.sigmoid(edge_gate)
                        * torch.tanh(edge_delta)
                    )
            value_output = self.value_heads[index](head_input) + value_residual
            movement_gate = torch.ones_like(value_output)
            if self.uncertainty_gate_enabled and belief_uncertainty is not None:
                gate_input = torch.cat([latent, belief_uncertainty], dim=-1)
                movement_gate = torch.sigmoid(
                    self.uncertainty_gate_heads[index](gate_input)
                )
            values.append(
                current[:, index : index + 1]
                + 0.75 * movement_gate * torch.tanh(value_output)
            )
            value_scales.append(
                nn.functional.softplus(
                    self.value_scale_heads[index](head_input.detach())
                )
                + 1e-3
            )
            if self.measurement_time_head:
                # Predict log1p(hours) for a positive, heavy-tailed process.
                # This head describes observation timing only; it never feeds
                # back into the physiological value forecast.
                next_measurement_times.append(
                    self.measurement_time_heads[index](
                        torch.cat(
                            [
                                adapted.detach(),
                                current[:, index : index + 1],
                                target_time,
                            ],
                            dim=-1,
                        )
                    )
                )
            persistence_gates.append(movement_gate)
            # Whether a target is measured is partly a hospital/workflow
            # decision. Keep that useful prediction task, but stop its loss
            # from reshaping the physiological context latent.
            observation_head_input = torch.cat(
                [
                    adapted.detach(),
                    current[:, index : index + 1],
                    target_time,
                ],
                dim=-1,
            )
            observed.append(
                self.observation_heads[index](observation_head_input)
            )
            reconstructed.append(self.reconstruction_heads[index](latent))
            temporal_latents.append(adapted)
            temporal_gates.append(gate.expand(latent.shape[0], 1))
        output = {
            "value": torch.cat(values, dim=-1),
            "value_scale": torch.cat(value_scales, dim=-1),
            "observation_logit": torch.cat(observed, dim=-1),
            "reconstruction": torch.cat(reconstructed, dim=-1),
            "target_temporal_latent": torch.stack(temporal_latents, dim=1),
            "target_temporal_gate": torch.cat(temporal_gates, dim=-1),
            "persistence_gate": torch.cat(persistence_gates, dim=-1),
            # Exposed for measurement-invariance regularization.  Keeping it
            # in the output also makes the learned body state auditable without
            # changing the forecast contract.
            "latent": latent,
        }
        if self.measurement_time_head:
            output["next_measurement_time_log"] = torch.cat(
                next_measurement_times, dim=-1
            )
        return output

    def forward(self, observations, current, mask, ages, presence, horizon):
        (
            latent,
            belief_state,
            belief_uncertainty,
            organ_latents,
            organ_presence,
        ) = self.encode_context_components(
            observations, current, mask, ages, presence
        )
        output = self._decode_latent(
            latent,
            current,
            horizon,
            ages=ages,
            current_mask=mask,
            belief_uncertainty=belief_uncertainty,
            module_presence=presence,
            treatment_observations=observations,
        )
        output["belief_state"] = belief_state
        output["belief_uncertainty"] = belief_uncertainty
        output["organ_latents"] = organ_latents
        output["organ_presence"] = organ_presence
        return output


class JointWholeBodyJEPAWorldModel(JointWholeBodyJEPA):
    """JEPA world-model candidate with an EMA future-state target encoder.

    The context path sees only anchor history/current observations.  The
    target path sees only the observed future state and is stop-gradient.  A
    predictor maps the context latent plus horizon to the future latent.  The
    numerical heads remain decoders for factual evaluation, not the primary
    JEPA training target.
    """

    def __init__(
        self,
        *args,
        target_momentum=0.99,
        target_min_observations=2,
        target_encoder_mode="full",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if target_encoder_mode not in {"full", "fast"}:
            raise ValueError("target_encoder_mode must be 'full' or 'fast'")
        self.target_encoder_mode = str(target_encoder_mode)
        self.target_momentum = float(target_momentum)
        self.target_min_observations = int(target_min_observations)
        self.target_variable_embedding = nn.Parameter(
            self.variable_embedding.detach().clone(), requires_grad=False
        )
        self.target_module_token_embedding = deepcopy(self.module_token_embedding)
        self.target_token_projection = deepcopy(self.token_projection)
        self.target_variable_mixer = deepcopy(self.variable_mixer)
        self.target_module_mixer = deepcopy(self.module_mixer)
        self.target_presence_projection = deepcopy(self.presence_projection)
        self.target_body_pool_projection = deepcopy(self.body_pool_projection)
        self.target_state_encoder = deepcopy(self.state_encoder)
        for module in (
            self.target_module_token_embedding,
            self.target_token_projection,
            self.target_variable_mixer,
            self.target_module_mixer,
            self.target_presence_projection,
            self.target_body_pool_projection,
            self.target_state_encoder,
        ):
            module.requires_grad_(False)
        self.target_variable_embedding.requires_grad_(False)
        self.latent_predictor = nn.Sequential(
            nn.Linear(self.latent + 4, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.latent),
        )
        self.target_future_predictors = nn.ModuleList()
        for _ in range(self.n_state):
            predictor = nn.Sequential(
                nn.Linear(self.latent + 1 + 4, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.SiLU(),
                nn.Linear(self.hidden, self.latent),
            )
            # The target-specific path starts from the shared JEPA predictor;
            # it must learn a residual future representation per variable.
            nn.init.zeros_(predictor[-1].weight)
            nn.init.zeros_(predictor[-1].bias)
            self.target_future_predictors.append(predictor)
        self.target_future_gates = nn.Parameter(torch.full((self.n_state,), -2.0))
        self._keep_target_encoder_in_eval_mode()

    def _keep_target_encoder_in_eval_mode(self):
        """Keep the stop-gradient EMA teacher deterministic during training."""

        for module in (
            self.target_module_token_embedding,
            self.target_token_projection,
            self.target_variable_mixer,
            self.target_module_mixer,
            self.target_presence_projection,
            self.target_body_pool_projection,
            self.target_state_encoder,
        ):
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        # ``Module.train`` is recursive, so without this reset the frozen EMA
        # target Transformer would enable dropout and present a moving,
        # stochastic supervision target to the online predictor.
        self._keep_target_encoder_in_eval_mode()
        return self

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA-update every target encoder component after an optimizer step."""

        momentum = min(max(float(self.target_momentum), 0.0), 0.99999)
        pairs = [
            (self.target_variable_embedding, self.variable_embedding),
            (self.target_module_token_embedding, self.module_token_embedding),
            (self.target_token_projection, self.token_projection),
            (self.target_variable_mixer, self.variable_mixer),
            (self.target_module_mixer, self.module_mixer),
            (self.target_presence_projection, self.presence_projection),
            (self.target_body_pool_projection, self.body_pool_projection),
            (self.target_state_encoder, self.state_encoder),
        ]
        for target, online in pairs:
            target_parameters = [target] if isinstance(target, nn.Parameter) else target.parameters()
            online_parameters = [online] if isinstance(online, nn.Parameter) else online.parameters()
            for target_parameter, online_parameter in zip(target_parameters, online_parameters):
                target_parameter.data.mul_(momentum).add_(online_parameter.data, alpha=1.0 - momentum)

    def _encode_target_specific_values(self, values, feature_mask):
        """Encode one future delta latent per target variable.

        Each target receives only its own paired delta.  This keeps the JEPA
        target contract target-specific while the context encoder still sees
        the complete measured body state.

        The target axis is batched in row chunks.  The previous implementation
        called the target Transformer once per variable, which made a full
        whole-body window cost roughly ``batch * horizon * target_count``
        Transformer launches.  Diagonalizing values/masks preserves the exact
        one-target-per-token contract while reducing it to one launch per row
        chunk.
        """

        batch_size = values.shape[0]
        if batch_size == 0:
            return (
                values.new_empty((0, self.n_state, self.latent)),
                feature_mask.new_empty((0, self.n_state), dtype=torch.bool),
            )

        # Keep the flattened target batch bounded.  This avoids constructing a
        # very large [batch, target, target] diagonal tensor for full gates,
        # while still reducing Transformer launches by approximately n_state.
        max_flattened_targets = 4096
        row_chunk = max(1, max_flattened_targets // max(1, self.n_state))
        membership = self.variable_module_membership.to(dtype=values.dtype)
        latents = []
        available = []
        for start in range(0, batch_size, row_chunk):
            stop = min(batch_size, start + row_chunk)
            chunk_values = values[start:stop]
            chunk_mask = feature_mask[start:stop]
            chunk_rows = stop - start

            # Each [row, target] item becomes one flattened example whose only
            # non-zero physiological feature is that target's own delta.
            target_values = torch.diag_embed(chunk_values * chunk_mask)
            target_mask = torch.diag_embed(chunk_mask)
            target_presence = (
                torch.matmul(target_mask, membership.t()) > 0
            ).to(values.dtype)
            flat = chunk_rows * self.n_state
            encoded = self._encode_body_latent(
                torch.zeros(
                    flat,
                    self.hidden,
                    dtype=values.dtype,
                    device=values.device,
                ),
                target_values.reshape(flat, self.n_state),
                target_mask.reshape(flat, self.n_state),
                torch.zeros(
                    flat,
                    self.n_state,
                    dtype=values.dtype,
                    device=values.device,
                ),
                target_presence.reshape(flat, self.module_count),
                target=True,
            )
            latents.append(encoded.reshape(chunk_rows, self.n_state, self.latent))
            available.append((chunk_mask > 0).to(torch.bool))
        return torch.cat(latents, dim=0), torch.cat(available, dim=0)

    def encode_target_window(
        self,
        future_window,
        future_window_mask,
        window_horizons,
        forecast_horizon,
        anchor_current=None,
    ):
        """Encode measurement-pure future deltas as stop-gradient targets.

        Future masks are used only for paired-delta evidence and masked token
        pooling. They are never exposed to the context encoder, so the
        forecast path cannot copy the future measurement schedule.
        """

        batch_size, width, _ = future_window.shape
        if anchor_current is None:
            delta_window = future_window
        else:
            delta_window = future_window - anchor_current.unsqueeze(1)
        target_values = delta_window * future_window_mask
        target_tokens = []
        target_specific_tokens = []
        target_specific_available = []
        for position in range(width):
            # The target encoder sees only paired, measured deltas.  This
            # prevents an unobserved variable from becoming a fake zero
            # change, while keeping the context path free of future masks.
            target_feature_mask = future_window_mask[:, position]
            target_presence = (
                torch.matmul(
                    target_feature_mask,
                    self.variable_module_membership.to(future_window.dtype).t(),
                )
                > 0
            ).to(future_window.dtype)
            target_latent = self._encode_body_latent(
                torch.zeros(
                    batch_size,
                    self.hidden,
                    dtype=future_window.dtype,
                    device=future_window.device,
                ),
                target_values[:, position],
                target_feature_mask,
                torch.zeros_like(target_feature_mask),
                target_presence,
                target=True,
            )
            # Keep the target trajectory aggregation deterministic.  A learned
            # temporal aggregator would be randomly initialized and frozen by
            # the stop-gradient target path, which would turn the target into
            # an arbitrary projection rather than a physiological trajectory.
            target_tokens.append(target_latent)
            specific_latent, specific_available = self._encode_target_specific_values(
                target_values[:, position], future_window_mask[:, position]
            )
            target_specific_tokens.append(specific_latent)
            target_specific_available.append(specific_available)
        sequence = torch.stack(target_tokens, dim=1)
        horizons = torch.as_tensor(
            window_horizons, dtype=future_window.dtype, device=future_window.device
        ).view(1, -1)
        valid_step = (
            future_window_mask.sum(dim=-1) >= self.target_min_observations
        ) & (
            horizons <= forecast_horizon.view(-1, 1) + 1e-6
        )
        weights = valid_step.float().unsqueeze(-1)
        pooled = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        specific_sequence = torch.stack(target_specific_tokens, dim=1)
        specific_available = torch.stack(target_specific_available, dim=1)
        horizon_allowed = (
            horizons <= forecast_horizon.view(-1, 1) + 1e-6
        ).unsqueeze(-1)
        specific_available = specific_available & horizon_allowed
        return pooled, sequence, valid_step, specific_sequence, specific_available

    def _predict_latent_trajectory(self, context_latent, window_horizons):
        """Predict one latent per horizon, preserving temporal order."""

        if window_horizons.ndim == 1:
            window_horizons = window_horizons.unsqueeze(0).expand(
                context_latent.shape[0], -1
            )
        batch_size, width = window_horizons.shape
        repeated_context = context_latent.unsqueeze(1).expand(-1, width, -1)
        time = self._time_features(window_horizons.reshape(-1)).reshape(
            batch_size, width, -1
        )
        predicted = self.latent_predictor(
            torch.cat([repeated_context, time], dim=-1).reshape(
                batch_size * width, -1
            )
        )
        return predicted.reshape(batch_size, width, -1)

    def _predict_target_latent_trajectory(
        self,
        context_latent,
        current,
        window_horizons,
        shared_trajectory=None,
        target_horizons=None,
        current_mask=None,
    ):
        """Predict one future latent trajectory per target variable."""

        if window_horizons.ndim == 1:
            window_horizons = window_horizons.unsqueeze(0).expand(
                context_latent.shape[0], -1
            )
        if shared_trajectory is None:
            shared_trajectory = self._predict_latent_trajectory(
                context_latent, window_horizons
            )
        batch_size, width, _ = shared_trajectory.shape
        if target_horizons is None:
            target_horizons = window_horizons.unsqueeze(-1).expand(
                -1, -1, self.n_state
            )
        repeated_context = context_latent.unsqueeze(1).expand(-1, width, -1)
        regime_ids = None
        if self.target_horizon_regime_adapter_enabled:
            regime_ids = self._infer_regime_ids(current, current_mask)
        time = self._time_features(target_horizons.reshape(-1)).reshape(
            batch_size, width, self.n_state, -1
        )
        target_latents = []
        for index in range(self.n_state):
            current_target = current[:, index : index + 1].unsqueeze(1).expand(
                -1, width, -1
            )
            predictor_input = torch.cat(
                [
                    repeated_context.unsqueeze(2).expand(-1, -1, self.n_state, -1)[:, :, index],
                    current_target,
                    time[:, :, index],
                ],
                dim=-1,
            ).reshape(batch_size * width, -1)
            residual = self.target_future_predictors[index](predictor_input).reshape(
                batch_size, width, -1
            )
            if regime_ids is not None:
                horizon_ids = self._horizon_bucket_ids(target_horizons[:, :, index])
                regime_index = regime_ids[:, index].unsqueeze(1).expand(-1, width)
                specialized = self.target_horizon_regime_table[index][
                    horizon_ids, regime_index
                ]
                specialized_gate = torch.sigmoid(
                    self.target_horizon_regime_gates[index][
                        horizon_ids, regime_index
                    ]
                ).unsqueeze(-1)
                residual = residual + 0.25 * specialized_gate * specialized
            gate = torch.sigmoid(self.target_future_gates[index])
            target_latents.append(
                shared_trajectory + 0.25 * gate * torch.tanh(residual)
            )
        return torch.stack(target_latents, dim=2)

    def forward(
        self,
        observations,
        current,
        mask,
        ages,
        presence,
        horizon,
        future=None,
        future_mask=None,
        future_window=None,
        future_window_mask=None,
        target_horizon=None,
        future_window_delta_t=None,
        window_horizons=None,
    ):
        (
            context_latent,
            belief_state,
            belief_uncertainty,
            organ_latents,
            organ_presence,
        ) = self.encode_context_components(
            observations, current, mask, ages, presence
        )
        time = self._time_features(horizon)
        predicted_latent = self.latent_predictor(
            torch.cat([context_latent, time], dim=-1)
        )
        if window_horizons is not None:
            trajectory_horizons = torch.as_tensor(
                window_horizons, dtype=horizon.dtype, device=horizon.device
            )
            if trajectory_horizons.ndim == 1:
                trajectory_horizons = trajectory_horizons.unsqueeze(0).expand(
                    horizon.shape[0], -1
                )
            predicted_trajectory_latent = self._predict_latent_trajectory(
                context_latent, trajectory_horizons
            )
        else:
            trajectory_horizons = horizon.unsqueeze(1)
            predicted_trajectory_latent = predicted_latent.unsqueeze(1)
        if target_horizon is None:
            target_horizon = horizon.unsqueeze(-1).expand(-1, self.n_state)
        target_horizon_window = target_horizon.unsqueeze(1)
        predicted_target_trajectory_latent = self._predict_target_latent_trajectory(
            context_latent,
            current,
            horizon.unsqueeze(1),
            shared_trajectory=predicted_latent.unsqueeze(1),
            target_horizons=target_horizon_window,
            current_mask=mask,
        )
        if window_horizons is not None:
            if future_window_delta_t is None:
                target_horizon_window = trajectory_horizons.unsqueeze(-1).expand(
                    -1, -1, self.n_state
                )
            else:
                target_horizon_window = future_window_delta_t
            full_predicted_target_trajectory_latent = self._predict_target_latent_trajectory(
                context_latent,
                current,
                trajectory_horizons,
                shared_trajectory=predicted_trajectory_latent,
                target_horizons=target_horizon_window,
                current_mask=mask,
            )
        else:
            full_predicted_target_trajectory_latent = predicted_target_trajectory_latent
        predicted_target_latent = predicted_target_trajectory_latent[:, 0]
        output = self._decode_latent(
            predicted_latent,
            current,
            horizon,
            ages=ages,
            target_latents=predicted_target_latent,
            target_horizons=target_horizon,
            current_mask=mask,
            belief_uncertainty=belief_uncertainty,
            module_presence=presence,
            treatment_observations=observations,
        )
        output["context_latent"] = context_latent
        output["belief_state"] = belief_state
        output["belief_uncertainty"] = belief_uncertainty
        output["organ_latents"] = organ_latents
        output["organ_presence"] = organ_presence
        output["predicted_latent"] = predicted_latent
        output["predicted_trajectory_latent"] = predicted_trajectory_latent
        output["predicted_target_latent"] = predicted_target_latent
        output["predicted_target_trajectory_latent"] = (
            full_predicted_target_trajectory_latent
        )
        output["target_horizon"] = target_horizon
        output["target_horizon_trajectory"] = target_horizon_window
        if future_window is not None and future_window_mask is not None:
            with torch.no_grad():
                (
                    target_latent,
                    target_sequence,
                    target_step_available,
                    target_specific_sequence,
                    target_specific_available,
                ) = self.encode_target_window(
                    future_window,
                    future_window_mask,
                    window_horizons,
                    horizon,
                    anchor_current=current,
                )
            output["target_latent"] = target_latent
            output["target_trajectory_latent"] = target_sequence
            output["target_step_available"] = target_step_available
            output["target_available"] = target_step_available.any(dim=1)
            output["target_specific_trajectory_latent"] = target_specific_sequence
            output["target_specific_available"] = target_specific_available
        elif future is not None and future_mask is not None:
            # Compatibility fallback for callers that have only one future
            # label. The training path uses future windows by default.
            paired_mask = future_mask * mask
            with torch.no_grad():
                target_specific_latent, target_specific_available = (
                    self._encode_target_specific_values(
                        (future - current) * paired_mask,
                        paired_mask,
                    )
                )
                target_latent = self._encode_body_latent(
                    torch.zeros(
                        future.shape[0],
                        self.hidden,
                        dtype=future.dtype,
                        device=future.device,
                    ),
                    (future - current) * paired_mask,
                    paired_mask,
                    torch.zeros_like(future_mask),
                    torch.ones(
                        future.shape[0],
                        self.module_count,
                        dtype=future.dtype,
                        device=future.device,
                    ),
                    target=True,
                )
            output["target_latent"] = target_latent
            output["target_trajectory_latent"] = target_latent.unsqueeze(1)
            output["target_step_available"] = (
                paired_mask.sum(dim=-1) >= self.target_min_observations
            ).unsqueeze(1)
            output["target_available"] = paired_mask.sum(dim=-1) >= self.target_min_observations
            output["target_specific_latent"] = target_specific_latent
            output["target_specific_available"] = target_specific_available
        return output


def _balanced_batches(frame, rows, batch_size, rng):
    groups = {}
    modules = frame["_module_count"].to_numpy(dtype=np.int16)
    horizons = frame["_source_horizon"].to_numpy(dtype=np.float32)
    presence_columns = [column for column in frame.columns if column.startswith("_module_present_")]
    presence = frame[presence_columns].to_numpy(dtype=np.float32) if presence_columns else None
    for row in np.asarray(rows, dtype=np.int64):
        if presence is not None:
            signature = tuple(np.flatnonzero(presence[row] > 0.5).tolist())
        else:
            signature = (int(modules[row]),)
        # Module count alone lets a frequent organ pair dominate training.
        # Sampling by the actual signature gives the shared encoder exposure
        # to rare but valid cross-system combinations.
        key = (float(horizons[row]), signature)
        groups.setdefault(key, []).append(int(row))
    keys = sorted(groups)
    steps = max(1, int(np.ceil(len(rows) / float(batch_size))))
    for _ in range(steps):
        candidates = np.asarray(groups[keys[int(rng.integers(0, len(keys)))]], dtype=np.int64)
        yield rng.choice(candidates, size=batch_size, replace=len(candidates) < batch_size)


def _future_task_balanced_batches(
    frame,
    rows,
    future_mask,
    delta_bin_ids,
    regime_ids,
    batch_size,
    rng,
):
    """Sample rows by inverse frequency of supervised future task cells.

    A row can supervise several targets, so its weight is the sum of inverse
    frequencies for its active ``target x nominal-horizon x actual-delta-bin x
    regime`` cells.  This keeps sparse future tasks visible without creating
    synthetic labels or using patient identifiers.
    """

    rows = np.asarray(rows, dtype=np.int64)
    horizon = frame["_source_horizon"].to_numpy(dtype=np.float32)
    counts = {}
    row_tasks = {}
    for row in rows:
        tasks = []
        active = np.flatnonzero(np.asarray(future_mask[row], dtype=np.float32) > 0.0)
        for index in active:
            delta_bin = int(delta_bin_ids[row, index])
            regime = int(regime_ids[row, index])
            if delta_bin < 0 or regime < 0:
                continue
            task = (int(index), float(horizon[row]), delta_bin, regime)
            tasks.append(task)
            counts[task] = counts.get(task, 0) + 1
        row_tasks[int(row)] = tasks
    weights = np.asarray(
        [
            sum(1.0 / max(1, counts[task]) for task in row_tasks[int(row)])
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(rows), dtype=np.float64)
    weights /= weights.sum()
    steps = max(1, int(np.ceil(len(rows) / float(batch_size))))
    for _ in range(steps):
        yield rng.choice(rows, size=batch_size, replace=True, p=weights)


def _target_balanced_loss(prediction, target, mask):
    element = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    count = mask.sum(dim=0)
    per_target = (element * mask).sum(dim=0) / count.clamp_min(1.0)
    active = count > 0
    return per_target[active].mean() if bool(active.any()) else element.new_zeros(())


def _target_balanced_scale_loss(prediction, target, scale, mask):
    """Calibrate target-specific uncertainty without moving point forecasts."""

    safe_scale = scale.clamp_min(1e-3)
    absolute_error = (prediction.detach() - target).abs()
    element = absolute_error / safe_scale + torch.log(safe_scale)
    count = mask.sum(dim=0)
    per_target = (element * mask).sum(dim=0) / count.clamp_min(1.0)
    active = count > 0
    return per_target[active].mean() if bool(active.any()) else element.new_zeros(())


SUPERVISION_DELTA_BIN_EDGES = np.asarray(
    [0.0, 1.5, 4.5, 9.0, 18.0, 36.0, 72.0, np.inf],
    dtype=np.float32,
)


def _actual_delta_bin_ids(delta_t):
    """Map actual event delays to stable, auditable time-scale bins."""

    delta_t = np.asarray(delta_t, dtype=np.float32)
    output = np.full(delta_t.shape, -1, dtype=np.int16)
    valid = np.isfinite(delta_t) & (delta_t > 0.0)
    output[valid] = np.digitize(
        delta_t[valid], SUPERVISION_DELTA_BIN_EDGES[1:-1], right=True
    ).astype(np.int16)
    return output


def _build_supervision_strata(arrays, fit_rows, regime_count=3):
    """Build target-specific anchor regimes from training rows only.

    Regimes use only current measured values, never future labels.  Each
    target gets its own quantile cut points because a heart-rate level and a
    creatinine level represent different physiological scales.  Unknown
    current targets remain ``-1`` and are excluded from stratified loss.
    """

    if int(regime_count) < 2:
        raise ValueError("regime_count must be at least 2")
    current = np.asarray(arrays[2], dtype=np.float32)
    mask = np.asarray(arrays[3], dtype=np.float32)
    future_delta_t = np.asarray(arrays[12], dtype=np.float32)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    n_state = current.shape[1]
    regimes = np.full((len(current), n_state), -1, dtype=np.int16)
    cutpoints = {}
    quantiles = np.linspace(0.0, 1.0, int(regime_count) + 1)[1:-1]
    for index in range(n_state):
        fit_valid = fit_rows[
            (mask[fit_rows, index] > 0.5) & np.isfinite(current[fit_rows, index])
        ]
        values = current[fit_valid, index]
        if len(values) < int(regime_count):
            cutpoints[str(index)] = []
            continue
        cuts = np.quantile(values, quantiles).astype(np.float32)
        cuts = np.maximum.accumulate(cuts)
        cutpoints[str(index)] = [float(value) for value in cuts]
        valid = (mask[:, index] > 0.5) & np.isfinite(current[:, index])
        regimes[valid, index] = np.digitize(
            current[valid, index], cuts, right=True
        ).astype(np.int16)
    return regimes, _actual_delta_bin_ids(future_delta_t), {
        "regime_count": int(regime_count),
        "delta_bin_edges_hours": [float(value) for value in SUPERVISION_DELTA_BIN_EDGES],
        "target_cutpoints_normalized": cutpoints,
    }


def _group_balanced_loss(element, mask, delta_bins, regimes, n_state):
    """Average active errors equally across target/time/regime cells."""

    if not bool(mask.any()):
        return element.new_zeros(())
    if element.ndim == 2:
        delta_bins = torch.as_tensor(delta_bins, device=element.device, dtype=torch.long)
        regimes = torch.as_tensor(regimes, device=element.device, dtype=torch.long)
        target_ids = torch.arange(n_state, device=element.device).unsqueeze(0).expand(element.shape[0], -1)
        valid = (mask > 0.0) & (delta_bins >= 0) & (regimes >= 0)
        keys = target_ids * (8 * 4) + delta_bins * 4 + regimes
    elif element.ndim == 3:
        delta_bins = torch.as_tensor(delta_bins, device=element.device, dtype=torch.long)
        regimes = torch.as_tensor(regimes, device=element.device, dtype=torch.long)
        target_ids = torch.arange(n_state, device=element.device).view(1, 1, -1).expand(element.shape[0], element.shape[1], -1)
        regime_grid = regimes.view(regimes.shape[0], 1, regimes.shape[1]).expand_as(delta_bins)
        valid = (mask > 0.0) & (delta_bins >= 0) & (regime_grid >= 0)
        keys = target_ids * (8 * 4) + delta_bins * 4 + regime_grid
    else:
        raise ValueError("group-balanced loss expects a 2D or 3D tensor")
    if not bool(valid.any()):
        return element.new_zeros(())
    errors = element[valid]
    group_keys = keys[valid]
    _, inverse = torch.unique(group_keys, sorted=True, return_inverse=True)
    sums = torch.zeros(
        int(inverse.max().item()) + 1,
        dtype=errors.dtype,
        device=errors.device,
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, errors)
    counts.scatter_add_(0, inverse, torch.ones_like(errors))
    return (sums / counts.clamp_min(1.0)).mean()


def _regime_balanced_value_loss(prediction, target, mask, delta_bins, regimes):
    element = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    stratified = _group_balanced_loss(element, mask, delta_bins, regimes, prediction.shape[1])
    return stratified if bool((mask > 0.0).any()) and bool(torch.isfinite(stratified)) else _target_balanced_loss(prediction, target, mask)


def _latent_consistency_loss(first, second):
    """Keep the body representation stable across missingness views."""

    first = nn.functional.normalize(first, dim=-1)
    second = nn.functional.normalize(second, dim=-1)
    return nn.functional.smooth_l1_loss(first, second)


def _masked_latent_prediction_loss(predicted, target, available):
    """JEPA loss with equal weight for each available horizon cell."""

    if not bool(available.any()):
        return predicted.new_zeros(())
    predicted = nn.functional.normalize(predicted, dim=-1)
    target = nn.functional.normalize(target.detach(), dim=-1)
    element = nn.functional.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
    counts = available.float().sum(dim=0)
    per_horizon = (element * available.float()).sum(dim=0) / counts.clamp_min(1.0)
    active = counts > 0
    return per_horizon[active].mean()


def _masked_target_specific_latent_prediction_loss(
    predicted,
    target,
    available,
    delta_bins=None,
    regimes=None,
):
    """JEPA loss with equal weight for each active target-horizon cell."""

    if not bool(available.any()):
        return predicted.new_zeros(())
    predicted = nn.functional.normalize(predicted, dim=-1)
    target = nn.functional.normalize(target.detach(), dim=-1)
    element = nn.functional.smooth_l1_loss(predicted, target, reduction="none").mean(
        dim=-1
    )
    if delta_bins is not None and regimes is not None:
        stratified = _group_balanced_loss(
            element,
            available.float(),
            delta_bins,
            regimes,
            predicted.shape[-2],
        )
        if bool(torch.isfinite(stratified)):
            return stratified
    counts = available.float().sum(dim=0)
    per_cell = (element * available.float()).sum(dim=0) / counts.clamp_min(1.0)
    active = counts > 0
    return per_cell[active].mean()


def _masked_batch_view(
    batch,
    current,
    mask,
    ages,
    presence,
    input_matrix,
    history,
    variables,
    module_names,
    module_variables,
    rng,
    masked_forecast_probability,
    module_mask_probability,
):
    """Create one causal missingness view of a batch.

    The target remains the unmasked current value, while the model sees only
    values that would have been available at the anchor.  This helper is used
    both by nowcast pretraining and by the future-forecast objective so the
    two stages share exactly the same observation contract.
    """

    batch_current = current[batch].copy()
    batch_mask = mask[batch].copy()
    batch_ages = ages[batch].copy()
    batch_presence = presence[batch].copy()
    batch_observations = input_matrix[history[batch]].copy()
    drop = (
        rng.random(batch_mask.shape) < float(masked_forecast_probability)
    ) & (batch_mask > 0.0)
    module_variable_indices = {
        module_index: [
            index for index, variable in enumerate(variables)
            if variable in set(module_variables.get(module, ()))
        ]
        for module_index, module in enumerate(module_names)
    }
    for row_index in range(len(batch)):
        observed_modules = np.flatnonzero(presence[batch[row_index]] > 0.5)
        if len(observed_modules) < 2 or rng.random() >= float(module_mask_probability):
            continue
        hidden_module = int(rng.choice(observed_modules))
        for index in module_variable_indices.get(hidden_module, ()):
            if batch_mask[row_index, index] > 0.0:
                drop[row_index, index] = True
        # Hide module metadata as well as its values.  Otherwise the model can
        # identify the missing organ from a presence bit alone.
        batch_presence[row_index, hidden_module] = 0.0
        presence_start = 3 * len(variables) + 1
        batch_observations[row_index, :, presence_start + hidden_module] = 0.0
    for index in range(len(variables)):
        selected = drop[:, index]
        if not bool(selected.any()):
            continue
        batch_current[selected, index] = 0.0
        batch_mask[selected, index] = 0.0
        batch_ages[selected, index] = 7.0
        batch_observations[selected, :, index] = 0.0
        batch_observations[selected, :, len(variables) + index] = 0.0
        batch_observations[selected, :, 2 * len(variables) + index] = 7.0
    # Recompute module availability after random and whole-module masking.
    for module_index, indices in module_variable_indices.items():
        if not indices:
            batch_presence[:, module_index] = 0.0
            continue
        observed_after_mask = batch_mask[:, indices].sum(axis=1) > 0.0
        batch_presence[:, module_index] = (
            presence[batch, module_index] * observed_after_mask.astype(np.float32)
        )
        presence_start = 3 * len(variables) + 1
        batch_observations[:, :, presence_start + module_index] = batch_presence[
            :, module_index, None
        ]
    return batch_current, batch_mask, batch_ages, batch_presence, batch_observations, drop


def _coupling_batch_view(
    batch,
    current,
    mask,
    ages,
    presence,
    input_matrix,
    history,
    future_mask,
    variable_module_membership,
    rng,
):
    """Hide one labelled target while retaining the other body systems.

    This is the explicit cross-organ training view.  Unlike random masking, it
    guarantees that each eligible row contributes a future loss for a target
    that must be inferred from the remaining observed systems.
    """

    batch_current = current[batch].copy()
    batch_mask = mask[batch].copy()
    batch_ages = ages[batch].copy()
    batch_presence = presence[batch].copy()
    batch_observations = input_matrix[history[batch]].copy()
    selected_targets = np.zeros((len(batch), current.shape[1]), dtype=np.float32)
    membership = np.asarray(variable_module_membership, dtype=np.float32)
    for row_index, source_row in enumerate(batch):
        observed_modules = np.flatnonzero(batch_presence[row_index] > 0.5)
        if len(observed_modules) < 2:
            continue
        eligible = []
        for target_index in np.flatnonzero(
            (batch_mask[row_index] > 0.0) & (future_mask[source_row] > 0.0)
        ):
            owners = np.flatnonzero(membership[:, target_index] > 0.5)
            if owners.size and any(
                int(owner) in observed_modules and len(observed_modules) > 1
                for owner in owners
            ):
                eligible.append(int(target_index))
        if not eligible:
            continue
        target_index = int(rng.choice(np.asarray(eligible, dtype=np.int64)))
        selected_targets[row_index, target_index] = 1.0
        batch_current[row_index, target_index] = 0.0
        batch_mask[row_index, target_index] = 0.0
        batch_ages[row_index, target_index] = 7.0
        batch_observations[:, :, target_index][row_index] = 0.0
        batch_observations[:, :, current.shape[1] + target_index][row_index] = 0.0
        batch_observations[:, :, 2 * current.shape[1] + target_index][row_index] = 7.0
        for module_index in np.flatnonzero(membership[:, target_index] > 0.5):
            remaining = np.flatnonzero(
                membership[module_index] > 0.5
            )
            if not np.any(batch_mask[row_index, remaining] > 0.0):
                batch_presence[row_index, module_index] = 0.0

    presence_start = 3 * current.shape[1] + 1
    for module_index in range(batch_presence.shape[1]):
        batch_observations[:, :, presence_start + module_index] = batch_presence[
            :, module_index, None
        ]
    return (
        batch_current,
        batch_mask,
        batch_ages,
        batch_presence,
        batch_observations,
        selected_targets,
    )


def _build_validated_edge_specs(
    module_names,
    variables,
    module_variables,
    allowed_keys=None,
):
    """Return conservative, previously validated cross-organ edges.

    This registry is deliberately small.  An edge adapter is allowed to learn
    only where an earlier held-out audit already found a reproducible
    observational relationship.  Adding an edge here is a scientific change,
    not a convenience default.
    """

    module_index = {name: index for index, name in enumerate(module_names)}
    variable_index = {name: index for index, name in enumerate(variables)}
    candidates = (
        ("cardiovascular_instability", "creatinine"),
        ("cardiovascular_instability", "bun"),
        ("sepsis", "map"),
    )
    # Transition tables intentionally repeat shared measurements such as
    # creatinine across several clinical modules.  Table membership is not
    # physiological ownership: treating every repeated copy as an owner made
    # cardio->creatinine look like a self-edge and silently discarded it.
    # These canonical owners are used only to construct masked edge training
    # views; the target value itself is still masked globally.
    canonical_target_owners = {
        "creatinine": "aki",
        "bun": "aki",
        "map": "cardiovascular_instability",
    }
    allowed = None if allowed_keys is None else {str(key) for key in allowed_keys}
    specs = []
    for source_module, target_variable in candidates:
        source_index = module_index.get(source_module)
        target_index = variable_index.get(target_variable)
        if source_index is None or target_index is None:
            continue
        preferred_owner = canonical_target_owners.get(target_variable)
        if preferred_owner in module_index:
            target_modules = (int(module_index[preferred_owner]),)
        else:
            target_modules = tuple(
                index
                for index, module in enumerate(module_names)
                if target_variable in set(module_variables.get(module, ()))
            )
        if not target_modules or source_index in target_modules:
            continue
        key = f"{source_module}->{target_variable}"
        if allowed is not None and key not in allowed:
            continue
        specs.append(
            {
                "key": key,
                "source_module": source_module,
                "source_module_index": int(source_index),
                "target_variable": target_variable,
                "target_index": int(target_index),
                "target_module_indices": [int(value) for value in target_modules],
            }
        )
    return tuple(specs)


def _build_source_group_adapter_specs(
    module_names,
    variables,
    module_variables,
    allowed_keys=None,
    contextual_regime_gate=False,
    treatment_history_columns=None,
):
    """Return only group adapters already supported by attribution evidence.

    The source-attribution registry authorizes an *adapter experiment*, never
    runtime movement.  Each spec therefore carries its allowed horizon and
    target-only masking policy into training.
    """

    module_index = {name: index for index, name in enumerate(module_names)}
    variable_index = {name: index for index, name in enumerate(variables)}
    treatment_history_columns = tuple(treatment_history_columns or ())
    treatment_index = {
        name: index for index, name in enumerate(treatment_history_columns)
    }
    treatment_start = 3 * len(variables) + 1 + len(module_names) + 1
    canonical_target_owners = {"creatinine": "aki", "bun": "aki", "map": "cardiovascular_instability"}
    allowed = None if allowed_keys is None else {str(key) for key in allowed_keys}
    specs = []
    for key, entry in ATTRIBUTION_VALIDATED_GROUP_ADAPTERS.items():
        if allowed is not None and key not in allowed:
            continue
        target_variable = str(entry["target"])
        target_index = variable_index.get(target_variable)
        source_modules = tuple(
            name for name in PHYSIOLOGY_GROUPS.get(str(entry["group"]), ())
            if name in module_index
        )
        if target_index is None or not source_modules:
            continue
        preferred_owner = canonical_target_owners.get(target_variable)
        if preferred_owner in module_index:
            target_modules = (int(module_index[preferred_owner]),)
        else:
            target_modules = tuple(
                index
                for index, module in enumerate(module_names)
                if target_variable in set(module_variables.get(module, ()))
            )
        if not target_modules:
            continue
        source_variable_indices = tuple(
            index
            for index, variable in enumerate(variables)
            if index != int(target_index)
            and any(
                variable in set(module_variables.get(module, ()))
                for module in source_modules
            )
        )
        if not source_variable_indices:
            continue
        selected_treatment_fields = tuple(
            field
            for field in entry.get("treatment_history_fields", ())
            if field in treatment_index
        )
        selected_treatment_indices = tuple(
            treatment_index[field] for field in selected_treatment_fields
        )
        treatment_context_indices = []
        for offset in (
            0,
            len(treatment_history_columns),
            2 * len(treatment_history_columns),
        ):
            treatment_context_indices.extend(
                treatment_start + offset + index
                for index in selected_treatment_indices
            )
        specs.append(
            {
                "key": str(key),
                "source_group": str(entry["group"]),
                "source_modules": list(source_modules),
                "source_module_indices": [int(module_index[name]) for name in source_modules],
                "target_variable": target_variable,
                "target_index": int(target_index),
                "target_module_indices": [int(value) for value in target_modules],
                "allowed_horizons": [float(value) for value in entry["horizons"]],
                "target_mask_scope": str(entry["target_mask_scope"]),
                "evidence": str(entry["evidence"]),
                # This is an experiment-only option.  It gives the adapter a
                # direct, as-of-only view of the attributed source variables
                # and a per-patient gate.  It never changes the attribution
                # evidence or authorizes runtime movement by itself.
                "contextual_regime_gate": bool(contextual_regime_gate),
                "source_variable_indices": [int(value) for value in source_variable_indices],
                "treatment_history_fields": list(selected_treatment_fields),
                "treatment_context_indices": [int(value) for value in treatment_context_indices],
            }
        )
    return tuple(specs)


def _validated_edge_batch_view(
    batch,
    current,
    mask,
    ages,
    presence,
    input_matrix,
    history,
    future_mask,
    horizon,
    edge_specs,
    variable_module_membership,
):
    """Mask the target organ while retaining its validated source organ.

    The loss mask contains only the selected edge target.  Rows without a
    valid source/target pair contribute no edge loss, so missingness is not
    silently converted into a negative physiological relation.
    """

    batch_current = current[batch].copy()
    batch_mask = mask[batch].copy()
    batch_ages = ages[batch].copy()
    batch_presence = presence[batch].copy()
    batch_observations = input_matrix[history[batch]].copy()
    selected_targets = np.zeros((len(batch), current.shape[1]), dtype=np.float32)
    for row_index, source_row in enumerate(batch):
        eligible = []
        for spec in edge_specs:
            target_index = int(spec["target_index"])
            source_values = spec.get("source_module_indices")
            if source_values is None:
                source_values = (int(spec["source_module_index"]),)
            source_modules = tuple(int(value) for value in source_values)
            target_modules = tuple(int(value) for value in spec["target_module_indices"])
            if not any(batch_presence[row_index, source_module] > 0.5 for source_module in source_modules):
                continue
            allowed_horizons = tuple(float(value) for value in spec.get("allowed_horizons", ()))
            if allowed_horizons and not any(
                np.isclose(float(horizon[source_row]), value, atol=1e-4)
                for value in allowed_horizons
            ):
                continue
            if not any(batch_presence[row_index, value] > 0.5 for value in target_modules):
                continue
            if batch_mask[row_index, target_index] <= 0.0:
                continue
            if future_mask[source_row, target_index] <= 0.0:
                continue
            eligible.append(spec)
        if not eligible:
            continue
        spec = eligible[row_index % len(eligible)]
        target_index = int(spec["target_index"])
        selected_targets[row_index, target_index] = 1.0
        n_state = current.shape[1]
        target_modules = tuple(int(value) for value in spec["target_module_indices"])
        if str(spec.get("target_mask_scope", "target_module")) == "target_only":
            target_indices = np.asarray([target_index], dtype=np.int64)
        else:
            target_indices = np.flatnonzero(
                np.any(
                    np.asarray(variable_module_membership, dtype=np.float32)[
                        list(target_modules)
                    ]
                    > 0.5,
                    axis=0,
                )
            )
        # Single-edge adapters hide the target organ.  The attribution-backed
        # metabolic-renal adapter hides only creatinine, retaining BUN and
        # electrolyte context exactly as in the source-ablation experiment.
        for target_index in target_indices:
            batch_current[row_index, target_index] = 0.0
            batch_mask[row_index, target_index] = 0.0
            batch_ages[row_index, target_index] = 7.0
            batch_observations[row_index, :, target_index] = 0.0
            batch_observations[row_index, :, n_state + target_index] = 0.0
            batch_observations[row_index, :, 2 * n_state + target_index] = 7.0
        if str(spec.get("target_mask_scope", "target_module")) != "target_only":
            for module_index in target_modules:
                batch_presence[row_index, module_index] = 0.0
                presence_start = 3 * n_state + 1
                batch_observations[row_index, :, presence_start + module_index] = 0.0
    return (
        batch_current,
        batch_mask,
        batch_ages,
        batch_presence,
        batch_observations,
        selected_targets,
    )


def fit_joint(
    frame,
    variables,
    module_names,
    rows,
    seed=7,
    epochs=2,
    batch_size=2048,
    cross_forecast_weight=0.0,
    measurement_process_mode="observed",
    masked_forecast_probability=0.50,
    module_mask_probability=0.75,
    nowcast_pretrain_epochs=0,
    measurement_consistency_weight=0.0,
    world_model=False,
    world_model_weight=1.0,
    target_momentum=0.99,
    future_target_mode="window",
    target_min_observations=2,
    regime_balanced_loss=False,
    regime_count=3,
    state_normalization="standard",
    hospital_invariance_weight=0.0,
    domain_invariance_columns=("hospitalid",),
    target_encoder_mode="fast",
    target_horizon_regime_adapter=False,
    patient_residual_adapter=False,
    future_task_balanced_sampling=False,
    coupling_preservation_weight=0.0,
    uncertainty_gate=False,
    validated_edge_adapters=False,
    validated_edge_weight=0.0,
    validated_edge_stage_epochs=0,
    validated_edge_targets=None,
    target_head_stage_epochs=0,
    source_group_adapters=False,
    source_group_contextual_regime_gate=False,
    source_group_targeted_treatment_context=False,
    source_group_adapter_weight=0.0,
    source_group_adapter_stage_epochs=0,
    source_group_adapter_targets=None,
    supervised_targets=None,
    measurement_time_head=False,
    measurement_time_weight=0.10,
    uncertainty_calibration_weight=0.05,
    include_treatment_context=False,
    belief_input_mode="observed",
):
    if future_target_mode not in {"window", "single"}:
        raise ValueError("future_target_mode must be 'window' or 'single'")
    if int(target_min_observations) < 1:
        raise ValueError("target_min_observations must be at least 1")
    if state_normalization not in {"standard", "robust"}:
        raise ValueError("state_normalization must be 'standard' or 'robust'")
    if float(hospital_invariance_weight) < 0.0:
        raise ValueError("hospital_invariance_weight must be non-negative")
    if isinstance(domain_invariance_columns, str):
        domain_invariance_columns = tuple(
            value.strip()
            for value in domain_invariance_columns.split(",")
            if value.strip()
        )
    else:
        domain_invariance_columns = tuple(domain_invariance_columns)
    if not domain_invariance_columns:
        raise ValueError("domain_invariance_columns must not be empty")
    unknown_domain_columns = sorted(
        set(domain_invariance_columns) - set(frame.columns)
    )
    if unknown_domain_columns:
        raise ValueError(
            "domain invariance columns missing from frame: "
            f"{unknown_domain_columns}"
        )
    if target_encoder_mode not in {"full", "fast"}:
        raise ValueError("target_encoder_mode must be 'full' or 'fast'")
    if not isinstance(target_horizon_regime_adapter, (bool, np.bool_)):
        raise ValueError("target_horizon_regime_adapter must be boolean")
    if float(coupling_preservation_weight) < 0.0:
        raise ValueError("coupling_preservation_weight must be non-negative")
    if float(validated_edge_weight) < 0.0:
        raise ValueError("validated_edge_weight must be non-negative")
    if int(validated_edge_stage_epochs) < 0:
        raise ValueError("validated_edge_stage_epochs must be non-negative")
    if int(target_head_stage_epochs) < 0:
        raise ValueError("target_head_stage_epochs must be non-negative")
    if float(source_group_adapter_weight) < 0.0:
        raise ValueError("source_group_adapter_weight must be non-negative")
    if int(source_group_adapter_stage_epochs) < 0:
        raise ValueError("source_group_adapter_stage_epochs must be non-negative")
    if bool(validated_edge_adapters) and bool(source_group_adapters):
        raise ValueError(
            "Run single-edge and source-group adapter experiments separately so their effects remain identifiable"
        )
    if bool(source_group_contextual_regime_gate) and not bool(source_group_adapters):
        raise ValueError(
            "source_group_contextual_regime_gate requires source_group_adapters"
        )
    if bool(source_group_targeted_treatment_context) and not (
        bool(source_group_adapters)
        and bool(source_group_contextual_regime_gate)
        and bool(include_treatment_context)
    ):
        raise ValueError(
            "source_group_targeted_treatment_context requires treatment history, "
            "source_group_adapters, and source_group_contextual_regime_gate"
        )
    if float(measurement_time_weight) < 0.0:
        raise ValueError("measurement_time_weight must be non-negative")
    if float(uncertainty_calibration_weight) < 0.0:
        raise ValueError("uncertainty_calibration_weight must be non-negative")
    if belief_input_mode not in {
        "observed",
        "current_only",
        "placebo_time_only",
    }:
        raise ValueError(
            "belief_input_mode must be observed, current_only, or "
            "placebo_time_only"
        )
    if supervised_targets is not None:
        unknown_targets = sorted(set(supervised_targets) - set(variables))
        if unknown_targets:
            raise ValueError(f"supervised_targets not in variables: {unknown_targets}")
    set_seed(seed)
    arrays = _joint_arrays(
        frame,
        variables,
        module_names,
        rows,
        measurement_process_mode=measurement_process_mode,
        state_normalization=state_normalization,
        include_treatment_context=include_treatment_context,
    )
    (
        _, input_matrix, current, mask, ages, future, future_mask, history,
        horizon, presence, future_window, future_window_mask, future_delta_t,
        future_window_delta_t, window_horizons, next_measurement_time,
        next_measurement_mask,
    ) = arrays
    regime_ids, delta_bin_ids, regime_metadata = _build_supervision_strata(
        arrays, rows, regime_count=regime_count
    )
    supervised_mask = np.ones(len(variables), dtype=np.float32)
    if supervised_targets is not None:
        supervised_mask[:] = 0.0
        for target in supervised_targets:
            supervised_mask[variables.index(target)] = 1.0
    module_variables = frame.attrs.get("module_variables", {})
    single_edge_specs = _build_validated_edge_specs(
        module_names,
        variables,
        module_variables,
        allowed_keys=validated_edge_targets,
    )
    source_group_specs = (
        _build_source_group_adapter_specs(
            module_names,
            variables,
            module_variables,
            allowed_keys=source_group_adapter_targets,
            contextual_regime_gate=source_group_contextual_regime_gate,
            treatment_history_columns=(
                frame.attrs.get("treatment_history_columns", ())
                if bool(source_group_targeted_treatment_context)
                else ()
            ),
        )
        if bool(source_group_adapters)
        else ()
    )
    validated_edge_specs = tuple(single_edge_specs) + tuple(source_group_specs)
    edge_adapters_enabled = bool(validated_edge_adapters or source_group_adapters)
    edge_weight = (
        float(source_group_adapter_weight)
        if bool(source_group_adapters)
        else float(validated_edge_weight)
    )
    edge_stage_epochs = (
        int(source_group_adapter_stage_epochs)
        if bool(source_group_adapters)
        else int(validated_edge_stage_epochs)
    )
    variable_module_ids = []
    variable_module_membership = np.zeros(
        (len(module_names), len(variables)), dtype=np.float32
    )
    for variable in variables:
        owners = [
            index for index, module in enumerate(module_names)
            if variable in set(module_variables.get(module, ()))
        ]
        variable_module_ids.append(owners[0] if owners else 0)
        for owner in owners:
            variable_module_membership[owner, len(variable_module_ids) - 1] = 1.0
    model_class = JointWholeBodyJEPAWorldModel if world_model else JointWholeBodyJEPA
    model_kwargs = {}
    if world_model:
        model_kwargs["target_momentum"] = target_momentum
        model_kwargs["target_min_observations"] = int(target_min_observations)
        model_kwargs["target_encoder_mode"] = target_encoder_mode
    model_kwargs["horizon_values"] = window_horizons
    model_kwargs["regime_count"] = int(regime_count)
    model_kwargs["target_horizon_regime_adapter"] = bool(
        target_horizon_regime_adapter
    )
    model_kwargs["patient_residual_adapter"] = bool(patient_residual_adapter)
    model_kwargs["uncertainty_gate"] = bool(uncertainty_gate)
    model_kwargs["validated_edge_adapters"] = edge_adapters_enabled
    model_kwargs["validated_edge_specs"] = validated_edge_specs
    model_kwargs["adapter_only_treatment_context"] = bool(
        source_group_targeted_treatment_context
    )
    model_kwargs["measurement_time_head"] = bool(
        measurement_time_head and bool(next_measurement_mask.any())
    )
    model_kwargs["belief_input_mode"] = str(belief_input_mode)
    model = model_class(
        len(variables),
        input_matrix.shape[1],
        len(module_names),
        variable_module_ids=variable_module_ids,
        variable_module_membership=variable_module_membership,
        **model_kwargs,
    )
    model.configure_supervision_strata(regime_metadata)
    # Domain identity is never an inference feature. During training only, an
    # adversarial head tries to recover the requested provenance combination
    # while gradient reversal makes the physiology encoder remove that proxy.
    domain_parts = []
    for column in domain_invariance_columns:
        values = frame[column].fillna("__missing__").astype(str)
        domain_parts.append(column + "=" + values)
    domain_key_series = domain_parts[0]
    for values in domain_parts[1:]:
        domain_key_series = domain_key_series + "|" + values
    domain_keys = domain_key_series.to_numpy()
    fit_domain_keys = sorted(
        set(domain_keys[np.asarray(rows, dtype=np.int64)].tolist())
    )
    domain_to_index = {
        key: index for index, key in enumerate(fit_domain_keys)
    }
    domain_labels = np.asarray(
        [domain_to_index.get(key, -1) for key in domain_keys], dtype=np.int64
    )
    domain_classifier = None
    domain_class_weights = None
    if float(hospital_invariance_weight) > 0.0 and len(fit_domain_keys) >= 2:
        domain_hidden = max(16, model.hidden // 2)
        domain_classifier = nn.Sequential(
            nn.Linear(model.latent, domain_hidden),
            nn.LayerNorm(domain_hidden),
            nn.SiLU(),
            nn.Linear(domain_hidden, len(fit_domain_keys)),
        )
        counts = np.bincount(
            domain_labels[np.asarray(rows, dtype=np.int64)],
            minlength=len(fit_domain_keys),
        ).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights /= max(float(weights.mean()), 1e-6)
        domain_class_weights = torch.from_numpy(weights.astype(np.float32))
    optimizer_parameters = list(model.parameters())
    if domain_classifier is not None:
        optimizer_parameters.extend(domain_classifier.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    nowcast_losses = []
    module_variables = frame.attrs.get("module_variables", {})

    # Stage 1: learn same-anchor body structure before asking the model to
    # learn sparse future labels.  This is a masked nowcast objective, not a
    # future-label shortcut: the target is the original current observation.
    for _ in range(int(nowcast_pretrain_epochs)):
        epoch = []
        model.train()
        for batch in _balanced_batches(frame, rows, batch_size, rng):
            view = _masked_batch_view(
                batch, current, mask, ages, presence, input_matrix, history,
                variables, module_names, module_variables, rng,
                masked_forecast_probability, module_mask_probability,
            )
            batch_current, batch_mask, batch_ages, batch_presence, batch_observations, drop = view
            output = model(
                torch.from_numpy(batch_observations).float(),
                torch.from_numpy(batch_current).float(),
                torch.from_numpy(batch_mask).float(),
                torch.from_numpy(batch_ages).float(),
                torch.from_numpy(batch_presence).float(),
                torch.from_numpy(horizon[batch]).float(),
            )
            reconstruction_target = torch.from_numpy(current[batch]).float()
            reconstruction_mask = torch.from_numpy(drop.astype(np.float32)).float()
            loss = _target_balanced_loss(
                output["reconstruction"], reconstruction_target, reconstruction_mask
            )
            if measurement_consistency_weight > 0.0:
                second = _masked_batch_view(
                    batch, current, mask, ages, presence, input_matrix, history,
                    variables, module_names, module_variables, rng,
                    masked_forecast_probability, module_mask_probability,
                )
                second_output = model(
                    torch.from_numpy(second[4]).float(),
                    torch.from_numpy(second[0]).float(),
                    torch.from_numpy(second[1]).float(),
                    torch.from_numpy(second[2]).float(),
                    torch.from_numpy(second[3]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )
                loss = loss + float(measurement_consistency_weight) * _latent_consistency_loss(
                    output["latent"], second_output["latent"]
                )
            if domain_classifier is not None:
                domain_latent = output.get("context_latent")
                if domain_latent is None:
                    domain_latent = output["latent"]
                batch_domains = torch.from_numpy(domain_labels[batch]).long()
                valid_domains = batch_domains >= 0
                if bool(valid_domains.any()):
                    domain_logits = domain_classifier(
                        gradient_reverse(domain_latent[valid_domains], 1.0)
                    )
                    domain_loss = nn.functional.cross_entropy(
                        domain_logits,
                        batch_domains[valid_domains],
                        weight=domain_class_weights,
                    )
            loss = loss + float(hospital_invariance_weight) * domain_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, 5.0)
            optimizer.step()
            epoch.append(float(loss.detach()))
        nowcast_losses.append(float(np.mean(epoch)))
    losses.extend(nowcast_losses)

    for epoch_index in range(epochs):
        epoch = []
        model.train()
        # Curriculum: first preserve ordinary factual forecasting, then
        # gradually add the harder cross-organ objective. Optimizing both at
        # full strength from step one made the shared latent sacrifice local
        # target accuracy before it had learned a stable body representation.
        epoch_cross_weight = float(cross_forecast_weight) * (
            epoch_index / max(1, epochs - 1)
        )
        batch_iterator = (
            _future_task_balanced_batches(
                frame,
                rows,
                future_mask,
                delta_bin_ids,
                regime_ids,
                batch_size,
                rng,
            )
            if future_task_balanced_sampling
            else _balanced_batches(frame, rows, batch_size, rng)
        )
        for batch in batch_iterator:
            view = _masked_batch_view(
                batch, current, mask, ages, presence, input_matrix, history,
                variables, module_names, module_variables, rng,
                masked_forecast_probability, module_mask_probability,
            )
            batch_current, batch_mask, batch_ages, batch_presence, batch_observations, drop = view
            y = torch.from_numpy(future[batch]).float()
            target_mask = torch.from_numpy(future_mask[batch]).float()
            target_mask = target_mask * torch.from_numpy(supervised_mask).float()
            if world_model:
                target_kwargs = {
                    "future": y,
                    "future_mask": target_mask,
                    "target_horizon": torch.from_numpy(
                        future_delta_t[batch]
                    ).float(),
                }
                if future_target_mode == "window":
                    target_kwargs.update(
                        {
                            "future_window": torch.from_numpy(
                                future_window[batch]
                            ).float(),
                    "future_window_mask": torch.from_numpy(
                        future_window_mask[batch]
                    ).float(),
                            "future_window_delta_t": torch.from_numpy(
                                future_window_delta_t[batch]
                            ).float(),
                    "window_horizons": torch.from_numpy(
                        window_horizons
                    ).float(),
                        }
                    )
                output = model(
                    torch.from_numpy(batch_observations).float(),
                    torch.from_numpy(batch_current).float(),
                    torch.from_numpy(batch_mask).float(),
                    torch.from_numpy(batch_ages).float(),
                    torch.from_numpy(batch_presence).float(),
                    torch.from_numpy(horizon[batch]).float(),
                    **target_kwargs,
                )
            else:
                output = model(
                    torch.from_numpy(batch_observations).float(),
                    torch.from_numpy(batch_current).float(),
                    torch.from_numpy(batch_mask).float(),
                    torch.from_numpy(batch_ages).float(),
                    torch.from_numpy(batch_presence).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )
            if regime_balanced_loss:
                value_loss = _regime_balanced_value_loss(
                    output["value"],
                    y,
                    target_mask,
                    delta_bin_ids[batch],
                    regime_ids[batch],
                )
            else:
                value_loss = _target_balanced_loss(output["value"], y, target_mask)
            measurement_time_loss = output["value"].new_zeros(())
            if (
                model.measurement_time_head
                and float(measurement_time_weight) > 0.0
                and bool(next_measurement_mask[batch].any())
            ):
                observed_time = torch.from_numpy(
                    next_measurement_mask[batch]
                ).float()
                time_target = torch.from_numpy(
                    np.log1p(
                        np.nan_to_num(
                            next_measurement_time[batch],
                            nan=168.0,
                            posinf=168.0,
                            neginf=168.0,
                        )
                    )
                ).float()
                time_prediction = nn.functional.softplus(
                    output["next_measurement_time_log"]
                )
                measurement_time_loss = _target_balanced_loss(
                    time_prediction, time_target, observed_time
                )
            uncertainty_calibration_loss = _target_balanced_scale_loss(
                output["value"],
                y,
                output["value_scale"],
                target_mask,
            )
            obs_loss = nn.functional.binary_cross_entropy_with_logits(output["observation_logit"], target_mask)
            reconstruction_target = torch.from_numpy(current[batch]).float()
            reconstruction_mask = torch.from_numpy(drop.astype(np.float32)).float()
            reconstruction_loss = _target_balanced_loss(
                output["reconstruction"], reconstruction_target, reconstruction_mask
            )
            cross_forecast_mask = target_mask * reconstruction_mask
            if regime_balanced_loss:
                cross_forecast_loss = _regime_balanced_value_loss(
                    output["value"],
                    y,
                    cross_forecast_mask,
                    delta_bin_ids[batch],
                    regime_ids[batch],
                )
            else:
                cross_forecast_loss = _target_balanced_loss(
                    output["value"], y, cross_forecast_mask
                )
            coupling_loss = output["value"].new_zeros(())
            if float(coupling_preservation_weight) > 0.0:
                coupling_view = _coupling_batch_view(
                    batch,
                    current,
                    mask,
                    ages,
                    presence,
                    input_matrix,
                    history,
                    future_mask,
                    variable_module_membership,
                    rng,
                )
                coupling_kwargs = {}
                if world_model:
                    coupling_kwargs["target_horizon"] = torch.from_numpy(
                        future_delta_t[batch]
                    ).float()
                coupling_output = model(
                    torch.from_numpy(coupling_view[4]).float(),
                    torch.from_numpy(coupling_view[0]).float(),
                    torch.from_numpy(coupling_view[1]).float(),
                    torch.from_numpy(coupling_view[2]).float(),
                    torch.from_numpy(coupling_view[3]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                    **coupling_kwargs,
                )
                coupling_mask = target_mask * torch.from_numpy(
                    coupling_view[5]
                ).float()
                if regime_balanced_loss:
                    coupling_loss = _regime_balanced_value_loss(
                        coupling_output["value"],
                        y,
                        coupling_mask,
                        delta_bin_ids[batch],
                        regime_ids[batch],
                    )
                else:
                    coupling_loss = _target_balanced_loss(
                        coupling_output["value"], y, coupling_mask
                    )
            edge_loss = output["value"].new_zeros(())
            joint_edge_weight = (
                0.0
                if int(edge_stage_epochs) > 0
                else float(edge_weight)
            )
            if joint_edge_weight > 0.0 and validated_edge_specs:
                edge_view = _validated_edge_batch_view(
                    batch,
                    current,
                    mask,
                    ages,
                    presence,
                    input_matrix,
                    history,
                    future_mask,
                    horizon,
                    validated_edge_specs,
                    variable_module_membership,
                )
                edge_output = model(
                    torch.from_numpy(edge_view[4]).float(),
                    torch.from_numpy(edge_view[0]).float(),
                    torch.from_numpy(edge_view[1]).float(),
                    torch.from_numpy(edge_view[2]).float(),
                    torch.from_numpy(edge_view[3]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )
                edge_mask = target_mask * torch.from_numpy(edge_view[5]).float()
                if bool(edge_mask.any()):
                    if regime_balanced_loss:
                        edge_loss = _regime_balanced_value_loss(
                            edge_output["value"],
                            y,
                            edge_mask,
                            delta_bin_ids[batch],
                            regime_ids[batch],
                        )
                    else:
                        edge_loss = _target_balanced_loss(
                            edge_output["value"], y, edge_mask
                        )
            loss = (
                value_loss
                + 0.2 * obs_loss
                + 0.3 * reconstruction_loss
                + epoch_cross_weight * cross_forecast_loss
                + float(coupling_preservation_weight) * coupling_loss
                + joint_edge_weight * edge_loss
                + float(measurement_time_weight) * measurement_time_loss
                + float(uncertainty_calibration_weight)
                * uncertainty_calibration_loss
            )
            if world_model:
                shared_jepa_loss = _masked_latent_prediction_loss(
                    output["predicted_trajectory_latent"],
                    output["target_trajectory_latent"],
                    output["target_step_available"],
                )
                target_jepa_loss = _masked_target_specific_latent_prediction_loss(
                    output["predicted_target_trajectory_latent"],
                    output["target_specific_trajectory_latent"],
                    output["target_specific_available"],
                    delta_bins=_actual_delta_bin_ids(future_window_delta_t[batch]),
                    regimes=regime_ids[batch],
                )
                loss = loss + float(world_model_weight) * 0.5 * (
                    shared_jepa_loss + target_jepa_loss
                )
            if measurement_consistency_weight > 0.0:
                second = _masked_batch_view(
                    batch, current, mask, ages, presence, input_matrix, history,
                    variables, module_names, module_variables, rng,
                    masked_forecast_probability, module_mask_probability,
                )
                second_output = model(
                    torch.from_numpy(second[4]).float(),
                    torch.from_numpy(second[0]).float(),
                    torch.from_numpy(second[1]).float(),
                    torch.from_numpy(second[2]).float(),
                    torch.from_numpy(second[3]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )
                loss = loss + float(measurement_consistency_weight) * _latent_consistency_loss(
                    output["latent"], second_output["latent"]
                )
            if domain_classifier is not None:
                domain_latent = output.get("context_latent")
                if domain_latent is None:
                    domain_latent = output["latent"]
                batch_domains = torch.from_numpy(domain_labels[batch]).long()
                valid_domains = batch_domains >= 0
                if bool(valid_domains.any()):
                    domain_logits = domain_classifier(
                        gradient_reverse(domain_latent[valid_domains], 1.0)
                    )
                    domain_loss = nn.functional.cross_entropy(
                        domain_logits,
                        batch_domains[valid_domains],
                        weight=domain_class_weights,
                    )
                    loss = loss + float(hospital_invariance_weight) * domain_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, 5.0)
            optimizer.step()
            if world_model:
                model.update_target_encoder()
            epoch.append(float(loss.detach()))
        losses.append(float(np.mean(epoch)))

    # Stage 2: preserve the jointly trained whole-body representation, then
    # let every target-specific temporal/value/uncertainty branch specialize
    # without moving the shared physiology latent. This is deliberately
    # applied to all supervised targets rather than test-selected cells.
    if int(target_head_stage_epochs) > 0:
        target_parameters = (
            list(model.target_temporal_adapters.parameters())
            + [model.target_temporal_gates]
            + list(model.value_heads.parameters())
            + list(model.value_scale_heads.parameters())
        )
        if model.patient_residual_adapter_enabled:
            target_parameters += list(model.patient_residual_heads.parameters())
            target_parameters += [model.patient_residual_gates]
        if model.target_horizon_regime_adapter_enabled:
            target_parameters += [
                model.target_horizon_regime_table,
                model.target_horizon_regime_gates,
                model.target_horizon_regime_value_bias,
                model.target_horizon_regime_value_gates,
            ]
        if model.uncertainty_gate_enabled:
            target_parameters += list(model.uncertainty_gate_heads.parameters())
        if world_model:
            target_parameters += list(model.target_future_predictors.parameters())
            target_parameters += [model.target_future_gates]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in target_parameters:
            parameter.requires_grad_(True)
        target_optimizer = torch.optim.AdamW(
            target_parameters, lr=3e-4, weight_decay=1e-4
        )
        for _ in range(int(target_head_stage_epochs)):
            target_epoch = []
            model.train()
            batch_iterator = (
                _future_task_balanced_batches(
                    frame,
                    rows,
                    future_mask,
                    delta_bin_ids,
                    regime_ids,
                    batch_size,
                    rng,
                )
                if future_task_balanced_sampling
                else _balanced_batches(frame, rows, batch_size, rng)
            )
            for batch in batch_iterator:
                view = _masked_batch_view(
                    batch,
                    current,
                    mask,
                    ages,
                    presence,
                    input_matrix,
                    history,
                    variables,
                    module_names,
                    module_variables,
                    rng,
                    masked_forecast_probability,
                    module_mask_probability,
                )
                target_mask = torch.from_numpy(future_mask[batch]).float()
                target_mask = target_mask * torch.from_numpy(
                    supervised_mask
                ).float()
                y = torch.from_numpy(future[batch]).float()
                target_kwargs = {}
                if world_model:
                    target_kwargs = {
                        "future": y,
                        "future_mask": target_mask,
                        "target_horizon": torch.from_numpy(
                            future_delta_t[batch]
                        ).float(),
                    }
                    if future_target_mode == "window":
                        target_kwargs.update(
                            {
                                "future_window": torch.from_numpy(
                                    future_window[batch]
                                ).float(),
                                "future_window_mask": torch.from_numpy(
                                    future_window_mask[batch]
                                ).float(),
                                "future_window_delta_t": torch.from_numpy(
                                    future_window_delta_t[batch]
                                ).float(),
                                "window_horizons": torch.from_numpy(
                                    window_horizons
                                ).float(),
                            }
                        )
                output = model(
                    torch.from_numpy(view[4]).float(),
                    torch.from_numpy(view[0]).float(),
                    torch.from_numpy(view[1]).float(),
                    torch.from_numpy(view[2]).float(),
                    torch.from_numpy(view[3]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                    **target_kwargs,
                )
                if regime_balanced_loss:
                    target_loss = _regime_balanced_value_loss(
                        output["value"],
                        y,
                        target_mask,
                        delta_bin_ids[batch],
                        regime_ids[batch],
                    )
                else:
                    target_loss = _target_balanced_loss(
                        output["value"], y, target_mask
                    )
                target_loss = target_loss + float(
                    uncertainty_calibration_weight
                ) * _target_balanced_scale_loss(
                    output["value"],
                    y,
                    output["value_scale"],
                    target_mask,
                )
                if world_model:
                    target_loss = target_loss + 0.5 * float(
                        world_model_weight
                    ) * _masked_target_specific_latent_prediction_loss(
                        output["predicted_target_trajectory_latent"],
                        output["target_specific_trajectory_latent"],
                        output["target_specific_available"],
                        delta_bins=_actual_delta_bin_ids(
                            future_window_delta_t[batch]
                        ),
                        regimes=regime_ids[batch],
                    )
                target_optimizer.zero_grad(set_to_none=True)
                target_loss.backward()
                torch.nn.utils.clip_grad_norm_(target_parameters, 5.0)
                target_optimizer.step()
                target_epoch.append(float(target_loss.detach()))
            if target_epoch:
                losses.append(float(np.mean(target_epoch)))

    # Stage 3: preserve the trained body representation and local forecasts,
    # then fit only the small, pre-validated edge adapters.  Joint training
    # made the edge objective compete with the factual objective; this stage
    # keeps the two responsibilities separate.
    if (
        int(edge_stage_epochs) > 0
        and validated_edge_specs
        and edge_adapters_enabled
        and hasattr(model, "edge_value_heads")
    ):
        edge_parameters = list(model.edge_value_heads.parameters()) + list(
            model.edge_gates.parameters()
        )
        if hasattr(model, "edge_source_context_encoders"):
            edge_parameters += list(model.edge_source_context_encoders.parameters())
        if hasattr(model, "edge_treatment_context_encoders"):
            edge_parameters += list(model.edge_treatment_context_encoders.parameters())
        if hasattr(model, "edge_contextual_gates"):
            edge_parameters += list(model.edge_contextual_gates.parameters())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in edge_parameters:
            parameter.requires_grad_(True)
        edge_optimizer = torch.optim.AdamW(
            edge_parameters, lr=4e-4, weight_decay=1e-4
        )
        for _ in range(int(edge_stage_epochs)):
            edge_epoch = []
            model.train()
            for batch in _balanced_batches(frame, rows, batch_size, rng):
                edge_view = _validated_edge_batch_view(
                    batch,
                    current,
                    mask,
                    ages,
                    presence,
                    input_matrix,
                    history,
                    future_mask,
                    horizon,
                    validated_edge_specs,
                    variable_module_membership,
                )
                edge_mask = torch.from_numpy(
                    future_mask[batch] * edge_view[5]
                ).float()
                if not bool(edge_mask.any()):
                    continue
                edge_output = model(
                    torch.from_numpy(edge_view[4]).float(),
                    torch.from_numpy(edge_view[0]).float(),
                    torch.from_numpy(edge_view[1]).float(),
                    torch.from_numpy(edge_view[2]).float(),
                    torch.from_numpy(edge_view[3]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )
                if regime_balanced_loss:
                    edge_loss = _regime_balanced_value_loss(
                        edge_output["value"],
                        torch.from_numpy(future[batch]).float(),
                        edge_mask,
                        delta_bin_ids[batch],
                        regime_ids[batch],
                    )
                else:
                    edge_loss = _target_balanced_loss(
                        edge_output["value"],
                        torch.from_numpy(future[batch]).float(),
                        edge_mask,
                    )
                edge_optimizer.zero_grad(set_to_none=True)
                edge_loss.backward()
                torch.nn.utils.clip_grad_norm_(edge_parameters, 5.0)
                edge_optimizer.step()
                edge_epoch.append(float(edge_loss.detach()))
            if edge_epoch:
                losses.append(float(np.mean(edge_epoch)))
    return model, arrays, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument(
        "--modules",
        default="sepsis,aki,respiratory,integumentary_skin_wound,toxic_metabolic,electrolyte_acid_base,endocrine_stress,gi_pancreatic_nutrition,cardiac_injury,musculoskeletal_rhabdo,immune_inflammatory,cardiovascular_instability,acute_neuro,hepatic_failure,coagulopathy_heme",
    )
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--alignment", choices=("exact", "asof"), default="asof")
    parser.add_argument("--asof-max-age-hours", type=float, default=6.0)
    parser.add_argument(
        "--asof-age-policy",
        choices=("fixed", "horizon_scaled"),
        default="horizon_scaled",
    )
    parser.add_argument(
        "--variable-age-limits",
        type=Path,
        default=None,
        help="Optional JSON object mapping variable names to maximum input age in hours.",
    )
    parser.add_argument("--future-label-tolerance-hours", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cross-forecast-weight", type=float, default=0.0)
    parser.add_argument("--masked-forecast-probability", type=float, default=0.50)
    parser.add_argument("--module-mask-probability", type=float, default=0.75)
    parser.add_argument(
        "--nowcast-pretrain-epochs",
        type=int,
        default=0,
        help="Masked same-anchor reconstruction epochs before future forecasting.",
    )
    parser.add_argument(
        "--measurement-consistency-weight",
        type=float,
        default=0.0,
        help="Weight for latent consistency across two independent missingness views.",
    )
    parser.add_argument(
        "--world-model",
        action="store_true",
        help="Train the true JEPA latent-prediction candidate with an EMA target encoder.",
    )
    parser.add_argument("--world-model-weight", type=float, default=1.0)
    parser.add_argument("--target-momentum", type=float, default=0.99)
    parser.add_argument(
        "--future-target-mode",
        choices=("window", "single"),
        default="window",
        help="JEPA target: causal future trajectory prefix or one future row.",
    )
    parser.add_argument(
        "--target-min-observations",
        type=int,
        default=2,
        help="Minimum observed variables required for a future JEPA step.",
    )
    parser.add_argument(
        "--regime-balanced-loss",
        action="store_true",
        help="Balance future supervision by target, actual delta-time bin, and anchor regime.",
    )
    parser.add_argument(
        "--regime-count",
        type=int,
        default=3,
        help="Number of target-specific anchor regimes used by the balanced loss.",
    )
    parser.add_argument(
        "--measurement-process-mode",
        choices=("observed", "neutralized"),
        default="observed",
    )
    parser.add_argument(
        "--state-normalization",
        choices=("standard", "robust"),
        default="standard",
        help="Training-only global normalization; robust uses median/MAD.",
    )
    parser.add_argument(
        "--hospital-invariance-weight",
        type=float,
        default=0.0,
        help="Training-only adversarial hospital-domain loss weight.",
    )
    parser.add_argument(
        "--domain-invariance-columns",
        default="hospitalid",
        help=(
            "Comma-separated provenance columns removed adversarially from "
            "the physiology latent."
        ),
    )
    parser.add_argument(
        "--target-encoder-mode",
        choices=("full", "fast"),
        default="fast",
        help="Future JEPA target encoder: full mixers or fast stop-gradient view.",
    )
    parser.add_argument(
        "--target-horizon-regime-adapter",
        action="store_true",
        help="Enable target-specific horizon x anchor-regime residual adapters.",
    )
    parser.add_argument(
        "--patient-residual-adapter",
        action="store_true",
        help="Enable target-specific residuals derived from each patient's history latent.",
    )
    parser.add_argument(
        "--future-task-balanced-sampling",
        action="store_true",
        help="Sample future task cells by inverse target-horizon-regime frequency.",
    )
    parser.add_argument(
        "--coupling-preservation-weight",
        type=float,
        default=0.0,
        help="Explicit loss weight for target-ablated cross-organ future views.",
    )
    parser.add_argument(
        "--uncertainty-gate",
        action="store_true",
        help="Gate forecast movement by belief uncertainty, with persistence fallback.",
    )
    parser.add_argument(
        "--validated-edge-adapters",
        action="store_true",
        help="Enable adapters only for previously validated cross-organ edges.",
    )
    parser.add_argument(
        "--validated-edge-weight",
        type=float,
        default=0.0,
        help="Train validated edges on source-present/target-masked views.",
    )
    parser.add_argument(
        "--validated-edge-stage-epochs",
        type=int,
        default=0,
        help="Freeze the body/local heads and train validated edge adapters separately.",
    )
    parser.add_argument(
        "--target-head-stage-epochs",
        type=int,
        default=0,
        help=(
            "Freeze the shared whole-body latent and specialize every "
            "target-specific temporal/value/uncertainty JEPA branch."
        ),
    )
    parser.add_argument(
        "--measurement-time-head",
        action="store_true",
        help="Train a separate next-measurement-time head from raw timestamp labels.",
    )
    parser.add_argument(
        "--measurement-time-weight",
        type=float,
        default=0.10,
        help="Loss weight for the separate next-measurement-time task.",
    )
    parser.add_argument(
        "--uncertainty-calibration-weight",
        type=float,
        default=0.05,
        help=(
            "Train target-specific residual scales on detached point errors "
            "for normalized conformal calibration."
        ),
    )
    parser.add_argument("--output", default="whole_body_joint_jepa_candidate.pt")
    parser.add_argument("--report", default="whole_body_joint_jepa_candidate.json")
    args = parser.parse_args()
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    variable_age_limits = None
    if args.variable_age_limits is not None:
        variable_age_limits = json.loads(args.variable_age_limits.read_text(encoding="utf-8"))
    frame, variables, module_names = load_joint_cohort(
        args.data_root,
        modules,
        horizons,
        args.max_stays,
        min_modules=args.min_modules,
        alignment=args.alignment,
        asof_max_age_hours=args.asof_max_age_hours,
        future_label_tolerance_hours=args.future_label_tolerance_hours,
        asof_age_policy=args.asof_age_policy,
        variable_age_limits=variable_age_limits,
    )
    rows = np.arange(len(frame), dtype=np.int64)
    model, arrays, losses = fit_joint(
        frame,
        variables,
        module_names,
        rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cross_forecast_weight=args.cross_forecast_weight,
        measurement_process_mode=args.measurement_process_mode,
        masked_forecast_probability=args.masked_forecast_probability,
        module_mask_probability=args.module_mask_probability,
        nowcast_pretrain_epochs=args.nowcast_pretrain_epochs,
        measurement_consistency_weight=args.measurement_consistency_weight,
        world_model=args.world_model,
        world_model_weight=args.world_model_weight,
        target_momentum=args.target_momentum,
        future_target_mode=args.future_target_mode,
        target_min_observations=args.target_min_observations,
        regime_balanced_loss=args.regime_balanced_loss,
        regime_count=args.regime_count,
        state_normalization=args.state_normalization,
        hospital_invariance_weight=args.hospital_invariance_weight,
        domain_invariance_columns=tuple(
            value.strip()
            for value in args.domain_invariance_columns.split(",")
            if value.strip()
        ),
        target_encoder_mode=args.target_encoder_mode,
        target_horizon_regime_adapter=args.target_horizon_regime_adapter,
        patient_residual_adapter=args.patient_residual_adapter,
        future_task_balanced_sampling=args.future_task_balanced_sampling,
        coupling_preservation_weight=args.coupling_preservation_weight,
        uncertainty_gate=args.uncertainty_gate,
        validated_edge_adapters=args.validated_edge_adapters,
        validated_edge_weight=args.validated_edge_weight,
        validated_edge_stage_epochs=args.validated_edge_stage_epochs,
        target_head_stage_epochs=args.target_head_stage_epochs,
        measurement_time_head=args.measurement_time_head,
        measurement_time_weight=args.measurement_time_weight,
        uncertainty_calibration_weight=args.uncertainty_calibration_weight,
    )
    checkpoint = {
        "model_type": "JointWholeBodyJEPA",
        "state_dict": model.state_dict(),
        "scaler": arrays[0].to_dict(),
        "variables": list(variables),
        "modules": list(module_names),
        "input_dim": model.input_dim,
        "hidden": model.hidden,
        "latent": model.latent,
        "promotion_status": "candidate_only",
        "architecture": {
            "joint_patient_time_alignment": True,
            "alignment_mode": args.alignment,
            "asof_max_age_hours": float(args.asof_max_age_hours),
            "asof_age_policy": args.asof_age_policy,
            "variable_age_limits": variable_age_limits,
            "future_label_tolerance_hours": float(args.future_label_tolerance_hours),
            "masked_body_state": True,
            "cross_system_variable_tokens": True,
            "module_aware_variable_tokens": True,
            "module_level_state_tokens": True,
            "cross_system_transformer_layers": 2,
            "hybrid_variable_and_module_pool": True,
            "masked_body_reconstruction": True,
            "masked_target_future_forecast": bool(args.cross_forecast_weight > 0.0),
            "cross_forecast_weight": float(args.cross_forecast_weight),
            "cross_forecast_curriculum": True,
            "masked_forecast_probability": float(args.masked_forecast_probability),
            "module_mask_probability": float(args.module_mask_probability),
            "nowcast_pretrain_epochs": int(args.nowcast_pretrain_epochs),
            "measurement_consistency_weight": float(args.measurement_consistency_weight),
            "world_model": bool(args.world_model),
            "world_model_weight": float(args.world_model_weight),
            "target_momentum": float(args.target_momentum),
            "target_min_observations": int(args.target_min_observations),
            "future_delta_target": True,
            "future_target_sequence": True,
            "future_label_source": "raw_eicu_event_value_nearest_nominal_target",
            "future_label_quality_gate": "strict_raw_event_only",
            "near_labels_are_audit_only": True,
            "target_specific_future_delta_t": True,
            "regime_balanced_loss": bool(args.regime_balanced_loss),
            "regime_count": int(args.regime_count),
            "measurement_pure": True,
            "measurement_process_mode": args.measurement_process_mode,
            "measurement_process_separated": True,
            "state_normalization": args.state_normalization,
            "hospital_invariance_weight": float(args.hospital_invariance_weight),
            "hospital_adversarial_training_only": bool(args.hospital_invariance_weight > 0.0),
            "domain_invariance_columns": [
                value.strip()
                for value in args.domain_invariance_columns.split(",")
                if value.strip()
            ],
            "hospital_id_inference_feature": False,
            "target_encoder_mode": args.target_encoder_mode,
            "target_encoder_mixer_skipped": bool(args.target_encoder_mode == "fast"),
            "target_horizon_regime_adapter": bool(args.target_horizon_regime_adapter),
            "patient_residual_adapter": bool(args.patient_residual_adapter),
            "future_task_balanced_sampling": bool(args.future_task_balanced_sampling),
            "coupling_preservation_weight": float(args.coupling_preservation_weight),
            "uncertainty_gate": bool(args.uncertainty_gate),
            "validated_edge_adapters": bool(args.validated_edge_adapters),
            "validated_edge_weight": float(args.validated_edge_weight),
            "validated_edge_stage_epochs": int(args.validated_edge_stage_epochs),
            "target_head_stage_epochs": int(args.target_head_stage_epochs),
            "next_measurement_time_head": bool(model.measurement_time_head),
            "next_measurement_time_weight": float(args.measurement_time_weight),
            "uncertainty_calibration_weight": float(
                args.uncertainty_calibration_weight
            ),
            "next_measurement_time_label_source": (
                "raw_timestamped_event" if model.measurement_time_head else "unavailable"
            ),
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "training": {
            "rows": len(frame),
            "subjects": int(frame.subject_id.nunique()),
            "joint_rows_min_modules": int(args.min_modules),
            "mean_modules_per_row": float(frame["_module_count"].mean()),
            "loss_history": losses,
        },
    }
    torch.save(checkpoint, args.output)
    report = {
        "schema": "whole_body_joint_jepa_candidate.v1",
        "checkpoint": str(Path(args.output).resolve()),
        "modules": list(module_names),
        "variables": list(variables),
        "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()),
        "mean_modules_per_row": float(frame["_module_count"].mean()),
        "alignment_mode": args.alignment,
        "asof_max_age_hours": float(args.asof_max_age_hours),
        "asof_age_policy": args.asof_age_policy,
        "variable_age_limits": variable_age_limits,
        "future_label_tolerance_hours": float(args.future_label_tolerance_hours),
        "future_label_source": "raw_eicu_event_value_nearest_nominal_target",
        "future_label_quality_gate": "strict_raw_event_only",
        "near_labels_are_audit_only": True,
        "target_specific_future_delta_t": True,
        "regime_balanced_loss": bool(args.regime_balanced_loss),
        "regime_count": int(args.regime_count),
        "measurement_process_mode": args.measurement_process_mode,
        "measurement_process_separated": True,
        "state_normalization": args.state_normalization,
        "hospital_invariance_weight": float(args.hospital_invariance_weight),
        "hospital_adversarial_training_only": bool(args.hospital_invariance_weight > 0.0),
        "domain_invariance_columns": [
            value.strip()
            for value in args.domain_invariance_columns.split(",")
            if value.strip()
        ],
        "hospital_id_inference_feature": False,
        "target_encoder_mode": args.target_encoder_mode,
        "target_encoder_mixer_skipped": bool(args.target_encoder_mode == "fast"),
        "target_horizon_regime_adapter": bool(args.target_horizon_regime_adapter),
        "patient_residual_adapter": bool(args.patient_residual_adapter),
        "future_task_balanced_sampling": bool(args.future_task_balanced_sampling),
        "coupling_preservation_weight": float(args.coupling_preservation_weight),
        "nowcast_pretrain_epochs": int(args.nowcast_pretrain_epochs),
        "measurement_consistency_weight": float(args.measurement_consistency_weight),
        "world_model": bool(args.world_model),
        "world_model_weight": float(args.world_model_weight),
        "target_momentum": float(args.target_momentum),
        "target_min_observations": int(args.target_min_observations),
        "future_target_mode": args.future_target_mode,
        "next_measurement_time_head": bool(model.measurement_time_head),
        "next_measurement_time_weight": float(args.measurement_time_weight),
        "uncertainty_calibration_weight": float(
            args.uncertainty_calibration_weight
        ),
        "next_measurement_time_label_source": (
            "raw_timestamped_event" if model.measurement_time_head else "unavailable"
        ),
        "loss_history": losses,
        "promotion_status": "candidate_only",
        "joint_latent_validated": False,
        "gate_required": [
            "patient_heldout",
            "hospital_heldout",
            "target_by_horizon_conformal",
            "cross_system_ablation",
        ],
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
