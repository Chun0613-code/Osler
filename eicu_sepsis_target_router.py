"""Nested factual target router for the full-eICU sepsis adapter.

This is the first Chapter-B disease expansion after the validated DKA router.
It intentionally uses disease-agnostic factual sources only:

* persistence: current observed value;
* population_delta: discovery-only mean delta for the target;
* ridge_realfit: a small discovery-only linear residual model using observed
  states, masks/ages, and factual observed-treatment evidence.

Discovery rows use out-of-fold predictions for source selection.  Held-out rows
use models fit only on discovery patients.  The report is aggregate-only and
does not claim treatment causality, counterfactual validity, or clinical
authority.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_transition_extract import ACTION_KEYS, TARGET_VARS


METHODS = ("persistence", "population_delta", "ridge_realfit")


def _round(value, digits: int = 6):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _bootstrap_ci(values, seed: int, samples: int) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
    return [
        round(float(np.quantile(means, 0.025)), 6),
        round(float(np.quantile(means, 0.975)), 6),
    ]


def _subject_column(frame: pd.DataFrame) -> str:
    return "subject_id" if "subject_id" in frame else "stay_id"


def split_subjects(
    frame: pd.DataFrame,
    seed: int,
    discovery_fraction: float,
    group_column: str | None = None,
) -> tuple[set[object], set[object]]:
    column = group_column or _subject_column(frame)
    groups = sorted(frame[column].dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    split = max(1, int(round(len(groups) * discovery_fraction)))
    split = min(split, max(1, len(groups) - 1))
    discovery = {groups[int(index)] for index in order[:split]}
    heldout = {groups[int(index)] for index in order[split:]}
    return discovery, heldout


def _group_folds(groups: np.ndarray, seed: int, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.asarray(sorted(pd.unique(groups).tolist()), dtype=object)
    if len(unique) < 2:
        return []
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_count = min(max(2, folds), len(unique))
    splits = np.array_split(unique, fold_count)
    output = []
    for validation_groups in splits:
        validation_mask = np.isin(groups, validation_groups)
        train_mask = ~validation_mask
        if train_mask.any() and validation_mask.any():
            output.append((np.flatnonzero(train_mask), np.flatnonzero(validation_mask)))
    return output


def _target_pairs(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_tp6"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def _feature_columns(frame: pd.DataFrame, target: str | None = None) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in {
            "stay_id",
            "subject_id",
            "onset",
            "onset_criteria",
            "t",
            "t_plus",
            "unittype",
        }:
            continue
        if column.endswith("_tp6"):
            continue
        if column.endswith("_t") or column.endswith("_age_hr"):
            columns.append(column)
            continue
        if column.startswith("hist_") or column.startswith("act_"):
            if target == "vasopressor_requirement" and column.startswith("act_"):
                # ``vasopressor_requirement_tp6`` is defined from future
                # vasopressor evidence, so future action features would leak the
                # answer.  This target may use current state and history only.
                continue
            if column.endswith("_sources") or column.endswith("_kinds"):
                continue
            columns.append(column)
            continue
        if column in ("hours_since_onset", "hospitalid"):
            columns.append(column)
    numeric_columns = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            numeric_columns.append(column)
    return sorted(set(numeric_columns))


def _prepare_features(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[columns].apply(pd.to_numeric, errors="coerce")
    predict_x = predict[columns].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
    medians = medians.fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    predict_x = predict_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return (
        ((train_x - mean) / std).to_numpy(dtype=np.float64),
        ((predict_x - mean) / std).to_numpy(dtype=np.float64),
    )


def _fit_ridge(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    alpha: float,
) -> np.ndarray:
    current = f"{target}_t"
    future = f"{target}_tp6"
    usable = train[train[current].notna() & train[future].notna()].copy()
    if len(usable) < max(20, len(feature_columns) + 2):
        return np.full(len(predict), np.nan, dtype=np.float64)
    x_train, x_predict = _prepare_features(usable, predict, feature_columns)
    y = (
        usable[future].to_numpy(dtype=np.float64)
        - usable[current].to_numpy(dtype=np.float64)
    )
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_predict = np.c_[np.ones(len(x_predict)), x_predict]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    delta = x_predict @ beta
    return predict[current].to_numpy(dtype=np.float64) + delta


def _fit_population_delta(train: pd.DataFrame, predict: pd.DataFrame, target: str) -> np.ndarray:
    current = f"{target}_t"
    future = f"{target}_tp6"
    usable = train[train[current].notna() & train[future].notna()]
    if usable.empty:
        return np.full(len(predict), np.nan, dtype=np.float64)
    delta = (
        usable[future].to_numpy(dtype=np.float64)
        - usable[current].to_numpy(dtype=np.float64)
    )
    mean_delta = float(np.nanmean(delta))
    return predict[current].to_numpy(dtype=np.float64) + mean_delta


def _target_scale(discovery: pd.DataFrame, target: str) -> float:
    future = pd.to_numeric(discovery[f"{target}_tp6"], errors="coerce")
    std = float(future.std(skipna=True))
    if target == "vasopressor_requirement":
        return 1.0
    if not np.isfinite(std) or std < 1e-6:
        current = pd.to_numeric(discovery[f"{target}_t"], errors="coerce")
        std = float(current.std(skipna=True))
    return max(std, 1.0) if np.isfinite(std) else 1.0


def _build_base_rows(frame: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    parts = []
    for target in TARGET_VARS:
        current = f"{target}_t"
        future = f"{target}_tp6"
        if current not in frame or future not in frame:
            continue
        selected = frame[frame[current].notna() & frame[future].notna()].copy()
        if selected.empty:
            continue
        subject = (
            selected["subject_id"]
            if "subject_id" in selected
            else selected["stay_id"].astype(str)
        )
        hospital = (
            selected["hospitalid"].astype("int64")
            if "hospitalid" in selected
            else pd.Series(-1, index=selected.index, dtype="int64")
        )
        part = pd.DataFrame({
            "row_index": selected.index.to_numpy(dtype=np.int64),
            "subject_id": subject.to_numpy(),
            "stay_id": selected["stay_id"].to_numpy(dtype=np.int64),
            "hospitalid": hospital.to_numpy(dtype=np.int64),
            "target": target,
            "active_sepsis": selected.get(
                "sepsis_active_t",
                pd.Series(False, index=selected.index),
            ).fillna(False).astype(bool).to_numpy(),
            "truth": selected[future].to_numpy(dtype=np.float64),
            "current": selected[current].to_numpy(dtype=np.float64),
            "scale": np.full(len(selected), float(scales.get(target, 1.0)), dtype=np.float64),
            "persistence": selected[current].to_numpy(dtype=np.float64),
            "population_delta": np.full(len(selected), np.nan, dtype=np.float64),
            "ridge_realfit": np.full(len(selected), np.nan, dtype=np.float64),
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _assign_predictions(
    rows: pd.DataFrame,
    frame: pd.DataFrame,
    target: str,
    method: str,
    predictions: np.ndarray,
) -> None:
    if len(predictions) != len(frame):
        raise ValueError("prediction length does not match frame length")
    mapping = {
        int(index): float(value)
        for index, value in zip(frame.index.to_numpy(dtype=np.int64), predictions)
        if np.isfinite(value)
    }
    mask = rows["target"] == target
    rows.loc[mask, method] = rows.loc[mask, "row_index"].map(mapping).to_numpy(dtype=np.float64)


def attach_sources(
    frame: pd.DataFrame,
    discovery_groups: set[object],
    heldout_groups: set[object],
    *,
    seed: int,
    inner_folds: int,
    ridge_alpha: float,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group_column = group_column or _subject_column(frame)
    discovery = frame[frame[group_column].isin(discovery_groups)].copy()
    heldout = frame[frame[group_column].isin(heldout_groups)].copy()
    scales = {
        target: _target_scale(_target_pairs(discovery, target), target)
        for target in TARGET_VARS
    }
    rows = _build_base_rows(pd.concat([discovery, heldout], axis=0), scales)
    features_by_target = {
        target: _feature_columns(frame, target)
        for target in TARGET_VARS
    }
    groups = discovery[group_column].to_numpy(dtype=object)
    folds = _group_folds(groups, seed=seed + 17, folds=inner_folds)
    diagnostics = {
        "feature_count_by_target": {
            target: int(len(columns))
            for target, columns in features_by_target.items()
        },
        "inner_oof_folds": int(len(folds)),
        "ridge_alpha": float(ridge_alpha),
        "sources": {
            "persistence": "current observed value",
            "population_delta": "discovery-only mean target delta",
            "ridge_realfit": "small linear residual model fit on discovery patients",
        },
    }

    for target in TARGET_VARS:
        current = f"{target}_t"
        future = f"{target}_tp6"
        if current not in frame or future not in frame:
            continue
        features = features_by_target[target]
        discovery_predictions = {
            "population_delta": pd.Series(np.nan, index=discovery.index, dtype="float64"),
            "ridge_realfit": pd.Series(np.nan, index=discovery.index, dtype="float64"),
        }
        for train_local, validation_local in folds:
            train_fold = discovery.iloc[train_local]
            validation_fold = discovery.iloc[validation_local]
            population = _fit_population_delta(train_fold, validation_fold, target)
            ridge = _fit_ridge(train_fold, validation_fold, target, features, ridge_alpha)
            discovery_predictions["population_delta"].loc[validation_fold.index] = population
            discovery_predictions["ridge_realfit"].loc[validation_fold.index] = ridge
        heldout_population = _fit_population_delta(discovery, heldout, target)
        heldout_ridge = _fit_ridge(discovery, heldout, target, features, ridge_alpha)
        combined_population = pd.concat(
            [discovery_predictions["population_delta"], pd.Series(heldout_population, index=heldout.index)]
        ).sort_index()
        combined_ridge = pd.concat(
            [discovery_predictions["ridge_realfit"], pd.Series(heldout_ridge, index=heldout.index)]
        ).sort_index()
        combined_frame = pd.concat([discovery, heldout], axis=0).sort_index()
        _assign_predictions(
            rows,
            combined_frame,
            target,
            "population_delta",
            combined_population.reindex(combined_frame.index).to_numpy(dtype=np.float64),
        )
        _assign_predictions(
            rows,
            combined_frame,
            target,
            "ridge_realfit",
            combined_ridge.reindex(combined_frame.index).to_numpy(dtype=np.float64),
        )
    return rows, diagnostics


def _scope(rows: pd.DataFrame, active_only: bool) -> pd.DataFrame:
    return rows[rows["active_sepsis"].astype(bool)] if active_only else rows


def _method_rows(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    if method not in rows:
        return rows.iloc[0:0].copy()
    selected = rows[np.isfinite(rows[method])].copy()
    selected = selected[np.isfinite(selected["truth"])]
    selected = selected[np.isfinite(selected["scale"])]
    return selected


def _mae(rows: pd.DataFrame, method: str, normalized: bool) -> float | None:
    selected = _method_rows(rows, method)
    if selected.empty:
        return None
    errors = np.abs(selected[method].to_numpy(dtype=np.float64) - selected["truth"].to_numpy(dtype=np.float64))
    if normalized:
        errors = errors / selected["scale"].to_numpy(dtype=np.float64)
    return float(errors.mean())


def _cluster_delta(
    rows: pd.DataFrame,
    method: str,
    normalized: bool,
    baseline: str = "persistence",
) -> pd.Series:
    selected = _method_rows(rows, method)
    selected = selected[np.isfinite(selected[baseline])]
    if selected.empty:
        return pd.Series(dtype="float64")
    scale = selected["scale"].to_numpy(dtype=np.float64) if normalized else 1.0
    delta = (
        np.abs(selected[method].to_numpy(dtype=np.float64) - selected["truth"].to_numpy(dtype=np.float64))
        - np.abs(selected[baseline].to_numpy(dtype=np.float64) - selected["truth"].to_numpy(dtype=np.float64))
    ) / scale
    temp = selected[["subject_id"]].copy()
    temp["delta"] = delta
    return temp.groupby("subject_id")["delta"].mean()


def _delta_summary(
    rows: pd.DataFrame,
    method: str,
    seed: int,
    samples: int,
    normalized: bool,
) -> dict[str, object]:
    deltas = _cluster_delta(rows, method, normalized=normalized)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_persistence": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=samples)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_persistence": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _method_stats(
    rows: pd.DataFrame,
    method: str,
    seed: int,
    samples: int,
    normalized: bool,
) -> dict[str, object]:
    selected = _method_rows(rows, method)
    return {
        "rows": int(len(selected)),
        "subjects": int(selected["subject_id"].nunique()) if len(selected) else 0,
        "stays": int(selected["stay_id"].nunique()) if len(selected) else 0,
        "mae": _round(_mae(selected, method, normalized=normalized)),
        "delta_vs_persistence": _delta_summary(
            selected,
            method,
            seed=seed,
            samples=samples,
            normalized=normalized,
        ),
    }


def _choose_selector(
    discovery: pd.DataFrame,
    active_only: bool,
    min_pairs: int,
    min_subjects: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, object]]:
    selected = {}
    details = {}
    scoped = _scope(discovery, active_only)
    for target_index, target in enumerate(TARGET_VARS):
        target_rows = scoped[scoped["target"] == target].copy()
        target_details = {}
        for method_index, method in enumerate(METHODS):
            target_details[method] = _method_stats(
                target_rows,
                method,
                seed=seed + 101 * target_index + method_index,
                samples=bootstrap_samples,
                normalized=False,
            )
        candidates = {}
        for method in METHODS:
            if method == "persistence":
                continue
            stats = target_details[method]
            delta = stats["delta_vs_persistence"]
            if stats["rows"] < min_pairs or stats["subjects"] < min_subjects:
                continue
            if not delta["significant"]:
                continue
            if stats["mae"] is None:
                continue
            candidates[method] = stats["mae"]
        if candidates:
            selected_method = min(candidates, key=candidates.get)
            reason = "lowest_discovery_mae_among_significant_supported_sources"
        else:
            selected_method = "persistence"
            reason = "no_source_passed_discovery_significance_gate"
        selected[target] = selected_method
        details[target] = {
            "selected_method": selected_method,
            "reason": reason,
            "min_pairs": int(min_pairs),
            "min_subjects": int(min_subjects),
            "method_stats": target_details,
        }
    return selected, details


def _selected_methods_for_rows(rows: pd.DataFrame, selector: dict[str, str]) -> np.ndarray:
    return rows["target"].map(selector).fillna("persistence").to_numpy()


def _router_predictions(rows: pd.DataFrame, selector: dict[str, str]) -> np.ndarray:
    if rows.empty:
        return np.asarray([], dtype=np.float64)
    selected = _selected_methods_for_rows(rows, selector)
    predictions = rows["persistence"].to_numpy(dtype=np.float64).copy()
    for method in METHODS:
        method_mask = selected == method
        if method == "persistence" or not method_mask.any():
            continue
        values = rows[method].to_numpy(dtype=np.float64)
        finite = method_mask & np.isfinite(values)
        predictions[finite] = values[finite]
    return predictions


def _router_errors(rows: pd.DataFrame, selector: dict[str, str], normalized: bool) -> np.ndarray:
    if rows.empty:
        return np.asarray([], dtype=np.float64)
    errors = np.abs(_router_predictions(rows, selector) - rows["truth"].to_numpy(dtype=np.float64))
    if normalized:
        errors = errors / rows["scale"].to_numpy(dtype=np.float64)
    return errors


def _router_delta(rows: pd.DataFrame, selector: dict[str, str], normalized: bool) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="float64")
    scale = rows["scale"].to_numpy(dtype=np.float64) if normalized else 1.0
    delta = (
        np.abs(_router_predictions(rows, selector) - rows["truth"].to_numpy(dtype=np.float64))
        - np.abs(rows["persistence"].to_numpy(dtype=np.float64) - rows["truth"].to_numpy(dtype=np.float64))
    ) / scale
    temp = pd.DataFrame({
        "subject_id": rows["subject_id"].to_numpy(),
        "delta": delta,
    })
    return temp.groupby("subject_id")["delta"].mean()


def _router_delta_summary(
    rows: pd.DataFrame,
    selector: dict[str, str],
    seed: int,
    samples: int,
    normalized: bool,
) -> dict[str, object]:
    deltas = _router_delta(rows, selector, normalized=normalized)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_persistence": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=samples)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_persistence": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _fixed_methods_summary(rows: pd.DataFrame, normalized: bool) -> dict[str, float | None]:
    return {method: _round(_mae(rows, method, normalized=normalized)) for method in METHODS}


def _scope_summary(
    rows: pd.DataFrame,
    selector: dict[str, str],
    active_only: bool,
    seed: int,
    samples: int,
) -> dict[str, object]:
    scoped = _scope(rows, active_only)
    router_errors = _router_errors(scoped, selector, normalized=True)
    per_target = {}
    for target_index, target in enumerate(TARGET_VARS):
        target_rows = scoped[scoped["target"] == target]
        target_selector = {target: selector.get(target, "persistence")}
        target_errors = _router_errors(target_rows, target_selector, normalized=False)
        per_target[target] = {
            "rows": int(len(target_rows)),
            "subjects": int(target_rows["subject_id"].nunique()) if len(target_rows) else 0,
            "stays": int(target_rows["stay_id"].nunique()) if len(target_rows) else 0,
            "selected_method": target_selector[target],
            "mae": {
                **_fixed_methods_summary(target_rows, normalized=False),
                "target_router": _round(float(target_errors.mean()) if len(target_errors) else None),
            },
            "router_delta_vs_persistence": _router_delta_summary(
                target_rows,
                target_selector,
                seed=seed + 100 + target_index,
                samples=samples,
                normalized=False,
            ),
        }
    return {
        "rows": int(len(scoped)),
        "subjects": int(scoped["subject_id"].nunique()) if len(scoped) else 0,
        "stays": int(scoped["stay_id"].nunique()) if len(scoped) else 0,
        "normalized_mae": {
            **_fixed_methods_summary(scoped, normalized=True),
            "target_router": _round(float(router_errors.mean()) if len(router_errors) else None),
        },
        "router_delta_vs_persistence": _router_delta_summary(
            scoped,
            selector,
            seed=seed,
            samples=samples,
            normalized=True,
        ),
        "per_target": per_target,
    }


def _split_report(
    frame: pd.DataFrame,
    discovery_groups: set[object],
    heldout_groups: set[object],
    seed: int,
    min_pairs: int,
    min_subjects: int,
    bootstrap_samples: int,
    inner_folds: int,
    ridge_alpha: float,
    group_column: str | None = None,
) -> dict[str, object]:
    group_column = group_column or _subject_column(frame)
    rows, source_diagnostics = attach_sources(
        frame,
        discovery_groups,
        heldout_groups,
        seed=seed,
        inner_folds=inner_folds,
        ridge_alpha=ridge_alpha,
        group_column=group_column,
    )
    discovery = rows[rows[group_column].isin(discovery_groups)].copy()
    heldout = rows[rows[group_column].isin(heldout_groups)].copy()
    selector, selector_details = _choose_selector(
        discovery,
        active_only=True,
        min_pairs=min_pairs,
        min_subjects=min_subjects,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 10_000,
    )
    return {
        "seed": int(seed),
        "group_column": group_column,
        "discovery_groups": int(len(discovery_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "overlap": int(len(discovery_groups & heldout_groups)),
        "source_diagnostics": source_diagnostics,
        "selected_methods": selector,
        "discovery_selector": selector_details,
        "heldout_all_windows": _scope_summary(
            heldout,
            selector,
            active_only=False,
            seed=seed + 20_000,
            samples=bootstrap_samples,
        ),
        "heldout_active_sepsis_only": _scope_summary(
            heldout,
            selector,
            active_only=True,
            seed=seed + 30_000,
            samples=bootstrap_samples,
        ),
    }


def _multi_seed_summary(reports: list[dict[str, object]]) -> dict[str, object]:
    def collect(scope: str) -> dict[str, object]:
        deltas = [
            report[scope]["router_delta_vs_persistence"]["point_delta"]
            for report in reports
            if report[scope]["router_delta_vs_persistence"]["point_delta"] is not None
        ]
        beats = [
            bool(report[scope]["router_delta_vs_persistence"]["beats_persistence"])
            for report in reports
        ]
        significant = [
            bool(report[scope]["router_delta_vs_persistence"]["significant"])
            for report in reports
        ]
        return {
            "splits": int(len(reports)),
            "beats_persistence_count": int(sum(beats)),
            "significant_count": int(sum(significant)),
            "delta_min": _round(min(deltas)) if deltas else None,
            "delta_median": _round(float(np.median(deltas))) if deltas else None,
            "delta_max": _round(max(deltas)) if deltas else None,
        }

    def method_counts() -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for report in reports:
            for target, method in report["selected_methods"].items():
                counts.setdefault(target, {})
                counts[target][method] = counts[target].get(method, 0) + 1
        return counts

    return {
        "target_router": {
            "active_sepsis_only": collect("heldout_active_sepsis_only"),
            "all_windows": collect("heldout_all_windows"),
        },
        "selected_method_counts_by_target": method_counts(),
    }


def _cohort_summary(frame: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "active_sepsis_rows": int(frame["sepsis_active_t"].fillna(False).sum()) if "sepsis_active_t" in frame else None,
    }
    for target in TARGET_VARS:
        current = f"{target}_t"
        future = f"{target}_tp6"
        if current in frame and future in frame:
            summary[f"{target}_pairs"] = int(
                (frame[current].notna() & frame[future].notna()).sum()
            )
    return summary


def _hospital_holdout_report(
    frame: pd.DataFrame,
    seed: int,
    discovery_fraction: float,
    min_pairs: int,
    min_subjects: int,
    bootstrap_samples: int,
    inner_folds: int,
    ridge_alpha: float,
) -> dict[str, object]:
    if "hospitalid" not in frame or frame["hospitalid"].nunique() < 3:
        return {
            "available": False,
            "reason": "cohort has fewer than three hospitals",
        }
    discovery, heldout = split_subjects(
        frame,
        seed,
        discovery_fraction,
        group_column="hospitalid",
    )
    return {
        "available": True,
        "split_type": "hospital-held-out",
        **_split_report(
            frame,
            discovery,
            heldout,
            seed=seed,
            min_pairs=min_pairs,
            min_subjects=min_subjects,
            bootstrap_samples=bootstrap_samples,
            inner_folds=inner_folds,
            ridge_alpha=ridge_alpha,
            group_column="hospitalid",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="eicu_sepsis_transitions_6h.parquet")
    parser.add_argument("--output", default="eicu_sepsis_target_router.json")
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [
        int(item.strip()) for item in args.seeds.split(",")
        if item.strip()
    ]
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    reports = []
    for seed in seeds:
        discovery, heldout = split_subjects(
            frame,
            seed,
            args.discovery_fraction,
        )
        reports.append(_split_report(
            frame,
            discovery,
            heldout,
            seed=seed,
            min_pairs=args.min_pairs,
            min_subjects=args.min_subjects,
            bootstrap_samples=args.bootstrap_samples,
            inner_folds=args.inner_folds,
            ridge_alpha=args.ridge_alpha,
        ))

    hospital = _hospital_holdout_report(
        frame,
        seed=9001,
        discovery_fraction=args.discovery_fraction,
        min_pairs=args.min_pairs,
        min_subjects=args.min_subjects,
        bootstrap_samples=args.bootstrap_samples,
        inner_folds=args.inner_folds,
        ridge_alpha=args.ridge_alpha,
    )

    report = {
        "experiment": "Full eICU sepsis nested factual target router",
        "cohort": Path(args.cohort).name,
        "cohort_summary": _cohort_summary(frame),
        "targets": TARGET_VARS,
        "action_channels": ACTION_KEYS,
        "sources": {
            "persistence": "current observed value",
            "population_delta": "discovery-only mean target delta; discovery selection uses inner OOF predictions",
            "ridge_realfit": "small discovery-only ridge residual; discovery selection uses inner OOF predictions",
        },
        "selector_gate": {
            "scope": "discovery active-sepsis rows",
            "min_pairs": int(args.min_pairs),
            "min_subjects": int(args.min_subjects),
            "rule": (
                "A non-persistence source can be selected only if its discovery "
                "subject-level bootstrap CI versus persistence excludes zero; "
                "among passing sources choose lowest discovery MAE. Otherwise "
                "fallback to persistence."
            ),
        },
        "random_patient_splits": {
            "seeds": seeds,
            "summary": _multi_seed_summary(reports),
            "runs": reports,
        },
        "hospital_holdout_split": hospital,
        "time_order_split": {
            "available": False,
            "reason": (
                "The public eICU release does not expose reliable chronological "
                "calendar order for inter-stay time splits in this adapter."
            ),
        },
        "safety_boundary": {
            "raw_rows_included": False,
            "patient_ids_included_in_report": False,
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "cohort_summary": report["cohort_summary"],
        "random_summary": report["random_patient_splits"]["summary"],
        "hospital_holdout_available": report["hospital_holdout_split"]["available"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
