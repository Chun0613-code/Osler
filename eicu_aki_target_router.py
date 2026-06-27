"""Nested factual target router for the full-eICU AKI adapter.

This is the Chapter-B AKI module.  It keeps the same router discipline as DKA
and sepsis:

* discovery-only source selection;
* out-of-fold predictions for discovery source comparisons;
* held-out patient and hospital evaluation;
* persistence fallback for unsupported targets;
* no causal, counterfactual, clinical, or checkpoint-promotion claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_aki_transition_extract import ACTION_KEYS, TARGET_VARS
from eicu_sepsis_target_router import (
    METHODS,
    _bootstrap_ci,
    _feature_columns,
    _fit_population_delta,
    _fit_ridge,
    _group_folds,
    _round,
    _subject_column,
    split_subjects,
)


ACTIVE_COLUMN = "aki_active_t"


def _target_pairs(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_tp6"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def _target_scale(discovery: pd.DataFrame, target: str) -> float:
    future = pd.to_numeric(discovery[f"{target}_tp6"], errors="coerce")
    std = float(future.std(skipna=True))
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
            "active_aki": selected.get(
                ACTIVE_COLUMN,
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
            discovery_predictions["population_delta"].loc[validation_fold.index] = (
                _fit_population_delta(train_fold, validation_fold, target)
            )
            discovery_predictions["ridge_realfit"].loc[validation_fold.index] = (
                _fit_ridge(train_fold, validation_fold, target, features, ridge_alpha)
            )
        heldout_population = _fit_population_delta(discovery, heldout, target)
        heldout_ridge = _fit_ridge(discovery, heldout, target, features, ridge_alpha)
        combined_frame = pd.concat([discovery, heldout], axis=0).sort_index()
        combined_population = pd.concat([
            discovery_predictions["population_delta"],
            pd.Series(heldout_population, index=heldout.index),
        ]).sort_index()
        combined_ridge = pd.concat([
            discovery_predictions["ridge_realfit"],
            pd.Series(heldout_ridge, index=heldout.index),
        ]).sort_index()
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
    return rows[rows["active_aki"].astype(bool)] if active_only else rows


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


def _cluster_delta(rows: pd.DataFrame, method: str, normalized: bool, baseline: str = "persistence") -> pd.Series:
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


def _delta_summary(rows: pd.DataFrame, method: str, seed: int, samples: int, normalized: bool) -> dict[str, object]:
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


def _method_stats(rows: pd.DataFrame, method: str, seed: int, samples: int, normalized: bool) -> dict[str, object]:
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


def _router_predictions(rows: pd.DataFrame, selector: dict[str, str]) -> np.ndarray:
    if rows.empty:
        return np.asarray([], dtype=np.float64)
    selected = rows["target"].map(selector).fillna("persistence").to_numpy()
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


def _router_delta_summary(rows: pd.DataFrame, selector: dict[str, str], seed: int, samples: int, normalized: bool) -> dict[str, object]:
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


def _scope_summary(rows: pd.DataFrame, selector: dict[str, str], active_only: bool, seed: int, samples: int) -> dict[str, object]:
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
        "heldout_active_aki_only": _scope_summary(
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

    counts: dict[str, dict[str, int]] = {}
    for report in reports:
        for target, method in report["selected_methods"].items():
            counts.setdefault(target, {})
            counts[target][method] = counts[target].get(method, 0) + 1

    return {
        "target_router": {
            "active_aki_only": collect("heldout_active_aki_only"),
            "all_windows": collect("heldout_all_windows"),
        },
        "selected_method_counts_by_target": counts,
    }


def _cohort_summary(frame: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "active_aki_rows": int(frame[ACTIVE_COLUMN].fillna(False).sum()) if ACTIVE_COLUMN in frame else None,
    }
    for target in TARGET_VARS:
        current = f"{target}_t"
        future = f"{target}_tp6"
        if current in frame and future in frame:
            summary[f"{target}_pairs"] = int((frame[current].notna() & frame[future].notna()).sum())
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
        return {"available": False, "reason": "cohort has fewer than three hospitals"}
    discovery, heldout = split_subjects(frame, seed, discovery_fraction, group_column="hospitalid")
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
    parser.add_argument("--cohort", default="eicu_aki_transitions_6h.parquet")
    parser.add_argument("--output", default="eicu_aki_target_router.json")
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
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    reports = []
    for seed in seeds:
        discovery, heldout = split_subjects(frame, seed, args.discovery_fraction)
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
        "experiment": "Full eICU AKI nested factual target router",
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
            "scope": "discovery active-AKI rows",
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
