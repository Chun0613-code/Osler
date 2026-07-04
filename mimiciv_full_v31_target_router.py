"""Nested target-gated router for the full MIMIC-IV DKA factual proxy.

The router chooses, per target, among persistence, fixed JEPA checkpoints, and a
real-data grey-box residual source.  The grey-box source is handled carefully:

* discovery rows use inner out-of-fold residual predictions for source selection;
* held-out rows use a residual model fit only on discovery stays;
* row-level predictions are never written.

This is a factual observed-treatment proxy only.  It does not identify treatment
effects and does not promote checkpoints, residual artifacts, or active rules.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dka_world_model import S_STD, load_checkpoint
from eicu_dka_per_target_ensemble import (
    TARGET_MAP,
    _active_dka,
    _float,
    _predict_physical,
    split_stays,
)
from osler_jepa.greybox_residual import GreyBoxResidualRuntime
from train_greybox_residual import build_examples, fit_network, rollout
from train_intervention_jepa import choose_device


METHODS = (
    "persistence",
    "v5",
    "presentation_only",
    "mechanism",
    "greybox_realfit",
)
BASE_METHODS = ("persistence", "v5", "presentation_only")

RESIDUAL_TARGETS = {
    "glucose": (0, lambda value: value),
    "anion_gap": (1, lambda value: value + 12.0),
    "bicarbonate": (2, lambda value: value),
    "potassium": (3, lambda value: value),
    "sodium": (4, lambda value: value),
    "creatinine": (5, lambda value: value),
}


def _round(value, digits=6):
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


def _cohort_summary(frame: pd.DataFrame, examples: list[dict]) -> dict:
    summary = {
        "rows": int(len(frame)),
        "examples_with_mechanism_rollout": int(len(examples)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "active_dka_rows": (
            int(frame["dka_active_t"].fillna(False).sum())
            if "dka_active_t" in frame else None
        ),
        "all_icd_supported": (
            bool(frame["icd_dka_support"].fillna(False).all())
            if "icd_dka_support" in frame else None
        ),
    }
    for target, (current, future, _) in TARGET_MAP.items():
        if current in frame and future in frame:
            summary[f"{target}_pairs"] = int(
                (frame[current].notna() & frame[future].notna()).sum()
            )
    return summary


def build_base_rows(frame: pd.DataFrame, models: dict, device) -> pd.DataFrame:
    rows = []
    for row_index, row in frame.iterrows():
        predictions = {}
        state = None
        for name, model in models.items():
            physical, state_for_row = _predict_physical(model, row, frame, device)
            if physical is None:
                predictions = {}
                break
            predictions[name] = physical
            state = state_for_row
        if not predictions:
            continue
        active = _active_dka(row, state)
        for target, (current_column, target_column, index) in TARGET_MAP.items():
            current = _float(row, current_column)
            truth = _float(row, target_column)
            if target == "osmolality":
                if np.isnan(current):
                    current = _float(row, "osmolality_derived_t")
                if np.isnan(truth):
                    truth = _float(row, "osmolality_derived_tp6")
            if np.isnan(current) or np.isnan(truth):
                continue
            entry = {
                "row_index": int(row_index),
                "stay_id": int(row["stay_id"]),
                "target": target,
                "target_index": int(index),
                "active_dka": bool(active),
                "truth": float(truth),
                "current": float(current),
                "scale": float(S_STD[index]),
                "persistence": float(current),
            }
            for name, physical in predictions.items():
                entry[name] = float(physical[index])
            rows.append(entry)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["mechanism"] = np.nan
    result["greybox_realfit"] = np.nan
    return result


def _example_indices_for_stays(
    examples: list[dict],
    stays: set[int],
) -> np.ndarray:
    return np.asarray(
        [
            index for index, item in enumerate(examples)
            if int(item["stay_id"]) in stays
        ],
        dtype=np.int64,
    )


def _validation_split(indices: np.ndarray, examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([examples[index]["stay_id"] for index in indices])
    unique = np.unique(groups)
    if len(unique) < 3:
        split = max(1, int(round(len(indices) * 0.8)))
        return indices[:split], indices[split:]
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=min(3, len(unique)))
    train_local, validation_local = next(splitter.split(indices, groups=groups))
    return indices[train_local], indices[validation_local]


def _group_folds(indices: np.ndarray, examples: list[dict], folds: int):
    groups = np.asarray([examples[index]["stay_id"] for index in indices])
    unique = np.unique(groups)
    if len(unique) < 2:
        yield indices, np.asarray([], dtype=np.int64)
        return
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=min(folds, len(unique)))
    for train_local, test_local in splitter.split(indices, groups=groups):
        yield indices[train_local], indices[test_local]


def _predict_with_greybox(
    examples: list[dict],
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    seed: int,
    epochs: int,
    bottleneck: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if len(train_indices) == 0 or len(predict_indices) == 0:
        return {}
    inner_train, validation = _validation_split(train_indices, examples)
    model, mean, std, validation_loss = fit_network(
        examples,
        inner_train,
        validation_indices=validation if len(validation) else None,
        seed=seed,
        epochs=epochs,
        bottleneck=bottleneck,
    )
    print(
        f"[target-router] greybox fit seed={seed} train={len(inner_train)} "
        f"validation={len(validation)} predict={len(predict_indices)} "
        f"validation_loss={validation_loss:.6f}",
        file=sys.stderr,
        flush=True,
    )
    runtime = GreyBoxResidualRuntime(model, mean, std)
    predictions = {}
    for offset, index in enumerate(predict_indices, start=1):
        if offset % 1000 == 0:
            print(
                f"[target-router] greybox rollout seed={seed} "
                f"rolled={offset}/{len(predict_indices)}",
                file=sys.stderr,
                flush=True,
            )
        item = examples[index]
        _, predicted, _ = rollout(item["row"], runtime)
        predictions[int(item["row"].name)] = (item["mechanism"], predicted)
    return predictions


def _nested_greybox_predictions(
    examples: list[dict],
    discovery_stays: set[int],
    heldout_stays: set[int],
    seed: int,
    epochs: int,
    bottleneck: int,
    inner_folds: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    discovery_indices = _example_indices_for_stays(examples, discovery_stays)
    heldout_indices = _example_indices_for_stays(examples, heldout_stays)
    predictions = {}
    fold_count = 0
    for fold, (train_indices, test_indices) in enumerate(
        _group_folds(discovery_indices, examples, inner_folds),
        start=1,
    ):
        fold_count += 1
        print(
            f"[target-router] seed={seed} discovery inner fold={fold} "
            f"train={len(train_indices)} oof={len(test_indices)}",
            file=sys.stderr,
            flush=True,
        )
        predictions.update(_predict_with_greybox(
            examples,
            train_indices,
            test_indices,
            seed=seed + 100 * fold,
            epochs=epochs,
            bottleneck=bottleneck,
        ))
    predictions.update(_predict_with_greybox(
        examples,
        discovery_indices,
        heldout_indices,
        seed=seed + 999,
        epochs=epochs,
        bottleneck=bottleneck,
    ))
    return predictions, {
        "discovery_examples": int(len(discovery_indices)),
        "heldout_examples": int(len(heldout_indices)),
        "inner_oof_folds": int(fold_count),
    }


def _residual_target_values(
    values: np.ndarray,
) -> dict[str, float]:
    result = {}
    for target, (index, transform) in RESIDUAL_TARGETS.items():
        result[target] = float(transform(float(values[index])))
    return result


def attach_residual_sources(
    rows: pd.DataFrame,
    examples: list[dict],
    greybox_predictions: dict[int, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    result = rows.copy()
    mechanism = {}
    greybox = {}
    for item in examples:
        row_index = int(item["row"].name)
        for target, value in _residual_target_values(item["mechanism"]).items():
            mechanism[(row_index, target)] = value
    for row_index, (mechanism_values, greybox_values) in greybox_predictions.items():
        for target, value in _residual_target_values(mechanism_values).items():
            mechanism[(row_index, target)] = value
        for target, value in _residual_target_values(greybox_values).items():
            greybox[(row_index, target)] = value
    result["mechanism"] = [
        mechanism.get((int(row.row_index), row.target), np.nan)
        for row in result.itertuples()
    ]
    result["greybox_realfit"] = [
        greybox.get((int(row.row_index), row.target), np.nan)
        for row in result.itertuples()
    ]
    return result


def _scope(rows: pd.DataFrame, active_only: bool) -> pd.DataFrame:
    return rows[rows["active_dka"].astype(bool)] if active_only else rows


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
    errors = np.abs(selected[method].astype(float) - selected["truth"].astype(float))
    if normalized:
        errors = errors / selected["scale"].astype(float)
    return float(errors.mean())


def _stay_delta(
    rows: pd.DataFrame,
    method: str,
    baseline: str = "persistence",
    normalized: bool = True,
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
    temp = selected[["stay_id"]].copy()
    temp["delta"] = delta
    return temp.groupby("stay_id")["delta"].mean()


def _delta_summary(
    rows: pd.DataFrame,
    method: str,
    seed: int,
    samples: int,
    normalized: bool,
) -> dict:
    deltas = _stay_delta(rows, method, normalized=normalized)
    if deltas.empty:
        return {
            "stays": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_persistence": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=samples)
    point = float(values.mean())
    return {
        "stays": int(len(values)),
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
) -> dict:
    selected = _method_rows(rows, method)
    return {
        "rows": int(len(selected)),
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
    min_stays: int,
    bootstrap_samples: int,
    seed: int,
    methods: tuple[str, ...],
) -> tuple[dict[str, str], dict]:
    selected = {}
    details = {}
    scoped = _scope(discovery, active_only)
    for target_index, target in enumerate(TARGET_MAP):
        target_rows = scoped[scoped["target"] == target].copy()
        target_details = {}
        for method_index, method in enumerate(methods):
            target_details[method] = _method_stats(
                target_rows,
                method,
                seed=seed + 101 * target_index + method_index,
                samples=bootstrap_samples,
                normalized=False,
            )
        candidates = {}
        for method in methods:
            if method == "persistence":
                continue
            stats = target_details[method]
            delta = stats["delta_vs_persistence"]
            if stats["rows"] < min_pairs or stats["stays"] < min_stays:
                continue
            if not delta["significant"]:
                continue
            if stats["mae"] is None:
                continue
            candidates[method] = stats["mae"]
        if candidates:
            selected_method = min(candidates, key=candidates.get)
            reason = "lowest_discovery_mae_among_significant_sources"
        else:
            selected_method = "persistence"
            reason = "no_source_passed_discovery_significance_gate"
        selected[target] = selected_method
        details[target] = {
            "selected_method": selected_method,
            "reason": reason,
            "min_pairs": int(min_pairs),
            "min_stays": int(min_stays),
            "method_stats": target_details,
        }
    return selected, details


def _selected_methods_for_rows(rows: pd.DataFrame, selector: dict[str, str]) -> np.ndarray:
    return rows["target"].map(selector).fillna("persistence").to_numpy()


def _router_errors(rows: pd.DataFrame, selector: dict[str, str], normalized: bool) -> np.ndarray:
    if rows.empty:
        return np.asarray([], dtype=np.float64)
    selected = _selected_methods_for_rows(rows, selector)
    predictions = []
    for method, (_, row) in zip(selected, rows.iterrows()):
        value = row.get(method, np.nan)
        if np.isfinite(value):
            predictions.append(float(value))
        else:
            predictions.append(float(row["persistence"]))
    if not predictions:
        return np.asarray([], dtype=np.float64)
    errors = np.abs(
        np.asarray(predictions, dtype=np.float64)
        - rows["truth"].to_numpy(dtype=np.float64)
    )
    if normalized:
        errors = errors / rows["scale"].to_numpy(dtype=np.float64)
    return errors


def _router_delta(
    rows: pd.DataFrame,
    selector: dict[str, str],
    normalized: bool,
) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="float64")
    selected_methods = _selected_methods_for_rows(rows, selector)
    temp_rows = []
    for method, (_, row) in zip(selected_methods, rows.iterrows()):
        prediction = row.get(method, np.nan)
        if not np.isfinite(prediction):
            prediction = row["persistence"]
        scale = float(row["scale"]) if normalized else 1.0
        delta = (
            abs(float(prediction) - float(row["truth"]))
            - abs(float(row["persistence"]) - float(row["truth"]))
        ) / scale
        temp_rows.append((int(row["stay_id"]), delta))
    if not temp_rows:
        return pd.Series(dtype="float64")
    temp = pd.DataFrame(temp_rows, columns=["stay", "delta"])
    return temp.groupby("stay")["delta"].mean()


def _router_delta_summary(
    rows: pd.DataFrame,
    selector: dict[str, str],
    seed: int,
    samples: int,
    normalized: bool,
) -> dict:
    deltas = _router_delta(rows, selector, normalized=normalized)
    if deltas.empty:
        return {
            "stays": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_persistence": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=samples)
    point = float(values.mean())
    return {
        "stays": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_persistence": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _fixed_methods_summary(
    rows: pd.DataFrame,
    normalized: bool,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    return {
        method: _round(_mae(rows, method, normalized=normalized))
        for method in methods
    }


def _scope_summary(
    rows: pd.DataFrame,
    selector: dict[str, str],
    active_only: bool,
    seed: int,
    samples: int,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    scoped = _scope(rows, active_only)
    router_errors = _router_errors(scoped, selector, normalized=True)
    per_target = {}
    for target_index, target in enumerate(TARGET_MAP):
        target_rows = scoped[scoped["target"] == target]
        target_selector = {target: selector.get(target, "persistence")}
        per_target[target] = {
            "rows": int(len(target_rows)),
            "stays": int(target_rows["stay_id"].nunique()) if len(target_rows) else 0,
            "selected_method": target_selector[target],
            "mae": {
                **_fixed_methods_summary(target_rows, normalized=False, methods=methods),
                "target_router": _round(
                    float(
                        _router_errors(
                            target_rows,
                            target_selector,
                            normalized=False,
                        ).mean()
                    )
                    if len(target_rows) else None
                ),
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
        "stays": int(scoped["stay_id"].nunique()) if len(scoped) else 0,
        "normalized_mae": {
            **_fixed_methods_summary(scoped, normalized=True, methods=methods),
            "target_router": _round(
                float(router_errors.mean()) if len(router_errors) else None
            ),
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
    base_rows: pd.DataFrame,
    examples: list[dict],
    seed: int,
    discovery_fraction: float,
    min_pairs: int,
    min_stays: int,
    bootstrap_samples: int,
    greybox_epochs: int,
    greybox_bottleneck: int,
    greybox_inner_folds: int,
) -> dict:
    discovery_stays, heldout_stays = split_stays(
        base_rows,
        seed,
        discovery_fraction,
    )
    print(
        f"[target-router] split seed={seed} discovery_stays={len(discovery_stays)} "
        f"heldout_stays={len(heldout_stays)}",
        file=sys.stderr,
        flush=True,
    )
    greybox_predictions, greybox_fit = _nested_greybox_predictions(
        examples,
        discovery_stays,
        heldout_stays,
        seed=seed,
        epochs=greybox_epochs,
        bottleneck=greybox_bottleneck,
        inner_folds=greybox_inner_folds,
    )
    rows = attach_residual_sources(base_rows, examples, greybox_predictions)
    discovery = rows[rows["stay_id"].isin(discovery_stays)].copy()
    heldout = rows[rows["stay_id"].isin(heldout_stays)].copy()
    selector, selector_details = _choose_selector(
        discovery,
        active_only=True,
        min_pairs=min_pairs,
        min_stays=min_stays,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 10_000,
        methods=METHODS,
    )
    base_selector, base_selector_details = _choose_selector(
        discovery,
        active_only=True,
        min_pairs=min_pairs,
        min_stays=min_stays,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 40_000,
        methods=BASE_METHODS,
    )
    return {
        "seed": int(seed),
        "discovery_stays": int(len(discovery_stays)),
        "heldout_stays": int(len(heldout_stays)),
        "overlap": int(len(discovery_stays & heldout_stays)),
        "greybox_fit": greybox_fit,
        "selected_methods": selector,
        "base_selected_methods": base_selector,
        "discovery_selector": selector_details,
        "base_discovery_selector": base_selector_details,
        "heldout_all_windows": _scope_summary(
            heldout,
            selector,
            active_only=False,
            seed=seed + 20_000,
            samples=bootstrap_samples,
            methods=METHODS,
        ),
        "heldout_active_dka_only": _scope_summary(
            heldout,
            selector,
            active_only=True,
            seed=seed + 30_000,
            samples=bootstrap_samples,
            methods=METHODS,
        ),
        "base_heldout_all_windows": _scope_summary(
            heldout,
            base_selector,
            active_only=False,
            seed=seed + 50_000,
            samples=bootstrap_samples,
            methods=BASE_METHODS,
        ),
        "base_heldout_active_dka_only": _scope_summary(
            heldout,
            base_selector,
            active_only=True,
            seed=seed + 60_000,
            samples=bootstrap_samples,
            methods=BASE_METHODS,
        ),
    }


def _multi_seed_summary(reports: list[dict]) -> dict:
    def collect(scope: str) -> dict:
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

    def method_counts(field: str) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for report in reports:
            for target, method in report[field].items():
                counts.setdefault(target, {})
                counts[target][method] = counts[target].get(method, 0) + 1
        return counts

    def residual_gain(scope: str, base_scope: str) -> dict:
        differences = []
        for report in reports:
            router = report[scope]["router_delta_vs_persistence"]["point_delta"]
            base = report[base_scope]["router_delta_vs_persistence"]["point_delta"]
            if router is not None and base is not None:
                differences.append(float(router) - float(base))
        return {
            "splits": int(len(differences)),
            "delta_minus_base_min": _round(min(differences)) if differences else None,
            "delta_minus_base_median": (
                _round(float(np.median(differences))) if differences else None
            ),
            "delta_minus_base_max": _round(max(differences)) if differences else None,
            "improves_base_count": int(sum(value < 0 for value in differences)),
        }

    return {
        "target_router": {
            "active_dka_only": collect("heldout_active_dka_only"),
            "all_windows": collect("heldout_all_windows"),
        },
        "base_router_without_realfit": {
            "active_dka_only": collect("base_heldout_active_dka_only"),
            "all_windows": collect("base_heldout_all_windows"),
        },
        "target_router_minus_base": {
            "active_dka_only": residual_gain(
                "heldout_active_dka_only",
                "base_heldout_active_dka_only",
            ),
            "all_windows": residual_gain(
                "heldout_all_windows",
                "base_heldout_all_windows",
            ),
        },
        "selected_method_counts_by_target": method_counts("selected_methods"),
        "base_selected_method_counts_by_target": method_counts("base_selected_methods"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--v5-checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument(
        "--presentation-checkpoint",
        default="dka_physionet_presentation_only_candidate.pt",
    )
    parser.add_argument("--output", default="mimiciv_full_v31_icd_target_router.json")
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--min-stays", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--greybox-epochs", type=int, default=120)
    parser.add_argument("--greybox-bottleneck", type=int, default=8)
    parser.add_argument("--greybox-inner-folds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [
        int(item.strip()) for item in args.seeds.split(",")
        if item.strip()
    ]
    device = choose_device(args.device)
    v5, _ = load_checkpoint(args.v5_checkpoint, device)
    presentation, _ = load_checkpoint(args.presentation_checkpoint, device)
    frame = pd.read_parquet(args.cohort).reset_index(drop=True)
    examples = build_examples(frame)
    print(
        f"[target-router] built examples={len(examples)} "
        f"stays={len({item['stay_id'] for item in examples})}",
        file=sys.stderr,
        flush=True,
    )
    base_rows = build_base_rows(
        frame,
        {"v5": v5, "presentation_only": presentation},
        device,
    )
    print(
        f"[target-router] base prediction rows={len(base_rows)} "
        f"stays={base_rows['stay_id'].nunique() if len(base_rows) else 0}",
        file=sys.stderr,
        flush=True,
    )
    split_reports = []
    for seed in seeds:
        split_reports.append(_split_report(
            base_rows,
            examples,
            seed=seed,
            discovery_fraction=args.discovery_fraction,
            min_pairs=args.min_pairs,
            min_stays=args.min_stays,
            bootstrap_samples=args.bootstrap_samples,
            greybox_epochs=args.greybox_epochs,
            greybox_bottleneck=args.greybox_bottleneck,
            greybox_inner_folds=args.greybox_inner_folds,
        ))

    report = {
        "experiment": "MIMIC-IV full v3.1 DKA nested target-gated router",
        "cohort": str(Path(args.cohort).resolve()),
        "cohort_summary": _cohort_summary(frame, examples),
        "sources": {
            "persistence": "current observed value",
            "v5": str(Path(args.v5_checkpoint).resolve()),
            "presentation_only": str(Path(args.presentation_checkpoint).resolve()),
            "mechanism": "DKABody mechanism rollout for residual-supported targets",
            "greybox_realfit": (
                "small constrained residual fit on discovery stays; discovery "
                "selection uses inner out-of-fold predictions"
            ),
        },
        "selector_gate": {
            "scope": "discovery active-DKA rows",
            "min_pairs": int(args.min_pairs),
            "min_stays": int(args.min_stays),
            "rule": (
                "A non-persistence source can be selected only if its discovery "
                "stay-level bootstrap CI versus persistence excludes zero; among "
                "passing sources choose lowest discovery MAE. Otherwise fallback "
                "to persistence."
            ),
        },
        "greybox_training": {
            "epochs": int(args.greybox_epochs),
            "bottleneck": int(args.greybox_bottleneck),
            "inner_oof_folds": int(args.greybox_inner_folds),
            "heldout_fit_uses_only_discovery_stays": True,
            "discovery_selection_uses_inner_oof_predictions": True,
        },
        "random_patient_splits": {
            "seeds": seeds,
            "summary": _multi_seed_summary(split_reports),
            "runs": split_reports,
        },
        "safety_boundary": {
            "row_level_predictions_written": False,
            "patient_identifiers_written": False,
            "timestamp_cutoffs_written": False,
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
            "residual_artifact_promotion_allowed": False,
            "active_rule_promotion_allowed": False,
        },
        "interpretation": [
            "This tests whether the validated per-target architecture improves when a real-fit residual is added as one gated source.",
            "The primary result is the discovery-selected held-out router across patient splits.",
            "The grey-box source is not allowed to see held-out outcomes during selection or fitting for that held-out split.",
            "This remains observational and treatment-confounded even when it beats persistence factually.",
        ],
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
