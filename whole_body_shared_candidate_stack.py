"""Nested candidate-library audit on one true-patient shared-JEPA contract.

Candidates are trained only on each outer discovery fold.  A second, disjoint
patient calibration fold selects a hard router or convex stack; the outer test
patients and hospitals are never used to choose sources or weights.  This is a
diagnostic bridge between per-target factual routers and the shared-JEPA
candidate, not an automatic promotion path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_library_complementarity_audit import (
    BASELINE,
    _fit_cell_stacks,
    _hard_prediction,
    _oracle_prediction,
    _select_hard_router,
    _stack_prediction,
    _summary,
)
from whole_body_shared_gate import SEEDS, _forward, _patient_split_column, _split_by_column
from whole_body_shared_jepa import fit_shared, load_shared_cohort
from whole_body_shared_jepa_v3 import fit_shared_v3


METHODS = ("prediction__ridge_realfit", "prediction__shared_v1", "prediction__shared_v3")
DEFAULT_TARGETS = (
    "glucose,map,bicarbonate,creatinine,potassium,heart_rate,o2sat,"
    "respiratory_rate,sodium,bun"
)


def _round(value):
    return round(float(value), 6) if value is not None and np.isfinite(value) else None


def _raw_matrix(frame: pd.DataFrame, variables: tuple[str, ...], suffix: str) -> np.ndarray:
    columns = []
    for variable in variables:
        column = f"future_{variable}" if suffix == "_future" else f"{variable}{suffix}"
        values = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
        columns.append(values.to_numpy(dtype=np.float64))
    return np.column_stack(columns)


def _ridge_features(frame: pd.DataFrame, variables: tuple[str, ...], module_count: int) -> np.ndarray:
    current = _raw_matrix(frame, variables, "_t")
    ages = _raw_matrix(frame, variables, "_age_hr")
    horizon = pd.to_numeric(frame["_source_horizon"], errors="coerce").to_numpy(dtype=np.float64)[:, None]
    onset_series = frame.get("hours_since_onset", pd.Series(0.0, index=frame.index))
    onset = pd.to_numeric(onset_series, errors="coerce").to_numpy(dtype=np.float64)[:, None]
    modules = frame["_shared_module"].to_numpy(dtype=np.int64)
    one_hot = np.zeros((len(frame), module_count), dtype=np.float64)
    one_hot[np.arange(len(frame)), np.clip(modules, 0, module_count - 1)] = 1.0
    return np.concatenate([current, ages, horizon, onset, one_hot], axis=1)


def _fit_ridge_predictions(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    train_rows: np.ndarray,
    prediction_rows: np.ndarray,
    module_count: int,
    alpha: float,
) -> np.ndarray:
    """Fit one train-only ridge residual per target and return raw values."""
    features = _ridge_features(frame, variables, module_count)
    current = _raw_matrix(frame, variables, "_t")
    future = _raw_matrix(frame, variables, "_future")
    output = current[prediction_rows].copy()
    for index in range(len(variables)):
        usable = train_rows[np.isfinite(current[train_rows, index]) & np.isfinite(future[train_rows, index])]
        if len(usable) < max(80, features.shape[1] + 10):
            continue
        x_train = features[usable]
        available = np.isfinite(x_train).any(axis=0)
        if int(available.sum()) == 0:
            continue
        x_train = x_train[:, available]
        medians = np.nanmedian(x_train, axis=0)
        medians = np.nan_to_num(medians, nan=0.0)
        x_train = np.where(np.isfinite(x_train), x_train, medians)
        scale = np.nanstd(x_train, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        x_train = (x_train - medians) / scale
        x_train = np.c_[np.ones(len(x_train)), x_train]
        y = future[usable, index] - current[usable, index]
        penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ (x_train.T @ y)
        x_test = features[prediction_rows][:, available]
        x_test = np.where(np.isfinite(x_test), x_test, medians)
        x_test = np.c_[np.ones(len(x_test)), (x_test - medians) / scale]
        valid = np.isfinite(current[prediction_rows, index])
        output[valid, index] = current[prediction_rows[valid], index] + x_test[valid] @ beta
    return output


def _jepa_prediction(fitter, frame, variables, module_count, train_rows, rows, seed, epochs, batch_size):
    model, arrays, loss = fitter(
        frame, variables, module_count, train_rows, seed=seed, epochs=epochs, batch_size=batch_size
    )
    predicted = _forward(model, arrays, frame, rows, batch_size)
    scaler = arrays[0]
    return predicted * scaler.scales + scaler.medians, loss


def _target_horizon_scales(frame, variables, train_rows, targets) -> dict[str, float]:
    future = _raw_matrix(frame, variables, "_future")
    horizon = pd.to_numeric(frame["_source_horizon"], errors="coerce").to_numpy(dtype=np.float64)
    scales = {}
    for index, target in enumerate(variables):
        if target not in targets:
            continue
        for value in np.unique(horizon[train_rows]):
            observed = future[train_rows[(horizon[train_rows] == value) & np.isfinite(future[train_rows, index])], index]
            if len(observed) < 20:
                continue
            scale = float(np.subtract(*np.quantile(observed, [0.75, 0.25])))
            if scale < 1e-6:
                scale = float(np.std(observed))
            scales[f"{target}@{value:g}h"] = max(scale, 1.0e-6)
    return scales


def _long_rows(frame, variables, rows, targets, candidate_predictions: dict[str, np.ndarray], scales: dict[str, float], module_names) -> pd.DataFrame:
    current = _raw_matrix(frame, variables, "_t")
    future = _raw_matrix(frame, variables, "_future")
    horizon = pd.to_numeric(frame["_source_horizon"], errors="coerce").to_numpy(dtype=np.float64)
    parts = []
    for index, target in enumerate(variables):
        if target not in targets:
            continue
        selected = rows[np.isfinite(current[rows, index]) & np.isfinite(future[rows, index])]
        if not len(selected):
            continue
        part = pd.DataFrame({
            "row_id": selected,
            "subject_id": frame.iloc[selected]["_patient_subject_id"].astype(str).to_numpy(),
            "hospitalid": frame.iloc[selected]["hospitalid"].to_numpy(),
            "module": [module_names[index] for index in frame.iloc[selected]["_shared_module"].to_numpy(dtype=np.int64)],
            "target": target,
            "horizon_hours": horizon[selected],
            "truth": future[selected, index],
            BASELINE: current[selected, index],
        })
        part["cell"] = part["target"].astype(str) + "@" + part["horizon_hours"].map(lambda x: f"{x:g}h")
        part["scale"] = part["cell"].map(scales).fillna(1.0)
        positions = np.searchsorted(rows, selected)
        for name, values in candidate_predictions.items():
            part[name] = values[positions, index]
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return combined


def _train_sources(frame, variables, module_count, train_rows, prediction_rows, seed, epochs, batch_size, ridge_alpha):
    v1, v1_loss = _jepa_prediction(fit_shared, frame, variables, module_count, train_rows, prediction_rows, seed, epochs, batch_size)
    v3, v3_loss = _jepa_prediction(fit_shared_v3, frame, variables, module_count, train_rows, prediction_rows, seed + 101, epochs, batch_size)
    ridge = _fit_ridge_predictions(frame, variables, train_rows, prediction_rows, module_count, ridge_alpha)
    return {
        "prediction__shared_v1": v1,
        "prediction__shared_v3": v3,
        "prediction__ridge_realfit": ridge,
    }, {"v1": [_round(value) for value in v1_loss], "v3": [_round(value) for value in v3_loss]}


def _meta_splits(frame, discovery_rows, seed):
    column = _patient_split_column(frame)
    local = frame.iloc[discovery_rows].reset_index(drop=True)
    fit_local, meta_local = _split_by_column(local, column, seed, fraction=0.40)
    meta_rows = discovery_rows[meta_local]
    meta_frame = frame.iloc[meta_rows].reset_index(drop=True)
    selection_local, interval_local = _split_by_column(
        meta_frame, column, seed + 1, fraction=0.50
    )
    return discovery_rows[fit_local], meta_rows[selection_local], meta_rows[interval_local]


def _report_methods(frame, predictions, seed, samples):
    values = {BASELINE: frame[BASELINE].to_numpy(dtype=np.float64), **predictions}
    normalized = frame.copy()
    for name, values_array in values.items():
        normalized[name] = values_array / normalized["scale"].to_numpy(dtype=np.float64)
    normalized["truth"] = normalized["truth"].to_numpy(dtype=np.float64) / normalized["scale"].to_numpy(dtype=np.float64)
    return {
        name: {
            "raw": _summary(frame, values_array, seed + index, samples),
            "target_horizon_normalized": _summary(
                normalized, normalized[name].to_numpy(dtype=np.float64), seed + 10_000 + index, samples
            ),
        }
        for index, (name, values_array) in enumerate(values.items())
    }


def _stack_conformal(calibration: pd.DataFrame, test: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    """Calibrate raw-unit 90% intervals on patients unused for source selection."""
    if calibration.empty or test.empty:
        return {"cells": {}, "eligible_cells": 0, "passing_cells": 0}
    calibration = calibration.copy()
    calibration["stack_prediction"] = _stack_prediction(
        calibration, calibration.attrs["stacks"]
    )
    cells = {}
    for cell, test_group in test.groupby("cell", sort=False):
        cal_group = calibration[calibration["cell"] == cell]
        if len(cal_group) < 30 or len(test_group) < 30:
            continue
        q90 = float(np.quantile(np.abs(
            cal_group["stack_prediction"].to_numpy(dtype=np.float64)
            - cal_group["truth"].to_numpy(dtype=np.float64)
        ), 0.90, method="higher"))
        positions = test_group.index.to_numpy(dtype=np.int64)
        observed = test_group["truth"].to_numpy(dtype=np.float64)
        predicted = prediction[positions]
        coverage = float(((observed >= predicted - q90) & (observed <= predicted + q90)).mean())
        cells[cell] = {
            "calibration_rows": int(len(cal_group)),
            "test_rows": int(len(test_group)),
            "q90": _round(q90),
            "coverage": _round(coverage),
            "pass": bool(0.87 <= coverage <= 0.93),
        }
    return {
        "cells": cells,
        "eligible_cells": int(len(cells)),
        "passing_cells": int(sum(bool(value["pass"]) for value in cells.values())),
    }


def _cell_metrics(frame: pd.DataFrame, predictions: dict[str, np.ndarray], seed: int, samples: int) -> dict[str, object]:
    """Per target×horizon held-out evidence; aggregate scores are insufficient."""
    output = {}
    for index, (cell, group) in enumerate(frame.groupby("cell", sort=False)):
        positions = group.index.to_numpy(dtype=np.int64)
        output[cell] = {
            name: _summary(group, values[positions], seed + 100 * index + offset, samples)
            for offset, (name, values) in enumerate(predictions.items())
        }
    return output


def _module_balance(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    """Require a stack to help broadly, rather than hiding a single-module win."""
    output = {}
    for (cell, module), group in frame.groupby(["cell", "module"], sort=False):
        if group["subject_id"].nunique() < 20 or len(group) < 30:
            continue
        positions = group.index.to_numpy(dtype=np.int64)
        candidate = np.abs(prediction[positions] - group["truth"].to_numpy(dtype=np.float64)).mean()
        baseline = np.abs(group[BASELINE].to_numpy(dtype=np.float64) - group["truth"].to_numpy(dtype=np.float64)).mean()
        output.setdefault(cell, []).append({
            "module": module,
            "rows": int(len(group)),
            "subjects": int(group["subject_id"].nunique()),
            "delta_vs_persistence": _round(candidate - baseline),
            "pass": bool(candidate < baseline),
        })
    return {
        cell: {
            "eligible_modules": int(len(entries)),
            "pass_rate": _round(np.mean([entry["pass"] for entry in entries])) if entries else None,
            "pass": bool(len(entries) >= 3 and np.mean([entry["pass"] for entry in entries]) >= 0.80),
            "modules": entries,
        }
        for cell, entries in output.items()
    }


def _run_outer(frame, variables, module_names, targets, train_rows, test_rows, seed, args, scope):
    module_count = len(module_names)
    fit_rows, selection_rows, interval_rows = _meta_splits(frame, train_rows, seed + 500)
    meta_rows = np.sort(np.concatenate([selection_rows, interval_rows]))
    meta_sources, meta_loss = _train_sources(
        frame, variables, module_count, fit_rows, meta_rows, seed + 1000,
        args.epochs, args.batch_size, args.ridge_alpha,
    )
    scales = _target_horizon_scales(frame, variables, fit_rows, targets)
    meta = _long_rows(frame, variables, meta_rows, targets, meta_sources, scales, module_names).dropna(subset=list(METHODS)).copy()
    selection = meta[meta["row_id"].isin(set(selection_rows.tolist()))].copy().reset_index(drop=True)
    interval = meta[meta["row_id"].isin(set(interval_rows.tolist()))].copy().reset_index(drop=True)
    selector = _select_hard_router(selection, list(METHODS), args.min_subjects, seed + 2000, args.bootstrap_samples)
    stacks = _fit_cell_stacks(selection, list(METHODS), args.min_subjects)
    test_sources, test_loss = _train_sources(
        frame, variables, module_count, train_rows, test_rows, seed + 3000,
        args.epochs, args.batch_size, args.ridge_alpha,
    )
    test = _long_rows(frame, variables, test_rows, targets, test_sources, scales, module_names).dropna(subset=list(METHODS)).copy().reset_index(drop=True)
    predictions = {name: test[name].to_numpy(dtype=np.float64) for name in METHODS}
    predictions["hard_router"] = _hard_prediction(test, selector)
    predictions["convex_stack"] = _stack_prediction(test, stacks)
    predictions["oracle_upper_bound_leaky"] = _oracle_prediction(test, list(METHODS))
    interval.attrs["stacks"] = stacks
    return {
        "scope": scope,
        "seed": int(seed),
        "train_rows": int(len(train_rows)),
        "meta_rows": int(len(meta_rows)),
        "test_rows": int(len(test_rows)),
        "common_test_label_rows": int(len(test)),
        "meta_training": {
            "fit_rows": int(len(fit_rows)),
            "selection_rows": int(len(selection)),
            "interval_calibration_rows": int(len(interval)),
            "cells": int(selection["cell"].nunique()),
            "loss": meta_loss,
        },
        "test_training": {"loss": test_loss},
        "methods": _report_methods(test, predictions, seed + 4000, args.bootstrap_samples),
        "test_cell_metrics": _cell_metrics(test, predictions, seed + 5000, args.bootstrap_samples),
        "module_balanced": _module_balance(test, predictions["convex_stack"]),
        "convex_stack_conformal": _stack_conformal(interval, test, predictions["convex_stack"]),
        "hard_router_cells": selector,
        "convex_stack_cells": stacks,
    }


def _gate(reports, method):
    values = [report["methods"][method]["target_horizon_normalized"] for report in reports]
    return {
        "splits": len(values),
        "significant_wins": int(sum(bool(value["significant_win"]) for value in values)),
        "all_splits_significant": bool(values) and all(bool(value["significant_win"]) for value in values),
        "median_delta_vs_persistence": _round(np.median([value["delta_vs_persistence"] for value in values])),
    }


def _runtime_registry(patient_reports, hospital_report):
    """Return only cells that have independently earned runtime movement."""
    keys = sorted(set().union(*(report["test_cell_metrics"].keys() for report in patient_reports)))
    registry = {}
    for cell in keys:
        patient_cells = [report["test_cell_metrics"].get(cell, {}).get("convex_stack") for report in patient_reports]
        patient_conformal = [report["convex_stack_conformal"]["cells"].get(cell) for report in patient_reports]
        patient_module = [report["module_balanced"].get(cell) for report in patient_reports]
        hospital_cell = hospital_report["test_cell_metrics"].get(cell, {}).get("convex_stack")
        hospital_conformal = hospital_report["convex_stack_conformal"]["cells"].get(cell)
        hospital_module = hospital_report["module_balanced"].get(cell)
        patient_value = bool(patient_cells) and all(item is not None and item["significant_win"] for item in patient_cells)
        patient_interval = bool(patient_conformal) and all(item is not None and item["pass"] for item in patient_conformal)
        patient_modules = bool(patient_module) and all(item is not None and item["pass"] for item in patient_module)
        hospital_value = bool(hospital_cell and hospital_cell["significant_win"])
        hospital_interval = bool(hospital_conformal and hospital_conformal["pass"])
        hospital_modules = bool(hospital_module and hospital_module["pass"])
        validated = all((patient_value, patient_interval, patient_modules, hospital_value, hospital_interval, hospital_modules))
        registry[cell] = {
            "patient_value": patient_value,
            "patient_conformal": patient_interval,
            "patient_module_balanced": patient_modules,
            "hospital_value": hospital_value,
            "hospital_conformal": hospital_interval,
            "hospital_module_balanced": hospital_modules,
            "validated_for_runtime": validated,
        }
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--modules", default="sepsis,aki,respiratory,electrolyte_acid_base,cardiovascular_instability")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--max-stays", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-subjects", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    targets = {value.strip() for value in args.targets.split(",") if value.strip()}
    frame, variables, module_names = load_shared_cohort(args.data_root, modules, horizons, args.max_stays)
    targets &= set(variables)
    if not targets:
        raise ValueError("No requested targets exist in the shared cohort")
    patient_column = _patient_split_column(frame)
    reports = []
    for seed in (int(value) for value in args.seeds.split(",") if value.strip()):
        train, test = _split_by_column(frame, patient_column, seed)
        reports.append(_run_outer(frame, variables, module_names, targets, train, test, seed, args, "patient"))
        print(json.dumps({"scope": "patient", "seed": seed, "stack": reports[-1]["methods"]["convex_stack"]}), flush=True)
    hospital_train, hospital_test = _split_by_column(frame, "hospitalid", 2026)
    hospital = _run_outer(frame, variables, module_names, targets, hospital_train, hospital_test, 2026, args, "hospital")
    registry = _runtime_registry(reports, hospital)
    validated_cells = sorted(
        key for key, value in registry.items() if value["validated_for_runtime"]
    )
    if not validated_cells:
        promotion_status = "candidate_only"
    elif len(validated_cells) == len(registry):
        promotion_status = "validated_shared_stack_research_only"
    else:
        promotion_status = "validated_target_gated_shared_stack_research_only"
    report = {
        "schema": "whole_body_shared_candidate_stack.v1",
        "data_contract": "shared module rows; true-patient outer splits; module-local histories",
        "rows": int(len(frame)),
        "subjects": int(frame[patient_column].nunique()),
        "module_local_history_subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "modules": list(module_names),
        "targets": sorted(targets),
        "candidate_sources": [BASELINE, *METHODS],
        "patient_heldout": reports,
        "patient_gate": {method: _gate(reports, method) for method in ("hard_router", "convex_stack", "oracle_upper_bound_leaky")},
        "hospital_heldout": hospital,
        "runtime_registry": registry,
        "validated_runtime_cells": validated_cells,
        "validated_runtime_cell_count": len(validated_cells),
        "interpretation": {
            "oracle": "Leaky per-row best-source upper bound only.",
            "decision": "An oracle failure means these candidates do not contain enough complementary signal for a router to create it.",
            "promotion": "Only runtime-registry cells that pass all patient, hospital, conformal, and module-balance gates may move in factual research runtime. All other cells fallback.",
        },
        "promotion_status": promotion_status,
        "runtime_promotion_allowed": bool(validated_cells),
        "clinical_promotion_allowed": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "patient_gate": report["patient_gate"], "hospital_stack": hospital["methods"]["convex_stack"]}, indent=2))


if __name__ == "__main__":
    main()
