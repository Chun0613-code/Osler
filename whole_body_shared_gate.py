"""Strict patient/hospital/module-balanced gate for the shared whole-body JEPA.

The shared encoder is a research candidate.  This gate is intentionally
harder than the smoke training: normalization is fit on the training patients
only; patient and hospital splits are independent; conformal intervals are
target-by-horizon; and module-balanced performance is reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import set_seed
from whole_body_shared_jepa import fit_shared as fit_shared_v1, load_shared_cohort


SEEDS = (7, 19, 31, 43, 59, 71, 83)


def _patient_split_column(frame: pd.DataFrame) -> str:
    """Use the real person ID, never the module-local history ID, for splits."""
    return "_patient_subject_id" if "_patient_subject_id" in frame.columns else "subject_id"


def _split_by_column(frame: pd.DataFrame, column: str, seed: int, fraction: float = 0.25):
    # Pandas may expose a read-only view here; the split must own a mutable
    # array because the seeded shuffle is part of the validation contract.
    values = frame[column].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    selected = set(values[: max(1, int(round(len(values) * fraction)))].tolist())
    test = frame[column].isin(selected).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _split_train_calibration(
    frame: pd.DataFrame,
    train_rows: np.ndarray,
    seed: int,
    patient_column: str | None = None,
):
    train_frame = frame.iloc[train_rows]
    patient_column = patient_column or _patient_split_column(train_frame)
    fit_local, cal_local = _split_by_column(
        train_frame.reset_index(drop=True), patient_column, seed, 0.20
    )
    return train_rows[fit_local], train_rows[cal_local]


def _forward(model, arrays, frame, rows, batch_size=2048):
    _, input_matrix, current, _, _, history, horizon = arrays
    module_ids = frame["_shared_module"].to_numpy(dtype=np.int64)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output.append(model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
                torch.from_numpy(module_ids[batch]),
            )["value"].numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, int(model.n_state)))


def _cell_metrics(frame, variables, arrays, prediction, rows):
    scaler, _, current, future, future_mask, _, horizon = arrays
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    pred_raw = prediction * scaler.scales + scaler.medians
    modules = frame["_shared_module"].to_numpy(dtype=np.int64)
    results = {}
    for index, target in enumerate(variables):
        for horizon_value in sorted(set(horizon[rows])):
            row_mask = (
                (horizon[rows] == float(horizon_value))
                & np.asarray(future_mask[rows, index], dtype=bool)
            )
            if row_mask.sum() < 30:
                continue
            selected = rows[row_mask]
            pred_index = np.flatnonzero(row_mask)
            candidate = np.abs(pred_raw[pred_index, index] - future_raw[selected, index])
            persistence = np.abs(current_raw[selected, index] - future_raw[selected, index])
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else float(horizon_value)}h"
            results[key] = {
                "target": target,
                "horizon_hours": float(horizon_value),
                "n": int(len(selected)),
                "candidate_mae": float(candidate.mean()),
                "persistence_mae": float(persistence.mean()),
                "delta_vs_persistence": float(candidate.mean() - persistence.mean()),
                "pass_value": bool(candidate.mean() < persistence.mean()),
            }
    return results


def _conformal(frame, variables, arrays, model, calibration_rows, test_rows):
    scaler, _, current, future, future_mask, _, horizon = arrays
    cal_pred = _forward(model, arrays, frame, calibration_rows)
    test_pred = _forward(model, arrays, frame, test_rows)
    future_raw = future * scaler.scales + scaler.medians
    cal_raw = cal_pred * scaler.scales + scaler.medians
    test_raw = test_pred * scaler.scales + scaler.medians
    output = {}
    for index, target in enumerate(variables):
        for horizon_value in sorted(set(horizon[calibration_rows]).intersection(horizon[test_rows])):
            cal_mask = (
                (horizon[calibration_rows] == float(horizon_value))
                & np.asarray(future_mask[calibration_rows, index], dtype=bool)
            )
            test_mask = (
                (horizon[test_rows] == float(horizon_value))
                & np.asarray(future_mask[test_rows, index], dtype=bool)
            )
            if cal_mask.sum() < 30 or test_mask.sum() < 30:
                continue
            residual = np.abs(cal_raw[cal_mask, index] - future_raw[calibration_rows[cal_mask], index])
            q90 = float(np.quantile(residual, 0.90, method="higher"))
            observed = future_raw[test_rows[test_mask], index]
            predicted = test_raw[test_mask, index]
            coverage = float(((observed >= predicted - q90) & (observed <= predicted + q90)).mean())
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else float(horizon_value)}h"
            output[key] = {
                "target": target,
                "horizon_hours": float(horizon_value),
                "calibration_n": int(cal_mask.sum()),
                "test_n": int(test_mask.sum()),
                "q90": q90,
                "coverage": coverage,
                "pass": bool(0.87 <= coverage <= 0.93),
            }
    return output


def _aggregate_split(split_results):
    keys = sorted(set().union(*(result.keys() for result in split_results)))
    output = {}
    for key in keys:
        cells = [result[key] for result in split_results if key in result]
        output[key] = {
            "n_splits": len(cells),
            "value_pass_splits": int(sum(cell["pass_value"] for cell in cells)),
            "median_delta": float(np.median([cell["delta_vs_persistence"] for cell in cells])),
            "pass_all_splits": bool(len(cells) == len(split_results) and all(cell["pass_value"] for cell in cells)),
        }
    return output


def run_gate(frame, variables, module_names, epochs=2, batch_size=2048, fitter=fit_shared_v1, architecture="v1"):
    patient_values, patient_conformal = [], []
    patient_column = _patient_split_column(frame)
    for seed in SEEDS:
        train_rows, test_rows = _split_by_column(frame, patient_column, seed)
        fit_rows, calibration_rows = _split_train_calibration(
            frame, train_rows, seed + 1001, patient_column
        )
        model, arrays, losses = fitter(
            frame, variables, len(module_names), fit_rows, seed=seed,
            epochs=epochs, batch_size=batch_size,
        )
        prediction = _forward(model, arrays, frame, test_rows, batch_size)
        patient_values.append(_cell_metrics(frame, variables, arrays, prediction, test_rows))
        patient_conformal.append(_conformal(frame, variables, arrays, model, calibration_rows, test_rows))
        print(json.dumps({"scope": "patient", "seed": seed, "cells": len(patient_values[-1]), "loss": losses[-1]}), flush=True)

    hospital_train, hospital_test = _split_by_column(frame, "hospitalid", 2026)
    fit_rows, calibration_rows = _split_train_calibration(
        frame, hospital_train, 3027, patient_column
    )
    model, arrays, losses = fitter(
        frame, variables, len(module_names), fit_rows, seed=2026,
        epochs=epochs, batch_size=batch_size,
    )
    hospital_values = _cell_metrics(frame, variables, arrays, _forward(model, arrays, frame, hospital_test), hospital_test)
    hospital_conformal = _conformal(frame, variables, arrays, model, calibration_rows, hospital_test)

    module_cells = []
    scaler, _, current, future, future_mask, _, horizon = arrays
    prediction = _forward(model, arrays, frame, hospital_test)
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    pred_raw = prediction * scaler.scales + scaler.medians
    hospital_modules = frame.iloc[hospital_test]["_shared_module"].to_numpy()
    for module_id, module in enumerate(module_names):
        scoped_positions = np.flatnonzero(hospital_modules == module_id)
        scoped = hospital_test[scoped_positions]
        if len(scoped) == 0:
            continue
        for index, target in enumerate(variables):
            for horizon_value in sorted(set(horizon[scoped])):
                mask = (
                    (horizon[scoped] == float(horizon_value))
                    & np.asarray(future_mask[scoped, index], dtype=bool)
                )
                if mask.sum() < 30:
                    continue
                selected = scoped[mask]
                selected_positions = scoped_positions[mask]
                candidate = np.abs(pred_raw[selected_positions, index] - future_raw[selected, index]).mean()
                persistence = np.abs(current_raw[selected, index] - future_raw[selected, index]).mean()
                module_cells.append({"module": module, "target": target, "horizon_hours": float(horizon_value), "n": int(mask.sum()), "delta": float(candidate - persistence), "pass": bool(candidate < persistence)})

    patient_aggregate = _aggregate_split(patient_values)
    conformal_keys = sorted(set().union(*(result.keys() for result in patient_conformal)))
    conformal_aggregate = {
        key: {
            "n_splits": sum(key in result for result in patient_conformal),
            "pass_splits": sum(bool(result.get(key, {}).get("pass")) for result in patient_conformal),
            "pass_all_splits": bool(all(result.get(key, {}).get("pass", False) for result in patient_conformal)),
            "median_coverage": float(np.median([result[key]["coverage"] for result in patient_conformal if key in result])) if any(key in result for result in patient_conformal) else None,
        }
        for key in conformal_keys
    }
    module_pass_rate = float(np.mean([cell["pass"] for cell in module_cells])) if module_cells else 0.0
    module_by_target_horizon = {}
    for cell in module_cells:
        key = f"{cell['target']}@{int(cell['horizon_hours']) if float(cell['horizon_hours']).is_integer() else cell['horizon_hours']}h"
        module_by_target_horizon.setdefault(key, []).append(cell)
    module_target_horizon = {
        key: {
            "n_modules": len(cells),
            "pass_rate": float(np.mean([cell["pass"] for cell in cells])),
            "pass_module_threshold": bool(
                len(cells) >= 3 and np.mean([cell["pass"] for cell in cells]) >= 0.80
            ),
        }
        for key, cells in sorted(module_by_target_horizon.items())
    }
    eligible_value = [cell for cell in patient_aggregate.values() if cell["n_splits"] == len(SEEDS)]
    eligible_conformal = [cell for cell in conformal_aggregate.values() if cell["n_splits"] == len(SEEDS)]
    strict = bool(
        eligible_value
        and eligible_conformal
        and all(cell["pass_all_splits"] for cell in eligible_value)
        and all(cell["pass_all_splits"] for cell in eligible_conformal)
        and all(cell.get("pass_value", False) for cell in hospital_values.values())
        and all(cell.get("pass", False) for cell in hospital_conformal.values())
        and module_pass_rate >= 0.80
    )
    validated_cells = []
    cell_registry = {}
    for key, value in patient_aggregate.items():
        module_evidence = module_target_horizon.get(key, {})
        cell_pass = bool(
            value.get("n_splits") == len(SEEDS)
            and value.get("pass_all_splits", False)
            and conformal_aggregate.get(key, {}).get("n_splits") == len(SEEDS)
            and conformal_aggregate.get(key, {}).get("pass_all_splits", False)
            and hospital_values.get(key, {}).get("pass_value", False)
            and hospital_conformal.get(key, {}).get("pass", False)
            and module_evidence.get("pass_module_threshold", False)
        )
        cell_registry[key] = {
            "validated": cell_pass,
            "patient_value": bool(value.get("pass_all_splits", False)),
            "patient_conformal": bool(conformal_aggregate.get(key, {}).get("pass_all_splits", False)),
            "hospital_value": bool(hospital_values.get(key, {}).get("pass_value", False)),
            "hospital_conformal": bool(hospital_conformal.get(key, {}).get("pass", False)),
            "module_evidence": module_evidence,
        }
        if cell_pass:
            validated_cells.append(key)
    return {
        "schema": f"whole_body_shared_jepa_gate.{architecture}",
        "architecture": architecture,
        "modules": list(module_names),
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame[_patient_split_column(frame)].nunique()),
        "module_local_history_subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "patient_heldout": {"seeds": list(SEEDS), "aggregate": patient_aggregate, "conformal": conformal_aggregate},
        "hospital_heldout": {"values": hospital_values, "conformal": hospital_conformal},
        "module_balanced": {"eligible_cells": len(module_cells), "pass_rate": module_pass_rate, "cells": module_cells},
        "module_balanced_by_target_horizon": module_target_horizon,
        "validated_target_horizon_cells": sorted(validated_cells),
        "validated_target_horizon_count": len(validated_cells),
        "target_horizon_registry": cell_registry,
        "promotion_status": "validated_shared_latent_research_only" if strict else "candidate_only",
        "shared_latent_validated": strict,
        "shared_latent_partially_validated": bool(validated_cells),
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "fail_closed_reason": None if strict else "global shared latent remains candidate; only validated target-horizon cells may be used, all other cells must use the per-module registry or persistence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--modules", default="dka,sepsis,aki,respiratory,integumentary_skin_wound,toxic_metabolic,electrolyte_acid_base,endocrine_stress,gi_pancreatic_nutrition,cardiac_injury,musculoskeletal_rhabdo,immune_inflammatory,cardiovascular_instability,acute_neuro,hepatic_failure,coagulopathy_heme")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--architecture", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--output", type=Path, default=Path("whole_body_shared_jepa_gate.json"))
    args = parser.parse_args()
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    frame, variables, module_names = load_shared_cohort(args.data_root, modules, horizons, args.max_stays)
    if args.architecture == "v2":
        from whole_body_shared_jepa_v2 import fit_shared_v2

        fitter = fit_shared_v2
    elif args.architecture == "v3":
        from whole_body_shared_jepa_v3 import fit_shared_v3

        fitter = fit_shared_v3
    else:
        fitter = fit_shared_v1
    report = run_gate(
        frame,
        variables,
        module_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fitter=fitter,
        architecture=args.architecture,
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "promotion_status": report["promotion_status"], "shared_latent_validated": report["shared_latent_validated"], "rows": report["rows"], "modules": report["modules"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
