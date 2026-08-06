"""Diagnose whether factual forecast candidates contain complementary signal.

This is deliberately an audit, not a promotion path.  It requires row-level
predictions for the *same* patient/anchor/target/horizon label contract.  The
script compares a discovery-selected hard router, a non-negative convex stack,
and a leaky per-row oracle against persistence on held-out patients and
hospitals.  Aggregate reports from unrelated experiments are not accepted as
candidate inputs because they cannot establish complementarity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = (7, 11, 19, 23, 37, 53, 71)
BASELINE = "persistence"
REQUIRED_COLUMNS = ("subject_id", "target", "horizon_hours", "truth", BASELINE)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("--predictions must be a .csv or .parquet file")


def _round(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None and np.isfinite(value) else None


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> list[float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [_round(np.quantile(means, 0.025)), _round(np.quantile(means, 0.975))]


def _subject_deltas(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    candidate_error = np.abs(prediction - frame["truth"].to_numpy(dtype=np.float64))
    baseline_error = np.abs(frame[BASELINE].to_numpy(dtype=np.float64) - frame["truth"].to_numpy(dtype=np.float64))
    delta = candidate_error - baseline_error
    grouped = pd.DataFrame({"subject_id": frame["subject_id"].astype(str), "delta": delta})
    return grouped.groupby("subject_id", sort=False)["delta"].mean().to_numpy(dtype=np.float64)


def _summary(frame: pd.DataFrame, prediction: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "subjects": 0, "mae": None, "delta_vs_persistence": None,
                "bootstrap_95_ci": [None, None], "significant_win": False}
    deltas = _subject_deltas(frame, prediction)
    ci = _bootstrap_ci(deltas, seed, samples)
    point = float(deltas.mean()) if len(deltas) else None
    return {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "mae": _round(np.abs(prediction - frame["truth"].to_numpy(dtype=np.float64)).mean()),
        "delta_vs_persistence": _round(point),
        "bootstrap_95_ci": ci,
        "significant_win": bool(point is not None and point < 0.0 and ci[1] is not None and ci[1] < 0.0),
    }


def _split_groups(values: pd.Series, seed: int, fraction: float = 0.67) -> tuple[set[str], set[str]]:
    groups = np.asarray(sorted(values.astype(str).unique()), dtype=object)
    if len(groups) < 3:
        raise ValueError("Need at least three independent groups for a split")
    rng = np.random.default_rng(seed)
    shuffled = groups[rng.permutation(len(groups))]
    cutoff = max(1, min(len(groups) - 1, int(round(len(groups) * fraction))))
    return set(shuffled[:cutoff]), set(shuffled[cutoff:])


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto non-negative weights that sum to one."""
    ordered = np.sort(vector)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    index = np.arange(1, len(vector) + 1)
    valid = ordered - cssv / index > 0
    rho = int(np.flatnonzero(valid)[-1]) if valid.any() else 0
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(vector - theta, 0.0)


def _fit_convex_weights(frame: pd.DataFrame, methods: list[str], steps: int = 400) -> np.ndarray:
    """Fit a conservative convex MSE stack on discovery rows only."""
    x = frame[methods].to_numpy(dtype=np.float64)
    y = frame["truth"].to_numpy(dtype=np.float64)
    # Standardising preserves the simplex constraint while preventing a target
    # with large units from making the gradient numerically unstable.
    center = float(np.mean(y))
    scale = max(float(np.std(y)), 1e-6)
    x = (x - center) / scale
    y = (y - center) / scale
    weights = np.full(len(methods), 1.0 / len(methods), dtype=np.float64)
    spectral = float(np.linalg.norm(x, ord=2) ** 2 / max(len(x), 1))
    rate = 1.0 / max(spectral, 1.0)
    for _ in range(steps):
        residual = x @ weights - y
        gradient = (2.0 / len(x)) * (x.T @ residual)
        weights = _project_simplex(weights - rate * gradient)
    return weights


def _group_key(frame: pd.DataFrame) -> pd.Series:
    horizon = pd.to_numeric(frame["horizon_hours"], errors="coerce")
    return frame["target"].astype(str) + "@" + horizon.map(lambda value: f"{value:g}h")


def _select_hard_router(discovery: pd.DataFrame, methods: list[str], min_subjects: int, seed: int, samples: int) -> dict[str, str]:
    selected: dict[str, str] = {}
    for index, (key, group) in enumerate(discovery.groupby("cell", sort=False)):
        choices = {BASELINE: _summary(group, group[BASELINE].to_numpy(dtype=np.float64), seed + index, samples)}
        for offset, method in enumerate(methods):
            choices[method] = _summary(group, group[method].to_numpy(dtype=np.float64), seed + 1000 + index + offset, samples)
        passing = {
            method: details["mae"]
            for method, details in choices.items()
            if method != BASELINE and details["subjects"] >= min_subjects and details["significant_win"] and details["mae"] is not None
        }
        selected[key] = min(passing, key=passing.get) if passing else BASELINE
    return selected


def _fit_cell_stacks(discovery: pd.DataFrame, methods: list[str], min_subjects: int) -> dict[str, dict[str, object]]:
    fitted: dict[str, dict[str, object]] = {}
    all_methods = [BASELINE, *methods]
    for key, group in discovery.groupby("cell", sort=False):
        if group["subject_id"].nunique() < min_subjects:
            fitted[key] = {"methods": [BASELINE], "weights": [1.0], "reason": "insufficient_discovery_subjects"}
            continue
        weights = _fit_convex_weights(group, all_methods)
        fitted[key] = {"methods": all_methods, "weights": [_round(value) for value in weights], "reason": "discovery_convex_mse"}
    return fitted


def _hard_prediction(frame: pd.DataFrame, selector: dict[str, str]) -> np.ndarray:
    choices = frame["cell"].map(selector).fillna(BASELINE).to_numpy(dtype=object)
    prediction = frame[BASELINE].to_numpy(dtype=np.float64).copy()
    for method in set(choices) - {BASELINE}:
        mask = choices == method
        prediction[mask] = frame.loc[mask, method].to_numpy(dtype=np.float64)
    return prediction


def _stack_prediction(frame: pd.DataFrame, stacks: dict[str, dict[str, object]]) -> np.ndarray:
    prediction = frame[BASELINE].to_numpy(dtype=np.float64).copy()
    # ``groups`` returns original DataFrame labels.  Held-out frames retain
    # their source labels, while ``prediction`` is indexed by local position.
    for key, positions in frame.groupby("cell", sort=False).indices.items():
        fit = stacks.get(key)
        if fit is None:
            continue
        methods = list(fit["methods"])
        weights = np.asarray(fit["weights"], dtype=np.float64)
        local = np.asarray(positions, dtype=np.int64)
        prediction[local] = (
            frame.iloc[local][methods].to_numpy(dtype=np.float64) @ weights
        )
    return prediction


def _oracle_prediction(frame: pd.DataFrame, methods: list[str]) -> np.ndarray:
    candidates = frame[[BASELINE, *methods]].to_numpy(dtype=np.float64)
    truth = frame["truth"].to_numpy(dtype=np.float64)[:, None]
    return candidates[np.arange(len(frame)), np.abs(candidates - truth).argmin(axis=1)]


def _evaluate_split(frame: pd.DataFrame, group_column: str, seed: int, methods: list[str], min_subjects: int, samples: int) -> dict[str, object]:
    discovery_groups, heldout_groups = _split_groups(frame[group_column], seed)
    discovery = frame[frame[group_column].astype(str).isin(discovery_groups)].copy()
    heldout = frame[frame[group_column].astype(str).isin(heldout_groups)].copy()
    hard_selector = _select_hard_router(discovery, methods, min_subjects, seed + 11, samples)
    stacks = _fit_cell_stacks(discovery, methods, min_subjects)
    values = {
        BASELINE: heldout[BASELINE].to_numpy(dtype=np.float64),
        "hard_router": _hard_prediction(heldout, hard_selector),
        "convex_stack": _stack_prediction(heldout, stacks),
        "oracle_upper_bound_leaky": _oracle_prediction(heldout, methods),
    }
    for method in methods:
        values[method] = heldout[method].to_numpy(dtype=np.float64)
    return {
        "seed": int(seed),
        "group_column": group_column,
        "discovery_groups": int(len(discovery_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "overlap": int(len(discovery_groups & heldout_groups)),
        "methods": {name: _summary(heldout, prediction, seed + 101 + index, samples) for index, (name, prediction) in enumerate(values.items())},
        "hard_router_cells": hard_selector,
        "convex_stack_cells": stacks,
    }


def _promotion_gate(reports: list[dict[str, object]], method: str) -> dict[str, object]:
    results = [report["methods"][method] for report in reports]
    return {
        "splits": int(len(results)),
        "significant_wins": int(sum(bool(item["significant_win"]) for item in results)),
        "all_splits_significant": bool(results) and all(bool(item["significant_win"]) for item in results),
        "median_delta_vs_persistence": _round(np.median([item["delta_vs_persistence"] for item in results if item["delta_vs_persistence"] is not None])) if results else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Long CSV/parquet with common-label candidate predictions")
    parser.add_argument("--candidates", default=None, help="Comma-separated prediction columns; defaults to prediction__* columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-subjects", type=int, default=50)
    parser.add_argument("--source-provenance", choices=("unknown", "cross_fitted"), default="unknown")
    args = parser.parse_args()

    frame = _read_frame(args.predictions)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if args.candidates:
        methods = [value.strip() for value in args.candidates.split(",") if value.strip()]
    else:
        methods = [column for column in frame.columns if column.startswith("prediction__")]
    if not methods:
        raise ValueError("Provide --candidates or prediction__* columns")
    missing = [column for column in methods if column not in frame]
    if missing:
        raise ValueError(f"Candidate columns absent from input: {missing}")
    for column in ["truth", BASELINE, *methods]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["horizon_hours"] = pd.to_numeric(frame["horizon_hours"], errors="coerce")
    frame = frame.dropna(subset=["subject_id", "target", "horizon_hours", "truth", BASELINE]).copy()
    source_coverage = {method: int(frame[method].notna().sum()) for method in methods}
    comparable = frame.dropna(subset=methods).copy()
    if comparable.empty:
        raise ValueError("No rows contain every candidate prediction under one common label contract")
    comparable["cell"] = _group_key(comparable)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    patient_reports = [
        _evaluate_split(comparable, "subject_id", seed, methods, args.min_subjects, args.bootstrap_samples)
        for seed in seeds
    ]
    hospital_report = None
    if "hospitalid" in comparable and comparable["hospitalid"].nunique() >= 3:
        hospital_report = _evaluate_split(comparable, "hospitalid", 9001, methods, args.min_subjects, args.bootstrap_samples)
    patient_gate = {method: _promotion_gate(patient_reports, method) for method in ("hard_router", "convex_stack")}
    hospital_gate = {
        method: (hospital_report["methods"][method] if hospital_report is not None else None)
        for method in ("hard_router", "convex_stack", "oracle_upper_bound_leaky")
    }
    output = {
        "schema": "candidate_library_complementarity_audit.v1",
        "purpose": "diagnose candidate complementarity; not a promotion path",
        "input": str(args.predictions.resolve()),
        "source_prediction_provenance": args.source_provenance,
        "common_label_contract": {
            "rows_before_common_intersection": int(len(frame)),
            "rows_with_all_candidates": int(len(comparable)),
            "subjects": int(comparable["subject_id"].nunique()),
            "hospitals": int(comparable["hospitalid"].nunique()) if "hospitalid" in comparable else None,
            "target_horizon_cells": int(comparable["cell"].nunique()),
            "candidate_row_coverage_before_intersection": source_coverage,
        },
        "candidates": [BASELINE, *methods],
        "patient_heldout": patient_reports,
        "patient_gate": patient_gate,
        "hospital_heldout": hospital_report,
        "hospital_gate": hospital_gate,
        "interpretation": {
            "oracle": "Leaky per-row best candidate; it is only an upper bound for available complementarity.",
            "decision_rule": (
                "If oracle is not a robust significant win, candidates share their errors and no router can create missing signal. "
                "If oracle wins but nested hard/convex routing fails, the remaining problem is selection, calibration, or regime conditioning."
            ),
            "promotion_boundary": (
                "This audit never promotes a model. Promotion needs cross-fitted source predictions generated inside each outer split, "
                "plus the existing patient, hospital, and conformal gates."
            ),
        },
        "promotion_status": "diagnostic_only",
        "clinical_promotion_allowed": False,
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "common_rows": len(comparable),
        "candidates": [BASELINE, *methods],
        "patient_gate": patient_gate,
        "hospital_stack": hospital_gate["convex_stack"],
    }, indent=2))


if __name__ == "__main__":
    main()
