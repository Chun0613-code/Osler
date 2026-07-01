"""Aggregate split-conformal coverage audit for validated eICU factual routers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eicu_body_system_target_router as body_router
import eicu_aki_target_router as aki_router
import eicu_respiratory_target_router as respiratory_router
import eicu_sepsis_target_router as sepsis_router
from eicu_body_system_configs import BODY_SYSTEM_CONFIGS, get_body_system_config


DEFAULT_SEEDS = (7, 11, 19, 23, 37, 53, 71)
DEFAULT_TASKS = (
    ("sepsis", Path("eicu_sepsis_transitions_6h.parquet"), "tp6"),
    ("aki", Path("eicu_aki_transitions_24h.parquet"), "tp24"),
    ("aki", Path("eicu_aki_transitions_48h.parquet"), "tp48"),
)


def conformal_quantile(residuals: np.ndarray, level: float) -> float | None:
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return None
    order = np.sort(residuals)
    rank = int(np.ceil((len(order) + 1) * float(level))) - 1
    rank = min(max(rank, 0), len(order) - 1)
    return float(order[rank])


def _coverage(values: np.ndarray, radius: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not np.isfinite(radius):
        return None
    return float((values <= float(radius)).mean())


def _method_values(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    if method not in rows:
        return rows.iloc[0:0].copy()
    selected = rows[np.isfinite(rows[method]) & np.isfinite(rows["truth"])].copy()
    selected["abs_residual"] = np.abs(
        selected[method].to_numpy(dtype=np.float64)
        - selected["truth"].to_numpy(dtype=np.float64)
    )
    return selected


def _evaluate_scope(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    selector: dict[str, str],
    level: float,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    targets = sorted(set(calibration["target"].dropna()) | set(test["target"].dropna()))
    for target in targets:
        method = selector.get(target, "persistence")
        cal = _method_values(calibration[calibration["target"] == target], method)
        tst = _method_values(test[test["target"] == target], method)
        radius = conformal_quantile(cal["abs_residual"].to_numpy(dtype=np.float64), level)
        coverage = _coverage(tst["abs_residual"].to_numpy(dtype=np.float64), radius if radius is not None else np.nan)
        output[target] = {
            "method": method,
            "level": float(level),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(tst)),
            "radius": round(radius, 6) if radius is not None else None,
            "mean_interval_width": round(2.0 * radius, 6) if radius is not None else None,
            "coverage": round(coverage, 6) if coverage is not None else None,
            "coverage_gate_passed": bool(
                radius is not None
                and len(cal) >= int(min_calibration_rows)
                and len(tst) >= int(min_test_rows)
                and coverage is not None
                and 0.87 <= coverage <= 0.93
            ),
        }
    return output


def _module_api(module: str):
    if module == "sepsis":
        return {
            "split_subjects": sepsis_router.split_subjects,
            "attach_sources": lambda frame, discovery, heldout, **kwargs: sepsis_router.attach_sources(
                frame, discovery, heldout, **kwargs
            ),
            "choose_selector": lambda discovery, **kwargs: sepsis_router._choose_selector(discovery, **kwargs),
            "active_scope": "active_sepsis",
            "future_suffix_arg": True,
            "hospital_scope": "hospitalid",
        }
    if module == "aki":
        return {
            "split_subjects": aki_router.split_subjects,
            "attach_sources": lambda frame, discovery, heldout, **kwargs: aki_router.attach_sources(
                frame, discovery, heldout, **kwargs
            ),
            "choose_selector": lambda discovery, **kwargs: aki_router._choose_selector(discovery, **kwargs),
            "active_scope": "active_aki",
            "future_suffix_arg": True,
            "hospital_scope": "hospitalid",
        }
    if module == "respiratory":
        return {
            "split_subjects": respiratory_router.split_subjects,
            "attach_sources": lambda frame, discovery, heldout, **kwargs: respiratory_router.attach_sources(
                frame, discovery, heldout, **kwargs
            ),
            "choose_selector": lambda discovery, **kwargs: respiratory_router._choose_selector(discovery, **kwargs),
            "active_scope": "active_respiratory",
            "future_suffix_arg": True,
            "hospital_scope": "hospitalid",
        }
    if module in BODY_SYSTEM_CONFIGS:
        config = get_body_system_config(module)
        return {
            "split_subjects": body_router.split_subjects,
            "attach_sources": lambda frame, discovery, heldout, **kwargs: body_router.attach_sources(
                frame, config, discovery, heldout, **kwargs
            ),
            "choose_selector": lambda discovery, **kwargs: body_router._choose_selector(
                discovery, config, **kwargs
            ),
            "active_scope": "active",
            "future_suffix_arg": True,
            "hospital_scope": "hospitalid",
        }
    raise ValueError(f"unsupported module: {module}")


def _fit_rows(
    frame: pd.DataFrame,
    module: str,
    discovery_groups: set[object],
    heldout_groups: set[object],
    seed: int,
    future_suffix: str,
    discovery_fraction: float,
    inner_folds: int,
    ridge_alpha: float,
    min_pairs: int,
    min_subjects: int,
    bootstrap_samples: int,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    api = _module_api(module)
    rows, _diagnostics = api["attach_sources"](
        frame,
        discovery_groups,
        heldout_groups,
        seed=seed,
        inner_folds=inner_folds,
        ridge_alpha=ridge_alpha,
        group_column=group_column,
        future_suffix=future_suffix,
    )
    group_column = group_column or ("subject_id" if "subject_id" in rows else "stay_id")
    discovery = rows[rows[group_column].isin(discovery_groups)].copy()
    heldout = rows[rows[group_column].isin(heldout_groups)].copy()
    selector, _details = api["choose_selector"](
        discovery,
        active_only=True,
        min_pairs=min_pairs,
        min_subjects=min_subjects,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 10_000,
    )
    return discovery, heldout, selector


def audit_task(
    module: str,
    cohort: Path,
    future_suffix: str,
    seeds: tuple[int, ...],
    level: float,
    discovery_fraction: float,
    inner_folds: int,
    ridge_alpha: float,
    min_pairs: int,
    min_subjects: int,
    bootstrap_samples: int,
    min_calibration_rows: int,
    min_test_rows: int,
) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    api = _module_api(module)
    random_runs = []
    for seed in seeds:
        discovery_groups, heldout_groups = api["split_subjects"](frame, seed, discovery_fraction)
        discovery, heldout, selector = _fit_rows(
            frame,
            module,
            discovery_groups,
            heldout_groups,
            seed,
            future_suffix,
            discovery_fraction,
            inner_folds,
            ridge_alpha,
            min_pairs,
            min_subjects,
            bootstrap_samples,
        )
        random_runs.append({
            "seed": int(seed),
            "selected_methods": selector,
            "active_only": _evaluate_scope(
                discovery[discovery[api["active_scope"]].astype(bool)],
                heldout[heldout[api["active_scope"]].astype(bool)],
                selector,
                level,
                min_calibration_rows,
                min_test_rows,
            ),
            "all_windows": _evaluate_scope(
                discovery,
                heldout,
                selector,
                level,
                min_calibration_rows,
                min_test_rows,
            ),
        })
    hospital = {"available": False}
    if "hospitalid" in frame and frame["hospitalid"].nunique() >= 3:
        discovery_groups, heldout_groups = api["split_subjects"](
            frame, 9001, discovery_fraction, group_column="hospitalid"
        )
        discovery, heldout, selector = _fit_rows(
            frame,
            module,
            discovery_groups,
            heldout_groups,
            9001,
            future_suffix,
            discovery_fraction,
            inner_folds,
            ridge_alpha,
            min_pairs,
            min_subjects,
            bootstrap_samples,
            group_column="hospitalid",
        )
        hospital = {
            "available": True,
            "selected_methods": selector,
            "active_only": _evaluate_scope(
                discovery[discovery[api["active_scope"]].astype(bool)],
                heldout[heldout[api["active_scope"]].astype(bool)],
                selector,
                level,
                min_calibration_rows,
                min_test_rows,
            ),
            "all_windows": _evaluate_scope(
                discovery,
                heldout,
                selector,
                level,
                min_calibration_rows,
                min_test_rows,
            ),
        }
    return {
        "module": module,
        "cohort": cohort.name,
        "future_suffix": future_suffix,
        "level": float(level),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "random_patient_splits": {
            "seeds": list(seeds),
            "runs": random_runs,
        },
        "hospital_holdout": hospital,
    }


def parse_tasks(raw: str) -> tuple[tuple[str, Path, str], ...]:
    if not raw.strip():
        return DEFAULT_TASKS
    tasks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        module, cohort, suffix = item.split(":", 2)
        tasks.append((module, Path(cohort), suffix))
    return tuple(tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated module:cohort:future_suffix entries. Defaults to sepsis 6h and AKI 24/48h.",
    )
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--min-calibration-rows", type=int, default=100)
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("whole_body_conformal_coverage_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    tasks = parse_tasks(args.tasks)
    reports = [
        audit_task(
            module=module,
            cohort=cohort,
            future_suffix=future_suffix,
            seeds=seeds,
            level=args.level,
            discovery_fraction=args.discovery_fraction,
            inner_folds=args.inner_folds,
            ridge_alpha=args.ridge_alpha,
            min_pairs=args.min_pairs,
            min_subjects=args.min_subjects,
            bootstrap_samples=args.bootstrap_samples,
            min_calibration_rows=args.min_calibration_rows,
            min_test_rows=args.min_test_rows,
        )
        for module, cohort, future_suffix in tasks
    ]
    output = {
        "artifact": "whole-body split-conformal coverage audit",
        "interval_method": "split_conformal_residual_interval_per_target_horizon_source",
        "primary_coverage_target": float(args.level),
        "accepted_coverage_range": [0.87, 0.93],
        "tasks": reports,
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "factual_prediction_only": True,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "tasks": [f"{module}:{cohort.name}:{suffix}" for module, cohort, suffix in tasks],
        "causal_claim_allowed": output["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
