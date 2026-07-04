"""Aggregate same-time nowcasting audit for sparse physiology variables.

Nowcasting is not future forecasting.  It estimates a currently unmeasured
target from other measurements available at the same anchor time.  The audit is
therefore deliberately stricter than a plain fit:

* no future ``*_tp`` columns;
* no future action-window ``act_*`` columns;
* no active flags, because they can be derived from target values;
* patient-heldout and hospital-heldout gates;
* a capacity-matched placebo model trained with permuted target values.

Only aggregate metrics are written.  Row-level cohorts stay local-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_aki_transition_extract import TARGET_VARS as AKI_TARGETS
from eicu_body_system_configs import BODY_SYSTEM_CONFIGS, get_body_system_config
from eicu_respiratory_transition_extract import TARGET_VARS as RESPIRATORY_TARGETS
from eicu_sepsis_target_router import _bootstrap_ci, _group_folds, split_subjects
from eicu_sepsis_transition_extract import TARGET_VARS as SEPSIS_TARGETS
from osler_jepa.state_completion import LEAKAGE_SIBLING_GROUPS


DEFAULT_SEEDS = (7, 11, 19, 23, 37, 53, 71)
DEFAULT_TASKS = (
    ("sepsis", Path("eicu_sepsis_transitions_6h.parquet")),
    ("aki", Path("eicu_aki_transitions_6h.parquet")),
    ("respiratory", Path("eicu_respiratory_transitions_6h.parquet")),
    *(
        (module, Path(f"eicu_{module}_transitions_6h.parquet"))
        for module in BODY_SYSTEM_CONFIGS
    ),
)

LEAKAGE_GROUPS = LEAKAGE_SIBLING_GROUPS


def _round(value, digits: int = 6):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def module_targets(module: str) -> tuple[str, ...]:
    if module == "sepsis":
        return tuple(SEPSIS_TARGETS)
    if module == "aki":
        return tuple(AKI_TARGETS)
    if module == "respiratory":
        return tuple(RESPIRATORY_TARGETS)
    if module in BODY_SYSTEM_CONFIGS:
        return tuple(get_body_system_config(module).targets)
    raise ValueError(f"unsupported module: {module}")


def active_column(module: str) -> str | None:
    if module == "sepsis":
        return "active_sepsis"
    if module == "aki":
        return "active_aki"
    if module == "respiratory":
        return "respiratory_active_t"
    if module in BODY_SYSTEM_CONFIGS:
        return f"{module}_active_t"
    return None


def target_column(target: str) -> str:
    return f"{target}_t"


def target_age_column(target: str) -> str:
    return f"{target}_age_hr"


def leakage_sibling_targets(target: str) -> set[str]:
    siblings: set[str] = set()
    for group in LEAKAGE_GROUPS:
        if target in group:
            siblings.update(group)
    siblings.discard(target)
    return siblings


def is_leakage_sibling_column(column: str, target: str) -> bool:
    siblings = leakage_sibling_targets(target)
    if not siblings:
        return False
    for sibling in siblings:
        if column == target_column(sibling) or column == target_age_column(sibling):
            return True
    return False


def is_future_column(column: str) -> bool:
    if "_tp" not in column:
        return False
    suffix = column.rsplit("_tp", 1)[-1]
    return suffix.replace(".", "", 1).isdigit()


def nowcast_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded_exact = {
        "stay_id",
        "subject_id",
        "hospitalid",
        "onset",
        "onset_criteria",
        "t",
        "t_plus",
        "unittype",
        "died_after_window",
        "hrs_to_death_from_cut",
        target_column(target),
        target_age_column(target),
    }
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded_exact:
            continue
        if is_leakage_sibling_column(column, target):
            continue
        if is_future_column(column):
            continue
        if column.startswith("act_"):
            continue
        if column.endswith("_active_t") or column.startswith("active_"):
            continue
        if column.endswith("_sources") or column.endswith("_kinds"):
            continue
        if column.endswith("_t") or column.endswith("_age_hr") or column.startswith("hist_"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                columns.append(column)
            continue
        if column == "hours_since_onset":
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                columns.append(column)
    return sorted(set(columns))


def observed_target_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    column = target_column(target)
    if column not in frame:
        return frame.iloc[0:0].copy()
    selected = frame[pd.to_numeric(frame[column], errors="coerce").notna()].copy()
    selected["truth"] = pd.to_numeric(selected[column], errors="coerce").astype("float64")
    return selected[np.isfinite(selected["truth"])].copy()


def prepare_features(train: pd.DataFrame, predict: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.zeros((len(train), 0), dtype=np.float64), np.zeros((len(predict), 0), dtype=np.float64)
    train_x = train[columns].apply(pd.to_numeric, errors="coerce")
    predict_x = predict[columns].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    predict_x = predict_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return (
        ((train_x - mean) / std).to_numpy(dtype=np.float64),
        ((predict_x - mean) / std).to_numpy(dtype=np.float64),
    )


def fit_median(train: pd.DataFrame, predict: pd.DataFrame) -> np.ndarray:
    if train.empty:
        return np.full(len(predict), np.nan, dtype=np.float64)
    value = float(np.nanmedian(train["truth"].to_numpy(dtype=np.float64)))
    return np.full(len(predict), value, dtype=np.float64)


def fit_ridge_nowcast(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    features: list[str],
    alpha: float,
    seed: int,
    placebo: bool = False,
) -> np.ndarray:
    if train.empty or len(train) < max(20, len(features) + 2):
        return np.full(len(predict), np.nan, dtype=np.float64)
    x_train, x_predict = prepare_features(train, predict, features)
    y = train["truth"].to_numpy(dtype=np.float64).copy()
    if placebo:
        rng = np.random.default_rng(int(seed))
        y = rng.permutation(y)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_predict = np.c_[np.ones(len(x_predict)), x_predict]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return x_predict @ beta


def clustered_delta(
    rows: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="float64")
    truth = rows["truth"].to_numpy(dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(baseline)
    if not finite.any():
        return pd.Series(dtype="float64")
    temp = pd.DataFrame({
        "subject_id": rows.loc[finite, "subject_id"].to_numpy(dtype=object),
        "delta": np.abs(candidate[finite] - truth[finite]) - np.abs(baseline[finite] - truth[finite]),
    })
    return temp.groupby("subject_id")["delta"].mean()


def delta_summary(rows: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    deltas = clustered_delta(rows, candidate, baseline)
    if deltas.empty:
        return {
            "subjects": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_baseline": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=samples)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_baseline": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def mae(values: np.ndarray, truth: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(truth)
    if not finite.any():
        return None
    return float(np.abs(values[finite] - truth[finite]).mean())


def evaluate_predictions(rows: pd.DataFrame, real: np.ndarray, baseline: np.ndarray, placebo: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(rows)),
        "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
        "mae": {
            "median_baseline": _round(mae(baseline, truth)),
            "placebo_ridge": _round(mae(placebo, truth)),
            "ridge_nowcast": _round(mae(real, truth)),
        },
        "delta_vs_median": delta_summary(rows, real, baseline, seed=seed, samples=samples),
        "delta_vs_placebo": delta_summary(rows, real, placebo, seed=seed + 1234, samples=samples),
    }


def oof_predictions(
    discovery: pd.DataFrame,
    target: str,
    features: list[str],
    seed: int,
    inner_folds: int,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    baseline = np.full(len(discovery), np.nan, dtype=np.float64)
    real = np.full(len(discovery), np.nan, dtype=np.float64)
    placebo = np.full(len(discovery), np.nan, dtype=np.float64)
    folds = _group_folds(discovery["subject_id"].to_numpy(dtype=object), seed=seed + 17, folds=inner_folds)
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        train = discovery.iloc[train_idx]
        val = discovery.iloc[val_idx]
        baseline[val_idx] = fit_median(train, val)
        real[val_idx] = fit_ridge_nowcast(train, val, features, ridge_alpha, seed=seed + 101 * fold_index)
        placebo[val_idx] = fit_ridge_nowcast(
            train,
            val,
            features,
            ridge_alpha,
            seed=seed + 10_000 + 101 * fold_index,
            placebo=True,
        )
    return {"median_baseline": baseline, "ridge_nowcast": real, "placebo_ridge": placebo}


def fit_predictions(
    discovery: pd.DataFrame,
    heldout: pd.DataFrame,
    features: list[str],
    seed: int,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    return {
        "median_baseline": fit_median(discovery, heldout),
        "ridge_nowcast": fit_ridge_nowcast(discovery, heldout, features, ridge_alpha, seed=seed + 20_000),
        "placebo_ridge": fit_ridge_nowcast(
            discovery,
            heldout,
            features,
            ridge_alpha,
            seed=seed + 30_000,
            placebo=True,
        ),
    }


def scope_rows(rows: pd.DataFrame, module: str, active_only: bool) -> pd.DataFrame:
    if not active_only:
        return rows
    column = active_column(module)
    if column is None or column not in rows:
        return rows.iloc[0:0].copy()
    return rows[rows[column].fillna(False).astype(bool)].copy()


def audit_split(
    frame: pd.DataFrame,
    module: str,
    discovery_groups: set[object],
    heldout_groups: set[object],
    seed: int,
    min_pairs: int,
    min_subjects: int,
    inner_folds: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    group_column: str = "subject_id",
) -> dict[str, object]:
    selected: dict[str, str] = {}
    target_reports: dict[str, object] = {}
    for target in module_targets(module):
        rows = observed_target_rows(frame, target)
        if rows.empty or group_column not in rows:
            selected[target] = "missing"
            target_reports[target] = {"status": "missing_target_or_group_column"}
            continue
        discovery = rows[rows[group_column].isin(discovery_groups)].copy()
        heldout = rows[rows[group_column].isin(heldout_groups)].copy()
        features = nowcast_feature_columns(frame, target)
        supported = (
            len(discovery) >= int(min_pairs)
            and discovery["subject_id"].nunique() >= int(min_subjects)
            and len(features) > 0
        )
        if not supported:
            selected[target] = "missing"
            target_reports[target] = {
                "status": "insufficient_support",
                "discovery_rows": int(len(discovery)),
                "discovery_subjects": int(discovery["subject_id"].nunique()) if len(discovery) else 0,
                "feature_count": int(len(features)),
            }
            continue
        oof = oof_predictions(discovery, target, features, seed, inner_folds, ridge_alpha)
        discovery_eval = evaluate_predictions(
            discovery,
            oof["ridge_nowcast"],
            oof["median_baseline"],
            oof["placebo_ridge"],
            seed=seed + 40_000,
            samples=bootstrap_samples,
        )
        passes_discovery = bool(
            discovery_eval["delta_vs_median"]["significant"]
            and discovery_eval["delta_vs_placebo"]["significant"]
        )
        selected[target] = "ridge_nowcast" if passes_discovery else "missing"
        heldout_pred = fit_predictions(discovery, heldout, features, seed, ridge_alpha)
        all_eval = evaluate_predictions(
            heldout,
            heldout_pred["ridge_nowcast"],
            heldout_pred["median_baseline"],
            heldout_pred["placebo_ridge"],
            seed=seed + 50_000,
            samples=bootstrap_samples,
        )
        active = scope_rows(heldout, module, active_only=True)
        if len(active):
            active_index = heldout.index.get_indexer(active.index)
            active_eval = evaluate_predictions(
                active,
                heldout_pred["ridge_nowcast"][active_index],
                heldout_pred["median_baseline"][active_index],
                heldout_pred["placebo_ridge"][active_index],
                seed=seed + 60_000,
                samples=bootstrap_samples,
            )
        else:
            active_eval = {"rows": 0, "subjects": 0}
        target_reports[target] = {
            "status": "evaluated",
            "selected_method": selected[target],
            "feature_count": int(len(features)),
            "discovery": discovery_eval,
            "heldout_all_windows": all_eval,
            "heldout_active_only": active_eval,
        }
    return {
        "seed": int(seed),
        "group_column": group_column,
        "selected_methods": selected,
        "targets": target_reports,
    }


def split_summary(reports: list[dict[str, object]], module: str) -> dict[str, object]:
    targets = module_targets(module)
    output: dict[str, object] = {}
    for target in targets:
        selected_count = sum(
            report["selected_methods"].get(target) == "ridge_nowcast"
            for report in reports
        )
        all_median = 0
        all_placebo = 0
        active_median = 0
        active_placebo = 0
        for report in reports:
            payload = report["targets"].get(target, {})
            all_scope = payload.get("heldout_all_windows", {})
            active_scope = payload.get("heldout_active_only", {})
            all_median += int(bool((all_scope.get("delta_vs_median") or {}).get("significant")))
            all_placebo += int(bool((all_scope.get("delta_vs_placebo") or {}).get("significant")))
            active_median += int(bool((active_scope.get("delta_vs_median") or {}).get("significant")))
            active_placebo += int(bool((active_scope.get("delta_vs_placebo") or {}).get("significant")))
        output[target] = {
            "selected_nowcast_splits": int(selected_count),
            "heldout_all_beats_median_splits": int(all_median),
            "heldout_all_beats_placebo_splits": int(all_placebo),
            "heldout_active_beats_median_splits": int(active_median),
            "heldout_active_beats_placebo_splits": int(active_placebo),
        }
    return output


def robust_nowcast_summary(
    random_summary: dict[str, object],
    hospital_report: dict[str, object] | None,
    split_count: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    hospital_targets = (hospital_report or {}).get("targets", {}) if hospital_report else {}
    for target, row in random_summary.items():
        hospital_payload = hospital_targets.get(target, {})
        hospital_all = hospital_payload.get("heldout_all_windows", {})
        hospital_pass = bool(
            (hospital_all.get("delta_vs_median") or {}).get("significant")
            and (hospital_all.get("delta_vs_placebo") or {}).get("significant")
            and hospital_payload.get("selected_method") == "ridge_nowcast"
        )
        validated = bool(
            row["selected_nowcast_splits"] == split_count
            and row["heldout_all_beats_median_splits"] == split_count
            and row["heldout_all_beats_placebo_splits"] == split_count
            and hospital_pass
        )
        output[target] = {
            **row,
            "hospital_all_gate_passed": hospital_pass,
            "nowcast_validated": validated,
        }
    return output


def audit_task(
    module: str,
    cohort: Path,
    seeds: tuple[int, ...],
    discovery_fraction: float,
    inner_folds: int,
    ridge_alpha: float,
    min_pairs: int,
    min_subjects: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    random_runs = []
    for seed in seeds:
        discovery_groups, heldout_groups = split_subjects(frame, seed, discovery_fraction)
        random_runs.append(audit_split(
            frame,
            module,
            discovery_groups,
            heldout_groups,
            seed,
            min_pairs,
            min_subjects,
            inner_folds,
            ridge_alpha,
            bootstrap_samples,
        ))
    hospital_report = None
    if "hospitalid" in frame and frame["hospitalid"].nunique() >= 3:
        discovery_groups, heldout_groups = split_subjects(
            frame,
            9001,
            discovery_fraction,
            group_column="hospitalid",
        )
        hospital_report = audit_split(
            frame,
            module,
            discovery_groups,
            heldout_groups,
            9001,
            min_pairs,
            min_subjects,
            inner_folds,
            ridge_alpha,
            bootstrap_samples,
            group_column="hospitalid",
        )
    summary = split_summary(random_runs, module)
    robust = robust_nowcast_summary(summary, hospital_report, split_count=len(seeds))
    return {
        "module": module,
        "cohort": cohort.name,
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "targets": list(module_targets(module)),
        "random_patient_splits": {
            "seeds": list(seeds),
            "summary": summary,
            "runs": random_runs,
        },
        "hospital_holdout": hospital_report or {"available": False},
        "validated_nowcast_targets": [
            target for target, payload in robust.items() if payload["nowcast_validated"]
        ],
        "robust_summary": robust,
    }


def parse_tasks(raw: str) -> tuple[tuple[str, Path], ...]:
    if not raw.strip():
        return DEFAULT_TASKS
    tasks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        module, cohort = item.split(":", 1)
        tasks.append((module, Path(cohort)))
    return tuple(tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated module:cohort entries. Defaults to sepsis, AKI, respiratory, and body-system 6h cohorts.",
    )
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("whole_body_nowcasting_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    tasks = parse_tasks(args.tasks)
    reports = [
        audit_task(
            module=module,
            cohort=cohort,
            seeds=seeds,
            discovery_fraction=args.discovery_fraction,
            inner_folds=args.inner_folds,
            ridge_alpha=args.ridge_alpha,
            min_pairs=args.min_pairs,
            min_subjects=args.min_subjects,
            bootstrap_samples=args.bootstrap_samples,
        )
        for module, cohort in tasks
    ]
    output = {
        "artifact": "whole-body same-time nowcasting audit",
        "nowcasting_definition": "estimate currently unmeasured target_t from other same-time and historical features",
        "tasks": reports,
        "gate": {
            "patient_split_seeds": list(seeds),
            "hospital_holdout_required": True,
            "must_beat": ["discovery_median_baseline", "capacity_matched_placebo_ridge"],
            "selection_scope": "discovery all-windows out-of-fold",
            "validation_scope": "heldout all-windows plus hospital-heldout all-windows",
            "leakage_sibling_groups": [sorted(group) for group in LEAKAGE_GROUPS],
            "leakage_policy": (
                "When target is in a deterministic or near-deterministic sibling group, "
                "same-group target_t and age features are excluded from ridge nowcast. "
                "Those completions belong in derived_formula or same_group_calibrated "
                "layers, not cross-system nowcast."
            ),
        },
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "same_time_imputation_only": True,
            "future_prediction_claim_allowed": False,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "tasks": [f"{module}:{cohort.name}" for module, cohort in tasks],
        "causal_claim_allowed": output["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
