"""Target-specific factual audits for cells that resist point forecasting.

This module deliberately changes the *question*, not the test set:

* oxygen saturation is also evaluated as a future hypoxemia event;
* creatinine is evaluated with a change/no-change hurdle before delta size.

All inputs are measurement-pure and available at or before the anchor.  Every
model is fit on disjoint patients and compared with both persistence and a
matched simpler baseline.  The output is research-only and cannot authorize a
causal or clinical claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from teacher_anchored_joint_jepa_audit import _crossfit_normalized_conformal

from whole_body_joint_gate import (
    _split_by_column,
    _split_forward_time,
    _split_known_groups,
)


PATIENT_BOOTSTRAP_SAMPLES = 2000
PATIENT_BOOTSTRAP_MIN_SUBJECTS = 30
DEFAULT_HYPOXEMIA_THRESHOLD = 92.0
DEFAULT_CREATININE_CHANGE = 0.1
TARGET_CHANGE_THRESHOLDS = {
    "creatinine": 0.1,
    "platelets": 10.0,
    "hemoglobin": 0.5,
    "hematocrit": 1.5,
    "wbc": 1.0,
    "calcium": 0.2,
    "magnesium": 0.2,
    "albumin": 0.2,
    "total_protein": 0.3,
    "potassium": 0.2,
    "temperature": 0.3,
    "sodium": 1.0,
    "chloride": 1.0,
    "bicarbonate": 1.0,
    "anion_gap": 1.0,
    "bun": 2.0,
    "phosphate": 0.2,
}
SIBLING_EXCLUSIONS = {
    "hemoglobin": ("hematocrit",),
    "hematocrit": ("hemoglobin",),
}


@dataclass
class MatrixTransform:
    columns: list[str]
    medians: np.ndarray
    scales: np.ndarray

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.reindex(columns=self.columns).to_numpy(dtype=np.float64)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians)
        return ((matrix - self.medians) / self.scales).astype(np.float32)


def _safe_float(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(
        frame.get(column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a causal state/history view without provenance or future fields."""

    value_columns = sorted(
        column
        for column in frame.columns
        if column.endswith("_t")
        and not column.startswith(("future_", "_source_", "_"))
        and column not in {"t"}
    )
    age_columns = sorted(column for column in frame.columns if column.endswith("_age_hr"))
    observed_columns = sorted(
        column for column in frame.columns if column.startswith("_observed_")
    )
    treatment_columns = sorted(
        column for column in frame.columns if column.startswith("hist_")
    )
    selected = value_columns + age_columns + observed_columns + treatment_columns
    output = frame[selected].apply(pd.to_numeric, errors="coerce").copy()

    # The previous observation and its per-hour slope carry patient-specific
    # trajectory information while remaining strictly before the anchor.
    order = np.argsort(
        pd.to_datetime(frame["anchor_time"], errors="coerce").astype("int64").to_numpy(),
        kind="stable",
    )
    ordered = frame.iloc[order]
    ordered_values = ordered[value_columns].apply(pd.to_numeric, errors="coerce")
    previous = ordered_values.groupby(ordered["subject_id"].astype(str), sort=False).shift(1)
    time_hours = (
        pd.to_datetime(ordered["anchor_time"], errors="coerce").astype("int64") / 3.6e12
    )
    delta_hours = time_hours.groupby(ordered["subject_id"].astype(str), sort=False).diff()
    slopes = ordered_values.subtract(previous).div(delta_hours.replace(0.0, np.nan), axis=0)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    previous = previous.iloc[inverse]
    slopes = slopes.iloc[inverse]
    previous.columns = [f"lag1__{column}" for column in value_columns]
    slopes.columns = [f"slope1h__{column}" for column in value_columns]
    output = pd.concat(
        [output.reset_index(drop=True), previous.reset_index(drop=True), slopes.reset_index(drop=True)],
        axis=1,
    )
    output["hours_since_onset"] = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    return output


def _exclude_sibling_features(features: pd.DataFrame, target: str) -> pd.DataFrame:
    siblings = SIBLING_EXCLUSIONS.get(target, ())
    if not siblings:
        return features
    blocked = [
        column
        for column in features.columns
        if any(sibling in column for sibling in siblings)
    ]
    return features.drop(columns=blocked)


def _fit_transform(features: pd.DataFrame, fit_rows: np.ndarray) -> MatrixTransform:
    train = features.iloc[np.asarray(fit_rows, dtype=np.int64)]
    usable = [
        column
        for column in train.columns
        if np.isfinite(pd.to_numeric(train[column], errors="coerce")).sum() >= 30
    ]
    matrix = train[usable].to_numpy(dtype=np.float64)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    scales = np.nanstd(filled, axis=0)
    keep = scales > 1e-7
    return MatrixTransform(
        columns=[column for column, include in zip(usable, keep) if include],
        medians=medians[keep],
        scales=scales[keep],
    )


def _split_fit_calibration(frame: pd.DataFrame, train_rows: np.ndarray, seed: int):
    local = frame.iloc[train_rows].reset_index(drop=True)
    fit_local, calibration_local = _split_by_column(local, "subject_id", seed, 0.20)
    return train_rows[fit_local], train_rows[calibration_local]


def _patient_bootstrap(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_loss: np.ndarray,
    comparator_loss: np.ndarray,
    *,
    seed: int,
) -> dict:
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    deltas = np.asarray(
        [
            np.mean(candidate_loss[subjects == subject] - comparator_loss[subjects == subject])
            for subject in np.unique(subjects)
        ],
        dtype=np.float64,
    )
    point = float(np.mean(deltas)) if len(deltas) else None
    if len(deltas) < PATIENT_BOOTSTRAP_MIN_SUBJECTS:
        return {
            "n_subjects": int(len(deltas)),
            "patient_equal_delta": point,
            "bootstrap_low": None,
            "bootstrap_high": None,
            "pass": False,
        }
    rng = np.random.default_rng(seed)
    sampled = deltas[
        rng.integers(0, len(deltas), size=(PATIENT_BOOTSTRAP_SAMPLES, len(deltas)))
    ].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "n_subjects": int(len(deltas)),
        "patient_equal_delta": point,
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
        "pass": bool(high < 0.0),
    }


def _finite_conformal_width(residual: np.ndarray, coverage: float = 0.90) -> float:
    residual = np.sort(np.asarray(residual, dtype=np.float64))
    if not len(residual):
        return float("nan")
    rank = int(np.ceil((len(residual) + 1) * float(coverage))) - 1
    return float(residual[min(max(rank, 0), len(residual) - 1)])


def _adaptive_conformal_width(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Choose a finite-sample quantile using patient-disjoint calibration CV."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds = [set(part.tolist()) for part in np.array_split(unique, min(5, len(unique)))]
    candidates = np.linspace(0.80, 0.94, 15)
    scores = []
    for nominal in candidates:
        covered = []
        for heldout in folds:
            test = np.asarray([subject in heldout for subject in subjects], dtype=bool)
            fit = ~test
            if fit.sum() < 30 or test.sum() < 10:
                continue
            width = _finite_conformal_width(residual[fit], nominal)
            covered.extend((residual[test] <= width).tolist())
        empirical = float(np.mean(covered)) if covered else float("nan")
        scores.append((abs(empirical - 0.90), nominal, empirical))
    _, selected, crossfit_coverage = min(
        (item for item in scores if np.isfinite(item[2])),
        key=lambda item: (item[0], abs(item[1] - 0.90)),
    )
    return (
        _finite_conformal_width(residual, selected),
        float(selected),
        float(crossfit_coverage),
    )


def _adaptive_normalized_conformal_widths(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select normalized conformal nominal coverage on held-out patients."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds = [set(part.tolist()) for part in np.array_split(unique, min(5, len(unique)))]
    scored = []
    for nominal in np.linspace(0.80, 0.94, 15):
        covered = []
        for fold_index, heldout in enumerate(folds):
            check = np.asarray([subject in heldout for subject in subjects], dtype=bool)
            fit = ~check
            if fit.sum() < 90 or check.sum() < 20:
                continue
            widths, _ = _crossfit_normalized_conformal(
                frame,
                rows[fit],
                residual[fit],
                calibration_features[fit],
                calibration_features[check],
                seed=seed + fold_index * 101,
                coverage=float(nominal),
                folds=4,
            )
            covered.extend((residual[check] <= widths).tolist())
        empirical = float(np.mean(covered)) if covered else float("nan")
        scored.append((abs(empirical - 0.90), float(nominal), empirical))
    valid_scored = [item for item in scored if np.isfinite(item[2])]
    if valid_scored:
        _, selected, crossfit_coverage = min(
            valid_scored,
            key=lambda item: (item[0], abs(item[1] - 0.90)),
        )
    else:
        selected, crossfit_coverage = 0.90, None
    widths, report = _crossfit_normalized_conformal(
        frame,
        rows,
        residual,
        calibration_features,
        test_features,
        seed=seed + 997,
        coverage=selected,
    )
    report.update(
        {
            "selected_nominal_coverage": selected,
            "calibration_outer_crossfit_coverage": crossfit_coverage,
            "selection_test_labels_used": False,
        }
    )
    return widths, report


def _regression_metrics(
    frame: pd.DataFrame,
    rows: np.ndarray,
    observed: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    ridge: np.ndarray,
    calibration_rows: np.ndarray,
    calibration_observed: np.ndarray,
    calibration_candidate: np.ndarray,
    *,
    seed: int,
    target: str,
) -> dict:
    candidate_error = np.abs(candidate - observed)
    persistence_error = np.abs(persistence - observed)
    ridge_error = np.abs(ridge - observed)
    calibration_residual = np.abs(calibration_candidate - calibration_observed)
    calibration_current = _safe_float(frame, f"{target}_t")[calibration_rows]
    test_current = persistence
    calibration_age = np.log1p(
        np.clip(_safe_float(frame, f"{target}_age_hr")[calibration_rows], 0.0, 168.0)
    )
    test_age = np.log1p(
        np.clip(_safe_float(frame, f"{target}_age_hr")[rows], 0.0, 168.0)
    )
    calibration_onset = np.log1p(
        np.clip(_safe_float(frame, "hours_since_onset")[calibration_rows], 0.0, 720.0)
    )
    test_onset = np.log1p(
        np.clip(_safe_float(frame, "hours_since_onset")[rows], 0.0, 720.0)
    )
    calibration_scale_features = np.column_stack(
        [
            calibration_current,
            calibration_candidate,
            np.abs(calibration_candidate - calibration_current),
            np.nan_to_num(calibration_age, nan=np.log1p(168.0)),
            np.nan_to_num(calibration_onset, nan=0.0),
        ]
    )
    test_scale_features = np.column_stack(
        [
            test_current,
            candidate,
            np.abs(candidate - test_current),
            np.nan_to_num(test_age, nan=np.log1p(168.0)),
            np.nan_to_num(test_onset, nan=0.0),
        ]
    )
    widths, normalized_report = _adaptive_normalized_conformal_widths(
        frame,
        calibration_rows,
        calibration_residual,
        calibration_scale_features,
        test_scale_features,
        seed=seed + 707,
    )
    coverage = float(np.mean(np.abs(candidate - observed) <= widths))
    versus_persistence = _patient_bootstrap(
        frame, rows, candidate_error, persistence_error, seed=seed
    )
    versus_ridge = _patient_bootstrap(
        frame, rows, candidate_error, ridge_error, seed=seed + 1
    )
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "candidate_mae": float(np.mean(candidate_error)),
        "persistence_mae": float(np.mean(persistence_error)),
        "plain_ridge_mae": float(np.mean(ridge_error)),
        "versus_persistence": versus_persistence,
        "versus_plain_ridge": versus_ridge,
        "conformal": {
            "nominal": 0.90,
            "width": float(np.median(widths)),
            "coverage": coverage,
            "method": normalized_report,
            "pass": bool(coverage is not None and 0.87 <= coverage <= 0.93),
        },
        "pass": bool(
            versus_persistence["pass"]
            and versus_ridge["pass"]
            and coverage is not None
            and 0.87 <= coverage <= 0.93
        ),
    }


def _classification_metrics(
    frame: pd.DataFrame,
    rows: np.ndarray,
    observed: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    prevalence: np.ndarray,
    *,
    seed: int,
) -> dict:
    candidate_loss = (candidate - observed) ** 2
    persistence_loss = (persistence - observed) ** 2
    prevalence_loss = (prevalence - observed) ** 2
    versus_persistence = _patient_bootstrap(
        frame, rows, candidate_loss, persistence_loss, seed=seed
    )
    versus_prevalence = _patient_bootstrap(
        frame, rows, candidate_loss, prevalence_loss, seed=seed + 1
    )
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "event_prevalence": float(np.mean(observed)),
        "candidate_brier": float(np.mean(candidate_loss)),
        "persistence_brier": float(np.mean(persistence_loss)),
        "prevalence_brier": float(np.mean(prevalence_loss)),
        "versus_persistence": versus_persistence,
        "versus_prevalence": versus_prevalence,
        "pass": bool(versus_persistence["pass"] and versus_prevalence["pass"]),
    }


def _fit_hurdle(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    target: str,
    change_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    features = _exclude_sibling_features(features, target)
    current = _safe_float(frame, f"{target}_t")
    future = _safe_float(frame, f"future_{target}")
    valid = np.isfinite(current) & np.isfinite(future)
    fit_rows = fit_rows[valid[fit_rows]]
    calibration_rows = calibration_rows[valid[calibration_rows]]
    test_rows = test_rows[valid[test_rows]]
    transform = _fit_transform(features, fit_rows)
    x_fit = transform.apply(features.iloc[fit_rows])
    x_cal = transform.apply(features.iloc[calibration_rows])
    x_test = transform.apply(features.iloc[test_rows])
    delta_fit = future[fit_rows] - current[fit_rows]
    changed_fit = np.abs(delta_fit) >= float(change_threshold)
    classifier = LogisticRegression(C=0.25, max_iter=300, solver="lbfgs")
    classifier.fit(x_fit, changed_fit.astype(np.int8))
    probability_cal = classifier.predict_proba(x_cal)[:, 1]
    probability_test = classifier.predict_proba(x_test)[:, 1]

    changed_rows = np.flatnonzero(changed_fit)
    delta_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=50,
        l2_regularization=10.0,
        random_state=seed,
    )
    delta_model.fit(x_fit[changed_rows], delta_fit[changed_rows])
    delta_cal = delta_model.predict(x_cal)
    delta_test = delta_model.predict(x_test)

    robust_all = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=50,
        l2_regularization=10.0,
        random_state=seed + 17,
    )
    robust_all.fit(x_fit, delta_fit)
    robust_cal = robust_all.predict(x_cal)
    robust_test = robust_all.predict(x_test)

    # Discovery-only clipping prevents a handful of extreme deltas from
    # turning a median-oriented factual forecast into an outlier generator.
    delta_limit = float(np.quantile(np.abs(delta_fit), 0.99))
    delta_limit = max(delta_limit, float(change_threshold))
    delta_cal = np.clip(delta_cal, -delta_limit, delta_limit)
    delta_test = np.clip(delta_test, -delta_limit, delta_limit)
    robust_cal = np.clip(robust_cal, -delta_limit, delta_limit)
    robust_test = np.clip(robust_test, -delta_limit, delta_limit)

    # Plain ridge is the matched non-hurdle comparator.
    plain = Ridge(alpha=25.0)
    plain.fit(x_fit, delta_fit)
    plain_cal = current[calibration_rows] + plain.predict(x_cal)
    plain_test = current[test_rows] + plain.predict(x_test)

    candidates = []
    for shrink in (0.25, 0.5, 0.75, 1.0):
        candidates.append(
            (
                f"robust_median_delta_{shrink:g}",
                current[calibration_rows] + shrink * robust_cal,
                current[test_rows] + shrink * robust_test,
            )
        )
    for shrink in (0.25, 0.5, 0.75, 1.0):
        candidates.append(
            (
                f"soft_p_x_delta_{shrink:g}",
                current[calibration_rows] + shrink * probability_cal * delta_cal,
                current[test_rows] + shrink * probability_test * delta_test,
            )
        )
        for threshold in (0.35, 0.5, 0.65):
            candidates.append(
                (
                    f"hard_p{threshold:g}_x_delta_{shrink:g}",
                    current[calibration_rows]
                    + shrink * (probability_cal >= threshold) * delta_cal,
                    current[test_rows]
                    + shrink * (probability_test >= threshold) * delta_test,
                )
            )
    selected_name, selected_cal, selected_test = min(
        candidates,
        key=lambda item: float(np.mean(np.abs(item[1] - future[calibration_rows]))),
    )
    return (
        test_rows,
        selected_test,
        plain_test,
        calibration_rows,
        selected_cal,
        selected_name,
    )


def _fit_hypoxemia(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    current = _safe_float(frame, "o2sat_t")
    future = _safe_float(frame, "future_o2sat")
    valid = np.isfinite(current) & np.isfinite(future)
    fit_rows = fit_rows[valid[fit_rows]]
    calibration_rows = calibration_rows[valid[calibration_rows]]
    test_rows = test_rows[valid[test_rows]]
    transform = _fit_transform(features, fit_rows)
    x_fit = transform.apply(features.iloc[fit_rows])
    x_cal = transform.apply(features.iloc[calibration_rows])
    x_test = transform.apply(features.iloc[test_rows])
    y_fit = (future[fit_rows] <= float(threshold)).astype(np.int8)
    model = LogisticRegression(C=0.20, max_iter=300, solver="lbfgs")
    model.fit(x_fit, y_fit)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    y_cal = (future[calibration_rows] <= float(threshold)).astype(np.float64)
    fit_prevalence = float(np.mean(y_fit))
    # Select calibration shrinkage without reading the test labels. This is a
    # reliability correction, not a new physiological feature.
    shrinkage = min(
        np.linspace(0.0, 1.0, 21),
        key=lambda alpha: float(
            np.mean((fit_prevalence + alpha * (raw_cal - fit_prevalence) - y_cal) ** 2)
        ),
    )
    candidate = fit_prevalence + float(shrinkage) * (raw_test - fit_prevalence)
    persistence = (current[test_rows] <= float(threshold)).astype(np.float64)
    prevalence = np.full(len(test_rows), fit_prevalence, dtype=np.float64)
    return test_rows, candidate, persistence, prevalence, float(shrinkage)


def _split_specs(frame: pd.DataFrame, patient_seeds: tuple[int, ...]):
    specs = [
        (f"patient_seed_{seed}", *_split_by_column(frame, "subject_id", seed), seed)
        for seed in patient_seeds
    ]
    specs.extend(
        [
            ("hospital", *_split_by_column(frame, "hospitalid", 92026), 92026),
            ("careunit", *_split_known_groups(frame, "careunit", 92027), 92027),
            ("forward_time", *_split_forward_time(frame), 92028),
        ]
    )
    return specs


def _audit_horizon(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    patient_seeds: tuple[int, ...],
    hypoxemia_threshold: float,
    creatinine_change: float,
) -> dict:
    view = frame[pd.to_numeric(frame["horizon_hours"], errors="coerce") == float(horizon)].copy()
    view["hours_since_onset"] = (
        view["anchor_time"]
        - view.groupby("stay_id", sort=False)["anchor_time"].transform("min")
    ).dt.total_seconds() / 3600.0
    view.sort_values(["subject_id", "anchor_time"], inplace=True)
    view.reset_index(drop=True, inplace=True)
    features = _feature_frame(view)
    reports = {}
    for name, train_rows, test_rows, seed in _split_specs(view, patient_seeds):
        if target != "o2sat":
            change_threshold = (
                float(creatinine_change)
                if target == "creatinine"
                else float(TARGET_CHANGE_THRESHOLDS[target])
            )
            (
                rows,
                candidate,
                ridge,
                calibration_rows,
                calibration_candidate,
                selected_name,
            ) = _fit_hurdle(
                view,
                features,
                train_rows,
                test_rows,
                seed=seed,
                target=target,
                change_threshold=change_threshold,
            )
            observed = _safe_float(view, f"future_{target}")[rows]
            persistence = _safe_float(view, f"{target}_t")[rows]
            calibration_observed = _safe_float(view, f"future_{target}")[calibration_rows]
            report = _regression_metrics(
                view,
                rows,
                observed,
                candidate,
                persistence,
                ridge,
                calibration_rows,
                calibration_observed,
                calibration_candidate,
                seed=seed,
                target=target,
            )
            report["selected_hurdle_policy"] = selected_name
            report["change_threshold"] = change_threshold
        else:
            rows, candidate, persistence, prevalence, shrinkage = _fit_hypoxemia(
                view,
                features,
                train_rows,
                test_rows,
                seed=seed,
                threshold=hypoxemia_threshold,
            )
            observed = (
                _safe_float(view, "future_o2sat")[rows] <= float(hypoxemia_threshold)
            ).astype(np.float64)
            report = _classification_metrics(
                view,
                rows,
                observed,
                candidate,
                persistence,
                prevalence,
                seed=seed,
            )
            report["discovery_calibrated_shrinkage"] = shrinkage
        reports[name] = report
    patient_keys = [f"patient_seed_{seed}" for seed in patient_seeds]
    external_keys = ["hospital", "careunit", "forward_time"]
    return {
        "target": target,
        "horizon_hours": int(horizon),
        "rows": int(len(view)),
        "subjects": int(view["subject_id"].nunique()),
        "splits": reports,
        "summary": {
            "patient_passes": int(sum(reports[key]["pass"] for key in patient_keys)),
            "patient_required": int(len(patient_keys)),
            "external_passes": int(sum(reports[key]["pass"] for key in external_keys)),
            "external_required": int(len(external_keys)),
            "validated": bool(
                all(reports[key]["pass"] for key in patient_keys + external_keys)
            ),
        },
    }


def run(args) -> dict:
    frame = pd.read_parquet(args.event_examples)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], errors="coerce")
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    unsupported = sorted(
        target
        for target in targets
        if target != "o2sat" and target not in TARGET_CHANGE_THRESHOLDS
    )
    if unsupported:
        raise ValueError(f"No target-specific change threshold for: {unsupported}")
    patient_seeds = tuple(range(int(args.patient_seeds)))
    reports = {}
    for target in targets:
        for horizon in horizons:
            key = f"{target}@{horizon}h"
            print(f"auditing {key}", flush=True)
            reports[key] = _audit_horizon(
                frame,
                target=target,
                horizon=horizon,
                patient_seeds=patient_seeds,
                hypoxemia_threshold=args.hypoxemia_threshold,
                creatinine_change=args.creatinine_change,
            )
            print(json.dumps({key: reports[key]["summary"]}, indent=2), flush=True)
    output = {
        "schema": "target_specific_event_hurdle_audit.v1",
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "event_examples": str(args.event_examples.resolve()),
        "hypoxemia_definition": f"future_o2sat <= {args.hypoxemia_threshold:g}",
        "creatinine_change_definition": f"abs(future-current) >= {args.creatinine_change:g} mg/dL",
        "input_policy": (
            "anchor-time measurements, causal lag/slope history, and strictly pre-anchor treatment context; "
            "hospital/care-unit/database provenance excluded from model inputs"
        ),
        "reports": reports,
        "validated_cells": sorted(
            key for key, report in reports.items() if report["summary"]["validated"]
        ),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--targets", default="creatinine,o2sat")
    parser.add_argument("--horizons", default="1,3,6,24")
    parser.add_argument("--patient-seeds", type=int, default=7)
    parser.add_argument("--hypoxemia-threshold", type=float, default=DEFAULT_HYPOXEMIA_THRESHOLD)
    parser.add_argument("--creatinine-change", type=float, default=DEFAULT_CREATININE_CHANGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run(args)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "validated_cells": output["validated_cells"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
