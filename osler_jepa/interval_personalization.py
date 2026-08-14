"""Leakage-safe selective and online conformal interval policies.

These functions operate after point prediction.  They never update the
physiology model and only use a patient's forecast error after the forecast's
target event has actually occurred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectiveIntervalResult:
    widths: np.ndarray
    selected: np.ndarray
    scale_threshold: float
    normalized_quantile: float


@dataclass(frozen=True)
class OnlineIntervalResult:
    widths: np.ndarray
    effective_alpha: np.ndarray
    prior_resolved_count: np.ndarray
    personalized: np.ndarray
    normalized_quantile: np.ndarray


def _finite(values) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    return output[np.isfinite(output)]


def patient_weighted_quantile(
    values,
    subjects,
    *,
    coverage: float,
) -> float:
    """Return a finite-sample quantile with equal total patient weight."""

    scores = np.asarray(values, dtype=np.float64)
    patient_ids = np.asarray(subjects, dtype=str)
    finite = np.isfinite(scores)
    scores = scores[finite]
    patient_ids = patient_ids[finite]
    if not len(scores):
        raise ValueError("Conformal scores must contain finite values")
    unique, inverse, counts = np.unique(
        patient_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    order = np.argsort(scores, kind="stable")
    corrected = min(
        1.0,
        float(coverage) * (len(unique) + 1.0) / max(len(unique), 1),
    )
    threshold = corrected * float(weights.sum())
    index = int(
        np.searchsorted(np.cumsum(weights[order]), threshold, side="left")
    )
    return float(scores[order[min(index, len(order) - 1)]])


def _patient_weighted_distribution_quantile(
    values,
    subjects,
    *,
    quantile: float,
) -> float:
    """Return an ordinary patient-equalized distribution quantile."""

    scores = np.asarray(values, dtype=np.float64)
    patient_ids = np.asarray(subjects, dtype=str)
    finite = np.isfinite(scores)
    scores = scores[finite]
    patient_ids = patient_ids[finite]
    if not len(scores):
        raise ValueError("Distribution values must contain finite values")
    _, inverse, counts = np.unique(
        patient_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    order = np.argsort(scores, kind="stable")
    threshold = float(np.clip(quantile, 0.0, 1.0)) * float(weights.sum())
    index = int(
        np.searchsorted(np.cumsum(weights[order]), threshold, side="left")
    )
    return float(scores[order[min(index, len(order) - 1)]])


def selective_interval_widths(
    calibration_scores,
    calibration_scales,
    calibration_subjects,
    test_scales,
    *,
    coverage: float = 0.90,
    selection_fraction: float = 0.50,
) -> SelectiveIntervalResult:
    """Calibrate a narrower interval only for low predicted-risk rows.

    Selection depends only on a scale available at forecast time.  High-risk
    rows are marked unselected and must abstain rather than receiving a
    deceptively narrow interval.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64)
    calibration_scale = np.asarray(calibration_scales, dtype=np.float64)
    subjects = np.asarray(calibration_subjects, dtype=str)
    test_scale = np.asarray(test_scales, dtype=np.float64)
    valid = (
        np.isfinite(scores)
        & np.isfinite(calibration_scale)
        & (calibration_scale > 0.0)
    )
    if int(valid.sum()) < 30:
        raise ValueError("Selective conformal needs at least 30 calibration rows")
    fraction = float(np.clip(selection_fraction, 0.10, 0.90))
    threshold = _patient_weighted_distribution_quantile(
        calibration_scale[valid],
        subjects[valid],
        quantile=fraction,
    )
    calibration_selected = valid & (calibration_scale <= threshold)
    if int(calibration_selected.sum()) < 30:
        raise ValueError("Low-risk calibration subset is too small")
    quantile = patient_weighted_quantile(
        scores[calibration_selected],
        subjects[calibration_selected],
        coverage=coverage,
    )
    selected = (
        np.isfinite(test_scale)
        & (test_scale > 0.0)
        & (test_scale <= threshold)
    )
    return SelectiveIntervalResult(
        widths=np.maximum(test_scale, 1.0e-8) * quantile,
        selected=selected,
        scale_threshold=float(threshold),
        normalized_quantile=float(quantile),
    )


def online_adaptive_interval_widths(
    calibration_scores,
    calibration_subjects,
    test_subjects,
    anchor_times,
    future_event_times,
    test_scales,
    point_predictions,
    observed_values,
    *,
    coverage: float = 0.90,
    gamma: float = 0.01,
    min_resolved_history: int = 3,
    min_alpha: float = 0.02,
    max_alpha: float = 0.25,
) -> OnlineIntervalResult:
    """Issue causal-order adaptive conformal widths for each patient.

    A forecast outcome enters the patient's update state only when its actual
    target-event timestamp is no later than the next forecast anchor.  Multiple
    forecasts resolved by the same measurement are de-duplicated so a single
    lab draw cannot be counted repeatedly.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64)
    calibration_ids = np.asarray(calibration_subjects, dtype=str)
    subject_ids = np.asarray(test_subjects, dtype=str)
    anchors = pd.to_datetime(anchor_times, errors="coerce").to_numpy()
    event_times = pd.to_datetime(future_event_times, errors="coerce").to_numpy()
    scales = np.asarray(test_scales, dtype=np.float64)
    points = np.asarray(point_predictions, dtype=np.float64)
    observed = np.asarray(observed_values, dtype=np.float64)
    count = len(subject_ids)
    if not all(
        len(values) == count
        for values in (anchors, event_times, scales, points, observed)
    ):
        raise ValueError("Online conformal arrays must have equal length")

    target_alpha = 1.0 - float(coverage)
    widths = np.full(count, np.nan, dtype=np.float64)
    alphas = np.full(count, target_alpha, dtype=np.float64)
    history_counts = np.zeros(count, dtype=np.int64)
    personalized = np.zeros(count, dtype=bool)
    quantiles = np.full(count, np.nan, dtype=np.float64)
    quantile_cache: dict[float, float] = {}

    for subject in np.unique(subject_ids):
        rows = np.flatnonzero(subject_ids == subject)
        order = rows[np.argsort(anchors[rows], kind="stable")]
        alpha = target_alpha
        resolved_count = 0
        pending: dict[np.datetime64, tuple[float, float, float]] = {}
        for row in order:
            anchor = anchors[row]
            if not np.isnat(anchor):
                resolved_keys = sorted(
                    key for key in pending if key <= anchor
                )
                for key in resolved_keys:
                    point, truth, issued_width = pending.pop(key)
                    missed = float(abs(truth - point) > issued_width)
                    alpha = float(
                        np.clip(
                            alpha + float(gamma) * (target_alpha - missed),
                            min_alpha,
                            max_alpha,
                        )
                    )
                    resolved_count += 1

            effective_alpha = (
                alpha
                if resolved_count >= int(min_resolved_history)
                else target_alpha
            )
            cache_key = round(float(effective_alpha), 8)
            quantile = quantile_cache.get(cache_key)
            if quantile is None:
                quantile = patient_weighted_quantile(
                    scores,
                    calibration_ids,
                    coverage=1.0 - effective_alpha,
                )
                quantile_cache[cache_key] = quantile
            width = float(max(scales[row], 1.0e-8) * quantile)
            widths[row] = width
            alphas[row] = effective_alpha
            history_counts[row] = resolved_count
            personalized[row] = resolved_count >= int(min_resolved_history)
            quantiles[row] = quantile

            event_time = event_times[row]
            if (
                not np.isnat(event_time)
                and not np.isnat(anchor)
                and event_time > anchor
                and np.isfinite(points[row])
                and np.isfinite(observed[row])
            ):
                # Keep the latest forecast for one realized measurement.
                pending[event_time] = (points[row], observed[row], width)

    return OnlineIntervalResult(
        widths=widths,
        effective_alpha=alphas,
        prior_resolved_count=history_counts,
        personalized=personalized,
        normalized_quantile=quantiles,
    )


def mondrian_online_adaptive_interval_widths(
    calibration_scores,
    calibration_subjects,
    calibration_regimes,
    test_subjects,
    test_regimes,
    anchor_times,
    future_event_times,
    test_scales,
    point_predictions,
    observed_values,
    *,
    coverage: float = 0.90,
    min_calibration_rows: int = 30,
) -> OnlineIntervalResult:
    """Calibrate causal online intervals within forecast-time-known regimes.

    Regimes must be derived exclusively from information available at the
    forecast anchor. Sparse regimes fall back to the global calibration pool.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64)
    calibration_ids = np.asarray(calibration_subjects, dtype=str)
    calibration_groups = np.asarray(calibration_regimes, dtype=str)
    test_ids = np.asarray(test_subjects, dtype=str)
    test_groups = np.asarray(test_regimes, dtype=str)
    count = len(test_ids)
    if len(scores) != len(calibration_ids) or len(scores) != len(calibration_groups):
        raise ValueError("Mondrian calibration arrays must have equal length")
    if len(test_groups) != count:
        raise ValueError("Mondrian test regimes must align with test rows")

    widths = np.full(count, np.nan, dtype=np.float64)
    alphas = np.full(count, np.nan, dtype=np.float64)
    history_counts = np.zeros(count, dtype=np.int64)
    personalized = np.zeros(count, dtype=bool)
    quantiles = np.full(count, np.nan, dtype=np.float64)
    anchors = np.asarray(anchor_times)
    futures = np.asarray(future_event_times)
    scales = np.asarray(test_scales, dtype=np.float64)
    points = np.asarray(point_predictions, dtype=np.float64)
    observed = np.asarray(observed_values, dtype=np.float64)

    for regime in np.unique(test_groups):
        test_rows = np.flatnonzero(test_groups == regime)
        calibration_rows = np.flatnonzero(calibration_groups == regime)
        if len(calibration_rows) < int(min_calibration_rows):
            calibration_rows = np.arange(len(scores))
        result = online_adaptive_interval_widths(
            scores[calibration_rows],
            calibration_ids[calibration_rows],
            test_ids[test_rows],
            anchors[test_rows],
            futures[test_rows],
            scales[test_rows],
            points[test_rows],
            observed[test_rows],
            coverage=coverage,
        )
        widths[test_rows] = result.widths
        alphas[test_rows] = result.effective_alpha
        history_counts[test_rows] = result.prior_resolved_count
        personalized[test_rows] = result.personalized
        quantiles[test_rows] = result.normalized_quantile

    return OnlineIntervalResult(
        widths=widths,
        effective_alpha=alphas,
        prior_resolved_count=history_counts,
        personalized=personalized,
        normalized_quantile=quantiles,
    )


__all__ = [
    "OnlineIntervalResult",
    "SelectiveIntervalResult",
    "online_adaptive_interval_widths",
    "mondrian_online_adaptive_interval_widths",
    "patient_weighted_quantile",
    "selective_interval_widths",
]
