"""Waveform feature extraction utilities for candidate precision audits.

The functions here deliberately produce simple, auditable aggregate features
from a bounded waveform window.  They do not make predictions and do not grant
forecast authority; they only prepare candidate inputs for a later held-out
accuracy audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import find_peaks

from osler_jepa.waveform import classify_signal_name


@dataclass(frozen=True)
class WaveformFeatureRecord:
    """Candidate high-frequency features for one waveform segment."""

    record_name: str
    source_path: str
    sampling_frequency_hz: float | None
    seconds_read: float
    signals: tuple[str, ...]
    signal_groups: tuple[str, ...]
    features: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return values


def robust_univariate_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return robust scalar features for a single waveform channel."""

    clean = _clean_values(values)
    if clean.size == 0:
        return {
            f"{prefix}_valid_fraction": 0.0,
            f"{prefix}_median": math.nan,
            f"{prefix}_iqr": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    q25, q75 = np.percentile(clean, [25, 75])
    return {
        f"{prefix}_valid_fraction": float(clean.size / max(len(values), 1)),
        f"{prefix}_median": float(np.median(clean)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_std": float(np.std(clean)),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
    }


def peak_interval_features(
    values: np.ndarray,
    sampling_frequency_hz: float | None,
    prefix: str,
    min_interval_seconds: float,
    max_interval_seconds: float,
) -> dict[str, float]:
    """Return simple peak/cycle features for a bounded waveform channel."""

    clean = _clean_values(values)
    if clean.size < 3 or not sampling_frequency_hz:
        return {
            f"{prefix}_peak_count": 0.0,
            f"{prefix}_peak_rate_per_min": math.nan,
            f"{prefix}_peak_interval_median_s": math.nan,
            f"{prefix}_peak_interval_iqr_s": math.nan,
        }
    centered = np.asarray(values, dtype=float)
    finite = np.isfinite(centered)
    if not finite.any():
        return {
            f"{prefix}_peak_count": 0.0,
            f"{prefix}_peak_rate_per_min": math.nan,
            f"{prefix}_peak_interval_median_s": math.nan,
            f"{prefix}_peak_interval_iqr_s": math.nan,
        }
    fill_value = float(np.nanmedian(centered[finite]))
    centered = np.where(finite, centered, fill_value)
    centered = centered - float(np.median(centered))
    scale = float(np.std(centered))
    if scale <= 1e-9:
        return {
            f"{prefix}_peak_count": 0.0,
            f"{prefix}_peak_rate_per_min": math.nan,
            f"{prefix}_peak_interval_median_s": math.nan,
            f"{prefix}_peak_interval_iqr_s": math.nan,
        }
    normalized = centered / scale
    min_distance = max(1, int(round(min_interval_seconds * sampling_frequency_hz)))
    peaks, _ = find_peaks(normalized, distance=min_distance, prominence=0.5)
    if len(peaks) < 2:
        peaks, _ = find_peaks(-normalized, distance=min_distance, prominence=0.5)
    duration_minutes = len(values) / sampling_frequency_hz / 60.0
    intervals = np.diff(peaks) / sampling_frequency_hz if len(peaks) >= 2 else np.array([])
    intervals = intervals[
        (intervals >= min_interval_seconds)
        & (intervals <= max_interval_seconds)
    ]
    if intervals.size:
        q25, q75 = np.percentile(intervals, [25, 75])
        rate = 60.0 / float(np.median(intervals))
        interval_median = float(np.median(intervals))
        interval_iqr = float(q75 - q25)
    else:
        rate = float(len(peaks) / duration_minutes) if duration_minutes > 0 else math.nan
        interval_median = math.nan
        interval_iqr = math.nan
    return {
        f"{prefix}_peak_count": float(len(peaks)),
        f"{prefix}_peak_rate_per_min": float(rate),
        f"{prefix}_peak_interval_median_s": interval_median,
        f"{prefix}_peak_interval_iqr_s": interval_iqr,
    }


def arterial_pressure_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return candidate arterial-pressure burden features."""

    clean = _clean_values(values)
    features = robust_univariate_features(values, prefix)
    if clean.size == 0:
        features.update(
            {
                f"{prefix}_hypotension_fraction_lt65": math.nan,
                f"{prefix}_hypertension_fraction_gt120": math.nan,
            }
        )
        return features
    features.update(
        {
            f"{prefix}_hypotension_fraction_lt65": float(np.mean(clean < 65.0)),
            f"{prefix}_hypertension_fraction_gt120": float(np.mean(clean > 120.0)),
        }
    )
    return features


def waveform_array_features(
    signal_values: np.ndarray,
    signal_names: list[str] | tuple[str, ...],
    sampling_frequency_hz: float | None = None,
) -> dict[str, float]:
    """Summarize waveform channels into candidate feature columns."""

    if signal_values.ndim != 2:
        raise ValueError("signal_values must be a 2D array")
    if signal_values.shape[1] != len(signal_names):
        raise ValueError("signal name count must match signal array columns")

    features: dict[str, float] = {}
    seen_group_counts: dict[str, int] = {}
    for column_index, signal_name in enumerate(signal_names):
        group = classify_signal_name(signal_name)
        if group is None:
            continue
        seen_group_counts[group] = seen_group_counts.get(group, 0) + 1
        prefix = f"wave_{group}_{seen_group_counts[group]}"
        column = signal_values[:, column_index]
        if group == "arterial_pressure":
            features.update(arterial_pressure_features(column, prefix))
            features.update(
                peak_interval_features(
                    column,
                    sampling_frequency_hz,
                    prefix,
                    min_interval_seconds=0.3,
                    max_interval_seconds=2.0,
                )
            )
        else:
            features.update(robust_univariate_features(column, prefix))
            if group in {"ecg", "pleth"}:
                features.update(
                    peak_interval_features(
                        column,
                        sampling_frequency_hz,
                        prefix,
                        min_interval_seconds=0.3,
                        max_interval_seconds=2.0,
                    )
                )
            elif group == "respiration":
                features.update(
                    peak_interval_features(
                        column,
                        sampling_frequency_hz,
                        prefix,
                        min_interval_seconds=1.0,
                        max_interval_seconds=10.0,
                    )
                )
    return features


def read_waveform_feature_record(record_path: str | Path, seconds: float = 60.0) -> WaveformFeatureRecord:
    """Read a bounded waveform segment and return candidate features."""

    return read_waveform_feature_window(record_path, sampfrom=0, seconds=seconds)


def read_waveform_feature_window(
    record_path: str | Path,
    sampfrom: int = 0,
    seconds: float = 60.0,
) -> WaveformFeatureRecord:
    """Read a bounded waveform window and return candidate features."""

    import wfdb  # Optional dependency; installed only for waveform work.

    record_path = Path(record_path)
    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs) if header.fs is not None else None
    sample_count = int(round(seconds * fs)) if fs else None
    sampfrom = max(0, int(sampfrom))
    if header.sig_len is not None and sample_count is not None:
        sample_count = min(sample_count, max(int(header.sig_len) - sampfrom, 0))
    if sample_count is None or sample_count <= 0:
        sample_count = max(int(header.sig_len or 0) - sampfrom, 0)
    if sample_count <= 0:
        raise ValueError(f"record has no readable samples: {record_path}")
    sampto = sampfrom + sample_count

    record = wfdb.rdrecord(
        str(record_path),
        sampfrom=sampfrom,
        sampto=sampto,
        physical=True,
    )
    signal_values = np.asarray(record.p_signal, dtype=float)
    signal_names = tuple(str(name) for name in record.sig_name)
    groups = tuple(sorted({group for group in (classify_signal_name(name) for name in signal_names) if group}))
    seconds_read = float(signal_values.shape[0] / fs) if fs else 0.0
    return WaveformFeatureRecord(
        record_name=str(record.record_name),
        source_path=str(record_path),
        sampling_frequency_hz=fs,
        seconds_read=seconds_read,
        signals=signal_names,
        signal_groups=groups,
        features=waveform_array_features(signal_values, signal_names, sampling_frequency_hz=fs),
    )


def aggregate_feature_summary(records: list[WaveformFeatureRecord]) -> dict[str, object]:
    """Return an aggregate-only summary for candidate waveform features."""

    feature_names = sorted({name for record in records for name in record.features})
    group_counts: dict[str, int] = {}
    for record in records:
        for group in record.signal_groups:
            group_counts[group] = group_counts.get(group, 0) + 1
    return {
        "record_count": len(records),
        "signal_group_counts": dict(sorted(group_counts.items())),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "seconds_read_median": (
            float(np.median([record.seconds_read for record in records]))
            if records
            else 0.0
        ),
        "boundary": {
            "raw_samples_committed": False,
            "row_level_features_committed": False,
            "prediction_authority": False,
        },
    }
