"""Fail-closed waveform support for the Osler-JEPA observation layer.

Waveforms are the remaining high-precision signal source for factual
forecasting, but they are a separate data modality from the table-based ICU
cohorts.  This module therefore exposes only a manifest/contract layer:
headers can be scanned, signal groups can be identified, and candidate feature
families can be reported.  No waveform feature is authorized for prediction
until a held-out accuracy audit validates it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping


WAVEFORM_SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "ecg": (
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "MCL",
        "MCL1",
        "ECG",
        "EKG",
    ),
    "arterial_pressure": (
        "ABP",
        "ART",
        "ART1",
        "ART2",
        "AR1",
        "AR2",
        "ARTERIAL",
        "ARTERIALBP",
        "ARTERIALPRESSURE",
    ),
    "pleth": (
        "PLETH",
        "PPG",
        "PHOTOPLETHYSMOGRAM",
    ),
    "respiration": (
        "RESP",
        "RESPIRATION",
        "RESPIMP",
        "RESPIRATORY",
    ),
    "oxygen_saturation": (
        "SPO2",
        "SAO2",
        "O2SAT",
    ),
    "capnography": (
        "CO2",
        "ETCO2",
        "CAPNO",
        "CAPNOGRAPHY",
    ),
    "central_venous_pressure": (
        "CVP",
    ),
}

WAVEFORM_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "ecg": (
        "beat_to_beat_heart_rate",
        "rr_interval_variability",
        "tachy_brady_burden",
    ),
    "arterial_pressure": (
        "beat_to_beat_sbp_dbp_map",
        "map_variability",
        "hypotension_burden",
        "pulse_pressure_variability",
    ),
    "pleth": (
        "pleth_variability",
        "perfusion_proxy",
        "desaturation_burden",
    ),
    "respiration": (
        "respiratory_rate_variability",
        "apnea_or_low_ventilation_burden",
    ),
    "oxygen_saturation": (
        "high_frequency_spo2",
        "desaturation_burden",
    ),
    "capnography": (
        "etco2_level_and_variability",
        "ventilation_burden",
    ),
    "central_venous_pressure": (
        "venous_pressure_level",
        "venous_pressure_variability",
    ),
}

WAVEFORM_CANDIDATE_TARGETS = (
    "heart_rate",
    "map",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "oxygen_saturation",
    "respiratory_rate",
)

WAVEFORM_DENIED_AUTHORITIES = (
    "causal_claim",
    "counterfactual_treatment_effect",
    "clinical_decision",
    "runtime_treatment_authority",
    "checkpoint_promotion",
    "active_rule_promotion",
)


@dataclass(frozen=True)
class WaveformHeader:
    """Aggregate-safe metadata parsed from a WFDB header."""

    record_name: str
    header_kind: str
    signal_count: int | None
    sampling_frequency_hz: float | None
    sample_count: int | None
    duration_seconds: float | None
    signals: tuple[str, ...]
    signal_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_signal_name(signal: str) -> str:
    """Normalize a waveform signal label for grouping."""

    return re.sub(r"[^A-Z0-9]", "", signal.upper())


def classify_signal_name(signal: str) -> str | None:
    """Return the coarse physiologic group for a waveform signal label."""

    normalized = normalize_signal_name(signal)
    for group, aliases in WAVEFORM_SIGNAL_GROUPS.items():
        if normalized in {normalize_signal_name(alias) for alias in aliases}:
            return group
    return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value.split("/")[0])
    except (TypeError, ValueError):
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(float(value.split("/")[0]))
    except (TypeError, ValueError):
        return None


def _signal_label_from_line(line: str) -> str | None:
    tokens = line.split()
    if not tokens:
        return None
    if len(tokens) >= 10 and tokens[-1].startswith("#"):
        return tokens[-2]
    label = tokens[-1]
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", label):
        return None
    return label


def parse_wfdb_header_text(text: str) -> WaveformHeader:
    """Parse the aggregate fields needed from a WFDB ``.hea`` header.

    The parser intentionally avoids patient identifiers beyond the local record
    name and does not read waveform samples.  It is designed to support a
    manifest audit before any high-frequency feature extraction exists.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("empty WFDB header")

    header = lines[0].split()
    record_token = header[0]
    is_multi_segment = "/" in record_token
    record_name = record_token.split("/", 1)[0]
    signal_count = _parse_int(header[1]) if len(header) > 1 else None
    sampling_frequency_hz = _parse_float(header[2]) if len(header) > 2 else None
    sample_count = _parse_int(header[3]) if len(header) > 3 else None
    duration_seconds = None
    if sampling_frequency_hz and sample_count is not None:
        duration_seconds = sample_count / sampling_frequency_hz

    expected_signals = 0 if is_multi_segment else signal_count or max(len(lines) - 1, 0)
    signal_lines = lines[1 : 1 + expected_signals]
    signals = tuple(
        label
        for label in (_signal_label_from_line(line) for line in signal_lines)
        if label
    )
    groups = tuple(
        sorted(
            {
                group
                for group in (classify_signal_name(signal) for signal in signals)
                if group
            }
        )
    )

    return WaveformHeader(
        record_name=record_name,
        header_kind="multi_segment_layout" if is_multi_segment else "signal_segment",
        signal_count=signal_count,
        sampling_frequency_hz=sampling_frequency_hz,
        sample_count=sample_count,
        duration_seconds=duration_seconds,
        signals=signals,
        signal_groups=groups,
    )


def summarize_waveform_headers(headers: Iterable[WaveformHeader]) -> dict[str, object]:
    """Return an aggregate-only summary of parsed waveform headers."""

    headers = tuple(headers)
    signal_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()
    sampling_counter: Counter[str] = Counter()
    for header in headers:
        signal_counter.update(normalize_signal_name(signal) for signal in header.signals)
        group_counter.update(header.signal_groups)
        if header.sampling_frequency_hz is not None:
            sampling_counter.update([str(round(header.sampling_frequency_hz, 3))])

    return {
        "record_count": len(headers),
        "signal_counts": dict(sorted(signal_counter.items())),
        "signal_group_counts": dict(sorted(group_counter.items())),
        "sampling_frequency_hz_counts": dict(sorted(sampling_counter.items())),
        "candidate_feature_groups": {
            group: WAVEFORM_FEATURE_GROUPS[group]
            for group in sorted(group_counter)
            if group in WAVEFORM_FEATURE_GROUPS
        },
        "safety": {
            "aggregate_only": True,
            "raw_waveform_samples_included": False,
            "patient_identifiers_included": False,
        },
    }


def waveform_feature_contract() -> dict[str, object]:
    """Return the candidate waveform feature contract."""

    return {
        "status": "candidate_only",
        "scope": "high_frequency_waveform_precision",
        "candidate_signal_groups": WAVEFORM_SIGNAL_GROUPS,
        "candidate_feature_groups": WAVEFORM_FEATURE_GROUPS,
        "candidate_targets": WAVEFORM_CANDIDATE_TARGETS,
        "allowed_uses": (
            "waveform_manifest_inspection",
            "feature_engineering_candidate",
            "capability_reporting",
        ),
        "denied_authorities": WAVEFORM_DENIED_AUTHORITIES,
        "promotion_gate": {
            "required": True,
            "baseline": "existing table-based factual forecast",
            "candidate": "same forecast plus aggregate waveform-derived features",
            "placebo": "same forecast plus capacity-matched random features",
            "splits": "patient-heldout plus careunit/time or hospital-heldout where available",
            "targets": WAVEFORM_CANDIDATE_TARGETS,
            "claim_allowed_before_gate": False,
        },
    }


def waveform_readiness(manifest_summary: Mapping[str, object] | None) -> dict[str, object]:
    """Fail-closed readiness summary for waveform work."""

    contract = waveform_feature_contract()
    record_count = int((manifest_summary or {}).get("record_count") or 0)
    group_counts = dict((manifest_summary or {}).get("signal_group_counts") or {})
    data_available = record_count > 0 and bool(group_counts)
    return {
        "status": "candidate_only",
        "data_available": data_available,
        "record_count": record_count,
        "signal_group_counts": group_counts,
        "feature_contract": contract,
        "accuracy_audit_required": True,
        "factual_prediction_authority": False,
        "reason": (
            "waveform headers found; feature extraction and held-out accuracy audit are still required"
            if data_available
            else "no waveform manifest with signal groups is available; fail closed"
        ),
    }
