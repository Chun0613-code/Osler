"""Shared route-aware DKA action contract with legacy five-channel support."""

from __future__ import annotations

import numpy as np


ACTION_KEYS = (
    "insulin_iv",
    "insulin_rapid_sc",
    "insulin_intermediate_sc",
    "insulin_basal_sc",
    "fluids",
    "kcl",
    "bicarbonate",
    "dextrose",
)
INSULIN_KEYS = ACTION_KEYS[:4]
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_KEYS)}
ACTION_SCALE = np.array(
    [20.0, 40.0, 40.0, 80.0, 500.0, 20.0, 50.0, 20.0],
    dtype=np.float32,
)

LEGACY_KEYS = ("insulin", "fluids", "kcl", "bicarbonate", "dextrose")


def expand_action(action) -> np.ndarray:
    """Return the eight-channel physical action vector.

    Five-element vectors retain their historical meaning and map insulin to IV.
    Dictionaries may use either route-aware keys or the legacy ``insulin`` key.
    """
    if isinstance(action, dict):
        values = np.zeros(len(ACTION_KEYS), dtype=np.float32)
        for key, value in action.items():
            if key == "insulin":
                values[ACTION_INDEX["insulin_iv"]] += float(value or 0.0)
            elif key in ACTION_INDEX:
                values[ACTION_INDEX[key]] += float(value or 0.0)
        return np.maximum(values, 0.0)

    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    if len(raw) <= len(LEGACY_KEYS):
        padded = np.zeros(len(LEGACY_KEYS), dtype=np.float32)
        padded[:len(raw)] = raw
        values = np.zeros(len(ACTION_KEYS), dtype=np.float32)
        values[ACTION_INDEX["insulin_iv"]] = padded[0]
        values[ACTION_INDEX["fluids"]:] = padded[1:]
        return np.maximum(values, 0.0)
    values = np.zeros(len(ACTION_KEYS), dtype=np.float32)
    values[:min(len(raw), len(values))] = raw[:len(values)]
    return np.maximum(values, 0.0)


def legacy_action(action) -> np.ndarray:
    values = expand_action(action)
    return np.array([
        values[:4].sum(),
        values[ACTION_INDEX["fluids"]],
        values[ACTION_INDEX["kcl"]],
        values[ACTION_INDEX["bicarbonate"]],
        values[ACTION_INDEX["dextrose"]],
    ], dtype=np.float32)


def insulin_total(action) -> float:
    return float(expand_action(action)[:4].sum())


def action_dict(action) -> dict[str, float]:
    return dict(zip(ACTION_KEYS, expand_action(action).astype(float).tolist()))
