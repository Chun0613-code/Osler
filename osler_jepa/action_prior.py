"""Empirical, route-aware DKA action prior compiled from observed action grids."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dka_action_contract import ACTION_KEYS


QUANTILE_LEVELS = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def _grid(value):
    if isinstance(value, str):
        value = json.loads(value)
    array = np.asarray(value, dtype=np.float32)
    return array if array.ndim == 2 and array.shape[1] == len(ACTION_KEYS) else None


@dataclass
class EmpiricalActionPrior:
    payload: dict

    @classmethod
    def fit(cls, frame):
        grids = [_grid(value) for value in frame.get("future_action_grid", [])]
        grids = [grid for grid in grids if grid is not None]
        if not grids:
            raise ValueError("No route-aware future_action_grid values available")
        cells = np.concatenate(grids, axis=0)
        channels = {}
        for index, name in enumerate(ACTION_KEYS):
            values = cells[:, index]
            positive = values[values > 1e-6]
            channels[name] = {
                "active_probability": float(np.mean(values > 1e-6)),
                "positive_cells": int(len(positive)),
                "quantile_levels": QUANTILE_LEVELS.tolist(),
                "positive_quantiles": (
                    np.quantile(positive, QUANTILE_LEVELS).astype(float).tolist()
                    if len(positive) else [0.0] * len(QUANTILE_LEVELS)
                ),
            }
        insulin_activity = np.asarray([
            channels[name]["active_probability"] for name in ACTION_KEYS[:4]
        ], dtype=np.float64)
        insulin_route_probability = (
            insulin_activity / insulin_activity.sum()
            if insulin_activity.sum() else np.asarray([1.0, 0.0, 0.0, 0.0])
        )
        return cls({
            "version": "1.0.0",
            "source_rows": int(len(frame)),
            "source_stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
            "action_keys": list(ACTION_KEYS),
            "channels": channels,
            "insulin_route_probability": insulin_route_probability.tolist(),
            "research_only": True,
        })

    @classmethod
    def from_parquet(cls, path):
        return cls.fit(pd.read_parquet(path))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def _positive_sample(self, name, rng):
        channel = self.payload["channels"][name]
        quantiles = np.asarray(channel["positive_quantiles"], dtype=np.float32)
        if not np.any(quantiles > 0):
            return 0.0
        return float(np.interp(
            rng.uniform(QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1]),
            QUANTILE_LEVELS,
            quantiles,
        ))

    def sample(self, rng):
        action = np.zeros(len(ACTION_KEYS), dtype=np.float32)
        insulin_any = min(0.85, sum(
            self.payload["channels"][name]["active_probability"]
            for name in ACTION_KEYS[:4]
        ))
        if rng.random() < insulin_any:
            route = int(rng.choice(
                4, p=np.asarray(self.payload["insulin_route_probability"])
            ))
            action[route] = self._positive_sample(ACTION_KEYS[route], rng)
        for index, name in enumerate(ACTION_KEYS[4:], start=4):
            probability = self.payload["channels"][name]["active_probability"]
            if rng.random() < probability:
                action[index] = self._positive_sample(name, rng)
        return action

    def constrain(self, action):
        constrained = np.asarray(action, dtype=np.float32).copy()
        for index, name in enumerate(ACTION_KEYS):
            channel = self.payload["channels"][name]
            upper = float(channel["positive_quantiles"][-1])
            if int(channel["positive_cells"]) == 0:
                constrained[index] = 0.0
                continue
            if upper > 0:
                constrained[index] = np.clip(constrained[index], 0.0, upper)
        return constrained


def load_or_fit_action_prior(prior_path=None, cohort_path=None):
    if prior_path and Path(prior_path).exists():
        return EmpiricalActionPrior.load(prior_path)
    if cohort_path and Path(cohort_path).exists():
        return EmpiricalActionPrior.from_parquet(cohort_path)
    return None
