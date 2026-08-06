"""Sparse-aware pure-JEPA candidate for factual body-state forecasting.

Sparse targets are not treated as ordinary dense regression targets.  For each
physiological variable the model learns four separate quantities:

* future value, conditional on the future value being observed;
* probability that the variable will be observed at the requested horizon;
* a heteroscedastic uncertainty scale for the value forecast;
* probability that the target remains stable within a small normalized band.

The observation and stability heads are descriptive.  They are not treatment
or clinical decision models.  Promotion still requires the existing
patient/hospital/conformal gates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from nonlinear_latent_rollout import (
    DEFAULT_HORIZONS,
    StateScaler,
    _history_indices,
    load_multihorizon_cohort,
    set_seed,
)


class SparseAwarePureJEPABodyModel(nn.Module):
    """Measurement-pure history encoder with target-specific sparse heads."""

    def __init__(self, n_state: int, input_dim: int, hidden: int = 64, latent: int = 48,
                 measurement_time_head: bool = False):
        super().__init__()
        self.n_state = int(n_state)
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.measurement_time_head = bool(measurement_time_head)
        self.history_encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden + n_state, latent),
            nn.LayerNorm(latent),
            nn.SiLU(),
        )
        head_input = latent + 1 + 4
        self.value_heads = nn.ModuleList()
        self.observation_heads = nn.ModuleList()
        self.uncertainty_heads = nn.ModuleList()
        self.stability_heads = nn.ModuleList()
        self.measurement_time_heads = nn.ModuleList()
        for _ in range(n_state):
            self.value_heads.append(
                nn.Sequential(nn.Linear(head_input, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            )
            self.observation_heads.append(
                nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
            )
            self.uncertainty_heads.append(
                nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
            )
            self.stability_heads.append(
                nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
            )
            if self.measurement_time_head:
                self.measurement_time_heads.append(
                    nn.Sequential(nn.Linear(head_input, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
                )

    @staticmethod
    def _time_features(horizon: torch.Tensor) -> torch.Tensor:
        horizon = horizon.clamp(min=0.05, max=168.0)
        log_horizon = torch.log1p(horizon)
        return torch.stack(
            [
                log_horizon / math.log1p(48.0),
                horizon / 6.0,
                torch.sin(log_horizon),
                torch.cos(log_horizon),
            ],
            dim=-1,
        )

    def forward(self, observations: torch.Tensor, current: torch.Tensor, horizon: torch.Tensor):
        _, hidden = self.history_encoder(observations)
        latent = self.state_encoder(torch.cat([hidden[-1], current], dim=-1))
        time = self._time_features(horizon)
        values, observed, scales, stable, next_times = [], [], [], [], []
        for index in range(self.n_state):
            head_input = torch.cat([latent, current[:, index : index + 1], time], dim=-1)
            values.append(current[:, index : index + 1] + 0.75 * torch.tanh(self.value_heads[index](head_input)))
            observed.append(self.observation_heads[index](head_input))
            scales.append(self.uncertainty_heads[index](head_input))
            stable.append(self.stability_heads[index](head_input))
            if self.measurement_time_head:
                next_times.append(self.measurement_time_heads[index](head_input))
        output = {
            "value": torch.cat(values, dim=-1),
            "observation_logit": torch.cat(observed, dim=-1),
            "log_scale": torch.cat(scales, dim=-1),
            "stable_logit": torch.cat(stable, dim=-1),
        }
        if self.measurement_time_head:
            output["next_measurement_time_log"] = torch.cat(next_times, dim=-1)
        return output


def _arrays(frame: pd.DataFrame, variables, rows):
    # Sparse labs often contain clinical outliers.  A standard deviation scale
    # lets one extreme value make the target look artificially stable.
    matrix = frame.iloc[rows][[f"{v}_t" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    scales = np.ones(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        finite = matrix[:, index][np.isfinite(matrix[:, index])]
        if len(finite) == 0:
            continue
        medians[index] = float(np.median(finite))
        q25, q75 = np.percentile(finite, [25, 75])
        mad = np.median(np.abs(finite - medians[index]))
        scale = max(float((q75 - q25) / 1.349), float(mad * 1.4826))
        if np.isfinite(scale) and scale >= 1e-6:
            scales[index] = scale
    scaler = StateScaler(tuple(variables), medians.astype(np.float32), scales.astype(np.float32))
    current, mask, ages = scaler.normalize_current(frame)
    future, future_mask = scaler.normalize_future(frame)
    onset = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    onset = np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0)
    onset = np.clip(onset, 0.0, 168.0) / 24.0
    input_matrix = np.concatenate([current, mask, ages, onset[:, None]], axis=1).astype(np.float32)
    history = _history_indices(frame, history=5)
    horizon = frame["_source_horizon"].to_numpy(dtype=np.float32)
    return scaler, input_matrix, current, future, future_mask, history, horizon


def _forward(model, arrays, rows, batch_size=1024):
    _, input_matrix, current, _, _, history, horizon = arrays
    output_keys = ["value", "observation_logit", "log_scale", "stable_logit"]
    if getattr(model, "measurement_time_head", False):
        output_keys.append("next_measurement_time_log")
    output = {key: [] for key in output_keys}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            result = model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
            )
            for key in output:
                output[key].append(result[key].numpy())
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def _split_subjects(frame, seed, fraction=0.25):
    groups = frame["subject_id"].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    selected = set(groups[: max(1, int(round(len(groups) * fraction)))].tolist())
    test = frame["subject_id"].isin(selected).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_target = (values * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
    supported = mask.sum(dim=0) > 0
    return per_target[supported].mean() if supported.any() else values.sum() * 0.0


def fit_one(frame, variables, train_rows, seed=7, epochs=2, batch_size=1024):
    set_seed(seed)
    arrays = _arrays(frame, variables, train_rows)
    _, input_matrix, current, future, future_mask, history, horizon = arrays
    time_columns = []
    for variable in variables:
        column = f"next_measurement_time_hr_{variable}"
        if column in frame.columns:
            time_columns.append(column)
    if not time_columns and "next_measurement_time_hr" in frame.columns:
        time_columns = ["next_measurement_time_hr"]
    has_measurement_time = bool(time_columns)
    model = SparseAwarePureJEPABodyModel(
        len(variables), input_matrix.shape[1], measurement_time_head=has_measurement_time
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(train_rows)
        model.train()
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            result = model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
            )
            y = torch.from_numpy(future[batch]).float()
            observed = torch.from_numpy(future_mask[batch]).float()
            value_loss = nn.functional.smooth_l1_loss(result["value"], y, reduction="none", beta=0.5)
            value_loss = _masked_mean(value_loss, observed)
            observation_loss = nn.functional.binary_cross_entropy_with_logits(
                result["observation_logit"], observed, reduction="none"
            ).mean(dim=0).mean()
            residual = (y - result["value"].detach()).abs()
            scale = nn.functional.softplus(result["log_scale"]) + 0.05
            uncertainty_loss = (((residual / scale) ** 2) + 2.0 * torch.log(scale))
            uncertainty_loss = _masked_mean(uncertainty_loss, observed)
            stable_label = ((y - torch.from_numpy(current[batch]).float()).abs() <= 0.25).float()
            stable_loss = nn.functional.binary_cross_entropy_with_logits(
                result["stable_logit"], stable_label, reduction="none"
            )
            stable_loss = _masked_mean(stable_loss, observed)
            loss = value_loss + 0.20 * observation_loss + 0.05 * uncertainty_loss + 0.10 * stable_loss
            if has_measurement_time:
                batch_frame = frame.iloc[batch]
                if len(time_columns) == 1 and time_columns[0] == "next_measurement_time_hr":
                    next_time = pd.to_numeric(
                        batch_frame[time_columns[0]], errors="coerce"
                    ).to_numpy(dtype=np.float32)[:, None]
                    next_time = np.repeat(next_time, len(variables), axis=1)
                else:
                    next_time = batch_frame[time_columns].apply(
                        pd.to_numeric, errors="coerce"
                    ).to_numpy(dtype=np.float32)
                    missing = len(variables) - next_time.shape[1]
                    if missing > 0:
                        next_time = np.concatenate(
                            [next_time, np.full((len(batch), missing), np.nan, dtype=np.float32)],
                            axis=1,
                        )
                time_mask = np.isfinite(next_time).astype(np.float32)
                time_target = np.log1p(np.nan_to_num(np.clip(next_time, 0.0, 168.0), nan=168.0))
                time_prediction = nn.functional.softplus(result["next_measurement_time_log"])
                time_loss = nn.functional.smooth_l1_loss(
                    time_prediction, torch.from_numpy(time_target).float(), reduction="none"
                )
                time_loss = (time_loss * torch.from_numpy(time_mask).float()).sum() / max(float(time_mask.sum()), 1.0)
                loss = loss + 0.10 * time_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    return model, arrays, losses


def evaluate(model, arrays, frame, variables, rows):
    scaler, _, current, future, future_mask, _, _ = arrays
    output = _forward(model, arrays, rows)
    truth = future[rows] * scaler.scales + scaler.medians
    current_raw = current[rows] * scaler.scales + scaler.medians
    prediction = output["value"] * scaler.scales + scaler.medians
    report = {}
    for index, target in enumerate(variables):
        valid = future_mask[rows, index].astype(bool)
        if valid.sum() < 20:
            report[target] = {"status": "insufficient_support", "n": int(valid.sum())}
            continue
        candidate_error = np.abs(prediction[valid, index] - truth[valid, index])
        persistence_error = np.abs(current_raw[valid, index] - truth[valid, index])
        observation = future_mask[rows, index].astype(np.float32)
        p_observed = 1.0 / (1.0 + np.exp(-output["observation_logit"][:, index]))
        p_stable = 1.0 / (1.0 + np.exp(-output["stable_logit"][:, index]))
        stable = (np.abs(future[rows, index] - current[rows, index]) <= 0.25).astype(np.float32)
        sigma = (np.log1p(np.exp(output["log_scale"][:, index])) + 0.05) * scaler.scales[index]
        lower = prediction[:, index] - 1.645 * sigma
        upper = prediction[:, index] + 1.645 * sigma
        interval_valid = np.isfinite(truth[:, index])
        coverage = float(((truth[interval_valid, index] >= lower[interval_valid]) & (truth[interval_valid, index] <= upper[interval_valid])).mean())
        width = upper - lower
        finite_width = width[np.isfinite(width)]
        report[target] = {
            "n": int(valid.sum()),
            "candidate_mae": float(candidate_error.mean()),
            "persistence_mae": float(persistence_error.mean()),
            "delta_vs_persistence": float(candidate_error.mean() - persistence_error.mean()),
            "observation_rate": float(observation.mean()),
            "observation_brier": float(np.mean((p_observed - observation) ** 2)),
            "observation_brier_baseline": float(
                np.mean((observation.mean() - observation) ** 2)
            ),
            "observation_brier_improvement": float(
                np.mean((observation.mean() - observation) ** 2)
                - np.mean((p_observed - observation) ** 2)
            ),
            "stable_rate": float(stable.mean()),
            "stable_brier": float(np.mean((p_stable - stable) ** 2)),
            "stable_brier_baseline": float(np.mean((stable.mean() - stable) ** 2)),
            "stable_brier_improvement": float(
                np.mean((stable.mean() - stable) ** 2)
                - np.mean((p_stable - stable) ** 2)
            ),
            "nominal_90_coverage_uncalibrated": coverage,
            "uncalibrated_interval_width": float(np.median(finite_width)) if len(finite_width) else None,
            "status": "candidate_only",
        }
    return report


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def conformal_calibration(model, arrays, variables, calibration_rows, test_rows):
    """Calibrate 90% intervals per target and source horizon.

    A pooled target interval can hide a horizon-specific failure.  The runtime
    therefore consumes ``by_horizon`` and refuses to display an interval when
    the requested horizon was not independently calibrated.
    """

    scaler, _, current, future, future_mask, _, horizon = arrays
    calibration = _forward(model, arrays, calibration_rows)
    test = _forward(model, arrays, test_rows)
    truth_cal = future[calibration_rows] * scaler.scales + scaler.medians
    truth_test = future[test_rows] * scaler.scales + scaler.medians
    output = {}
    for index, target in enumerate(variables):
        cal_valid = future_mask[calibration_rows, index].astype(bool)
        test_valid = future_mask[test_rows, index].astype(bool)
        by_horizon = {}
        calibration_horizons = horizon[calibration_rows]
        test_horizons = horizon[test_rows]
        for value in sorted(set(calibration_horizons).intersection(test_horizons)):
            cal_scope = cal_valid & (calibration_horizons == value)
            test_scope = test_valid & (test_horizons == value)
            key = str(int(value)) if float(value).is_integer() else str(float(value))
            if cal_scope.sum() < 30 or test_scope.sum() < 30:
                by_horizon[key] = {
                    "status": "needs_calibration_audit",
                    "calibration_n": int(cal_scope.sum()),
                    "test_n": int(test_scope.sum()),
                }
                continue
            cal_pred = calibration["value"][cal_scope, index] * scaler.scales[index] + scaler.medians[index]
            test_pred = test["value"][test_scope, index] * scaler.scales[index] + scaler.medians[index]
            residual = np.abs(cal_pred - truth_cal[cal_scope, index])
            q = float(np.quantile(residual, 0.90, method="higher"))
            lower = test_pred - q
            upper = test_pred + q
            observed = truth_test[test_scope, index]
            coverage = float(((observed >= lower) & (observed <= upper)).mean())
            by_horizon[key] = {
                "status": "PASS" if 0.87 <= coverage <= 0.93 else "REJECT",
                "calibration_n": int(cal_scope.sum()),
                "test_n": int(test_scope.sum()),
                "q90": q,
                "coverage": coverage,
                "interval_width": 2.0 * q,
            }
        passed = [cell for cell in by_horizon.values() if cell.get("status") in {"PASS", "REJECT"}]
        output[target] = {
            "status": "PASS" if passed and all(cell.get("status") == "PASS" for cell in passed) else "REJECT",
            "scope": "target_by_horizon",
            "by_horizon": by_horizon,
            # Backward-compatible pooled value is retained only as a legacy
            # artifact. Runtime never uses it for a different horizon.
            "q90": float(max((cell["q90"] for cell in passed if cell.get("q90") is not None), default=np.nan)),
        }
    return output


def run_module(data_root, module, horizons, seed, epochs, batch_size, max_stays, checkpoint_dir=None):
    frame, variables = load_multihorizon_cohort(
        data_root, module, horizons, max_stays_per_horizon=max_stays, history=5, seed=seed
    )
    train, test = _split_subjects(frame, seed)
    fit, calibration = _split_subjects(frame.iloc[train].reset_index(drop=True), seed + 1001, 0.20)
    calibration = train[calibration]
    fit = train[fit]
    model, arrays, losses = fit_one(frame, variables, fit, seed, epochs, batch_size)
    measurement_time_head = bool(
        getattr(model, "measurement_time_head", False)
        and any(f"next_measurement_time_hr_{variable}" in frame.columns for variable in variables)
    )
    result = {
        "module": module,
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "train_subjects": int(frame.iloc[train]["subject_id"].nunique()),
        "test_subjects": int(frame.iloc[test]["subject_id"].nunique()),
        "loss_history": losses,
        "metrics": evaluate(model, arrays, frame, variables, test),
        "conformal": conformal_calibration(model, arrays, variables, calibration, test),
        "architecture": {
            "measurement_pure": True,
            "history_encoder": "GRU",
            "target_specific_heads": len(variables),
            "observation_process_head": True,
            "uncertainty_head": True,
            "stability_head": True,
            "measurement_time_head": measurement_time_head,
            "next_measurement_time_validated_targets": [],
            "next_measurement_time_label_source": (
                "raw_eicu_event_offset_strictly_after_anchor" if measurement_time_head else "unavailable"
            ),
            "hand_beliefs_used": False,
            "causal_claim_allowed": False,
            "promotion_status": "candidate_only",
        },
    }
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / f"{module}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "SparseAwarePureJEPABodyModel",
                "state_dict": model.state_dict(),
                "scaler": arrays[0].to_dict(),
                "module": module,
                "horizons": list(horizons),
                "variables": list(variables),
                "input_dim": int(model.input_dim),
                "hidden": int(model.hidden),
                "latent": int(model.latent),
                "architecture": result["architecture"],
                "promotion_status": "candidate_only",
                "conformal": result["conformal"],
            },
            checkpoint_path,
        )
        result["checkpoint"] = str(checkpoint_path.resolve())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--modules", required=True)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output", default="/private/tmp/sparse_aware_pure_jepa_report.json")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    result = {}
    for module in [value.strip() for value in args.modules.split(",") if value.strip()]:
        print(f"running sparse-aware audit: {module}", flush=True)
        try:
            result[module] = run_module(
                args.data_root,
                module,
                horizons,
                args.seed,
                args.epochs,
                args.batch_size,
                args.max_stays,
                args.checkpoint_dir,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            result[module] = {"status": "not_available", "error": str(exc)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
