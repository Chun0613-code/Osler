"""Self-supervised ICU JEPA pretraining on PhysioNet Challenge 2019.

This data path is intentionally separate from the DKA intervention JEPA.  The
Challenge 2019 PSV files contain hourly ICU observations and sepsis labels, but
not explicit medication/action channels.  The model trained here can therefore
learn general patient-state dynamics and observation-mask structure; it must not
be promoted as an intervention or counterfactual model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DYNAMIC_KEYS = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
)
STATIC_KEYS = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")
LABEL_KEY = "SepsisLabel"
TIME_KEY = "ICULOS"
MAX_OBSERVATION_AGE_HOURS = 24.0


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psv_files(data_root: Path, max_patients: int | None = None) -> list[Path]:
    paths = sorted(data_root.glob("**/*.psv"))
    if max_patients is not None:
        paths = paths[:max_patients]
    if not paths:
        raise FileNotFoundError(f"No .psv files found under {data_root}")
    return paths


def read_patient(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    if not rows:
        raise ValueError(f"Empty PSV file: {path}")

    dynamic = np.full((len(rows), len(DYNAMIC_KEYS)), np.nan, dtype=np.float32)
    static = np.full((len(rows), len(STATIC_KEYS)), np.nan, dtype=np.float32)
    sepsis = np.zeros(len(rows), dtype=np.float32)
    time_index = np.arange(len(rows), dtype=np.float32)

    for row_index, row in enumerate(rows):
        for col_index, key in enumerate(DYNAMIC_KEYS):
            value = row.get(key, "")
            if value and value != "NaN":
                dynamic[row_index, col_index] = float(value)
        for col_index, key in enumerate(STATIC_KEYS):
            value = row.get(key, "")
            if value and value != "NaN":
                static[row_index, col_index] = float(value)
        value = row.get(LABEL_KEY, "0")
        sepsis[row_index] = float(value) if value and value != "NaN" else 0.0
        value = row.get(TIME_KEY, "")
        if value and value != "NaN":
            time_index[row_index] = float(value)

    return {
        "dynamic": dynamic,
        "static": static,
        "sepsis": sepsis,
        "time": time_index,
    }


@dataclass
class Normalizer:
    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray

    def to_json(self) -> dict[str, object]:
        return {
            "dynamic_keys": list(DYNAMIC_KEYS),
            "static_keys": list(STATIC_KEYS),
            "dynamic_mean": self.dynamic_mean.round(6).tolist(),
            "dynamic_std": self.dynamic_std.round(6).tolist(),
            "static_mean": self.static_mean.round(6).tolist(),
            "static_std": self.static_std.round(6).tolist(),
        }


def split_paths(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    train_end = int(0.70 * len(paths))
    validation_end = int(0.85 * len(paths))
    return {
        "train": [paths[int(i)] for i in order[:train_end]],
        "validation": [paths[int(i)] for i in order[train_end:validation_end]],
        "test": [paths[int(i)] for i in order[validation_end:]],
    }


def fit_normalizer(paths: list[Path]) -> Normalizer:
    dyn_sum = np.zeros(len(DYNAMIC_KEYS), dtype=np.float64)
    dyn_sum_sq = np.zeros(len(DYNAMIC_KEYS), dtype=np.float64)
    dyn_count = np.zeros(len(DYNAMIC_KEYS), dtype=np.float64)
    static_sum = np.zeros(len(STATIC_KEYS), dtype=np.float64)
    static_sum_sq = np.zeros(len(STATIC_KEYS), dtype=np.float64)
    static_count = np.zeros(len(STATIC_KEYS), dtype=np.float64)

    for path in paths:
        patient = read_patient(path)
        dynamic = patient["dynamic"]
        mask = np.isfinite(dynamic)
        dyn_sum += np.nan_to_num(dynamic, nan=0.0).sum(axis=0)
        dyn_sum_sq += np.nan_to_num(dynamic * dynamic, nan=0.0).sum(axis=0)
        dyn_count += mask.sum(axis=0)

        static = patient["static"][:1]
        static_mask = np.isfinite(static)
        static_sum += np.nan_to_num(static, nan=0.0).sum(axis=0)
        static_sum_sq += np.nan_to_num(static * static, nan=0.0).sum(axis=0)
        static_count += static_mask.sum(axis=0)

    dyn_count = np.maximum(dyn_count, 1.0)
    static_count = np.maximum(static_count, 1.0)
    dyn_mean = dyn_sum / dyn_count
    static_mean = static_sum / static_count
    dyn_var = np.maximum(dyn_sum_sq / dyn_count - dyn_mean * dyn_mean, 1e-6)
    static_var = np.maximum(static_sum_sq / static_count - static_mean * static_mean, 1e-6)
    return Normalizer(
        dyn_mean.astype(np.float32),
        np.sqrt(dyn_var).astype(np.float32),
        static_mean.astype(np.float32),
        np.sqrt(static_var).astype(np.float32),
    )


def carry_forward(raw: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return imputed values, direct-observation mask, and measurement age."""
    values = np.empty_like(raw, dtype=np.float32)
    mask = np.isfinite(raw).astype(np.float32)
    age = np.full_like(raw, MAX_OBSERVATION_AGE_HOURS, dtype=np.float32)
    last_value = mean.astype(np.float32).copy()
    last_seen = np.full(raw.shape[1], -1e9, dtype=np.float32)
    for row in range(raw.shape[0]):
        observed = np.isfinite(raw[row])
        last_value[observed] = raw[row, observed]
        last_seen[observed] = float(row)
        values[row] = last_value
        age[row] = np.clip(float(row) - last_seen, 0.0, MAX_OBSERVATION_AGE_HOURS)
    return values, mask, age


def patient_transitions(
    path: Path,
    normalizer: Normalizer,
    horizon_hours: int,
) -> dict[str, np.ndarray] | None:
    patient = read_patient(path)
    if len(patient["time"]) <= horizon_hours:
        return None
    values, mask, age = carry_forward(patient["dynamic"], normalizer.dynamic_mean)
    static_values = patient["static"][0].copy()
    static_mask = np.isfinite(static_values).astype(np.float32)
    static_values = np.where(np.isfinite(static_values), static_values, normalizer.static_mean)

    x_values, y_values = values[:-horizon_hours], values[horizon_hours:]
    x_mask, y_mask = mask[:-horizon_hours], mask[horizon_hours:]
    x_age, y_age = age[:-horizon_hours], age[horizon_hours:]
    x_time, y_time = patient["time"][:-horizon_hours], patient["time"][horizon_hours:]
    delta = np.clip(y_time - x_time, 1.0, float(horizon_hours)).astype(np.float32)
    static = np.repeat(static_values[None, :], len(x_values), axis=0)
    static_observed = np.repeat(static_mask[None, :], len(x_values), axis=0)
    sepsis = patient["sepsis"][horizon_hours:].astype(np.float32)

    return {
        "x": ((x_values - normalizer.dynamic_mean) / normalizer.dynamic_std).astype(np.float32),
        "x_mask": x_mask.astype(np.float32),
        "x_age": (x_age / MAX_OBSERVATION_AGE_HOURS).astype(np.float32),
        "y": ((y_values - normalizer.dynamic_mean) / normalizer.dynamic_std).astype(np.float32),
        "y_mask": y_mask.astype(np.float32),
        "y_age": (y_age / MAX_OBSERVATION_AGE_HOURS).astype(np.float32),
        "static": ((static - normalizer.static_mean) / normalizer.static_std).astype(np.float32),
        "static_mask": static_observed.astype(np.float32),
        "delta": (delta / max(float(horizon_hours), 1.0)).reshape(-1, 1).astype(np.float32),
        "sepsis": sepsis.reshape(-1, 1),
    }


def build_split_arrays(
    paths: list[Path],
    normalizer: Normalizer,
    horizon_hours: int,
    max_transitions: int | None,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "x": [], "x_mask": [], "x_age": [], "y": [], "y_mask": [], "y_age": [],
        "static": [], "static_mask": [], "delta": [], "sepsis": [],
    }
    total = 0
    for path in paths:
        transition = patient_transitions(path, normalizer, horizon_hours)
        if transition is None:
            continue
        remaining = None if max_transitions is None else max_transitions - total
        if remaining is not None and remaining <= 0:
            break
        take = len(transition["x"]) if remaining is None else min(len(transition["x"]), remaining)
        for key in parts:
            parts[key].append(transition[key][:take])
        total += take
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def build_cache(
    data_root: Path,
    cache_path: Path,
    max_patients: int | None,
    max_transitions_per_split: int | None,
    horizon_hours: int,
    seed: int,
) -> dict[str, object]:
    paths = psv_files(data_root, max_patients=max_patients)
    splits = split_paths(paths, seed)
    normalizer = fit_normalizer(splits["train"])
    arrays = {}
    for split, split_paths_ in splits.items():
        arrays[split] = build_split_arrays(
            split_paths_, normalizer, horizon_hours, max_transitions_per_split
        )
    payload = {
        "normalizer_json": json.dumps(normalizer.to_json()),
        "horizon_hours": np.array([horizon_hours], dtype=np.float32),
    }
    for split, split_arrays in arrays.items():
        for key, value in split_arrays.items():
            payload[f"{split}_{key}"] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return cache_summary(cache_path)


def cache_summary(cache_path: Path) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as data:
        normalizer = json.loads(str(data["normalizer_json"]))
        splits = {}
        for split in ("train", "validation", "test"):
            x_key = f"{split}_x"
            if x_key in data:
                splits[split] = {
                    "transitions": int(data[x_key].shape[0]),
                    "direct_observation_density": round(float(data[f"{split}_x_mask"].mean()), 6),
                    "future_target_density": round(float(data[f"{split}_y_mask"].mean()), 6),
                    "sepsis_positive_rate": round(float(data[f"{split}_sepsis"].mean()), 6),
                }
    return {
        "cache": str(cache_path),
        "dynamic_features": normalizer["dynamic_keys"],
        "static_features": normalizer["static_keys"],
        "splits": splits,
        "intervention_claim_allowed": False,
        "intended_use": "general ICU state-dynamics pretraining only",
    }


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, cache: np.lib.npyio.NpzFile, split: str):
        self.arrays = {
            key: cache[f"{split}_{key}"]
            for key in (
                "x", "x_mask", "x_age", "y", "y_mask", "y_age",
                "static", "static_mask", "delta", "sepsis",
            )
        }

    def __len__(self) -> int:
        return int(self.arrays["x"].shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.as_tensor(self.arrays[key][index], dtype=torch.float32)
            for key in (
                "x", "x_mask", "x_age", "y", "y_mask", "y_age",
                "static", "static_mask", "delta", "sepsis",
            )
        )


def mlp(input_dim: int, output_dim: int, hidden: int = 192) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, output_dim),
    )


class PhysioNetICUJepa(nn.Module):
    def __init__(self, dynamic_dim: int, static_dim: int, latent_dim: int = 48):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        state_input = dynamic_dim * 3 + static_dim * 2
        self.E = mlp(state_input, latent_dim)
        self.Ebar = mlp(state_input, latent_dim)
        self.Ebar.load_state_dict(self.E.state_dict())
        for parameter in self.Ebar.parameters():
            parameter.requires_grad_(False)
        self.P = mlp(latent_dim + 1, latent_dim)
        self.D = mlp(latent_dim, dynamic_dim)
        self.Risk = mlp(latent_dim, 1, hidden=96)

    def _features(self, value, mask, age, static, static_mask):
        return torch.cat([value, mask, age, static, static_mask], dim=-1)

    def encode(self, value, mask, age, static, static_mask):
        return self.E(self._features(value, mask, age, static, static_mask))

    def encode_target(self, value, mask, age, static, static_mask):
        return self.Ebar(self._features(value, mask, age, static, static_mask))

    def predict_latent(self, latent, delta):
        return latent + 0.2 * self.P(torch.cat([latent, delta], dim=-1))

    def forward(self, x, x_mask, x_age, static, static_mask, delta):
        latent = self.encode(x, x_mask, x_age, static, static_mask)
        future_latent = self.predict_latent(latent, delta)
        return self.D(future_latent), self.Risk(future_latent), latent, future_latent

    @torch.no_grad()
    def ema(self, tau: float = 0.995):
        for target, online in zip(self.Ebar.parameters(), self.E.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)


def vicreg(latent: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    if latent.shape[0] < 2:
        return latent.new_tensor(0.0)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    variance = torch.relu(gamma - std).mean()
    centered = latent - latent.mean(dim=0)
    covariance = centered.T @ centered / max(latent.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return variance + off_diagonal.square().sum() / latent.shape[1]


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def batch_loss(model: PhysioNetICUJepa, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    x, x_mask, x_age, y, y_mask, y_age, static, static_mask, delta, sepsis = batch
    prediction, risk_logit, latent, future_latent = model(
        x, x_mask, x_age, static, static_mask, delta
    )
    with torch.no_grad():
        target_latent = model.encode_target(y, y_mask, y_age, static, static_mask)
    state = masked_mse(prediction, y, y_mask)
    reconstruction = masked_mse(model.D(latent), x, x_mask)
    latent_loss = F.mse_loss(future_latent, target_latent)
    risk = F.binary_cross_entropy_with_logits(risk_logit, sepsis)
    anti_collapse = vicreg(latent)
    total = state + 0.5 * latent_loss + 0.25 * reconstruction + 0.1 * risk + 0.05 * anti_collapse
    return {
        "total": total,
        "state": state,
        "latent": latent_loss,
        "reconstruction": reconstruction,
        "risk": risk,
        "anti_collapse": anti_collapse,
    }


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def train_model(
    model: PhysioNetICUJepa,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_state = None
    best_validation = math.inf
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        parts = []
        for batch in train_loader:
            losses = batch_loss(model, move_batch(batch, device))
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.ema()
            parts.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        scheduler.step()

        validation = evaluate_loss(model, validation_loader, device)
        summary = {key: float(np.mean([part[key] for part in parts])) for key in parts[0]}
        summary["validation"] = validation
        summary["epoch"] = epoch
        history.append(summary)
        if validation < best_validation:
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch {epoch:03d}/{epochs} train={summary['total']:.4f} "
            f"val={validation:.4f} state={summary['state']:.4f} "
            f"risk={summary['risk']:.4f} collapse={summary['anti_collapse']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state or model.state_dict()


@torch.no_grad()
def evaluate_loss(
    model: PhysioNetICUJepa,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        losses.append(float(batch_loss(model, move_batch(batch, device))["total"].cpu()))
    return float(np.mean(losses)) if losses else math.inf


@torch.no_grad()
def evaluate_model(
    model: PhysioNetICUJepa,
    loader: torch.utils.data.DataLoader,
    normalizer: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    dynamic_std = torch.as_tensor(normalizer["dynamic_std"], dtype=torch.float32, device=device)
    abs_error_sum = torch.zeros(len(DYNAMIC_KEYS), device=device)
    persistence_error_sum = torch.zeros(len(DYNAMIC_KEYS), device=device)
    observed_count = torch.zeros(len(DYNAMIC_KEYS), device=device)
    latents = []
    risk_logits = []
    risk_targets = []
    total_state_loss = []
    for batch in loader:
        x, x_mask, x_age, y, y_mask, y_age, static, static_mask, delta, sepsis = move_batch(batch, device)
        prediction, risk_logit, latent, _ = model(x, x_mask, x_age, static, static_mask, delta)
        abs_error_sum += ((prediction - y).abs() * dynamic_std * y_mask).sum(dim=0)
        persistence_error_sum += ((x - y).abs() * dynamic_std * y_mask).sum(dim=0)
        observed_count += y_mask.sum(dim=0)
        total_state_loss.append(float(masked_mse(prediction, y, y_mask).cpu()))
        latents.append(latent.cpu())
        risk_logits.append(risk_logit.cpu())
        risk_targets.append(sepsis.cpu())

    latent_matrix = torch.cat(latents, dim=0).numpy()
    std = latent_matrix.std(axis=0)
    covariance = np.cov(latent_matrix, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    effective_rank = float(np.exp(entropy))
    model_mae = (abs_error_sum / observed_count.clamp_min(1.0)).cpu().numpy()
    persistence_mae = (
        persistence_error_sum / observed_count.clamp_min(1.0)
    ).cpu().numpy()
    per_feature = {}
    for index, key in enumerate(DYNAMIC_KEYS):
        per_feature[key] = {
            "model_mae": round(float(model_mae[index]), 4),
            "persistence_mae": round(float(persistence_mae[index]), 4),
            "beats_persistence": bool(model_mae[index] < persistence_mae[index]),
            "observed_targets": int(observed_count[index].cpu()),
        }
    return {
        "masked_state_mse": round(float(np.mean(total_state_loss)), 6),
        "per_feature_physical_mae": per_feature,
        "latent_collapse": {
            "mean_latent_std": round(float(std.mean()), 6),
            "min_latent_std": round(float(std.min()), 6),
            "effective_rank": round(effective_rank, 3),
            "active_dimensions_std_gt_0_01": int((std > 0.01).sum()),
            "collapsed": bool(std.mean() < 0.05 or effective_rank < 3.0),
        },
    }


def save_checkpoint(
    model: PhysioNetICUJepa,
    path: Path,
    normalizer: dict[str, object],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "dynamic_keys": list(DYNAMIC_KEYS),
            "static_keys": list(STATIC_KEYS),
            "normalizer": normalizer,
            "metadata": {
                **metadata,
                "intervention_claim_allowed": False,
                "counterfactual_claim_allowed": False,
                "intended_use": "general ICU state-dynamics pretraining only",
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/chunyouchang/mimic/physionet.org/files/challenge-2019/1.0.0/training"),
    )
    parser.add_argument("--cache", type=Path, default=Path("physionet2019_cache.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("physionet2019_icu_jepa.pt"))
    parser.add_argument("--report", type=Path, default=Path("physionet2019_icu_jepa_report.json"))
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-transitions-per-split", type=int, default=None)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.rebuild_cache or not args.cache.exists():
        summary = build_cache(
            args.data_root,
            args.cache,
            args.max_patients,
            args.max_transitions_per_split,
            args.horizon_hours,
            args.seed,
        )
    else:
        summary = cache_summary(args.cache)
    if args.profile_only:
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    cache = np.load(args.cache, allow_pickle=False)
    normalizer = json.loads(str(cache["normalizer_json"]))
    train_set = TransitionDataset(cache, "train")
    validation_set = TransitionDataset(cache, "validation")
    test_set = TransitionDataset(cache, "test")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False
    )
    device = choose_device(args.device)
    model = PhysioNetICUJepa(len(DYNAMIC_KEYS), len(STATIC_KEYS))
    history, _ = train_model(
        model, train_loader, validation_loader, args.epochs,
        args.learning_rate, device
    )
    validation_metrics = evaluate_model(model, validation_loader, normalizer, device)
    test_metrics = evaluate_model(model, test_loader, normalizer, device)
    metadata = {
        "data_root": str(args.data_root),
        "cache": str(args.cache),
        "patients_seen_cap": args.max_patients,
        "max_transitions_per_split": args.max_transitions_per_split,
        "horizon_hours": args.horizon_hours,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    save_checkpoint(model, args.checkpoint, normalizer, metadata)
    report = {
        "dataset": summary,
        "metadata": metadata,
        "training_history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "promotion_boundary": {
            "may_initialize_or_pretrain_generic_encoder": True,
            "may_replace_dka_intervention_checkpoint": False,
            "reason": "PhysioNet 2019 has no explicit DKA treatment action channels.",
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "report": str(args.report),
        "test_masked_state_mse": report["test"]["masked_state_mse"],
        "collapsed": report["test"]["latent_collapse"]["collapsed"],
    }, indent=2))


if __name__ == "__main__":
    main()
