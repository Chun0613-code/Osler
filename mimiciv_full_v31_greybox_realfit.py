"""Aggregate-only real-cohort grey-box residual evaluation.

This script asks one narrow question:

Can a constrained DKABody residual, fit on real MIMIC-IV DKA transitions with
patient-held-out cross-fitting, beat persistence on observed factual outcomes?

It intentionally writes aggregate reports only. It does not write row-level
predictions, stay lists, patient identifiers, or timestamp cutoffs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from dka_world_model import S_STD
from osler_jepa.greybox_residual import GreyBoxResidualRuntime, RESIDUAL_KEYS
from train_greybox_residual import build_examples, fit_network, rollout


SCALE_BY_STATE = {
    "G": float(S_STD[0]),
    "Ket": float(S_STD[3]),
    "HCO3": float(S_STD[2]),
    "Ke": float(S_STD[4]),
    "Na": float(S_STD[8]),
    "Cr": float(S_STD[10]),
}

CURRENT_COLUMNS = {
    "G": "glucose_t",
    "Ket": "anion_gap_t",
    "HCO3": "bicarbonate_t",
    "Ke": "potassium_t",
    "Na": "sodium_t",
    "Cr": "creatinine_t",
}

STATE_LABELS = {
    "G": "glucose",
    "Ket": "ketone_proxy_from_anion_gap",
    "HCO3": "bicarbonate",
    "Ke": "serum_potassium",
    "Na": "sodium",
    "Cr": "creatinine",
}


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _current(row: pd.Series, state: str) -> float:
    value = _finite(row.get(CURRENT_COLUMNS[state]))
    if state == "Ket" and np.isfinite(value):
        value = max(0.0, value - 12.0)
    return value


def _round(value, digits=6):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _bootstrap_ci(values, seed: int, samples: int) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        sample = rng.choice(values, size=len(values), replace=True)
        draws.append(float(sample.mean()))
    return [
        round(float(np.quantile(draws, 0.025)), 6),
        round(float(np.quantile(draws, 0.975)), 6),
    ]


def _cohort_summary(frame: pd.DataFrame, examples: list[dict]) -> dict:
    summary = {
        "rows": int(len(frame)),
        "examples_with_mechanism_rollout": int(len(examples)),
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else None,
        "active_dka_rows": (
            int(frame["dka_active_t"].fillna(False).sum())
            if "dka_active_t" in frame else None
        ),
        "all_icd_supported": (
            bool(frame["icd_dka_support"].fillna(False).all())
            if "icd_dka_support" in frame else None
        ),
    }
    for state, current in CURRENT_COLUMNS.items():
        future = {
            "G": "glucose_tp6",
            "Ket": "anion_gap_tp6",
            "HCO3": "bicarbonate_tp6",
            "Ke": "potassium_tp6",
            "Na": "sodium_tp6",
            "Cr": "creatinine_tp6",
        }[state]
        if current in frame and future in frame:
            summary[f"{STATE_LABELS[state]}_pairs"] = int(
                (frame[current].notna() & frame[future].notna()).sum()
            )
    return summary


def _crossfit_rows(
    examples: list[dict],
    seed: int,
    bottleneck: int,
    epochs: int,
    folds: int,
) -> pd.DataFrame:
    groups = np.asarray([item["stay_id"] for item in examples])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        raise RuntimeError("At least four stays are required for grouped evaluation")
    splitter = GroupKFold(n_splits=min(folds, len(unique_groups)))
    rows = []
    for fold, (train, test) in enumerate(
        splitter.split(np.arange(len(examples)), groups=groups)
    ):
        print(
            f"[greybox-realfit] seed={seed} fold={fold + 1}/"
            f"{splitter.n_splits} train={len(train)} heldout={len(test)}",
            file=sys.stderr,
            flush=True,
        )
        inner_groups = groups[train]
        inner_splitter = GroupKFold(n_splits=min(3, len(np.unique(inner_groups))))
        inner_train_local, inner_validation_local = next(
            inner_splitter.split(train, groups=inner_groups)
        )
        inner_train = train[inner_train_local]
        inner_validation = train[inner_validation_local]
        model, mean, std, validation_loss = fit_network(
            examples,
            inner_train,
            validation_indices=inner_validation,
            seed=seed + fold,
            epochs=epochs,
            bottleneck=bottleneck,
        )
        runtime = GreyBoxResidualRuntime(model, mean, std)
        print(
            f"[greybox-realfit] seed={seed} fold={fold + 1} "
            f"validation_loss={validation_loss:.6f}; rolling heldout",
            file=sys.stderr,
            flush=True,
        )
        for offset, index in enumerate(test, start=1):
            if offset % 1000 == 0:
                print(
                    f"[greybox-realfit] seed={seed} fold={fold + 1} "
                    f"rolled={offset}/{len(test)}",
                    file=sys.stderr,
                    flush=True,
                )
            item = examples[index]
            _, prediction, _ = rollout(item["row"], runtime)
            row = item["row"]
            active_dka = bool(row.get("dka_active_t", False))
            for state_index, state in enumerate(RESIDUAL_KEYS):
                if not item["mask"][state_index]:
                    continue
                current = _current(row, state)
                if not np.isfinite(current):
                    continue
                truth = float(item["target"][state_index])
                rows.append({
                    "fold": int(fold),
                    "validation_loss": float(validation_loss),
                    "stay_id": int(item["stay_id"]),
                    "state": state,
                    "active_dka": active_dka,
                    "truth": truth,
                    "scale": float(SCALE_BY_STATE[state]),
                    "persistence": float(current),
                    "mechanism": float(item["mechanism"][state_index]),
                    "greybox": float(prediction[state_index]),
                })
    return pd.DataFrame(rows)


def _selected_rows(
    rows: pd.DataFrame,
    active_only: bool,
    states: list[str] | None = None,
) -> pd.DataFrame:
    selected = rows
    if active_only:
        selected = selected[selected["active_dka"].astype(bool)]
    if states is not None:
        selected = selected[selected["state"].isin(states)]
    return selected.copy()


def _stay_deltas(
    rows: pd.DataFrame,
    method: str,
    active_only: bool,
    states: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    selected = _selected_rows(rows, active_only=active_only, states=states)
    if selected.empty:
        return np.asarray([], dtype=np.float64), 0
    for column in ("truth", "scale", "persistence", method):
        selected = selected[np.isfinite(selected[column])]
    if selected.empty:
        return np.asarray([], dtype=np.float64), 0
    selected = selected.assign(
        method_error=(selected[method] - selected["truth"]).abs() / selected["scale"],
        persistence_error=(
            selected["persistence"] - selected["truth"]
        ).abs() / selected["scale"],
    )
    grouped = selected.groupby("stay_id")[["method_error", "persistence_error"]].mean()
    return (
        (grouped["method_error"] - grouped["persistence_error"]).to_numpy(),
        int(len(grouped)),
    )


def _scope_summary(
    rows: pd.DataFrame,
    method: str,
    active_only: bool,
    seed: int,
    bootstrap_samples: int,
    states: list[str] | None = None,
) -> dict:
    selected = _selected_rows(rows, active_only=active_only, states=states)
    valid = selected.copy()
    for column in ("truth", "scale", "persistence", method):
        if column in valid:
            valid = valid[np.isfinite(valid[column])]
    if valid.empty:
        return {
            "rows": 0,
            "stays": 0,
            "persistence_normalized_mae": None,
            f"{method}_normalized_mae": None,
            "delta_vs_persistence": None,
        }
    persistence_error = (
        valid["persistence"] - valid["truth"]
    ).abs() / valid["scale"]
    method_error = (valid[method] - valid["truth"]).abs() / valid["scale"]
    deltas, stays = _stay_deltas(
        rows, method, active_only=active_only, states=states
    )
    ci = _bootstrap_ci(deltas, seed=seed, samples=bootstrap_samples)
    point_delta = float(np.mean(deltas)) if len(deltas) else None
    return {
        "rows": int(len(valid)),
        "stays": stays,
        "persistence_normalized_mae": _round(persistence_error.mean()),
        f"{method}_normalized_mae": _round(method_error.mean()),
        "delta_vs_persistence": {
            "point_delta": _round(point_delta),
            "bootstrap_95_ci": ci,
            "beats_persistence": bool(point_delta is not None and point_delta < 0),
            "significant": bool(ci is not None and ci[1] < 0),
        },
    }


def _per_state_summary(
    rows: pd.DataFrame,
    method: str,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    result = {}
    for offset, state in enumerate(RESIDUAL_KEYS):
        result[state] = {
            "label": STATE_LABELS[state],
            "all_windows": _scope_summary(
                rows,
                method,
                active_only=False,
                seed=seed + offset,
                bootstrap_samples=bootstrap_samples,
                states=[state],
            ),
            "active_dka_only": _scope_summary(
                rows,
                method,
                active_only=True,
                seed=seed + 100 + offset,
                bootstrap_samples=bootstrap_samples,
                states=[state],
            ),
        }
    return result


def _seed_report(
    rows: pd.DataFrame,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    return {
        "seed": int(seed),
        "prediction_rows": int(len(rows)),
        "active_prediction_rows": int(rows["active_dka"].sum()) if not rows.empty else 0,
        "folds": int(rows["fold"].nunique()) if not rows.empty else 0,
        "validation_loss_mean": _round(rows["validation_loss"].mean()) if not rows.empty else None,
        "methods": {
            "mechanism": {
                "all_residual_states": {
                    "all_windows": _scope_summary(
                        rows,
                        "mechanism",
                        active_only=False,
                        seed=seed + 1000,
                        bootstrap_samples=bootstrap_samples,
                    ),
                    "active_dka_only": _scope_summary(
                        rows,
                        "mechanism",
                        active_only=True,
                        seed=seed + 2000,
                        bootstrap_samples=bootstrap_samples,
                    ),
                },
                "per_state": _per_state_summary(
                    rows,
                    "mechanism",
                    seed=seed + 3000,
                    bootstrap_samples=bootstrap_samples,
                ),
            },
            "greybox": {
                "all_residual_states": {
                    "all_windows": _scope_summary(
                        rows,
                        "greybox",
                        active_only=False,
                        seed=seed + 4000,
                        bootstrap_samples=bootstrap_samples,
                    ),
                    "active_dka_only": _scope_summary(
                        rows,
                        "greybox",
                        active_only=True,
                        seed=seed + 5000,
                        bootstrap_samples=bootstrap_samples,
                    ),
                },
                "per_state": _per_state_summary(
                    rows,
                    "greybox",
                    seed=seed + 6000,
                    bootstrap_samples=bootstrap_samples,
                ),
            },
        },
    }


def _multi_seed_summary(seed_reports: list[dict]) -> dict:
    def collect(method: str, scope: str) -> dict:
        results = [
            report["methods"][method]["all_residual_states"][scope][
                "delta_vs_persistence"
            ]
            for report in seed_reports
        ]
        deltas = [
            item["point_delta"] for item in results
            if item["point_delta"] is not None
        ]
        return {
            "seeds": int(len(results)),
            "beats_persistence_count": int(sum(bool(item["beats_persistence"]) for item in results)),
            "significant_count": int(sum(bool(item["significant"]) for item in results)),
            "delta_min": _round(min(deltas)) if deltas else None,
            "delta_median": _round(float(np.median(deltas))) if deltas else None,
            "delta_max": _round(max(deltas)) if deltas else None,
        }

    return {
        "mechanism": {
            "all_windows": collect("mechanism", "all_windows"),
            "active_dka_only": collect("mechanism", "active_dka_only"),
        },
        "greybox": {
            "all_windows": collect("greybox", "all_windows"),
            "active_dka_only": collect("greybox", "active_dka_only"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--output", default="mimiciv_full_v31_icd_greybox_realfit.json")
    parser.add_argument("--seeds", default="7,19,37")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--bottleneck", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [
        int(item.strip()) for item in args.seeds.split(",")
        if item.strip()
    ]
    frame = pd.read_parquet(args.cohort)
    examples = build_examples(frame)
    print(
        f"[greybox-realfit] built examples={len(examples)} "
        f"stays={len({item['stay_id'] for item in examples})}",
        file=sys.stderr,
        flush=True,
    )
    seed_reports = []
    for seed in seeds:
        print(f"[greybox-realfit] starting seed={seed}", file=sys.stderr, flush=True)
        rows = _crossfit_rows(
            examples,
            seed=seed,
            bottleneck=args.bottleneck,
            epochs=args.epochs,
            folds=args.folds,
        )
        seed_reports.append(_seed_report(
            rows,
            seed=seed,
            bootstrap_samples=args.bootstrap_samples,
        ))
        print(f"[greybox-realfit] finished seed={seed}", file=sys.stderr, flush=True)

    report = {
        "experiment": "MIMIC-IV full v3.1 DKA real-data grey-box residual crossfit",
        "cohort": str(Path(args.cohort).resolve()),
        "cohort_summary": _cohort_summary(frame, examples),
        "model": {
            "type": "constrained grey-box residual ODE",
            "training_target": (
                "residual derivative between DKABody mechanism rollout and observed "
                "6h factual outcome"
            ),
            "residual_keys": list(RESIDUAL_KEYS),
            "mechanism_owned_hard_states": [
                "total_body_potassium",
                "fluid_volume",
                "insulin_pk_depots",
                "dose_mass_balance",
                "osmotic_injury",
                "critical_burdens",
            ],
            "bottleneck": int(args.bottleneck),
            "epochs": int(args.epochs),
            "folds": int(args.folds),
        },
        "evaluation": {
            "unit_of_evidence": "ICU stay; transition rows are averaged within stay",
            "split": "GroupKFold by ICU stay; no stay appears in both train and heldout fold",
            "baseline": "persistence/current observed value",
            "delta_definition": (
                "normalized MAE(method) - normalized MAE(persistence); negative is better"
            ),
            "row_level_predictions_written": False,
            "patient_identifiers_written": False,
            "timestamp_cutoffs_written": False,
        },
        "random_patient_crossfit": {
            "seeds": seeds,
            "summary": _multi_seed_summary(seed_reports),
            "runs": seed_reports,
        },
        "reference_baselines": {
            "validated_per_target_ensemble": (
                "mimiciv_full_v31_icd_per_target_ensemble.json is the factual "
                "advisory baseline. This grey-box audit uses only residual states, "
                "so scores are not target-identical to the ensemble report."
            )
        },
        "safety_boundary": {
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
            "candidate_artifact_written": False,
        },
        "interpretation": [
            "This tests whether a small residual learned from real DKA transitions adds useful factual signal over persistence.",
            "A pass would justify a candidate-only downstream JEPA experiment, not causal or clinical promotion.",
            "A fail keeps the validated per-target ensemble as the current factual advisory artifact.",
        ],
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
