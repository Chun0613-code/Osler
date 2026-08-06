"""Patient-held-out per-target ensemble for eICU DKA factual proxy tests.

This script compares two existing DKA JEPA checkpoints on the same
observed-treatment cohort, learns a per-target selector on discovery patient
stays, and evaluates the selector on held-out patient stays.

It is a factual observed-treatment proxy only. It does not estimate causal
treatment effects and does not promote either checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dka_action_contract import expand_action
from dka_body import henderson
from dka_world_model import (
    A_DIM,
    ACTION_KEYS,
    DT,
    HISTORY_HOURS,
    S_MEAN,
    S_STD,
    a2vec,
    load_checkpoint,
    observation_context,
    s2vec,
    treatment_history_features,
)
from osler_jepa.actions import treatment_event_features
from train_intervention_jepa import choose_device


TARGET_MAP = {
    "glucose": ("glucose_t", "glucose_tp6", 0),
    "ph": ("ph_t", "ph_tp6", 1),
    "bicarbonate": ("bicarbonate_t", "bicarbonate_tp6", 2),
    "anion_gap": ("anion_gap_t", "anion_gap_tp6", 3),
    "potassium": ("potassium_t", "potassium_tp6", 4),
    "map": ("map_t", "map_tp6", 5),
    "sodium": ("sodium_t", "sodium_tp6", 8),
    "osmolality": ("osmolality_t", "osmolality_tp6", 9),
    "creatinine": ("creatinine_t", "creatinine_tp6", 10),
    "urine_output": ("urine_output_t", "urine_output_tp6", 11),
    "BHB": ("BHB_t", "BHB_tp6", 12),
}
METHODS = ("persistence", "v5", "presentation_only")


def _missing(value):
    return value is None or pd.isna(value)


def _float(row, name, default=np.nan):
    value = row.get(name, default)
    return default if _missing(value) else float(value)


def _core_state(row):
    glucose = _float(row, "glucose_t")
    potassium = _float(row, "potassium_t")
    map_value = _float(row, "map_t")
    bicarbonate = _float(row, "bicarbonate_t")
    if any(np.isnan(value) for value in (glucose, potassium, map_value, bicarbonate)):
        return None

    ph = _float(row, "ph_t", henderson(bicarbonate))
    anion_gap = _float(row, "anion_gap_t", float(S_MEAN[3]))
    sodium = _float(row, "sodium_t", 138.0)
    creatinine = _float(row, "creatinine_t", 1.2)
    osmolality = _float(row, "osmolality_t")
    if np.isnan(osmolality):
        osmolality = _float(row, "osmolality_derived_t")
    if np.isnan(osmolality):
        osmolality = 2.0 * sodium + glucose / 18.0
    urine_output = _float(row, "urine_output_t", 100.0)
    bhb = _float(row, "BHB_t", max(0.0, anion_gap - 12.0) * 0.75)

    return {
        "G": glucose,
        "pH": ph,
        "HCO3": bicarbonate,
        "anion_gap": anion_gap,
        "Ke": potassium,
        "MAP": map_value,
        "V": float(np.clip((map_value - 30.0) / 60.0 * 15.0, 5.0, 18.0)),
        "I": 1.0,
        "Na": sodium,
        "osmolality": osmolality,
        "creatinine": creatinine,
        "urine_output": urine_output,
        "BHB": bhb,
        "K_store": float(np.clip(
            120.0 - 45.0 * max(0.0, 7.35 - ph)
            - 8.0 * max(0.0, creatinine - 1.2),
            55.0,
            145.0,
        )),
        "osmotic_injury": 0.0,
    }


def _observation_context(row, frame, state):
    reported = {
        "G": _float(row, "glucose_t"),
        "HCO3": _float(row, "bicarbonate_t"),
        "Ke": _float(row, "potassium_t"),
        "MAP": _float(row, "map_t"),
    }
    observed_fields = (
        ("pH", "ph_t"),
        ("anion_gap", "anion_gap_t"),
        ("Na", "sodium_t"),
        ("osmolality", "osmolality_t"),
        ("creatinine", "creatinine_t"),
        ("urine_output", "urine_output_t"),
        ("BHB", "BHB_t"),
    )
    for model_name, column in observed_fields:
        value = row.get(column)
        if not _missing(value):
            reported[model_name] = float(value)

    age_map = {
        model_name: float(row[column])
        for model_name, column in (
            ("G", "glucose_age_hr"),
            ("pH", "ph_age_hr"),
            ("HCO3", "bicarbonate_age_hr"),
            ("anion_gap", "anion_gap_age_hr"),
            ("Ke", "potassium_age_hr"),
            ("MAP", "map_age_hr"),
            ("Na", "sodium_age_hr"),
            ("osmolality", "osmolality_age_hr"),
            ("creatinine", "creatinine_age_hr"),
            ("urine_output", "urine_output_age_hr"),
            ("BHB", "BHB_age_hr"),
        )
        if column in frame and not _missing(row.get(column))
    }
    return observation_context(reported, age_map)[1:]


def _action_context(row, frame):
    detailed = "future_action_grid" in frame.columns
    if detailed and isinstance(row.get("future_action_grid"), str):
        physical_sequence = np.asarray(
            json.loads(row["future_action_grid"]), dtype=np.float32
        )
        if physical_sequence.ndim != 2 or physical_sequence.shape[1] != A_DIM:
            return None
        sequence = np.asarray([a2vec(action) for action in physical_sequence])
        if isinstance(row.get("future_treatment_event_grid"), str):
            event_sequence = np.asarray(
                json.loads(row["future_treatment_event_grid"]), dtype=np.float32
            )
        else:
            event_sequence = treatment_event_features(physical_sequence)
    else:
        action = np.array([
            4.0 if row.get("act_insulin", 0) else 0.0,
            250.0 if row.get("act_fluids", 0) else 0.0,
            10.0 if (
                row.get("act_kcl", 0) or row.get("act_potassium", 0)
            ) else 0.0,
            25.0 if row.get("act_bicarbonate", 0) else 0.0,
            5.0 if row.get("act_dextrose", 0) else 0.0,
        ], dtype=np.float32)
        physical_sequence = np.repeat(expand_action(action)[None, :], 12, axis=0)
        sequence = np.repeat(a2vec(action)[None, :], 12, axis=0)
        event_sequence = treatment_event_features(physical_sequence)

    history = None
    if "history_action_grid" in frame.columns and isinstance(
        row.get("history_action_grid"), str
    ):
        history_grid = np.asarray(
            json.loads(row["history_action_grid"]), dtype=np.float32
        )
        history = treatment_history_features(history_grid, DT, HISTORY_HOURS)
    return sequence, event_sequence, history


@torch.no_grad()
def _predict_physical(model, row, frame, device):
    state = _core_state(row)
    action_context = _action_context(row, frame)
    if state is None or action_context is None:
        return None, None
    observed_mask, observed_age = _observation_context(row, frame, state)
    sequence, event_sequence, history = action_context
    predicted, _, _ = model.rollout(
        torch.as_tensor(s2vec(state), dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0),
        None if history is None else torch.as_tensor(
            history, dtype=torch.float32, device=device
        ).unsqueeze(0),
        observation_mask=torch.as_tensor(
            observed_mask, dtype=torch.float32, device=device
        ).unsqueeze(0),
        observation_age=torch.as_tensor(
            observed_age, dtype=torch.float32, device=device
        ).unsqueeze(0),
        treatment_events=torch.as_tensor(
            event_sequence, dtype=torch.float32, device=device
        ).unsqueeze(0),
    )
    return predicted[0, -1].cpu().numpy() * S_STD + S_MEAN, state


def _active_dka(row, state):
    value = row.get("dka_active_t", np.nan)
    if not _missing(value):
        return bool(value)
    return bool(
        (
            state["G"] >= 200.0
            and (state["HCO3"] < 18.0 or state["anion_gap"] > 12.0)
        )
        or state["BHB"] >= 3.0
    )


def build_long_predictions(frame, models, device):
    rows = []
    for _, row in frame.iterrows():
        predictions = {}
        state = None
        for name, model in models.items():
            physical, state_for_row = _predict_physical(model, row, frame, device)
            if physical is None:
                predictions = {}
                break
            predictions[name] = physical
            state = state_for_row
        if not predictions:
            continue

        active = _active_dka(row, state)
        for target, (current_column, target_column, index) in TARGET_MAP.items():
            current = _float(row, current_column)
            truth = _float(row, target_column)
            if target == "osmolality":
                if np.isnan(current):
                    current = _float(row, "osmolality_derived_t")
                if np.isnan(truth):
                    truth = _float(row, "osmolality_derived_tp6")
            if np.isnan(current) or np.isnan(truth):
                continue
            entry = {
                "stay_id": int(row["stay_id"]),
                "target": target,
                "target_index": index,
                "active_dka": active,
                "truth": truth,
                "current": current,
                "scale": float(S_STD[index]),
                "persistence": current,
            }
            for name, physical in predictions.items():
                entry[name] = float(physical[index])
            rows.append(entry)
    return pd.DataFrame(rows)


def split_stays(rows, seed, discovery_fraction):
    stays = np.asarray(sorted(rows["stay_id"].dropna().unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(stays)
    split = max(1, int(round(discovery_fraction * len(stays))))
    split = min(split, max(1, len(stays) - 1))
    return set(stays[:split].tolist()), set(stays[split:].tolist())


def _scope(rows, active_only):
    return rows[rows["active_dka"].astype(bool)] if active_only else rows


def mae(rows, method, normalized=False):
    if rows.empty:
        return None
    errors = np.abs(rows[method].astype(float) - rows["truth"].astype(float))
    if normalized:
        errors = errors / rows["scale"].astype(float)
    return float(errors.mean())


def method_for_rows(rows, selector):
    return rows["target"].map(selector).fillna("persistence")


def ensemble_errors(rows, selector, normalized=False):
    if rows.empty:
        return np.asarray([], dtype=np.float64)
    selected = method_for_rows(rows, selector).to_numpy()
    prediction = np.asarray([
        float(row[method]) for method, (_, row) in zip(selected, rows.iterrows())
    ])
    errors = np.abs(prediction - rows["truth"].to_numpy(dtype=np.float64))
    if normalized:
        errors = errors / rows["scale"].to_numpy(dtype=np.float64)
    return errors


def ensemble_mae(rows, selector, normalized=False):
    errors = ensemble_errors(rows, selector, normalized)
    return float(errors.mean()) if len(errors) else None


def summarize_fixed_methods(rows, normalized=False):
    return {
        method: round(mae(rows, method, normalized), 6)
        if mae(rows, method, normalized) is not None else None
        for method in METHODS
    }


def choose_selector(discovery, active_only, min_pairs, min_stays):
    selected = {}
    details = {}
    scoped = _scope(discovery, active_only)
    for target in TARGET_MAP:
        target_rows = scoped[scoped["target"] == target]
        n = int(len(target_rows))
        stays = int(target_rows["stay_id"].nunique()) if n else 0
        method_mae = summarize_fixed_methods(target_rows)
        if n < min_pairs or stays < min_stays:
            selected[target] = "persistence"
            reason = "insufficient_discovery_support"
        else:
            finite = {
                method: value for method, value in method_mae.items()
                if value is not None and np.isfinite(value)
            }
            selected[target] = min(finite, key=finite.get) if finite else "persistence"
            reason = "lowest_discovery_mae"
        details[target] = {
            "n": n,
            "stays": stays,
            "mae": method_mae,
            "selected_method": selected[target],
            "reason": reason,
            "selected_beats_persistence": bool(
                method_mae.get(selected[target]) is not None
                and method_mae.get("persistence") is not None
                and method_mae[selected[target]] < method_mae["persistence"]
            ),
        }
    return selected, details


def stay_level_delta(rows, selector, baseline="persistence", normalized=True):
    if rows.empty:
        return pd.Series(dtype="float64")
    selected = method_for_rows(rows, selector).to_numpy()
    predicted = np.asarray([
        float(row[method]) for method, (_, row) in zip(selected, rows.iterrows())
    ])
    truth = rows["truth"].to_numpy(dtype=np.float64)
    baseline_prediction = rows[baseline].to_numpy(dtype=np.float64)
    scale = rows["scale"].to_numpy(dtype=np.float64) if normalized else 1.0
    delta = (np.abs(predicted - truth) - np.abs(baseline_prediction - truth)) / scale
    temp = rows[["stay_id"]].copy()
    temp["delta"] = delta
    return temp.groupby("stay_id")["delta"].mean()


def bootstrap_delta(rows, selector, seed, samples, normalized=True):
    deltas = stay_level_delta(rows, selector, normalized=normalized)
    if deltas.empty:
        return {
            "stays": 0,
            "point_delta": None,
            "bootstrap_95_ci": [None, None],
            "beats_persistence": None,
            "significant": False,
        }
    values = deltas.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        boot.append(float(rng.choice(values, size=len(values), replace=True).mean()))
    ci = np.quantile(boot, [0.025, 0.975])
    point = float(values.mean())
    return {
        "stays": int(len(values)),
        "point_delta": round(point, 6),
        "bootstrap_95_ci": [round(float(ci[0]), 6), round(float(ci[1]), 6)],
        "beats_persistence": bool(point < 0.0),
        "significant": bool(ci[1] < 0.0),
    }


def summarize_scope(rows, selector, active_only, seed, samples):
    scoped = _scope(rows, active_only)
    fixed = summarize_fixed_methods(scoped, normalized=True)
    fixed["per_target_ensemble"] = (
        round(ensemble_mae(scoped, selector, normalized=True), 6)
        if not scoped.empty else None
    )
    per_target = {}
    for index, target in enumerate(TARGET_MAP):
        target_rows = scoped[scoped["target"] == target]
        target_selector = {target: selector.get(target, "persistence")}
        target_mae = summarize_fixed_methods(target_rows, normalized=False)
        target_mae["per_target_ensemble"] = (
            round(ensemble_mae(target_rows, target_selector), 6)
            if not target_rows.empty else None
        )
        per_target[target] = {
            "n": int(len(target_rows)),
            "stays": int(target_rows["stay_id"].nunique()) if len(target_rows) else 0,
            "selected_method": selector.get(target, "persistence"),
            "mae": target_mae,
            "ensemble_delta_vs_persistence": bootstrap_delta(
                target_rows,
                target_selector,
                seed + 101 + index,
                samples,
                normalized=False,
            ),
        }
    return {
        "n_rows": int(len(scoped)),
        "stays": int(scoped["stay_id"].nunique()) if len(scoped) else 0,
        "normalized_mae": fixed,
        "ensemble_delta_vs_persistence": bootstrap_delta(
            scoped, selector, seed, samples, normalized=True
        ),
        "per_target": per_target,
    }


def oracle_selector(rows, active_only, min_pairs, min_stays):
    scoped = _scope(rows, active_only)
    selector = {}
    for target in TARGET_MAP:
        target_rows = scoped[scoped["target"] == target]
        if len(target_rows) < min_pairs or target_rows["stay_id"].nunique() < min_stays:
            selector[target] = "persistence"
            continue
        scores = summarize_fixed_methods(target_rows)
        finite = {
            method: value for method, value in scores.items()
            if value is not None and np.isfinite(value)
        }
        selector[target] = min(finite, key=finite.get) if finite else "persistence"
    return selector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="eicu_dka_transitions_6h_demo.parquet")
    parser.add_argument("--v5-checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument(
        "--presentation-checkpoint",
        default="dka_physionet_presentation_only_candidate.pt",
    )
    parser.add_argument("--output", default="eicu_dka_per_target_ensemble.json")
    parser.add_argument("--predictions-csv")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--discovery-fraction", type=float, default=0.67)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-stays", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    v5, _ = load_checkpoint(args.v5_checkpoint, device)
    presentation, _ = load_checkpoint(args.presentation_checkpoint, device)
    frame = pd.read_parquet(args.cohort)
    rows = build_long_predictions(
        frame,
        {"v5": v5, "presentation_only": presentation},
        device,
    )
    if args.predictions_csv:
        rows.to_csv(args.predictions_csv, index=False)

    discovery_stays, heldout_stays = split_stays(
        rows, args.seed, args.discovery_fraction
    )
    discovery = rows[rows["stay_id"].isin(discovery_stays)].copy()
    heldout = rows[rows["stay_id"].isin(heldout_stays)].copy()

    selector, selector_details = choose_selector(
        discovery,
        active_only=True,
        min_pairs=args.min_pairs,
        min_stays=args.min_stays,
    )
    oracle = oracle_selector(
        heldout,
        active_only=True,
        min_pairs=args.min_pairs,
        min_stays=args.min_stays,
    )
    report = {
        "experiment": "eicu_patient_heldout_per_target_ensemble",
        "cohort": str(Path(args.cohort).resolve()),
        "models": {
            "v5": str(Path(args.v5_checkpoint).resolve()),
            "presentation_only": str(Path(args.presentation_checkpoint).resolve()),
        },
        "split": {
            "seed": args.seed,
            "discovery_fraction": args.discovery_fraction,
            "discovery_stays": len(discovery_stays),
            "heldout_stays": len(heldout_stays),
            "overlap": len(discovery_stays & heldout_stays),
            "prediction_rows": int(len(rows)),
            "active_prediction_rows": int(rows["active_dka"].sum()),
        },
        "selector_training_scope": "active_dka_only",
        "selector_gate": {
            "min_pairs": args.min_pairs,
            "min_stays": args.min_stays,
            "rule": (
                "Choose the lowest-MAE method on discovery active-DKA stays; "
                "otherwise fall back to persistence."
            ),
        },
        "discovery_selector": selector_details,
        "selected_methods": selector,
        "heldout": {
            "all_windows": summarize_scope(
                heldout, selector, False, args.seed + 1000, args.bootstrap_samples
            ),
            "active_dka_only": summarize_scope(
                heldout, selector, True, args.seed + 2000, args.bootstrap_samples
            ),
        },
        "oracle_upper_bound_heldout": {
            "warning": (
                "This selector is fit on held-out outcomes and is leakage; use only "
                "as an upper-bound sanity check."
            ),
            "selected_methods": oracle,
            "active_dka_only": summarize_scope(
                heldout, oracle, True, args.seed + 3000, args.bootstrap_samples
            ),
        },
        "decision": "no_checkpoint_promotion_research_proxy_only",
        "safety_boundary": {
            "factual_observed_treatment_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "checkpoint_promotion_allowed": False,
            "row_level_predictions_written": bool(args.predictions_csv),
            "row_level_predictions_versioned": False,
        },
        "interpretation": [
            "The primary result is the discovery-selected held-out ensemble, not the oracle upper bound.",
            "The selector is target-wise and patient-held-out, so target outcomes from held-out stays do not choose methods.",
            "Negative bootstrap deltas mean the ensemble beats persistence; intervals crossing zero are not significant.",
            "This remains observational and treatment-confounded even when it beats persistence factually.",
        ],
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
