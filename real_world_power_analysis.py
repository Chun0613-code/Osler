"""Patient-stay power analysis for DKA JEPA promotion gates.

The unit of evidence is the ICU stay, not the transition row.  A method must
show a negative paired error delta versus persistence on held-out stays before
more power can make it promotable.  If the observed delta is positive, the
candidate is currently worse than persistence and the required-N calculation is
intentionally withheld.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from dka_world_model import S_STD
from osler_jepa.greybox_residual import GreyBoxResidualRuntime, RESIDUAL_KEYS
from train_greybox_residual import build_examples, fit_network, rollout


Z_ALPHA_TWO_SIDED_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
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


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _current(row, state):
    value = _finite(row.get(CURRENT_COLUMNS[state]))
    if state == "Ket" and np.isfinite(value):
        value = max(0.0, value - 12.0)
    return value


def bootstrap_ci(values, seed=17, samples=5000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(sample.mean()))
    return [
        round(float(np.quantile(means, 0.025)), 6),
        round(float(np.quantile(means, 0.975)), 6),
    ]


def required_stays_for_win(mean_delta, sd_delta, alpha=Z_ALPHA_TWO_SIDED_95,
                           power=Z_POWER_80):
    """Return paired-stay N needed to show method MAE < persistence MAE."""
    if mean_delta >= 0:
        return None
    if sd_delta <= 1e-12:
        return 3
    effect = abs(mean_delta)
    return int(math.ceil(((alpha + power) * sd_delta / effect) ** 2))


def summarize_delta(values, current_stays):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ci = bootstrap_ci(values)
    required = required_stays_for_win(mean, sd)
    if mean >= 0:
        interpretation = (
            "observed_candidate_not_better_than_persistence; increasing N alone "
            "does not promote this candidate"
        )
    elif ci and ci[1] < 0:
        interpretation = "current_demo_ci_already_excludes_zero"
    else:
        interpretation = "underpowered_for_current_observed_effect"
    return {
        "stay_count": int(current_stays),
        "mean_method_minus_persistence": round(mean, 6),
        "sd_stay_delta": round(sd, 6),
        "bootstrap_95_ci": ci,
        "required_stays_for_95ci_80power": required,
        "additional_stays_needed_from_current_demo": (
            max(0, required - int(current_stays)) if required is not None else None
        ),
        "interpretation": interpretation,
    }


def stay_level_method_deltas(frame, method, baseline="persistence",
                             states=None, active_only=False):
    selected = frame.copy()
    if active_only and "active_dka" in selected.columns:
        selected = selected[selected["active_dka"].astype(bool)]
    if states is not None:
        selected = selected[selected["state"].isin(states)]
    if selected.empty:
        return np.asarray([]), 0
    selected = selected[np.isfinite(selected["truth"])]
    selected = selected[np.isfinite(selected[method])]
    selected = selected[np.isfinite(selected[baseline])]
    if selected.empty:
        return np.asarray([]), 0
    selected = selected.assign(
        method_error=(selected[method] - selected["truth"]).abs() / selected["scale"],
        baseline_error=(selected[baseline] - selected["truth"]).abs() / selected["scale"],
    )
    grouped = selected.groupby("stay_id")[["method_error", "baseline_error"]].mean()
    return (grouped["method_error"] - grouped["baseline_error"]).to_numpy(), len(grouped)


def summarize_oof_predictions(path):
    frame = pd.read_csv(path)
    methods = [
        "base_jepa", "ensemble_adapter", "safe_hybrid", "active_safe_hybrid",
    ]
    dense_states = ["G", "Na", "creatinine"]
    report = {
        "source": Path(path).name,
        "stay_count": int(frame["stay_id"].nunique()),
        "methods": {},
    }
    for method in methods:
        if method not in frame.columns:
            continue
        method_report = {"all_states": {}, "dense_core": {}}
        for active_only in (False, True):
            label = "active_dka_only" if active_only else "all_windows"
            deltas, stays = stay_level_method_deltas(
                frame, method, active_only=active_only
            )
            method_report["all_states"][label] = summarize_delta(deltas, stays)
            dense, dense_stays = stay_level_method_deltas(
                frame, method, states=dense_states, active_only=active_only
            )
            method_report["dense_core"][label] = summarize_delta(dense, dense_stays)
        per_state = {}
        for state in sorted(frame["state"].unique()):
            deltas, stays = stay_level_method_deltas(frame, method, states=[state])
            per_state[state] = summarize_delta(deltas, stays)
        method_report["per_state"] = per_state
        report["methods"][method] = method_report
    return report


def greybox_oof_rows(cohort, seed=7, bottleneck=8):
    frame = pd.read_parquet(cohort)
    examples = build_examples(frame)
    groups = np.asarray([item["stay_id"] for item in examples])
    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    rows = []
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(examples)), groups=groups)):
        inner_groups = groups[train]
        inner_splitter = GroupKFold(n_splits=min(3, len(np.unique(inner_groups))))
        inner_train_local, inner_validation_local = next(
            inner_splitter.split(train, groups=inner_groups)
        )
        inner_train = train[inner_train_local]
        inner_validation = train[inner_validation_local]
        model, mean, std, _ = fit_network(
            examples, inner_train, validation_indices=inner_validation,
            seed=seed + fold, bottleneck=bottleneck,
        )
        runtime = GreyBoxResidualRuntime(model, mean, std)
        for index in test:
            item = examples[index]
            _, prediction, _ = rollout(item["row"], runtime)
            row = item["row"]
            for state_index, state in enumerate(RESIDUAL_KEYS):
                if not item["mask"][state_index]:
                    continue
                current = _current(row, state)
                if not np.isfinite(current):
                    continue
                rows.append({
                    "stay_id": int(item["stay_id"]),
                    "state": state,
                    "truth": float(item["target"][state_index]),
                    "current": float(current),
                    "scale": SCALE_BY_STATE[state],
                    "persistence": float(current),
                    "mechanism": float(item["mechanism"][state_index]),
                    "greybox": float(prediction[state_index]),
                })
    return pd.DataFrame(rows)


def summarize_greybox(cohort, seed=7, bottleneck=8, predictions_output=None):
    rows = greybox_oof_rows(cohort, seed=seed, bottleneck=bottleneck)
    if predictions_output:
        rows.to_csv(predictions_output, index=False)
    report = {
        "source": Path(cohort).name,
        "stay_count": int(rows["stay_id"].nunique()) if not rows.empty else 0,
        "methods": {},
    }
    for method in ("mechanism", "greybox"):
        method_report = {"all_residual_states": {}, "per_state": {}}
        deltas, stays = stay_level_method_deltas(rows, method)
        method_report["all_residual_states"]["all_windows"] = summarize_delta(
            deltas, stays
        )
        for state in RESIDUAL_KEYS:
            deltas, stays = stay_level_method_deltas(rows, method, states=[state])
            method_report["per_state"][state] = summarize_delta(deltas, stays)
        report["methods"][method] = method_report
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-predictions", default="dka_real_world_oof_predictions_v2.csv")
    parser.add_argument("--cohort", default="dka_transitions_6h_demo_v4.parquet")
    parser.add_argument("--greybox-predictions", default="dka_greybox_oof_predictions_v1.csv")
    parser.add_argument("--output", default="dka_persistence_power_analysis_v1.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bottleneck", type=int, default=8)
    args = parser.parse_args()

    result = {
        "analysis": "patient-held-out power analysis versus persistence",
        "unit_of_evidence": "ICU stay; transition rows are averaged within stay",
        "alpha": 0.05,
        "target_power": 0.80,
        "delta_definition": "normalized MAE(method) - normalized MAE(persistence); negative is better",
        "factual_forecast_promotion_allowed": False,
        "causal_claim_allowed": False,
        "oof_jepa_adapter": summarize_oof_predictions(args.oof_predictions),
        "greybox_residual": summarize_greybox(
            args.cohort, seed=args.seed, bottleneck=args.bottleneck,
            predictions_output=args.greybox_predictions,
        ),
        "conclusion": (
            "Power can only rescue candidates with a negative stay-level delta. "
            "For dense glucose/sodium/creatinine targets in this demo cohort, "
            "the dominant result remains either wrong-signed or too uncertain; "
            "promotion requires a causal-grade, larger held-out cohort."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
