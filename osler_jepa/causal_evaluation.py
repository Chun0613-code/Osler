"""Patient-grouped retrospective treatment-effect diagnostics for DKA.

These estimators expose overlap and confounding failure modes. They are not a
substitute for randomization, complete treatment histories, or external review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class TargetTrialSpec:
    name: str
    treatment_column: str
    outcome_column: str
    baseline_outcome_column: str
    horizon_hours: float = 6.0
    treatment_grace_hours: float = 6.0


DEFAULT_TRIALS = (
    TargetTrialSpec("insulin_to_glucose", "act_insulin", "glucose_tp6", "glucose_t"),
    TargetTrialSpec(
        "insulin_to_bicarbonate", "act_insulin", "bicarbonate_tp6", "bicarbonate_t"
    ),
    TargetTrialSpec(
        "insulin_to_anion_gap", "act_insulin", "anion_gap_tp6", "anion_gap_t"
    ),
    TargetTrialSpec("fluids_to_map", "act_fluids", "map_tp6", "map_t"),
    TargetTrialSpec(
        "potassium_to_serum_potassium",
        "act_kcl", "potassium_tp6", "potassium_t",
    ),
)


def baseline_covariates(frame):
    state = [
        column for column in frame
        if column.endswith("_t") and column != "t"
        and not column.startswith("act_")
    ]
    ages = [column for column in frame if column.endswith("_age_hr")]
    history = [
        column for column in frame
        if column.startswith("hist_") and not column.endswith("_grid")
    ]
    context = [
        column for column in ("hours_since_onset", "dka_active_t")
        if column in frame
    ]
    return list(dict.fromkeys(state + ages + history + context))


def _impute_standardize(train_x, test_x):
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train_x), train_x, median)
    test = np.where(np.isfinite(test_x), test_x, median)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def _crossfit_nuisance(x, treatment, outcome, groups, folds=4):
    propensity = np.full(len(outcome), np.nan, dtype=float)
    mu0 = np.full(len(outcome), np.nan, dtype=float)
    mu1 = np.full(len(outcome), np.nan, dtype=float)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return propensity, mu0, mu1
    splitter = GroupKFold(n_splits=min(folds, len(unique_groups)))
    for train, test in splitter.split(x, treatment, groups):
        train_x, test_x = _impute_standardize(x[train], x[test])
        train_a = treatment[train]
        if len(np.unique(train_a)) < 2:
            continue
        propensity_model = LogisticRegression(
            C=0.25, max_iter=1000, class_weight="balanced"
        )
        propensity_model.fit(train_x, train_a)
        propensity[test] = propensity_model.predict_proba(test_x)[:, 1]
        for arm, destination in ((0, mu0), (1, mu1)):
            selected = train_a == arm
            if selected.sum() < 3:
                destination[test] = outcome[train].mean()
                continue
            model = Ridge(alpha=10.0)
            model.fit(train_x[selected], outcome[train][selected])
            destination[test] = model.predict(test_x)
    return propensity, mu0, mu1


def _cluster_interval(values, groups, samples=1000, seed=19):
    unique = np.unique(groups)
    if len(unique) < 2 or len(values) == 0:
        return None
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        combined = np.concatenate([by_group[group] for group in sampled])
        estimates.append(float(np.mean(combined)))
    return [
        round(float(np.quantile(estimates, 0.025)), 6),
        round(float(np.quantile(estimates, 0.975)), 6),
    ]


def _smd(x, treatment):
    treated, control = x[treatment == 1], x[treatment == 0]
    if not len(treated) or not len(control):
        return np.full(x.shape[1], np.nan)
    pooled = np.sqrt((treated.var(axis=0) + control.var(axis=0)) / 2.0)
    pooled[pooled < 1e-6] = 1.0
    return (treated.mean(axis=0) - control.mean(axis=0)) / pooled


def _matched_att(x, treatment, outcome, groups, propensity, caliper_scale=0.2):
    clipped = np.clip(propensity, 1e-4, 1.0 - 1e-4)
    logits = np.log(clipped / (1.0 - clipped))
    caliper = caliper_scale * max(float(np.std(logits)), 1e-6)
    controls = np.flatnonzero(treatment == 0)
    pairs = []
    for treated in np.flatnonzero(treatment == 1):
        eligible = controls[groups[controls] != groups[treated]]
        if not len(eligible):
            continue
        distance = np.abs(logits[eligible] - logits[treated])
        control = eligible[int(np.argmin(distance))]
        if float(np.min(distance)) <= caliper:
            pairs.append((treated, control))
    if not pairs:
        return None
    treated_index = np.asarray([pair[0] for pair in pairs])
    control_index = np.asarray([pair[1] for pair in pairs])
    effects = outcome[treated_index] - outcome[control_index]
    after_x = np.concatenate([x[treated_index], x[control_index]])
    after_a = np.concatenate([
        np.ones(len(pairs), dtype=int), np.zeros(len(pairs), dtype=int)
    ])
    return {
        "estimate": round(float(effects.mean()), 6),
        "cluster_bootstrap_95_ci": _cluster_interval(
            effects, groups[treated_index]
        ),
        "matched_pairs": int(len(pairs)),
        "unique_treated_stays": int(len(np.unique(groups[treated_index]))),
        "same_stay_matches": int(np.sum(groups[treated_index] == groups[control_index])),
        "caliper_logit": round(caliper, 6),
        "max_abs_smd_after": round(float(np.nanmax(np.abs(_smd(after_x, after_a)))), 6),
    }


def evaluate_trial(frame, spec, bootstrap_samples=1000):
    required = {
        spec.treatment_column, spec.outcome_column,
        spec.baseline_outcome_column, "stay_id",
    }
    if not required <= set(frame):
        return {"available": False, "missing_columns": sorted(required - set(frame))}
    covariates = baseline_covariates(frame)
    selected = frame[
        frame[spec.treatment_column].notna()
        & frame[spec.outcome_column].notna()
        & frame[spec.baseline_outcome_column].notna()
    ].copy()
    treatment = (selected[spec.treatment_column].to_numpy(float) > 0).astype(int)
    if len(selected) < 20 or len(np.unique(treatment)) < 2:
        return {"available": False, "reason": "insufficient treatment support"}
    x_raw = selected[covariates].to_numpy(float)
    _, x = _impute_standardize(x_raw, x_raw)
    outcome = (
        selected[spec.outcome_column].to_numpy(float)
        - selected[spec.baseline_outcome_column].to_numpy(float)
    )
    groups = selected["stay_id"].to_numpy()
    propensity, mu0, mu1 = _crossfit_nuisance(
        x_raw, treatment, outcome, groups
    )
    complete = np.isfinite(propensity) & np.isfinite(mu0) & np.isfinite(mu1)
    propensity = np.clip(propensity, 0.05, 0.95)
    overlap = complete & (propensity >= 0.10) & (propensity <= 0.90)
    pseudo = (
        mu1 - mu0
        + treatment * (outcome - mu1) / propensity
        - (1 - treatment) * (outcome - mu0) / (1.0 - propensity)
    )
    matched = _matched_att(
        x[complete], treatment[complete], outcome[complete], groups[complete],
        propensity[complete],
    ) if complete.any() else None
    other_actions = [
        column for column in selected
        if column.startswith("act_") and column != spec.treatment_column
        and not column.endswith("_total") and not column.endswith("_rate_mean")
    ]
    treated_mask = treatment == 1
    concomitant = float(
        (selected.loc[treated_mask, other_actions].fillna(0).to_numpy() > 0).any(axis=1).mean()
    ) if other_actions and treated_mask.any() else 0.0
    assignment_aligned = bool(
        "future_treatment_event_grid" in selected
        or spec.treatment_grace_hours == 0.0
    )
    readiness_failures = []
    if not assignment_aligned:
        readiness_failures.append("treatment_assignment_not_aligned_to_time_zero")
    if overlap.mean() < 0.80:
        readiness_failures.append("insufficient_propensity_overlap")
    if concomitant > 0.50:
        readiness_failures.append("high_concomitant_treatment_rate")
    if matched is None or matched["max_abs_smd_after"] > 0.20:
        readiness_failures.append("matched_covariate_balance_inadequate")
    if int(selected["stay_id"].nunique()) < 30:
        readiness_failures.append("too_few_patient_stays")
    return {
        "available": bool(overlap.sum() >= 10),
        "target_trial": {
            **asdict(spec),
            "time_zero": "anchor state at t",
            "eligibility": "observed baseline and six-hour outcome",
            "treatment_strategy": "action recorded during treatment grace window",
            "outcome": "six-hour change from baseline",
            "assignment_is_time_aligned": assignment_aligned,
        },
        "support": {
            "rows": int(len(selected)),
            "stays": int(selected["stay_id"].nunique()),
            "treated_rows": int(treatment.sum()),
            "control_rows": int((1 - treatment).sum()),
            "overlap_rows": int(overlap.sum()),
            "overlap_fraction": round(float(overlap.mean()), 6),
            "propensity_range": [
                round(float(np.nanmin(propensity)), 6),
                round(float(np.nanmax(propensity)), 6),
            ],
            "concomitant_treatment_rate_among_treated": round(concomitant, 6),
            "max_abs_smd_before": round(float(np.nanmax(np.abs(_smd(x, treatment)))), 6),
        },
        "aipw_ate": {
            "estimate": round(float(np.mean(pseudo[overlap])), 6)
            if overlap.any() else None,
            "cluster_bootstrap_95_ci": _cluster_interval(
                pseudo[overlap], groups[overlap], bootstrap_samples
            ) if overlap.any() else None,
        },
        "propensity_matched_att": matched,
        "promotion_readiness": {
            "passes": bool(not readiness_failures),
            "failures": readiness_failures,
        },
        "causal_claim_allowed": False,
        "limitations": [
            "Treatment is observational and may have unmeasured confounding.",
            "Concurrent treatments make action-specific effects difficult to identify.",
            "A treatment grace window longer than zero can introduce time-alignment bias.",
            "Confidence intervals reflect sampling variability, not structural bias.",
        ],
    }


def evaluate_target_trials(frame, specs=DEFAULT_TRIALS, bootstrap_samples=1000):
    return {
        "evaluation": "patient-stay grouped target-trial diagnostics",
        "trials": {
            spec.name: evaluate_trial(frame, spec, bootstrap_samples)
            for spec in specs
        },
        "causal_claim_allowed": False,
    }
