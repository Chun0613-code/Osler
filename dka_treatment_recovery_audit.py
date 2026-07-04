"""Falsify DKABody treatment-recovery dynamics against observed cohort windows.

The audit keeps two judges separate:

1. Fixed clinical priors define broad physiologic boundaries.
2. Observed ICU cohorts provide external direction and magnitude checks.

The script never fits simulator parameters. A runtime change is justified only
when a sourced boundary and external falsification identify the same mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dka_action_contract import ACTION_KEYS
from dka_body import DKABody
from dka_fidelity_replay import DT, init_body


TARGETS = {
    "glucose": ("G", "glucose_t", "glucose_tp6"),
    "bicarbonate": ("HCO3", "bicarbonate_t", "bicarbonate_tp6"),
    "ph": ("pH", "ph_t", "ph_tp6"),
    "anion_gap": ("anion_gap", "anion_gap_t", "anion_gap_tp6"),
    "potassium": ("Ke", "potassium_t", "potassium_tp6"),
    "map": ("MAP", "map_t", "map_tp6"),
}

CONSENSUS_CITATION = (
    "Umpierrez GE et al. Hyperglycemic Crises in Adults With Diabetes: "
    "A Consensus Report. Diabetes Care. 2024;47(8):1257-1275."
)
CONSENSUS_URL = "https://doi.org/10.2337/dci24-0032"

SOURCE_PRIORS = {
    "initial_insulin_glucose_decline_mg_dl_hr": {
        "lower": 50.0,
        "upper": 75.0,
        "applies_when": (
            "active DKA, glucose >=250 mg/dL, insulin delivered, no dextrose, "
            "and glucose remains above the dextrose-transition threshold"
        ),
        "source": CONSENSUS_CITATION,
        "source_url": CONSENSUS_URL,
        "role": "broad protocol-response boundary, not patient-specific target",
    },
    "dka_resolution": {
        "beta_hydroxybutyrate_max_mmol_l": 0.6,
        "venous_ph_min": 7.3,
        "bicarbonate_min_mmol_l": 18.0,
        "source": CONSENSUS_CITATION,
        "source_url": CONSENSUS_URL,
        "role": "resolution boundary, not a fixed hourly acid-base slope",
    },
    "potassium_safety": {
        "hold_insulin_below_mmol_l": 3.5,
        "source": CONSENSUS_CITATION,
        "source_url": CONSENSUS_URL,
        "role": "safety boundary, not a fitted potassium trajectory",
    },
}


def parse_grid(value) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D action grid, got {array.shape}")
    if array.shape[1] < len(ACTION_KEYS):
        padding = np.zeros((array.shape[0], len(ACTION_KEYS) - array.shape[1]))
        array = np.concatenate([array, padding], axis=1)
    return array[:, :len(ACTION_KEYS)]


def grid_actions(grid: np.ndarray, start_hour: float = 0.0):
    actions = []
    for index, values in enumerate(grid):
        action = {
            key: float(value)
            for key, value in zip(ACTION_KEYS, values)
            if np.isfinite(value) and float(value) > 0.0
        }
        action["t"] = start_hour + index * DT
        actions.append(action)
    return actions


def initial_state(row: pd.Series):
    mapping = {
        "glucose": "glucose_t",
        "HCO3": "bicarbonate_t",
        "K": "potassium_t",
        "anion_gap": "anion_gap_t",
        "MAP": "map_t",
        "Na": "sodium_t",
        "creatinine": "creatinine_t",
        "urine_output": "urine_output_t",
        "BHB": "BHB_t",
    }
    output = {}
    for name, column in mapping.items():
        value = row.get(column, np.nan)
        if pd.notna(value) and np.isfinite(float(value)):
            output[name] = float(value)
    return output


def replay_row(row: pd.Series):
    history_grid = parse_grid(row["history_action_grid"])
    future_grid = parse_grid(row["future_action_grid"])
    body = init_body(
        initial_state(row),
        grid_actions(history_grid, -len(history_grid) * DT),
    )
    start = body.observe()
    first_hour = None
    for index, values in enumerate(future_grid):
        action = {
            key: float(value)
            for key, value in zip(ACTION_KEYS, values)
            if np.isfinite(value) and float(value) > 0.0
        }
        body.step(action, dt=DT)
        if index == 1:
            first_hour = body.observe()
        if not body.alive:
            break
    if first_hour is None:
        first_hour = body.observe()
    end = body.observe()
    result = {
        "sim_alive": bool(body.alive),
        "sim_death_cause": body.death_cause,
        "insulin_total": float(future_grid[:, :4].sum() * DT),
        "fluids_total": float(future_grid[:, 4].sum() * DT),
        "kcl_total": float(future_grid[:, 5].sum() * DT),
        "bicarbonate_total": float(future_grid[:, 6].sum() * DT),
        "dextrose_total": float(future_grid[:, 7].sum() * DT),
        "insulin_evidence": bool(row.get(
            "act_insulin_evidence", row.get("act_insulin_total", 0.0) > 0.0
        )),
        "insulin_evidence_only": bool(row.get(
            "act_insulin_evidence_only", False
        )),
        "first_hour_iv_rate_mean": float(future_grid[:2, 0].mean()),
        "first_hour_insulin_total": float(future_grid[:2, :4].sum() * DT),
        "first_hour_dextrose_total": float(future_grid[:2, 7].sum() * DT),
        "glucose_sim_1h": float(first_hour["G"]),
    }
    for target, (state_key, current_column, future_column) in TARGETS.items():
        current = row.get(current_column, np.nan)
        future = row.get(future_column, np.nan)
        result[f"{target}_sim_start"] = float(start[state_key])
        result[f"{target}_sim_end"] = float(end[state_key])
        result[f"{target}_real_start"] = (
            float(current) if pd.notna(current) else np.nan
        )
        result[f"{target}_real_end"] = (
            float(future) if pd.notna(future) else np.nan
        )
    return result


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        name: round(float(np.quantile(array, probability)), 6)
        for name, probability in (
            ("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
            ("p75", 0.75), ("p90", 0.90),
        )
    }


def target_summary(frame: pd.DataFrame, target: str):
    current = frame[f"{target}_real_start"].to_numpy(dtype=np.float64)
    real = frame[f"{target}_real_end"].to_numpy(dtype=np.float64)
    sim = frame[f"{target}_sim_end"].to_numpy(dtype=np.float64)
    mask = np.isfinite(current) & np.isfinite(real) & np.isfinite(sim)
    current, real, sim = current[mask], real[mask], sim[mask]
    if len(real) == 0:
        return {"n": 0}
    real_rate = (real - current) / 6.0
    sim_rate = (sim - current) / 6.0
    return {
        "n": int(len(real)),
        "stays": int(frame.loc[mask, "stay_id"].nunique()),
        "real_delta_per_hour": {
            "mean": round(float(real_rate.mean()), 6),
            "quantiles": quantiles(real_rate),
        },
        "sim_delta_per_hour": {
            "mean": round(float(sim_rate.mean()), 6),
            "quantiles": quantiles(sim_rate),
        },
        "simulator_mae": round(float(np.mean(np.abs(sim - real))), 6),
        "persistence_mae": round(float(np.mean(np.abs(current - real))), 6),
        "direction_agreement": round(
            float(np.mean(np.sign(sim - current) == np.sign(real - current))), 6
        ),
        "median_absolute_rate_gap": round(
            float(np.median(np.abs(sim_rate - real_rate))), 6
        ),
    }


def clustered_bootstrap_delta(frame, target, samples, seed):
    columns = [
        "stay_id", f"{target}_real_start", f"{target}_real_end",
        f"{target}_sim_end",
    ]
    valid = frame[columns].dropna()
    if valid.empty:
        return {"n_stays": 0, "delta_mae_sim_minus_persistence": None, "ci95": None}
    by_stay = {}
    for stay_id, group in valid.groupby("stay_id"):
        sim_error = np.abs(
            group[f"{target}_sim_end"].to_numpy()
            - group[f"{target}_real_end"].to_numpy()
        )
        persistence_error = np.abs(
            group[f"{target}_real_start"].to_numpy()
            - group[f"{target}_real_end"].to_numpy()
        )
        by_stay[int(stay_id)] = float(np.mean(sim_error - persistence_error))
    stays = np.asarray(list(by_stay), dtype=np.int64)
    point = float(np.mean([by_stay[int(stay)] for stay in stays]))
    if len(stays) < 2:
        return {
            "n_stays": int(len(stays)),
            "delta_mae_sim_minus_persistence": round(point, 6),
            "ci95": None,
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(stays, size=len(stays), replace=True)
        estimates[index] = np.mean([by_stay[int(stay)] for stay in selected])
    return {
        "n_stays": int(len(stays)),
        "delta_mae_sim_minus_persistence": round(point, 6),
        "ci95": [
            round(float(np.quantile(estimates, 0.025)), 6),
            round(float(np.quantile(estimates, 0.975)), 6),
        ],
        "interpretation": "negative favors DKABody; crossing zero is inconclusive",
    }


def eicu_protocol_dose_response(frame: pd.DataFrame):
    # The consensus range describes the initial response to protocol-dose
    # insulin, not the six-hour average of every nonzero insulin exposure.
    # With no patient weight in the proxy cohort, 4-15 U/hr is a conservative
    # adult protocol-dose band around 0.1 U/kg/hr.
    mask = (
        frame["dka_active_t"].fillna(False).astype(bool)
        & (frame["glucose_real_start"] >= 250.0)
        & frame["first_hour_iv_rate_mean"].between(4.0, 15.0)
        & (frame["first_hour_dextrose_total"] == 0.0)
        & np.isfinite(frame["glucose_sim_1h"])
    )
    selected = frame[mask]
    if selected.empty:
        return {"n": 0}
    sim_decline = (
        selected["glucose_real_start"] - selected["glucose_sim_1h"]
    )
    sim_median = float(np.median(sim_decline))
    return {
        "n": int(len(selected)),
        "stays": int(selected["stay_id"].nunique()),
        "sim_decline_mg_dl_hr": {
            "median": round(sim_median, 6),
            "quantiles": quantiles(sim_decline),
        },
        "dose_proxy": "first-hour IV insulin mean 4-15 U/hr",
        "warning": (
            "Descriptive only. Heterogeneous patient state, prior insulin, and "
            "unobserved co-treatment make this unsuitable as a hard source gate."
        ),
    }


def reference_protocol_boundary():
    """Check the neutral simulator against a fixed, sourced treatment scenario."""
    body = DKABody()
    start = body.observe()
    action = {"insulin_iv": 6.0, "fluids": 500.0, "kcl": 10.0}
    end, _, dead, info = body.step(action, dt=1.0)
    decline = float(start["G"] - end["G"])
    prior = SOURCE_PRIORS["initial_insulin_glucose_decline_mg_dl_hr"]
    return {
        "scenario": {
            "initial_glucose_mg_dl": float(start["G"]),
            "insulin_iv_u_hr": 6.0,
            "isotonic_fluid_ml_hr": 500.0,
            "kcl_meq_hr": 10.0,
            "dextrose_g_hr": 0.0,
            "duration_hours": 1.0,
        },
        "glucose_decline_mg_dl_hr": round(decline, 6),
        "boundary": [prior["lower"], prior["upper"]],
        "passes_sourced_boundary": bool(
            prior["lower"] <= decline <= prior["upper"]
        ),
        "alive": not dead,
        "death_cause": info.get("cause"),
        "end_state": {
            key: round(float(end[key]), 6)
            for key in ("G", "HCO3", "pH", "Ke", "anion_gap", "MAP")
        },
        "parameters_fitted_to_eicu": False,
    }


def death_coverage_summary(frame: pd.DataFrame):
    deaths = frame[~frame["sim_alive"]].copy()
    if deaths.empty:
        return {"total": 0, "by_cause": {}, "by_insulin_coverage": {}}
    deaths["insulin_coverage"] = np.select(
        [
            deaths["insulin_total"] > 0.0,
            deaths["insulin_evidence"],
        ],
        [
            "numeric_dose_captured",
            "presence_evidence_only",
        ],
        default="no_insulin_evidence",
    )
    return {
        "total": int(len(deaths)),
        "by_cause": {
            str(key): int(value)
            for key, value in deaths["sim_death_cause"].fillna(
                "unknown"
            ).value_counts().items()
        },
        "by_insulin_coverage": {
            str(key): int(value)
            for key, value in deaths["insulin_coverage"].value_counts().items()
        },
        "cause_by_insulin_coverage": {
            str(cause): {
                str(key): int(value)
                for key, value in group["insulin_coverage"].value_counts().items()
            }
            for cause, group in deaths.groupby(
                deaths["sim_death_cause"].fillna("unknown")
            )
        },
        "interpretation": (
            "Deaths with presence evidence but no numeric dose are explicitly "
            "coverage-limited. No-evidence deaths may still reflect missing "
            "capture, but this observational cohort cannot prove treatment occurred."
        ),
    }


def treatment_strata(frame: pd.DataFrame):
    masks = {
        "all_active_dka": np.ones(len(frame), dtype=bool),
        "numeric_dose_captured": frame["insulin_total"] > 0.0,
        "insulin_without_dextrose": (
            (frame["insulin_total"] > 0.0)
            & (frame["dextrose_total"] == 0.0)
        ),
        "insulin_with_dextrose": (
            (frame["insulin_total"] > 0.0)
            & (frame["dextrose_total"] > 0.0)
        ),
        "no_captured_insulin": frame["insulin_total"] == 0.0,
        "insulin_evidence_only": (
            (frame["insulin_total"] == 0.0)
            & frame["insulin_evidence"]
        ),
        "no_insulin_evidence": (
            (frame["insulin_total"] == 0.0)
            & ~frame["insulin_evidence"]
        ),
        "fluids_without_insulin": (
            (frame["fluids_total"] > 0.0)
            & (frame["insulin_total"] == 0.0)
        ),
        "protocol_iv_first_hour": (
            frame["first_hour_iv_rate_mean"].between(4.0, 15.0)
            & (frame["first_hour_dextrose_total"] == 0.0)
        ),
    }
    return {
        name: {
            "rows": int(np.asarray(mask).sum()),
            "stays": int(frame.loc[mask, "stay_id"].nunique()),
            "targets": {
                target: target_summary(frame.loc[mask], target)
                for target in TARGETS
            },
        }
        for name, mask in masks.items()
    }


def decision(report):
    insulin = report["sourced_boundary_checks"]["reference_protocol"]
    summaries = report["treatment_strata"]["insulin_without_dextrose"]["targets"]
    failures = []
    if insulin.get("passes_sourced_boundary") is False:
        failures.append("insulin_glucose_recovery_outside_sourced_boundary")
    for target in ("glucose", "bicarbonate", "ph", "potassium"):
        summary = summaries.get(target, {})
        if (
            summary.get("n", 0) >= 20
            and summary.get("direction_agreement", 1.0) < 0.55
        ):
            failures.append(
                f"{target}_insulin_stratum_external_direction_falsification"
            )
    return {
        "falsifications": failures,
        "runtime_change_allowed": (
            "insulin_glucose_recovery_outside_sourced_boundary" in failures
            and (
                "glucose_insulin_stratum_external_direction_falsification"
                in failures
            )
        ),
        "automatic_parameter_fit_allowed": False,
        "checkpoint_promotion_allowed": False,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "stop_rule": (
            "Change a mechanism only when a sourced boundary and external "
            "falsification agree. Do not tune constants to minimize this cohort."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="eicu_dka_transitions_6h_demo.parquet")
    parser.add_argument("--output", default="dka_treatment_recovery_audit.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    source = pd.read_parquet(args.cohort)
    required = {
        "stay_id", "history_action_grid", "future_action_grid", "dka_active_t",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"cohort is missing required columns: {missing}")

    rows = []
    for _, row in source.iterrows():
        replayed = replay_row(row)
        replayed["stay_id"] = int(row["stay_id"])
        replayed["dka_active_t"] = bool(row.get("dka_active_t", False))
        rows.append(replayed)
    replay_frame = pd.DataFrame(rows)
    active = replay_frame[replay_frame["dka_active_t"]].reset_index(drop=True)

    target_summaries = {
        target: target_summary(active, target)
        for target in TARGETS
    }
    report = {
        "experiment": "sourced DKABody treatment-recovery falsification",
        "cohort": str(Path(args.cohort).resolve()),
        "data_boundary": {
            "dataset_role": "observational external falsification only",
            "patient_rows_in_output": False,
            "patient_identifiers_in_output": False,
            "treatment_effect_identified": False,
            "parameters_fitted_to_eicu": False,
        },
        "source_priors": SOURCE_PRIORS,
        "evaluated_rows": int(len(replay_frame)),
        "active_dka_rows": int(len(active)),
        "active_dka_stays": int(active["stay_id"].nunique()),
        "simulated_deaths": death_coverage_summary(active),
        "target_summaries": target_summaries,
        "treatment_strata": treatment_strata(active),
        "patient_bootstrap": {
            target: clustered_bootstrap_delta(
                active, target, args.bootstrap_samples, args.seed + index
            )
            for index, target in enumerate(TARGETS)
        },
        "sourced_boundary_checks": {
            "reference_protocol": reference_protocol_boundary(),
            "observed_protocol_dose_response": eicu_protocol_dose_response(active),
        },
    }
    report["decision"] = decision(report)
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output,
        "active_dka_rows": report["active_dka_rows"],
        "active_dka_stays": report["active_dka_stays"],
        "simulated_deaths": report["simulated_deaths"],
        "sourced_boundary_checks": report["sourced_boundary_checks"],
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
