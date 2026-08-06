"""Audit graph-propagated whole-body coupling beyond explicit latent baseline.

Baseline:
  ordinary features + all five belief families + explicit whole-body latent axes

Candidate:
  baseline + graph-propagated coupling features

Placebo:
  baseline + same number of random columns

Passing means the graph layer adds incremental factual signal beyond both the
all-belief baseline and the previous explicit latent coupling layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from eicu_all_model_belief_coupling_audit import (  # noqa: E402
    FULL_COHORTS,
    SEEDS,
    _resolve_path,
    attach_all_beliefs_cached,
    auto_targets,
    target_rows,
)
from eicu_body_system_target_router import _fit_ridge  # noqa: E402
from eicu_sepsis_target_router import _feature_columns, _round, _subject_column, split_subjects  # noqa: E402
from eicu_whole_body_coupling_latent_audit import delta_summary, is_validated, placebo_columns  # noqa: E402
from whole_body_coupling_latent import whole_body_coupling_latent_features  # noqa: E402
from whole_body_graph_coupling import whole_body_graph_coupling_features  # noqa: E402


def summarize(target_reports: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [report for report in target_reports if report.get("status") == "evaluated"]
    both = [
        report for report in evaluated
        if report["candidate_vs_explicit_latent_baseline"]["significant"] and report["candidate_vs_placebo"]["significant"]
    ]
    baseline_deltas = [
        float(report["candidate_vs_explicit_latent_baseline"]["point_delta"])
        for report in evaluated
        if report["candidate_vs_explicit_latent_baseline"]["point_delta"] is not None
    ]
    placebo_deltas = [
        float(report["candidate_vs_placebo"]["point_delta"])
        for report in evaluated
        if report["candidate_vs_placebo"]["point_delta"] is not None
    ]
    return {
        "evaluated_splits": int(len(evaluated)),
        "pass_both_count": int(len(both)),
        "median_delta_vs_explicit_latent_baseline": _round(float(np.median(baseline_deltas))) if baseline_deltas else None,
        "median_delta_vs_placebo": _round(float(np.median(placebo_deltas))) if placebo_deltas else None,
    }


def graph_is_validated(report: dict[str, object]) -> bool:
    patient = report["patient_splits"]
    holdout = report.get("holdout") or {}
    return bool(
        patient["pass_both_count"] == len(SEEDS)
        and holdout.get("status") == "evaluated"
        and holdout.get("candidate_vs_explicit_latent_baseline", {}).get("significant")
        and holdout.get("candidate_vs_placebo", {}).get("significant")
    )


def split_target_report(
    frame: pd.DataFrame,
    target: str,
    suffix: str,
    belief_columns: list[str],
    latent_columns: list[str],
    graph_columns: list[str],
    seed: int,
    group_column: str | None = None,
    discovery_fraction: float = 0.67,
    active_only: bool = True,
) -> dict[str, object]:
    group_column = group_column or _subject_column(frame)
    discovery_groups, heldout_groups = split_subjects(
        frame,
        seed=seed,
        discovery_fraction=discovery_fraction,
        group_column=group_column,
    )
    discovery = frame[frame[group_column].isin(discovery_groups)].copy()
    heldout = frame[frame[group_column].isin(heldout_groups)].copy()
    train = target_rows(discovery, target, suffix)
    test = target_rows(heldout, target, suffix)
    active_column = next((column for column in test.columns if column.endswith("_active_t")), None)
    if active_only and active_column is not None:
        test = test[test[active_column].fillna(False).astype(bool)].copy()
    if train.empty or test.empty:
        return {"status": "insufficient_pairs", "rows": int(len(test)), "subjects": 0}

    excluded = set(belief_columns) | set(latent_columns) | set(graph_columns)
    ordinary_features = [column for column in _feature_columns(train, target) if column not in excluded]
    usable_beliefs = [
        column for column in belief_columns
        if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    usable_latents = [
        column for column in latent_columns
        if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    usable_graph = [
        column for column in graph_columns
        if column in train and column in test and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    baseline_features = sorted(set(ordinary_features + usable_beliefs + usable_latents))
    candidate_features = sorted(set(baseline_features + usable_graph))

    baseline = _fit_ridge(train, test, target, baseline_features, 10.0, suffix)
    candidate = _fit_ridge(train, test, target, candidate_features, 10.0, suffix)
    placebo_train, placebo = placebo_columns(train, len(usable_graph), seed + 301)
    placebo_test, _ = placebo_columns(test, len(usable_graph), seed + 402)
    placebo_pred = _fit_ridge(placebo_train, placebo_test, target, sorted(set(baseline_features + placebo)), 10.0, suffix)

    truth = test[f"{target}_{suffix}"].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "group_column": group_column,
        "rows": int(len(test)),
        "subjects": int(test[_subject_column(test)].nunique()) if len(test) else 0,
        "feature_counts": {
            "ordinary": int(len(ordinary_features)),
            "belief": int(len(usable_beliefs)),
            "latent": int(len(usable_latents)),
            "graph": int(len(usable_graph)),
            "baseline": int(len(baseline_features)),
            "candidate": int(len(candidate_features)),
            "placebo": int(len(placebo)),
        },
        "candidate_vs_explicit_latent_baseline": delta_summary(test, truth, candidate, baseline, seed + 311),
        "candidate_vs_placebo": delta_summary(test, truth, candidate, placebo_pred, seed + 323),
    }


def audit_cohort(spec: dict[str, object], *, cache_dir: Path, rebuild_cache: bool = False) -> dict[str, object]:
    print(f"Loading {spec['name']}", flush=True)
    cohort_path = _resolve_path(str(spec["path"]))
    if not cohort_path.exists():
        return {"name": spec["name"], "status": "missing_cohort", "cohort": str(spec["path"]), "targets": {}}
    frame = pd.read_parquet(cohort_path).reset_index(drop=True)
    frame, belief_columns, cache_info = attach_all_beliefs_cached(
        frame,
        spec=spec,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    latent = whole_body_coupling_latent_features(frame).reset_index(drop=True)
    latent_columns = list(latent.columns)
    frame = pd.concat([frame.reset_index(drop=True), latent], axis=1)
    graph = whole_body_graph_coupling_features(frame).reset_index(drop=True)
    graph_columns = list(graph.columns)
    frame = pd.concat([frame.reset_index(drop=True), graph], axis=1)
    print(
        f"Attached beliefs={len(belief_columns)} latent={len(latent_columns)} "
        f"graph={len(graph_columns)} rows={len(frame)} cache_rebuilt={cache_info.get('rebuilt')}",
        flush=True,
    )

    requested = spec["targets"]
    targets = auto_targets(frame, str(spec["future_suffix"])) if requested == "auto" else tuple(requested)
    reports = {}
    for target in targets:
        split_reports = [
            split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                latent_columns,
                graph_columns,
                seed,
            )
            for seed in SEEDS
        ]
        holdout = None
        if "hospitalid" in frame and frame["hospitalid"].nunique(dropna=True) >= 2:
            holdout = split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                latent_columns,
                graph_columns,
                9911,
                group_column="hospitalid",
            )
        elif spec.get("holdout_group_column") in frame and frame[str(spec["holdout_group_column"])].nunique(dropna=True) >= 2:
            holdout = split_target_report(
                frame,
                str(target),
                str(spec["future_suffix"]),
                belief_columns,
                latent_columns,
                graph_columns,
                9911,
                group_column=str(spec["holdout_group_column"]),
            )
        reports[str(target)] = {
            "patient_splits": summarize(split_reports),
            "holdout": holdout,
        }
        print(spec["name"], target, reports[str(target)]["patient_splits"], flush=True)

    return {
        "name": spec["name"],
        "cohort": str(spec["path"]),
        "future_suffix": str(spec["future_suffix"]),
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()) if len(frame) else 0,
        "hospitals": int(frame["hospitalid"].nunique()) if "hospitalid" in frame else None,
        "holdout_group_column": "hospitalid" if "hospitalid" in frame else spec.get("holdout_group_column"),
        "belief_feature_count": int(len(belief_columns)),
        "latent_feature_count": int(len(latent_columns)),
        "graph_feature_count": int(len(graph_columns)),
        "cache": cache_info,
        "targets": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="whole_body_graph_coupling_audit.json")
    parser.add_argument("--cohorts", default="cardiovascular_full_6h,sepsis_6h_full,aki_24h_full,aki_48h_full,mimiciv_observation_6h")
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/osler_belief_cache_full"))
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {item.strip() for item in args.cohorts.split(",") if item.strip()}
    specs = [spec for spec in FULL_COHORTS if str(spec["name"]) in requested]
    cohorts = [audit_cohort(spec, cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache) for spec in specs]
    validated = []
    for cohort in cohorts:
        if cohort.get("status") == "missing_cohort":
            continue
        for target, report in cohort["targets"].items():
            if graph_is_validated(report):
                patient = report["patient_splits"]
                validated.append({
                    "cohort": cohort["name"],
                    "target": target,
                    "median_delta_vs_explicit_latent_baseline": patient["median_delta_vs_explicit_latent_baseline"],
                    "median_delta_vs_placebo": patient["median_delta_vs_placebo"],
                })

    output = {
        "artifact": "graph-propagated whole-body coupling audit",
        "definition": "explicit latent baseline vs explicit latent plus graph propagation features vs capacity-matched placebo",
        "validated": validated,
        "cohorts": cohorts,
        "boundary": {
            "factual_observed_inputs_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_authority": False,
            "row_level_outputs_written": False,
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validated": validated, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
