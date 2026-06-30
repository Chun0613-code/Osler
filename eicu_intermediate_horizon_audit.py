"""Run aggregate 1h/3h/12h factual move-audits for eICU routers.

This script fills the trajectory grid introduced by
``WHOLE_BODY_TRAJECTORY_UNCERTAINTY_LAYER.md``.  It only writes aggregate JSON
reports.  Row-level transition parquet cohorts are generated as local-only
inputs and are ignored by git.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_HORIZONS = (1.0, 3.0, 12.0)
DEFAULT_MODULES = ("sepsis", "aki")


def horizon_suffix(horizon_hours: float) -> str:
    value = float(horizon_hours)
    if value.is_integer():
        return f"tp{int(value)}"
    return "tp" + str(value).replace(".", "p")


def horizon_label(horizon_hours: float) -> str:
    value = float(horizon_hours)
    if value.is_integer():
        return f"{int(value)}h"
    return str(value).replace(".", "p") + "h"


def module_paths(module: str, horizon_hours: float) -> dict[str, Path]:
    label = horizon_label(horizon_hours)
    if module == "sepsis":
        return {
            "cohort": Path(f"eicu_sepsis_transitions_{label}.parquet"),
            "transition_report": Path(f"eicu_sepsis_transition_report_{label}.json"),
            "router_report": Path(f"eicu_sepsis_target_router_{label}.json"),
            "extractor": Path("eicu_sepsis_transition_extract.py"),
            "router": Path("eicu_sepsis_target_router.py"),
        }
    if module == "aki":
        return {
            "cohort": Path(f"eicu_aki_transitions_{label}.parquet"),
            "transition_report": Path(f"eicu_aki_transition_report_{label}.json"),
            "router_report": Path(f"eicu_aki_target_router_{label}.json"),
            "extractor": Path("eicu_aki_transition_extract.py"),
            "router": Path("eicu_aki_target_router.py"),
        }
    raise ValueError(f"unsupported module: {module}")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_reports(
    module: str,
    horizon_hours: float,
    data_root: Path,
    bootstrap_samples: int,
    force: bool,
) -> dict[str, Path]:
    paths = module_paths(module, horizon_hours)
    if force or not paths["cohort"].exists() or not paths["transition_report"].exists():
        run_command([
            sys.executable,
            str(paths["extractor"]),
            "--data-root",
            str(data_root),
            "--output",
            str(paths["cohort"]),
            "--report",
            str(paths["transition_report"]),
            "--horizon-hours",
            str(horizon_hours),
        ])
    if force or not paths["router_report"].exists():
        run_command([
            sys.executable,
            str(paths["router"]),
            "--cohort",
            str(paths["cohort"]),
            "--output",
            str(paths["router_report"]),
            "--future-suffix",
            horizon_suffix(horizon_hours),
            "--bootstrap-samples",
            str(bootstrap_samples),
        ])
    return paths


def active_scope_name(module: str) -> str:
    return "active_sepsis_only" if module == "sepsis" else "active_aki_only"


def target_validation_summary(module: str, router_report: dict[str, object]) -> dict[str, dict[str, object]]:
    scope = active_scope_name(module)
    runs = router_report["random_patient_splits"]["runs"]
    target_counts = router_report["random_patient_splits"]["summary"]["selected_method_counts_by_target"]
    split_count = len(runs)
    hospital = router_report.get("hospital_holdout_split") or {}
    hospital_scope = hospital.get(f"heldout_{scope}", {}) if hospital.get("available") else {}
    output: dict[str, dict[str, object]] = {}
    for target, counts in target_counts.items():
        non_persistence_count = int(sum(count for method, count in counts.items() if method != "persistence"))
        significant_count = 0
        beat_count = 0
        for run in runs:
            per_target = run[f"heldout_{scope}"]["per_target"].get(target, {})
            delta = per_target.get("router_delta_vs_persistence", {})
            significant_count += int(bool(delta.get("significant")))
            beat_count += int(bool(delta.get("beats_persistence")))
        hospital_target = (hospital_scope.get("per_target") or {}).get(target, {})
        hospital_delta = hospital_target.get("router_delta_vs_persistence", {})
        output[target] = {
            "selected_method_counts": counts,
            "non_persistence_selected_splits": non_persistence_count,
            "heldout_beats_persistence_splits": beat_count,
            "heldout_significant_splits": significant_count,
            "hospital_holdout_significant": bool(hospital_delta.get("significant")),
            "move_validated": bool(
                non_persistence_count == split_count
                and significant_count == split_count
                and hospital_delta.get("significant")
            ),
        }
    return output


def active_transition_count(module: str, transition_report: dict[str, object]) -> int | None:
    for key in (
        "active_transitions",
        f"active_{module}_transitions",
        f"active_{module}_windows",
    ):
        if key in transition_report:
            value = transition_report.get(key)
            return int(value) if value is not None else None
    return None


def build_audit(
    modules: tuple[str, ...],
    horizons: tuple[float, ...],
    data_root: Path,
    bootstrap_samples: int,
    force: bool,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for module in modules:
        module_results = {}
        for horizon in horizons:
            paths = ensure_reports(module, horizon, data_root, bootstrap_samples, force)
            transition_report = json.loads(paths["transition_report"].read_text(encoding="utf-8"))
            router_report = json.loads(paths["router_report"].read_text(encoding="utf-8"))
            module_results[horizon_label(horizon)] = {
                "cohort": paths["cohort"].name,
                "transition_report": paths["transition_report"].name,
                "router_report": paths["router_report"].name,
                "future_suffix": horizon_suffix(horizon),
                "transition_summary": {
                    "stays": transition_report.get("stays"),
                    "subjects": transition_report.get("subjects"),
                    "hospitals": transition_report.get("hospitals"),
                    "transitions": transition_report.get("transitions"),
                    "active_transitions": active_transition_count(module, transition_report),
                    "target_pair_counts": transition_report.get("target_pair_counts", {}),
                },
                "router_summary": router_report["random_patient_splits"]["summary"],
                "hospital_holdout_available": bool(router_report["hospital_holdout_split"]["available"]),
                "target_validation": target_validation_summary(module, router_report),
            }
        results[module] = module_results
    return {
        "artifact": "eICU intermediate-horizon move audit",
        "horizons_hours": [float(value) for value in horizons],
        "modules": list(modules),
        "data_root": data_root.name,
        "bootstrap_samples": int(bootstrap_samples),
        "results": results,
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "factual_prediction_only": True,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("eicu-collaborative-research-database-2.0"))
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("whole_body_intermediate_horizon_move_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    horizons = tuple(float(item.strip()) for item in args.horizons.split(",") if item.strip())
    report = build_audit(
        modules=modules,
        horizons=horizons,
        data_root=args.data_root,
        bootstrap_samples=args.bootstrap_samples,
        force=bool(args.force),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "modules": list(modules),
        "horizons_hours": list(horizons),
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
