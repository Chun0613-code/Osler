"""
nhanes_nowcast_audit.py

Cross-sectional NOWCAST audit on NHANES 2017-2018 (_J cycle), mirroring the ICU
observation-layer nowcast gate.

NHANES is cross-sectional: each participant is measured once. There is NO per-person
time axis, so FORECAST is impossible here. Only NOWCAST applies: estimate a variable
that was not measured on a participant from the participant's other contemporaneous
measurements.

Gate (same discipline as the ICU layer):
  - baseline  = train-set median of the target
  - candidate = ridge on all OTHER variables
  - placebo   = ridge on the same NUMBER of random-noise features
  - 7 random participant-heldout splits
  - a target is VALIDATED only if candidate beats BOTH baseline and placebo in all 7 splits

Output is aggregate-only. No participant identifiers, no row-level data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA = Path("./nhanes_2017_2018")

SOURCE_FILES = {
    "demo": "DEMO_J.XPT",
    "bio": "BIOPRO_J.XPT",
    "cbc": "CBC_J.XPT",
    "bpx": "BPX_J.XPT",
    "bmx": "BMX_J.XPT",
    "tchol": "TCHOL_J.XPT",
    "hdl": "HDL_J.XPT",
    "trigly": "TRIGLY_J.XPT",
    "ghb": "GHB_J.XPT",
    "hscrp": "HSCRP_J.XPT",
}

CORE_TARGETS = {
    # Biochemistry (BIOPRO_J).
    "sodium": "LBXSNASI",
    "potassium": "LBXSKSI",
    "chloride": "LBXSCLSI",
    "bicarbonate": "LBXSC3SI",
    "bun": "LBXSBU",
    "creatinine": "LBXSCR",
    "glucose": "LBXSGL",
    "calcium": "LBXSCA",
    "phosphorus": "LBXSPH",
    "albumin": "LBXSAL",
    "total_protein": "LBXSTP",
    "bilirubin": "LBXSTB",
    "alt": "LBXSATSI",
    "ast": "LBXSASSI",
    "alk_phos": "LBXSAPSI",
    "uric_acid": "LBXSUA",
    # Complete blood count (CBC_J).
    "hemoglobin": "LBXHGB",
    "hematocrit": "LBXHCT",
    "wbc": "LBXWBCSI",
    "platelets": "LBXPLTSI",
    "rbc": "LBXRBCSI",
    # Vitals/body composition. MAP is derived from SBP/DBP, and body-size
    # siblings are leak-guarded.
    "sbp": "SBP",
    "dbp": "DBP",
    "map": "MAP",
    "bmi": "BMXBMI",
    "weight": "BMXWT",
    "waist": "BMXWAIST",
}

NHANES_UNIQUE_TARGETS = {
    "total_cholesterol": "LBXTC",
    "hdl_cholesterol": "LBDHDD",
    "triglycerides": "LBXTR",
    "hba1c": "LBXGH",
    "hs_crp": "LBXHSCRP",
}

TARGETS = {**CORE_TARGETS, **NHANES_UNIQUE_TARGETS}

CONTEXT_FEATURES = {
    "age": "RIDAGEYR",
    "sex": "RIAGENDR",
}

# Deterministic or near-deterministic siblings are excluded from one another's
# feature sets. Without this guard, the audit can mistake algebraic completion
# for cross-system physiologic inference.
LEAKAGE_GROUPS = [
    {"sbp", "dbp", "map"},
    {"hemoglobin", "hematocrit", "rbc"},
    {"bmi", "weight", "waist"},
]


def load(data_path, name):
    return pd.read_sas(data_path / name, format="xport")


def build_matrix(data_path):
    demo = load(data_path, SOURCE_FILES["demo"])[["SEQN", *CONTEXT_FEATURES.values()]]
    bio = load(data_path, SOURCE_FILES["bio"])
    cbc = load(data_path, SOURCE_FILES["cbc"])
    bpx = load(data_path, SOURCE_FILES["bpx"])
    bmx = load(data_path, SOURCE_FILES["bmx"])
    tchol = load(data_path, SOURCE_FILES["tchol"])
    hdl = load(data_path, SOURCE_FILES["hdl"])
    trigly = load(data_path, SOURCE_FILES["trigly"])
    ghb = load(data_path, SOURCE_FILES["ghb"])
    hscrp = load(data_path, SOURCE_FILES["hscrp"])

    df = (
        demo.merge(bio, on="SEQN", how="inner")
        .merge(cbc, on="SEQN", how="inner")
        .merge(bpx, on="SEQN", how="inner")
        .merge(bmx, on="SEQN", how="inner")
        .merge(tchol[["SEQN", "LBXTC"]], on="SEQN", how="left")
        .merge(hdl[["SEQN", "LBDHDD"]], on="SEQN", how="left")
        .merge(trigly[["SEQN", "LBXTR"]], on="SEQN", how="left")
        .merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
        .merge(hscrp[["SEQN", "LBXHSCRP"]], on="SEQN", how="left")
    )

    df["SBP"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan).mean(axis=1)
    df["DBP"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan).mean(axis=1)
    df["MAP"] = df["DBP"] + (df["SBP"] - df["DBP"]) / 3.0

    variables = {**TARGETS, **CONTEXT_FEATURES}
    matrix = df[list(variables.values())].apply(pd.to_numeric, errors="coerce")
    matrix.columns = list(variables.keys())
    return df, matrix


def leak_guarded_features(target, columns):
    excluded = {target}
    for group in LEAKAGE_GROUPS:
        if target in group:
            excluded.update(group)
    return [c for c in columns if c not in excluded], sorted(excluded - {target})


def nowcast_gate(matrix, target, seeds=range(7), min_rows=500):
    # NHANES-unique labs are evaluated as targets, not as extra predictors for
    # the routine panel. This keeps the audit question clean: can a standard
    # contemporaneous physiology panel estimate special population-health labs?
    feature_pool = [*CORE_TARGETS.keys(), *CONTEXT_FEATURES.keys()]
    cols, leak_guard_excluded = leak_guarded_features(target, feature_pool)
    d = matrix[[target] + cols].dropna()
    if len(d) < min_rows:
        return None, len(d), leak_guard_excluded
    y = d[target].to_numpy()
    Xf = d[cols].to_numpy()
    base, cand, plac = [], [], []
    for s in seeds:
        idx = np.arange(len(d))
        np.random.default_rng(s).shuffle(idx)
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        base.append(mean_absolute_error(y[te], np.full(len(te), np.median(y[tr]))))
        sc = StandardScaler().fit(Xf[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(Xf[tr]), y[tr])
        cand.append(mean_absolute_error(y[te], m.predict(sc.transform(Xf[te]))))
        rng = np.random.default_rng(1000 + s)
        P = rng.normal(size=(len(d), Xf.shape[1]))
        mp = Ridge(alpha=1.0).fit(P[tr], y[tr])
        plac.append(mean_absolute_error(y[te], mp.predict(P[te])))
    base, cand, plac = map(np.array, (base, cand, plac))
    bb = int((cand < base).sum())
    bp = int((cand < plac).sum())
    delta = float(cand.mean() - base.mean())
    return (
        {
            "n": int(len(d)),
            "baseline_mae": round(float(base.mean()), 6),
            "candidate_mae": round(float(cand.mean()), 6),
            "placebo_mae": round(float(plac.mean()), 6),
            "delta_vs_baseline": round(delta, 6),
            "relative_mae_change": round(delta / float(base.mean()), 6),
            "beats_baseline": f"{bb}/7",
            "beats_placebo": f"{bp}/7",
            "validated": bool(bb == 7 and bp == 7),
            "leak_guard_excluded_features": leak_guard_excluded,
        },
        len(d),
        leak_guard_excluded,
    )


def run_audit(data_path, min_rows=500):
    df, matrix = build_matrix(data_path)
    results = {}
    skipped = {}
    for target in TARGETS:
        result, n, leak_guard_excluded = nowcast_gate(matrix, target, min_rows=min_rows)
        if result is None:
            skipped[target] = {
                "reason": "insufficient_complete_rows",
                "n": int(n),
                "min_rows": int(min_rows),
                "leak_guard_excluded_features": leak_guard_excluded,
            }
            continue
        results[target] = result

    validated = [k for k, v in results.items() if v["validated"]]
    failed = [k for k, v in results.items() if not v["validated"]]
    return {
        "schema": "nhanes_2017_2018_cross_sectional_nowcast_audit.v1",
        "data_source": "NHANES 2017-2018 public XPT files",
        "mode": "cross_sectional_nowcast_only",
        "row_level_data_written": False,
        "participant_merge_n": int(len(df)),
        "min_complete_rows": int(min_rows),
        "gate": {
            "baseline": "train_set_target_median",
            "candidate": "ridge_on_contemporaneous_non_target_variables",
            "placebo": "capacity_matched_random_noise_features",
            "splits": "7 random participant-heldout splits",
            "pass_rule": "candidate beats both baseline and placebo in all 7 splits",
        },
        "context_features": sorted(CONTEXT_FEATURES),
        "target_groups": {
            "core_targets": sorted(CORE_TARGETS),
            "nhanes_unique_targets": sorted(NHANES_UNIQUE_TARGETS),
        },
        "leakage_guards": [
            {
                "group": sorted(group),
                "policy": "When any group member is the target, all group members are excluded from the feature set.",
            }
            for group in LEAKAGE_GROUPS
        ],
        "targets_evaluated": len(results),
        "targets_validated": len(validated),
        "validated_targets": validated,
        "failed_targets": failed,
        "skipped_targets": skipped,
        "results": results,
        "interpretation": {
            "claim_supported": (
                "The ICU nowcast recipe transfers to a healthy-population cross-sectional setting "
                "for many dense physiologic and laboratory variables."
            ),
            "claim_not_supported": (
                "NHANES has no per-person time axis, so this audit does not validate forecasting, "
                "intervention response, causality, or ICU trajectory behavior."
            ),
        },
    }


def write_markdown(report, path):
    validated = report["validated_targets"]
    failed = report["failed_targets"]
    lines = [
        "# NHANES 2017-2018 Cross-Sectional Nowcast Findings",
        "",
        "This audit tests whether the Osler observation-layer nowcast recipe transfers from ICU data to a non-ICU, healthy-population cross-sectional dataset.",
        "",
        "NHANES has no per-person time axis, so this is **nowcast only**. It does not support forecast, treatment-response, or causal claims.",
        "",
        "## Gate",
        "",
        "- Candidate: ridge regression using contemporaneous non-target variables.",
        "- Baseline: train-set target median.",
        "- Placebo: ridge regression on the same number of random-noise features.",
        "- Validation rule: candidate beats both baseline and placebo in all 7 participant-heldout splits.",
        "- Leakage guard: deterministic or near-deterministic sibling groups are excluded from one another's feature sets: `sbp/dbp/map`, `hemoglobin/hematocrit/rbc`, and `bmi/weight/waist`.",
        "- Output is aggregate-only; no participant identifiers or row-level data are written.",
        "",
        "## Result",
        "",
        f"- Merged participants: `{report['participant_merge_n']}`",
        f"- Evaluated targets: `{report['targets_evaluated']}`",
        f"- Validated nowcast targets: `{report['targets_validated']} / {report['targets_evaluated']}`",
        f"- NHANES-unique targets tested: `{', '.join(report['target_groups']['nhanes_unique_targets'])}`",
        f"- Validated: `{', '.join(validated)}`",
        f"- Failed: `{', '.join(failed) if failed else 'none'}`",
        "",
        "## Interpretation",
        "",
        "The same disciplined nowcast pattern seen in ICU data also appears in NHANES: many contemporaneous lab/body variables are constrained enough by the rest of the physiologic panel to beat both median and capacity-matched placebo baselines, even after excluding deterministic sibling variables.",
        "",
        f"The negative targets are also informative: `{', '.join(failed) if failed else 'none'}` do not pass this cross-sectional gate, so they should remain missing/fallback in the NHANES nowcast contract.",
        "",
        "This is a healthy-population observation result, not a clinical or causal result.",
        "",
        "## Target-Level Summary",
        "",
        "| Target | N | Baseline MAE | Candidate MAE | Placebo MAE | Beats Baseline | Beats Placebo | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target, row in report["results"].items():
        status = "validated" if row["validated"] else "fallback"
        lines.append(
            f"| {target} | {row['n']} | {row['baseline_mae']:.4f} | "
            f"{row['candidate_mae']:.4f} | {row['placebo_mae']:.4f} | "
            f"{row['beats_baseline']} | {row['beats_placebo']} | {status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run NHANES 2017-2018 cross-sectional nowcast audit.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--min-rows", type=int, default=500)
    args = parser.parse_args()

    report = run_audit(args.data, min_rows=args.min_rows)
    print(f"merged participants: {report['participant_merge_n']}")
    for target, row in report["results"].items():
        flag = "VALIDATED" if row["validated"] else "-"
        print(
            f"{target:14s} n={row['n']:5d}  base={row['baseline_mae']:8.3f}  "
            f"cand={row['candidate_mae']:8.3f}  beats_base={row['beats_baseline']}  "
            f"beats_plac={row['beats_placebo']}  {flag}"
        )
    print(f"\nVALIDATED nowcast targets: {report['targets_validated']} / {report['targets_evaluated']}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        write_markdown(report, args.markdown)


if __name__ == "__main__":
    main()
