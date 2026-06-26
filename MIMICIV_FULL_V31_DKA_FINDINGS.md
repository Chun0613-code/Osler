# MIMIC-IV Full v3.1 DKA Findings

Date: 2026-06-25

This run uses the local credentialed MIMIC-IV v3.1 files to build an
ICD-supported, lab-defined DKA transition cohort. Patient-level parquet output is
local-only and ignored by git. The committed JSON files are aggregate reports
only.

## Cohort

`dka_transition_extract.py` now uses a full-data path:

1. find lab co-occurring DKA onset in DuckDB SQL;
2. require DKA ICD support for the promotion-facing cohort;
3. pull measurements, inputevents, eMAR/eMAR-detail administrations, and
   ingredientevents only for those stays.

The ICD-supported cohort has:

- 755 subjects
- 1,010 ICU stays
- 10,785 6-hour transitions
- 3,739 active-DKA rows in the treatment-recovery audit
- 4,950 glucose pairs, 5,094 potassium pairs, 4,749 bicarbonate pairs, and
  10,311 MAP pairs
- exact dose/time grids, explicit treatment start/stop event grids, six-hour
  pre-anchor treatment history, and maintenance/nutrition context

Action coverage is now strong enough for factual proxy testing:

- insulin in 83.9% of transitions
- fluids in 76.5%
- KCl in 39.4%
- dextrose in 42.5%
- bicarbonate in 3.2%

This remains observational EHR data. These reports do not identify treatment
effects, allow causal claims, or permit clinical claims.

## Factual Proxy

Single checkpoints are not promotable:

- v5 loses active-DKA glucose, bicarbonate, potassium, sodium, osmolality,
  creatinine, urine output, and nearly ties MAP.
- the PhysioNet presentation-only candidate wins active-DKA glucose and
  anion gap, but loses most other targets.

The patient-held-out per-target ensemble is the important result. It chooses
methods on discovery stays and evaluates on held-out patients:

- held-out active-DKA normalized MAE improves from persistence `0.390721` to
  ensemble `0.369611`
- bootstrap delta vs persistence is `-0.027072`, 95% CI
  `[-0.036244, -0.018760]`, significant
- glucose improves from `144.46` to `108.77` mg/dL MAE, significant
- anion gap improves from `5.04` to `4.68`, significant
- MAP is not significant and should not be claimed

This is the first robust patient-held-out MIMIC result in this project where the
guarded ensemble beats persistence. It is still a factual observed-treatment
proxy, not counterfactual validation.

## Robustness

The first significant result was pressure-tested before any promotion claim.
`mimiciv_full_v31_robustness.py` reuses the same prediction logic and writes only
aggregate metrics. It does not write row-level predictions, patient identifiers,
stay lists, or timestamp cutoffs.

Across seven random patient-held-out split seeds:

- active-DKA: 7/7 splits beat persistence and 7/7 are significant
- active-DKA stay-level delta range: `-0.027468` to `-0.023236`
- active-DKA median delta: `-0.025305`
- all windows: 7/7 splits beat persistence and 7/7 are significant
- all-window median delta: `-0.009065`

The time-order split also holds:

- early-stay discovery, late-stay held-out
- active-DKA normalized MAE improves from persistence `0.386096` to ensemble
  `0.366090`
- bootstrap delta `-0.025508`, 95% CI `[-0.032894, -0.018012]`

The diagnosis-pure check is already satisfied for this parquet because it is
restricted to ICD-supported DKA stays. Hospital/careunit robustness is not yet
tested: the current parquet does not include aggregate-safe careunit/site
metadata.

Conclusion: the factual ensemble win is not a single-split accident. It is
stable across random patient splits and a time-order split, while remaining
observational.

## Causal Feasibility

`dka_causal_evaluation.py` was rerun on the full MIMIC-IV v3.1 cohort to check
whether causal target-trial diagnostics are now feasible. The answer is no: N is
large enough to run diagnostics, but treatment-effect claims are still blocked.

All tested target trials are available, but none pass promotion readiness:

- insulin -> glucose
- insulin -> bicarbonate
- insulin -> anion gap
- fluids -> MAP
- KCl -> serum potassium

The repeated blockers are:

- insufficient propensity overlap
- high concomitant treatment rate
- matched covariate balance inadequate

Examples: insulin trials have overlap around `0.675` and concomitant treatment
rate `1.0`; fluids -> MAP has overlap `0.191`; KCl -> potassium has concomitant
treatment rate `0.956`.

Conclusion: the full MIMIC cohort opens the factual prediction gate, but not the
causal gate. The next modeling branch should prioritize real-data factual
training/fine-tuning unless a stricter causal design is added.

## Treatment-Recovery Audit

The DKABody treatment-recovery audit found:

- 3,739 active-DKA rows across 878 stays
- 48 simulated deaths
- death causes: 26 hyperkalemia, 14 acidosis, 7 circulatory collapse, and
  1 hypokalemia
- no external falsification strong enough to change runtime physiology

The sourced protocol check still passes: a neutral DKABody patient receiving
IV insulin 6 U/hr, isotonic fluid 500 mL/hr, and KCl 10 mEq/hr lowers glucose by
66.07 mg/dL/hr, inside the 50-75 mg/dL/hr sourced boundary.

Decision remains conservative:

- no runtime physiology change
- no automatic parameter fit
- no checkpoint promotion
- no causal claim
- no clinical claim

## Symbolic Rule Test

The full MIMIC symbolic test produced:

- factual direction accuracy: 0.564 overall
- changed-only direction accuracy: 0.6053
- mean proposal confidence: 0.9401
- 42 observational candidate rules
- 23 retrospectively validated rules
- 0 active-rule conflicts
- 0 automatically promoted rules

The candidate sandbox worked as designed: useful hypotheses were captured, but
the active symbolic engine was not modified.

## Boundary

The local parquet cohort contains patient-level transitions and is ignored by
git. The versioned artifacts are aggregate-only reports:

- `mimiciv_full_v31_icd_dka_transition_report.json`
- `mimiciv_full_v31_icd_v5_evaluation.json`
- `mimiciv_full_v31_icd_presentation_only_evaluation.json`
- `mimiciv_full_v31_icd_per_target_ensemble.json`
- `mimiciv_full_v31_icd_robustness.json`
- `mimiciv_full_v31_icd_treatment_recovery_audit.json`
- `mimiciv_full_v31_icd_causal_diagnostics.json`
- `mimiciv_full_v31_icd_symbolic_real_test_v5.json`
