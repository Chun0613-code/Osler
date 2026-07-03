# Cardiovascular Predict-Update Belief Findings

This audit tests whether the AKI renal-belief idea generalizes to a second
system: cardiovascular/perfusion physiology.  The candidate is an online
predict-update shock/perfusion belief built from each patient's own MAP,
lactate, heart-rate, vasopressor, inotrope, and fluid context.

The state is **not** a measured clinical variable.  It can only be used if it
improves downstream observable prediction beyond both a population ridge
baseline and a capacity-matched placebo.

## Bounded Cohort

- Dataset: eICU cardiovascular instability / shock / heart-failure module.
- Rows: `16,047`
- Subjects: `1,211`
- Stays: `1,382`
- Hospitals: `12`
- Active cardiovascular-instability rows: `4,734`
- Horizon: `6h`

This bounded engineering cohort produced a heart-rate near-miss: `6/7`
patient splits passed both baseline and capacity-matched placebo, and the
hospital-heldout gate passed.  It was therefore kept candidate-only until the
same test could be run on the full eICU cardiovascular cohort.

## Full Cohort Replication

- Dataset: full unbounded eICU cardiovascular instability / shock /
  heart-failure module.
- Rows: `682,172`
- Subjects: `45,486`
- Stays: `55,080`
- Hospitals: `206`
- Active cardiovascular-instability rows: `298,744`
- Horizon: `6h`
- Extraction report: `eicu_cardiovascular_instability_full_transition_report.json`
- Audit report: `eicu_cardiovascular_belief_full_heart_rate_audit.json`

Only `heart_rate` was re-tested on the full cohort, because it was the only
bounded-cohort target with a real near-miss.  MAP, lactate, O2 saturation, and
respiratory rate remain unsupported for cardiovascular belief promotion.

## Gate

- Baseline: ordinary factual `ridge_realfit`.
- Candidate: `ridge_realfit + cardiovascular belief features`.
- Placebo: `ridge_realfit + same-number random features`.
- Required: significant held-out improvement over both baseline and placebo in
  all 7 patient splits, plus hospital-heldout support before validation.
- Output is aggregate-only; no row-level or patient identifiers are stored.

## Bounded Result

| Target | Patient Splits Passing Both | Median Delta vs Baseline | Median Delta vs Placebo | Hospital-Heldout |
|---|---:|---:|---:|---|
| heart_rate | 6 / 7 | -0.1995 | -0.2061 | pass |
| map | 0 / 7 | -0.0050 | -0.0114 | fail |
| lactate | 0 / 7 | +0.1258 | +0.0581 | fail |
| o2sat | 0 / 7 | +0.0056 | +0.0085 | fail |
| respiratory_rate | 0 / 7 | +0.0075 | +0.0036 | fail |

Hospital-heldout heart-rate details:

- baseline MAE: `8.9350`
- cardiovascular-belief MAE: `8.7859`
- placebo MAE: `8.9412`
- delta vs baseline: `-0.2685`, 95% CI `[-0.4454, -0.0964]`
- delta vs placebo: `-0.2734`, 95% CI `[-0.4293, -0.0996]`

## Full Cohort Heart-Rate Result

| Target | Patient Splits Passing Both | Median Delta vs Baseline | Median Delta vs Placebo | Hospital-Heldout |
|---|---:|---:|---:|---|
| heart_rate | 7 / 7 | -0.1122 | -0.1125 | pass |

Hospital-heldout full-cohort details:

- held-out rows: `288,969`
- held-out subjects: `18,671`
- baseline MAE: `8.4157`
- cardiovascular-belief MAE: `8.3192`
- placebo MAE: `8.4160`
- delta vs baseline: `-0.1087`, 95% CI `[-0.1224, -0.0954]`
- delta vs placebo: `-0.1091`, 95% CI `[-0.1238, -0.0957]`

## Interpretation

The cardiovascular belief state finds a real personalization signal for
heart-rate forecasting.  The bounded cohort was underpowered by one split, but
the full cohort passes the required `7/7` patient-split gate and the
hospital-heldout gate while also beating a capacity-matched placebo.

This validates a second online personalized belief component after AKI renal
belief: heart-rate forecasting can use a patient-specific perfusion/shock state
when the cohort is large enough.  The result is target-specific.  It does not
validate the hidden state as a directly measured clinical quantity, and it does
not validate cardiovascular belief for MAP, lactate, O2 saturation, respiratory
rate, causal effects, or treatment planning.

## Boundary

Allowed:

- factual heart-rate prediction with the validated cardiovascular belief state;
- candidate-only reporting for unsupported cardiovascular belief targets;
- future experiments on alternate horizons or additional dense cardiovascular
  targets.

Not allowed:

- clinical, causal, or counterfactual claims;
- runtime treatment authority;
- checkpoint or active-rule promotion;
- claiming direct hidden-state accuracy.

The validated personalized predict-update belief components are now:

- AKI renal belief state for creatinine/BUN;
- cardiovascular perfusion/shock belief state for heart rate.
