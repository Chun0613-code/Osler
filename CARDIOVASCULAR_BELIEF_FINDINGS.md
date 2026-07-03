# Cardiovascular Predict-Update Belief Findings

This audit tests whether the AKI renal-belief idea generalizes to a second
system: cardiovascular/perfusion physiology.  The candidate is an online
predict-update shock/perfusion belief built from each patient's own MAP,
lactate, heart-rate, vasopressor, inotrope, and fluid context.

The state is **not** a measured clinical variable.  It can only be used if it
improves downstream observable prediction beyond both a population ridge
baseline and a capacity-matched placebo.

## Cohort

- Dataset: eICU cardiovascular instability / shock / heart-failure module.
- Rows: `16,047`
- Subjects: `1,211`
- Stays: `1,382`
- Hospitals: `12`
- Active cardiovascular-instability rows: `4,734`
- Horizon: `6h`

## Gate

- Baseline: ordinary factual `ridge_realfit`.
- Candidate: `ridge_realfit + cardiovascular belief features`.
- Placebo: `ridge_realfit + same-number random features`.
- Required: significant held-out improvement over both baseline and placebo in
  all 7 patient splits, plus hospital-heldout support before validation.
- Output is aggregate-only; no row-level or patient identifiers are stored.

## Result

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

## Interpretation

The cardiovascular belief state finds a real personalization signal for
heart-rate forecasting, but it is not stable enough to validate: it reaches
`6/7` patient splits and passes hospital-heldout, but misses the required
`7/7` split gate.

This is a useful near-miss, not a promotion.  It says the patient-specific
perfusion/shock state carries individualized information, but not enough to
become a validated online digital-twin component under the current cohort and
6h horizon.

## Boundary

Allowed:

- candidate-only capability reporting;
- future experiments on larger cardiovascular cohorts or alternate horizons.

Not allowed:

- clinical, causal, or counterfactual claims;
- runtime treatment authority;
- checkpoint or active-rule promotion;
- claiming direct hidden-state accuracy.

The AKI renal belief state remains the only validated personalized
predict-update belief component.  Cardiovascular belief is candidate-only.
