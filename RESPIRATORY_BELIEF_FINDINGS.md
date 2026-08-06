# Respiratory / Gas-Exchange Belief Audit

Status: **validated for selected downstream observables**

This audit tests whether a patient-specific predict-update respiratory belief state improves factual downstream prediction beyond both:

- a baseline ridge model using the usual observed state/action variables
- a capacity-matched placebo ridge model using random features of the same width

The belief state is an inferred research feature set. It has no direct hidden-state accuracy claim, no causal authority, no clinical authority, no runtime decision authority, and no checkpoint or active-rule promotion authority.

## Cohort

Source: eICU respiratory-failure / hypoxemia-like 6h transition cohort, re-extracted at full scale.

- Rows: 310,172
- Active respiratory rows: 214,130
- Stays: 22,077
- Subjects: 19,438
- Hospitals: 200
- Horizon: 6h

## Gate

A target is validated only if `respiratory_belief_ridge` significantly beats both:

- `baseline_ridge`
- `placebo_belief_ridge`

on all 7 patient-heldout splits and also passes hospital-heldout.

## Results

| Target | 7-split pass count | Median delta vs baseline | Median delta vs placebo | Hospital-heldout | Status |
| --- | ---: | ---: | ---: | --- | --- |
| o2sat | 7/7 | -0.132723 | -0.133133 | pass | validated |
| respiratory_rate | 7/7 | -0.192376 | -0.192345 | pass | validated |
| heart_rate | 7/7 | -0.023963 | -0.024208 | pass | validated |
| bicarbonate | 7/7 | -0.023749 | -0.026564 | pass | validated |
| map | 5/7 | -0.010996 | -0.011821 | fail | candidate-only |
| paco2 | 0/7 | -0.020118 | -0.037053 | fail vs baseline | candidate-only |
| ph | 0/7 | -0.000251 | -0.000288 | fail vs baseline | candidate-only |

Hospital-heldout details for validated targets:

- O2 saturation MAE: 2.920065 baseline → 2.745684 with belief
- Respiratory rate MAE: 4.009301 baseline → 3.767697 with belief
- Heart rate MAE: 8.432626 baseline → 8.412765 with belief
- Bicarbonate MAE: 2.132781 baseline → 2.105294 with belief

## Interpretation

This is the fourth validated personalization site in the observation layer:

1. renal belief state for creatinine/BUN
2. cardiovascular belief state for heart-rate prediction
3. electrolyte / acid-base belief state for potassium, bicarbonate, anion gap, and creatinine
4. respiratory / gas-exchange belief state for O2 saturation, respiratory rate, heart rate, and bicarbonate

The strongest validated gains are in oxygenation and respiratory rate, which fits the intended role of the belief state: patient-specific gas-exchange and ventilatory trajectory. Bicarbonate also passes, suggesting the respiratory belief carries useful acid-base compensation signal.

PaCO2 and pH show partial signal but fail the strict gate. MAP is a near-miss on patient splits but fails hospital-heldout. These are intentionally not promoted.

## Safety Boundary

- Factual observed-treatment prediction only
- No causal claim
- No counterfactual claim
- No clinical claim
- No runtime decision authority
- No checkpoint promotion
- No active-rule promotion
- No direct hidden-state accuracy claim
- Aggregate-only artifact; no raw rows or patient identifiers

