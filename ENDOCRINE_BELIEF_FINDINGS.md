# Endocrine / Glycemic-Stress Belief Audit

Status: **validated for selected downstream observables**

This audit tests whether a patient-specific predict-update endocrine-metabolic belief state improves factual downstream prediction beyond both:

- a baseline ridge model using the usual observed state/action variables
- a capacity-matched placebo ridge model using random features of the same width

The belief state is an inferred research feature set. It has no direct hidden-state accuracy claim, no causal authority, no clinical authority, no runtime decision authority, and no checkpoint or active-rule promotion authority.

## Cohort

Source: eICU endocrine / glycemic / adrenal-thyroid stress 6h transition cohort, re-extracted at full scale.

- Rows: 369,605
- Active endocrine rows: 266,061
- Stays: 31,393
- Subjects: 24,597
- Hospitals: 202
- Horizon: 6h
- Bounded engineering cohort: false

## Gate

A target is validated only if `endocrine_belief_ridge` significantly beats both:

- `baseline_ridge`
- `placebo_belief_ridge`

on all 7 patient-heldout splits and also passes hospital-heldout.

## Results

| Target | 7-split pass count | Median delta vs baseline | Median delta vs placebo | Hospital-heldout | Status |
| --- | ---: | ---: | ---: | --- | --- |
| glucose | 7/7 | -0.671441 | -0.682936 | pass | validated |
| anion_gap | 7/7 | -0.130909 | -0.135452 | pass | validated |
| bicarbonate | 7/7 | -0.056937 | -0.057628 | pass | validated |
| sodium | 7/7 | -0.025322 | -0.028349 | pass | validated |
| potassium | 7/7 | -0.005223 | -0.005500 | pass | validated |
| map | 7/7 | -0.419226 | -0.420753 | pass | validated |
| serum_ketones | 0/7 | -0.637425 | -0.651334 | fail, too sparse | candidate-only |
| serum_osmolality | 0/7 | 0.073754 | -0.373079 | fail | candidate-only |
| temperature | 0/7 | 0.001151 | -0.000137 | fail | candidate-only |
| tsh | 0/7 | n/a | n/a | fail, too sparse | candidate-only |
| free_t4 | 0/7 | n/a | n/a | fail, too sparse | candidate-only |
| cortisol | 0/7 | n/a | n/a | fail, too sparse | candidate-only |

Hospital-heldout details for validated targets:

- Glucose MAE: 42.959333 baseline -> 42.587148 with belief
- Anion gap MAE: 2.516132 baseline -> 2.334629 with belief
- Bicarbonate MAE: 2.152542 baseline -> 2.091248 with belief
- Sodium MAE: 2.255098 baseline -> 2.230525 with belief
- Potassium MAE: 0.369383 baseline -> 0.362452 with belief
- MAP MAE: 10.657422 baseline -> 10.098679 with belief

## Interpretation

This is the fifth validated personalization site in the observation layer:

1. renal belief state for creatinine/BUN
2. cardiovascular belief state for heart-rate prediction
3. electrolyte / acid-base belief state for potassium, bicarbonate, anion gap, and creatinine
4. respiratory / gas-exchange belief state for O2 saturation, respiratory rate, heart rate, and bicarbonate
5. endocrine / glycemic-stress belief state for glucose, anion gap, bicarbonate, sodium, potassium, and MAP

The bounded engineering endocrine cohort showed only partial signal. The full cohort converts the signal into robust validation across 202 hospitals, which reinforces a recurring pattern in this project: personalization often needs the full patient population, not a bounded engineering sample.

The validated targets are dense glycemic, osmotic, acid-base, electrolyte, and perfusion observables. Sparse endocrine hormone markers and ketones do not pass and remain candidate-only.

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

