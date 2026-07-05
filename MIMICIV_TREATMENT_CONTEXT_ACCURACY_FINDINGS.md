# MIMIC-IV Observed Treatment Context Accuracy Findings

Date: 2026-07-04

## Question

Does factual observed treatment context improve MIMIC-IV 6h physiologic forecasts?

This audit compares three feature sets on the same MIMIC-IV v3.1 whole-ICU transition cohort:

- Baseline: ridge residual forecast without `hist_*` / `act_*` treatment features.
- Candidate: same forecast plus observed factual treatment context from `inputevents`, `emar`, and `procedureevents`.
- Placebo: baseline plus an equal number of random-noise features.

This is a factual forecast audit only. It does not estimate treatment effects and does not allow causal, counterfactual, clinical, or treatment recommendation claims.

## Cohort

- ICU stays selected: 93,224
- Candidate anchors: 644,387
- Evaluable transition rows after target-pair filtering: 640,164
- Subjects: 63,307
- Treatment evidence rows: 16,513,272
- Stays with treatment evidence: 92,308

The generated row-level cohort was written to `/tmp/mimiciv_observation_transitions_6h_treatment_context.parquet` and is not committed.

## Gate

Candidate must beat both baseline and placebo under:

- 7 patient-heldout split seeds.
- Careunit-heldout check.
- Time-heldout check.
- Clustered bootstrap over subjects.

## Result

3 of 4 treatment-sensitive targets pass the full gate.

| Target | Rows | Subjects | Treatment Features | 7-Split Selected | 7-Split vs Baseline | 7-Split vs Placebo | Careunit | Time | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| glucose | 80,003 | 27,283 | 320 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| potassium | 80,008 | 25,316 | 320 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| bicarbonate | 76,879 | 29,302 | 320 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| map | 80,022 | 8,125 | 320 | 1/7 | 7/7 | 7/7 | fail | pass | fallback |

## Heldout Effect Sizes

### Glucose

- Careunit MAE: baseline 33.0257, candidate 31.3197, placebo 33.0331
- Careunit candidate vs baseline: delta -1.5354, 95% CI [-2.5616, -0.5641]
- Time MAE: baseline 33.0347, candidate 31.8589, placebo 33.1936
- Time candidate vs baseline: delta -1.4341, 95% CI [-1.6353, -1.2358]

### Potassium

- Careunit MAE: baseline 0.3737, candidate 0.3645, placebo 0.3755
- Careunit candidate vs baseline: delta -0.0109, 95% CI [-0.0192, -0.0030]
- Time MAE: baseline 0.3717, candidate 0.3625, placebo 0.3735
- Time candidate vs baseline: delta -0.0082, 95% CI [-0.0103, -0.0061]

### Bicarbonate

- Careunit MAE: baseline 1.8491, candidate 1.7978, placebo 1.8568
- Careunit candidate vs baseline: delta -0.0479, 95% CI [-0.0777, -0.0137]
- Time MAE: baseline 1.8114, candidate 1.7921, placebo 1.8189
- Time candidate vs baseline: delta -0.0168, 95% CI [-0.0237, -0.0101]

### MAP

- Careunit MAE: baseline 10.3863, candidate 10.3250, placebo 10.3955
- Careunit candidate vs baseline: delta -0.1005, 95% CI [-0.2598, 0.0366]
- Time MAE: baseline 9.5581, candidate 9.4685, placebo 9.5955
- Time candidate vs baseline: delta -0.0549, 95% CI [-0.1008, -0.0081]
- Interpretation: MAP has a favorable point estimate and passes time-heldout, but fails careunit-heldout and discovery selection stability. It remains fallback.

## Interpretation

Observed treatment context improves factual forecast accuracy for the variables most directly driven by recorded treatment:

- Glucose improves with observed insulin/dextrose/nutrition context.
- Potassium improves with observed potassium repletion, renal replacement, diuretics, insulin, and related context.
- Bicarbonate improves with observed bicarbonate, fluids, ventilation, renal replacement, and disease-treatment context.

MAP shows a weak positive signal but does not pass the full gate. This is consistent with MAP being affected by treatment, measurement noise, unit practice, shock severity, and incomplete dose/route capture.

## Boundary

This finding supports adding observed treatment context as an input to factual forecasts. It does not support:

- causal treatment-effect claims;
- counterfactual intervention planning;
- clinical recommendations;
- active rule promotion;
- runtime treatment authority.

