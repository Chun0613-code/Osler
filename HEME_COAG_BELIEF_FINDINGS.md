# Heme/Coagulation Belief Findings

Date: 2026-06-28

This is the hematology Chapter B-deep audit.  The goal was to test whether
first-class hemoglobin, hematocrit, INR, PTT, fibrinogen, platelets, perfusion,
and observed transfusion/anticoagulation evidence can support an inferred
bleeding/coagulation reserve belief.

The belief is not a measured lab and is not allowed to claim direct hidden-state
accuracy.  It survives only if it improves downstream observable prediction
versus both:

- baseline `ridge_realfit`;
- a capacity-matched placebo with the same number of additional columns.

## Implemented Artifacts

- `heme_coag_belief.py`
- `eicu_heme_coag_belief_audit.py`
- `eicu_heme_coag_belief_audit.json`
- `eicu_heme_coag_state_belief_audit.json`

The row-level cohort remains local-only:

- `eicu_coagulopathy_heme_transitions_6h.parquet`

## Belief Candidates

Feature belief columns:

- `belief_bleeding_burden`
- `belief_coagulation_reserve`
- `belief_oxygen_carrying_reserve`
- `belief_platelet_reserve`
- `belief_transfusion_pressure`
- `belief_anticoagulation_pressure`
- `belief_hemostatic_stress`
- `belief_observation_confidence`

Explicit predict-update state columns:

- `state_belief_coag_reserve_mean`
- `state_belief_coag_reserve_sd`
- `state_belief_bleeding_burden`
- `state_belief_oxygen_reserve`
- `state_belief_platelet_reserve`
- `state_belief_observation_confidence`
- `state_belief_delta_since_prior`

## Gate

For each target and each split:

- candidate: `ridge_realfit + heme/coag belief`
- baseline: `ridge_realfit`
- placebo: `ridge_realfit + capacity-matched noise`

A target passes only when the candidate significantly beats both baseline and
placebo on held-out patients.

## Feature Belief Result

Random patient-split summary:

| Target | Beats Baseline | Beats Placebo | Passes Both | Median Delta vs Baseline | Median Delta vs Placebo |
|---|---:|---:|---:|---:|---:|
| Hematocrit | 4/7 | 5/7 | 4/7 | -0.084192 | -0.091013 |
| Hemoglobin | 3/7 | 4/7 | 3/7 | -0.023827 | -0.027921 |
| INR | 1/7 | 0/7 | 0/7 | -0.002566 | -0.022477 |
| PTT | 0/7 | 1/7 | 0/7 | +0.096339 | -0.554865 |
| Platelets | 0/7 | 0/7 | 0/7 | +0.014296 | -0.393311 |
| Fibrinogen | 0/7 | 0/7 | 0/7 | n/a | n/a |

Hospital-heldout did not pass any target significantly.  Hemoglobin and
hematocrit had enough rows for evaluation but crossed zero confidence intervals.
INR, PTT, and fibrinogen were too sparse for a stable hospital-heldout claim.

## Explicit State Result

The predict-update state belief was more online-compatible but weaker:

| Target | Beats Baseline | Beats Placebo | Passes Both | Median Delta vs Baseline | Median Delta vs Placebo |
|---|---:|---:|---:|---:|---:|
| Hematocrit | 1/7 | 1/7 | 1/7 | -0.045252 | -0.046552 |
| Hemoglobin | 2/7 | 2/7 | 2/7 | -0.015589 | -0.016422 |
| INR | 0/7 | 2/7 | 0/7 | +0.037404 | +0.040559 |
| PTT | 0/7 | 2/7 | 0/7 | +0.972253 | +0.313681 |
| Platelets | 0/7 | 0/7 | 0/7 | -0.383969 | -0.520747 |
| Fibrinogen | 0/7 | 0/7 | 0/7 | n/a | n/a |

The explicit state is therefore not validated as a robust online heme/coag
belief state.  It should remain a candidate research feature only.

## Interpretation

The heme/coag belief audit gives a useful boundary:

- Hemoglobin and hematocrit carry partial personalization signal.  This likely
  reflects observed anemia/transfusion/perfusion context.
- Coagulation reserve itself is not robustly validated in this 6h eICU window.
  INR/PTT/fibrinogen are sparse, hospital-heldout confidence intervals cross
  zero, and the explicit state does not stabilize the signal.
- Platelets remain better handled by the existing factual router or
  persistence fallback.

This is different from the AKI renal belief result.  The renal belief passed
the strict placebo gate; the heme/coag belief does not.  The correct promotion
boundary is therefore narrow: keep the heme/coag belief as candidate-only
research scaffolding, not as a validated digital-twin state.

## Safety Boundary

- No direct hidden-state accuracy claim is allowed.
- No causal treatment-effect claim is allowed.
- No counterfactual transfusion or anticoagulation claim is allowed.
- No clinical recommendation claim is allowed.
- No checkpoint promotion is allowed.
- No active symbolic-rule promotion is allowed.
- Row-level parquet cohorts remain local-only and ignored by git.

## Next Step

The next useful heme/coag depth work is not another 6h belief tweak.  It is:

- better transfusion-dose normalization;
- longer-horizon hemoglobin/platelet/coagulation evaluation;
- first-class blood-product subtype separation: PRBC, platelets, FFP,
  cryoprecipitate;
- external validation or a larger cohort for sparse INR/PTT/fibrinogen windows.

The longer-horizon audit has since been run.  It validates 24h hemoglobin and
hematocrit factual routing, but INR/PTT/fibrinogen/platelets still fall back to
persistence at 24h/48h.  This supports the same boundary: longer horizons help
observable Hgb/Hct forecasting, but they do not validate the current hidden
coagulation-reserve belief state.

Until then, the validated heme/coag module remains the factual router with
first-class hematology variables, while the bleeding/coagulation belief remains
candidate-only.
