# AKI Renal Belief Findings

Date: 2026-06-27

This is the first Chapter B-deep experiment: a patient-specific renal belief
layer for AKI.  It is not designed to beat persistence directly.  The
long-horizon router already does that.  The question here is narrower:

Can an interpretable renal reserve/GFR proxy improve downstream observable
prediction beyond an already-strong `ridge_realfit` source, and beyond a
capacity-matched placebo?

## Candidate

Implemented files:

- `aki_renal_belief.py`
- `eicu_aki_belief_audit.py`
- `eicu_aki_renal_belief_audit.json`

Belief features:

- renal reserve proxy
- azotemia load
- oliguria burden
- perfusion deficit
- RRT context
- observation confidence

These are not measured labs and are not treated as direct ground truth.  The
only validation gate is downstream observable prediction.

## Gate

For each target and horizon, compare:

- baseline: `ridge_realfit`
- candidate: `ridge_realfit + renal_belief_features`
- placebo: `ridge_realfit + capacity_matched_noise_features`

The candidate passes only if it significantly beats both baseline and placebo on
held-out patients.  This prevents mistaking extra model capacity for a real
belief signal.

## Random Patient Split Results

Seven patient split seeds were evaluated at 24h and 48h with 500 bootstrap
samples per target gate.

| Horizon | Target | Candidate Beats Baseline | Candidate Beats Placebo | Median Delta vs Baseline | Median Delta vs Placebo |
| --- | --- | ---: | ---: | ---: | ---: |
| 24h | creatinine | 7/7 | 7/7 | -0.005324 | -0.005387 |
| 24h | BUN | 7/7 | 7/7 | -0.104042 | -0.105492 |
| 24h | urine output | 7/7 | 7/7 | -0.973778 | -0.980189 |
| 48h | creatinine | 7/7 | 7/7 | -0.006845 | -0.007025 |
| 48h | BUN | 7/7 | 7/7 | -0.129525 | -0.132678 |
| 48h | urine output | 7/7 | 7/7 | -0.828486 | -0.834799 |

## Hospital-Heldout Results

At 24h, all three renal targets pass:

- creatinine: candidate delta vs baseline -0.003355,
  95% CI [-0.006763, -0.000357]
- BUN: candidate delta vs baseline -0.104589,
  95% CI [-0.137591, -0.071917]
- urine output: candidate delta vs baseline -0.542393,
  95% CI [-0.790525, -0.303014]

At 48h, creatinine and BUN pass; urine output improves directionally but does
not pass hospital-heldout significance:

- creatinine: candidate delta vs baseline -0.005779,
  95% CI [-0.008679, -0.003069]
- BUN: candidate delta vs baseline -0.110460,
  95% CI [-0.157313, -0.065230]
- urine output: candidate delta vs baseline -0.407611,
  95% CI [-0.883438, 0.089881]

## Interpretation

The renal belief layer passes its downstream gate for creatinine and BUN at both
24h and 48h.  It also passes for 24h urine output and is directionally positive
for 48h urine output.

This is the correct role for mechanism/belief after the long-horizon audit:

- not to rescue 6h slow-target persistence;
- not to make causal treatment claims;
- not to claim direct hidden-state accuracy;
- yes to patient-specific personalization that improves observable renal
  forecasts beyond baseline and placebo.

The next B-deep step should convert this feature belief into an explicit
predict-update renal reserve/GFR belief state and test the same downstream gate.

## Explicit State Follow-Up

The explicit predict-update state version has now been tested separately in
`AKI_RENAL_BELIEF_STATE_FINDINGS.md`.

Result:

- BUN passes the state-belief gate at 24h/48h;
- urine output passes at 24h and is directionally positive at 48h;
- creatinine does not pass.

So the feature belief remains the broader personalization source, while the
explicit online-compatible state belief is currently validated only for BUN and
part of urine-output forecasting.

## Boundary

- No raw rows are committed.
- No patient identifiers are included.
- No direct hidden-state accuracy claim is allowed.
- No causal claim is allowed.
- No counterfactual claim is allowed.
- No clinical claim is allowed.
- No checkpoint or active-rule promotion is allowed.
