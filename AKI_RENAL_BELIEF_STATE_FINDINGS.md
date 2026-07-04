# AKI Renal Belief State Findings

Date: 2026-06-27

This audit upgrades the previous renal belief features into an explicit
predict-update belief state.  The state tracks a Gaussian renal reserve/GFR
proxy with:

- prediction from the previous belief and elapsed time;
- process uncertainty growth under renal stress;
- update from current creatinine/BUN/urine/MAP observations;
- posterior uncertainty;
- downstream-observable validation only.

It is still not a measured GFR and it does not grant hidden-state accuracy,
causal, counterfactual, clinical, checkpoint, or active-rule claims.

## Candidate

Implemented files:

- `aki_renal_belief.py`
- `eicu_aki_belief_audit.py`
- `eicu_aki_renal_belief_state_audit.json`

State-belief columns:

- `state_belief_renal_reserve_mean`
- `state_belief_renal_reserve_sd`
- `state_belief_renal_stress`
- `state_belief_observation_confidence`
- `state_belief_delta_since_prior`

## Gate

The same downstream gate is used:

- baseline: `ridge_realfit`
- candidate: `ridge_realfit + explicit_renal_belief_state`
- placebo: `ridge_realfit + capacity_matched_noise_features`

A target passes only if candidate significantly beats both baseline and placebo
on held-out patients.

## Random Patient Splits

Seven split seeds were evaluated at 24h and 48h with 500 bootstrap samples.

| Horizon | Target | Passes Both Gates | Median Delta vs Baseline | Median Delta vs Placebo |
| --- | --- | ---: | ---: | ---: |
| 24h | creatinine | 0/7 | 0.000860 | 0.000717 |
| 24h | BUN | 7/7 | -0.061289 | -0.060574 |
| 24h | urine output | 7/7 | -0.869213 | -0.874726 |
| 48h | creatinine | 0/7 | 0.001270 | 0.001260 |
| 48h | BUN | 6/7 | -0.067651 | -0.068582 |
| 48h | urine output | 7/7 | -0.788519 | -0.799984 |

## Hospital-Heldout

At 24h:

- BUN passes:
  - delta vs baseline -0.070984
  - 95% CI [-0.097137, -0.047994]
- urine output passes:
  - delta vs baseline -0.509590
  - 95% CI [-0.832112, -0.177190]
- creatinine does not pass:
  - delta vs baseline 0.000977
  - 95% CI [-0.000703, 0.002546]

At 48h:

- BUN passes:
  - delta vs baseline -0.065847
  - 95% CI [-0.114217, -0.021796]
- urine output is directionally positive but not significant:
  - delta vs baseline -0.280333
  - 95% CI [-0.831174, 0.247708]
- creatinine does not pass:
  - delta vs baseline 0.000932
  - 95% CI [-0.001586, 0.003255]

## Interpretation

The explicit predict-update renal belief state is partially validated:

- it reliably improves BUN forecasting at 24h/48h;
- it improves 24h urine output and is directionally positive at 48h;
- it does not improve creatinine over the already-strong baseline.

This is exactly the type of boundary the belief gate is supposed to reveal.  The
state belief should be promoted only as a candidate personalization signal for
BUN and short-to-mid horizon urine output, not as a universal renal target
improver.

The earlier feature-belief audit remains stronger for creatinine.  The explicit
state version is more online-compatible, but the current state equations lose
some creatinine signal.  The next refinement should focus on the predict-update
equation, not on relaxing the gate.

## V2 Follow-Up

`AKI_RENAL_BELIEF_STATE_V2_FINDINGS.md` implements that refinement by adding a
creatinine-specific predict-update state dimension.  The v2 state fixes the main
creatinine failure:

- 24h creatinine passes 7/7 random splits and hospital-heldout
- 48h creatinine passes 5/7 random splits and hospital-heldout
- BUN remains strongly positive

The v1 state should therefore be treated as a useful intermediate audit, not the
current best explicit state.

## Boundary

- No raw rows are committed.
- No patient identifiers are included.
- No direct hidden-state accuracy claim is allowed.
- No causal claim is allowed.
- No counterfactual claim is allowed.
- No clinical claim is allowed.
- No checkpoint or active-rule promotion is allowed.
