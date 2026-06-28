# AKI Renal Belief State V2 Findings

Date: 2026-06-27

This audit deepens the explicit renal belief state by giving creatinine its own
predict-update state dimension.  The previous state belief helped BUN and part
of urine-output forecasting, but did not pass the downstream gate for
creatinine.  The hypothesis here is that creatinine should not be forced through
one generic renal reserve proxy; it needs its own level, slope, uncertainty, and
innovation state.

## Candidate

Implemented files:

- `aki_renal_belief.py`
- `eicu_aki_belief_audit.py`
- `eicu_aki_renal_belief_state_v2_audit.json`

State-v2 belief columns:

- renal reserve mean
- renal reserve uncertainty
- renal stress
- observation confidence
- reserve delta since prior
- creatinine level
- creatinine slope
- creatinine uncertainty
- creatinine innovation
- creatinine observation confidence

The state is online-compatible: it uses current and previous observations within
the stay, not future target columns.

## Gate

The same downstream gate is preserved:

- baseline: `ridge_realfit`
- candidate: `ridge_realfit + explicit_renal_belief_state_v2`
- placebo: `ridge_realfit + capacity_matched_noise_features`

The candidate passes only if it significantly beats both baseline and placebo on
held-out patients.

## Random Patient Splits

Seven split seeds were evaluated at 24h and 48h with 500 bootstrap samples.

| Horizon | Target | Passes Both Gates | Median Delta vs Baseline | Median Delta vs Placebo |
| --- | --- | ---: | ---: | ---: |
| 24h | creatinine | 7/7 | -0.006026 | -0.006057 |
| 24h | BUN | 7/7 | -0.206556 | -0.209537 |
| 24h | urine output | 7/7 | -0.698308 | -0.732736 |
| 48h | creatinine | 5/7 | -0.006051 | -0.006155 |
| 48h | BUN | 7/7 | -0.226826 | -0.228928 |
| 48h | urine output | 7/7 | -0.789695 | -0.803576 |

## Hospital-Heldout

At 24h, all three targets pass:

- creatinine:
  - delta vs baseline -0.005013
  - 95% CI [-0.008286, -0.002116]
- BUN:
  - delta vs baseline -0.210704
  - 95% CI [-0.250995, -0.168905]
- urine output:
  - delta vs baseline -0.412348
  - 95% CI [-0.747311, -0.071996]

At 48h, creatinine and BUN pass; urine output remains directionally positive but
not significant on hospital-heldout:

- creatinine:
  - delta vs baseline -0.004563
  - 95% CI [-0.008617, -0.000331]
- BUN:
  - delta vs baseline -0.180596
  - 95% CI [-0.250693, -0.113156]
- urine output:
  - delta vs baseline -0.186820
  - 95% CI [-0.717781, 0.344232]

## Interpretation

The creatinine-specific state dimension fixes the main failure of the previous
explicit state belief.

Validated scope:

- 24h creatinine, BUN, and urine output;
- 48h creatinine and BUN;
- 48h urine output is positive but not hospital-heldout significant.

This is now a real online-compatible personalization layer: it is a
predict-update state, it beats baseline, and it beats capacity-matched placebo.
It still does not claim direct hidden-state accuracy or causal treatment effect.

## Boundary

- No raw rows are committed.
- No patient identifiers are included.
- No direct hidden-state accuracy claim is allowed.
- No causal claim is allowed.
- No counterfactual claim is allowed.
- No clinical claim is allowed.
- No checkpoint or active-rule promotion is allowed.

