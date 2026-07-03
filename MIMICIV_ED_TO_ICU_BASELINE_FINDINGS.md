# MIMIC-IV-ED to Early-ICU Baseline Findings

Date: 2026-07-02

This pass tests the uncertain second use of ED data: whether a patient's ED
trajectory improves early ICU factual prediction beyond what is already known
from the early ICU state.

The gate is intentionally stricter than the ED-scene audit.  The candidate is
`ICU current state + prior ED trajectory`.  To validate, it must beat all three
baselines:

- persistence;
- ICU-only ridge;
- capacity-matched ED-feature placebo ridge.

It must also pass seven patient-heldout random splits, first-careunit heldout,
and chronological heldout.  Row-level outputs remain local-only.

## Cohorts

`mimiciv_ed_to_icu_baseline_extract.py` links MIMIC-IV-ED stays to MIMIC-IV ICU
stays only when the ED stay belongs to the same admission and ends before ICU
`intime`.  This avoids using ED observations that occur after ICU admission.

| Horizon | Rows | Subjects | Linked Prior ED->ICU Stays | Eligible Paired Targets |
|---:|---:|---:|---:|---:|
| 1h | 21,885 | 18,285 | 27,761 | 47 |
| 3h | 22,140 | 18,463 | 27,692 | 51 |
| 6h | 22,283 | 18,574 | 27,476 | 52 |

The audit evaluates the dense early-ICU vital targets:

- `heart_rate`
- `map`
- `o2sat`
- `respiratory_rate`
- `temperature`

## Results

No target validates:

| Horizon | Evaluated Targets | Validated ED-Increment Targets | Validated Intervals |
|---:|---:|---:|---:|
| 1h | 5 | 0 | 0 |
| 3h | 5 | 0 | 0 |
| 6h | 5 | 0 | 0 |

This means ED trajectory features are not yet a validated early-ICU baseline
under the strict contract.

## Near-Miss Structure

The result is not flat.  Heart rate and MAP show real heldout signal at longer
horizons, but they do not pass the full nested gate.

Examples:

| Horizon | Target | Random Heldout Beats ICU-Only | First-Careunit Delta vs ICU-Only | Time Delta vs ICU-Only |
|---:|---|---:|---:|---:|
| 1h | `map` | 1 / 7 | -0.105 | -0.068 |
| 3h | `heart_rate` | 7 / 7 | -0.102 | -0.104 |
| 3h | `map` | 7 / 7 | -0.175 | -0.176 |
| 6h | `heart_rate` | 7 / 7 | -0.223 | -0.204 |
| 6h | `map` | 7 / 7 | -0.224 | -0.196 |

The failure point is selection stability: the ED+ICU model does not get selected
in all seven discovery splits.  For example, `heart_rate` at 6h is selected in
6/7 splits and `map` at 6h is selected in 4/7 splits.  Under Osler-JEPA's
promotion discipline, that is candidate-only.

## Interpretation

The prior expectation was mostly right:

> ED data works as its own scene, but ED history does not automatically improve
> early ICU forecasting once the early ICU state is already known.

The useful distinction is:

- ED-scene prediction is validated: dense ED vitals can be forecast inside the
  ED setting.
- ED-to-ICU transfer is not validated: prior ED trajectory does not robustly add
  incremental signal beyond early ICU state under the current gate.

The heart-rate/MAP near-miss is worth keeping as a research clue, but it should
not enter runtime prediction or calibrated intervals.

## Boundary

Allowed:

- capability reporting;
- research-only follow-up on `heart_rate` and `map` ED-to-ICU transfer.

Forbidden:

- early-ICU runtime forecasting from ED trajectory;
- numeric uncertainty intervals for ED-to-ICU transfer;
- ED medication-effect claims;
- causal or counterfactual treatment claims;
- clinical recommendation authority;
- checkpoint promotion;
- active symbolic-rule promotion.
