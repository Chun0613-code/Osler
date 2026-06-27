# AKI Long-Horizon Router Findings

Date: 2026-06-27

This audit tests the hypothesis that the failed 6-hour AKI mechanism candidate
was not mainly a mechanism failure, but a horizon mismatch.  Creatinine and BUN
are slow renal accumulation targets; at 6 hours, persistence is often the best
factual predictor because there is little movement to forecast.

## Implementation

The AKI extractor and router were made horizon-aware:

- `eicu_aki_transition_extract.py --horizon-hours 24`
- `eicu_aki_transition_extract.py --horizon-hours 48`
- `eicu_aki_target_router.py --future-suffix tp24`
- `eicu_aki_target_router.py --future-suffix tp48`

The feature builder was also hardened to exclude any future target suffix such
as `_tp24` or `_tp48`, preventing long-horizon leakage into `ridge_realfit`.
The 24h/48h router audits used seven patient split seeds and 500 bootstrap
samples per gate.

## Cohorts

| Horizon | Stays | Subjects | Hospitals | Transitions | Active-AKI Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6h | 43,748 | 37,119 | 204 | 667,799 | 219,417 |
| 24h | 33,975 | 29,508 | 203 | 530,265 | 178,294 |
| 48h | 24,098 | 21,427 | 198 | 397,897 | 135,870 |

Target-pair support for the slow renal targets:

| Horizon | Creatinine Pairs | BUN Pairs | Urine Output Pairs |
| --- | ---: | ---: | ---: |
| 6h | 55,920 | 55,715 | 255,976 |
| 24h | 73,026 | 72,849 | 199,603 |
| 48h | 52,389 | 52,269 | 147,867 |

## Results

Random patient splits:

| Horizon | Active-AKI Median Delta | Significant Splits | Creatinine Source | BUN Source | Urine Source |
| --- | ---: | ---: | --- | --- | --- |
| 6h | -0.044771 | 7/7 | persistence 7/7 | persistence 7/7 | mostly persistence |
| 24h | -0.067735 | 7/7 | ridge_realfit 7/7 | ridge_realfit 7/7 | population_delta 7/7 |
| 48h | -0.106262 | 7/7 | ridge_realfit 7/7 | ridge_realfit 7/7 | population_delta 7/7 |

Hospital-held-out split:

| Horizon | Active Delta | 95% CI | Creatinine Delta | BUN Delta | Urine Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| 6h | -0.039494 | [-0.042375, -0.036591] | 0.000000 | 0.000000 | 0.000000 |
| 24h | -0.072671 | [-0.076890, -0.068759] | -0.027636 | -0.650428 | -1.298092 |
| 48h | -0.110150 | [-0.116736, -0.103889] | -0.032039 | -0.756922 | -2.776454 |

## Interpretation

The 6-hour negative mechanism result was real, but its cause is sharper now:

- at 6h, creatinine/BUN move too slowly, so persistence is the correct factual
  fallback;
- at 24h and 48h, slow renal targets accumulate enough signal for real-fit
  prediction to beat persistence;
- the useful axis is horizon length and observability, not another simple 6h
  mechanism prior.

This does not prove causal treatment effects.  It says a long-horizon factual
AKI router can forecast renal accumulation targets better than persistence when
the prediction task is placed on the right time scale.

## Next Mechanism Boundary

The next renal mechanism attempt should not be another simple 6-hour drift
prior.  It should target the long-horizon setting and add:

- patient-specific renal reserve or GFR belief state;
- RRT dose and ultrafiltration observability;
- fluid balance and urine-output reliability checks;
- explicit handling of intervening treatment between `t` and `t+h`.

Until then, the validated shape is:

- 6h: persistence fallback for creatinine/BUN;
- 24-48h: real-fit source for creatinine/BUN/BUN-like slow renal targets.
