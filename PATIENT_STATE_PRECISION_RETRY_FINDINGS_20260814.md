# Renal Patient-State Precision Retry Findings

Date: 2026-08-14

## Question

Could the six closest renal cells be rescued without relaxing the frozen point,
placebo, external, conformal, or interval-width gates?

## Methods

- BUN 6h/24h: fit-only heteroscedastic residual scale for the matched population
  interval. The patient model and 0.87-0.93 coverage gate were unchanged.
- Creatinine 24h: candidate-only nominal coverage selected inside the calibration
  patients. The population interval remained fixed at 0.90.
- Urine output 1h/3h/6h: renal belief v3 added a causal online urine level,
  slope, uncertainty, innovation, and observation-confidence state. Candidate
  and placebo retained identical neural architecture and parameter count.

All final decisions used deterministic CPU execution, seven patient splits,
hospital, care-unit, and late-stage heldout splits, patient-paired bootstrap,
matched coverage, and significant interval narrowing.

## Results

| Cell | Point gates | Interval gates | Decision |
|---|---:|---:|---|
| creatinine@24h | 7/7 patient + 3/3 external | 7/7 + 3/3 | promoted |
| BUN@24h | 7/7 + 3/3 | 7/7 + 3/3 | promoted |
| urine_output@6h | 7/7 + 3/3 | 7/7 + 3/3 | promoted |
| BUN@6h | 7/7 + 3/3 | 6/7 + 3/3 | rejected |
| urine_output@1h | 6/7 + 2/3 | 7/7 + 3/3 | rejected |
| urine_output@3h | 7/7 + 2/3 | 7/7 + 3/3 | rejected |

Urine v3 improved urine_output@3h from 5/7 to 7/7 patient point gates, but the
care-unit point gate remained unstable. This is useful evidence, not promotion.

## Runtime

The first retry raised the registry to six serialized cells. A second,
predeclared retry used temporal-Mondrian calibration, urine-belief v3, and two
perfusion targets already present in the kidney/perfusion feature contract. It
did not relax any gate. The final registry contains 12 serialized cells:

- creatinine@3h, creatinine@12h, creatinine@24h
- BUN@3h, BUN@6h, BUN@12h, BUN@24h, BUN@48h
- urine_output@6h, urine_output@12h
- MAP@3h, MAP@6h

All six newly added cells pass 7/7 patient splits and 3/3 hospital, care-unit,
and late-stage splits for point accuracy, conformal coverage, and significant
matched interval narrowing. `creatinine@12h` selects its nominal coverage using
calibration patients only. `MAP@3h/6h` are cardiovascular/perfusion targets
predicted by a kidney-local patient-state adapter; the registry records this
distinction explicitly rather than calling MAP a kidney measurement.

Each artifact includes neural weights, the population anchor, preprocessing,
patient-weighted temporal-Mondrian conformal quantiles, metadata, and a SHA-256
manifest. Runtime selects renal belief v2 or v3 and the early/middle/late
calibration regime from artifact metadata. Unsupported cells remain on their
existing validated population source.

## Boundary

These are factual patient-specific forecasts and calibrated intervals. They are
not direct hidden-state truths, treatment effects, clinical recommendations, or
causal claims.
