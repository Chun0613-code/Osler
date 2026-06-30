# Whole-Body Trajectory Validation Findings

Date: 2026-06-30

This audit turns the whole-body trajectory and uncertainty contract from a
definition into a partially validated artifact. It fills two pieces:

- intermediate-horizon movement gates for 1h, 3h, and 12h;
- split-conformal interval calibration for selected validated routers.

The result preserves the same rule as the rest of Osler-JEPA: the forecast
object is complete, but motion and intervals are selective.

## Intermediate Horizon Move Audit

Artifact:

- `whole_body_intermediate_horizon_move_audit.json`

The audit ran eICU sepsis and AKI routers at 1h, 3h, and 12h with the same
patient-heldout and hospital-heldout gates used by the 6h/24h/48h routers.

Validated move cells:

| Module | Horizon | Validated moving targets |
|---|---:|---|
| sepsis | 1h | MAP |
| sepsis | 3h | heart rate, lactate, MAP, O2 saturation, respiratory rate, urine output |
| sepsis | 12h | heart rate, lactate, MAP, O2 saturation, respiratory rate, urine output |
| AKI | 1h | MAP |
| AKI | 3h | bicarbonate, MAP, potassium, sodium |
| AKI | 12h | bicarbonate, BUN, MAP, potassium, sodium, urine output |

Median active-window normalized deltas:

| Module | Horizon | Median delta vs persistence |
|---|---:|---:|
| sepsis | 1h | -0.007543 |
| sepsis | 3h | -0.028564 |
| sepsis | 12h | -0.068102 |
| AKI | 1h | -0.007428 |
| AKI | 3h | -0.032122 |
| AKI | 12h | -0.064360 |

Interpretation:

- 1h movement is narrow and mostly perfusion-driven.
- 3h exposes electrolyte, acid-base, and respiratory/inflammatory movement.
- 12h begins to expose slower renal physiology, including AKI BUN and urine
  output.
- Unsupported cells remain persistence fallback. The layer does not pretend
  every target-horizon pair is learned.

## Split-Conformal Coverage Audit

Artifact:

- `whole_body_conformal_coverage_audit.json`

The audit calibrated residual intervals per target, horizon, and selected
source, with a primary 90% interval target. A cell passes only when empirical
coverage falls inside 0.87-0.93 across patient-heldout splits and
hospital-heldout evaluation.

Results:

- sepsis 6h active-only: 8/8 hospital-heldout targets passed.
- sepsis 6h all-windows: 7/8 hospital-heldout targets passed.
- AKI 24h active-only and all-windows: 7/7 targets passed.
- AKI 48h active-only and all-windows: 7/7 targets passed.

The only systematic miss was sepsis all-windows vasopressor requirement. It is a
persistence fallback target with a zero-width interval; hospital-heldout
coverage was 0.930301 and random-split all-window coverage was usually slightly
above the accepted upper bound. It stays marked as not interval-calibrated for
all-window display.

This means the calibrated interval layer is valid for:

- sepsis 6h active-window forecasts;
- AKI 24h forecasts;
- AKI 48h forecasts;
- most sepsis 6h all-window physiologic targets except the vasopressor proxy.

Cells without a passing calibration artifact must continue to expose
`interval_status = needs_calibration_audit` and no numeric lower/upper band.

## Boundary

These findings validate factual trajectory and interval behavior only. They do
not allow causal treatment-effect claims, counterfactual treatment planning,
clinical authority, runtime treatment authority, checkpoint promotion,
complete-human-simulation claims, or active symbolic-rule promotion.
