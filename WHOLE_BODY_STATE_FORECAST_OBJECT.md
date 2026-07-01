# Whole-Body State Forecast Object

Date: 2026-07-01

This artifact closes the non-causal observation/prediction assembly step.  It
defines one queryable object, `whole_body_state_forecast`, that combines the
three separately validated objects:

1. `nowcast` - current-state completion for missing present-tense values.
2. `forecast` - future factual physiology cells at validated horizons.
3. `interval` - calibrated uncertainty when a conformal coverage gate passed.

The object is complete in shape but selective in authority.  It may describe a
cell as observed, nowcastable, forecastable, fallback, uncalibrated, or missing.
It may not invent unsupported physiology.

## Output Shape

```text
whole_body_state_forecast
  patient_time
    patient_key
    anchor_time

  current_state
    target/module cell
      point_estimate
      source: observed | same_time_nowcast_ridge | missing
      status: observed | validated_nowcast_available | unsupported | missing
      can_estimate
      can_move = false

  future_trajectory
    target/module/horizon cell
      point_estimate
      lower
      upper
      interval_level
      source
      status
      can_estimate
      can_move
      interval_status

  uncertainty
    split-conformal calibration policy
    calibrated only when a target/horizon/source gate passed
```

The committed template contains no runtime patient values.  Runtime code may
fill numeric values only when a cell is directly observed or has a validated
nowcast/forecast source.

## Current-State Axis

The full numeric-variable sweep validates 71 module-target current-state
completion cells across 15 cohorts.  These cells can estimate currently missing
values from same-time and historical physiology:

- renal chemistry and electrolyte panels;
- hemoglobin and hematocrit;
- albumin and total protein;
- direct bilirubin and platelets;
- selected dense vital and perfusion variables.

A validated nowcast does not imply that the same variable can be moved into the
future.  It only says the current body state contains enough information to
estimate that value when it is missing.

## Future Axis

The future trajectory axis uses the canonical horizons:

- 1h
- 3h
- 6h
- 12h
- 24h
- 48h

The template now includes:

- validated core 6h fast-physiology cells;
- validated 24-48h renal accumulation cells;
- the 79 all-module intermediate-horizon move cells from the full-body audit.
- the 59 full-variable 6h forecast cells from the eICU numeric coverage sweep.

Unsupported target/horizon pairs fall back to persistence or missing.  This is
the same humility reflex as before, now placed inside one object.

## Uncertainty Axis

Numeric intervals are allowed only when a split-conformal coverage audit passes
for the target, horizon, and source.  The current aggregate interval evidence is:

- 51 full-variable 6h forecast cells pass;
- 135/378 active-window full-body cells pass;
- 151/378 all-window full-body cells pass.

Cells without a passing interval gate must set `interval_status` to
`needs_calibration_audit` and leave `lower`/`upper` empty.

## Human Analogy

This object is the bedside monitor version of Osler-JEPA:

- nowcast = "what do I think the missing current value is?"
- forecast = "what future values am I allowed to move?"
- interval = "how honest is my confidence?"
- fallback/missing = "I do not know enough to say."

It is closer to a disciplined whole-body observer than to a treatment planner.
The treatment-planning key remains outside this artifact and requires external
causal evidence.

## Boundary

This artifact allows:

- shadow observation;
- same-time state completion for validated module-target pairs;
- factual future prediction for validated target/horizon cells;
- calibrated intervals only where coverage passed;
- capability reporting.

This artifact forbids:

- causal treatment-effect claims;
- counterfactual treatment planning;
- clinical recommendation authority;
- runtime treatment authority;
- checkpoint promotion;
- active symbolic-rule promotion;
- complete-human-simulation claims.

## Implementation

The machine-readable template is exposed by:

- `osler_jepa/observation_layer.py`
  - `nowcast_state_grid()`
  - `trajectory_prediction_grid()`
  - `whole_body_state_forecast_schema()`
  - `whole_body_state_forecast_template()`
- `whole_body_state_forecast_contract.json`

The aggregate evidence comes from:

- `WHOLE_BODY_NOWCASTING_FINDINGS.md`
- `whole_body_nowcasting_audit.json`
- `EICU_FULL_VARIABLE_COVERAGE_FINDINGS.md`
- `eicu_full_variable_coverage_audit.json`
- `WHOLE_BODY_TRAJECTORY_UNCERTAINTY_LAYER.md`
- `whole_body_rollout_uncertainty_contract.json`
- `whole_body_all_modules_intermediate_horizon_move_audit.json`
- `whole_body_all_modules_conformal_coverage_audit.json`
