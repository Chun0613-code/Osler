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

The MIMIC-IV cross-database pass independently validates 23 target-level
same-time nowcasts on 63,307 subjects.  This is external-validity evidence, not
a new disease-module runtime grant.

The first MIMIC-IV Note pass extracts timestamped chest radiology findings
from free text, but those `rad_*` targets remain candidate-only: 0/6 same-time
nowcast targets pass robust patient, careunit, time, and placebo gates. A
timestamp-valid report finding may be displayed as observed evidence; an
unobserved radiology finding must remain missing.

The eICU neuro-note pass extracts timestamped GCS and delirium/CAM evidence
from nursing/progress-note flowsheets, but those `neuro_*` targets also remain
candidate-only: 0/6 same-time nowcast targets pass robust patient, hospital,
and placebo gates. A timestamp-valid neuro finding may be displayed as observed
evidence; an unobserved neuro-note target must remain missing.

The all-ICU eICU neuro-note follow-up scales the same question to 100,862
subjects and still validates 0/2 GCS/delirium nowcast targets. This rejects the
simple explanation that the bounded acute-neuro miss was only a power problem.

The MIMIC-IV-ED scene audit extends current-state completion to a pre-ICU ED
setting. ED triage and vital-sign tables validate 3/9 same-time nowcast targets
at 1h cohort support, 5/9 at 3h, and 4/9 at 6h. This is a scene extension for
dense ED physiology, not a note-imputation or treatment-effect module.

The ED-to-early-ICU baseline audit tests a different question: whether prior ED
trajectory improves early ICU prediction after the early ICU state is known. It
validates 0/5 dense vital targets at 1h, 3h, and 6h. Heart rate and MAP show
candidate-only near-miss signal at 3h-6h, but no ED-to-ICU transfer cell may
move or display an interval until a later audit passes the full gate.

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

The MIMIC-IV cross-database pass independently validates 18 target-level 6h
forecast cells using patient-heldout, first-careunit-heldout, and chronological
heldout gates.

The MIMIC-IV radiology note pass does not validate any 6h future imaging
finding forecast cells. Those targets stay outside the future trajectory grid
until a later audit passes.

The eICU neuro-note pass does not validate any 6h future GCS, delirium, or
structured neurologic-exam forecast cells. Those targets stay outside the
future trajectory grid until a later audit passes.

The all-ICU follow-up also validates 0/2 GCS/delirium 6h forecast targets;
persistence remains stronger than the note-augmented ridge forecast.

The MIMIC-IV-ED scene audit validates ED factual vital-sign forecasting with
calibrated intervals: 2/8 targets at 1h, 5/8 at 3h, and 6/8 at 6h. These cells
are ED-scene factual predictions only; they do not imply ED medication effects
or causal planning.

The ED-to-ICU baseline pass validates no future ICU forecast cells. Prior ED
trajectory stays outside the ICU future trajectory grid and remains
candidate-only.

Unsupported target/horizon pairs fall back to persistence or missing.  This is
the same humility reflex as before, now placed inside one object.

## Uncertainty Axis

Numeric intervals are allowed only when a split-conformal coverage audit passes
for the target, horizon, and source.  The current aggregate interval evidence is:

- 51 full-variable 6h forecast cells pass;
- 135/378 active-window full-body cells pass;
- 151/378 all-window full-body cells pass.
- 14 MIMIC-IV target-level 6h intervals pass external-database calibration.

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
- `MIMICIV_CROSS_DATABASE_COVERAGE_FINDINGS.md`
- `mimiciv_cross_database_coverage_audit.json`
- `MIMICIV_ED_OBSERVATION_FINDINGS.md`
- `mimiciv_ed_observation_coverage_audit_1h.json`
- `mimiciv_ed_observation_coverage_audit_3h.json`
- `mimiciv_ed_observation_coverage_audit_6h.json`
- `MIMICIV_ED_TO_ICU_BASELINE_FINDINGS.md`
- `mimiciv_ed_to_icu_baseline_audit_1h.json`
- `mimiciv_ed_to_icu_baseline_audit_3h.json`
- `mimiciv_ed_to_icu_baseline_audit_6h.json`
- `MIMICIV_RADIOLOGY_NOTE_OBSERVATION_FINDINGS.md`
- `mimiciv_radiology_note_coverage_audit.json`
- `EICU_NEURO_NOTE_OBSERVATION_FINDINGS.md`
- `eicu_neuro_note_coverage_audit.json`
- `EICU_NEURO_NOTE_ALL_ICU_FINDINGS.md`
- `eicu_neuro_note_all_icu_coverage_audit.json`
