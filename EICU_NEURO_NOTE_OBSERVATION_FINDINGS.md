# eICU Neuro-Note Observation Findings

Date: 2026-07-02

This pass tests whether eICU nursing/progress-note structured flowsheets can
deepen the acute-neuro observation layer beyond physiologic proxies. It reuses
the radiology-note lesson: text-derived observations are allowed only when
timestamp-valid, and prediction/imputation authority requires the same robust
held-out gates as numeric targets.

## What Was Built

`eicu_neuro_note_observation_extract.py` extracts timestamped neurologic
observations from:

- `nurseCharting.csv.gz`
- `nurseAssessment.csv.gz`
- `physicalExam.csv.gz`

Target contract:

- `neuro_gcs`
- `neuro_sedation_score`
- `neuro_delirium_present`
- `neuro_mental_abnormal`
- `neuro_pupils_abnormal`
- `neuro_motor_abnormal`

The first bounded run attaches these targets to the existing
`eicu_acute_neuro_transitions_6h.parquet` cohort.

## Leakage Guard

- current `neuro_*_t` observations use only rows with `note_time <= anchor`,
  with a 24h lookback window;
- future `neuro_*_tp6` labels use rows after the anchor and nearest to
  `anchor + 6h` within +/- 2h;
- same-time nowcasting excludes every `neuro_*` feature, preventing one
  neurologic assessment row from predicting another row in the same assessment;
- future forecasting may use current `neuro_*_t` observations because those are
  known before the forecast horizon.

## Extraction Scale

Row-level outputs remain local-only and are not committed.

- acute-neuro transition rows: 16,238
- stays: 1,390
- subjects: 1,183
- hospitals: 12
- extracted neuro-note events: 108,418
- event-covered stays: 1,359

Paired current/future support:

| Target | Paired Rows |
|---|---:|
| `neuro_delirium_present` | 11,714 |
| `neuro_gcs` | 11,385 |
| `neuro_sedation_score` | 0 |
| `neuro_mental_abnormal` | 0 |
| `neuro_pupils_abnormal` | 0 |
| `neuro_motor_abnormal` | 0 |

The broad eICU tables contain pupil, motor, and mental-status fields, but those
fields did not have usable support inside this bounded acute-neuro transition
window. The first deep neuro-note target is therefore GCS plus delirium/CAM
status, not the full neurologic exam.

## Gate Result

`eicu_neuro_note_coverage_audit.py` ran the same aggregate gate family used by
the numeric observation layer:

- seven patient-heldout split seeds;
- hospital-heldout validation;
- capacity-matched placebo ridge control;
- median baseline for same-time nowcast;
- persistence baseline for 6h forecast;
- conformal interval gate for any validated forecast target.

No neuro-note target passed the robust validation gate:

| Axis | Eligible Targets | Validated Targets |
|---|---:|---:|
| same-time nowcast | 6 | 0 |
| 6h future forecast | 6 | 0 |
| calibrated interval | 6 | 0 |

Important near-miss signals:

- `neuro_gcs` nowcast beats baseline and placebo on the hospital-heldout
  evaluation, but discovery selection is not stable across seven patient splits.
- `neuro_delirium_present` forecast beats persistence and placebo on the
  hospital-heldout evaluation, but seven-seed patient splits are not stable.

These are candidate signals, not validated capabilities.

## Interpretation

This is the second clean note-backed measurement-depth result after radiology.
The pattern is now clearer:

- free text / structured note rows can expand what Osler-JEPA can observe;
- those observations do not automatically become values that can be imputed or
  forecast from structured physiology;
- local signals must still pass discovery, patient-heldout, hospital-heldout,
  and placebo gates before entering the validated observation layer.

For neuro specifically, eICU flowsheets meaningfully add observed GCS and
delirium status. They do not yet validate robust GCS/delirium imputation or
forecasting. The correct contract is candidate-only observed evidence.

## Safety Boundary

Allowed:

- timestamp-valid shadow observation of extracted neuro-note findings;
- capability reporting;
- future expansion to richer note targets under the same timestamp contract.

Forbidden:

- same-time imputation of `neuro_*` targets;
- 6h forecasting of `neuro_*` targets;
- numeric uncertainty display for `neuro_*` targets;
- causal treatment-effect claims;
- counterfactual treatment planning;
- clinical recommendation authority;
- runtime treatment authority;
- checkpoint promotion;
- active symbolic-rule promotion.
