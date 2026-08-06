# MIMIC-IV-ED Observation Scene Findings

Date: 2026-07-02

This pass tests whether the validated observation recipe transfers from ICU
data to a new care scene: the emergency department. MIMIC-IV-ED is structured,
timestamped, and table-based, so it tests a different axis from note-derived
measurement depth. It is not a treatment or causal module.

## Cohorts

`mimiciv_ed_observation_extract.py` maps ED `triage.csv.gz` and
`vitalsign.csv.gz` into the same aggregate observation contract used by the ICU
whole-body layer. Row-level parquet outputs remain local-only and are not
committed.

| Horizon | Rows | Subjects | Eligible Paired Targets |
|---:|---:|---:|---:|
| 1h | 348,633 | 107,445 | 8 |
| 3h | 515,782 | 137,831 | 8 |
| 6h | 345,629 | 93,811 | 8 |

Supported ED variables:

- acuity
- systolic blood pressure
- diastolic blood pressure
- calculated MAP
- heart rate
- respiratory rate
- oxygen saturation
- temperature
- pain score

## Gates

`mimiciv_ed_observation_coverage_audit.py` uses the same fail-closed audit
shape as the ICU observation layer:

- seven patient-heldout random splits;
- arrival-transport heldout;
- chronological heldout;
- same-time nowcast must beat median baseline and capacity-matched placebo;
- future forecast must beat persistence and capacity-matched placebo;
- split-conformal intervals are shown only when calibrated.

## Results

| Horizon | Validated Nowcast | Validated Forecast | Validated Interval |
|---:|---:|---:|---:|
| 1h | 3 / 9 | 2 / 8 | 2 / 8 |
| 3h | 5 / 9 | 5 / 8 | 5 / 8 |
| 6h | 4 / 9 | 6 / 8 | 6 / 8 |

Validated nowcast targets:

- 1h cohort: `dbp`, `map`, `sbp`
- 3h cohort: `acuity`, `dbp`, `map`, `pain`, `sbp`
- 6h cohort: `acuity`, `dbp`, `map`, `sbp`

Validated forecast and interval targets:

- 1h: `map`, `sbp`
- 3h: `dbp`, `map`, `respiratory_rate`, `sbp`, `temperature`
- 6h: `dbp`, `heart_rate`, `map`, `respiratory_rate`, `sbp`, `temperature`

External stress-test deltas are consistently negative for validated future
targets. Examples:

| Horizon | Target | Arrival-Transport Heldout Delta | Time Heldout Delta |
|---:|---|---:|---:|
| 1h | `map` | -0.885 | -0.968 |
| 1h | `sbp` | -1.123 | -1.306 |
| 3h | `respiratory_rate` | -0.374 | -0.274 |
| 3h | `temperature` | -0.119 | -0.117 |
| 6h | `heart_rate` | -2.619 | -2.181 |
| 6h | `map` | -2.408 | -2.169 |
| 6h | `sbp` | -2.356 | -2.662 |

## Interpretation

The ED scene succeeds where note-derived targets did not. The difference is
observability:

- note-derived radiology and neuro findings increased observed evidence but did
  not become robust predictive coverage;
- ED vitals are dense, timestamped physiology, so the existing observation
  recipe transfers cleanly.

This adds a new coverage axis: care setting. Osler-JEPA is no longer only an ICU
observation layer; it now has validated pre-ICU ED-scene factual observation
coverage for dense vital-sign physiology.

## Boundary

Allowed:

- ED-scene same-time state completion for validated nowcast targets;
- ED-scene factual vital-sign forecasting for validated target-horizon cells;
- calibrated 90% interval display for validated forecast cells;
- capability reporting.

Forbidden:

- causal treatment-effect claims;
- ED medication effect claims;
- counterfactual planning;
- clinical recommendation authority;
- runtime treatment authority;
- checkpoint promotion;
- active symbolic-rule promotion;
- complete-human simulation claims.
