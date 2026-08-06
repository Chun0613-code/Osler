# MIMIC-IV Cross-Database Observation Coverage Findings

Date: 2026-07-01

This artifact records the first whole-body observation-layer external-validity
pass on credentialed MIMIC-IV v3.1.  It asks whether the eICU-derived
nowcast/forecast/interval recipe still works when moved to a different ICU
database, schema, measurement culture, and patient population.

The audit is non-causal and aggregate-only.  It does not claim treatment
effects, counterfactual planning, clinical authority, runtime authority,
checkpoint promotion, complete-human simulation, or active symbolic-rule
promotion.

## What Was Built

`mimiciv_observation_transition_extract.py` maps MIMIC-IV raw ICU/lab/vital
tables into the same observation contract used by the eICU whole-body layer:

```text
subject_id, stay_id, careunit, t, t_plus
  + target_t
  + target_age_hr
  + target_tp6
```

Only observation variables are extracted: labs, vitals, respiratory charting,
and urine output.  Treatment/action channels are intentionally outside this
audit.

The full local MIMIC-IV extraction produced:

| Quantity | Count |
|---|---:|
| selected ICU stays | 93,224 |
| evaluable stays | 90,190 |
| subjects | 63,307 |
| 6h transition rows | 640,164 |
| care units | 17 |
| numeric targets with current/future pairs | 55 |

The row-level parquet stays local-only under `/private/tmp` and is not
versioned.

## Gate

Because MIMIC-IV is effectively single-center, this audit does not use a
hospital-heldout gate.  It uses three MIMIC-appropriate gates:

- 7 random patient-heldout splits;
- first-careunit heldout;
- chronological early-to-late heldout.

Every candidate ridge source also has to beat:

- persistence for future forecasting;
- median baseline for same-time nowcasting;
- capacity-matched placebo ridge.

Forecast intervals use split-conformal residual calibration and must land in
the same 90% empirical coverage band: 0.87-0.93.

For tractability, each target/mode is evaluated with a deterministic
subject-level cap of 80,000 rows.  The full cohort is still the source
population; the cap prevents one dense target from turning the full matrix gate
into an unbounded runtime job.

## Result

All 55 eligible MIMIC-IV numeric targets were evaluated.

| Gate | Eligible targets | Validated |
|---|---:|---:|
| same-time nowcast | 55 | 23 |
| 6h factual forecast | 55 | 18 |
| 90% conformal interval | forecast-validated targets only | 14 |

Validated same-time nowcast targets:

```text
albumin, anion_gap, bicarbonate, bun, calcium, chloride, fibrinogen,
heart_rate, hematocrit, hemoglobin, ionized_calcium, magnesium, map,
minute_volume, paco2, pao2, ph, phosphate, platelets, potassium,
respiratory_rate, serum_osmolality, sodium
```

Validated 6h factual forecast targets:

```text
anion_gap, bicarbonate, calcium, chloride, heart_rate, hemoglobin, magnesium,
map, minute_volume, paco2, pao2, ph, phosphate, potassium, respiratory_rate,
temperature, urine_output, wbc
```

Validated 90% interval targets:

```text
anion_gap, bicarbonate, calcium, heart_rate, magnesium, map, minute_volume,
paco2, pao2, ph, phosphate, potassium, respiratory_rate, temperature
```

## Interpretation

The eICU pattern survives an external database test, but in bounded form.

What transfers:

- electrolyte and acid-base targets;
- perfusion and vital-sign targets;
- respiratory gas/ventilation-adjacent targets when charted well;
- same-time chemistry-panel completion;
- calibrated intervals for a subset of fast physiology.

What remains closed:

- sparse cardiac injury biomarkers such as troponin and BNP;
- sparse inflammatory markers such as CRP, ESR, and ferritin;
- most endocrine hormones;
- many GI/hepatic enzymes for future movement;
- INR/PTT future movement, despite some same-time fibrinogen nowcasting;
- any treatment-effect or planning claim.

This is the same law seen inside eICU, now under external pressure:

> Dense, mechanistically linked physiology can move or be completed; sparse
> deep biomarkers remain fallback or missing unless the data actually support
> them.

## Why This Matters

The previous eICU sweep proved local full-variable coverage.  This MIMIC-IV
pass proves that the observation-layer recipe is not merely an eICU artifact.
It can be rebuilt on an independent ICU database and still passes strict
held-out gates for a coherent subset of whole-body physiology.

The result does not make Osler a complete human simulator.  It makes Osler a
more externally tested whole-body observer: broad, selective, and explicit about
where it does not know enough.

## Runtime Contract

The MIMIC-IV audit adds external-validation evidence to the
`whole_body_state_forecast` object.  It does not automatically add new
module-specific eICU runtime cells, because this MIMIC pass is target-level
and all-ICU rather than disease-module-specific.

Runtime policy remains:

- current values may be filled only by direct observation or validated nowcast;
- future values may move only in validated forecast cells;
- numeric intervals may be shown only when conformal calibration passed;
- unsupported cells stay at persistence or missing;
- no causal, counterfactual, clinical, or runtime treatment authority.

## Artifacts

- `mimiciv_observation_transition_extract.py`
- `mimiciv_cross_database_coverage_audit.py`
- `mimiciv_observation_transition_report.json`
- `mimiciv_cross_database_coverage_audit.json`

The two JSON reports are aggregate-only.  The MIMIC row-level transition parquet
is local-only and intentionally not versioned.
