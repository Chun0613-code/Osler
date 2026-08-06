# Target-Specific Rescue Findings (2026-08-05)

## Scope

This audit tested three alternatives to repeatedly changing the shared JEPA
architecture:

1. remove the 1,000-stay cap for a near-miss target;
2. represent oxygenation as a calibrated future hypoxemia event;
3. represent creatinine with a change/no-change hurdle and a robust
   conditional-median delta model.

All inputs are anchor-time measurements, causal history, and strictly
pre-anchor treatment context. Hospital, care-unit, and database provenance are
retained for audit splits but excluded from model features. Every result is
factual, research-only, non-causal, and non-clinical.

## Result 1: The heart-rate failure was a cohort cap

The earlier external audit used at most 1,000 stays. Its care-unit holdout had
only 42 subjects and failed. Re-running the unchanged hierarchical candidate
on the complete aligned event cohort used:

- 100,328 rows;
- 1,799 subjects;
- 185 hospitals.

Both cells passed every patient, hospital, care-unit, forward-time,
module-balanced, bootstrap, SVD, and conformal requirement:

| Cell | Held-out rows | Held-out subjects | Patient-equal MAE delta | 95% CI | 90% coverage |
|---|---:|---:|---:|---:|---:|
| heart_rate@3h | 21,102 | 1,297 | -0.0305 | [-0.0349, -0.0265] | 0.8842 |
| heart_rate@6h | 19,762 | 1,263 | -0.0448 | [-0.0497, -0.0401] | 0.9064 |

The prior failure was therefore a power/sampling problem, not evidence that
the target required a different latent architecture.

## Result 2: Oxygen carried event signal, not stable point-value signal

The initial future-hypoxemia classifier beat current-state persistence in
almost every split but usually lost to a prevalence-only probability. Its raw
probabilities were too variable. A discovery-only calibration split selected a
shrinkage factor toward training prevalence; test labels were never used for
selection.

After calibration, both event cells passed 7/7 patient splits and all three
external domains:

```text
hypoxemia_event@1h
hypoxemia_event@3h
```

Across the ten patient/external splits, selected shrinkage ranged from 0.15 to
0.70. The event model consistently beat both:

- persistence event status: `current_o2sat <= 92`;
- discovery prevalence probability.

This result does **not** promote a continuous O2-saturation value forecast.
The runtime target is explicitly `hypoxemia_event`, with a probability output
and the definition `future_o2sat <= 92`.

## Result 3: Creatinine needed a robust 24-hour change model

A linear changed-only delta model was unstable and could not beat persistence.
Replacing its conditional mean with a robust conditional-median tree model,
selecting hurdle/shrinkage policy only on patient-disjoint calibration rows,
and clipping deltas using the discovery distribution produced consistent
24-hour value gains.

The remaining failures were uncertainty failures, not value failures:

- a fixed interval over-covered one patient split and the care-unit domain;
- normalized conformal fixed those but initially over-covered the later-stay
  forward-time domain;
- adding legal `hours_since_onset` to the residual-scale model calibrated the
  later-stay error without changing the point forecast.

Final `creatinine@24h` result:

- 7/7 patient splits passed persistence, plain-ridge, and conformal gates;
- hospital heldout passed;
- care-unit heldout passed;
- forward-time heldout passed;
- every value comparison had patient-bootstrap CI below zero;
- every 90% interval coverage was within 0.87-0.93.

## Runtime boundary

The validated outputs are kept under separate task contracts:

| Runtime cell | Output type | Registry |
|---|---|---|
| heart_rate@3h | continuous value | `heart_rate_full_cohort_validated_registry_20260805.json` |
| heart_rate@6h | continuous value | `heart_rate_full_cohort_validated_registry_20260805.json` |
| hypoxemia_event@1h | binary probability | `hypoxemia_event_validated_registry_20260805.json` |
| hypoxemia_event@3h | binary probability | `hypoxemia_event_validated_registry_20260805.json` |
| creatinine@24h | continuous hurdle value | `creatinine24_hurdle_validated_registry_20260805.json` |

Catalog `whole_body_runtime_candidate_catalog_v7.json` records these sources
without removing any older validated cell. Contract and unit mismatches fail
closed to the existing validated source or persistence.

## Main conclusion

The remaining cells do not all have the same failure cause. This round found
three distinct and actionable causes:

1. insufficient external-test support (heart rate);
2. wrong output object plus probability miscalibration (oxygenation);
3. wrong loss/statistic plus heteroscedastic uncertainty (creatinine).

Therefore the next improvements should begin with a per-cell failure audit,
not another global increase in JEPA latent capacity.
