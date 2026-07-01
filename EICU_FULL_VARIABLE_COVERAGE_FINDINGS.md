# eICU Full Numeric-Variable Coverage Findings

Date: 2026-07-01

This artifact records the first full numeric-variable coverage sweep for the
eICU whole-body observation layer.  It is the literal "no measured variable left
unevaluated" pass inside the existing local eICU transition cohorts.

The audit is non-causal and aggregate-only.  It does not claim treatment
effects, counterfactual planning, clinical authority, runtime authority,
checkpoint promotion, or active symbolic-rule promotion.

## What Was Swept

The audit scans every numeric `*_t` target in the configured 6h eICU transition
cohorts, excluding active labels and identifiers.  A target enters:

- the nowcast gate when its current `target_t` exists;
- the forecast gate when both `target_t` and `target_tp6` exist;
- the interval gate when its forecast gate passes.

Total coverage:

| Gate | Eligible module-targets | Validated |
|---|---:|---:|
| same-time nowcast | 242 | 71 |
| 6h factual forecast | 242 | 59 |
| 90% conformal interval | forecast-validated targets only | 51 |

All 242 eligible numeric module-targets were evaluated.  Targets that did not
pass are explicitly left as fallback or missing.

## New Coverage Compared With The Curated Layer

The earlier curated layer had validated 41 same-time nowcast cells.  The full
numeric sweep expands that to 71.  New or strengthened coverage includes:

- sepsis: bicarbonate, pH, platelets, potassium, sodium, temperature, WBC;
- AKI: heart rate, respiratory rate, temperature;
- respiratory: BUN and creatinine nowcast;
- skin/wound: BUN and potassium nowcast;
- toxic/metabolic: BUN, chloride, sodium nowcast;
- endocrine stress: BUN nowcast;
- GI/nutrition: bicarbonate nowcast;
- cardiac injury: BUN nowcast;
- hepatic failure: pH, potassium, WBC nowcast;
- heme/coagulation: BUN and potassium nowcast.

The forecast sweep also validates 59 module-target 6h cells.  Many are dense or
moderately dense physiology targets: MAP, heart rate, respiratory rate,
oxygenation, glucose, potassium, bicarbonate, pH, platelets, WBC, and
temperature in contexts where the held-out gates support motion.

Of those forecast cells, 51 also pass the split-conformal interval gate.

## What Stayed Closed

The full sweep does not make every measured variable predictable.  The same
walls remain:

- immune/inflammatory nowcast and forecast stay closed under the strict gate;
- rhabdomyolysis nowcast remains closed and only potassium forecasts at 6h;
- cardiovascular and acute-neuro nowcasting stay closed, though some vital-sign
  forecasts pass;
- sparse deep biomarkers such as troponin, BNP, CK-MB, fibrinogen, INR/PTT,
  endocrine hormones, and many hepatic/GI enzymes still fail one or more gates.

This is the desired behavior.  The sweep proves the variables were evaluated,
not that Osler is allowed to fill or move all of them.

## Interpretation

The eICU observation layer is now covered in the strictest local sense:

> every numeric target visible in the configured 6h eICU transition cohorts has
> passed through nowcast, forecast, and interval eligibility gates.

The result confirms the earlier pattern at larger variable coverage:

- dense vital/perfusion variables often forecast;
- cross-sectional chemistry panels often nowcast;
- some chemistry variables can both nowcast and forecast;
- sparse deep biomarkers remain fallback/missing.

This makes the whole-body object more complete without making it less honest.

## Runtime Contract Update

The unified `whole_body_state_forecast` object now uses:

- 71 validated nowcast module-target cells;
- 59 validated full-variable 6h forecast cells;
- 51 calibrated full-variable 6h interval cells;
- explicit fallback/missing status for all non-validated numeric targets.

The full sweep updates coverage only inside eICU.  It does not prove
cross-database generalization; that is the next external-validity chapter.

## Safety Boundary

- Row-level outputs are not committed.
- Patient identifiers are not included.
- Same-time state completion is allowed only for validated nowcast cells.
- Future movement is allowed only for validated forecast cells.
- Numeric intervals are allowed only for validated interval cells.
- Causal, counterfactual, clinical, runtime-treatment, checkpoint-promotion,
  complete-human-simulation, and active-rule-promotion claims remain closed.

