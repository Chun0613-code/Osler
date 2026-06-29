# Body-System Edge-Specific Coupling Findings

Date: 2026-06-29

This is the third cross-system coupling pass after body-system breadth
completion:

1. static upstream feature concatenation;
2. generic temporal predict-update burden states;
3. edge-specific shared physiologic states.

The third pass tests bespoke shared states for each directed organ-system edge,
for example renal-electrolyte buffering, respiratory CO2/ventilation mismatch,
immune capillary-leak/shock burden, heme oxygen-delivery debt, and
cardio-renal perfusion stress.

The gate is unchanged:

- baseline: downstream ridge model without edge-specific coupling features;
- candidate: baseline plus edge-specific shared-state features;
- placebo: baseline plus the same number of random features;
- pass rule: candidate must significantly beat both baseline and placebo on
  held-out patients.

This report is aggregate-only and grants no causal, counterfactual, clinical,
runtime, checkpoint-promotion, or active-rule authority.

## Result

Edge-specific shared states produce more physiologically plausible weak signals
than generic temporal coupling, but still no edge passes the robust 7/7
active-window promotion boundary.  The layer remains candidate-only.

| Edge | Edge-Specific Shared State | Active Pass-Both Targets | All-Window Notes | Interpretation |
|---|---|---|---|---|
| renal -> electrolyte / acid-base | renal reserve x acid buffer / K handling / sodium-water state | bicarbonate 2/7, anion gap 1/7 | bicarbonate 2/7 | best current weak edge; not promoted |
| respiratory -> acid-base | CO2/ventilation mismatch proxy | bicarbonate 1/7 | bicarbonate 3/7 | weak acid-base signal; needs CO2/ventilator detail |
| endocrine -> electrolyte | osmotic/glycemic x electrolyte state | sodium 1/7, bicarbonate 1/7, anion gap 1/7 | sodium 2/7, bicarbonate 1/7 | broader weak signal; not promoted |
| heme/coag -> perfusion | oxygen-delivery debt | heart rate 1/7 | MAP 1/7, heart rate 1/7 | weak oxygen-delivery signal; not promoted |
| cardiovascular -> renal | shock burden x renal reserve / creatinine kinetics | creatinine 1/7 | creatinine median deltas improve but not significant enough | promising direction; likely needs 24h/48h |
| hepatic -> coagulation / platelets | hepatic burden x platelet/acid-base state | none | none | rejected |
| immune / inflammatory -> hemodynamics | capillary-leak / shock state | none | platelets 1/7 | weak all-window immune/coag signal only |

## What Improved

Compared with the generic temporal audit, edge-specific states spread weak
signals into more plausible target channels:

- renal buffering adds anion-gap signal and keeps bicarbonate signal;
- respiratory CO2/ventilation mismatch improves bicarbonate all-window support;
- endocrine osmotic/electrolyte interactions add sodium, bicarbonate, and
  anion-gap weak signals;
- oxygen-delivery debt adds heart-rate signal;
- cardio-renal perfusion stress adds a creatinine active-window signal.

This is the right direction, but not enough to promote a whole-body coupling
layer.

## Interpretation

The audit now establishes a graded picture:

- isolated body-system routers: validated;
- AKI renal belief state: validated for renal downstream observables;
- static cross-system coupling: rejected;
- generic temporal coupling: weak candidate signals only;
- edge-specific shared-state coupling: stronger weak signals, still
  candidate-only.

The limiting factor is no longer "we have no body systems."  It is now
observability and edge-specific physiology.  Several edges likely need richer
state or a longer horizon:

- renal-electrolyte buffering should be retested at longer horizons and with
  explicit potassium/bicarbonate store dynamics;
- cardiovascular -> renal should move to 24h/48h because creatinine/urine output
  are slow targets;
- respiratory -> acid-base needs PaCO2, ventilator settings, or richer
  ventilation semantics;
- heme -> perfusion needs oxygen-delivery / bleeding-source / transfusion
  protocol context;
- immune -> hemodynamics needs capillary-leak and vasoplegia observability.

## Safety Boundary

No edge-specific coupling edge is promoted.  The current artifact is a
research-only candidate map for future whole-body digital-twin work.
