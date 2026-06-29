# Body-System Temporal Coupling Findings

Date: 2026-06-29

This is the second cross-system coupling pass after full body-system breadth
completion.  The first pass tested static upstream feature concatenation and
found no robust promoted edges.  This pass tests a more body-like candidate:
each upstream system is converted into temporal predict-update belief features
before being added to downstream factual prediction.

The gate is unchanged:

- baseline: downstream ridge model without temporal coupling belief features;
- candidate: baseline plus temporal predict-update coupling belief features;
- placebo: baseline plus the same number of random features;
- pass rule: candidate must significantly beat both baseline and placebo on
  held-out patients.

The report is aggregate-only and grants no causal, counterfactual, clinical,
runtime, checkpoint-promotion, or active-rule authority.

## Result

Temporal coupling adds weak, physiologically plausible signals, but no edge
passes the robust 7/7 active-window promotion boundary.  Therefore the temporal
whole-body coupling layer remains candidate-only.

| Edge | Temporal Belief | Active Pass-Both Targets | Hospital-Heldout Notes | Interpretation |
|---|---|---|---|---|
| renal -> electrolyte / acid-base | renal reserve + creatinine kinetics | bicarbonate 2/7, sodium 1/7 | sodium passes hospital-only | weak renal/electrolyte signal, not promoted |
| respiratory -> acid-base | ventilation / oxygenation burden | none | none | rejected |
| endocrine -> electrolyte | glycemic / osmotic burden | none | sodium passes hospital-only | rejected; hospital-only signal is not enough |
| heme/coag -> perfusion | coagulation / oxygen-carrying reserve | none | none | rejected |
| cardiovascular -> renal | perfusion shock burden | none | none | rejected at 6h |
| hepatic -> coagulation / platelets | bilirubin / hepatic burden | none | none | rejected |
| immune / inflammatory -> hemodynamics | inflammatory burden | MAP 1/7 | none | weak inflammatory/hemodynamic signal, not promoted |

## What Improved Versus Static Coupling

The temporal layer produced more plausible weak signals than static
feature-concat coupling:

- renal reserve/creatinine kinetics helped bicarbonate in 2/7 patient splits;
- renal temporal belief helped sodium in 1/7 patient splits and hospital-only
  evaluation;
- inflammatory temporal burden helped MAP in 1/7 patient splits.

These are not enough to leave candidate status, but they are useful because they
identify where deeper body coupling should focus first: renal-electrolyte and
immune-hemodynamic interactions.

## What This Rules Out

This rules out a generic one-size-fits-all temporal coupling layer for now.  The
system cannot simply generate a burden state for every upstream organ and expect
whole-body physiology to emerge.

The next layer needs edge-specific physiology:

- renal reserve should couple explicitly to potassium, bicarbonate, sodium, and
  acid-base buffering, likely with a longer horizon for slower renal effects;
- immune/inflammatory burden should couple to vascular tone, capillary leak,
  albumin, platelets, and lactate through sepsis-like hemodynamic state;
- respiratory burden should couple to pH through ventilation/CO2 support, which
  is not fully observed in the current eICU numeric state map;
- heme/perfusion coupling likely needs bleeding source, procedure context,
  transfusion protocol semantics, and oxygen delivery rather than generic
  coagulation reserve.

## Safety Boundary

No temporal coupling edge is promoted.  The current whole-body status is:

- body-system surface coverage: validated as isolated factual routers;
- AKI renal belief state: validated for specific downstream renal observables;
- static cross-system coupling: rejected;
- temporal cross-system coupling: weak candidate signals only.

This is a research baseline for future whole-body digital-twin work, not a
complete human simulator.
