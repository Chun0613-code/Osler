# Whole-Body Observation And Prediction Layer

Date: 2026-06-30

This artifact defines the non-causal Osler-JEPA observation layer.  Its purpose
is deliberately narrower than treatment planning: observe the patient state,
predict validated near-future physiology, expose uncertainty through fallback,
and refuse unsupported targets.

It does not claim causal treatment effects, counterfactual treatment response,
clinical authority, runtime treatment authority, checkpoint promotion, or active
symbolic-rule promotion.

## Validated Capabilities

The layer now has four kinds of validated factual prediction capability:

1. Disease routers:
   - DKA: glucose, anion gap, potassium, bicarbonate, and MAP at 6h.
   - Sepsis: MAP, heart rate, lactate, creatinine, BUN, and oxygen saturation at
     6h.
   - AKI: creatinine, BUN, and urine output at 24-48h.
   - Respiratory failure / hypoxemia: oxygen saturation, respiratory rate, MAP,
     heart rate, pH, and bicarbonate at 6h.

2. Body-system surface routers:
   - Twelve adult-ICU body-system modules cover cardiovascular, respiratory,
     neurologic proxy, renal/urinary, electrolyte/acid-base, endocrine,
     hepatic/GI/pancreatic/nutrition, hematologic/coagulation, immune,
     musculoskeletal, integumentary proxy, and toxic/metabolic physiology.
   - These routers repeatedly validate the same rule: dense fast variables can
     move away from persistence; sparse, slow, or poorly observed variables must
     fall back.

3. Cross-system coupling:
   - Cardiovascular/perfusion -> renal at 24-48h for creatinine and BUN.
   - Sepsis/immune-inflammatory -> cardiovascular MAP at 6h.

4. Multi-hop and hidden-state observation:
   - Sepsis/immune -> MAP at 6h -> renal at 24h validates for creatinine and
     BUN.
   - AKI renal belief state v2 validates an online-compatible hidden renal
     reserve/GFR state for downstream creatinine and BUN prediction.

## Prediction Policy

The policy is simple:

- Use a validated capability only at its validated target and horizon.
- Use 6h for fast physiology and acute perfusion targets.
- Use 24-48h for slow renal accumulation targets.
- Use the validated multi-hop path only for sepsis -> MAP6 -> renal24
  creatinine/BUN prediction.
- Fall back to persistence for unsupported targets, sparse variables, wrong
  horizons, candidate-only edges, and any future UI/API request that falls
  outside the aggregate evidence.

This is why the layer can be broad without pretending to be omniscient.  It is
allowed to say, "this variable should not be moved by the model yet."

## Candidate Or Closed Frontiers

The following remain candidate-only under the current aggregate gates:

- Respiratory -> acid-base coupling, even after first-class PaCO2, FiO2, PEEP,
  tidal volume, and ventilator-mode features.
- Hepato -> renal coupling at 6h, 24h, and 48h.
- Renal -> electrolyte store coupling without better KCl/bicarbonate treatment
  capture.
- Sepsis -> MAP6 -> renal48 and urine-output multi-hop prediction.
- Heme/coagulation hidden-state belief after blood-product subtype and dose
  normalization.

These failures are part of the observation contract.  They tell Osler where not
to invent motion.

## Human Analogy

At this stage, Osler-JEPA is not a treatment planner.  It is closer to a
whole-body monitor with short-to-medium horizon physiology intuition.

In human terms:

- The symbolic engine is the doctor who understands rules, danger signs, and
  mechanism language.
- The factual routers are the bedside observer who has learned which numbers
  usually move soon and which numbers usually do not.
- The renal belief state is the beginning of an internal hidden-organ estimate:
  not just "what is creatinine now," but "what renal reserve seems to be behind
  the observed curve."
- Persistence fallback is the humility reflex.  If Osler has not earned the
  right to move a variable, it keeps the variable still.

So the current goal is not "simulate every treatment."  The current goal is:
given the observed body, predict the next believable body state as broadly as
the data supports, and refuse everything else.

## Implementation

The machine-readable readiness map lives in:

- `osler_jepa/observation_layer.py`
- `whole_body_observation_contract.json`

The aggregate evidence comes from the already committed disease, body-system,
coupling, multihop, and renal-belief findings.  No row-level cohorts, patient
identifiers, timestamps, or treatment-policy claims are included in this
artifact.
