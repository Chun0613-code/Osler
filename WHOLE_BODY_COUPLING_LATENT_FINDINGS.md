# Whole-Body Coupling Latent Findings

Date: 2026-07-05

This audit asks whether Osler-JEPA can be strengthened beyond simply
concatenating the current per-system belief models.

The baseline is intentionally strong:

- ordinary table/action ridge features;
- plus all five current personalized belief families:
  renal, cardiovascular, electrolyte/acid-base, respiratory, endocrine.

The candidate adds only explicit whole-body coupling latent features:

- hemodynamic axis;
- respiratory axis;
- metabolic/endocrine axis;
- electrolyte/acid-base axis;
- renal axis;
- global instability;
- cross-axis interactions such as hemodynamic x renal reserve,
  respiratory x acid-base, endocrine x electrolyte, and global x organ axes.

The placebo adds the same number of random coupling features.  Passing means
the explicit whole-body coupling layer adds incremental factual signal beyond
the all-belief baseline and beyond added capacity.

This remains factual and observational.  It grants no causal, counterfactual,
clinical, runtime, treatment-planning, checkpoint-promotion, or active-rule
authority.

## Result

The explicit coupling latent validates 10 incremental target/cohort cells:

| Cohort | Target | Median delta vs all-belief baseline | Median delta vs placebo | Interpretation |
|---|---|---:|---:|---|
| cardiovascular_full_6h | lactate | -0.041106 | -0.049929 | hemodynamic-metabolic coupling |
| sepsis_6h_full | lactate | -0.039607 | -0.042979 | inflammatory/hemodynamic-metabolic coupling |
| sepsis_6h_full | O2 saturation | -0.010591 | -0.010871 | sepsis cardiopulmonary coupling |
| AKI 24h | BUN | -0.021879 | -0.026502 | systemic renal-solute coupling |
| MIMIC-IV 6h | anion gap | -0.007704 | -0.008581 | metabolic-acid-base coupling |
| MIMIC-IV 6h | BUN | -0.007167 | -0.010018 | renal/systemic coupling |
| MIMIC-IV 6h | creatinine | -0.000861 | -0.001093 | small renal/systemic coupling |
| MIMIC-IV 6h | lactate | -0.017113 | -0.018080 | replicated hemodynamic-metabolic coupling |
| MIMIC-IV 6h | PaCO2 | -0.038039 | -0.040195 | respiratory-acid-base coupling |
| MIMIC-IV 6h | phosphate | -0.001761 | -0.002305 | renal/electrolyte coupling |

The strongest repeated new signal is lactate:

```text
cardiovascular_full_6h lactate
sepsis_6h_full lactate
MIMIC-IV 6h lactate
```

That matters because lactate is not a single-organ variable.  It sits at the
intersection of perfusion, oxygen delivery, metabolism, renal/hepatic clearance,
and systemic stress.  It is exactly the kind of target that should benefit from
a whole-body coupling representation if that representation is real.

## What Did Not Improve

Many targets that were already handled by local/all-belief models did not gain
extra signal from explicit coupling:

- heart rate;
- MAP in most full-cohort settings;
- glucose;
- many core electrolytes such as sodium/potassium/magnesium;
- most sparse specialty variables.

That is a useful boundary.  The coupling layer is not a universal score booster.
It adds signal mostly where the target is physiologically cross-system by
nature.

The following remain closed:

- troponin;
- thyroid tests;
- bilirubin;
- INR/PTT/fibrinogen;
- CRP/ESR/ferritin;
- serum ketones/osmolality;
- most sparse specialty markers.

## Interpretation

Before this audit, the whole-body layer had:

1. isolated organ-system routers;
2. validated single-hop edges such as cardio -> renal and sepsis -> MAP;
3. one validated multi-hop path: sepsis -> MAP6 -> renal24;
4. all-model belief concatenation, which showed shared latent states help.

This audit adds a stricter fourth layer:

```text
per-system belief states
  -> explicit whole-body latent axes
  -> incremental prediction of cross-system targets
```

The result is a real strengthening of coupling, but not a claim of complete
human simulation.  It says explicit body-level interaction features improve
held-out factual prediction for cross-system physiology.  It does not say the
model understands causal treatment effects.

## Practical Status

Validated for shadow factual observation:

- lactate as a whole-body coupling target;
- BUN and small creatinine increments as renal/systemic coupling targets;
- anion gap and PaCO2 as metabolic/respiratory-acid-base coupling targets;
- phosphate as renal/electrolyte coupling target.

Candidate-only:

- MAP extra coupling beyond all-belief baseline;
- O2 saturation outside the sepsis cohort;
- pH, which reached 6/7 in MIMIC but did not pass the full gate;
- platelets, which reached 4/7 but did not pass.

Rejected:

- specialty biomarkers and sparse deep variables.

## Boundary

Allowed:

- use explicit whole-body latent coupling as a research/shadow feature source
  for validated cross-system factual targets;
- report lactate/BUN/anion-gap/PaCO2/phosphate coupling as incremental over
  all-belief baseline;
- keep nonvalidated targets in fallback or candidate-only mode.

Not allowed:

- causal claims;
- counterfactual treatment planning;
- clinical recommendations;
- runtime treatment authority;
- claiming the latent axes are directly measured physiologic truth.
