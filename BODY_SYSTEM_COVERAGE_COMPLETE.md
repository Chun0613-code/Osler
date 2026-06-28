# Body-System Coverage Complete

Date: 2026-06-28

This document closes the Chapter-B breadth-first pass.  Osler-JEPA now has
isolated factual-router contracts for the adult-ICU body-system buckets that
eICU can support with observed physiology.

This is not a complete human simulator.  It is a bounded factual physiology
coverage layer: each module has its own cohort definition, targets, observed
action evidence, discovery-only source selection, held-out router gate, and
persistence fallback.  No module grants causal, counterfactual, clinical,
runtime, checkpoint-promotion, or active-rule authority.

## Coverage Map

| Standard System | Osler-JEPA Coverage | Status |
|---|---|---|
| Cardiovascular | cardiovascular instability + cardiac injury | covered |
| Respiratory | respiratory failure / hypoxemia | covered |
| Nervous system | acute neuro physiologic proxy | covered with proxy caveat |
| Renal / urinary | AKI + renal belief state | covered; deepest non-DKA belief state |
| Fluid / electrolyte / acid-base | electrolyte / acid-base / osmotic instability | covered; cross-hospital caveat |
| Endocrine / metabolic | DKA + endocrine stress + toxic-metabolic overlap | covered |
| Digestive / hepatic / pancreatic / nutrition | hepatic failure + GI/pancreatic/nutrition | covered with proxy caveats |
| Hematologic / coagulation | coagulopathy / heme first-class variables | covered; heme belief remains candidate-only |
| Immune / lymphatic | sepsis + immune/inflammatory proxy | covered |
| Musculoskeletal | rhabdomyolysis / muscle-injury proxy | covered |
| Integumentary | skin / wound / burn proxy | covered with text-state caveat |
| Toxicologic | poisoning / overdose / toxic-metabolic proxy | covered with toxin-level caveat |
| Reproductive / obstetric | not implemented | eICU data ceiling |

## Final Breadth Modules

The last two feasible adult-ICU breadth modules were run as bounded 1,500-stay
full-eICU engineering cohorts.

| Module | Stays | Transitions | Active Rows | Active Median Delta | Random Splits | Hospital-Heldout Delta | Stable 7/7 Ridge Targets |
|---|---:|---:|---:|---:|---:|---:|---|
| Integumentary / skin / wound | 1,377 | 17,327 | 10,023 | -0.058359 | 7/7 significant | -0.030696 | glucose, heart rate, hematocrit, hemoglobin, MAP |
| Toxicologic / metabolic | 1,333 | 11,318 | 5,390 | -0.094078 | 7/7 significant | -0.014730 | glucose, heart rate, MAP, oxygen saturation, potassium, respiratory rate |

## What This Proves

The same rule now holds across the body-system surface:

- dense, fast, physiologic targets can beat persistence;
- sparse, slow, or unobserved targets should fall back to persistence;
- the router's core value is selective movement, not universal movement.

The breadth-first pass is therefore complete enough for the current data source.
Future work should move from adding more disease names to increasing depth:
structured skin/wound state, toxin-level trajectories, microbiology phenotypes,
procedure-specific cardiac/neuro variables, treatment dose normalization, and
additional validated predict-update belief states.

## Data Ceilings

The following are explicit data ceilings in the current eICU contract:

- reproductive/obstetric physiology: adult ICU eICU does not provide reliable
  pregnancy, fetal, obstetric intervention, or reproductive hormone trajectories;
- integumentary depth: skin stage, wound size, drainage, and burn surface area
  are mostly free text or absent from the numeric state map;
- toxicologic depth: toxin concentration, ingestion timing, exposure dose, and
  poison-control protocol context are not reliable first-class trajectories;
- neurologic depth: detailed exam trajectories and imaging/procedure context are
  incomplete;
- causal planning: observational EHR remains fail-closed without external
  randomized, instrumental-variable, or front-door identification evidence.

