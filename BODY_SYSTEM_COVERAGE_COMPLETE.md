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

## Cross-System Coupling Baseline

The first cross-system depth audit has now been run in
`eicu_body_system_coupling_audit.json` and summarized in
`BODY_SYSTEM_COUPLING_FINDINGS.md`.

It tests whether upstream organ-system features improve downstream factual
prediction beyond both a no-upstream baseline and a capacity-matched placebo.
No directed edge passes the robust 7/7 active-window promotion gate.  The only
weak active-window signals are endocrine -> electrolyte potassium and sodium,
each at 1/7 patient split seeds, so they remain candidate-only.

This is a useful boundary: isolated full-body surface coverage is not yet a
whole-body digital twin.  The next depth layer should be temporal and
belief-based rather than static feature concatenation, for example renal reserve
belief -> electrolyte/acid-base updates, perfusion shock burden -> renal reserve
updates, and respiratory burden -> acid-base updates.

That temporal belief-based coupling layer has now been tested as
`eicu_body_system_temporal_coupling_audit.json`.  It improves the shape of the
signal but still does not promote any edge: renal -> bicarbonate reaches 2/7
active-window patient splits, renal -> sodium reaches 1/7, and immune -> MAP
reaches 1/7.  These weak signals stay candidate-only.  The next depth work
should therefore be edge-specific shared state, not a generic all-edge coupling
feature block.

The edge-specific shared-state pass has also been run as
`eicu_body_system_edge_specific_coupling_audit.json`.  It adds bespoke
renal-electrolyte buffering, respiratory CO2/ventilation mismatch,
endocrine-electrolyte osmotic interaction, heme oxygen-delivery debt,
cardio-renal perfusion stress, hepatic/coagulation, and immune capillary-leak
states.  This improves weak signal coverage but still does not promote any edge.
The whole-body layer therefore remains a candidate map, not an online digital
twin coupling graph.

The focused follow-up audit has now been run as
`eicu_body_system_focused_coupling_audit.json`.  It rejects the same-window
renal-electrolyte explicit store candidate, but validates the first
cross-system factual coupling edge: cardiovascular perfusion / shock burden to
renal creatinine and BUN at 24h/48h.  Urine output remains partial.  This means
the whole-body layer now has one validated factual coupling line, while all
causal and runtime treatment gates remain closed.

The next focused pass adds two more clinically grounded edges.  Sepsis/immune
burden to cardiovascular tone validates MAP at 6h, becoming the second
cross-system factual coupling line.  Hepatic burden to renal stress improves to
a 3/7 candidate signal for creatinine and BUN at 6h but does not promote.  The
current validated factual coupling map is therefore cardio -> renal long horizon
and sepsis/immune -> MAP short horizon; hepato-renal remains a candidate for a
longer-horizon or richer-observability audit.

The longer-horizon hepato-renal audit has also been run on the same hepatic stay
set.  It does not validate the edge: creatinine reaches 2/7 at 24h and 3/7 at
48h, while BUN remains 0/7 active-window splits at both horizons.  Hepato-renal
therefore needs richer hepatic state and treatment/procedure context rather than
another horizon-only pass.

The first multi-hop audit has been run as
`eicu_body_system_multihop_coupling_audit.json`.  It tests
sepsis/immune -> MAP -> renal against a direct sepsis-to-renal baseline.  BUN
passes 7/7 active-window patient splits, creatinine reaches 6/7, and urine
output fails; hospital-heldout support is mixed.  This is the first strong
multi-hop whole-body candidate, but not a promoted path.

A respiratory acid-base observability scan shows that eICU contains abundant
PaCO2, FiO2, PEEP, tidal-volume, ventilator-mode, and minute-ventilation labels.
Respiratory -> acid-base is therefore blocked by feature extraction, not by
absence of source measurements.

That feature-extraction pass has now been run as a bounded deterministic
5,000-stay observed respiratory cohort.  PaCO2, FiO2, PEEP, tidal-volume, and
ventilator-mode features are promoted to first-class state variables, but the
focused respiratory -> acid-base gate still does not validate: bicarbonate
reaches only 1/7 active-window patient splits, while pH, PaCO2, and lactate are
0/7.  Respiratory -> acid-base therefore remains candidate-only under the
current eICU factual contract; the blocker is no longer simple absence of
PaCO2/PEEP/FiO2 extraction.

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
