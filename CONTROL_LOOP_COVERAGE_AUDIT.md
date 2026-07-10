# Control-Loop Coverage Audit

Date: 2026-07-10

This document maps the current Osler-JEPA body model against physiologic
control loops, not just disease names.  It answers: which feedback loops are
validated, which are only partially represented, and which remain data ceilings?

All entries are factual / observational only.  No entry grants causal,
counterfactual, treatment-planning, clinical, runtime, checkpoint-promotion, or
active-rule authority.

## Validated Belief Families

The current promoted predict-update belief families are:

1. renal reserve / creatinine kinetics;
2. cardiovascular perfusion / shock burden;
3. electrolyte / acid-base state;
4. respiratory / gas-exchange state;
5. endocrine / glycemic-adrenal stress;
6. immune / inflammatory host-response state;
7. GI / pancreatic / nutrition / gut-perfusion state;
8. musculoskeletal / rhabdomyolysis perfusion-stress state.

These are the loops where patient-specific hidden-state features have already
shown held-out improvement over both a strong baseline and a capacity-matched
placebo for at least one downstream observable.

## Loop-by-Loop Map

| Control loop | Current status | Evidence / boundary |
|---|---|---|
| Blood pressure / perfusion / cardiac output | validated | cardiovascular belief and all-model coupling validate HR/MAP/O2/RR signals; cardio -> renal long-horizon coupling validates kidney impact |
| Blood volume / sodium-water balance | partially validated | renal + cardiovascular + electrolyte beliefs; sodium is candidate-only in electrolyte belief but validates in some routers/all-model contexts |
| Oxygenation / CO2 / respiratory drive | validated | respiratory belief validates O2 saturation, respiratory rate, heart rate, bicarbonate; PaCO2/ventilator extraction exists but respiratory -> acid-base coupling remains candidate-only |
| Acid-base buffering | validated | electrolyte and respiratory beliefs validate bicarbonate / anion gap signals; DKA and AKI routers support horizon-specific acid-base forecasting |
| Potassium homeostasis | validated | electrolyte belief validates potassium at 6h; DKA/AKI routers also support potassium in specific contexts |
| Glucose regulation | validated | endocrine/DKA/presentation routers and treatment-context audits validate glucose forecasting in ICU contexts |
| Inflammation / host response | validated | immune belief validates sepsis MAP, creatinine, O2 saturation, HR, RR, and vasopressor requirement |
| Renal clearance / azotemia | validated | AKI renal belief validates creatinine/BUN at 24-48h; all-model coupling and multi-hop sepsis -> MAP -> renal path validate selected renal targets |
| GI / nutrition / gut perfusion | validated but bounded | GI/nutrition belief validates MAP; albumin/protein/bilirubin remain mostly observation/nowcast or fallback targets |
| Mineral-bone / divalent minerals | partially represented | electrolyte belief includes a `divalent_mineral` state over calcium/magnesium/phosphate; magnesium is patient-split positive but hospital-heldout fail; phosphate is candidate-only; calcium/ionized calcium are rejected |
| Thyroid axis | data ceiling / fallback | thyroid tests appear as sparse specialty targets in MIMIC-IV and are rejected/fallback; no validated TSH-T4-T3 belief |
| HPA/adrenal stress | partially validated | endocrine/glycemic-adrenal belief contributes validated MAP coupling, but ACTH/cortisol are not first-class validated hormone trajectories |
| ADH / thirst / osmolality | partially represented | sodium-water/osmotic components exist in electrolyte/renal beliefs, but serum osmolality itself is rejected in electrolyte belief |
| Growth hormone / IGF | data ceiling | no reliable ICU time-series support; not implemented as a belief |
| Reproductive / HPG | data ceiling | adult eICU lacks reliable pregnancy/fetal/reproductive-hormone trajectories; explicitly not implemented |
| EPO / red-cell production | partially observed, not validated as belief | Hgb/Hct first-class heme targets validate in routers/nowcast; heme belief and explicit production/control belief remain candidate-only |
| Coagulation / hemostasis | partial observation, belief not validated | INR/PTT/fibrinogen are first-class where available, but coagulation belief remains candidate-only and sparse |
| Iron / hepcidin | data ceiling | not supported as a reliable ICU time-series loop |
| Bilirubin / bile | observed/nowcast, not predictive belief | hepatic router and nowcast recover some bilirubin information; hepato-renal coupling fails even at 24-48h |
| Ammonia / urea cycle | data ceiling | ammonia is not reliable enough as a first-class repeated trajectory in the current contract |
| Albumin / oncotic pressure / protein synthesis | partially represented | GI/nutrition belief includes albumin/protein burden; albumin/protein mostly remain nowcast/observation rather than forecast wins |
| Thermoregulation | target-level only | temperature is used in immune/endocrine contexts, but no dedicated hypothalamic thermoregulation belief has validated |
| Consciousness / arousal | observation-only | GCS/delirium notes can be timestamped and observed; prediction remains unstable |
| Intracranial pressure regulation | data ceiling | no reliable first-class ICP/exam/procedure trajectory in current eICU contract |
| Autonomic balance | indirect only | cardiovascular and endocrine beliefs touch HR/BP/stress; no explicit sympathetic/parasympathetic belief has validated |
| Musculoskeletal / rhabdomyolysis | bounded validated | isolated muscle belief validates MAP at 3h and 12h; connected all-belief layer preserves 3h MAP only; muscle-to-kidney and muscle-to-electrolyte targets remain rejected |
| Cardiac injury / myocardial biomarkers | connected candidate-only | eight-belief connected audit validates 0 targets; 12h heart rate reaches 6/7 but does not promote; troponin/BNP/CK-MB remain sparse specialty markers |

## What Changed In This Pass

The new action taken in this pass was to test a distinct
musculoskeletal/rhabdomyolysis belief family and then rerun it across nearby
horizons after a 6h MAP near-miss.  The result is horizon-specific:

- rows: `16,837`;
- subjects: `1,205`;
- hospitals: `22`;
- features: `59`;
- 1h: no promoted target; MAP reaches `5 / 7`;
- 3h: MAP promotes at `7 / 7`, median delta `-0.4197` versus baseline and
  `-0.4244` versus placebo;
- 6h: no promoted target; MAP reaches `6 / 7`;
- 12h: MAP promotes at `7 / 7`, median delta `-0.3003` versus baseline and
  `-0.3060` versus placebo.

Because the validated signal is MAP-only, this expands the whole-body belief
layer narrowly: it adds a muscle/perfusion-stress loop, not a validated
muscle-to-kidney or muscle-to-electrolyte loop.

The connected 8-belief follow-up preserves the 3h MAP signal at `7 / 7`
(`-0.4051` versus baseline, `-0.4874` versus placebo).  The 12h MAP signal
drops to `6 / 7` when connected to all previous belief families, so it remains
candidate-only in the connected whole-body layer.

The same connected 8-belief approach was tested on cardiac injury / myocardial
stress cohorts.  It validates no target: 12h heart rate reaches `6 / 7`, but
troponin I, BNP, CK-MB, CPK, potassium, creatinine, MAP, and lactate fail the
full gate.  This keeps cardiac injury as a candidate-only control loop under
the current table-data contract.

## Practical Reading

The model is getting closer to a whole-body observer, but not a complete human
simulator.  It has robust acute ICU loops for perfusion, kidney, respiratory,
acid-base, potassium, glucose, inflammation, GI/nutrition stress, and bounded
muscle/perfusion stress.  The
remaining gaps are mostly one of three kinds:

1. **Sparse specialty hormones or deep labs**: thyroid, cortisol, ACTH, IGF,
   hepcidin, ammonia, INR/PTT/fibrinogen.
2. **Text/procedure-heavy states**: neuro exam, skin/wound staging, ICP,
   hepatic encephalopathy.
3. **Slow biology not visible at ICU forecast horizons**: erythropoiesis,
   bone-mineral regulation, protein synthesis, iron regulation.

The next high-yield depth work should therefore be selective, not another broad
surface sweep: build only loops with enough repeated, timestamped observability
to pass the same baseline/placebo gates.
