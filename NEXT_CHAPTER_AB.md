# Next Chapter A/B Contract

Date: 2026-06-26

The DKA factual arc is closed at a stable research artifact: a nested
per-target router. The current validated shape is:

- glucose -> PhysioNet presentation-only checkpoint
- anion gap -> real-fit grey-box residual
- all other targets -> persistence

This router beats persistence across seven patient split seeds and improves over
the no-realfit base router in all seven splits. It remains an observational
factual advisory artifact. It is not causal, clinical, or runtime-authoritative.

## Chapter A: Causal What-If Planning

The causal chapter is not opened by more observational EHR alone. Full MIMIC-IV
DKA causal diagnostics are runnable, but the target trials fail overlap,
co-treatment, and balance readiness. That is a structural confounding result, not
just a sample-size result.

Implemented now:

- `osler_jepa/causal_readiness.py`
- `next_chapter_ab.py`
- `next_chapter_ab_contract.json`
- `CHAPTER_A_CAUSAL_READINESS_FINDINGS.md`

The causal gate accepts only explicit external identification evidence:

- randomized trial data;
- instrumental-variable designs with external review;
- front-door designs with external review.

Current status is fail-closed:

- BioLINCC: candidate source, not in workspace
- Vivli: candidate source, not in workspace
- YODA: candidate source, not in workspace
- observational EHR: available for factual forecasting and negative confounding
  diagnostics only

When external evidence arrives, it must pass:

- required assignment, treatment, outcome, patient, and baseline columns;
- adequate subject count;
- balanced two-arm assignment for randomized data;
- acceptable outcome missingness;
- acceptable treatment adherence;
- acceptable baseline standardized mean differences.

Even a passing causal-readiness gate permits only a research causal-effect
estimate. It does not grant clinical recommendation authority, runtime action
authority, checkpoint promotion, or active-rule promotion.

## Current Product Focus: Observation And Prediction

The current engineering focus is the non-causal observation layer, not treatment
planning.  The validated layer is documented in
`WHOLE_BODY_OBSERVATION_PREDICTION_LAYER.md`,
`whole_body_observation_contract.json`, and `osler_jepa/observation_layer.py`.

Its allowed role is:

- shadow observation;
- factual physiology prediction at validated target/horizon pairs;
- capability reporting;
- persistence fallback when a target is unsupported, sparse, slow at the wrong
  horizon, or candidate-only.

Its forbidden role remains:

- causal treatment-effect estimation;
- counterfactual treatment planning;
- clinical recommendation authority;
- runtime treatment action authority;
- checkpoint or active symbolic-rule promotion.

## Chapter B: Multi-Disease Expansion

The multi-disease chapter should reuse the DKA pattern, not mix all diseases into
one latent space immediately.

Implemented now:

- `osler_jepa/disease_router.py`
- `eicu_body_system_configs.py`
- `eicu_body_system_transition_extract.py`
- `eicu_body_system_target_router.py`
- `next_chapter_ab.py`
- `next_chapter_ab_contract.json`

The reusable architecture is:

```text
disease-specific cohort
  -> disease-specific target/action contract
  -> observed-treatment transitions with masks and ages
  -> candidate sources
  -> discovery-only per-target source selection
  -> held-out patient/time/care-unit evaluation
  -> persistence fallback for unsupported targets
```

Starter modules:

- sepsis
- acute kidney injury
- respiratory failure / hypoxemia

Each disease starts as a factual research router:

- no causal claim unless `causal_readiness` passes;
- no clinical claim;
- no active rule promotion;
- no shared latent space until each disease has its own validated contract.

## Chapter B Implemented Disease Modules

Implemented on 2026-06-27:

Sepsis:

- `eicu_sepsis_transition_extract.py`
- `eicu_sepsis_target_router.py`
- `eicu_sepsis_transition_report.json`
- `eicu_sepsis_target_router.json`
- `EICU_SEPSIS_ROUTER_FINDINGS.md`

- 30,198 evaluable stays
- 25,175 subjects
- 204 hospitals
- 422,244 six-hour transitions
- 7/7 random patient splits significantly beat persistence
- hospital-held-out split also significantly beats persistence

- creatinine -> persistence
- heart rate, lactate, MAP, oxygen saturation, respiratory rate, urine output -> ridge realfit
- vasopressor-support proxy -> persistence after leakage guard

AKI:

- `eicu_aki_transition_extract.py`
- `eicu_aki_target_router.py`
- `eicu_aki_transition_report.json`
- `eicu_aki_target_router.json`
- `EICU_AKI_ROUTER_FINDINGS.md`
- `aki_mechanism.py`
- `eicu_aki_mechanism_target_router.json`
- `AKI_MECHANISM_CANDIDATE_FINDINGS.md`

- 43,748 evaluable stays
- 37,119 subjects
- 204 hospitals
- 667,799 six-hour transitions
- 7/7 random patient splits significantly beat persistence
- hospital-held-out split also significantly beats persistence

- creatinine, BUN -> persistence
- urine output -> mostly persistence
- bicarbonate, MAP, potassium, sodium -> ridge realfit
- renal mechanism candidate -> rejected; it did not robustly improve
  creatinine/BUN over persistence in the 6-hour factual window

It does not open Chapter A.  Antibiotic, fluid, ventilation, vasopressor, and
renal-replacement causal claims remain closed.  AKI diuretic, nephrotoxin,
fluid, vasopressor, and renal-replacement causal claims also remain closed.

Respiratory failure / hypoxemia:

- `eicu_respiratory_transition_extract.py`
- `eicu_respiratory_target_router.py`
- `eicu_respiratory_transition_report.json`
- `eicu_respiratory_target_router.json`
- `EICU_RESPIRATORY_ROUTER_FINDINGS.md`

- bounded deterministic cohort from full eICU: 4,170 evaluable stays
- 3,618 subjects
- 55 hospitals
- 57,673 six-hour transitions
- 26,558 active respiratory transitions
- 7/7 random patient splits significantly beat persistence
- hospital-heldout split also significantly beats persistence

- oxygen saturation, respiratory rate, heart rate, MAP, and pH -> ridge realfit
- bicarbonate -> ridge realfit in 5/7 splits, otherwise persistence

This respiratory module validates the same factual-router recipe in a fourth
domain, but it carries a bounded-cohort caveat.  The full respiratory cohort is
much larger and slower than sepsis/AKI because respiratory diagnoses and
hypoxemia are common in ICU data.  The extractor and router are ready for a
background full-scale run, but the committed result is the bounded engineering
cohort.

Body-system coverage layer:

- `eicu_body_system_configs.py`
- `eicu_body_system_transition_extract.py`
- `eicu_body_system_target_router.py`
- `BODY_SYSTEM_COVERAGE_FINDINGS.md`

New bounded modules:

- cardiovascular instability / shock / heart failure
- acute neurologic injury / seizure / coma physiologic proxy
- hepatic failure / cirrhosis proxy
- coagulopathy / thrombocytopenia / hematologic instability proxy

All four bounded modules pass the same factual router gate:

- 7/7 random patient splits significantly beat persistence on active windows
- hospital-heldout split also significantly beats persistence
- no causal, counterfactual, clinical, runtime, checkpoint, or active-rule claim

This expands body coverage to cardiovascular, nervous-system proxy,
hepatic/GI proxy, and hematologic/coagulation systems.  The proxy caveat still
matters for neuro and hepatic modules: neuro lacks detailed exam trajectories,
and hepatic lacks first-class ammonia/encephalopathy state.  Heme/coagulation
has moved beyond proxy status with first-class hemoglobin, hematocrit, INR,
PTT, fibrinogen, and transfusion evidence, but it remains factual rather than
causal.

Expanded full-body coverage pass:

- electrolyte / acid-base / osmotic instability
- endocrine stress / glycemic / adrenal-thyroid proxy
- GI / pancreatic / nutrition failure proxy
- cardiac injury / myocardial stress biomarkers
- musculoskeletal injury / rhabdomyolysis proxy
- immune / inflammatory activation proxy

All six bounded modules pass 7/7 random patient split seeds on active windows.
Five of the six also beat persistence on hospital-heldout normalized MAE.
Electrolyte/acid-base is the cautionary exception: it passes random splits but
regresses on hospital-heldout, likely from cross-hospital measurement or
practice shift.  This expands body-system factual physiology coverage, not
causal treatment planning or complete human simulation.

Breadth completion pass:

- integumentary / skin / wound / burn proxy
- toxicologic / metabolic poisoning proxy
- reproductive / obstetric marked as eICU data ceiling

Both feasible final modules pass 7/7 random patient split seeds and
hospital-heldout evaluation as bounded physiologic proxy routers.  Reproductive
and obstetric physiology is not forced into the router because this adult ICU
dataset lacks reliable pregnancy, fetal, obstetric intervention, and
reproductive hormone trajectories.

## Boundary

This A/B contract contains no patient rows or identifiers. It is a build
contract, not a clinical product claim.

The immediate broad engineering step for B has now been tested with a fourth
respiratory module.  DKA, sepsis, AKI, and bounded respiratory failure all
validate the factual-router pattern.
The generic body-system adapter now pushes the same pattern further across
cardiovascular, neurologic-proxy, hepatic-proxy, and hematologic/coagulation
systems.
The expanded body-system pass now pushes the same pattern into electrolyte,
endocrine, GI/pancreatic/nutrition, cardiac injury, musculoskeletal, and
immune/inflammatory systems.  The next coverage work should add missing
first-class variables and deeper hidden-state belief objects rather than only
adding more disease names.  Remaining high-value gaps include skin/wound state,
detailed neurologic exam trajectories, reproductive physiology, microbiology and
immune phenotype depth, procedure-specific cardiac/neuro variables, and
high-resolution treatment dosing.
The final breadth pass adds integumentary and toxicologic/metabolic proxy
coverage, then marks reproductive/obstetric physiology as an eICU data ceiling.
Chapter B breadth-first expansion is therefore closed for the current data
source.  The next Chapter-B work should be depth-first: structured wound state,
toxin-level/exposure trajectories, microbiology phenotypes, procedure-specific
variables, normalized treatment dosing, and more predict-update belief states.
The AKI renal mechanism audit shows that the first simple mechanism candidate is
not enough for slow creatinine/BUN targets at 6 hours.  The long-horizon AKI
audit answers the next question: at 24-48 hours, creatinine and BUN move from
persistence to `ridge_realfit` in 7/7 patient splits and also pass
hospital-heldout evaluation.  The next deeper renal step should therefore use
the 24-48h setting, a patient-specific renal reserve/GFR belief state, and richer
RRT/fluid-balance observability.  Asthma/respiratory exacerbation remains a
named template, but its peak-flow and work-of-breathing targets may need data
beyond ICU EHR.  The immediate next external-data step for A is unchanged:
obtain or map a randomized or otherwise externally identified treatment dataset
into the causal-readiness contract.

The first B-deep renal belief audit has now been run.  A transparent renal
reserve/GFR proxy improves long-horizon AKI downstream observable prediction
beyond both baseline `ridge_realfit` and a capacity-matched placebo for
creatinine and BUN at 24h/48h.  This does not open causal or hidden-state
accuracy claims.  It does establish the next research layer: convert the feature
belief into an explicit predict-update renal reserve belief state and keep the
same downstream-observable gate.

That explicit state version has also been tested.  It is partially validated:
BUN passes at 24h/48h, urine output passes at 24h and is directionally positive
at 48h, while creatinine does not pass.  The next B-deep refinement should focus
on improving the predict-update renal reserve equation for creatinine without
relaxing the placebo gate.

The creatinine-specific v2 state has now done exactly that.  Adding a separate
creatinine level/slope/innovation belief dimension recovers creatinine: 24h
passes 7/7 random splits and hospital-heldout, and 48h passes hospital-heldout
with 5/7 random splits.  The next B-deep step is not another feature audit; it is
to expose this v2 state as a shadow-mode digital-twin state object while keeping
all causal and clinical gates closed.

The first hematology B-deep belief audit has also been run.  It adds
bleeding/coagulation reserve features and an explicit predict-update coagulation
state, then tests both against baseline `ridge_realfit` and a capacity-matched
placebo.  The feature belief shows partial hemoglobin/hematocrit signal, but no
target reaches robust 7/7 validation and hospital-heldout does not pass.  The
explicit state is weaker.  Therefore heme/coagulation depth remains
candidate-only.  The next hematology step should be better transfusion-dose and
blood-product subtype observability plus longer horizons, not promotion of the
current hidden state.

The longer-horizon heme/coag audit has now been run on the same stay set as the
6h cohort.  It strengthens the factual router at 24h: active median delta
improves to -0.133450, with hemoglobin and hematocrit still selecting
`ridge_realfit` in 7/7 random splits.  At 48h the router still beats persistence
in 7/7 splits, but heme lab selection weakens.  INR, PTT, fibrinogen, and
platelets remain persistence at both 24h and 48h.  This closes the simple
"just extend the horizon" hypothesis for coagulation cascade targets: horizon
helps Hgb/Hct, but coagulation depth now needs blood-product subtype and dose
observability rather than another hidden-state promotion attempt.

That transfusion observability step has now been run.  The extractor separates
PRBC, plasma, platelet, cryoprecipitate, whole-blood, and unknown blood-product
evidence and adds `volume_like_ml` / `unit_like_count` features while preserving
the original binary `transfusion` channel.  The features are non-empty at 6h
(PRBC 1,509 future windows, plasma 254, platelets 306, cryoprecipitate 72), and
the heme router remains significant at 6h/24h/48h.  The strongest new belief
signal is 24h feature belief for hemoglobin (5/7 pass-both) and hematocrit
(4/7 pass-both), but it still fails the 7/7 promotion boundary and does not
unlock INR/PTT/fibrinogen/platelets.  Heme/coag therefore stays candidate-only.
The next heme step is not another hidden-state promotion attempt; it requires
stronger unit semantics, transfusion protocol context, bleeding-source/procedure
context, anticoagulation reversal evidence, or external identified data.

The first cross-system coupling audit has now also been run.  It tests seven
directed organ-system edges with a fail-closed gate: coupled ridge must beat both
a no-upstream baseline and a capacity-matched placebo.  No edge passes the
robust 7/7 active-window boundary; endocrine -> electrolyte potassium and sodium
show only 1/7 weak candidate signals.  This closes the static feature-concat
coupling hypothesis.  Whole-body depth should now move to temporal
predict-update coupling: renal reserve -> electrolyte/acid-base updates,
perfusion shock burden -> renal reserve and lactate updates, respiratory burden
-> acid-base updates, heme oxygen-carrying capacity -> perfusion/lactate updates,
and inflammatory burden -> hemodynamic/albumin/platelet updates.  These remain
candidate-only until they improve downstream observables beyond both baseline
and placebo.

That temporal predict-update coupling audit has now been run.  It moves in the
right direction but still does not pass promotion: renal -> bicarbonate reaches
2/7 active-window splits, renal -> sodium reaches 1/7, and immune -> MAP reaches
1/7.  No edge reaches 7/7, so the whole-body coupling layer remains
candidate-only.  The next Chapter-B depth step should be edge-specific shared
state rather than generic burden features: renal-electrolyte buffering,
immune-hemodynamic/capillary-leak state, respiratory CO2/ventilation state, and
heme oxygen-delivery state.

The edge-specific shared-state audit has now been run.  It produces broader weak
signals but still no promoted edge: renal -> bicarbonate 2/7, renal -> anion gap
1/7, respiratory -> bicarbonate 1/7, endocrine -> sodium/bicarbonate/anion gap
1/7 each, heme -> heart rate 1/7, and cardiovascular -> creatinine 1/7.  The
next depth step should stop trying all edges generically and focus on the two
most plausible frontiers: renal-electrolyte buffering with explicit
potassium/bicarbonate store dynamics, and cardio-renal coupling at 24h/48h where
creatinine and urine output have time to move.

That focused coupling audit has now been run.  It gives the first robust
cross-system factual coupling edge.  The renal-electrolyte explicit
potassium/bicarbonate store candidate does not promote: bicarbonate and anion
gap reach only 1/7 active-window patient splits, and potassium/sodium/phosphate
stay at 0/7.  Cardio-renal coupling does promote as a factual research edge at
the right time scale: creatinine and BUN pass 7/7 active-window splits at both
24h and 48h, with hospital-heldout support; urine output is partial
(5/7 at 24h and 4/7 at 48h).

The next B-deep step is therefore to expose the cardio-renal 24h/48h coupling as
a shadow factual state object, not as a treatment policy.  Renal-electrolyte
coupling should stay candidate-only until treatment timing, KCl/bicarbonate
dosing, urine electrolyte evidence, and acid-base/ventilation context are richer.

The next focused coupling pass has now tested two more clinically grounded
edges.  Sepsis/immune -> cardiovascular passes for MAP at 6h: MAP reaches 7/7
active-window patient splits with hospital-heldout support, while heart rate
and lactate remain partial.  Hepatic -> renal improves over the broad audit but
does not promote: creatinine and BUN reach only 3/7 active-window patient
splits at 6h, despite hospital-heldout creatinine signal.

The whole-body factual coupling map now has two validated research edges:
cardio-renal long-horizon coupling and sepsis/immune -> MAP.  The next depth
step should expose both as shadow factual state objects while keeping all
treatment-effect, causal, and clinical gates closed.

Hepato-renal has now been pushed to 24h/48h on the same hepatic stay set.  The
longer horizon does not rescue it: creatinine reaches only 2/7 active splits at
24h and 3/7 at 48h, while BUN remains 0/7 at both horizons.  This rejects the
simple "like cardio-renal, just make the horizon longer" hypothesis.  The next
hepato-renal step is richer hepatic observability and context, not another
horizon-only audit.

The first multi-hop coupling audit has also been run.  It tests
sepsis/immune -> MAP -> renal on the full eICU sepsis cohort, using a
discovery-only MAP mediator and comparing against a direct sepsis-to-renal
baseline plus a capacity-matched placebo.  The path is a strong candidate but
not fully promoted: BUN passes 7/7 active-window patient splits, creatinine
passes 6/7, urine output fails, and hospital-heldout support is mixed.  This is
the first evidence that body-system edges can start to form a path, but it
should stay candidate-only until a compatible long-horizon sepsis/MAP/renal
cohort or stronger hospital-heldout support exists.

That compatible long-horizon cohort has now been generated and tested.  The
time-scale-corrected path uses sepsis -> MAP at 6h as the mediator and renal
targets at 24h/48h as the downstream hop.  The 24h path validates for
creatinine and BUN: both pass 7/7 active-window patient splits and pass
hospital-heldout active-window comparisons against both the direct
sepsis-to-renal baseline and a capacity-matched placebo.  Urine output fails,
and the 48h path remains candidate-only.  This is the first validated multi-hop
whole-body factual path; it can be exposed only as a shadow factual state, not a
treatment or causal policy.

Respiratory -> acid-base now has a clear observability path.  A full-eICU label
scan finds abundant PaCO2, FiO2, PEEP, tidal-volume, ventilator-mode, and
minute-ventilation evidence in lab and respiratory charting tables.  The next
respiratory coupling pass has now promoted PaCO2, FiO2, PEEP, tidal-volume, and
ventilator-mode columns to first-class observed respiratory state variables in a
bounded deterministic 5,000-stay cohort.  That retest still does not validate
respiratory -> acid-base: bicarbonate reaches only 1/7 active-window splits,
while pH, PaCO2, and lactate are 0/7.  The current next step is therefore not
another generic feature block.  Either build a specific ventilator/ABG
trajectory contract with usable minute-ventilation and repeated ABG windows, or
keep this edge closed under the present eICU factual router contract.
