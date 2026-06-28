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

## Chapter B: Multi-Disease Expansion

The multi-disease chapter should reuse the DKA pattern, not mix all diseases into
one latent space immediately.

Implemented now:

- `osler_jepa/disease_router.py`
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

## Boundary

This A/B contract contains no patient rows or identifiers. It is a build
contract, not a clinical product claim.

The immediate broad engineering step for B has now been tested with a fourth
respiratory module.  DKA, sepsis, AKI, and bounded respiratory failure all
validate the factual-router pattern.
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
