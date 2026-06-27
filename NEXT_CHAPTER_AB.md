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

The causal gate accepts only explicit external identification evidence:

- randomized trial data;
- instrumental-variable designs with external review;
- front-door designs with external review.

Current status is fail-closed:

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
- asthma exacerbation

Each disease starts as a factual research router:

- no causal claim unless `causal_readiness` passes;
- no clinical claim;
- no active rule promotion;
- no shared latent space until each disease has its own validated contract.

## Chapter B Sepsis Status

Implemented on 2026-06-27:

- `eicu_sepsis_transition_extract.py`
- `eicu_sepsis_target_router.py`
- `eicu_sepsis_transition_report.json`
- `eicu_sepsis_target_router.json`
- `EICU_SEPSIS_ROUTER_FINDINGS.md`

The full eICU sepsis router is now the first Chapter-B disease expansion:

- 30,198 evaluable stays
- 25,175 subjects
- 204 hospitals
- 422,244 six-hour transitions
- 7/7 random patient splits significantly beat persistence
- hospital-held-out split also significantly beats persistence

The selected router remains factual and observational:

- creatinine -> persistence
- heart rate, lactate, MAP, oxygen saturation, respiratory rate, urine output -> ridge realfit
- vasopressor-support proxy -> persistence after leakage guard

It does not open Chapter A.  Antibiotic, fluid, ventilation, vasopressor, and
renal-replacement causal claims remain closed.

## Boundary

This A/B contract contains no patient rows or identifiers. It is a build
contract, not a clinical product claim.

The immediate next engineering step for B is to choose the next disease module
after sepsis, likely AKI because its target/action contract is closer to the
existing renal, fluid, electrolyte, and urine-output machinery.  The immediate
next external-data step for A is unchanged: obtain or map a randomized or
otherwise externally identified treatment dataset into the causal-readiness
contract.
