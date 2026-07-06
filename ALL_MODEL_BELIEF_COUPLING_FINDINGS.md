# All-Model Belief Coupling Findings

Date: 2026-07-05

This audit asks what happens when the current personalized physiology models
are connected together rather than used one system at a time.

The connected candidate uses all five current predict-update belief feature
families:

- renal reserve / creatinine kinetics;
- cardiovascular perfusion / shock burden;
- electrolyte and acid-base state;
- respiratory / gas-exchange state;
- endocrine / glycemic-adrenal stress.

The gate is unchanged:

- baseline: ordinary table/action ridge forecast;
- candidate: baseline plus all current belief model outputs;
- placebo: baseline plus the same number of random features;
- validation: 7 patient-heldout splits plus hospital-heldout support;
- authority: factual observation/prediction only.

This does **not** grant causal, counterfactual, treatment-planning, clinical,
runtime, checkpoint-promotion, or active-rule authority.

## First-Pass Scope

The first pass intentionally used medium-sized cohorts where all five online
belief models can be computed without pre-caching:

| Cohort | Rows | Horizon |
|---|---:|---:|
| respiratory failure / hypoxemia | 57,673 | 6h |
| electrolyte / acid-base | 17,387 | 6h |
| endocrine / glycemic-adrenal stress | 17,009 | 6h |
| cardiovascular instability | 16,047 | 6h |

Full sepsis and AKI cohorts were **not** included in this first pass because
computing five online belief states over 400k-500k row tables is too slow in
the current non-cached implementation. That is an engineering limit of the
audit runner, not a biological conclusion.

## Result

All-model coupling validated six target/cohort cells:

| Cohort | Target | Patient splits | Median delta vs baseline | Median delta vs placebo | Interpretation |
|---|---|---:|---:|---:|---|
| respiratory_6h | O2 saturation | 7 / 7 | -0.216237 | -0.228680 | reconfirms respiratory/gas-exchange personalization |
| respiratory_6h | respiratory rate | 7 / 7 | -0.251531 | -0.272223 | reconfirms respiratory/gas-exchange personalization |
| respiratory_6h | MAP | 7 / 7 | -0.399305 | -0.456304 | **new cross-system MAP coupling** |
| electrolyte_6h | MAP | 7 / 7 | -0.405835 | -0.486711 | **new cross-system MAP coupling** |
| endocrine_6h | glucose | 7 / 7 | -1.394339 | -1.884862 | reconfirms endocrine/glycemic personalization |
| endocrine_6h | MAP | 7 / 7 | -0.339332 | -0.454693 | reconfirms endocrine/hemodynamic personalization |

The cardiovascular bounded cohort showed heart-rate signal but did not promote:

| Cohort | Target | Patient splits | Median delta vs baseline | Median delta vs placebo | Status |
|---|---|---:|---:|---:|---|
| cardiovascular_6h | heart rate | 5 / 7 | -0.514239 | -0.628450 | candidate-only in this bounded all-model pass |

## What Is New

The new signal is not "all models magically improve everything." The clean new
coupling is:

```text
endocrine / adrenal-hemodynamic stress belief
  -> MAP prediction
  in respiratory and electrolyte/acid-base cohorts
```

This is a whole-body coupling because a stress/endocrine state improves blood
pressure prediction outside the endocrine module itself.

## MAP Ablation

A focused ablation decomposed the two new MAP hits by belief family.

### Respiratory Cohort -> MAP

| Belief family | Patient splits | Median delta vs baseline | Median delta vs placebo | Hospital-heldout |
|---|---:|---:|---:|---|
| endocrine | 7 / 7 | -0.351169 | -0.357220 | pass |
| cardiovascular | 3 / 7 | -0.026249 | -0.029145 | fail |
| renal | 2 / 7 | -0.053782 | -0.056196 | fail |
| electrolyte | 0 / 7 | -0.009973 | -0.028389 | fail |
| respiratory | 0 / 7 | +0.014676 | +0.006267 | fail |

### Electrolyte Cohort -> MAP

| Belief family | Patient splits | Median delta vs baseline | Median delta vs placebo | Hospital-heldout |
|---|---:|---:|---:|---|
| endocrine | 7 / 7 | -0.363824 | -0.406441 | pass |
| renal | 3 / 7 | -0.072128 | -0.077342 | patient-split fail, hospital pass |
| cardiovascular | 2 / 7 | -0.044021 | -0.045808 | fail |
| respiratory | 0 / 7 | -0.028456 | -0.035286 | patient-split fail, hospital pass |
| electrolyte | 0 / 7 | +0.029388 | -0.004187 | fail |

The ablation makes the mechanism clear: the new MAP coupling is carried by the
endocrine/adrenal-hemodynamic belief, not by generic extra capacity and not by
the local respiratory/electrolyte belief states.

## Interpretation

This audit adds a second kind of cross-system connection:

1. previously validated path-style coupling:
   `sepsis -> MAP6 -> renal24`;
2. newly validated shared-state coupling:
   `endocrine/adrenal-hemodynamic stress -> MAP` across respiratory and
   electrolyte contexts.

That means the whole-body layer is beginning to show two forms of coupling:

- directed temporal paths between systems;
- shared latent stress states that generalize across disease modules.

The result is still factual and observational. It says this hidden endocrine /
hemodynamic stress state improves held-out MAP prediction. It does **not** say
changing hormones, glucose, steroids, fluids, vasopressors, oxygen, or any
treatment would causally change MAP.

## Boundary

Allowed:

- report the endocrine/adrenal-hemodynamic belief as a validated factual MAP
  coupling source in respiratory and electrolyte contexts;
- use the result for shadow observation-layer research;
- use it as a guide for future cached all-model coupling audits.

Not allowed:

- causal or treatment-effect claims;
- clinical or runtime action authority;
- checkpoint promotion;
- active rule promotion;
- claiming direct hidden-state truth.

## Next Engineering Step

Cache belief features per cohort before running the full all-model scan. The
current online builders are correct but too slow for 400k-500k row sepsis/AKI
tables when recomputed inside a single probe.

Once cached, rerun the same gate on:

- full sepsis 6h;
- AKI 24h/48h;
- MIMIC-IV cross-database cohorts where matching belief features exist.
