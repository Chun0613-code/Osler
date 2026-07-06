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

## Cached Full Rerun

The engineering limit has now been removed by caching all five belief feature
families once per cohort and reusing the cached features across every split.
The row-level caches remain local-only and are not committed.

Cached full cohorts:

| Cohort | Rows | Holdout support | Belief features |
|---|---:|---|---:|
| full cardiovascular instability | 682,172 | 206 hospitals | 156 |
| full sepsis | 422,244 | hospital-heldout | 156 |
| AKI 24h | 530,265 | hospital-heldout | 156 |
| AKI 48h | 397,897 | hospital-heldout | 156 |
| MIMIC-IV observation 6h | 640,164 | careunit-heldout | 156 |

The rerun validates the following cells across 7 patient splits and the
available hospital/careunit holdout, always against both baseline and
capacity-matched placebo.

| Cohort | Validated targets |
|---|---|
| cardiovascular_full_6h | heart_rate, MAP, O2 saturation, respiratory_rate |
| sepsis_6h_full | MAP, creatinine, urine_output, O2 saturation, heart_rate, respiratory_rate, vasopressor_requirement |
| aki_24h_full | creatinine, potassium, bicarbonate, MAP, BUN, sodium |
| aki_48h_full | creatinine, potassium, bicarbonate, MAP, BUN, sodium |
| mimiciv_observation_6h | anion_gap, bicarbonate, BUN, calcium, chloride, creatinine, FiO2, glucose, heart_rate, hematocrit, hemoglobin, lactate, magnesium, MAP, minute_volume, O2 saturation, PaCO2, PaO2, pH, phosphate, potassium, respiratory_rate, sodium, temperature, urine_output |

Key deltas from the cached full rerun:

| Cohort | Target | Median delta vs baseline | Median delta vs placebo |
|---|---|---:|---:|
| cardiovascular_full_6h | heart_rate | -0.247222 | -0.249876 |
| cardiovascular_full_6h | MAP | -0.369981 | -0.371737 |
| sepsis_6h_full | MAP | -0.306954 | -0.311221 |
| sepsis_6h_full | urine_output | -2.197772 | -2.370074 |
| AKI 24h | creatinine | -0.053483 | -0.054585 |
| AKI 24h | BUN | -0.648400 | -0.679650 |
| AKI 48h | creatinine | -0.053978 | -0.056186 |
| AKI 48h | BUN | -0.794947 | -0.816441 |
| MIMIC-IV 6h | glucose | -0.564341 | -0.607153 |
| MIMIC-IV 6h | MAP | -0.273235 | -0.276151 |
| MIMIC-IV 6h | PaO2 | -0.382534 | -0.452251 |
| MIMIC-IV 6h | urine_output | -1.042320 | -1.077995 |

This firms up two previously incomplete findings:

1. the cardiovascular heart-rate near-miss becomes a full validation at scale;
2. the shared belief layer generalizes beyond eICU into MIMIC-IV, where it
   improves many dense physiology targets while still rejecting sparse deep
   markers such as troponin, thyroid tests, bilirubin, INR/PTT, and inflammatory
   specialty markers.

The result is not that the connected model predicts everything. It is more
precise: cached all-model belief features add stable signal for dense,
mechanistically coupled physiology across cardiovascular, sepsis, AKI, and
MIMIC-IV observation cohorts; sparse specialty targets still fall back.

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

## Engineering Status

The cache layer is now implemented in `eicu_all_model_belief_coupling_audit.py`.
It writes local row-level belief parquet caches and small manifests, then reuses
those features for the full scan. `.belief_cache/` is ignored by Git, and the
committed output is aggregate-only.
