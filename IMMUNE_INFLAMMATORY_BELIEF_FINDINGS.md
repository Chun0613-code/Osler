# Immune / Inflammatory Belief Findings

Date: 2026-07-06

This audit tests a sixth online predict-update belief family for Osler-JEPA:
immune / inflammatory physiology.

The candidate compresses each patient's own observed:

- fever / hypothermia;
- WBC leukocytosis / leukopenia;
- lactate / MAP perfusion stress;
- heart-rate, respiratory-rate, and oxygenation response;
- creatinine / platelet / albumin systemic stress;
- observed antibiotics, fluids, vasopressor, and steroid context.

The gate is unchanged:

- baseline: ordinary table/action ridge forecast;
- candidate: baseline plus immune belief features;
- placebo: baseline plus the same number of random features;
- validation: 7 patient-heldout splits plus hospital-heldout support;
- authority: factual observation/prediction only.

This grants no causal, counterfactual, clinical, treatment-planning, runtime,
checkpoint-promotion, or active-rule authority.

## Bounded Immune Cohort

The bounded eICU immune/inflammatory module has 18,392 rows.

It shows real signal but does not promote any target:

| Target | Split result | Median delta vs baseline | Median delta vs placebo | Status |
|---|---:|---:|---:|---|
| respiratory_rate | 6/7 | -0.150390 | -0.161594 | near-miss |
| MAP | 5/7 | -0.328954 | -0.365127 | near-miss |
| heart_rate | 3/7 | -0.166098 | -0.166425 | candidate-only |
| O2 saturation | 3/7 | -0.059572 | -0.066808 | candidate-only |
| WBC | 1/7 | -0.329090 | -0.740092 | candidate-only |

This bounded cohort is underpowered for promotion, but it correctly identifies
cardiopulmonary and perfusion targets as the likely signal carriers.

## Full Sepsis Cohort

The same immune belief was then run on the full eICU sepsis transition cohort
with 422,244 rows.

The larger cohort validates six downstream observable targets:

| Target | Patient splits | Median delta vs baseline | Median delta vs placebo | Status |
|---|---:|---:|---:|---|
| MAP | 7/7 | -0.295895 | -0.297295 | validated |
| creatinine | 7/7 | -0.015808 | -0.016520 | validated |
| O2 saturation | 7/7 | -0.114907 | -0.115244 | validated |
| heart_rate | 7/7 | -0.201006 | -0.203133 | validated |
| respiratory_rate | 7/7 | -0.157555 | -0.158313 | validated |
| vasopressor_requirement | 7/7 | -0.066870 | -0.066877 | validated |

Near-misses:

| Target | Split result | Median delta vs baseline | Status |
|---|---:|---:|---|
| urine_output | 5/7 | -0.421228 | candidate-only |
| WBC | 4/7 | -0.019322 | candidate-only |
| temperature | 2/7 | -0.005982 | candidate-only |
| lactate | 1/7 | -0.006273 | candidate-only |

## Interpretation

This is the sixth validated personalization/belief family in the observation
layer:

```text
renal
cardiovascular
electrolyte / acid-base
respiratory / gas exchange
endocrine / glycemic-adrenal stress
immune / inflammatory host response
```

It strengthens the body model in exactly the place that was previously thin:
immune/inflammatory stress as an upstream whole-body state.  The validated
targets are not purely "immune" variables; they are coupled downstream
physiology:

- perfusion: MAP and vasopressor requirement;
- cardiopulmonary response: O2 saturation, heart rate, respiratory rate;
- renal/systemic effect: creatinine.

That is why this belief matters for whole-body coupling.  It adds an
inflammatory host-response state that can connect sepsis/inflammation to
cardiovascular, respiratory, and renal observables.

## Boundary

Allowed:

- use the immune belief as a validated factual feature source for the six full
  sepsis targets listed above;
- include immune belief features in future all-model belief coupling audits;
- treat WBC, temperature, lactate, and urine_output as candidate-only until they
  pass the full gate.

Not allowed:

- causal or treatment-effect claims;
- claiming antibiotics, fluids, vasopressors, or steroids would change outcomes;
- clinical recommendations;
- active rule promotion;
- claiming the hidden immune state is directly measured physiologic truth.
