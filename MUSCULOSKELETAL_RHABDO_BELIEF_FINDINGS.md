# Musculoskeletal / Rhabdomyolysis Belief Findings

Date: 2026-07-10

This audit tests whether a patient-specific musculoskeletal / rhabdomyolysis
belief state can become an additional whole-body personalization component.

The candidate compresses each patient's own muscle-injury markers, renal /
electrolyte release pattern, perfusion-clearance stress, and observed factual
treatment context into online-compatible features:

- muscle injury: CPK, myoglobin, LDH;
- renal/electrolyte release: potassium, phosphate, calcium, creatinine, BUN;
- clearance/perfusion stress: urine output, MAP, lactate, bicarbonate;
- observed context: fluids, bicarbonate, renal replacement, diuretics, calcium
  and potassium repletion, vasopressor.

The gate is unchanged:

- baseline: ordinary factual `ridge_realfit`;
- candidate: `ridge_realfit + musculoskeletal belief features`;
- placebo: `ridge_realfit + same-number random features`;
- validation: 7 patient-heldout splits plus hospital-heldout support;
- authority: factual observation/prediction only.

This does **not** grant causal, treatment-planning, clinical, runtime,
checkpoint-promotion, or active-rule authority.

## Cohort

- Dataset: `eicu_musculoskeletal_rhabdo_transitions_6h.parquet`
- Rows: `16,837`
- Subjects: `1,205`
- Hospitals: `22`
- Horizon: `6h`
- Belief feature count: `59`
- Audit report: `eicu_musculoskeletal_belief_audit.json`

## Result

No target passed the full promotion gate.

| Target | Patient Splits Passing Both | Median Delta vs Baseline | Median Delta vs Placebo | Status |
|---|---:|---:|---:|---|
| MAP | 6 / 7 | -0.2802 | -0.3312 | near-miss, candidate-only |
| BUN | 0 / 7 | -0.0939 | -0.6097 | rejected |
| creatinine | 0 / 7 | +0.0227 | -0.0256 | rejected |
| potassium | 0 / 7 | +0.0182 | +0.0105 | rejected |
| bicarbonate | 0 / 7 | +0.0529 | -0.0182 | rejected |
| calcium | 0 / 7 | +0.0225 | -0.0116 | rejected |
| urine output | 0 / 7 | +6.5239 | +1.7554 | rejected |
| pH | 0 / 7 | +0.0053 | +0.0017 | rejected |
| CPK / myoglobin / LDH / phosphate / ionized calcium / lactate | 0 / 7 | insufficient or unstable | insufficient or unstable | rejected |

Hospital-heldout did not rescue the candidate. MAP showed useful direction but
the confidence interval crossed zero versus baseline, so it remains
candidate-only. Several targets improved versus placebo but worsened versus the
baseline, which is exactly why the capacity-matched placebo gate is necessary
but not sufficient.

## Interpretation

This is a clean negative result for a new organ-specific belief family.

The musculoskeletal surface is covered by the body-system router, but the
predict-update muscle-injury state does not yet add robust held-out information
beyond the ordinary table/action model.  In the current eICU contract, muscle
injury is mostly a bounded proxy: CPK, myoglobin, LDH, and downstream renal /
electrolyte changes are too sparse or too treatment/measurement dependent to
support a promoted 6h personalization layer.

The strongest signal is MAP, not CPK or renal/electrolyte targets.  That
suggests the candidate is mostly detecting nonspecific perfusion/critical-illness
stress, which is already partly covered by the validated cardiovascular,
immune, endocrine, renal, respiratory, and GI belief families.

## Boundary

Allowed:

- report musculoskeletal / rhabdomyolysis belief as a tested candidate-only
  loop;
- keep the feature module for future richer-observability or alternate-horizon
  experiments;
- use this audit as evidence that musculoskeletal coupling has not yet met the
  same validation bar as renal, cardiovascular, electrolyte, respiratory,
  endocrine, immune, or GI/nutrition beliefs.

Not allowed:

- adding musculoskeletal belief to the all-model promoted builder;
- claiming muscle-to-kidney or muscle-to-electrolyte coupling is validated;
- causal, counterfactual, clinical, or runtime claims.
