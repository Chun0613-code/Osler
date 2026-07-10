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

## Initial 6h Result

No target passed the full 6h promotion gate.

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

At 6h, hospital-heldout did not rescue the candidate. MAP showed useful
direction but the confidence interval crossed zero versus baseline, so it
remained candidate-only at that horizon. Several targets improved versus
placebo but worsened versus the baseline, which is exactly why the
capacity-matched placebo gate is necessary but not sufficient.

## Horizon Follow-Up

Because MAP was a 6/7 near-miss at 6h, the same audit was rerun at 1h, 3h, and
12h using the already-built musculoskeletal transition contracts.

| Horizon | Validated Targets | MAP Patient Splits | MAP Median Delta vs Baseline | MAP Median Delta vs Placebo | MAP Hospital-Heldout |
|---|---|---:|---:|---:|---|
| 1h | none | 5 / 7 | -0.2418 | -0.2643 | candidate-only; baseline CI crosses zero |
| 3h | MAP | 7 / 7 | -0.4197 | -0.4244 | pass |
| 6h | none | 6 / 7 | -0.2802 | -0.3312 | candidate-only; baseline CI crosses zero |
| 12h | MAP | 7 / 7 | -0.3003 | -0.3060 | pass |

Hospital-heldout details for the promoted horizons:

| Horizon | Baseline MAE | Belief MAE | Placebo MAE | Delta vs Baseline 95% CI | Delta vs Placebo 95% CI |
|---|---:|---:|---:|---:|---:|
| 3h MAP | 9.4424 | 8.9379 | 9.4771 | [-0.5661, -0.2551] | [-0.5625, -0.2715] |
| 12h MAP | 10.5045 | 10.0022 | 10.5485 | [-0.5784, -0.2075] | [-0.5868, -0.1783] |

## Interpretation

The horizon follow-up changes the conclusion.

Musculoskeletal / rhabdomyolysis belief validates as a bounded personalization
component for **MAP at 3h and 12h**, but not for muscle biomarkers or downstream
renal/electrolyte injury.  The useful signal is therefore:

```text
muscle injury / rhabdomyolysis / perfusion-clearance stress -> MAP
```

It is **not** yet:

```text
muscle injury -> kidney injury
muscle injury -> potassium/phosphate/calcium dynamics
```

The distinction matters.  The belief state appears to capture a short-to-middle
horizon perfusion / critical-illness stress component in rhabdomyolysis-like
patients.  It does not robustly recover CPK, myoglobin, LDH, creatinine, BUN,
potassium, phosphate, calcium, bicarbonate, pH, urine output, or lactate beyond
the baseline/placebo gate.

The most likely explanation is observability.  CPK, myoglobin, LDH, phosphate,
and ionized calcium are sparse, while renal/electrolyte consequences are
strongly treatment- and measurement-dependent.  MAP is dense enough for the
belief state to add signal.

## Boundary

Allowed:

- report musculoskeletal / rhabdomyolysis belief as a bounded validated factual
  MAP personalization source at 3h and 12h;
- add the musculoskeletal belief builder to future all-model coupling reruns,
  with cache-manifest mismatch protection;
- keep 1h and 6h MAP as candidate-only horizon cells.

Not allowed:

- claiming muscle-to-kidney or muscle-to-electrolyte coupling is validated;
- promoting CPK, myoglobin, LDH, creatinine, BUN, potassium, phosphate, calcium,
  bicarbonate, pH, urine output, or lactate from this audit;
- causal, counterfactual, clinical, or runtime claims.
