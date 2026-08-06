# GI / Pancreatic / Nutrition Belief Findings

Date: 2026-07-06

This audit tests an online predict-update belief family for gastrointestinal,
pancreatic, and nutrition physiology.

The candidate compresses each patient's own observed:

- pancreatic injury proxies: lipase, amylase, triglycerides;
- nutrition reserve proxies: albumin, total protein, calcium;
- gut/perfusion/metabolic stress: lactate, MAP, bilirubin, bicarbonate,
  creatinine;
- observed fluids, nutrition, PPI, octreotide, albumin, antibiotics,
  vasopressor, and insulin context.

The gate is unchanged:

- baseline: ordinary table/action ridge forecast;
- candidate: baseline plus GI/nutrition belief features;
- placebo: baseline plus the same number of random features;
- validation: 7 patient-heldout splits plus hospital-heldout support;
- authority: factual observation/prediction only.

This grants no causal, counterfactual, clinical, nutrition/treatment-planning,
runtime, checkpoint-promotion, or active-rule authority.

## Result

One downstream observable validates:

| Target | Patient splits | Median delta vs baseline | Median delta vs placebo | Status |
|---|---:|---:|---:|---|
| MAP | 7/7 | -0.392584 | -0.459293 | validated |

Nonvalidated targets:

| Target | Split result | Median delta vs baseline | Status |
|---|---:|---:|---|
| albumin | 0/7 | +0.031788 | rejected |
| total_protein | 0/7 | +0.002338 | rejected |
| calcium | 0/7 | +0.019199 | rejected |
| glucose | 0/7 | +0.498442 | rejected |
| lactate | 0/7 | +0.053627 | rejected |
| bilirubin | 0/7 | +0.142133 | rejected |
| lipase / amylase / triglycerides | 0/7 | n/a | too sparse |

## Interpretation

This is not a complete GI digital twin.  The deep GI/pancreatic/nutrition
targets remain too sparse or too weakly constrained in the 6h eICU table data.

The validated signal is narrower but still useful:

```text
GI / nutrition / gut-perfusion stress belief -> MAP
```

That makes this the seventh validated personalization/belief family, but with a
tight boundary.  It adds a gut/nutrition/perfusion state that helps blood
pressure prediction in the GI/nutrition cohort.  It does not validate
pancreatic enzyme, bilirubin, albumin, protein, calcium, glucose, or lactate
forecasting.

## Boundary

Allowed:

- use the GI/nutrition belief as a validated factual feature source for MAP in
  the GI/nutrition cohort;
- include GI/nutrition belief features in future all-model coupling audits;
- report non-MAP GI targets as rejected or candidate-only according to the gate.

Not allowed:

- causal or treatment-effect claims;
- nutrition, albumin, PPI, octreotide, fluids, insulin, or vasopressor planning;
- clinical recommendations;
- claiming the hidden GI/nutrition state is directly measured physiologic truth.
