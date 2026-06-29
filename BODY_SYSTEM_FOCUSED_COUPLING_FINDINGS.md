# Body-System Focused Coupling Findings

Date: 2026-06-29

This is the fourth cross-system coupling pass after body-system breadth
completion.  It stops testing all edges generically and focuses on the two most
plausible frontiers from the edge-specific audit:

1. renal-electrolyte buffering with explicit potassium and bicarbonate store
   dynamics;
2. cardio-renal long-horizon coupling, where perfusion stress has enough time to
   change creatinine, BUN, and urine output.

The gate is unchanged:

- baseline: downstream ridge model without focused coupling features;
- candidate: baseline plus focused coupling features;
- placebo: baseline plus the same number of random features;
- pass rule: candidate must significantly beat both baseline and placebo on
  held-out patients.

This report is aggregate-only.  It grants no causal, counterfactual, clinical,
runtime, checkpoint-promotion, or active-rule authority.

## Result

Focused coupling produces the first robust validated cross-system factual edge:
cardio-renal long-horizon coupling.  Renal-electrolyte explicit store dynamics
remain candidate-only.

| Focus | Rows | Subjects | Hospitals | Active Pass-Both Targets | Hospital-Heldout | Interpretation |
|---|---:|---:|---:|---|---|---|
| renal-electrolyte store, 6h | 17,387 | 1,249 | 26 | bicarbonate 1/7, anion gap 1/7; potassium/sodium/phosphate 0/7 | no promoted target | rejected for promotion |
| cardio-renal, 24h | 530,265 | 29,508 | 203 | creatinine 7/7, BUN 7/7, urine output 5/7 | creatinine, BUN, and urine output pass all-window; creatinine/BUN pass active-window | validated factual coupling edge |
| cardio-renal, 48h | 397,897 | 21,427 | 198 | creatinine 7/7, BUN 7/7, urine output 4/7 | creatinine and BUN pass; urine output does not robustly pass hospital-heldout | validated for creatinine/BUN; urine partial |

## Detailed Signal

At 24h, cardio-renal focused features improve active-window normalized MAE over
both baseline and placebo in all seven patient split seeds for creatinine and
BUN:

| Target | All-Window Passes | Active Passes | Active Median Delta vs Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| creatinine | 7/7 | 7/7 | -0.019332 | -0.019409 |
| BUN | 7/7 | 7/7 | -0.012082 | -0.012057 |
| urine output | 7/7 | 5/7 | -0.002525 | -0.002680 |

At 48h, the same edge remains robust for creatinine and BUN:

| Target | All-Window Passes | Active Passes | Active Median Delta vs Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| creatinine | 7/7 | 7/7 | -0.021573 | -0.021702 |
| BUN | 7/7 | 7/7 | -0.018239 | -0.018301 |
| urine output | 7/7 | 4/7 | -0.002734 | -0.002762 |

Renal-electrolyte store features do not pass.  The explicit potassium store and
bicarbonate buffer state produce only weak bicarbonate/anion-gap signals and no
potassium signal:

| Target | All-Window Passes | Active Passes |
|---|---:|---:|
| bicarbonate | 2/7 | 1/7 |
| anion gap | 1/7 | 1/7 |
| potassium | 0/7 | 0/7 |
| sodium | 0/7 | 0/7 |
| phosphate | 0/7 | 0/7 |

## Interpretation

This closes the first whole-body coupling question with a split answer:

- same-window renal-electrolyte buffering is still not observable enough to
  promote;
- long-horizon cardio-renal coupling is validated as a factual predictive edge.

The physiology is coherent.  Perfusion burden and renal reserve do not need to
change creatinine/BUN inside a six-hour window to be useful.  At 24-48 hours,
the time scale matches renal kinetics, and the focused state passes both the
baseline and capacity-matched placebo gates.

This is the first Osler-JEPA result that moves beyond isolated organ-system
routers into a validated cross-organ factual coupling:

```text
cardiovascular perfusion / shock burden
  -> renal reserve / afterload / recovery-drive state
  -> creatinine and BUN at 24-48h
```

## Safety Boundary

The validated cardio-renal edge is a research factual coupling only.  It does
not say that changing vasopressors, fluids, or inotropes causally changes renal
outcomes.  It does not authorize runtime treatment decisions, active symbolic
rule edits, checkpoint promotion, or clinical deployment.

The renal-electrolyte focused store remains candidate-only.  Its next step is
richer observability: treatment timing, KCl/bicarbonate dosing, urine
electrolytes when available, and better acid-base/ventilation context.
