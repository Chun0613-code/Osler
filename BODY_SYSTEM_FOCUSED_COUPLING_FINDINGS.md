# Body-System Focused Coupling Findings

Date: 2026-06-29

This is the fourth cross-system coupling pass after body-system breadth
completion.  It stops testing all edges generically and focuses on the most
plausible frontiers from the edge-specific audit and clinical physiology:

1. renal-electrolyte buffering with explicit potassium and bicarbonate store
   dynamics;
2. cardio-renal long-horizon coupling, where perfusion stress has enough time to
   change creatinine, BUN, and urine output.
3. sepsis/immune burden to cardiovascular tone and lactate clearance;
4. hepatic burden to renal stress.

The gate is unchanged:

- baseline: downstream ridge model without focused coupling features;
- candidate: baseline plus focused coupling features;
- placebo: baseline plus the same number of random features;
- pass rule: candidate must significantly beat both baseline and placebo on
  held-out patients.

This report is aggregate-only.  It grants no causal, counterfactual, clinical,
runtime, checkpoint-promotion, or active-rule authority.

## Result

Focused coupling now produces two validated cross-system factual edges:
cardio-renal long-horizon coupling and sepsis/immune burden to MAP.  Hepato-
renal coupling shows a stronger candidate signal than the broad edge-specific
pass, but it does not reach the 7/7 boundary.  Renal-electrolyte explicit store
dynamics remain candidate-only.

| Focus | Rows | Subjects | Hospitals | Active Pass-Both Targets | Hospital-Heldout | Interpretation |
|---|---:|---:|---:|---|---|---|
| renal-electrolyte store, 6h | 17,387 | 1,249 | 26 | bicarbonate 1/7, anion gap 1/7; potassium/sodium/phosphate 0/7 | no promoted target | rejected for promotion |
| cardio-renal, 24h | 530,265 | 29,508 | 203 | creatinine 7/7, BUN 7/7, urine output 5/7 | creatinine, BUN, and urine output pass all-window; creatinine/BUN pass active-window | validated factual coupling edge |
| cardio-renal, 48h | 397,897 | 21,427 | 198 | creatinine 7/7, BUN 7/7, urine output 4/7 | creatinine and BUN pass; urine output does not robustly pass hospital-heldout | validated for creatinine/BUN; urine partial |
| sepsis/immune -> cardiovascular, 6h | 422,244 | 25,175 | 204 | MAP 7/7, heart rate 5/7, lactate 4/7 | MAP and heart rate pass; lactate is placebo-only in hospital-heldout | validated for MAP; HR/lactate partial |
| hepatic -> renal, 6h | 17,730 | 1,096 | 88 | creatinine 3/7, BUN 3/7 | creatinine passes hospital-heldout but not random patient splits | candidate-only |
| hepatic -> renal, 24h | 13,269 | 864 | 82 | creatinine 2/7, BUN 0/7 | creatinine and BUN pass hospital-heldout, but random splits fail | candidate-only; heterogeneous |
| hepatic -> renal, 48h | 9,316 | 596 | 73 | creatinine 3/7, BUN 0/7 | no target passes hospital-heldout | candidate-only; not a simple horizon fix |

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

Sepsis/immune to cardiovascular coupling passes for MAP at 6h.  This edge uses
inflammatory burden, vasoplegia, capillary-leak, and treatment-context features
from the full eICU sepsis cohort:

| Target | All-Window Passes | Active Passes | Active Median Delta vs Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| MAP | 7/7 | 7/7 | -0.001883 | -0.001869 |
| heart rate | 6/7 | 5/7 | -0.001198 | -0.001253 |
| lactate | 4/7 | 4/7 | -0.002611 | -0.003664 |

Hepatic to renal coupling improves over the prior broad audit but does not
promote.  Its 6h active-window signals are 3/7 for both creatinine and BUN:

| Target | All-Window Passes | Active Passes | Active Median Delta vs Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| creatinine | 3/7 | 3/7 | -0.034476 | -0.044751 |
| BUN | 2/7 | 3/7 | -0.016291 | -0.019774 |

Longer horizons do not rescue the hepato-renal edge.  This explicitly tests the
analogy to cardio-renal slow coupling and rejects it under the current eICU
hepatic proxy contract:

| Horizon | Target | All-Window Passes | Active Passes | Active Median Delta vs Baseline | Active Median Delta vs Placebo |
|---|---|---:|---:|---:|---:|
| 24h | creatinine | 1/7 | 2/7 | -0.027727 | -0.030759 |
| 24h | BUN | 0/7 | 0/7 | -0.008791 | -0.011425 |
| 48h | creatinine | 3/7 | 3/7 | -0.030555 | -0.048204 |
| 48h | BUN | 1/7 | 0/7 | -0.008718 | -0.024667 |

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
- sepsis/immune burden carries reproducible 6h factual signal for MAP;
- hepatic burden carries renal signal, but not reproducibly enough at 6h, 24h,
  or 48h.

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

sepsis / immune-inflammatory burden
  -> vasoplegia / capillary-leak state
  -> MAP at 6h
```

## Safety Boundary

The validated cardio-renal and sepsis-MAP edges are research factual couplings
only.  They do not say that changing vasopressors, fluids, antibiotics,
inotropes, or steroids causally changes renal or hemodynamic outcomes.  They do
not authorize runtime treatment decisions, active symbolic rule edits,
checkpoint promotion, or clinical deployment.

The renal-electrolyte focused store remains candidate-only.  Its next step is
richer observability: treatment timing, KCl/bicarbonate dosing, urine
electrolytes when available, and better acid-base/ventilation context.

The hepato-renal focused candidate also remains candidate-only.  The 24h/48h
test shows that horizon alone is not enough.  Its next step is richer hepatic
observability: INR/synthetic function, ammonia or encephalopathy severity,
ascites/volume status, albumin therapy, paracentesis/procedure context, and
cleaner hepatorenal syndrome labels.
