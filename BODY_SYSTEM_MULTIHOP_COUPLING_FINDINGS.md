# Body-System Multihop Coupling Findings

Date: 2026-06-29

This audit tests whether validated single-hop physiology can be chained into a
multi-hop whole-body factual path.

The first tested path is:

```text
sepsis / immune-inflammatory burden
  -> MAP mediator
  -> renal targets
```

The baseline is intentionally strong: a direct sepsis-to-renal ridge model.  The
candidate adds discovery-only MAP mediator features.  The placebo adds the same
number of random features.  Passing means the mediated MAP path adds downstream
factual signal beyond both the direct edge and capacity-matched noise.

The follow-up long-horizon audit tests the cleaner physiologic composition:

```text
sepsis / immune-inflammatory burden
  -> MAP at 6h
  -> renal targets at 24h and 48h
```

This uses the validated short-horizon sepsis -> MAP edge as the first hop and
the validated cardio-renal 24-48h time scale as the downstream window.

This remains factual and observational.  It grants no causal, counterfactual,
clinical, runtime, checkpoint-promotion, or active-rule authority.

## Result

The original 6h multi-hop path is a strong candidate, but not a full promoted
whole-body path under the same conservative standard used for single edges.  The
time-scale-corrected 24h path validates for creatinine and BUN.  The 48h path
is candidate-only.

| Path | Rows | Subjects | Hospitals | Active Pass-Both Targets | Hospital-Heldout | Interpretation |
|---|---:|---:|---:|---|---|---|
| sepsis -> MAP -> renal, 6h | 422,244 | 25,175 | 204 | BUN 7/7, creatinine 6/7, urine output 0/7 | creatinine active passes; BUN all-window passes but active baseline comparison is not significant; urine output fails | strong candidate-only multi-hop path |
| sepsis -> MAP6 -> renal, 24h | 328,759 | 20,478 | 202 | creatinine 7/7, BUN 7/7, urine output 1/7 | creatinine and BUN pass active-window and all-window; urine output fails | validated factual multi-hop path for creatinine/BUN |
| sepsis -> MAP6 -> renal, 48h | 238,649 | 14,529 | 197 | creatinine 5/7, BUN 5/7, urine output 0/7 | creatinine/BUN do not beat direct baseline significantly | candidate-only |

## Detailed Signal

| Target | All-Window Passes | Active Passes | Active Median Delta vs Direct Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| BUN | 6/7 | 7/7 | -0.001108 | -0.001240 |
| creatinine | 6/7 | 6/7 | -0.001693 | -0.001707 |
| urine output | 0/7 | 0/7 | +0.000075 | -0.000034 |

The time-scale-corrected 24h path validates.  It uses `map_tp6` as the
discovery-only mediator label and predicts 24h renal targets:

| Target | All-Window Passes | Active Passes | Active Median Delta vs Direct Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| BUN | 7/7 | 7/7 | -0.001967 | -0.002098 |
| creatinine | 7/7 | 7/7 | -0.002556 | -0.002587 |
| urine output | 0/7 | 1/7 | -0.000115 | -0.000179 |

Hospital-heldout support is strong for the 24h path:

- creatinine active-window and all-window pass both direct-baseline and placebo
  comparisons;
- BUN active-window and all-window pass both direct-baseline and placebo
  comparisons;
- urine output fails.

At 48h, the path weakens and does not promote:

| Target | All-Window Passes | Active Passes | Active Median Delta vs Direct Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| BUN | 3/7 | 5/7 | -0.001410 | -0.001781 |
| creatinine | 7/7 | 5/7 | -0.002382 | -0.002512 |
| urine output | 0/7 | 0/7 | +0.000099 | -0.000214 |

Hospital-heldout support is mixed:

- creatinine active-window passes both direct-baseline and placebo comparisons;
- BUN all-window passes, but active-window candidate-vs-direct-baseline does not
  reach significance;
- urine output fails.

## Interpretation

This is the first validated evidence that Osler-JEPA's body-system layer can
express more than isolated edges.  The key was matching the downstream renal
time scale: forcing the whole path into 6h produced only a strong candidate,
while using MAP at 6h and renal targets at 24h validates creatinine and BUN.

The right status is:

```text
single-hop factual edges:
  cardio -> renal                 validated
  sepsis / immune -> MAP          validated

multi-hop path:
  sepsis -> MAP6 -> renal24       validated for creatinine/BUN
  sepsis -> MAP -> renal6         strong candidate, not promoted
  sepsis -> MAP6 -> renal48       candidate-only
```

## Safety Boundary

This audit does not claim that changing antibiotics, fluids, vasopressors,
steroids, MAP, or any cardiovascular treatment causally changes renal outcomes.
It only says that a discovery-only 6h MAP mediator carries held-out factual
renal-prediction signal beyond a direct sepsis-to-renal model at 24h.

The next multi-hop step should expose the 24h sepsis -> MAP6 -> renal path as a
shadow factual state object and keep the 48h path candidate-only.  It should not
be promoted to causal or clinical use without external identification evidence.
