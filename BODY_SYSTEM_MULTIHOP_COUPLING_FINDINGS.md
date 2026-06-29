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

This remains factual and observational.  It grants no causal, counterfactual,
clinical, runtime, checkpoint-promotion, or active-rule authority.

## Result

The multi-hop path is a strong candidate, but not a full promoted whole-body
path under the same conservative standard used for single edges.

| Path | Rows | Subjects | Hospitals | Active Pass-Both Targets | Hospital-Heldout | Interpretation |
|---|---:|---:|---:|---|---|---|
| sepsis -> MAP -> renal, 6h | 422,244 | 25,175 | 204 | BUN 7/7, creatinine 6/7, urine output 0/7 | creatinine active passes; BUN all-window passes but active baseline comparison is not significant; urine output fails | strong candidate-only multi-hop path |

## Detailed Signal

| Target | All-Window Passes | Active Passes | Active Median Delta vs Direct Baseline | Active Median Delta vs Placebo |
|---|---:|---:|---:|---:|
| BUN | 6/7 | 7/7 | -0.001108 | -0.001240 |
| creatinine | 6/7 | 6/7 | -0.001693 | -0.001707 |
| urine output | 0/7 | 0/7 | +0.000075 | -0.000034 |

Hospital-heldout support is mixed:

- creatinine active-window passes both direct-baseline and placebo comparisons;
- BUN all-window passes, but active-window candidate-vs-direct-baseline does not
  reach significance;
- urine output fails.

## Interpretation

This is the first evidence that Osler-JEPA's body-system layer can express more
than isolated edges.  A validated first-hop sepsis/MAP state appears to add
incremental renal signal, especially for BUN.  But the path is not yet robust
enough to promote as a validated multi-hop edge because one target is 6/7 rather
than 7/7 and hospital-heldout support is not uniformly active-window strong.

The right status is:

```text
single-hop factual edges:
  cardio -> renal                 validated
  sepsis / immune -> MAP          validated

multi-hop path:
  sepsis -> MAP -> renal          strong candidate, not promoted
```

## Safety Boundary

This audit does not claim that changing antibiotics, fluids, vasopressors,
steroids, MAP, or any cardiovascular treatment causally changes renal outcomes.
It only says that a discovery-only MAP mediator carries some held-out factual
renal-prediction signal beyond a direct sepsis-to-renal model.

The next multi-hop step should either:

- test the same path at 24h/48h if a long-horizon sepsis cohort is generated; or
- use the already validated cardio-renal long-horizon edge as the downstream
  hop once a compatible sepsis/MAP long-horizon cohort exists.
