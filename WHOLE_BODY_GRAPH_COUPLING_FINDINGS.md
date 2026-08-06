# Whole-Body Graph Coupling Findings

Date: 2026-07-06

This audit asks whether Osler-JEPA can be coupled more tightly after the
previous explicit whole-body latent layer.

The baseline is deliberately strong:

- ordinary table/action features;
- all five current personalized belief families;
- explicit whole-body latent axes and cross-axis interactions.

The candidate adds one fixed physiology-inspired graph propagation step across
the organ-system axes:

```text
hemodynamic <-> respiratory <-> electrolyte/acid-base <-> renal
                     |                 |
                 metabolic/endocrine
```

The placebo adds the same number of random graph columns.  Passing means the
graph propagation layer adds incremental factual signal beyond both the
all-belief layer and the earlier explicit whole-body latent layer.

This remains a factual observation/prediction audit.  It grants no causal,
counterfactual, clinical, runtime, treatment-planning, checkpoint-promotion, or
active-rule authority.

## Result

Only one target/cohort cell fully validates:

| Cohort | Target | Median delta vs explicit latent baseline | Median delta vs placebo | Interpretation |
|---|---|---:|---:|---|
| MIMIC-IV observation 6h | creatinine | -0.000680 | -0.000937 | small renal/systemic graph increment |

The validated gain is small but real under the gate: 7/7 patient splits and the
held-out group gate both beat the explicit latent baseline and the
capacity-matched placebo.

## Near Misses

Several targets show directionally useful signal but fail the full gate:

| Cohort | Target | Split result | Median delta vs explicit latent baseline | Status |
|---|---|---:|---:|---|
| MIMIC-IV observation 6h | BUN | 5/7 | -0.005825 | candidate-only |
| MIMIC-IV observation 6h | respiratory rate | 5/7 | -0.001212 | candidate-only |
| MIMIC-IV observation 6h | anion gap | 4/7 | -0.003005 | candidate-only |
| AKI 24h | MAP | 4/7 | -0.006564 | candidate-only |
| AKI 24h | BUN | 3/7 | -0.007880 | candidate-only |
| MIMIC-IV observation 6h | lactate | 3/7 | -0.001580 | candidate-only |

These are not promoted.  They are useful evidence about where graph propagation
almost helps, not validated capabilities.

## What This Means

The answer to "can the model be more coupled?" is:

```text
Yes, but only a little with the current observed-table inputs.
```

The explicit latent layer already captures most of the stable cross-system
signal.  A second graph-propagation step can add a small renal/systemic
increment for creatinine, but it does not broadly improve the model.  In many
targets it becomes redundant with the existing latent axes or mildly
over-smooths local physiology.

This is an important boundary:

- validated: small graph increment for creatinine;
- candidate-only: BUN, respiratory rate, anion gap, MAP, lactate;
- rejected: most specialty biomarkers, coagulation markers, liver markers,
  glucose, heart rate, oxygen saturation, and core electrolytes.

## Updated Coupling Stack

The current whole-body coupling stack is now:

```text
per-system routers
  -> per-system predict-update belief states
  -> explicit whole-body latent axes
  -> graph-propagated coupling candidate layer
```

Only the following part of the graph layer is validated:

```text
graph-propagated renal/systemic signal -> MIMIC-IV 6h creatinine
```

Everything else remains candidate-only or fallback.

## Boundary

Allowed:

- report graph propagation as a small validated increment for MIMIC-IV 6h
  creatinine;
- keep BUN, respiratory rate, anion gap, MAP, and lactate as candidate-only
  graph near-misses;
- use the graph feature code for future shadow/research audits.

Not allowed:

- claim broad graph-level whole-body simulation;
- claim treatment effect or causal coupling;
- override the explicit latent layer for nonvalidated targets;
- use graph features for clinical recommendations or treatment planning.
