# Physiological Modes of the Learned Linear Dynamics

Date: 2026-07-10

This audit turns the learned linear model into interpretable physiology by
decomposing it with linear algebra, and then tests which parts of that
decomposition are real and which are artifacts.

## Method

We fit, per cohort, the linearized 6h dynamics matrix `A`:

```text
delta_state(6h)  ~=  A @ current_state        (both standardized)
```

Predicting the **change**, not the level, removes the trivial persistence
identity, so `A` estimates the system Jacobian around the homeostatic operating
point. Each row of `A` is a ridge fit on the rows where that target's delta was
observed. Then:

- **SVD** `A = U S V^T` — every linear map is rotate → stretch → rotate.
  Right vectors = combinations of current variables that drive motion; left
  vectors = combinations that move; singular values = coupling strength;
  effective rank = how many independent directions the dynamics actually uses.
- **Eigendecomposition** of `A` — relaxation modes. Negative real part means the
  mode decays back toward setpoint; `|real|` is speed per 6h step.

Implementation: `physiological_modes_svd.py`. Aggregate-only output; no
row-level data or identifiers.

## Cohorts (identical 14-variable state)

`glucose, potassium, sodium, bicarbonate, bun, creatinine, heart_rate, map,
o2sat, respiratory_rate, temperature, lactate, wbc, platelets`

| Cohort | Rows |
|---|---:|
| eICU sepsis 6h | 422,244 |
| MIMIC-IV all-ICU 6h | 640,164 |
| MIMIC-IV sepsis-only 6h (ICD-selected) | 139,472 |

## Result 1 — invariants that replicate across database AND disease

| Quantity | eICU sepsis | MIMIC all-ICU | MIMIC sepsis |
|---|---:|---:|---:|
| Effective rank (of 14) | 12.06 | 12.02 | 12.34 |
| Mean-reverting modes | 14/14 | 14/14 | 14/14 |
| Fastest relaxation, half-life | 5.3 h | 4.9 h | 5.2 h |
| Second relaxation, half-life | 6.2 h | 6.3 h | 6.0 h |
| Leading singular value | 0.556 | 0.579 | 0.562 |

Three findings replicate across two independent databases and across disease
state:

1. Short-term physiology uses about **12 of 14 independent directions** — the
   dynamics is high-dimensional and distributed, not collapsed onto a few modes.
2. **Every** mode is mean-reverting. The linearized dynamics is stable in every
   direction.
3. The two fastest relaxation modes have half-lives of roughly **5 h and 6 h**,
   tightly reproduced in all three cohorts.

## Result 2 — a refuted hypothesis: mode composition is database-driven

The *composition* of the leading modes differed between eICU and MIMIC:

- eICU sepsis mode 1 was dominated by a single variable (o2sat, loading 0.92).
- MIMIC all-ICU mode 1 was mixed (respiratory rate 0.62, o2sat 0.54,
  temperature −0.45, MAP 0.33).

We hypothesized this reflected **disease** (sepsis physiology couples
differently). Restricting MIMIC to ICD-selected sepsis stays tests this
directly. It does not hold:

- MIMIC **sepsis** mode 1: respiratory rate 0.58, temperature −0.53, o2sat 0.53,
  MAP 0.31 — essentially the MIMIC all-ICU mode, **not** the eICU sepsis mode.

Restricting to the same disease did not make MIMIC resemble eICU. Therefore the
mode composition is a property of the **database** — its charting frequency,
measurement practice, and anchor construction — not of the physiology or the
disease. The hypothesis is rejected.

**Consequence: specific mode identities must not be over-interpreted.** Only the
aggregate invariants above are trustworthy.

## Caveat that observational data cannot remove

The diagonal of `A` (a variable's own mean-reversion) is confounded with
**regression to the mean**: a value measured high is partly high because of
measurement noise, so the next measurement is lower. This produces mean-reversion
that looks identical to homeostatic restoration.

Consequently "14/14 modes are mean-reverting" is **homeostasis plus regression to
the mean**, and observational data cannot separate them — cross-database
agreement does not help, because both databases carry measurement noise.

The **off-diagonal** structure (cross-variable loadings) is *not* explained by
regression to the mean, and is the part that reflects genuine coupling. But per
Result 2, its specific composition is database-dependent.

Separating true homeostatic restoration from regression to the mean requires
either an explicit measurement-error model or **perturbation data**: apply a
known external input and observe how the loop responds. This is the same
conclusion reached by the control-loop and RD audits.

## What is and is not established

Established (replicated across two databases and two disease states):

- effective dimensionality of short-term dynamics (~12/14);
- global stability of the linearized dynamics (all modes mean-reverting, subject
  to the regression-to-the-mean caveat);
- relaxation half-lives of the two fastest modes (~5 h and ~6 h).

Not established:

- the physiological identity of individual modes (database-dependent);
- that mean-reversion is homeostasis rather than measurement regression;
- anything causal.

## Boundary

Factual and observational only. This grants no causal, counterfactual,
treatment-effect, clinical, runtime, checkpoint-promotion, or active-rule
authority. The eigenvectors are descriptive modes of an empirical linear fit,
not verified physiological control loops.
