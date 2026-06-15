# Grey-Box and Safe Hybrid Audit

Date: 2026-06-15

## Existing EHR Action Recovery

The strict MIMIC-IV demo extraction now includes `ingredientevents` maintenance
context in addition to `inputevents` and eMAR:

- 15 usable stays and 252 six-hour transitions
- 399 maintenance events in the strict DKA cohort
- free water, oral intake, enteral nutrition, parenteral nutrition, and explicit
  carbohydrate have separate channels
- calories are not converted to carbohydrate
- unknown nutrition composition remains context-only

The demo contained no safely attributable TPN/enteral carbohydrate mass in these
DKA windows. Replaying the richer trajectories therefore remained 11/16 simulated
deaths with essentially unchanged MAE. This rejects the assumption that the demo
alone contains enough hidden maintenance treatment to explain the residual gap.

## Empirical Action Prior

`dka_action_prior_demo_v1.json` compiles the observed route and dose support.
Randomized simulator branches and warm-up treatment now use empirical active
probabilities and positive-dose quantiles. Insulin route sampling is mutually
exclusive. Supplied protocol doses are clipped to the observed 99th percentile.

This prior is a demo-domain support constraint, not a clinical dosing policy.

## Constrained Grey-Box Residual

The candidate residual ODE has 286 parameters and can correct only concentration
derivatives. It cannot directly modify total-body potassium, fluid volume,
insulin PK depots, administered-dose mass balance, or osmotic injury.

Nested patient-group cross-fit results at six hours:

| State | Mechanism MAE | Grey-box MAE |
| --- | ---: | ---: |
| Glucose | 323.50 | 195.24 |
| Ketone pool | 8.62 | 6.04 |
| HCO3 | 7.43 | 5.83 |
| Serum K | 0.80 | 0.71 |
| Sodium | 8.28 | 4.87 |
| Creatinine | 1.03 | 0.95 |

The first experimental run incorrectly used the outer test fold for early
stopping and was discarded. The reported result uses an inner patient-group
validation split and a sealed outer test fold.

Despite consistent improvement, the artifact remains `candidate_only` and
`promotion_allowed: false` because only 15 stays are available and treatment
confounding remains unresolved.

## Safe Hybrid Contract

The hybrid v2 runtime now reports, per state:

- candidate source and prediction
- selected source and prediction
- fallback source
- ensemble direction agreement
- abstention decision and reason
- requested and supported horizon

Promotion requires at least three of four outer folds to select the same
non-persistence method and lower out-of-fold MAE than persistence. No state passed
both conditions. MAP was the closest candidate, but only two folds selected the
adapter and its all-window safe-hybrid MAE was slightly worse than persistence.
The generated v2 artifact therefore selects persistence for every state.

Overall safe-hybrid versus persistence normalized MAE difference was +0.00398
with 95% bootstrap interval [-0.00231, 0.01212]. Active-DKA hybrid was worse.

## Long-Horizon Check

The factual 6/12/24-hour test found:

- MAP beat persistence at 6h and 12h with 15 stays, but lost at 24h.
- Potassium beat persistence at 12h.
- Anion gap and potassium beat persistence at 24h, but each had only three
  measured targets and cannot support promotion.
- Glucose, sodium, creatinine, and urine output remained worse than persistence.

Longer horizon is therefore useful as an evaluation axis, but it is not evidence
that the current JEPA generally beats persistence.

## Not Executed

eICU/HiRID transfer pretraining was not run because those datasets are not present
in the workspace. Literature-curve calibration was not populated with unsourced
constants. Both can use the same transition/action contracts once a licensed
dataset or an explicitly approved target-curve source is supplied.
