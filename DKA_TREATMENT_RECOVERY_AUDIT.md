# DKA Treatment-Recovery Audit

Date: 2026-06-25

## Question

Does `DKABody` lack the treatment-driven recovery dynamics needed to move
glucose, acid-base state, and potassium toward a viable state?

The audit separates four things that must not be mixed:

1. a fixed clinical-prior check;
2. numeric dose-captured eICU falsification;
3. treatment-presence evidence without dose;
4. no-evidence windows.

No parameter is fitted to eICU. The clinical-prior check cites:

> Umpierrez GE et al. *Hyperglycemic Crises in Adults With Diabetes: A
> Consensus Report.* Diabetes Care. 2024;47(8):1257-1275.
> <https://doi.org/10.2337/dci24-0032>

## Fixed Protocol Check

The neutral `DKABody` was given a fixed one-hour reference treatment:

```text
IV insulin 6 U/hr + isotonic fluid 500 mL/hr + KCl 10 mEq/hr
```

Glucose fell from `480.0` to `413.93 mg/dL`, a decline of
`66.07 mg/dL/hr`. This is inside the fixed `50-75 mg/dL/hr` research boundary.
The patient remained alive; HCO3, pH, and anion gap all moved in the expected
recovery direction.

This check passes without fitting the scenario to eICU.

## eICU Action Boundary

The audit was rerun after correcting the eICU treatment contract:

- `infusionDrug` rows with defensible U/hr rates enter numeric action grids;
- `medication` rows are orders, not administrations;
- `treatment` rows are coarse presence text;
- unknown-concentration infusion rows are presence evidence only.

This removes medication-order leakage from the numeric action grid.

## Active-DKA Treatment Strata

The audit replayed all 367 active-DKA windows from 182 stays using the stricter
six-hour action grid and prior treatment history.

| Stratum | Rows | Stays |
|---|---:|---:|
| Numeric insulin dose captured | 35 | 22 |
| Insulin evidence only | 181 | 104 |
| No insulin evidence | 151 | 83 |

### Numeric insulin dose captured

| Target | Rows | Real change/hr | Sim change/hr | Direction agreement | Simulator MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 34 | -24.50 | -13.38 | 79.4% | 94.76 | 150.03 |
| HCO3 | 19 | +0.98 | +1.32 | 89.5% | 5.56 | 5.96 |
| Anion gap | 16 | -1.31 | -0.78 | 62.5% | 5.35 | 8.38 |
| Potassium | 18 | -0.060 | -0.017 | 66.7% | 0.71 | 0.81 |

The numeric-dose stratum remains physiologically encouraging: glucose and
acid-base direction are mostly correct, and several targets beat persistence.
It is also too small for promotion.

### Insulin evidence only

| Target | Rows | Real change/hr | Sim change/hr | Direction agreement | Simulator MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 157 | -22.66 | +17.95 | 17.2% | 249.01 | 152.64 |
| HCO3 | 114 | +0.74 | -0.48 | 20.2% | 7.99 | 5.44 |
| Anion gap | 85 | -1.26 | +1.31 | 9.4% | 15.67 | 8.41 |
| Potassium | 111 | -0.077 | +0.072 | 36.9% | 0.98 | 0.68 |

These rows are the key evidence for the coverage diagnosis. The raw data says
insulin likely existed, but the demo does not provide a usable numeric dose. The
simulator is therefore replaying a falsely untreated patient.

### No insulin evidence

| Target | Rows | Real change/hr | Sim change/hr | Direction agreement | Simulator MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 119 | -12.51 | +15.21 | 16.8% | 171.65 | 90.40 |
| HCO3 | 74 | +0.49 | -0.28 | 21.6% | 5.42 | 3.98 |
| Anion gap | 51 | -0.92 | +1.02 | 13.7% | 11.70 | 6.13 |
| Potassium | 60 | -0.074 | +0.035 | 46.7% | 0.79 | 0.68 |

These windows may still contain missing treatment, but the demo does not prove
it. They are not permission to weaken untreated DKA physiology.

## Death Coverage

The simulator produced 26 deaths:

| Cause | Total | Numeric dose | Evidence only | No evidence |
|---|---:|---:|---:|---:|
| Acidosis | 22 | 1 | 16 | 5 |
| Circulatory collapse | 3 | 1 | 0 | 2 |
| Hyperkalemia | 1 | 0 | 1 | 0 |

Seventeen of 26 deaths have insulin presence evidence but no numeric dose.
Those are explicitly coverage-limited. The two deaths with numeric dose captured
remain useful falsification cases, but are too few and heterogeneous to identify
a new equation.

## Decision

`runtime_change_allowed: false`

The standard protocol passes its sourced boundary, and the numeric-dose stratum
does not fail the external direction gate. Therefore:

- do not tune insulin, glucose, HCO3, or potassium constants to this cohort;
- do not add generic setpoint mean reversion;
- keep untreated DKA progressive;
- do not use medication orders or treatment text as numeric administrations;
- improve dose-resolved treatment capture before revisiting evidence-only deaths;
- do not promote a checkpoint or make causal or clinical claims.

The reusable gate is now stricter: a mechanism may change only when a fixed
source boundary and a dose-observed external falsification agree.
