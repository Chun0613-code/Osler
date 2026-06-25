# DKA Treatment-Recovery Audit

Date: 2026-06-25

## Question

Does `DKABody` lack the treatment-driven recovery dynamics needed to move
glucose, acid-base state, and potassium toward a viable state?

The audit separates three things that must not be mixed:

1. a fixed clinical-prior check;
2. observational eICU falsification;
3. treatment-coverage failure.

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

## eICU Treatment Strata

The audit replayed all 367 active-DKA windows from 182 stays using their
six-hour treatment grids and prior treatment history.

The full cohort looks poor because most windows have no captured insulin. The
treatment-bearing strata tell a different story.

### Captured insulin without dextrose

| Target | Rows | Real change/hr | Sim change/hr | Direction agreement | Simulator MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 24 | -28.10 | -17.08 | 91.7% | 82.24 | 168.58 |
| HCO3 | 15 | +0.97 | +1.93 | 100.0% | 7.00 | 5.95 |
| Anion gap | 14 | -1.32 | -1.48 | 71.4% | 5.56 | 11.10 |
| Potassium | 15 | -0.052 | -0.027 | 60.0% | 0.84 | 0.67 |

### Captured insulin with dextrose

| Target | Rows | Real change/hr | Sim change/hr | Direction agreement | Simulator MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 65 | -27.40 | -12.51 | 73.8% | 123.73 | 173.62 |
| HCO3 | 50 | +0.91 | +1.17 | 84.0% | 4.98 | 5.84 |
| Anion gap | 36 | -1.41 | -0.70 | 77.8% | 5.40 | 8.58 |
| Potassium | 49 | -0.103 | -0.043 | 81.6% | 0.56 | 0.87 |

The captured-treatment windows do not falsify the treatment theory. Glucose,
acid-base, anion-gap, and potassium directions are mostly correct, and several
targets beat persistence.

## Coverage Failure

Among active-DKA windows, 270 have no captured insulin. In that stratum:

- real glucose changes by `-15.49 mg/dL/hr`;
- simulated glucose changes by `+17.11 mg/dL/hr`;
- direction agreement is only `18.6%`;
- real HCO3 and anion gap improve while untreated simulation worsens.

That pattern is consistent with missing treatment capture. Weakening untreated
DKA physiology to imitate these windows would encode an extraction artifact as
biology.

The simulator produced 17 deaths:

| Cause | Total | No captured insulin | Captured insulin |
|---|---:|---:|---:|
| Acidosis | 13 | 12 | 1 |
| Circulatory collapse | 3 | 2 | 1 |
| Hyperkalemia | 1 | 0 | 1 |

Fourteen of 17 deaths are therefore coverage-limited. The remaining three are
useful falsification cases, but are too few and heterogeneous to identify a new
equation.

## Decision

`runtime_change_allowed: false`

The standard protocol passes its sourced boundary, and captured-insulin windows
do not fail the external direction gate. Therefore:

- do not tune insulin, glucose, HCO3, or potassium constants to this cohort;
- do not add generic setpoint mean reversion;
- keep untreated DKA progressive;
- improve treatment capture before revisiting the 14 coverage-limited deaths;
- retain the three captured-treatment deaths as future falsification cases;
- do not promote a checkpoint or make causal or clinical claims.

The main theoretical correction is now encoded as a gate: a mechanism may
change only when a fixed source boundary and a treatment-bearing external
falsification agree.
