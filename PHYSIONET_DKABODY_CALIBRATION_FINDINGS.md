# PhysioNet DKABody Calibration Findings

## What Was Calibrated

`physionet2019_calibrate_dkabody.py` builds an aggregate DKABody calibration
artifact from the local PhysioNet/CinC Challenge 2019 cohort. The artifact uses
40,336 patient files and 1,552,210 hourly ICU rows. It contains no patient rows
or identifiers.

The calibration covers four safe simulator priors:

1. **Presentation joint distribution**
   - DKA-like definition: `Glucose >= 200` and (`HCO3 <= 18` or `pH <= 7.30`).
   - Requires `Glucose`, `HCO3`, `Potassium`, and `MAP`.
   - First qualifying row per patient.
   - Records: 673 DKA-like presentations, 653 complete for the multivariate
     joint model.

2. **Observable patient-variability proxies**
   - Uses aggregate presentation proxies for renal reserve, vascular tone,
     fluid retention, counterregulatory drive, potassium-store scale, and
     baseline creatinine.
   - Keeps weight, insulin sensitivity, endogenous insulin, and sodium as
     documented unobserved defaults because Challenge 2019 lacks the needed
     measurements/actions.

3. **Action-unobserved drift**
   - Computes 1h/2h/4h/6h factual drift from DKA-like anchors.
   - This is not a clean no-treatment causal estimate because PhysioNet 2019 has
     no explicit medication/action channels.
   - It is stored as a simulator review target, not as treatment-effect truth.

4. **Measurement model**
   - Stores feature-level observation rates, measurement interval quantiles, and
     consecutive-change noise proxies.
   - This feeds optional state-specific observation masks and measurement ages
     during synthetic JEPA training.

## Key Aggregate Priors

Presentation quantiles from complete DKA-like rows:

| Variable | p05 | p50 | p95 |
|---|---:|---:|---:|
| Glucose | 203.0 | 246.0 | 481.6 |
| HCO3 | 10.0 | 17.0 | 25.0 |
| Potassium | 3.2 | 4.3 | 5.9 |
| MAP | 55.534 | 75.5 | 105.4 |

Measurement realism examples:

| Variable | Observation Rate | Median Interval | p90 Interval |
|---|---:|---:|---:|
| Glucose | 0.171057 | 3h | 10h |
| HCO3 | 0.041894 | 6h | 24h |
| pH | 0.069303 | 6h | 9h |
| Potassium | 0.093109 | 6h | 23h |
| MAP | 0.875487 | 1h | 1h |

## No-Action Audit

`physionet2019_dkabody_calibration_audit.py` samples calibrated DKABody
presentations and simulates no-action trajectories. It compares simulator drift
against PhysioNet action-unobserved factual drift.

The important result:

| 6h Glucose Drift | Median Delta / Hour |
|---|---:|
| DKABody no-action | +23.878624 |
| PhysioNet action-unobserved | -12.5 |
| Gap | +36.378624 |

Interpretation:

> This gap is expected and useful. DKABody no-action physiology lets untreated
> hyperglycemia worsen. PhysioNet action-unobserved drift often improves because
> real ICU patients are being treated, but treatment actions are not represented
> in the Challenge 2019 files. Therefore the drift artifact must not be used as a
> no-treatment causal target.

The calibration is safe because it upgrades the simulator's starting states,
observable heterogeneity, and measurement process without pretending that
PhysioNet identifies DKA treatment effects.

## Training Integration

`train_intervention_jepa.py` now accepts:

```bash
python train_intervention_jepa.py \
  --physionet-calibration physionet2019_dkabody_calibration.json
```

When enabled, synthetic DKABody branches use:

- PhysioNet-calibrated DKA-like presentation sampling.
- PhysioNet-derived observable patient-variability proxies.
- PhysioNet-derived observation mask and measurement-age sampling.

The default training path remains unchanged unless the flag is provided.

## Boundary

Allowed:

- More realistic simulated presentation distribution.
- More realistic missingness and measurement intervals.
- Simulator auditing against real ICU drift.
- Candidate-only JEPA experiments.

Not allowed:

- Claiming PhysioNet drift is untreated DKA drift.
- Claiming treatment-effect calibration from PhysioNet 2019.
- Replacing `dka_symbolic_jepa_v5.pt` without the same external gates.
- Clinical claims.
