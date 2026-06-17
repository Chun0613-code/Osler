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

## Full-Budget Candidate Result

The calibrated simulator was tested at the same full candidate budget used for
the v6 and PhysioNet-transfer closing experiments:

- 1,000 simulated DKA scenarios
- 12-step branches
- 55 epochs
- Grey-box residual enabled
- Demo action prior enabled
- PhysioNet-calibrated presentation/profile/mask-age model enabled

Result: do not promote the full checkpoint over v5.

The result is not a flat failure. It is the first DKA candidate in this series to
beat persistence on active-DKA glucose:

| Active-DKA Target | v5 JEPA | PhysioNet-Calibrated | Persistence | Result |
|---|---:|---:|---:|---|
| Glucose | 148.3214 | 94.4103 | 96.6471 | small win |
| pH | 0.0870 | 0.0575 | 0.0600 | small win |
| Anion gap | 3.0388 | 3.2290 | 3.2500 | tiny win vs persistence |
| MAP | 13.0363 | 9.8653 | 16.2683 | clear win |
| Potassium | 0.6896 | 0.9137 | 0.4400 | loses |
| Bicarbonate | 5.2205 | 6.4371 | 2.7500 | loses |
| Creatinine | 0.4819 | 6.0188 | 0.3400 | loses |

Simulator-side counterfactual sign accuracy at 6 steps was `0.8437`, below v5
and v6. The latent space did not collapse, but effective rank was only `6.055`,
which is a low-rank warning.

Interpretation:

> PhysioNet calibration changed the evidence in the right direction for glucose,
> pH, anion gap, and MAP, but it did not produce a safe full-model replacement.
> The strongest likely contribution is the real measurement mask/age model and
> more realistic presentations. The remaining failures still point to unresolved
> treated dynamics, especially potassium and bicarbonate.

The committed comparison report is
`dka_v5_vs_physionet_calibrated_comparison.json`.

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
