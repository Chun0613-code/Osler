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

The result initially looked promising because it was the first DKA candidate in
this series to beat persistence on active-DKA glucose:

| Active-DKA Target | v5 JEPA | PhysioNet-Calibrated | Persistence | Result |
|---|---:|---:|---:|---|
| Glucose | 148.3214 | 94.4103 | 96.6471 | small win |
| pH | 0.0870 | 0.0575 | 0.0600 | small win |
| Anion gap | 3.0388 | 3.2290 | 3.2500 | tiny win vs persistence |
| MAP | 13.0363 | 9.8653 | 16.2683 | clear win |
| Potassium | 0.6896 | 0.9137 | 0.4400 | loses |
| Bicarbonate | 5.2205 | 6.4371 | 2.7500 | loses |
| Creatinine | 0.4819 | 6.0188 | 0.3400 | loses |

However, the bottom-line metrics reveal this as a warning result rather than a
runtime improvement. Simulator-side counterfactual sign accuracy at 6 steps was
only `0.8437`, below v5 and v6. Factual normalized MSE at 6h was `1.115254`,
far worse than v5. The latent space did not formally collapse, but effective
rank was only `6.055`, triggering `low_rank_physiology_warning: true`.

Interpretation:

> PhysioNet calibration changed the glucose point estimate, but the full
> measurement-mask version regressed on dynamics and representation quality. The
> glucose win is therefore not sufficient evidence of a better treatment world
> model.

The committed comparison report is
`dka_v5_vs_physionet_calibrated_comparison.json`.

## Presentation-Only Ablation

The full calibration bundled presentation/profile priors with a real ICU
measurement mask/age model. To separate the useful component from the harmful
one, a second full-budget arm was run with the same presentation/profile priors
but with the measurement model disabled:

```bash
python train_intervention_jepa.py \
  --physionet-calibration physionet2019_dkabody_calibration.json \
  --disable-physionet-measurement-model \
  --scenarios 1000 \
  --sequence-length 12 \
  --epochs 55
```

This ablation supports the hypothesis that presentation/profile calibration is
useful, while the full real missingness model induces regression-to-prior.

| Metric | Full Calibration | Presentation Only | Direction |
|---|---:|---:|---|
| Effective latent rank | 6.055 | 9.344 | improves |
| Low-rank warning | true | false | improves |
| Simulator factual MSE 6h | 1.115254 | 0.479113 | improves |
| Counterfactual sign accuracy 6h | 0.8437 | 0.8806 | improves |
| Active-DKA glucose MAE | 94.4103 | 81.8959 | improves |
| Active-DKA potassium MAE | 0.9137 | 0.8133 | improves but still loses persistence |
| External symbolic changed-only | 0.6071 | 0.5357 | worsens |

Interpretation:

> The glucose improvement survives without the real measurement model and becomes
> stronger. The representation and simulator-side dynamics also recover
> substantially. Therefore the good part is likely the calibrated
> presentation/profile prior. The heavy real missingness model is not safe to use
> as-is because it pushes the model toward averaging under sparse observations.

Even presentation-only is not promotable: potassium, bicarbonate, sodium,
osmolality, creatinine, and urine output still fail persistence, and external
symbolic direction accuracy worsens. The committed ablation report is
`dka_physionet_calibration_ablation_comparison.json`.

## Presentation-Only LTC Candidate

The continuous-time candidate was run as an apples-to-apples dynamics test:
PhysioNet presentation/profile priors were enabled, the real ICU measurement
model was disabled, and the only intended change was
`dynamics_cell = "ltc"` instead of the presentation-only residual MLP baseline.

| Gate | Presentation-Only Residual MLP | Presentation-Only LTC | Result |
|---|---:|---:|---|
| Effective latent rank | 9.344 | 9.002 | both safe; LTC slightly lower |
| Low-rank warning | false | false | pass |
| Simulator factual MSE 6h | 0.479113 | 0.464397 | LTC slightly better |
| Action sensitivity, shuffled - factual | 0.454763 | 0.457847 | LTC slightly better |
| Counterfactual sign accuracy 6h | 0.8806 | 0.8791 | LTC no better |
| Counterfactual sign accuracy 12h | 0.8706 | 0.8750 | LTC slightly better |
| External symbolic changed-only | 0.5357 | 0.5357 | tie |
| Active-DKA glucose MAE | 81.8959 | 103.2699 | LTC loses persistence |
| Active-DKA glucose persistence | 96.6471 | 96.6471 | fixed comparator |

Decision: reject the LTC checkpoint and keep the presentation-only residual MLP
baseline as the current calibration ablation keeper.

Interpretation:

> LTC is viable as an implementation: it does not collapse and it slightly
> improves simulator-side factual/action-conditioning metrics. But the gains do
> not survive the real-proxy promotion gates. The candidate loses the active-DKA
> glucose persistence win that made presentation-only interesting, and it adds no
> external symbolic changed-only signal. It remains research-only.

The committed comparison report is
`dka_physionet_presentation_only_ltc_comparison.json`.

## Buffer Hypothesis Audit

A follow-up audit tested whether the remaining persistence failures could be
explained by DKABody being too jumpy or under-buffered for blood-buffered
variables such as HCO3 and serum potassium.

The audit compares PhysioNet action-unobserved factual dynamics with DKABody
no-action trajectories. It is not a treatment-effect calibration target because
PhysioNet 2019 has no medication action channels.

Key one-hour results:

| Feature | Sim/PhysioNet p90 abs-delta ratio | Autocorr gap, sim - PhysioNet | Mean-reversion beta gap, sim - PhysioNet |
|---|---:|---:|---:|
| HCO3 | 0.621 | +0.071 | -0.065 |
| Potassium | 0.452 | +0.105 | +0.061 |
| pH | 0.113 | +0.272 | -0.315 |
| Glucose | 0.478 | +0.135 | -0.180 |

Interpretation:

> The audit does not support the simple under-buffering story. DKABody no-action
> is not more volatile or less autocorrelated than PhysioNet for HCO3 or
> potassium. The stronger signal is a recovery mismatch: PhysioNet
> action-unobserved HCO3, pH, and glucose tend to move toward setpoint, while
> DKABody no-action does not. That points more to missing observed
> treatment/recovery dynamics than to passive blood buffering alone.

The committed aggregate report is `dka_buffer_hypothesis_audit.json`.

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
