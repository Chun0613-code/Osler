# PhysioNet 2019 ICU JEPA Findings

## What Changed

The local PhysioNet/CinC Challenge 2019 cohort adds a real-data, patient-split
pretraining check for Osler JEPA. Unlike the 15-stay DKA demo, this is a
large-scale real ICU time-series task with explicit observation masks and
measurement ages.

This path is not action-conditioned. It validates general ICU state forecasting,
not DKA treatment effects.

## Main Result

On the held-out PhysioNet 2019 test split, the general ICU JEPA beats
persistence on 22 of 34 dynamic variables. The strongest strategically relevant
wins are on the same dense physiology targets that failed in the small DKA demo:

| Variable | JEPA MAE | Persistence MAE | Observed Targets | Result |
|---|---:|---:|---:|---|
| Glucose | 29.7517 | 34.5768 | 16,144 | JEPA wins |
| Potassium | 0.3875 | 0.4446 | 8,554 | JEPA wins |
| HCO3 | 2.1490 | 2.2472 | 3,954 | JEPA wins |
| pH | 0.0450 | 0.0495 | 5,554 | JEPA wins |
| MAP | 9.4334 | 10.4268 | 88,752 | JEPA wins |
| Lactate | 1.0971 | 1.2174 | 2,050 | JEPA wins |

The latent space did not collapse:

- Mean latent std: `0.213926`
- Minimum latent std: `0.177820`
- Effective rank: `23.037`
- Active dimensions: `48 / 48`

## Interpretation

This changes the strategic reading of the earlier DKA demo failures. The
wrong-signed glucose behavior in the 15-stay DKA demo is now better explained by
small N, simulator-to-real mismatch, and treatment confounding than by JEPA being
intrinsically unable to learn dense ICU physiology.

The important proof-of-concept is this:

> With enough real patient time-series data, the same broad JEPA style can beat
> persistence on glucose, potassium, acid-base markers, and MAP.

That makes the full MIMIC/eICU DKA cohort path a justified investment rather
than a blind bet.

## Boundary

This result must not be over-claimed.

- PhysioNet 2019 has no explicit DKA treatment action channels.
- It does not validate insulin, KCl, fluids, bicarbonate, or dextrose effects.
- It does not support counterfactual or causal treatment claims.
- It does not replace `dka_symbolic_jepa_v5.pt`.

Allowed use:

- Generic ICU state encoder pretraining.
- Missingness and measurement-age representation research.
- Controlled transfer experiments into the DKA model.

Disallowed use:

- Runtime promotion over the DKA symbolic JEPA.
- Clinical treatment recommendation.
- Counterfactual treatment-effect claims.

## Next Controlled Experiment

The only transfer that is worth testing is feature-aligned encoder
initialization. Shared variables can seed the DKA encoder:

| DKA State | PhysioNet Variable |
|---|---|
| `G` | `Glucose` |
| `pH` | `pH` |
| `HCO3` | `HCO3` |
| `Ke` | `Potassium` |
| `MAP` | `MAP` |
| `creatinine` | `Creatinine` |

All other DKA-specific states, including `anion_gap`, `V`, `I`, `Na`,
`osmolality`, `urine_output`, `BHB`, `K_store`, and `osmotic_injury`, must remain
randomly initialized or simulator-trained. No transfer experiment may bypass the
same persistence, symbolic, and safety gates used for v5/v6.

## Transfer Experiment Result

The feature-aligned transfer experiment was run at the same full candidate
budget used for the v6 closing experiment:

- 1,000 simulated DKA scenarios
- 12-step branches
- 55 epochs
- Grey-box residual enabled
- Demo action prior enabled
- Initial checkpoint: `dka_physionet_encoder_init.pt`

Result: do not promote.

| Active-DKA Target | v5 JEPA | PhysioNet Transfer | Persistence | Transfer Result |
|---|---:|---:|---:|---|
| Glucose | 148.3214 | 295.2941 | 96.6471 | loses to v5 and persistence |
| Potassium | 0.6896 | 0.7243 | 0.4400 | loses to v5 and persistence |
| MAP | 13.0363 | 11.2647 | 16.2683 | wins |
| Anion gap | 3.0388 | 3.1598 | 3.2500 | wins persistence, loses v5 |
| pH | 0.0870 | 0.0614 | 0.0600 | near persistence, wins v5 |
| Bicarbonate | 5.2205 | 10.7297 | 2.7500 | loses |

The simulator-side metrics remained strong: 6-step counterfactual sign accuracy
was `0.9655`, action shuffling worsened MSE by `0.602902`, and the latent space
did not collapse. The external MIMIC factual gate still failed for the dense DKA
targets that matter most.

Interpretation:

> Generic ICU pretraining proves that JEPA can learn factual physiology at
> scale, but direct feature-aligned encoder transfer does not solve the DKA
> intervention/persistence failure. The remaining bottleneck is still
> action-conditioned, causal-grade DKA data rather than generic ICU encoder
> weights.
