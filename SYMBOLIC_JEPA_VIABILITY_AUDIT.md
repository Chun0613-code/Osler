# Symbolic JEPA Viability Audit

Date: 2026-06-15

## Question

Can Prolog-compiled patient state and a homeostatic world-model objective improve
the DKA JEPA without weakening the existing numerical model?

## Controlled runs

All three candidates used 1,000 simulated patient scenarios, 11 intervention
branches, 12 steps, 35 epochs, the same seed, and the same MIMIC-IV demo proxy.
No candidate checkpoint was promoted or committed.

| Model | Validation loss | Latent effective rank | 12-step effect sign accuracy |
|---|---:|---:|---:|
| Existing v5 | 0.3689 | 13.338 | 0.9666 |
| Compiler in dynamics + scalar viability | 0.2898 | 8.462 | 0.9737 |
| Compiler in dynamics + residual viability | 0.4350 | 8.344 | 0.9693 |
| Compiler isolated to symbolic heads | 0.4142 | 8.708 | 0.9688 |

Active-DKA MIMIC MAE:

| Model | Glucose | pH | HCO3 | Anion gap | Potassium | MAP |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 96.65 | 0.0600 | 2.75 | 3.25 | 0.44 | 16.27 |
| Existing v5 | 148.32 | 0.0870 | 5.22 | 3.04 | 0.69 | 13.04 |
| Scalar viability | 194.12 | 0.0476 | 8.08 | 2.31 | 0.70 | 12.68 |
| Residual viability | 194.34 | 0.0421 | 7.78 | 2.33 | 0.74 | 11.25 |
| Symbolic-only compiler | 191.61 | 0.0485 | 8.29 | 1.76 | 0.73 | 11.60 |

## Decision

- Keep dka_symbolic_jepa_v5.pt as the current checkpoint.
- Do not promote any v6 candidate.
- Keep the state compiler as shared Prolog/symbolic-head context only.
- Prohibit compiler context from changing continuous dynamics; this is covered
  by a regression test.
- Keep viability, intervention-sensitivity, and burden-ordering terms as reported
  audits with zero default dynamics weight.
- Train uncertainty on detached rollout latents so calibration cannot distort
  the world model.
- Keep every rule proposal candidate-only and every causal claim disabled.

## Interpretation

The experiments improved simulator-side pH, anion-gap, MAP, and direction
metrics, but worsened glucose, bicarbonate, potassium, and latent rank on the
real-data proxy. Lower simulator validation loss was therefore not evidence of
better patient dynamics. The current bottleneck is the simulator-to-real
distribution gap, not the Prolog-to-JEPA interface.

The MIMIC proxy contains only 12 stays and is not sufficient for causal
validation. These results are an engineering rejection gate, not clinical
evidence.
