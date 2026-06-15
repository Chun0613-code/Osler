# Symbolic-grounded JEPA v5 findings

## What was built

- Four supervised JEPA heads: future direction, Osler status, proof path, and
  intervention-effect rule proposal.
- Joint losses for direction, status, proof path, proposal direction, Osler
  contradiction, and action contrast, in addition to the existing numerical
  JEPA, rollout, risk, and anti-collapse losses.
- A versioned symbolic schema and a structured inference response consumed by
  Osler.
- A guarded rule sandbox. Automated code can write candidate rules only; active
  rules require external provenance, safety tests, and human review.
- A patient-stay-held-out MIMIC test with explicit confounding and persistence
  checks.

## Simulator held-out result

The v5 model was trained on 1,000 simulated patient scenarios, each branched into
11 intervention paths for 55 epochs.

| Metric | v4 | v5 |
|---|---:|---:|
| 6 h glucose MAE | 36.5608 | 34.5893 |
| 6 h bicarbonate MAE | 1.5308 | 1.4967 |
| 6 h potassium MAE | 0.1868 | 0.1622 |
| 6 h K-store MAE | 3.4085 | 3.3343 |
| 6 h osmotic-injury MAE | 0.4844 | 0.4513 |
| 6 h counterfactual sign accuracy | 0.9622 | 0.9666 |
| Effective latent rank | 12.484 | 13.338 |

V5 is not collapsed: all 48 latent dimensions are active. Symbolic proposal
direction accuracy is 0.9665 on changed simulator effects; status accuracy is
0.9858 and proof-path F1 is 0.9977. Contradicted status remains the weakest class
at 0.6757 accuracy because only 37 held-out examples belong to that class.

## MIMIC-IV demo real test

The 12 stays were split into 8 discovery stays and 4 held-out stays with zero
patient overlap. Only 16 discovery records and 5 held-out records met the active
DKA and action requirements.

- Changed-only factual direction accuracy: 0.6429 (28 comparisons).
- Mean proposal confidence: 0.9372, showing substantial external overconfidence.
- Active-DKA glucose MAE: 148.3214 versus persistence 96.6471.
- V5 beats persistence for active-DKA anion gap and MAP only.
- Two observational pH hypotheses survived discovery filtering.
- Zero rules passed the stricter held-out/action-specificity gate.
- Zero rules were promoted into the active Osler engine.

The first permissive gate incorrectly treated co-treatment associations as
action-specific evidence. The final gate now requires discovery consistency,
three independent patient stays, held-out replication, improvement over stable
persistence, and isolated-action support. Regression tests prevent this failure
from returning.

## Decision

The symbolic bridge works as an engineering interface: JEPA produces structured
transitions and proof-path predictions, Osler can verify or reject them, and the
candidate sandbox prevents autonomous rule changes. The v5 checkpoint is useful
for simulator research and hypothesis generation.

It is not yet a reliable real-patient intervention world model. The MIMIC demo is
too small and confounded, external calibration is poor, and persistence still
wins most numerical targets. No candidate rule should enter the active engine.
