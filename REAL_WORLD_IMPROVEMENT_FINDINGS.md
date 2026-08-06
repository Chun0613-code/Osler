# Real-world JEPA improvement without additional data

## Experiment

The existing 12-stay MIMIC-IV demo was reused without adding records. Evaluation
used four-fold patient-stay cross-fitting. Hyperparameters and the choice between
persistence, raw JEPA, and an adapter were selected only inside each outer
training fold.

The following methods were tested:

- treatment-episode purity weighting;
- per-action propensity-overlap weighting;
- combined propensity and episode weighting;
- a ridge residual adapter over JEPA factual forecasts;
- a 20-member patient-bootstrap adapter ensemble;
- state-wise fallback to persistence or raw JEPA;
- cross-fitted temperature scaling;
- abstention based on ensemble direction agreement.

## Numerical result

On active-DKA windows, normalized MAE changed as follows:

| Method | Normalized MAE |
|---|---:|
| Raw v5 JEPA | 0.56781 |
| Unweighted ensemble adapter | 0.38101 |
| Episode-weighted ensemble | 0.37124 |
| Propensity-weighted ensemble | 0.37474 |
| Combined weighting | 0.38495 |
| Persistence | 0.35789 |
| State-wise safe hybrid | 0.34099 |

The safe hybrid improves over raw JEPA by 0.228 normalized MAE. Its improvement
over persistence is 0.0164, but the stay-bootstrap 95% interval is
`[-0.05299, 0.01787]`; this is a promising signal, not statistically stable proof.

Important active-DKA state results:

| State | Raw JEPA | Persistence | Safe hybrid |
|---|---:|---:|---:|
| Glucose, mg/dL | 148.3214 | 96.6471 | 70.0533 |
| Bicarbonate, mEq/L | 5.2205 | 2.7500 | 2.6650 |
| Anion gap, mEq/L | 3.0388 | 3.2500 | 2.8710 |
| MAP, mmHg | 13.0363 | 16.2683 | 14.8592 |
| Effective osmolality | 22.1216 | 9.5340 | 9.3094 |

For pH, potassium, creatinine, and urine output, inner patient validation usually
selected persistence. The deployment artifact preserves that fallback instead
of forcing an adapter onto every variable.

## Confidence and abstention

Temperature scaling did not change direction accuracy, but it corrected severe
overconfidence:

| Metric | Before | After |
|---|---:|---:|
| Mean confidence | 0.9183 | 0.4297 |
| Expected calibration error | 0.4103 | 0.0783 |
| Multiclass Brier score | 0.8795 | 0.6425 |

For changed directions, accepting only predictions with complete agreement among
the adapter ensemble produced 75.76% accuracy at 32.74% coverage. A 90% agreement
threshold produced 71.32% accuracy at 52.58% coverage. The runtime therefore
marks lower-agreement adapter predictions as `abstain`.

## What did not work

- Propensity weighting alone did not beat episode weighting.
- Combining propensity and episode weights was worse than either alone on
  active-DKA windows.
- Applying the residual adapter to every state remained worse than persistence.
- Selecting only on active-DKA inner windows overfit some sparse states and was
  less stable than the general state-wise selector.
- Calibration fixed confidence, not the underlying direction accuracy.

## Decision

The guarded factual adapter is worth retaining for research. It substantially
reduces the simulator-to-EHR error of the raw JEPA and shows a meaningful glucose
improvement. It is not yet proven superior to persistence across states.

The adapter is restricted to factual forecasting under an observed treatment
sequence. It must not be used to estimate a causal treatment effect or to update
active Osler rules. Treatment confounding remains because propensity scores
cannot adjust for unmeasured severity or recover isolated interventions that do
not exist in the data.
