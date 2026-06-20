# MIMIC-III Demo DKA Transition Findings

## What Was Added

`mimiciii_demo_dka_transition_extract.py` adapts the local PhysioNet MIMIC-III Clinical Database Demo v1.4 into the same observed-treatment 6-hour DKA proxy contract used by the MIMIC-IV and eICU evaluators:

```text
state_t + history_action_grid + future_action_grid + treatment lifecycle + state_t+6h
```

The extractor reads ICU stays, labs, charted ICU vitals, urine output, and `INPUTEVENTS_CV` / `INPUTEVENTS_MV` treatment administrations. It does not treat `PRESCRIPTIONS` rows as administered medication. Prescription matches are reported only as an action-capture quality signal.

## Cohort Summary

The local MIMIC-III demo files contain no ICD-coded DKA admissions. This run is therefore a lab-defined DKA-like schema and power audit, not a diagnosis-confirmed DKA cohort.

| Metric | Value |
|---|---:|
| Lab-defined DKA-like stays before filtering | 42 |
| ICD-confirmed DKA stays | 0 |
| Evaluable stays | 40 |
| 6-hour transitions | 482 |
| Active-DKA transitions | 66 |
| Rows with any model comparison | 233 |
| Rows with active-DKA comparison | 56 |

Core target pair counts:

| Target | Paired rows |
|---|---:|
| Glucose | 332 |
| Potassium | 124 |
| Bicarbonate | 95 |
| pH | 128 |
| MAP | 455 |

Observed action-target support remains below prior reference gates:

| Channel | Stays | Active-DKA stays | Reference status |
|---|---:|---:|---|
| Fluids -> MAP | 37 | 29 | Below 42-stay MAP reference |
| Insulin any -> glucose | 18 | 7 | Below 77-stay glucose reference |
| KCl -> potassium | 9 | 1 | Below reference |
| Bicarbonate -> HCO3 | 5 | 1 | Underpowered |

## Factual Proxy Results

These are observational factual forecasts under recorded treatment grids. They do not estimate causal treatment effects.

### V5 Symbolic JEPA

| Scope | Target | JEPA MAE | Persistence MAE | Point estimate |
|---|---|---:|---:|---|
| All windows | Glucose | 159.9953 | 47.7303 | Loss |
| All windows | Potassium | 0.7320 | 0.4758 | Loss |
| All windows | Bicarbonate | 4.4986 | 2.5385 | Loss |
| All windows | pH | 0.0846 | 0.0569 | Loss |
| All windows | MAP | 10.9318 | 11.3247 | Small win |
| Active DKA | Glucose | 180.9064 | 98.4706 | Loss |
| Active DKA | Potassium | 0.8014 | 0.7571 | Loss |
| Active DKA | Bicarbonate | 4.1669 | 4.9167 | Small win |
| Active DKA | pH | 0.1471 | 0.1391 | Loss |
| Active DKA | MAP | 14.3401 | 13.8485 | Loss |

### PhysioNet Presentation-Only Candidate

| Scope | Target | JEPA MAE | Persistence MAE | Point estimate |
|---|---|---:|---:|---|
| All windows | Glucose | 81.4488 | 47.7303 | Loss |
| All windows | Potassium | 0.6916 | 0.4758 | Loss |
| All windows | Bicarbonate | 4.4060 | 2.5385 | Loss |
| All windows | pH | 0.0639 | 0.0569 | Loss |
| All windows | MAP | 11.3519 | 11.3247 | Tie/loss |
| Active DKA | Glucose | 101.4579 | 98.4706 | Slight loss |
| Active DKA | Potassium | 0.7380 | 0.7571 | Small win |
| Active DKA | Bicarbonate | 7.4982 | 4.9167 | Loss |
| Active DKA | pH | 0.1334 | 0.1391 | Small win |
| Active DKA | MAP | 14.1600 | 13.8485 | Loss |

## Symbolic Real Test

`symbolic_real_test.py` now labels `mimiciii_*` cohorts as MIMIC-III demo rather than MIMIC-IV demo.

Patient-held-out symbolic rule test summary:

| Metric | Value |
|---|---:|
| Discovery stays | 27 |
| Held-out stays | 13 |
| Discovery records | 29 |
| Held-out records | 14 |
| Direction accuracy, all | 0.5658 |
| Direction accuracy, changed-only | 0.6087 |
| Mean proposal confidence | 0.8829 |
| Candidate rules proposed | 10 |
| Retrospectively validated | 2 |
| Automatically promoted to active | 0 |

The rule sandbox behaved correctly: candidate rules stayed candidate-only and no active Osler rule was modified.

## Interpretation

This is useful as a third external schema smoke test. It proves that the MIMIC-III demo can be mapped into the same observed-treatment action/state contract as MIMIC-IV and eICU, including treatment timing, dose grids, lifecycle events, and pre-anchor history.

It does not prove model superiority. The cohort has 0 diagnosis-confirmed DKA stays, action-target support is below the prior power references, and the model wins are sparse point estimates. No checkpoint should be promoted from this result.

## Decision

- Checkpoint promotion: no.
- Causal or counterfactual claim: no.
- Clinical claim: no.
- Artifact role: aggregate external schema/power smoke test.
- Next useful step: use this adapter pattern for larger credentialed MIMIC-III/eICU-style cohorts where diagnosis confirmation and action-target support are adequate.
