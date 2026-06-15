# DKA Realism and Promotion Audit

Date: 2026-06-15

## Implemented

- Split serum potassium shifts from total-body potassium reserve. Insulin can
  lower serum potassium but no longer destroys potassium mass.
- Bound urinary potassium loss by urine flow and a plausible urinary potassium
  concentration. KCl replenishes the total-body reserve.
- Added lagged renal perfusion and a resolving counter-regulatory stress state.
- Added fluid formulation metadata so replay distinguishes normal saline,
  half-normal saline, lactated Ringer, Plasma-Lyte, and dextrose water.
- Added route-aware dose/time provenance, treatment start/stop events, and
  explicit confidence fields. KCl carrier volume is never treated as mEq.
- Tightened DKA onset to glucose, acidosis, and ketosis evidence within one
  four-hour window. ICD DKA codes are recorded as support, not used as a
  substitute for physiologic onset.
- Added a persistence gate. Viability/world-truth losses cannot optimize JEPA
  dynamics unless a patient-held-out report has at least 30 stays, complete
  treatment timing/history, and JEPA beats persistence on glucose, potassium,
  bicarbonate, and MAP.

## Demo Cohort Result

The strict extractor completed against the local MIMIC-IV demo:

- 14 subjects
- 15 stays with usable transitions
- 252 transitions
- Core paired targets: glucose 80, potassium 85, bicarbonate 78, MAP 239
- Exact action grids, pre-anchor history, and treatment lifecycle events present
- Evaluation gate: failed because the cohort has fewer than 30 stays
- Causal claim allowed: false

The demo is useful for integration testing. It is not large enough for model
promotion, fine-tuning, or causal claims.

## Fidelity Replay Result

The new 24-hour replay used 16 strict-onset trajectories:

| Variable | MAE |
| --- | ---: |
| Glucose | 248.11 mg/dL |
| Potassium | 0.98 mEq/L |
| HCO3 | 4.47 mEq/L |
| pH | 0.10 |
| Anion gap | 7.03 mEq/L |
| Sodium | 8.37 mEq/L |
| Creatinine | 0.61 mg/dL |

Simulated death occurred in 11/16 trajectories, compared with 12/12 in the
previous audit. This is a structural improvement, but the simulator still fails
the realism gate. Most residual failures are late hyperosmolar rebound or
hypokalemia in trajectories with incomplete captured maintenance treatment.
Lowering death thresholds or injecting inferred future treatment would make the
score look better without making the world model more truthful, so neither was
done.

## Promotion Decision

The current v5 checkpoint remains active and research-only. The new dynamics
training path is disabled by default. Passing `--enable-viability-dynamics` with
the current v5 report is rejected because:

- only 12 held-out stays were evaluated;
- explicit treatment lifecycle events were absent from that old cohort;
- glucose and bicarbonate had too few active-DKA pairs;
- potassium did not beat persistence.

No v6 checkpoint was created or promoted.

## Remaining Bottleneck

The Prolog-to-JEPA interface is not the limiting factor. The remaining work is:

1. Run the strict extractor on credentialed full MIMIC-IV, not the demo.
2. Re-evaluate by held-out patient and require all four core targets to beat
   persistence before enabling dynamics constraints.
3. Limit the acute DKA simulator to the interval where recorded treatment is
   sufficiently complete, or add an explicitly observed maintenance/nutrition
   model. Do not infer hidden treatment from future outcomes.
