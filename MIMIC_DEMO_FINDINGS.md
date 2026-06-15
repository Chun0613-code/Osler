# MIMIC-IV Demo Findings

## Cohort

- Source: MIMIC-IV demo, 140 ICU stays.
- Lab-defined DKA: 12 stays.
- Six-hour transitions: 187.
- Complete core JEPA inputs: 103 transitions.
- Active-DKA comparison windows: 42.
- Exact 30-minute action grids: IV insulin, rapid-SC insulin, NPH/intermediate,
  basal insulin, fluids, KCl, bicarbonate, and dextrose.
- Prior context: six hours of dose exposure and action recency.
- BHB observations: none in this demo subset.

## JEPA Versus Persistence

V4 beats persistence for MAP and anion gap on active-DKA windows. It remains worse
for the other observed targets:
windows:

| Target | JEPA MAE | Persistence MAE |
|---|---:|---:|
| Glucose | 138.46 mg/dL | 96.65 mg/dL |
| pH | 0.083 | 0.060 |
| Bicarbonate | 5.15 mmol/L | 2.75 mmol/L |
| Anion gap | 2.77 mmol/L | 3.25 mmol/L |
| Potassium | 0.679 mmol/L | 0.440 mmol/L |
| MAP | 12.97 mmHg | 16.27 mmHg |
| Sodium | 10.70 mmol/L | 4.16 mmol/L |
| Creatinine | 0.498 mg/dL | 0.340 mg/dL |

The failure is not latent collapse. It is simulator-to-real domain mismatch.
After route separation and support expansion, only 6.2% of IV insulin cells exceed
the 20 U/hr training support; no rapid-SC or basal cells exceed their support.

## Physiological Replay

Replaying actual treatment histories through `DKABody` gives:

- Glucose MAE: 195.45 mg/dL.
- HCO3 MAE: 5.74 mmol/L.
- Potassium MAE: 0.89 mmol/L.
- 12/12 simulated deaths, although these are observed hospital trajectories.
- Route-aware PK improves glucose MAE from v3 235.32 to 195.45 mg/dL.
- Osmolality replay MAE improves from 27.81 to 10.40 mOsm/kg.

Four-fold patient-held-out calibration improves glucose MAE to 156.97 mg/dL but
still produces 83% simulated deaths. `K_HEP` reaches its lower bound and
`K_UPTAKE_INS` reaches its upper bound, showing that the current equations are
missing structure rather than merely using slightly wrong constants.

## Implemented In V4

1. Separate IV, rapid-SC, intermediate/NPH, and basal insulin PK depots.
2. Add latent total-body potassium reserve and expose its estimate to Osler.
3. Replace instantaneous hyperosmolar death with cumulative injury burden.
4. Train a 15-state, eight-action, 16-history-feature JEPA checkpoint.

## Remaining Model Changes

1. Add observation masks and uncertainty; do not fill every missing variable with
   one population default without exposing that uncertainty.
2. Learn a posterior belief update from repeated observations instead of relying
   only on a mechanistic K-store prior at the anchor.
3. Calibrate potassium loss and hyperosmolar injury duration on a larger cohort.
4. Validate on a larger patient-held-out cohort before any real-data fine-tuning.

The demo is sufficient to validate extraction and expose failure modes. It is not
sufficient to estimate causal intervention effects or train a clinically reliable
world model.
