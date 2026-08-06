# AKI Mechanism Candidate Findings

Date: 2026-06-27

This audit tests the next Chapter-B question after the validated AKI factual
router: can a narrow renal mechanism source help the slow renal accumulation
targets, especially creatinine and BUN, where the 6-hour factual router falls
back to persistence?

## Candidate

Implemented source:

- `aki_mechanism.py`
- router method name: `renal_mechanism`

Design boundary:

- predicts only creatinine and BUN;
- uses a transparent renal stress proxy from urine output, MAP, vasopressor
  context, nephrotoxin context, elevated creatinine/BUN, and RRT context;
- contains no fitted cohort coefficients;
- returns `NaN` for unsupported targets so the router can fall back;
- makes no causal, counterfactual, clinical, checkpoint, or active-rule claim.

## Result

The source was evaluated through the same nested AKI router discipline:

- discovery-only per-target source selection;
- out-of-fold discovery predictions;
- held-out patient evaluation over seven seeds;
- hospital-held-out evaluation;
- persistence fallback.

Aggregate comparison:

- baseline AKI router active-AKI median normalized delta: `-0.044771`
- AKI router with `renal_mechanism`: `-0.044744`
- baseline all-window median normalized delta: `-0.041323`
- AKI router with `renal_mechanism`: `-0.041212`

Selected methods across seven random patient splits:

- creatinine: persistence in 7/7 splits
- BUN: persistence in 6/7 splits, `renal_mechanism` in 1/7 split
- bicarbonate, MAP, potassium, sodium: `ridge_realfit` in 7/7 splits
- urine output: mostly persistence

Hospital-held-out evaluation did not select the mechanism for creatinine or BUN:

- BUN persistence MAE: `5.558835`
- BUN `renal_mechanism` MAE: `5.614559`
- creatinine persistence MAE: `0.405716`
- creatinine `renal_mechanism` MAE: `0.410854`

The one random split that selected `renal_mechanism` for BUN did not generalize
on held-out patients. Its BUN held-out MAE was worse than persistence, and the
held-out bootstrap delta was positive.

## Conclusion

Rejected. The simple renal mechanism candidate does not add robust value over
persistence or the validated AKI router.

This is still useful knowledge. It says the slow renal targets are not fixed by
a transparent 6-hour accumulation prior alone. To beat persistence on
creatinine/BUN, the next mechanism attempt likely needs one or more of:

- a longer horizon, such as 24-48 hours;
- a patient-specific renal reserve or GFR belief state;
- better RRT dose, ultrafiltration, and fluid-balance observability;
- a stricter creatinine/BUN generation model with validated external support.

Until then, the safe AKI router behavior remains unchanged: creatinine and BUN
fall back to persistence over the 6-hour factual window.

