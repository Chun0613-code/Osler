# Whole-Body Trajectory And Uncertainty Layer

Date: 2026-06-30

This artifact extends the observation layer from single-point factual forecasts
into a trajectory contract. The goal is not to make every variable move. The
goal is to expose a complete target x horizon map where each cell says whether
Osler-JEPA is allowed to move the value, must fall back to persistence, or must
remain missing.

## Horizon Grid

The canonical horizons are:

- 1h
- 3h
- 6h
- 12h
- 24h
- 48h

Current validation is sparse by design:

- Fast physiology validates primarily at 6h.
- Slow renal accumulation validates at 24-48h.
- Sparse or poorly observed variables remain fallback or missing.

This means a full trajectory object can exist without pretending that every
point is a learned prediction. The object is complete; the motion is selective.

## Current Move Rules

Validated 6h fast-physiology cells include:

- glucose
- anion gap
- potassium
- bicarbonate in bounded contexts
- pH in respiratory contexts
- MAP
- heart rate
- oxygen saturation
- respiratory rate
- lactate

Validated 24-48h slow-renal cells include:

- creatinine
- BUN
- urine output under the AKI long-horizon factual router

All other currently represented horizons for those targets fall back unless a
future held-out audit validates that target-horizon pair.

Sparse or observability-limited targets such as bilirubin, INR, PTT,
fibrinogen, and PaCO2 remain fallback/missing under this contract.

## Uncertainty Policy

Every future numeric interval must pass a calibration gate before it can be
shown as a confidence interval.

The default interval method is split conformal residual calibration per target,
horizon, and source. The primary target is 90% empirical coverage.

The gate is:

- patient-heldout coverage across seven split seeds;
- hospital-heldout coverage;
- observed 90% interval coverage inside the accepted range 0.87-0.93;
- no interval shown when the calibration artifact is missing.

Until that gate exists for a cell, the layer may expose:

- target;
- horizon;
- point estimate for validated cells;
- source;
- can-move flag;
- interval status = `needs_calibration_audit`;
- no numeric lower/upper interval.

This preserves the "humility reflex": the model can predict, but it must also
say when it has not earned a calibrated confidence band.

## Output Shape

The intended output object is:

```text
whole_body_state_forecast
  patient_time
  horizons: [1h, 3h, 6h, 12h, 24h, 48h]
  variables:
    target:
      horizon:
        point_estimate
        lower
        upper
        interval_level
        source
        status
        can_move
        interval_status
```

The important part is not that every cell contains a learned model output. The
important part is that every cell has an honest state.

## Boundary

This remains factual and observational only. It does not permit causal
treatment-effect claims, counterfactual treatment planning, clinical authority,
runtime treatment authority, checkpoint promotion, active-rule promotion, or a
complete-human-simulation claim.

The implementation contract lives in:

- `osler_jepa/observation_layer.py`
- `whole_body_rollout_uncertainty_contract.json`
- `test_observation_layer.py`
