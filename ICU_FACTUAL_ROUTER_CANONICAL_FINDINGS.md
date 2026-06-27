# ICU Factual Router Canonical Findings

Date: 2026-06-27

This file closes the first Chapter-B factual-router arc across DKA, sepsis, and
AKI.  The goal is not to claim one universal ICU world model.  The finding is
narrower and more useful: a per-disease, per-target factual router can reliably
beat persistence where the target has enough short-horizon physiological signal,
and it should refuse to move where persistence is the correct short-horizon
baseline.

## Validated Pattern

Across DKA, sepsis, and AKI, the stable architecture is:

```text
disease-specific cohort
  -> disease-specific observed-treatment transitions
  -> candidate sources
  -> discovery-only per-target source selection
  -> held-out patient and hospital/time evaluation
  -> persistence fallback when no source passes
```

The important part is the fallback.  The router is valuable because it knows
when not to predict change.

## Cross-Disease Rule

At a 6-hour factual horizon:

- fast-changing electrolyte, acid-base, perfusion, or treatment-responsive
  targets can beat persistence with a real-fit source;
- slow cumulative or sparsely measured targets often correctly fall back to
  persistence.

Observed examples:

- DKA: glucose/anion-gap signal emerges only when enough real DKA data and
  target gating are available; unsupported targets fall back.
- Sepsis: MAP, lactate, respiratory and oxygenation targets show real-fit
  signal; creatinine falls back.
- AKI: bicarbonate, MAP, potassium, and sodium show real-fit signal at 6h;
  creatinine and BUN fall back at 6h.

This is a physiological result, not just a modeling trick.  A 6-hour window is
long enough for fast ICU targets, but too short for many renal accumulation
targets.

## Horizon Result

The AKI long-horizon audit tests the obvious counterfactual to the 6-hour
failure mode: if creatinine/BUN are slow targets, does a longer factual horizon
make them predictable?

It does.

- At 6h, creatinine and BUN select persistence in 7/7 patient splits.
- At 24h, creatinine and BUN select `ridge_realfit` in 7/7 patient splits.
- At 48h, creatinine and BUN select `ridge_realfit` in 7/7 patient splits.

This means the next axis of progress is not "a smarter 6h mechanism" for slow
renal targets.  It is the time horizon, plus better observability for
intervening treatments and renal reserve.

## Boundary

This finding remains factual and observational.

- No causal treatment-effect claim is allowed.
- No clinical recommendation claim is allowed.
- No runtime action authority is granted.
- No active rule promotion is granted.
- No shared multi-disease latent space is implied.

The router is a research-grade advisory artifact: useful for factual state
forecasting and hypothesis generation, not for autonomous treatment choice.

