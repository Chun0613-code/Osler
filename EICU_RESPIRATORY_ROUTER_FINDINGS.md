# eICU Respiratory Factual Router Findings

Date: 2026-06-28

This is the fourth Chapter-B factual-router disease module after DKA, sepsis,
and AKI. The cohort is deliberately narrower than "all pulmonary disease":
ARDS / ventilator-coded respiratory failure plus severe measured hypoxemia. The
broad full-scale cohort was too large for an interactive run, so this artifact
uses a deterministic bounded 5,000-stay extraction from the same full eICU
source. It is a validated engineering cohort, not yet the final full-scale
canonical respiratory result.

## Cohort

- Source: full PhysioNet eICU Collaborative Research Database 2.0
- Evaluable stays: 4,170
- Subjects: 3,618
- Hospitals: 55
- Six-hour transitions: 57,673
- Active respiratory transitions: 26,558

Targets:

- oxygen saturation
- respiratory rate
- heart rate
- MAP
- bicarbonate
- pH

Observed action evidence:

- ventilation
- bronchodilator
- systemic steroid
- antibiotics
- fluids
- vasopressor

All actions are factual observed-treatment evidence only. Medication orders and
coarse treatment rows are not treated as causal administrations.

## Router Result

The nested discovery-only router significantly beats persistence:

- active respiratory windows: 7/7 random patient splits beat persistence
- active respiratory windows: 7/7 random patient splits are significant
- median normalized delta vs persistence: -0.334511
- all windows: 7/7 random patient splits beat persistence
- all windows: 7/7 random patient splits are significant
- all-window median normalized delta vs persistence: -0.147597

Hospital-heldout evaluation also passes:

- active respiratory point delta vs persistence: -0.305452
- active respiratory 95% CI: [-0.342021, -0.26934]
- significant: true

Discovery source selection is stable:

- oxygen saturation: `ridge_realfit` in 7/7 splits
- respiratory rate: `ridge_realfit` in 7/7 splits
- heart rate: `ridge_realfit` in 7/7 splits
- MAP: `ridge_realfit` in 7/7 splits
- pH: `ridge_realfit` in 7/7 splits
- bicarbonate: `ridge_realfit` in 5/7 splits, persistence in 2/7 splits

## Interpretation

This extends the Chapter-B factual-router pattern into a respiratory domain.
The strongest signal is exactly where expected: dense, fast-moving oxygenation,
respiratory-rate, perfusion, and acid-base targets. The result supports the
cross-disease rule that per-target routers can beat persistence when the target
has enough physiological signal at the chosen horizon.

The bounded-cohort caveat matters. The full respiratory contract is larger and
slower than the sepsis/AKI adapters because respiratory diagnoses and hypoxemia
are common in ICU data. The extractor is full-scale-ready, but the canonical
full respiratory result should be run as a background job after the cohort
definition and table-scan optimizations are locked.

## Boundary

This artifact remains factual and observational.

- No causal treatment-effect claim is allowed.
- No counterfactual treatment claim is allowed.
- No clinical recommendation claim is allowed.
- No runtime action authority is granted.
- No checkpoint promotion is allowed.
- No active symbolic-rule promotion is allowed.
- No raw rows or patient identifiers are included in this report.

