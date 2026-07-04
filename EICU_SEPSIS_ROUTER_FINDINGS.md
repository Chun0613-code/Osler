# Full eICU Sepsis Router Findings

Date: 2026-06-27

This instantiates Chapter B for sepsis using the full PhysioNet eICU
Collaborative Research Database 2.0.  It reuses the validated DKA pattern:
disease-specific observed-treatment transitions, discovery-only source
selection, held-out patient evaluation, and persistence fallback.

## Cohort

- Sepsis-like stays before evaluable filtering: 31,770
- Evaluable stays: 30,198
- Subjects: 25,175
- Hospitals: 204
- Six-hour transitions: 422,244
- Active-sepsis transitions: 252,177

Targets:

- MAP
- lactate
- creatinine
- urine output
- oxygen saturation
- heart rate
- respiratory rate
- observed vasopressor-support proxy

Observed treatment evidence:

- vasopressors
- fluids
- antibiotics
- ventilation
- renal replacement context

## Router

Candidate sources:

- `persistence`: current observed value
- `population_delta`: discovery-only mean target delta
- `ridge_realfit`: discovery-only linear residual model using observed state,
  observation ages, history, and factual treatment evidence

Selection rule:

Only a non-persistence source whose discovery subject-bootstrap CI beats
persistence can be selected.  Otherwise the target falls back to persistence.
Discovery source selection uses out-of-fold predictions.

Leakage guard:

The vasopressor-support proxy is defined from future vasopressor evidence, so
the router forbids all future `act_*` features for that target.  After this
guard, vasopressor support correctly falls back to persistence.

## Results

Random patient splits, seven seeds:

- Active-sepsis: 7/7 splits beat persistence, 7/7 significant
- Active-sepsis median normalized delta vs persistence: -0.043957
- All windows: 7/7 splits beat persistence, 7/7 significant
- All-window median normalized delta vs persistence: -0.027930

Hospital-held-out split:

- Active-sepsis normalized MAE:
  - persistence: 0.477462
  - target router: 0.450429
  - subject-bootstrap delta: -0.040571, 95% CI [-0.042658, -0.038214]
- All-window normalized MAE:
  - persistence: 0.466830
  - target router: 0.446845
  - subject-bootstrap delta: -0.024206, 95% CI [-0.025653, -0.022600]

Stable selected sources across seven random patient splits:

- creatinine -> persistence
- heart rate -> ridge_realfit
- lactate -> ridge_realfit
- MAP -> ridge_realfit
- oxygen saturation -> ridge_realfit
- respiratory rate -> ridge_realfit
- urine output -> ridge_realfit
- vasopressor-support proxy -> persistence

## Interpretation

This is the first full-size Chapter-B disease router.  It shows that the DKA
per-target factual-router recipe generalizes to sepsis when the disease module
is kept separate and evaluated with held-out patients.

The win is factual and observational only.  It does not identify antibiotic,
fluid, ventilation, vasopressor, or renal-replacement causal effects.

## Boundary

- No row-level predictions are committed.
- No patient identifiers are included in aggregate reports.
- No causal claim is allowed.
- No counterfactual claim is allowed.
- No clinical claim is allowed.
- No checkpoint promotion is allowed.
