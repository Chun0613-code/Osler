# Full eICU AKI Router Findings

Date: 2026-06-27

This instantiates Chapter B for acute kidney injury using the full PhysioNet
eICU Collaborative Research Database 2.0.  It reuses the validated DKA/sepsis
pattern: disease-specific observed-treatment transitions, discovery-only source
selection, held-out patient evaluation, hospital-held-out evaluation, and
persistence fallback.

## Cohort

AKI-like cohort definition:

- diagnosis text/code support for AKI / acute renal failure / acute tubular
  necrosis; or
- KDIGO-like creatinine rise:
  - creatinine increase >= 0.3 mg/dL within 48 hours; or
  - creatinine ratio >= 1.5 versus the previous 7-day minimum.

Aggregate support:

- AKI-like stays before evaluable filtering: 56,132
- Evaluable stays: 43,748
- Subjects: 37,119
- Hospitals: 204
- Six-hour transitions: 667,799
- Active-AKI transitions: 219,417

Targets:

- creatinine
- urine output
- potassium
- bicarbonate
- MAP
- BUN
- sodium

Observed treatment evidence:

- fluids
- vasopressors
- diuretics
- renal replacement therapy context
- nephrotoxin exposure context

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

## Results

Random patient splits, seven seeds:

- Active-AKI: 7/7 splits beat persistence, 7/7 significant
- Active-AKI median normalized delta vs persistence: -0.044771
- All windows: 7/7 splits beat persistence, 7/7 significant
- All-window median normalized delta vs persistence: -0.041323

Hospital-held-out split:

- Active-AKI normalized MAE:
  - persistence: 0.437640
  - target router: 0.408232
  - subject-bootstrap delta: -0.039494, 95% CI [-0.042375, -0.036591]
- All-window normalized MAE:
  - persistence: 0.495730
  - target router: 0.465980
  - subject-bootstrap delta: -0.035677, 95% CI [-0.037538, -0.033738]

Stable selected sources across seven random patient splits:

- creatinine -> persistence
- BUN -> persistence
- urine output -> persistence in 6/7 splits, population delta in 1/7
- bicarbonate -> ridge_realfit
- MAP -> ridge_realfit
- potassium -> ridge_realfit
- sodium -> ridge_realfit

## Interpretation

The AKI module validates the Chapter-B recipe on a second non-DKA disease.  It
also exposes an important physiology pattern:

- slow renal accumulation targets such as creatinine and BUN remain hard to
  beat over a 6-hour factual horizon and correctly fall back to persistence;
- electrolyte, acid-base, and perfusion-related targets show stable real-fit
  signal across patient and hospital splits.

This is consistent with the earlier DKA/sepsis lesson: the stable architecture
is not a single monolithic model, but a per-target router that lets unsupported
targets refuse to move.

## Mechanism Candidate Audit

After the validated factual router, a narrow renal mechanism source was tested
for the slow targets where the router fell back to persistence.

Candidate:

- file: `aki_mechanism.py`
- source name: `renal_mechanism`
- supported targets: creatinine and BUN only
- design: transparent renal stress proxy, no fitted cohort coefficients

Result:

- baseline active-AKI median normalized delta: -0.044771
- with `renal_mechanism`: -0.044744
- baseline all-window median normalized delta: -0.041323
- with `renal_mechanism`: -0.041212
- creatinine selected `persistence` in 7/7 random patient splits
- BUN selected `persistence` in 6/7 random patient splits and
  `renal_mechanism` in 1/7, but that held-out BUN result was worse than
  persistence
- hospital-held-out creatinine and BUN both selected `persistence`

Conclusion:

`renal_mechanism` is rejected as a source candidate. The negative result is
informative: a simple 6-hour renal accumulation prior does not beat persistence
on slow creatinine/BUN targets. The next mechanism attempt should require a
longer horizon, a patient-specific renal reserve/GFR belief state, or richer RRT
and fluid-balance observability.

## Long-Horizon Audit

The longer-horizon audit confirms that the 6-hour creatinine/BUN fallback is a
horizon effect, not simply a modeling failure.

At 24 hours:

- 530,265 transitions
- 178,294 active-AKI rows
- active-AKI median normalized delta: -0.067735
- 7/7 random patient splits beat persistence significantly
- creatinine and BUN selected `ridge_realfit` in 7/7 splits

At 48 hours:

- 397,897 transitions
- 135,870 active-AKI rows
- active-AKI median normalized delta: -0.106262
- 7/7 random patient splits beat persistence significantly
- creatinine and BUN selected `ridge_realfit` in 7/7 splits

Hospital-held-out evaluation also supports the horizon effect:

- 24h creatinine delta: -0.027636
- 24h BUN delta: -0.650428
- 48h creatinine delta: -0.032039
- 48h BUN delta: -0.756922

Interpretation:

Slow renal accumulation targets do not provide enough 6-hour factual signal to
beat persistence. At 24-48 hours, the signal appears and the router moves
creatinine/BUN from persistence to `ridge_realfit`.

## Renal Belief Audit

The first B-deep AKI personalization layer was tested after the long-horizon
audit.  It adds transparent renal reserve/GFR proxy features to `ridge_realfit`
and compares them against both the baseline and a capacity-matched placebo.

Result:

- 24h creatinine, BUN, and urine output pass in 7/7 patient splits and
  hospital-heldout evaluation
- 48h creatinine and BUN pass in 7/7 patient splits and hospital-heldout
  evaluation
- 48h urine output passes in 7/7 patient splits but is only directionally
  positive on hospital-heldout

Interpretation:

The belief layer is useful as a patient-specific personalization signal for
long-horizon renal forecasting.  It does not claim direct hidden-state accuracy,
causal treatment effect, clinical authority, or active-rule promotion.

## Boundary

- No row-level predictions are committed.
- No patient identifiers are included in aggregate reports.
- No causal claim is allowed.
- No counterfactual claim is allowed.
- No clinical claim is allowed.
- No checkpoint promotion is allowed.
