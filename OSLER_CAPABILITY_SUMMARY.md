# Osler-JEPA — What This System Can Do

*A one-page, plain-language summary. Last updated 2026-08-03.*

## In one sentence

Osler-JEPA is a **validated, whole-body physiology observation-and-prediction system** for
critically ill patients. Given a patient's observed body state, it estimates what is
currently unmeasured, predicts near-future physiology where the data support it, attaches
honest confidence, and — importantly — **refuses to move any value it has not earned the
right to move.**

It is a disciplined bedside observer, **not** a treatment planner.

## The core result

The same method that failed on a tiny confounded cohort succeeded, robustly and
statistically significantly, once given enough real, treatment-resolved data — first on
**755 real ICU diabetic-ketoacidosis patients** (MIMIC-IV), then across **204 hospitals**
for sepsis and acute kidney injury (eICU), then again on an **independent second database**
(MIMIC-IV, ~90,000 ICU stays), and finally in a **healthy-population cross-sectional
setting** (NHANES 2017-2018).

The lesson, proven rather than asserted: **the bottleneck was the data, not the method** —
and the whole system was built without ever faking a small-sample win.

## What it can do (validated on held-out patients or participants; hospital-held-out where applicable)

- **Forecast near-future physiology** across four disease modules (DKA, sepsis, AKI,
  respiratory failure) and a whole-body surface of **12–14 organ-system modules**.
- **Predict over multiple time horizons** (1h → 48h). Fast perfusion/vital signs move
  first; slow renal chemistry becomes predictable only at 24–48h.
- **Three kinds of estimate, kept strictly separate:**
  - *nowcast* — estimate a currently-unmeasured value from the rest of the observed body;
  - *forecast* — predict a validated future value;
  - *interval* — show a calibrated 90% confidence band, only where coverage was verified.
- **Cross-organ coupling:** e.g. cardiovascular perfusion → kidney (24–48h), sepsis burden
  → blood pressure (6h), and a validated three-organ chain **sepsis → blood pressure →
  kidney**. Cached all-model belief reruns connect personalized physiology
  states at scale; they validate heart-rate/MAP/oxygenation/respiratory-rate signal in
  full cardiovascular cohorts, broad sepsis physiology, AKI 24–48h kidney/electrolyte
  targets, and many dense MIMIC-IV observation targets. A stricter explicit
  whole-body latent layer adds incremental signal beyond the all-belief baseline for
  cross-system targets such as lactate, BUN, anion gap, PaCO2, and phosphate.
  A still stricter graph-propagated coupling layer adds only a small validated
  renal/systemic increment for MIMIC-IV 6h creatinine; broader graph gains remain
  candidate-only, which keeps the coupling claims bounded.
- **Individualization:** online hidden-state estimates now validate in seven belief families:
  kidney reserve for creatinine/BUN, cardiovascular perfusion/shock state for
  heart-rate prediction, and electrolyte/acid-base state for potassium, bicarbonate,
  anion gap, and creatinine, plus respiratory/gas-exchange state for O2 saturation,
  respiratory rate, heart rate, and bicarbonate, and endocrine/glycemic-stress state
  for glucose, anion gap, bicarbonate, sodium, potassium, and MAP, plus
  immune/inflammatory host-response state for sepsis MAP, creatinine, O2 saturation,
  heart rate, respiratory rate, and vasopressor requirement, plus a bounded
  GI/nutrition/gut-perfusion belief for MAP, and a bounded musculoskeletal /
  rhabdomyolysis perfusion-stress belief for MAP. Each improves
  downstream observable prediction beyond a strong baseline and a capacity-matched
  placebo. In the cached full rerun, the cardiovascular heart-rate near-miss becomes
  a full validation at scale, and the connected belief layer generalizes to MIMIC-IV.
  The connected 8-belief audit preserves musculoskeletal MAP at 3h; its 12h
  isolated signal is candidate-only after whole-body connection.
  Muscle-to-kidney and muscle-to-electrolyte coupling remain candidate-only.
  A connected cardiac-injury scan was also run; it validates no target, with
  12h heart rate as a 6/7 near-miss and myocardial biomarkers still fallback.
- **Two care settings:** ICU **and** the pre-ICU emergency department.
- **Three population/data axes:** multi-hospital ICU data (eICU), independent ICU data
  (MIMIC-IV), and healthy/general-population cross-sectional data (NHANES).
- **Healthy-population nowcasting:** in NHANES, the same nowcast recipe validates
  **29 / 32** targets after sibling-variable leakage guards, including electrolytes,
  kidney, liver, CBC, blood pressure, body composition, lipids, and HbA1c.
- **Observed treatment context as factual input:** strict as-of MIMIC-IV
  `inputevents`, `emar`, `emar_detail`, and `procedureevents` features improve
  6h forecasts for glucose, MAP, and bicarbonate across the required
  care-unit/time checks. Potassium remains fallback because its care-unit gate
  did not pass. This means the model can use treatment that was actually
  observed; it still does **not** claim what a different treatment would have
  done.
- An observed-evidence layer drawn from clinical notes (imaging findings, GCS, delirium
  status).

## The one law that governs everything

Confirmed independently across databases and at 100,000-patient scale:

> **Dense, mechanistically-linked physiology can be predicted. Sparse, deep, or
> text-derived variables can only be *observed* — and this is a limit of the data, not of
> scale or effort.**

This is why the system is broad without pretending to be omniscient. For any variable it
cannot support, it says so and falls back — the humility is built in.

One especially clean example is **glucose**. In ICU/DKA, glucose becomes predictable
because illness, treatment, and repeated measurements create a trajectory. In healthy
NHANES cross-section data, glucose fails the nowcast gate because a single glucose value
depends heavily on unobserved timing such as meals and fasting state. The same variable can
be predictable in one physiologic context and correctly unknowable in another.

## What it explicitly does NOT do

These boundaries are enforced in code and never crossed:

- **No causal or treatment-effect claims.** It forecasts the trajectory under the treatment
  that *was* given; it does not claim what a *different* treatment would do.
- **No counterfactual / what-if treatment planning.**
- **No clinical recommendation or runtime treatment authority.**
- **No claim to be a complete human simulator.**

Everything above is **research-grade and observational.** Nothing has been clinically
validated or deployed to influence care.

## Remaining Frontiers

- **Factual accuracy frontier:** high-frequency waveforms. A candidate MIMIC-IV
  Waveform manifest, raw feature reader, and first held-out 6h precision audit now
  exist for ECG, arterial pressure, plethysmography, respiration, and related
  signals. The first audit validated **0 targets**, so waveform remains
  candidate-only; this specific 60-second feature layer did not improve the
  table-plus-treatment baseline.
- **Control-loop frontier:** sparse specialty loops. A new control-loop audit maps
  validated versus missing physiologic feedback loops. Mineral-bone, coagulation,
  liver detoxification, thermoregulation, neuro-arousal, autonomic balance, and
  cardiac-biomarker loops are partly observed but not yet validated as promoted
  predict-update belief families.
- **Causal frontier:** what-if treatment planning. Observational data cannot unlock this
  by itself; it requires external randomized-trial evidence (e.g. BioLINCC critical-care
  trials). That door is built, fail-closed, and waiting.

## Whole-Body Integration Update (2026-07-24)

- A canonical causal event ledger now separates measurement, medication,
  infusion, and procedure events while preserving actual timestamps, units,
  source provenance, measurement age, and strictly pre-anchor treatment
  context.
- The joint JEPA now exposes learned organ tokens alongside the shared body
  latent, neural predict-update belief state, target-specific future latents,
  and measurement-time head.
- `whole_body_state_forecast.v2` returns observed/current state, factual future
  point, actual target-specific horizon, uncertainty status, organ source,
  treatment context, and an explicit fallback reason for every cell.
- Promotion now requires the existing patient/hospital/conformal/cross-system
  gate plus direction, delta correlation, comparison with the current
  validated router, and learned-latent SVD stability.
- The latent SVD audit uses rotation-invariant principal-subspace overlap.
  A bounded two-module smoke had active latent rank in every split and passed
  this corrected stability check in 7/7 splits. It still validated no factual
  target cell, so the model remains target-gated and fail-closed.

## Final Joint-JEPA Gate Update (2026-07-29)

- The canonical event-ledger full gate ran on 566,376 rows from 3,290 patients
  and 12 hospitals, with actual measurement time, measurement age, quality,
  multiscale history, and strictly pre-anchor treatment context.
- The learned whole-body latent did not collapse. In the final independent
  seven-seed SVD audit, its median effective rank was about 29 and its median
  held-out mode cosine was 0.946; all seven splits passed latent stability.
- The joint model beat persistence on 18 evaluated target-horizon cells in all
  seven patient splits. It did not, however, beat the stronger existing
  target-specific router across every patient, hospital, care-unit, time,
  uncertainty, and cross-system gate.
- No target-horizon cell was promoted. The closest was `AST@1h`, which passed
  every point-performance gate but had one genuinely under-covered patient
  split. It remains fallback.
- Own-target ablation showed that broad whole-body context added a stable
  7/7 increment only for `bicarbonate@24h` in the bounded scoring audit.
  Therefore shared physiology is present, but its incremental predictive value
  is sparse and target-specific rather than universally useful.

The practical conclusion is precise: the joint JEPA is a real, active
whole-body representation and a useful research model, but it has not earned
permission to replace the validated factual router. Runtime continues to use
the strongest validated source per target and falls back to persistence or
missing whenever evidence is incomplete.

Verification on 2026-07-29: the complete `medical_jepa` unit suite passed
304/304 tests.

## Teacher-Anchored Accuracy Experiment (2026-07-30)

A new candidate stopped asking one joint model to replace the validated
target-specific router. It kept that router as a fixed teacher and learned
only a zero-initialized, bounded JEPA residual using a shared slow body latent
and target-specific fast future latents.

On 63,228 rows from 480 patients and 12 hospitals, the hierarchical candidate
beat the teacher and persistence in 15/36 target-horizon cells across all
seven patient splits. Three cells (`heart_rate@6h`, `heart_rate@12h`, and
`map@3h`) also passed hospital, care-unit, forward-time, module-balanced,
direction, delta-correlation, patient-bootstrap, and external conformal gates.
The learned latent stayed active and stable (effective rank 12.65-18.34;
principal-subspace overlap 0.795-0.981).

Explicit dynamic cross-organ routing did not beat its no-route ablation in
7/7 splits for any cell. PCGrad removed all measured negative gradient
conflicts but did not improve final accuracy. The reliable gain therefore
came from teacher anchoring plus target-specific fast/slow JEPA dynamics, not
from a newly validated organ-routing graph.

The first three point forecasts remained candidate-only because fixed nominal
90% conformal coverage was slightly over-conservative. Follow-up calibration
experiments rejected an unstable adaptive split and an unstable normalized
error-scale model. A fixed nominal 89% setting was then selected using two
development split batches and frozen before a third, previously unused
seven-seed patient audit.

In that final audit, `heart_rate@6h` passed point accuracy, direction,
delta-correlation, patient bootstrap, SVD, and conformal coverage in 7/7
splits. It also passed held-out hospital, care-unit, and forward-time gates.
The v2 registry now validates exactly this one target-horizon cell for
target-gated factual research runtime. `heart_rate@12h`, `map@3h`, and every
other unvalidated cell still use the existing router or persistence fallback.
This is not a whole-checkpoint or cross-organ-attribution promotion.

Canonical findings:
`TEACHER_ANCHORED_JOINT_JEPA_FINDINGS_20260730.md`.

Verification on 2026-07-30: the complete `medical_jepa` unit suite passed
311/311 tests.

The calibrated registry is
`teacher_anchored_joint_jepa_validated_registry_20260731.json`.

Verification on 2026-07-31: the complete `medical_jepa` unit suite passed
315/315 tests.

## Full-Cohort Joint-JEPA Update (2026-07-31)

The teacher-anchored experiment was rerun on the full canonical eICU cohort,
not the earlier 500-stay sample:

- 566,376 joint rows;
- 3,290 patients;
- 12 hospitals;
- six targets across 1/3/6/12/24/48h;
- seven independent patient splits;
- patient-equalized target/horizon conformal calibration.

The whole-body latent passed SVD stability in 7/7 splits. Nine cells passed
the internal teacher, persistence, and patient-cluster conformal gates. Three
also passed held-out hospital, care-unit, forward-time, module-balanced,
patient-bootstrap, direction, delta-correlation, and uncertainty gates:

```text
creatinine@6h
creatinine@12h
heart_rate@3h
```

These cells may use the bounded JEPA residual in factual research runtime.
All other cells still use the validated target router or persistence. This is
evidence that the joined whole-body representation can improve selected
targets; it is not a universal whole-body, cross-organ-attribution, causal, or
clinical promotion.

The remaining scientific limitation is now narrower: shared physiology is
useful but target- and timescale-specific. Dynamic organ routing has still not
passed its own no-route ablation, and sparse variables remain limited by
future-label support. The appropriate architecture remains shared context plus
target-specific dynamics, strict per-cell uncertainty gates, and fallback.

Verification after the full-cohort update: the complete `medical_jepa` unit
suite passed 318/318 tests.

---

*Discipline note: across the entire development arc, every reported result survived
patient-held-out, and where possible hospital-held-out and cross-database, testing against
a "predict no change" baseline and a capacity-matched placebo. Negative and candidate
results are recorded as faithfully as the wins.*
# 2026-08-01 whole-body sparse-expert update

Target/horizon sparse experts were tested on 1,000 eICU stays and seven
patient-held-out seeds. They consistently improved point forecasts for
glucose at 1h/3h and heart rate at 3h/6h/12h versus the factual teacher,
persistence, and a matched no-expert ablation. Fixed, adaptive, and normalized
post-hoc intervals were insufficient. A two-stage target-specific
distributional scale head preserved point predictions and calibrated
patient-dependent uncertainty. `glucose@3h` subsequently passed all seven
patient gates plus hospital, care-unit, forward-time, module-balanced,
conformal, direction/delta, no-expert, and SVD gates. It is validated as a
target-gated factual research source; all other cells retain their previous
source or persistence fallback. MIMIC-IV patient-cluster irregular-event
intervals remain validated for INR at 6--24h and 24--72h. See
`SPARSE_EXPERT_AND_IRREGULAR_INTERVAL_FINDINGS_20260801.md`.

## Two-Level All-Target JEPA Update (2026-08-01)

The successful sparse-distributional method was applied jointly to every
canonical target, rather than only the six development targets. The model is
one connected hierarchy: a shared whole-body latent, 11 disjoint canonical
organ latent tokens, and 53 target-specific sparse experts, residual heads,
and uncertainty scales.

On 1,000 eICU stays, 922 patients, 12 hospitals, six horizons, and seven
patient-held-out seeds, all 53 targets entered the same model. Forty-seven had
enough measured future-label support for at least one formal cell, producing
192 evaluated target-horizon cells. Eleven cells passed all seven point
forecast splits. Glucose at 1h, 3h, and 6h also passed all seven
patient-equal distributional conformal gates.

External retraining validated glucose@1h and glucose@3h across hospital,
care-unit, and forward-time holdouts, including module-balanced and matched
no-expert tests. glucose@6h failed the care-unit accuracy and conformal
requirements and therefore remains on fallback.

This supports the two-level design and shows that joint all-target training
can add a validated horizon without erasing the earlier one. It does not show
universal improvement: sparse supervision and target-specific uncertainty
remain the limiting factors for most cells. See
`TWO_LEVEL_ALL_TARGET_JEPA_FINDINGS_20260801.md`.

## Interval-Normalized Urine Output Update (2026-08-02)

An audit found that legacy eICU urine extraction mixed irregular interval
volumes, counts, occurrences, and an arbitrary fixed-time conversion. Urine
events were rebuilt as volume divided by their actual preceding collection
interval, with explicit `mL/hour`, interval, source, and quality provenance.

With zero patient overlap between development and formal holdout,
`urine_output@3h` passed 7/7 patient gates and the hospital, care-unit,
forward-time, no-expert, module-balanced, conformal, direction/delta, and SVD
gates. It is now a target-gated factual research source under the dedicated
`joint_asof_whole_body_state_interval_urine.v1` contract. Runtime rejects old
raw-volume input or a unit mismatch and falls back to persistence.

This adds one validated non-glucose whole-body JEPA cell and demonstrates that
measurement semantics can be the limiting factor even when model architecture
and raw data volume are adequate. It remains non-causal, non-clinical, and
requires fresh external confirmation because the conformal allocation method
was refined during this research cycle. See
`INTERVAL_NORMALIZED_URINE_OUTPUT_FINDINGS_20260802.md`.

## eICU Treatment-Time Alignment Update (2026-08-02)

An event-ledger audit found that eICU measurements and treatment events used
synthetic origins 130 years apart. Consequently, prior canonical examples had
zero usable `hist_*` treatment columns even when treatment context was
requested. All eICU event adapters now share one origin and fail closed on
cross-kind timestamp misalignment.

The rebuilt examples preserve 5.40 million events and now expose 96 strictly
pre-anchor treatment-context columns. A seven-seed patient-heldout run plus
hospital, care-unit, and forward-time evaluation validated
`urine_output@1h` under the dedicated aligned-treatment contract. No other new
cell passed every gate. The earlier measurement-only `urine_output@3h` cell
remains separate and valid only under its original contract.

This fixes a real extraction defect and adds one treatment-aware factual
research cell; it does not establish causal treatment effects or a universal
whole-body improvement. See
`EICU_EVENT_TIME_ALIGNMENT_FINDINGS_20260802.md`.

## Three Additional Joint-JEPA Runtime Cells (2026-08-03)

The runtime coverage comparison now uses the union of target-horizon cells,
not the number of model sources. Relative to catalog v2, catalog v4 grows from
30 to 33 cells and removes none. The three genuinely new cells are:

```text
glucose@1h
respiratory_rate@3h
respiratory_rate@24h
```

All three passed seven patient splits plus hospital, care-unit, forward-time,
module-balanced, bootstrap, SVD, and conformal gates. The respiratory models
also passed the sparse-expert ablation gate. A patient-grouped crossfit
adaptive conformal estimator fixed the remaining respiratory 24h care-unit
interval instability without reading forecast test labels or relaxing the
0.87-0.93 coverage requirement.

Exact input-contract and unit matching remain mandatory. Newly materialized
JEPA sources for `glucose@3h` and `glucose@12h` remain useful alternatives,
but those cells already existed in the shared router and are not counted as
new system coverage. All cells remain factual research forecasts with
persistence fallback and no causal or clinical authority. See
`THREE_ADDITIONAL_JEPA_CELLS_FINDINGS_20260803.md`.

Verification on 2026-08-03: the complete `medical_jepa` unit suite passed
357/357 tests.

## Target-Specific Rescue Update (2026-08-05)

Three alternatives to further shared-latent enlargement were evaluated. Using
the full 1,799-subject aligned eICU cohort rescued `heart_rate@3h` and
`heart_rate@6h`; the earlier care-unit failure was caused by the 1,000-stay
sampling cap. Reframing oxygenation as a discovery-calibrated future
hypoxemia probability validated `hypoxemia_event@1h` and
`hypoxemia_event@3h` across 7/7 patient splits plus hospital, care-unit, and
forward-time holdouts. This is an event probability, not a continuous O2sat
value claim.

A robust change-hurdle model also validated `creatinine@24h`. Conditional-
median deltas fixed outlier sensitivity, while patient-grouped normalized
conformal intervals using current value, predicted change, measurement age,
and hours since stay onset passed the 0.87-0.93 coverage requirement in every
patient and external split. Catalog v7 records all five new cells under exact
output contracts with persistence fallback and no causal or clinical
authority. See `TARGET_SPECIFIC_RESCUE_FINDINGS_20260805.md`.

## Target-Specific Grid Expansion (2026-08-05)

The validated change-hurdle formulation was applied to the remaining
stable-plus-change targets without relaxing any patient, external-domain,
bootstrap, comparator, or conformal gate. Eight additional cells passed:

```text
platelets@3h
hemoglobin@3h, @6h, @12h
hematocrit@3h, @6h
bicarbonate@24h
bun@24h
```

Hemoglobin and hematocrit passed with their near-algebraic sibling excluded
from all value/history features. Runtime catalog v8 therefore contains 45
unique validated target-horizon cells, up from 37 in v7. A subsequent
domain-heldout conformal audit added `platelets@1h`, bringing catalog v9 to 46
unique cells. The same interval method did not rescue anion gap, BUN, or
calcium, so those cells remain fallback-only. Many other cells
remained fail-closed: some lacked stable point gains, while a distinct group
won on point error but failed only cross-domain interval calibration. See
`TARGET_SPECIFIC_GRID_EXPANSION_FINDINGS_20260805.md`.

Runtime verification on 2026-08-06 materialized all nine new cells across six
source registries. The serving adapter now preserves the actual source model
and task type, and releases intervals only when patient, hospital, care-unit,
and forward-time conformal evidence are all present. All nine cells passed a
direct runtime invocation test; the complete suite passed 374/374 tests.
