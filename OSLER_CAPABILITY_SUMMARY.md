# Osler-JEPA — What This System Can Do

*A one-page, plain-language summary. Last updated 2026-07-04.*

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
  kidney**.
- **Individualization:** online hidden-state estimates now validate in five places:
  kidney reserve for creatinine/BUN, cardiovascular perfusion/shock state for
  heart-rate prediction, and electrolyte/acid-base state for potassium, bicarbonate,
  anion gap, and creatinine, plus respiratory/gas-exchange state for O2 saturation,
  respiratory rate, heart rate, and bicarbonate, and endocrine/glycemic-stress state
  for glucose, anion gap, bicarbonate, sodium, potassium, and MAP. Each improves
  downstream observable prediction beyond a strong baseline and a capacity-matched
  placebo.
- **Two care settings:** ICU **and** the pre-ICU emergency department.
- **Three population/data axes:** multi-hospital ICU data (eICU), independent ICU data
  (MIMIC-IV), and healthy/general-population cross-sectional data (NHANES).
- **Healthy-population nowcasting:** in NHANES, the same nowcast recipe validates
  **29 / 32** targets after sibling-variable leakage guards, including electrolytes,
  kidney, liver, CBC, blood pressure, body composition, lipids, and HbA1c.
- **Observed treatment context as factual input:** MIMIC-IV `inputevents`, `emar`,
  and `procedureevents` features improve 6h forecasts for glucose, potassium, and
  bicarbonate. This means the model can use treatment that was actually observed;
  it still does **not** claim what a different treatment would have done.
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
  Waveform manifest/contract now exists for ECG, arterial pressure, plethysmography,
  respiration, and related signals, but it has no prediction authority until a new
  signal-processing pipeline and held-out accuracy audit pass.
- **Causal frontier:** what-if treatment planning. Observational data cannot unlock this
  by itself; it requires external randomized-trial evidence (e.g. BioLINCC critical-care
  trials). That door is built, fail-closed, and waiting.

---

*Discipline note: across the entire development arc, every reported result survived
patient-held-out, and where possible hospital-held-out and cross-database, testing against
a "predict no change" baseline and a capacity-matched placebo. Negative and candidate
results are recorded as faithfully as the wins.*
