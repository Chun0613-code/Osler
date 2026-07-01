# Whole-Body Same-Time Nowcasting Findings

Date: 2026-06-30

This artifact records the same-time nowcasting pass for the whole-body
observation layer.  Nowcasting is current-state completion, not future
forecasting: it estimates a currently unmeasured `target_t` from other
same-time and historical features available at the same anchor.

It does not claim causal treatment effects, counterfactual treatment response,
clinical authority, runtime treatment authority, checkpoint promotion, or active
symbolic-rule promotion.

## Why This Audit Exists

The trajectory layer already learned a strict rule: sparse or slow variables
often should not be moved into the future.  That is correct, but it leaves a
different question open:

> If a sparse variable is not measured right now, can the observed body state
> estimate its current value?

That is a different task from future prediction.  The audit therefore tests a
separate object: same-time imputation under held-out gates.

## Gate

For each module and target, the audit uses:

- same-time and historical numeric features only;
- no future `*_tp*` columns;
- no `act_*` future-window action columns;
- no active flags, because those can be derived from target values;
- no target value or target age as features;
- discovery-only out-of-fold source selection;
- median baseline and a capacity-matched placebo ridge model;
- seven patient split seeds;
- hospital-heldout validation.

A nowcast is validated only when it is selected in all seven patient splits,
beats the median baseline in all seven held-out splits, beats the
capacity-matched placebo in all seven held-out splits, and also passes the
hospital-heldout gate.

## Headline Result

The full audit covers 15 module cohorts and validates 41 module-target
same-time nowcasts.

| Module | Validated nowcast targets |
|---|---|
| sepsis | MAP, creatinine, heart rate, respiratory rate, vasopressor requirement |
| AKI | creatinine, potassium, bicarbonate, MAP, BUN, sodium |
| respiratory | respiratory rate, MAP, bicarbonate |
| integumentary / skin / wound | hemoglobin, hematocrit, albumin |
| toxic / metabolic | anion gap, bicarbonate |
| electrolyte / acid-base | sodium, potassium, chloride, bicarbonate, anion gap, calcium, phosphate, creatinine |
| endocrine stress | anion gap, bicarbonate, potassium |
| GI / pancreatic / nutrition | albumin, total protein, calcium, MAP |
| cardiac injury | potassium |
| hepatic failure | direct bilirubin, platelets, bicarbonate, creatinine |
| heme / coagulation | hemoglobin, hematocrit |
| musculoskeletal / rhabdo | none |
| immune / inflammatory | none |
| cardiovascular instability | none |
| acute neuro | none |

## What It Means

Nowcasting rescues a real part of the "missing current body" problem.  It helps
when a sparse current variable is tightly constrained by other current
physiology:

- renal chemistry: creatinine and BUN;
- electrolyte/acid-base panels: sodium, potassium, chloride, bicarbonate,
  anion gap, calcium, phosphate;
- heme state: hemoglobin and hematocrit;
- nutrition/protein state: albumin and total protein;
- hepatic proxy state: direct bilirubin and platelets.

This is not the same as proving those variables can be moved forward in time.
For many of them, forward prediction remains persistence fallback unless their
target-horizon gate separately passes.

## What It Does Not Rescue

The audit does not validate a universal current-state imputer.  Several modules
remain unvalidated under the same strict gate:

- musculoskeletal/rhabdomyolysis;
- immune/inflammatory proxy;
- cardiovascular instability as a separate nowcast module;
- acute neurologic proxy.

The failure mode is consistent with the rest of the project: if the EHR does not
contain a reliable observable counterpart for a deep state, the model must leave
it missing rather than fabricate a confident value.

## Product Contract

The observation layer now has three distinct non-causal objects:

1. `nowcast`: estimate the current value of validated targets when the target is
   not measured at the anchor.
2. `forecast`: predict validated future target-horizon cells.
3. `interval`: show calibrated uncertainty only for cells with split-conformal
   coverage validation.

These objects must stay separate.  A validated nowcast target is not
automatically a validated future forecast target, and neither one authorizes a
treatment decision.

## Safety Boundary

- Row-level outputs are not committed.
- Patient identifiers are not included in this artifact.
- Same-time imputation is allowed only for validated module-target pairs.
- Future prediction claims require separate target-horizon validation.
- Causal, counterfactual, clinical, runtime-treatment, checkpoint-promotion,
  and active-rule-promotion claims remain closed.

