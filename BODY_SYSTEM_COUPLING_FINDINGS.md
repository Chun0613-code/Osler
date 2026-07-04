# Body-System Coupling Findings

Date: 2026-06-29

This is the first cross-system depth pass after Chapter-B body-system breadth
completion.  The breadth layer validated isolated factual routers for each
adult-ICU body-system bucket.  This audit asks a harder question: does upstream
organ-system state improve downstream factual prediction beyond a baseline
router that already sees the downstream cohort?

The tested candidate is deliberately simple and conservative:

- baseline: ridge model with the upstream organ-system feature block removed;
- candidate: baseline plus upstream organ-system features;
- placebo: baseline plus the same number of random features;
- pass rule: the candidate must significantly beat both baseline and placebo on
  held-out patients.

The output is aggregate-only.  Passing this gate would mean factual predictive
coupling signal, not causality, counterfactual validity, clinical authority, or
runtime treatment authority.

## Result

No cross-system edge passes the robust promotion boundary.  The static
feature-concatenation coupling layer therefore stays fail-closed.

| Edge | Rows | Subjects | Hospitals | Active Pass-Both Targets | Interpretation |
|---|---:|---:|---:|---|---|
| renal -> electrolyte / acid-base | 17,387 | 1,249 | 26 | none | rejected; hospital-only sodium signal is not reproducible across random patient splits |
| respiratory -> acid-base | 57,673 | 3,618 | 55 | none | rejected; hospital-only bicarbonate signal is not enough |
| endocrine -> electrolyte | 17,009 | 1,149 | 12 | potassium 1/7, sodium 1/7 | weak candidate only, not promoted |
| heme/coag -> perfusion | 12,379 | 907 | 12 | none | rejected |
| cardiovascular -> renal | 16,047 | 1,211 | 12 | none | rejected at 6h |
| hepatic -> coagulation / platelets | 17,730 | 1,096 | 88 | none | rejected |
| immune / inflammatory -> hemodynamics | 18,392 | 1,240 | 40 | none | rejected |

## What This Means

This does not mean the body systems are independent.  It means the first naive
coupling attempt is too shallow: adding same-window upstream features to a ridge
router does not reliably create human-like organ interaction.

That is an important boundary.  It prevents Osler-JEPA from pretending that
isolated body-system coverage is already a whole-body digital twin.

The pattern is now:

- isolated target routers work across the feasible adult-ICU body surface;
- long horizons help slow renal targets;
- renal predict-update belief state can add individual hidden-state signal;
- static cross-system feature coupling does not yet pass a robust gate.

## Next Coupling Layer

The next whole-body depth step should not be another static feature audit.  It
should be temporal and belief-based:

- renal reserve belief -> electrolyte / acid-base belief updates;
- perfusion shock burden -> renal reserve and lactate clearance updates;
- respiratory ventilation/oxygenation burden -> pH / bicarbonate update;
- heme oxygen-carrying capacity -> perfusion and lactate burden update;
- inflammatory burden -> hemodynamic instability and albumin/platelet update.

Each edge must keep the same gate: downstream observable improvement beyond
baseline and capacity-matched placebo, with patient-heldout and hospital-heldout
checks.  Until an edge passes that gate, it remains candidate-only and
fail-closed.

That temporal coupling audit has now been run and is documented in
`BODY_SYSTEM_TEMPORAL_COUPLING_FINDINGS.md`.  It produced weak candidate signals
but still no promoted edge: renal -> bicarbonate passed 2/7 active-window splits,
renal -> sodium passed 1/7, and immune -> MAP passed 1/7.  This confirms that
temporal predict-update state is a better direction than static concatenation,
but the current generic layer is not strong enough to leave fail-closed status.

## Safety Boundary

This audit grants no causal claim, counterfactual claim, clinical claim,
checkpoint promotion, runtime decision authority, or active symbolic-rule
promotion.  It is a research baseline for future whole-body coupling work.
