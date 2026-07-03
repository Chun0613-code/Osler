# Electrolyte / Acid-Base Predict-Update Belief Findings

This audit tests whether patient-specific electrolyte / acid-base belief can
become the third validated personalization layer after AKI renal belief and
cardiovascular heart-rate belief.

The candidate is an online predict-update state built from each patient's own
sodium-water, potassium, acid-base, divalent-mineral, osmotic/renal, treatment,
and short-term kinetic trajectory.  The state is **not** a measured clinical
variable.  It can only be used if it improves downstream observable prediction
beyond both:

- ordinary factual `ridge_realfit`; and
- a capacity-matched placebo with the same number of random features.

## Full Cohort

- Dataset: full unbounded eICU electrolyte / acid-base / osmotic instability
  module.
- Rows: `259,944`
- Subjects: `17,727`
- Stays: `20,542`
- Hospitals: `200`
- Active electrolyte / acid-base rows: `192,258`
- Horizon: `6h`
- Extraction report: `eicu_electrolyte_acid_base_full_transition_report.json`
- Audit report: `eicu_electrolyte_belief_full_audit.json`

## Gate

- Baseline: ordinary factual `ridge_realfit`.
- Candidate: `ridge_realfit + electrolyte belief features`.
- Placebo: `ridge_realfit + same-number random features`.
- Required: significant held-out improvement over both baseline and placebo in
  all 7 patient splits, plus hospital-heldout support before validation.
- Output is aggregate-only; no row-level or patient identifiers are stored.

## Result

| Target | Patient Splits Passing Both | Median Delta vs Baseline | Median Delta vs Placebo | Hospital-Heldout | Status |
|---|---:|---:|---:|---|---|
| potassium | 7 / 7 | -0.0078 | -0.0082 | pass | validated |
| bicarbonate | 7 / 7 | -0.0385 | -0.0407 | pass | validated |
| anion_gap | 7 / 7 | -0.0382 | -0.0405 | pass | validated |
| creatinine | 7 / 7 | -0.0130 | -0.0145 | pass | validated |
| magnesium | 7 / 7 | -0.0040 | -0.0062 | fail | candidate-only |
| sodium | 6 / 7 | -0.0185 | -0.0235 | fail | candidate-only |
| chloride | 6 / 7 | -0.0306 | -0.0339 | pass | candidate-only |
| map | 4 / 7 | -0.0110 | -0.0147 | pass | candidate-only |
| calcium | 1 / 7 | -0.0017 | -0.0027 | fail | rejected |
| phosphate | 1 / 7 | -0.0067 | -0.0117 | pass | candidate-only |
| ionized_calcium | 0 / 7 | +0.0262 | -0.0083 | fail | rejected |
| serum_osmolality | 0 / 7 | +1.0220 | +0.0104 | fail | rejected |

Hospital-heldout details for validated targets:

| Target | Baseline MAE | Belief MAE | Placebo MAE | Delta vs Baseline 95% CI | Delta vs Placebo 95% CI |
|---|---:|---:|---:|---:|---:|
| potassium | 0.3835 | 0.3773 | 0.3842 | [-0.0075, -0.0027] | [-0.0082, -0.0033] |
| bicarbonate | 2.1097 | 2.0705 | 2.1132 | [-0.0520, -0.0277] | [-0.0541, -0.0290] |
| anion_gap | 2.2686 | 2.1529 | 2.2760 | [-0.1394, -0.1032] | [-0.1519, -0.1110] |
| creatinine | 0.3133 | 0.3013 | 0.3153 | [-0.0178, -0.0083] | [-0.0201, -0.0097] |

## Interpretation

Electrolyte belief validates as a third personalization component, but only for
the targets where short-term patient-specific kinetics are dense and
mechanistically linked:

- potassium;
- bicarbonate;
- anion gap;
- creatinine.

This expands the personalization ladder:

1. AKI renal belief: creatinine / BUN at 24-48h;
2. cardiovascular belief: heart rate at 6h;
3. electrolyte / acid-base belief: potassium, bicarbonate, anion gap, and
   creatinine at 6h.

The result is target-specific.  Sodium and chloride show near-miss behavior but
do not pass the full gate.  Magnesium passes all patient splits but fails
hospital-heldout, so it remains candidate-only.  Sparse/deep variables such as
ionized calcium and serum osmolality remain rejected.

## Boundary

Allowed:

- factual 6h prediction for validated electrolyte belief targets;
- capability reporting for candidate-only targets;
- future experiments on alternate horizons or richer treatment capture.

Not allowed:

- clinical, causal, or counterfactual claims;
- runtime treatment authority;
- checkpoint or active-rule promotion;
- claiming direct hidden-state accuracy;
- promoting sodium, chloride, magnesium, calcium, phosphate, MAP, ionized
  calcium, or serum osmolality from this audit.
