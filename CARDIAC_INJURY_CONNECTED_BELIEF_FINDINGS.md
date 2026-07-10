# Cardiac Injury Connected-Belief Findings

Date: 2026-07-10

This audit asks whether the expanded eight-belief whole-body layer adds
incremental forecast signal in cardiac injury / myocardial stress cohorts.

The candidate is the ordinary table/action ridge baseline plus all current
promoted belief families:

1. renal;
2. cardiovascular;
3. electrolyte / acid-base;
4. respiratory;
5. endocrine;
6. immune;
7. GI / nutrition;
8. musculoskeletal / rhabdomyolysis.

The comparator is unchanged:

- baseline: ordinary factual table/action ridge;
- candidate: baseline plus all eight belief families;
- placebo: baseline plus the same number of random features;
- validation: 7 patient-heldout splits plus hospital-heldout support;
- authority: factual observation/prediction only.

This does **not** grant causal, treatment-planning, clinical, runtime,
checkpoint-promotion, or active-rule authority.

## Cohorts

| Cohort | Rows | Subjects | Belief Features | Horizon |
|---|---:|---:|---:|---:|
| cardiac_injury_3h | 30,496 | 1,937 | 319 | 3h |
| cardiac_injury_12h | 26,271 | 1,794 | 319 | 12h |

Audit report: `eicu_all_model_belief_coupling_cardiac_injury_audit.json`

## Result

No cardiac injury target passed the full connected whole-body gate.

| Cohort | Target | Patient Splits Passing Both | Median Delta vs Baseline | Median Delta vs Placebo | Status |
|---|---|---:|---:|---:|---|
| cardiac_injury_3h | MAP | 3 / 7 | -0.1469 | -0.2257 | candidate-only |
| cardiac_injury_3h | heart rate | 0 / 7 | -0.0682 | -0.1480 | rejected |
| cardiac_injury_3h | troponin I | 0 / 7 | +1.7718 | -4.3459 | rejected |
| cardiac_injury_3h | potassium | 0 / 7 | +0.0178 | -0.0179 | rejected |
| cardiac_injury_3h | creatinine | 0 / 7 | +0.0227 | -0.1369 | rejected |
| cardiac_injury_12h | heart rate | 6 / 7 | -0.2494 | -0.3305 | near-miss, candidate-only |
| cardiac_injury_12h | MAP | 4 / 7 | -0.2183 | -0.3312 | candidate-only |
| cardiac_injury_12h | troponin I | 0 / 7 | +2.7035 | -20.5924 | rejected |
| cardiac_injury_12h | potassium | 0 / 7 | +0.0080 | -0.0279 | rejected |
| cardiac_injury_12h | creatinine | 0 / 7 | +0.0277 | -0.2346 | rejected |

BNP, CK-MB, CPK, and lactate were too sparse or unstable to produce a promoted
connected signal.

## Interpretation

This is a useful negative result.

The current eight-belief whole-body layer does not add robust myocardial-biomarker
forecasting.  Troponin, BNP, CK-MB, and CPK remain sparse specialty markers.
The only near-miss is 12h heart rate, which is clinically plausible but does not
meet the 7/7 patient-split gate.

This argues against adding a separate cardiac-injury belief right now.  The
available cardiac injury signal overlaps heavily with the already-validated
cardiovascular/perfusion and endocrine-stress beliefs, while the myocardial
biomarkers themselves remain data-limited.

## Boundary

Allowed:

- report cardiac injury connected-belief audit as candidate-only;
- use the result to avoid duplicating cardiovascular belief with a weak
  cardiac-biomarker wrapper;
- revisit if richer ECG/waveform, catheterization, echo, or serial biomarker
  data become first-class.

Not allowed:

- promoting cardiac injury as a new belief family;
- claiming troponin/BNP/CK-MB dynamics are predicted by the current whole-body
  belief layer;
- causal, counterfactual, clinical, or runtime claims.
