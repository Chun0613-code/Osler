# Organ-Local Patient-State Adapter Findings

## Question

Does a formal patient-specific state improve a population forecast because it
contains real longitudinal patient information, rather than merely adding
parameters?

## Architecture

The implementation unifies three existing components:

1. a causal neural predict-update history filter;
2. the validated renal reserve/creatinine belief state;
3. patient-weighted split conformal with causal-order online adaptation.

The adapter is attached only to the anatomical kidney token. It cannot write
directly to the functional system layer or whole-body token. The population
ridge remains the anchor and the learned correction is bounded and initialized
to zero.

The capacity-matched placebo uses the exact same 15,891-parameter model but
receives history and belief tensors from a different patient. Therefore a win
over placebo isolates patient-aligned longitudinal information from extra model
capacity.

## Cohort And Gate

- eICU AKI 24-hour transition cohort;
- 198,211 patient-balanced rows;
- 29,508 subjects, 203 hospitals, 8 care-unit types;
- target: `creatinine@24h`;
- seven patient-heldout seeds;
- hospital-heldout, care-unit-heldout, and late-disease-stage heldout;
- patient-level paired bootstrap against both the population anchor and the
  capacity-matched placebo;
- 90% adaptive conformal coverage gate of 0.87-0.93.

## Initial Result And Reproducibility Correction

The initial accelerator run passed all gates for `creatinine@24h`:

- patient-heldout point gate: 7/7;
- patient-heldout conformal gate: 7/7;
- hospital, care-unit, and late-stage point gates: 3/3;
- hospital, care-unit, and late-stage conformal gates: 3/3;
- median patient-equalized MAE change versus population anchor: `-0.063243`
  mg/dL;
- median change versus equal-capacity placebo: `-0.008958` mg/dL;
- external median changes: `-0.055599` versus population and `-0.008892`
  versus placebo.

The point improvement was stable. A later deterministic CPU reproduction,
however, measured late-stage coverage of `0.930244`, just above the frozen
`0.93` ceiling. The initial interval promotion is therefore withdrawn. No
threshold was changed, and `creatinine@24h` is not included in the precision
runtime registry.

## Renal-Target Expansion

The identical adapter and unchanged gates were then applied to the other two
predeclared renal targets.

### `BUN@24h`

`BUN@24h` passed every point-personalization gate:

- patient-heldout point and conformal gates: 7/7;
- hospital, care-unit, and late-stage point and conformal gates: 3/3;
- median patient-equalized MAE change versus population: `-0.795596` mg/dL;
- median change versus equal-capacity placebo: `-0.201375` mg/dL;
- external median changes: `-0.830828` versus population and `-0.246066`
  versus placebo.

Its matched-width audit passed 7/7 patient gates but only 2/3 external gates.
It remains a point-personalization result and is not authorized to emit a
narrower patient-specific interval.

### `urine_output@24h`

Urine output remains candidate-only. It beat the population anchor in every
patient and external split, but it significantly beat the patient-mismatched
placebo in only 2/7 patient splits. One patient conformal split and the
late-stage conformal split also exceeded the 0.93 upper coverage bound. The
model improvement is therefore not yet attributable to stable patient-specific
history, and no urine artifact was produced.

## Full Renal Matched 90% Interval-Width Sweep

All 18 `creatinine / BUN / urine_output × 1/3/6/12/24/48h` cells were compared
with their population anchors
using the same rows, calibration order, patient weighting, nominal 90% target,
and online adaptive-conformal policy. Narrowing was accepted only when both
intervals had empirical coverage in `0.87-0.93` and the patient-bootstrap upper
confidence bound for personalized-minus-population half-width was below zero.

Only three cells reproduced both better point prediction and narrower intervals
under every patient and external gate:

| Cell | Patient half-width delta | External half-width delta | Gates |
|---|---:|---:|---|
| `BUN@3h` | `-0.938355` mg/dL | `-0.723564` mg/dL | 7/7 + 3/3 |
| `BUN@12h` | `-1.688074` mg/dL | `-1.569791` mg/dL | 7/7 + 3/3 |
| `BUN@48h` | `-2.931549` mg/dL | `-2.054347` mg/dL | 7/7 + 3/3 |

Several rejected cells still narrowed intervals, but failed point accuracy or
one coverage domain. For example, urine output at 1/3/6h narrowed in all width
audits but did not beat both population and equal-capacity placebo point
forecasts. These cells remain on their validated population source rather than
emitting deceptively precise individualized values.

The deterministic CPU reproduction is the canonical result. Accelerator-only
promotions are not accepted.

## Artifacts

Each precision-promoted artifact contains:

- `model.pt`: neural patient-state weights;
- `population_anchor.joblib`: fitted population ridge;
- `preprocessing_and_conformal.npz`: train-only scaling and conformal quantiles;
- `metadata.json`: exact target, horizon, features, cohort counts, and safety
  boundary;
- `MANIFEST.sha256`: artifact integrity hashes.

The artifact contains no patient rows or identifiers.

The authorized artifacts are `patient_state_precision_bun3_v1/`,
`patient_state_precision_bun12_v1/`, and
`patient_state_precision_bun48_v1/`. Each has a verified `MANIFEST.sha256`.

`renal_patient_state_precision_registry_20260813.json` records all 18 cells and
fails closed for the 15 unsupported cells. The individual audit reports are
stored as `patient_state_interval_width_<target><horizon>_full_20260813.json`.

## Boundary

This validates factual precision personalization for `BUN@3h`, `BUN@12h`, and
`BUN@48h`; it does not prove
that the latent equals true renal reserve, does not identify treatment effects,
and has no clinical decision authority. Other targets require their own
population-versus-placebo and uncertainty gates.

## Canonical Update: 2026-08-14

The three-cell result above is retained as the original deterministic sweep,
but it is no longer the current runtime result. Follow-up retries without gate
relaxation validated 12 of 20 evaluated kidney/perfusion cells:

- creatinine at 3h, 12h, and 24h;
- BUN at 3h, 6h, 12h, 24h, and 48h;
- urine output at 6h and 12h;
- MAP at 3h and 6h.

The added interval policy separates early (0-24h), middle (24-72h), and late
(72h+) calibration residuals. Every promoted cell still requires 7/7 patient
splits, hospital, care-unit, and late-stage heldout gates, population and
equal-capacity placebo wins, 0.87-0.93 empirical coverage, and significant
matched interval narrowing. The current registry and deployable weights are in
`renal_patient_state_precision_registry_20260813.json` and the corresponding
`patient_state_precision_*_v1/` directories.
