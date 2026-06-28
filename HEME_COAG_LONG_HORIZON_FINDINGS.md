# Heme/Coagulation Long-Horizon Findings

Date: 2026-06-28

This audit tests whether hematology/coagulation targets become more predictable
when the factual horizon is extended from 6h to 24h/48h.

The key design choice is apples-to-apples: both long-horizon cohorts are
restricted internally to the stay set from the existing 6h heme/coag cohort.
Only the horizon changes.  Reports include only aggregate counts and the source
filename; row-level stay identifiers remain local-only.

## Artifacts

Local-only row-level cohorts:

- `eicu_coagulopathy_heme_transitions_24h.parquet`
- `eicu_coagulopathy_heme_transitions_48h.parquet`

Committed aggregate reports:

- `eicu_coagulopathy_heme_transition_report_24h.json`
- `eicu_coagulopathy_heme_transition_report_48h.json`
- `eicu_coagulopathy_heme_target_router_24h.json`
- `eicu_coagulopathy_heme_target_router_48h.json`

Extractor support:

- `eicu_body_system_transition_extract.py` now supports
  `--restrict-stays-from`, allowing fixed-cohort horizon audits without writing
  patient identifiers into reports.

## Cohorts

| Horizon | Restricted Source | Restricted Stays | Evaluable Stays | Transitions | Active Transitions |
|---|---|---:|---:|---:|---:|
| 24h | `eicu_coagulopathy_heme_transitions_6h.parquet` | 1,382 | 1,044 | 12,379 | 6,266 |
| 48h | `eicu_coagulopathy_heme_transitions_6h.parquet` | 1,382 | 650 | 8,480 | 4,468 |

## Router Result

| Horizon | Active Median Delta | Random Splits | Hospital-Heldout Delta | Hospital-Heldout CI |
|---|---:|---:|---:|---|
| 6h | -0.083132 | 7/7 significant | -0.087116 | reported in body-system audit |
| 24h | -0.133450 | 7/7 significant | -0.141781 | [-0.201937, -0.091589] |
| 48h | -0.121074 | 7/7 significant | -0.021492 | [-0.034186, -0.009517] |

The long-horizon router still beats persistence robustly.  The strongest point
estimate is at 24h, not 48h.

## Target Selection

24h random patient splits:

- Hemoglobin: `ridge_realfit` 7/7
- Hematocrit: `ridge_realfit` 7/7
- MAP: `ridge_realfit` 7/7
- WBC: `population_delta` 4/7, persistence 3/7
- INR, PTT, fibrinogen, platelets: persistence 7/7

48h random patient splits:

- MAP: `ridge_realfit` 7/7
- Hemoglobin: `ridge_realfit` 4/7, persistence 3/7
- Hematocrit: `ridge_realfit` 2/7, persistence 5/7
- Bicarbonate: mixed weak signal
- INR, PTT, fibrinogen, platelets: persistence 7/7

Hospital-heldout is more conservative:

- 24h selects hemoglobin and MAP as active non-persistence sources.
- 48h selects no heme/coag lab as a stable non-persistence source.

## Interpretation

The 24h result validates the time-scale hypothesis for hemoglobin/hematocrit:
6h was not the only useful horizon, and 24h gives a stronger factual router
signal.

The hypothesis does not generalize to the coagulation cascade targets:

- INR remains persistence at 24h/48h.
- PTT remains persistence at 24h/48h.
- Fibrinogen is too sparse even at 24h/48h.
- Platelets remain persistence at both horizons.

So the heme/coag depth frontier is now sharper:

- Hemoglobin/hematocrit: validated factual targets, strongest around 24h.
- Coagulation reserve: not unlocked by horizon alone.
- Platelet/coagulation dynamics: need better blood-product subtype separation,
  transfusion dose normalization, anticoagulation context, and larger/specific
  observed windows before a belief state should be promoted.

This is consistent with the belief audit.  The bleeding/coagulation belief had
partial Hgb/Hct signal but did not pass robust 7/7 placebo-gated validation.
Longer horizons improve the factual router, not the hidden-state promotion
boundary.

## Boundary

- Factual observed-treatment forecasting only.
- No transfusion treatment-effect claim.
- No anticoagulation causal claim.
- No counterfactual claim.
- No clinical recommendation claim.
- No checkpoint promotion.
- No active symbolic-rule promotion.
- Row-level parquet cohorts remain local-only and ignored by git.
