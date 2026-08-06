# Target-Specific Grid Expansion Findings (2026-08-05)

## Scope

The target-specific event/change-hurdle method was expanded beyond the three
development targets. Each target retained measurement-pure anchor-time input,
patient-disjoint discovery/calibration/test partitions, seven patient seeds,
hospital, care-unit, and forward-time holdouts, paired patient bootstrap
comparisons against both persistence and plain ridge, and 0.87-0.93 conformal
coverage. No gate was relaxed.

## Newly validated cells

Eight additional target-horizon cells passed every required gate:

| Target | Validated horizons | Runtime output |
|---|---|---|
| platelets | 3h | continuous value with 10 K/uL change hurdle |
| hemoglobin | 3h, 6h, 12h | continuous value with 0.5 g/dL change hurdle |
| hematocrit | 3h, 6h | continuous value with 1.5 percentage-point change hurdle |
| bicarbonate | 24h | continuous value with 1 mmol/L change hurdle |
| BUN | 24h | continuous value with 2 mg/dL change hurdle |

Hemoglobin and hematocrit were evaluated with their sibling variable removed
from all current, lag, and slope features. Their gains therefore do not rely
on the near-algebraic Hgb-Hct relation.

The fail-closed runtime catalog initially grew from 37 to 45 unique validated cells.
Each new source preserves its threshold, units, adaptive normalized conformal
contract, exact input contract, persistence fallback, and factual-only status.

## Negative and near-miss results

The expanded audit also rejected unsupported cells. No registry was created
for them.

- The calibrated hypoxemia event remained valid only at 1-3h; 6-48h failed.
- Creatinine 48h, hematocrit 12h, WBC 6-24h, and several chemistry cells were
  patient-split near misses.
- Calcium, magnesium, sodium, anion gap, BUN 12h, and platelets 1h frequently
  passed point forecasting but failed one external conformal interval gate.
- Albumin, total protein, chloride, phosphate, and temperature showed broader
  point or temporal instability and remain fallback-only.

## What this changes

The former `candidate_only` group did not have one common failure cause.
Target-specific output geometry rescued stable-plus-change mixtures that a
single unconditional delta regression blurred together. It was especially
useful for blood counts and for slower renal/acid-base targets at 24h.

The remaining near misses now divide into two distinct engineering questions:

1. **point-forecast instability**: one or more patient/domain splits do not
   beat persistence;
2. **domain-conditional uncertainty instability**: the point forecast wins,
   but a held-out hospital, care unit, or later-time cohort misses the required
   interval coverage.

The second group should next test domain-robust or regime-conditional conformal
calibration without changing the point model. The first group should remain on
persistence unless a target-specific temporal/event formulation is supported
by physiology and independently validated.

## Domain-aware uncertainty follow-up

A second audit changed only interval calibration for cells whose point model
already passed. Calibration rows were partitioned by hospital, care unit, or
chronological domain; nominal coverage was selected by leaving these domains
out inside calibration. Forecast test labels were never used.

`platelets@1h` moved from a near miss to full validation:

- patient splits remained 7/7;
- hospital, care-unit, and forward-time splits became 3/3;
- the point model and 10 K/uL hurdle were unchanged.

The same method did not rescue `anion_gap@12h`, `bun@12h`, or
`calcium@3h/6h`; these remain fallback-only. A patient-equal-only conformal
variant was also rejected because it reduced platelets patient passes from
7/7 to 5/7. Runtime catalog v9 therefore adds only `platelets@1h`, reaching 46
unique validated cells. This demonstrates that external-domain interval
calibration is useful but target-specific, not a general promotion shortcut.

## Boundary

All results are factual research forecasts. They do not identify treatment
effects, authorize clinical use, or convert unsupported cells into numeric
outputs. Missing or unvalidated cells remain on their prior source or
persistence fallback.

## Runtime materialization verification (2026-08-06)

The six source registries now expose each cell's real `source_model`, task
type, units, change threshold, and patient/hospital/care-unit/time conformal
evidence. `whole_body_joint_runtime.py` reports that provenance instead of
labeling every source as generic joint JEPA, and calibrated intervals are
released only when all four conformal fields are present.

A permanent integration test loaded all six registry files and invoked the
serving runtime for all nine newly validated cells. Every cell moved under the
exact input contract and units, returned its source-specific hurdle task, and
released a calibrated interval. Contract, unit, horizon, or gate mismatch
continues to fall back. The complete project suite passed 374/374 tests.
