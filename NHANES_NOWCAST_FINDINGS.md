# NHANES 2017-2018 Cross-Sectional Nowcast Findings

This audit tests whether the Osler observation-layer nowcast recipe transfers from ICU data to a non-ICU, healthy-population cross-sectional dataset.

NHANES has no per-person time axis, so this is **nowcast only**. It does not support forecast, treatment-response, or causal claims.

## Gate

- Candidate: ridge regression using contemporaneous non-target variables.
- Baseline: train-set target median.
- Placebo: ridge regression on the same number of random-noise features.
- Validation rule: candidate beats both baseline and placebo in all 7 participant-heldout splits.
- Leakage guard: deterministic or near-deterministic sibling groups are excluded from one another's feature sets: `sbp/dbp/map`, `hemoglobin/hematocrit/rbc`, and `bmi/weight/waist`.
- Output is aggregate-only; no participant identifiers or row-level data are written.

## Result

- Merged participants: `6401`
- Evaluated targets: `27`
- Validated nowcast targets: `25 / 27`
- Validated: `sodium, potassium, chloride, bicarbonate, bun, creatinine, calcium, phosphorus, albumin, total_protein, bilirubin, alt, ast, uric_acid, hemoglobin, hematocrit, wbc, platelets, rbc, sbp, dbp, map, bmi, weight, waist`
- Failed: `glucose, alk_phos`

## Interpretation

The same disciplined nowcast pattern seen in ICU data also appears in NHANES: many contemporaneous lab/body variables are constrained enough by the rest of the physiologic panel to beat both median and capacity-matched placebo baselines, even after excluding deterministic sibling variables.

The negative targets are also informative: glucose and alkaline phosphatase do not pass this cross-sectional gate, so they should remain missing/fallback in the NHANES nowcast contract.

This is a healthy-population observation result, not a clinical or causal result.

## Target-Level Summary

| Target | N | Baseline MAE | Candidate MAE | Placebo MAE | Beats Baseline | Beats Placebo | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| sodium | 5374 | 2.1390 | 1.3400 | 2.1618 | 7/7 | 7/7 | validated |
| potassium | 5374 | 0.2759 | 0.2586 | 0.2791 | 7/7 | 7/7 | validated |
| chloride | 5374 | 2.0675 | 1.2548 | 2.0969 | 7/7 | 7/7 | validated |
| bicarbonate | 5374 | 1.9492 | 1.5994 | 1.9557 | 7/7 | 7/7 | validated |
| bun | 5374 | 4.0065 | 3.2149 | 4.1102 | 7/7 | 7/7 | validated |
| creatinine | 5374 | 0.2035 | 0.1738 | 0.2109 | 7/7 | 7/7 | validated |
| glucose | 5374 | 15.4775 | 17.6523 | 17.8505 | 0/7 | 5/7 | fallback |
| calcium | 5374 | 0.2873 | 0.2241 | 0.2898 | 7/7 | 7/7 | validated |
| phosphorus | 5374 | 0.4548 | 0.3927 | 0.4581 | 7/7 | 7/7 | validated |
| albumin | 5374 | 0.2600 | 0.1813 | 0.2636 | 7/7 | 7/7 | validated |
| total_protein | 5374 | 0.3384 | 0.2795 | 0.3401 | 7/7 | 7/7 | validated |
| bilirubin | 5374 | 0.1819 | 0.1751 | 0.1941 | 7/7 | 7/7 | validated |
| alt | 5374 | 9.2077 | 5.5223 | 10.1063 | 7/7 | 7/7 | validated |
| ast | 5374 | 5.9980 | 3.8813 | 6.5709 | 7/7 | 7/7 | validated |
| alk_phos | 5374 | 28.2722 | 29.6287 | 30.7496 | 1/7 | 7/7 | fallback |
| uric_acid | 5374 | 1.1573 | 0.9151 | 1.1649 | 7/7 | 7/7 | validated |
| hemoglobin | 5374 | 1.1921 | 0.8618 | 1.1969 | 7/7 | 7/7 | validated |
| hematocrit | 5374 | 3.2241 | 2.3636 | 3.2362 | 7/7 | 7/7 | validated |
| wbc | 5374 | 1.8276 | 1.6802 | 1.8727 | 7/7 | 7/7 | validated |
| platelets | 5374 | 47.3881 | 41.6286 | 47.8264 | 7/7 | 7/7 | validated |
| rbc | 5374 | 0.3914 | 0.3018 | 0.3928 | 7/7 | 7/7 | validated |
| sbp | 5374 | 14.8685 | 11.8266 | 15.1677 | 7/7 | 7/7 | validated |
| dbp | 5374 | 9.8108 | 8.9068 | 9.8557 | 7/7 | 7/7 | validated |
| map | 5374 | 10.0043 | 8.4799 | 10.0771 | 7/7 | 7/7 | validated |
| bmi | 5521 | 5.5974 | 4.7152 | 5.7022 | 7/7 | 7/7 | validated |
| weight | 5529 | 17.5123 | 14.5878 | 17.7645 | 7/7 | 7/7 | validated |
| waist | 5382 | 14.2129 | 11.0388 | 14.2741 | 7/7 | 7/7 | validated |
