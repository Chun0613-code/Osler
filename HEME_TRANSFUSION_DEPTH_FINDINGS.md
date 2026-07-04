# Heme Transfusion Depth Findings

Date: 2026-06-28

This is the first Chapter-B depth pass after body-system breadth completion.  It
tests whether better blood-product observability improves the hematology /
coagulation factual router or the bleeding/coagulation reserve belief gate.

The change is observational only:

- blood products are split into PRBC, plasma, platelet, cryoprecipitate,
  whole-blood, and unknown subtypes;
- intakeOutput numeric values are exposed as `volume_like_ml` and
  `unit_like_count` evidence;
- medication/treatment text contributes subtype presence when possible;
- the original `transfusion` action channel remains available for backward
  compatibility;
- no transfusion, anticoagulation, or blood-product causal claim is allowed.

## Observability

The 6h heme/coag cohort remains the same bounded stay set:

- 1,382 stays
- 16,870 transitions
- 8,055 active heme/coag windows

Subtype support is now visible in the aggregate report:

| Product | Future Windows | Future Stays | Dose-Observed Windows | Unit-Like Count | Volume-Like mL |
|---|---:|---:|---:|---:|---:|
| PRBC | 1,509 | 515 | 1,509 | 2,408.191536 | 722,457.46 |
| Plasma | 254 | 111 | 252 | 542.654080 | 135,663.52 |
| Platelet | 306 | 102 | 306 | 348.624031 | 104,587.21 |
| Cryoprecipitate | 72 | 26 | 72 | 190.550000 | 19,055.00 |
| Unknown blood product | 14 | 7 | 14 | 161.100000 | 48,330.00 |
| Whole blood | 0 | 0 | 0 | 0.000000 | 0.00 |

This means the depth pass did improve treatment observability.  The question is
whether that observability changes validated downstream prediction.

## Router Results

| Horizon | Active Median Delta | Random Splits | Hospital-Heldout Delta | Stable 7/7 Real-Fit Targets |
|---|---:|---:|---:|---|
| 6h | -0.084253 | 7/7 significant | -0.026300 | hemoglobin, hematocrit, MAP |
| 24h | -0.133439 | 7/7 significant | -0.012920 | hemoglobin, hematocrit, MAP |
| 48h | -0.109033 | 7/7 significant | -0.009951 | none for heme labs; MAP 6/7 |

Compared with the previous heme/coag audits, the factual router is essentially
stable.  Subtype/dose features do not unlock INR, PTT, fibrinogen, or platelets.
Hgb/Hct remain the strongest heme targets, especially at 24h.

## Belief Gate Results

The belief gate remains strict: a candidate must significantly beat both the
baseline ridge model and a capacity-matched placebo.

| Horizon / Belief | Hgb Pass-Both | Hct Pass-Both | INR Pass-Both | Platelets Pass-Both | PTT Pass-Both | Fibrinogen Pass-Both |
|---|---:|---:|---:|---:|---:|---:|
| 6h feature | 0/7 | 1/7 | 0/7 | 0/7 | 0/7 | 0/7 |
| 6h state | 0/7 | 1/7 | 0/7 | 0/7 | 0/7 | 0/7 |
| 24h feature | 5/7 | 4/7 | 2/7 | 0/7 | 0/7 | 0/7 |
| 24h state | 4/7 | 3/7 | 1/7 | 0/7 | 1/7 | 0/7 |
| 48h feature | 2/7 | 3/7 | 0/7 | 1/7 | 0/7 | 0/7 |
| 48h state | 2/7 | 2/7 | 0/7 | 2/7 | 0/7 | 0/7 |

The new observability creates a real 24h signal for Hgb/Hct, but it still fails
the promotion boundary.  No heme/coag belief state is promoted.

## Interpretation

This is a useful bounded negative result:

- blood-product subtype/dose extraction works and is non-empty;
- the extra observability improves 24h Hgb/Hct belief signal;
- it does not robustly unlock coagulation cascade targets;
- INR/PTT/fibrinogen remain limited by sparse measurement, treatment bundling,
  and observational confounding;
- platelet behavior remains unstable and does not pass the placebo-gated belief
  boundary.

The correct next heme step is not to relax the gate.  It is to improve external
observability and identification:

- product-specific dose normalization with stronger unit semantics;
- timestamped transfusion protocol context;
- bleeding source/procedure context;
- anticoagulation reversal evidence;
- external randomized or otherwise identified transfusion data for Chapter A.

The heme/coag module therefore stays factual and candidate-only for hidden
states.

