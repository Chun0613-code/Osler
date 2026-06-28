# Body-System Coverage Findings

Date: 2026-06-28

This extends Chapter B from individual disease modules into broad body-system
coverage.  The new layer is a reusable eICU adapter:

- `eicu_body_system_configs.py`
- `eicu_body_system_transition_extract.py`
- `eicu_body_system_target_router.py`

Each system still has its own isolated disease contract, target/action list,
active-window rule, discovery-only source selection, and held-out router gate.
No shared latent space, causal claim, clinical claim, or runtime action authority
is granted.

## Current Body Coverage

| System | Module | Status |
|---|---|---|
| Endocrine / metabolic | DKA | validated factual router |
| Infectious / systemic | Sepsis | validated factual router |
| Renal | AKI | validated factual router + long-horizon renal belief |
| Pulmonary | Respiratory failure / hypoxemia | validated bounded factual router |
| Cardiovascular | Cardiovascular instability / shock / heart failure | validated bounded factual router |
| Nervous system | Acute neurologic injury / seizure / coma | validated bounded physiologic-proxy router |
| Hepatic / GI | Hepatic failure / cirrhosis | validated bounded proxy router |
| Hematologic | Coagulopathy / thrombocytopenia / bleeding | validated bounded first-class heme/coag router + long-horizon audit + candidate-only belief audit |

## New Bounded Router Results

All four new body-system modules were run on deterministic bounded full-eICU
cohorts with seven patient split seeds and a hospital-heldout split.

| Module | Stays | Transitions | Active Rows | Active Median Delta | Random Splits | Hospital-Heldout Delta |
|---|---:|---:|---:|---:|---:|---:|
| Cardiovascular instability | 1,382 | 16,047 | 4,734 | -0.145857 | 7/7 significant | -0.178511 |
| Acute neuro proxy | 1,390 | 16,238 | 11,418 | -0.086287 | 7/7 significant | -0.065234 |
| Hepatic failure proxy | 1,396 | 17,730 | 10,717 | -0.073574 | 7/7 significant | -0.064441 |
| Coagulopathy / heme v2 | 1,382 | 16,870 | 8,055 | -0.083132 | 7/7 significant | -0.087116 |

The consistent pattern remains the same as DKA/sepsis/AKI/respiratory:

- dense fast variables select `ridge_realfit`;
- slow or sparse variables fall back to persistence or population delta;
- the router wins because it refuses to move unsupported targets.

## Per-System Notes

Cardiovascular:

- `ridge_realfit` is stable for MAP, heart rate, and potassium.
- Creatinine, urine output, lactate, and most bicarbonate splits fall back.
- This is a hemodynamic factual router, not a vasopressor/inotrope causal model.

Acute neuro:

- `ridge_realfit` is stable for glucose, heart rate, MAP, oxygen saturation, and
  respiratory rate.
- Sodium and pH are mixed/fallback targets.
- This is a physiologic proxy module because ICU EHR does not reliably expose
  neurologic exam trajectories.

Hepatic:

- MAP provides the strongest six-hour signal.
- Bilirubin, direct bilirubin, and creatinine fall back to persistence.
- This likely needs longer horizons and better coagulation/encephalopathy state
  variables before becoming a deeper hepatic digital-twin module.

Hematologic:

- Hemoglobin, hematocrit, INR, PTT, fibrinogen, and transfusion evidence are now
  first-class inputs.
- Hemoglobin and hematocrit select `ridge_realfit` in 7/7 splits.
- INR has partial signal (`ridge_realfit` in 2/7 splits); PTT and fibrinogen
  correctly fall back to persistence in this bounded 6h window.
- Transfusion evidence is now captured from both treatment text and
  intakeOutput blood-product rows: 577 stays / 1,860 windows, with 1,858
  dose-observed windows.
- This is no longer just a platelet/WBC proxy, but it is still not a
  transfusion or anticoagulation causal model.
- Long-horizon heme/coag audits were run on the same stay set as the 6h cohort.
  The router remains significant at 24h and 48h, with the strongest active
  median delta at 24h (-0.133450).  Hemoglobin and hematocrit keep `ridge_realfit`
  in 7/7 splits at 24h, but weaken by 48h.  INR, PTT, fibrinogen, and platelets
  still fall back to persistence at 24h/48h, so horizon alone does not unlock
  coagulation-cascade targets.
- A bleeding/coagulation reserve belief audit was run with a strict
  capacity-matched placebo gate.  Feature beliefs showed partial Hct/Hgb signal
  (Hct 4/7, Hgb 3/7 passes-both), but no target reached robust 7/7 validation,
  hospital-heldout did not pass, and the explicit predict-update state was
  weaker.  The heme/coag belief therefore remains candidate-only, unlike the
  validated AKI renal belief state.

## Boundary

These artifacts are factual and observational.

- No causal treatment-effect claim is allowed.
- No counterfactual treatment claim is allowed.
- No clinical recommendation claim is allowed.
- No runtime action authority is granted.
- No checkpoint promotion is allowed.
- No active symbolic-rule promotion is allowed.
- Row-level parquet cohorts remain local-only and ignored by git.

## Next Coverage Work

The most valuable next step is not just adding more disease names.  It is adding
missing first-class variables so the proxy systems become deeper:

- hematology: bleeding/coagulation reserve belief state, transfusion dose
  normalization, blood-product subtype separation, and longer-horizon
  hemoglobin/coagulation evaluation;
- hepatic: INR, ammonia/encephalopathy proxy, paracentesis/bleeding context;
- neurologic: GCS/mental-status proxies, ICP/EVD/procedure evidence;
- cardiovascular: rhythm/procedure evidence, inotrope dose normalization;
- endocrine beyond DKA: HHS/hypoglycemia and thyroid/adrenal crisis contracts.
