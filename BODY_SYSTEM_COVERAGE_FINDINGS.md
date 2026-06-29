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
| Fluid / electrolyte / acid-base | Electrolyte, acid-base, osmotic instability | validated bounded factual router; hospital-heldout caveat |
| Endocrine / metabolic broad | Endocrine stress / glycemic / adrenal-thyroid proxy | validated bounded factual router |
| GI / pancreatic / nutrition | GI, pancreatic, nutrition failure proxy | validated bounded factual router |
| Cardiac / myocardial | Cardiac injury and myocardial stress biomarkers | validated bounded factual router |
| Musculoskeletal | Rhabdomyolysis / muscle-injury proxy | validated bounded factual router |
| Immune / inflammatory | Immune and inflammatory activation proxy | validated bounded factual router |
| Integumentary | Skin / wound / burn proxy | validated bounded factual router |
| Toxicologic / metabolic | Poisoning / overdose / toxic-metabolic proxy | validated bounded factual router |

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
- Blood-product subtype and dose observability were added after breadth
  completion.  The 6h cohort now exposes PRBC, plasma, platelet,
  cryoprecipitate, whole-blood, and unknown blood-product channels plus
  `volume_like_ml` and `unit_like_count` features.  Aggregate future-window
  support is PRBC 1,509 windows / 515 stays, plasma 254 / 111, platelets
  306 / 102, cryoprecipitate 72 / 26, and unknown 14 / 7.
- This is no longer just a platelet/WBC proxy, but it is still not a
  transfusion or anticoagulation causal model.
- Long-horizon heme/coag audits were run on the same stay set as the 6h cohort.
  With subtype/dose observability, the router remains significant at 24h and
  48h, with the strongest active median delta at 24h (-0.133439).  Hemoglobin
  and hematocrit keep `ridge_realfit`
  in 7/7 splits at 24h, but weaken by 48h.  INR, PTT, fibrinogen, and platelets
  still fall back to persistence at 24h/48h, so horizon alone does not unlock
  coagulation-cascade targets.
- A bleeding/coagulation reserve belief audit was rerun with the new subtype/dose
  features and the same strict capacity-matched placebo gate.  The strongest
  signal is 24h feature belief for hemoglobin (5/7 pass-both) and hematocrit
  (4/7 pass-both), with weaker 24h state belief (Hgb 4/7, Hct 3/7).  No target
  reaches robust 7/7 validation, so the heme/coag belief remains candidate-only,
  unlike the validated AKI renal belief state.

## Expanded Full-Body Coverage Pass

The second body-system pass adds six broader organ/system modules and promotes
additional labs to first-class shared state variables:

- electrolytes / acid-base / osmolality: chloride, calcium, ionized calcium,
  magnesium, phosphate, anion gap, serum osmolality;
- endocrine-metabolic stress: glucose, ketones/osmolality, adrenal/thyroid
  marker proxies;
- GI / pancreatic / nutrition: lipase, amylase, triglycerides, albumin,
  prealbumin, total protein;
- cardiac / myocardial injury: troponin-I/T, BNP, CPK/CK-MB, LDH, myoglobin;
- musculoskeletal injury: CPK, myoglobin, LDH plus renal/electrolyte downstream
  targets;
- immune / inflammatory activation: CRP/CRP-hs, ESR, ferritin, WBC, temperature,
  lactate, albumin.

All six modules were run as bounded 1,500-stay full-eICU engineering cohorts.
The table reports aggregate metrics only; row-level transition parquets remain
local-only and ignored by git.

| Module | Stays | Transitions | Active Rows | Active Median Delta | Random Splits | Hospital-Heldout Delta | Stable 7/7 Ridge Targets |
|---|---:|---:|---:|---:|---:|---:|---|
| Electrolyte / acid-base | 1,385 | 17,387 | 15,255 | -0.061541 | 7/7 significant | +0.100273 | calcium, chloride, magnesium, MAP, potassium, sodium |
| Endocrine stress | 1,420 | 17,009 | 15,710 | -0.085487 | 7/7 significant | -0.044598 | glucose, MAP |
| GI / pancreatic / nutrition | 1,387 | 16,043 | 12,045 | -0.073193 | 7/7 significant | -0.018420 | glucose, MAP |
| Cardiac injury | 1,386 | 17,231 | 10,824 | -0.067364 | 7/7 significant | -0.042691 | heart rate, MAP, potassium |
| Musculoskeletal / rhabdo | 1,397 | 16,837 | 13,751 | -0.005510 | 7/7 significant | -0.023751 | potassium |
| Immune / inflammatory | 1,411 | 18,392 | 11,957 | -0.078651 | 7/7 significant | -0.040749 | none |

The expanded pass preserves the central Chapter-B rule:

- dense fast variables such as glucose, MAP, heart rate, potassium, sodium,
  chloride, calcium, and magnesium can support real-fit factual forecasting;
- sparse or slow biomarkers such as troponin, BNP, CK-MB, CPK, myoglobin,
  lipase, amylase, CRP, ESR, ferritin, thyroid markers, cortisol, serum ketones,
  osmolality, creatinine, and BUN usually fall back to persistence in a six-hour
  factual window;
- the router's value is still selective movement, not universal movement.

The electrolyte/acid-base module is a useful caution.  It passes all seven
random patient splits, but its hospital-heldout normalized MAE delta is worse
than persistence.  That suggests electrolyte measurement/practice shift across
hospitals is stronger than the random split suggests, so this module remains
validated only as a bounded engineering cohort and should not be treated as
cross-hospital robust without further calibration.

This pass moves Osler-JEPA closer to whole-body factual physiology coverage, but
it is not a complete human simulator.  Major remaining gaps include skin/wound
state, reproductive/endocrine physiology beyond ICU proxies, detailed neurologic
exam trajectories, procedure-specific cardiac/neuro variables, microbiology and
immune phenotype depth, and high-resolution physical exam states.

## Breadth Completion Pass

The final breadth pass adds the remaining feasible adult-ICU coverage modules:

- `integumentary_skin_wound`: pressure ulcer, wound, burn, cellulitis, skin
  infection, and related physiologic proxy coverage;
- `toxic_metabolic`: poisoning, overdose, toxic ingestion, and metabolic
  derangement proxy coverage.

| Module | Stays | Transitions | Active Rows | Active Median Delta | Random Splits | Hospital-Heldout Delta | Stable 7/7 Ridge Targets |
|---|---:|---:|---:|---:|---:|---:|---|
| Integumentary / skin / wound | 1,377 | 17,327 | 10,023 | -0.058359 | 7/7 significant | -0.030696 | glucose, heart rate, hematocrit, hemoglobin, MAP |
| Toxicologic / metabolic | 1,333 | 11,318 | 5,390 | -0.094078 | 7/7 significant | -0.014730 | glucose, heart rate, MAP, oxygen saturation, potassium, respiratory rate |

Both modules preserve the same pattern:

- dense physiologic downstream targets can beat persistence;
- skin/wound healing state, toxin concentration, and exposure-level trajectories
  are not first-class in the current eICU state map;
- sparse or unobserved system-specific targets must remain data ceilings, not
  hallucinated state variables.

With these two modules, the adult-ICU breadth pass now covers all standard
organ-system buckets that eICU can support with factual observed physiology:
cardiovascular, respiratory, nervous, renal/urinary, endocrine/metabolic,
digestive/hepatic/pancreatic/nutrition, hematologic/coagulation, immune/lymphatic
proxy, musculoskeletal, integumentary, and fluid/electrolyte/acid-base.

Reproductive/obstetric physiology is marked as a data ceiling in this adult ICU
dataset.  It is not implemented as a router because eICU lacks reliable
high-density pregnancy, fetal, obstetric intervention, or reproductive hormone
trajectories for this contract.

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
  normalization with stronger unit semantics, blood-product protocol context,
  bleeding-source/procedure context, and anticoagulation reversal evidence;
- hepatic: INR, ammonia/encephalopathy proxy, paracentesis/bleeding context;
- neurologic: GCS/mental-status proxies, ICP/EVD/procedure evidence;
- cardiovascular: rhythm/procedure evidence, inotrope dose normalization;
- endocrine beyond DKA: HHS/hypoglycemia and thyroid/adrenal crisis contracts.
- integumentary: structured wound stage/size/drainage and burn surface-area
  trajectories;
- toxicologic: measured toxin levels, ingestion timing, antidote dose
  normalization, and poison-control protocol context;
- reproductive/obstetric: external obstetric ICU datasets would be required;
  eICU is treated as a data ceiling for this system.
