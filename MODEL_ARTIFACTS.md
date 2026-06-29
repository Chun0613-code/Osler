# Model artifacts

These local research artifacts are versioned with the implementation so a clone
can reproduce the documented inference paths without retraining.

| Artifact | Size | SHA-256 | Purpose |
|---|---:|---|---|
| `dka_intervention_jepa.pt` | 942 KiB | `fc0f4e3b1983bc0e9566d00d87394da5a20c4114b93cffb48f3b118b20943bc4` | Earlier action-conditioned DKA JEPA checkpoint. |
| `dka_intervention_jepa_v4.pt` | 952 KiB | `71505038bed6ab2a78343abc080c01a6aaa0ed48d572e2bb7a2f227854cad7d7` | Route-aware insulin PK, potassium belief state, and hyperosmolar-risk checkpoint. |
| `dka_symbolic_jepa_v5.pt` | 1.3 MiB | `f4767fec387af5093e59c4adf0eb6f53d9ce236d40741bdc48defd9ff4c79d27` | Symbolic-grounded checkpoint with direction, status, proof-path, and rule-proposal heads. |
| `dka_symbolic_jepa_v6_candidate.pt` | 1.4 MiB | `febba04985aa3fd878bd61a2223f7af69931c92bbc930bb88c336e4337ba2c73` | Candidate-only checkpoint trained from repaired DKABody plus grey-box residual; not promoted over v5. |
| `dka_symbolic_jepa_v6_full_candidate.pt` | 1.4 MiB | `ae885f6ac7a01c1866e0cf327895a2e7776a41422247b3e152e1dff95e1355ae` | Full-budget 1,000-scenario/55-epoch v6 candidate; rejected for runtime promotion versus v5. |
| `dka_real_world_adapter_v1.joblib` | 66 KiB | `f68f1604b679a240c6bf5fe9e8f1887b7831ccdd986ae04c804d9ca8d4956a4a` | Guarded factual residual adapter with ensemble uncertainty and state-wise fallback. |

## Data boundary

The repository does not include MIMIC parquet cohorts, fidelity JSONL files,
patient-level out-of-fold CSV predictions, or candidate-rule files containing
stay identifiers. Aggregate evaluation reports are included.

PhysioNet/CinC Challenge 2019 PSV files and generated `physionet2019_cache.npz`
are local-only patient data artifacts and are ignored by git. The local
`physionet2019_icu_jepa.pt` checkpoint is also ignored by default; the committed
`physionet2019_icu_jepa_report.json` records aggregate, non-identified training
metrics for the general ICU pretraining path.

PhysioNet eICU demo CSV files under `physionet.org/`, generated
`eicu_demo_cache.npz`, and local `eicu_demo_icu_jepa.pt` are also local-only.
The committed `eicu_demo_icu_jepa_report.json` records aggregate generic ICU
pretraining metrics. The committed `eicu_demo_action_audit.json` records
aggregate treatment-table coverage for DKA-relevant action channels and contains
no raw rows or patient identifiers.

`eicu_dka_transitions_6h_demo.parquet` is local-only and ignored by git. It is
the eICU demo DKA-like transition cohort used for factual proxy evaluation.
Only defensible `infusionDrug` administrations enter numeric action grids;
eICU medication orders, treatment text, and unknown-dose infusions are retained
as treatment-presence evidence only. Committed aggregate reports include
`eicu_dka_transition_report.json`, `eicu_dka_v5_evaluation.json`,
`eicu_dka_physionet_presentation_only_evaluation.json`,
`eicu_dka_real_proxy_comparison.json`, `eicu_dka_symbolic_real_test_v5.json`,
and `eicu_dka_per_target_ensemble.json`. These reports contain cohort-level
counts and metrics only; raw rows, timestamps, and patient identifiers are not
versioned.

`dka_transitions_6h_mimiciv_full_v31_icd.parquet` is local-only and ignored by
git. It is the credentialed MIMIC-IV v3.1 ICD-supported, lab-defined DKA
transition cohort used for full-scale factual proxy testing. Committed aggregate
reports include `mimiciv_full_v31_icd_dka_transition_report.json`,
`mimiciv_full_v31_icd_v5_evaluation.json`,
`mimiciv_full_v31_icd_presentation_only_evaluation.json`,
`mimiciv_full_v31_icd_per_target_ensemble.json`,
`mimiciv_full_v31_icd_robustness.json`,
`mimiciv_full_v31_icd_treatment_recovery_audit.json`,
`mimiciv_full_v31_icd_causal_diagnostics.json`,
`mimiciv_full_v31_icd_greybox_realfit.json`,
`mimiciv_full_v31_icd_target_router.json`, and
`mimiciv_full_v31_icd_symbolic_real_test_v5.json`. These reports contain
cohort-level counts and metrics only; raw rows, timestamps, and patient
identifiers are not versioned. The patient-held-out per-target ensemble beats
persistence on active-DKA factual proxy MAE and remains significant across seven
random patient split seeds and a time-order split, but this remains
observational. The real-patient grey-box residual fit learns target-specific
glucose and ketone/anion-gap recovery signal, but it is not promoted as a
whole-state residual model because sodium and creatinine regress versus
persistence. The nested target router safely gates that residual to anion gap,
keeps presentation-only for glucose, and falls back to persistence for the other
targets; it improves over the no-realfit base router in all seven patient split
seeds. Causal diagnostics are runnable but fail readiness gates; these artifacts
do not permit causal, counterfactual, clinical, checkpoint-promotion,
residual-artifact-promotion, or active-rule-promotion claims.

`next_chapter_ab_contract.json` is an aggregate-only post-DKA build contract.
It contains no patient rows or identifiers. It records that the causal chapter
is fail-closed until randomized, instrumental-variable, or front-door evidence is
mapped into `osler_jepa/causal_readiness.py`, and that multi-disease expansion
should reuse the disease-specific target-router template in
`osler_jepa/disease_router.py`.

Full eICU sepsis and AKI router artifacts follow the same aggregate/report
boundary. Local transition parquet cohorts, including
`eicu_aki_transitions_6h.parquet`, `eicu_aki_transitions_24h.parquet`,
`eicu_aki_transitions_48h.parquet`, and `eicu_sepsis_transitions_6h.parquet`,
are row-level research cohorts and are not versioned. Committed aggregate AKI
reports include `eicu_aki_transition_report.json`,
`eicu_aki_target_router.json`, `eicu_aki_mechanism_target_router.json`,
`eicu_aki_transition_report_24h.json`, `eicu_aki_target_router_24h.json`,
`eicu_aki_transition_report_48h.json`, and
`eicu_aki_target_router_48h.json`. The 24h/48h AKI reports show that
creatinine and BUN fall back to persistence at 6h but select `ridge_realfit` in
7/7 patient splits at both 24h and 48h. These artifacts remain factual,
observational, and non-clinical.

`eicu_aki_renal_belief_audit.json` is also aggregate-only. It evaluates
candidate renal reserve/GFR belief features on 24h/48h AKI cohorts. The belief
candidate improves downstream creatinine and BUN prediction beyond both baseline
`ridge_realfit` and a capacity-matched placebo in 7/7 patient splits at both
horizons. The artifact does not contain raw rows or patient identifiers and does
not permit direct hidden-state accuracy, causal, counterfactual, clinical,
checkpoint-promotion, or active-rule-promotion claims.

`eicu_respiratory_transitions_6h.parquet` is local-only and ignored by git. It
is a bounded deterministic respiratory-failure / severe-hypoxemia transition
cohort extracted from full eICU for the fourth Chapter-B disease module.
Committed aggregate reports include `eicu_respiratory_transition_report.json`
and `eicu_respiratory_target_router.json`. The bounded cohort has 4,170
evaluable stays, 57,673 transitions, and 26,558 active respiratory transitions.
The nested respiratory router beats persistence in 7/7 random patient splits and
in hospital-heldout evaluation. This remains factual, observational,
non-clinical, and bounded-cohort only; no causal or runtime treatment authority
is granted.

The generic body-system adapter produces additional local-only row-level
parquets: `eicu_cardiovascular_instability_transitions_6h.parquet`,
`eicu_acute_neuro_transitions_6h.parquet`,
`eicu_hepatic_failure_transitions_6h.parquet`, and
`eicu_coagulopathy_heme_transitions_6h.parquet`. These are ignored by git.
Committed aggregate reports include the matching `*_transition_report.json` and
`*_target_router.json` files plus `BODY_SYSTEM_COVERAGE_FINDINGS.md`. These
bounded modules extend factual-router coverage to cardiovascular, neurologic
proxy, hepatic/GI proxy, and hematology/coagulation systems. The heme/coag
module now includes first-class hemoglobin, hematocrit, INR, PTT, fibrinogen,
and transfusion evidence. They remain observational, non-clinical,
bounded-cohort artifacts with no causal or runtime treatment authority.

The expanded full-body coverage pass produces additional local-only row-level
parquets: `eicu_electrolyte_acid_base_transitions_6h.parquet`,
`eicu_endocrine_stress_transitions_6h.parquet`,
`eicu_gi_pancreatic_nutrition_transitions_6h.parquet`,
`eicu_cardiac_injury_transitions_6h.parquet`,
`eicu_musculoskeletal_rhabdo_transitions_6h.parquet`, and
`eicu_immune_inflammatory_transitions_6h.parquet`. These are ignored by git.
Committed aggregate reports include the matching `*_transition_report.json` and
`*_target_router.json` files plus updated `BODY_SYSTEM_COVERAGE_FINDINGS.md`.
They widen factual physiology coverage to electrolyte/acid-base, endocrine,
GI/pancreatic/nutrition, cardiac injury, musculoskeletal/rhabdomyolysis, and
immune/inflammatory systems.  The pass remains bounded, observational,
non-clinical, and non-causal; it does not make Osler-JEPA a complete human
simulator.

The final breadth-completion pass adds local-only row-level parquets:
`eicu_integumentary_skin_wound_transitions_6h.parquet` and
`eicu_toxic_metabolic_transitions_6h.parquet`. These are ignored by git.
Committed aggregate reports include the matching `*_transition_report.json` and
`*_target_router.json` files plus `BODY_SYSTEM_COVERAGE_COMPLETE.md`. The pass
closes the feasible adult-ICU body-system breadth layer with integumentary and
toxicologic/metabolic proxy modules, while marking reproductive/obstetric
physiology as an eICU data ceiling.  It remains factual, observational,
bounded, non-clinical, and non-causal.

`eicu_aki_renal_belief_state_audit.json` is the explicit predict-update version
of the renal belief audit. It is aggregate-only. The state belief passes
downstream gates for BUN at 24h/48h and for urine output at 24h, but it does not
pass for creatinine. This artifact validates a narrower online-compatible belief
state and preserves the same safety boundary.

`eicu_aki_renal_belief_state_v2_audit.json` is the creatinine-specific
predict-update refinement. It is aggregate-only. The v2 state adds a separate
creatinine level/slope/innovation dimension and passes the downstream gate for
24h creatinine/BUN/urine output and 48h creatinine/BUN. It remains factual,
observational, and non-clinical.

`eicu_heme_coag_belief_audit.json`,
`eicu_heme_coag_state_belief_audit.json`,
`eicu_heme_coag_belief_audit_24h.json`,
`eicu_heme_coag_state_belief_audit_24h.json`,
`eicu_heme_coag_belief_audit_48h.json`, and
`eicu_heme_coag_state_belief_audit_48h.json` are aggregate-only hematology
candidate-belief audits. They evaluate bleeding/coagulation reserve features and
an explicit predict-update coagulation state against both baseline
`ridge_realfit` and a capacity-matched placebo. After adding blood-product
subtype and normalized dose evidence, the strongest signal is 24h feature belief
for hemoglobin (5/7 pass-both) and hematocrit (4/7 pass-both), but no target
passes robust 7/7 validation. These artifacts do not validate a promoted
heme/coag hidden state; they keep the belief candidate-only and preserve the
same factual, observational, non-clinical boundary.

`eicu_coagulopathy_heme_transitions_24h.parquet` and
`eicu_coagulopathy_heme_transitions_48h.parquet` are local-only long-horizon
heme/coag row-level cohorts and are ignored by git. They are restricted
internally to the existing 6h heme/coag stay set for apples-to-apples horizon
comparison. Aggregate reports are committed as
`eicu_coagulopathy_heme_transition_report_24h.json`,
`eicu_coagulopathy_heme_transition_report_48h.json`,
`eicu_coagulopathy_heme_target_router_24h.json`, and
`eicu_coagulopathy_heme_target_router_48h.json`. The 24h router has the
strongest heme/coag active-window delta and preserves 7/7 hemoglobin/hematocrit
`ridge_realfit` selection; 48h remains significant overall but heme lab
selection weakens. Coagulation cascade targets remain persistence fallback.
`HEME_TRANSFUSION_DEPTH_FINDINGS.md` records the subtype/dose depth pass:
PRBC/plasma/platelet/cryo/unknown blood-product evidence is now separated with
`volume_like_ml` and `unit_like_count` features, but the improvement remains
partial and candidate-only.

`eicu_body_system_coupling_audit.json` is the first aggregate-only
cross-system coupling audit after breadth completion. It tests seven directed
body-system edges by comparing a coupled ridge candidate against both a
no-upstream baseline and a capacity-matched placebo. No edge passes the robust
7/7 active-window promotion boundary; endocrine -> electrolyte potassium and
sodium show only 1/7 weak candidate signals. `BODY_SYSTEM_COUPLING_FINDINGS.md`
records the fail-closed interpretation and the next temporal/belief-based
coupling direction. The artifact contains no raw rows or patient identifiers
and grants no causal, counterfactual, clinical, checkpoint-promotion, runtime,
or active-rule authority.

`eicu_body_system_temporal_coupling_audit.json` is the second aggregate-only
cross-system coupling audit. It tests temporal predict-update coupling belief
features instead of static upstream feature concatenation. The temporal layer
produces weak plausible signals, including renal -> bicarbonate in 2/7 patient
splits, renal -> sodium in 1/7, and immune -> MAP in 1/7, but no edge reaches the
robust 7/7 active-window promotion boundary. `BODY_SYSTEM_TEMPORAL_COUPLING_FINDINGS.md`
records the candidate-only interpretation. This artifact contains no raw rows or
patient identifiers and grants no causal, counterfactual, clinical,
checkpoint-promotion, runtime, or active-rule authority.

`eicu_body_system_edge_specific_coupling_audit.json` is the third
aggregate-only cross-system coupling audit. It tests bespoke shared-state
features for each directed edge, including renal-electrolyte buffering,
respiratory CO2/ventilation mismatch, endocrine osmotic/electrolyte interaction,
heme oxygen-delivery debt, cardio-renal perfusion stress, hepatic/coagulation
state, and immune capillary-leak/shock state. Edge-specific features produce
more plausible weak signals than generic temporal coupling, but still no edge
passes the robust 7/7 active-window promotion boundary. The artifact remains
candidate-only and contains no raw rows or patient identifiers.

`eicu_body_system_focused_coupling_audit.json` is the fourth aggregate-only
cross-system coupling audit. It narrows the search to explicit renal-electrolyte
store dynamics, long-horizon cardio-renal coupling, sepsis/immune to
cardiovascular coupling, and hepatic to renal coupling. The renal-electrolyte
store candidate remains rejected for promotion, with only weak bicarbonate and
anion-gap signals. Cardio-renal long-horizon coupling is the first validated
cross-system factual edge: creatinine and BUN pass the candidate-versus-baseline
and candidate-versus-placebo gate in 7/7 patient splits at both 24h and 48h,
with hospital-heldout support. Sepsis/immune -> cardiovascular is the second
validated factual edge for MAP at 6h. Hepato-renal improves to candidate signal
but does not promote at 6h, 24h, or 48h, so it is not a simple horizon mismatch.
Urine output, heart rate, and lactate remain partial. The matching findings are
recorded in `BODY_SYSTEM_FOCUSED_COUPLING_FINDINGS.md`. This artifact contains
no raw rows or patient identifiers and grants no causal, counterfactual,
clinical, checkpoint-promotion, runtime, or active-rule authority.

`eicu_hepatic_failure_transitions_24h.parquet` and
`eicu_hepatic_failure_transitions_48h.parquet` are local-only long-horizon
hepatic row-level cohorts and are ignored by git. They are restricted internally
to the existing 6h hepatic stay set for apples-to-apples horizon comparison.
Aggregate reports are committed as
`eicu_hepatic_failure_transition_report_24h.json` and
`eicu_hepatic_failure_transition_report_48h.json`; focused coupling audit
summaries are committed as
`eicu_body_system_focused_hepato_renal_24h_audit.json` and
`eicu_body_system_focused_hepato_renal_48h_audit.json`. These artifacts remain
candidate-only.

`eicu_body_system_multihop_coupling_audit.json` is the first aggregate-only
multi-hop coupling audit. It tests whether a discovery-only MAP mediator adds
renal prediction signal beyond a direct sepsis-to-renal ridge baseline and a
capacity-matched placebo on the full eICU sepsis cohort. BUN passes 7/7 active
patient splits, creatinine passes 6/7, and urine output fails; hospital-heldout
support is mixed. The matching findings are recorded in
`BODY_SYSTEM_MULTIHOP_COUPLING_FINDINGS.md`. This is a strong candidate-only
whole-body path, not a promoted clinical or causal edge.

`eicu_respiratory_acid_base_observability_audit.json` is an aggregate-only label
coverage scan for the next respiratory -> acid-base coupling attempt. It shows
that eICU has substantial PaCO2, FiO2, PEEP, tidal-volume, ventilator-mode, and
minute-ventilation observability in lab and respiratory charting tables. The
next respiratory coupling pass should promote these to first-class features
before retesting respiratory -> acid-base.

`mimiciii_dka_transitions_6h_demo.parquet` is local-only and ignored by git. It
is the observed-treatment MIMIC-III demo lab-defined DKA-like transition cohort
used for schema and factual proxy smoke testing. Committed aggregate reports
include `mimiciii_dka_transition_report.json`, `mimiciii_dka_v5_evaluation.json`,
`mimiciii_dka_presentation_only_evaluation.json`,
`mimiciii_dka_symbolic_real_test_v5.json`, and
`mimiciii_dka_real_proxy_comparison.json`. These reports contain cohort-level
counts and metrics only; raw rows, timestamps, and patient identifiers are not
versioned. The local demo has 0 ICD-confirmed DKA stays, so this artifact must
not be used for checkpoint promotion or clinical claims.

`dka_physionet_encoder_init.pt` is a local-only candidate initialization
checkpoint. It contains feature-aligned encoder transfer from the PhysioNet ICU
JEPA into a fresh DKA WorldModel, but it is not a promoted DKA checkpoint. The
committed `dka_physionet_encoder_init_report.json` records the transfer map and
promotion boundary.

`dka_physionet_transfer_candidate.pt` is also local-only. It was trained at the
full 1,000-scenario/55-epoch candidate budget and rejected for runtime
promotion. The committed `dka_v5_vs_physionet_transfer_comparison.json` and
`dka_symbolic_real_test_physionet_transfer_candidate.json` record the aggregate
evaluation.

`physionet2019_dkabody_calibration.json` and
`physionet2019_dkabody_calibration_audit.json` are aggregate simulator-prior
artifacts. They contain no patient rows or identifiers. They calibrate DKABody
presentation priors, observable heterogeneity proxies, action-unobserved drift
targets, and measurement masks/ages. The drift section is not a no-treatment
causal estimate because PhysioNet 2019 has no medication action channels.

`dka_physionet_calibrated_candidate.pt` is local-only. It was trained at the
full 1,000-scenario/55-epoch candidate budget using the PhysioNet DKABody
calibration artifact plus the existing grey-box residual and action prior. The
full checkpoint was rejected for runtime promotion. Its glucose point estimate
improved, but simulator factual MSE, counterfactual sign accuracy, latent rank,
and external symbolic changed-only accuracy regressed. Aggregate reports are
committed as `dka_physionet_calibrated_candidate_report.json`,
`dka_symbolic_real_test_physionet_calibrated_candidate.json`, and
`dka_v5_vs_physionet_calibrated_comparison.json`.

`dka_physionet_presentation_only_candidate.pt` is local-only. It was trained at
the same full candidate budget with PhysioNet presentation/profile priors while
disabling the real ICU measurement mask/age model. This ablation improved
glucose, latent rank, simulator factual MSE, and counterfactual sign accuracy
relative to the full calibrated candidate, but it still failed promotion gates
for potassium, bicarbonate, and external symbolic direction accuracy. Aggregate
reports are committed as `dka_physionet_presentation_only_candidate_report.json`,
`dka_symbolic_real_test_physionet_presentation_only_candidate.json`, and
`dka_physionet_calibration_ablation_comparison.json`.

`dka_physionet_presentation_only_ltc_candidate.pt` is local-only. It was trained
with the same presentation-only calibration budget as
`dka_physionet_presentation_only_candidate.pt`, changing only the dynamics cell
from the residual MLP to LTC. It was rejected for promotion: it remained
non-collapsed and slightly improved simulator MSE/action sensitivity, but failed
active-DKA glucose versus persistence and did not improve external symbolic
changed-only accuracy. Aggregate reports are committed as
`dka_physionet_presentation_only_ltc_candidate_report.json`,
`dka_symbolic_real_test_physionet_presentation_only_ltc_candidate.json`, and
`dka_physionet_presentation_only_ltc_comparison.json`.

`dka_buffer_hypothesis_audit.json` is an aggregate simulator-integrity report.
It contains no patient rows or identifiers. It tests whether DKABody no-action
trajectories are too volatile or under-buffered versus PhysioNet 2019
action-unobserved ICU dynamics. The current result does not justify changing
runtime buffer constants; it points instead to missing observed
treatment/recovery dynamics.

The JEPA checkpoints were trained on the DKA simulator. The residual adapter was
fitted on the local MIMIC-IV demo after patient-cross-fitted evaluation; it does
not contain raw rows, but it has no untouched external validation set.

## Safety boundary

All artifacts are research-only and are not clinically validated. The residual
adapter may correct factual forecasts under an observed treatment sequence. It
must not be used to claim causal intervention effects. JEPA-generated rules stay
in the candidate sandbox and cannot automatically modify active Osler rules.
