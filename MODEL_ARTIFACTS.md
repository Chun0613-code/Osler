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

`eicu_aki_renal_belief_state_audit.json` is the explicit predict-update version
of the renal belief audit. It is aggregate-only. The state belief passes
downstream gates for BUN at 24h/48h and for urine output at 24h, but it does not
pass for creatinine. This artifact validates a narrower online-compatible belief
state and preserves the same safety boundary.

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
