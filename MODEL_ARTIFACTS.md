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

The JEPA checkpoints were trained on the DKA simulator. The residual adapter was
fitted on the local MIMIC-IV demo after patient-cross-fitted evaluation; it does
not contain raw rows, but it has no untouched external validation set.

## Safety boundary

All artifacts are research-only and are not clinically validated. The residual
adapter may correct factual forecasts under an observed treatment sequence. It
must not be used to claim causal intervention effects. JEPA-generated rules stay
in the candidate sandbox and cannot automatically modify active Osler rules.
