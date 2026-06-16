# Osler: Symbolic Engine + Numerical JEPA

This repository now contains two intentional systems:

1. `demo/` + `engine/`: the deterministic symbolic drug-recommendation demo.
2. `dka_body.py` + `dka_world_model.py` + `dka_osler.py`: the numerical,
   action-conditioned JEPA world model and its symbolic safety shield.

The former generic CTM/MIMIC JEPA path was removed. It used large trajectory
datasets and a 118-state vocabulary, but its latent representation collapsed and
its future predictor did not beat persistence reliably.

The repository was reorganized (see structure below). The recommendation logic is
100% symbolic and deterministic; an LLM is used only to parse free-text cases and to
explain results in chat — it never makes a medical decision.

> ⚠️ **Decision support / research demo only — not clinically validated.** The drug and
> disease `delta` values are hand-authored estimates; `data/demo_clinical_data.json` is
> illustrative, not real FDA labels. Direction (↑/↓) is reliable; magnitudes are soft.
> A licensed clinician makes the final call. See `demo/README.md`.

## Quick start

```bat
py -m pip install -r requirements.txt
py demo\demo_app.py
```

Open http://127.0.0.1:5000 . Paste an API key in the top-right box to enable the chat /
free-text parsing (OpenAI by default, or Gemini); without a key it still runs with
rule-based parsing + the full reasoning graph.

## Repository structure

```
Osler/
├── demo/        The product: web UI + agent orchestrator.
│               demo_app.py (Flask), agent.py, case_parser.py, case_targets.py,
│               disease_world.py, llm_client.py, case_demo.html, sample_cases.json,
│               demo_clinical_data.json, README.md
│
├── engine/      The live symbolic pharmacology engine the demo depends on (7 modules):
│               reasoning_engine.py, patient_profile.py, drug_safety_gate.py,
│               drug_identity.py, clinical_role.py, drug_profile.py, clinical_data.py
│
├── data/        Live knowledge files only: drugs_pkpd.json and retained organ models.
│
├── dka_body.py              Numerical DKA physiology simulator.
├── dka_world_model.py       Action-conditioned JEPA world model + MPC.
├── train_intervention_jepa.py  Branched counterfactual multi-horizon trainer.
├── predict_dka_intervention.py Compare a proposed intervention with no treatment.
├── dka_osler.py             Symbolic shield, mechanism checks, reasoning trace.
├── dka_causal_evaluation.py Matched-control and AIPW confounding audit.
├── osler_jepa/              Shared ontology, action schema, curriculum, validator.
├── physionet2019_pretrain.py  General ICU self-supervised JEPA pretraining.
├── rules/active/dka_embodied.pl  Human-owned Prolog action/effect rules.
├── dka_*.py                 Calibration and real-data validation utilities.
├── SYSTEM_FLOW.md           Current symbolic and numerical JEPA flows.
├── OSLER_JEPA_ROADMAP.md    Target architecture, promotion gates, next build order.
├── SYMBOLIC_JEPA_VIABILITY_AUDIT.md  Same-scale compiler/viability ablation.
├── requirements.txt
└── README.md
```

### How it fits together (live path)

```
demo/demo_app.py ──uses──▶ demo/agent.py
        │                      ├─ case_parser   (free-text → fields; LLM or rules)
        │                      ├─ case_targets   (indication → treatment targets)
        │                      ├─ engine/reasoning_engine.recommend()   ◀── data/drugs_pkpd.json
        │                      └─ disease_world  (indication → disease model ◀── data/<organ>.json)
        └──chat──▶ llm_client (OpenAI / Gemini), grounded in the engine result
```

`demo/demo_app.py` puts `engine/` on `sys.path`; the engine reads from `data/`.
The removed legacy diagnosis prototype was not part of this live path.

## Numerical JEPA

```bash
python train_intervention_jepa.py
python predict_dka_intervention.py --checkpoint dka_intervention_jepa_v4.pt \
  --insulin 6 --insulin-formulation iv --fluids 500 --kcl 10
python dka_osler.py

# Real-data rebuild and factual evaluation
MIMIC_DIR=/path/to/mimic-iv OUT_PATH=dka_transitions_6h.parquet \
  python dka_transition_extract.py
python evaluate_dka_mimic.py --mimic dka_transitions_6h.parquet

# Symbolic-grounded training and patient-held-out rule proposal test
python train_intervention_jepa.py --checkpoint dka_symbolic_jepa_v5.pt \
  --report dka_symbolic_jepa_v5_report.json
python symbolic_real_test.py --checkpoint dka_symbolic_jepa_v5.pt \
  --mimic dka_transitions_6h_demo_v4.parquet

# No-new-data domain adaptation, calibration, and abstention experiment
python real_world_improvement.py --checkpoint dka_symbolic_jepa_v5.pt \
  --mimic dka_transitions_6h_demo_v4.parquet \
  --output dka_real_world_hybrid_v2_report.json \
  --artifact dka_real_world_adapter_v2.joblib \
  --predictions dka_real_world_oof_predictions_v2.csv

# Compile observed dose/route support and train the candidate grey-box residual
python build_action_prior.py dka_transitions_6h_demo_v4.parquet \
  --output dka_action_prior.json
python train_greybox_residual.py dka_transitions_6h_demo_v4.parquet

# Quantify whether any factual forecaster has enough stay-level power to beat
# persistence; negative deltas are required before an N estimate is meaningful.
python real_world_power_analysis.py \
  --oof-predictions dka_real_world_oof_predictions_v2.csv \
  --cohort dka_transitions_6h_demo_v4.parquet

# Turn persistence into the runtime anchor. Candidate residuals may leave it
# only when the direction is Prolog-supported and ensemble agreement is high.
python anchored_residual_hybrid.py \
  --checkpoint dka_symbolic_jepa_v5.pt \
  --mimic dka_transitions_6h_demo_v4.parquet

# Feed the candidate grey-box residual back into the synthetic simulator used to
# train JEPA. This is a candidate research path, not a promotion shortcut.
python train_intervention_jepa.py \
  --greybox-residual dka_greybox_residual_candidate_v1.pt \
  --action-prior dka_action_prior_demo_v1.json \
  --mimic dka_transitions_6h_demo_v4.parquet \
  --checkpoint dka_symbolic_jepa_greybox_candidate.pt

# Produce a research-only what-if artifact that compares protocols inside the
# grey-box simulator instead of scoring factual forecasts against persistence.
python counterfactual_shadow_demo.py \
  --trajectory-jsonl dka_fidelity_demo_v4.jsonl \
  --greybox-residual dka_greybox_residual_candidate_v1.pt

# Trace real-survived simulated deaths to the hard mechanism that failed first.
python dka_viability_falsification_audit.py \
  /tmp/dka_maintenance_trajectories.jsonl \
  --output dka_viability_falsification_audit_v4.json

# Audit numerical floors/ceilings separately from physiology.
python dka_numeric_artifact_audit.py \
  /tmp/dka_maintenance_trajectories.jsonl \
  --output dka_numeric_artifact_audit_v1.json

# Candidate-only integration: repaired DKABody + grey-box residual -> JEPA.
python train_intervention_jepa.py \
  --scenarios 200 --sequence-length 8 --epochs 8 --batch-size 10 \
  --checkpoint dka_symbolic_jepa_v6_candidate.pt \
  --report dka_symbolic_jepa_v6_candidate_report.json \
  --greybox-residual dka_greybox_residual_candidate_v1.pt \
  --action-prior dka_action_prior_demo_v1.json \
  --mimic dka_transitions_6h_demo_v4.parquet
python symbolic_real_test.py \
  --checkpoint dka_symbolic_jepa_v6_candidate.pt \
  --mimic dka_transitions_6h_demo_v4.parquet \
  --output dka_symbolic_real_test_v6_candidate.json
python train_intervention_jepa.py \
  --scenarios 1000 --sequence-length 12 --epochs 55 --batch-size 12 \
  --checkpoint dka_symbolic_jepa_v6_full_candidate.pt \
  --report dka_symbolic_jepa_v6_full_candidate_report.json \
  --greybox-residual dka_greybox_residual_candidate_v1.pt \
  --action-prior dka_action_prior_demo_v1.json \
  --mimic dka_transitions_6h_demo_v4.parquet
python symbolic_real_test.py \
  --checkpoint dka_symbolic_jepa_v6_full_candidate.pt \
  --mimic dka_transitions_6h_demo_v4.parquet \
  --output dka_symbolic_real_test_v6_full_candidate.json

# Evaluate where persistence should become weaker
python long_horizon_real_test.py trajectories.jsonl \
  --checkpoint dka_symbolic_jepa_v5.pt --horizons 6,12,24
```

### PhysioNet 2019 ICU pretraining

The PhysioNet/CinC Challenge 2019 PSV files can now be used for a separate
general ICU state-dynamics JEPA:

```bash
python physionet2019_pretrain.py \
  --data-root /Users/chunyouchang/mimic/physionet.org/files/challenge-2019/1.0.0/training \
  --rebuild-cache \
  --max-transitions-per-split 100000 \
  --epochs 5 \
  --batch-size 1024 \
  --cache physionet2019_cache.npz \
  --checkpoint physionet2019_icu_jepa.pt \
  --report physionet2019_icu_jepa_report.json
```

This path learns from hourly ICU vitals/labs, explicit observation masks,
measurement ages, static demographics, irregular time deltas, and future
`SepsisLabel`. It is intentionally not wired into the DKA intervention model
because the Challenge 2019 data has no explicit DKA treatment action channels
such as insulin route, KCl, fluids, bicarbonate, or dextrose. Its allowed role is
generic ICU encoder/world-model pretraining and representation research. It may
not replace `dka_symbolic_jepa_v5.pt` or support counterfactual treatment claims.

This JEPA reasons over 15 continuous physiological variables: glucose, pH,
bicarbonate, anion gap, potassium, MAP, volume, insulin, sodium, effective
osmolality, creatinine, urine output, beta-hydroxybutyrate, latent total-body
potassium reserve, and cumulative hyperosmolar injury. Its eight action channels
separate IV, rapid-SC, intermediate/NPH, and basal insulin, followed by fluids,
KCl, bicarbonate, and dextrose. A 16-feature history encoder summarizes exposure
and recency over the preceding six hours. Training branches the same patient into
11 interventions, follows a four-stage curriculum, learns one-step through
six-hour open-loop outcomes, and checks counterfactual effects, Osler mechanism
consistency, action sensitivity, risk calibration, and latent collapse.

Before neural encoding, `osler_jepa/state_compiler.py` converts each partial DKA
observation into the same grounded facts consumed by active Prolog plus signed
numeric residuals from physiologic reference ranges. Missing measurements cannot
assert a fact or residual. A zero-initialized compiler encoder conditions the
symbolic direction, proof, status, and rule-proposal heads without entering the
continuous dynamics path. Old checkpoints and numerical predictions therefore
retain their legacy behavior.

Training now exposes a decomposed grounded homeostatic objective: future-state
truth, physiologic-burden fidelity, intervention-effect sensitivity,
counterfactual burden ordering, calibrated per-state uncertainty, and the
existing Prolog contradiction penalty. The viability terms remain reported
research audits with zero default dynamics weight because same-scale retraining
regressed on the MIMIC persistence gate. Viability reward is computed only from
the observed or simulated outcome. The model is therefore penalized for an
optimistic but false recovery forecast; prediction surprise is a learning signal,
not evidence that a treatment is clinically beneficial.

The v5 runtime also has an explicit partial-observation contract. Every state is
encoded as a value, observed/missing mask, and measurement age in hours. Rollouts
accept irregular time intervals rather than assuming every event is exactly 30
minutes apart. Training applies 25% observation dropout while retaining the full
future state as the supervision target. A predict-update
`PotassiumStoreBelief` tracks the latent total-body potassium reserve, its
uncertainty, documented KCl replacement, and estimated renal loss. The estimate
is always labeled as a belief rather than a measured lab.

Existing checkpoints remain loadable because the new observation-context encoder
is zero-initialized. Complete fresh observations therefore preserve their legacy
behavior. A newly trained checkpoint is required before claiming improved
performance from masks or measurement ages.

Treatment timing is no longer represented only as a rate grid. MIMIC extraction
now emits exact aggregate `start` and `stop` lifecycle records plus aligned event
grids for every route-aware action. A zero-initialized event encoder adds those
signals to JEPA dynamics while preserving old rate-only checkpoint behavior.
`ingredientevents` now adds observed free water, oral intake, enteral nutrition,
parenteral nutrition, and explicitly measured carbohydrate as a separate
maintenance context grid. Calories are never reverse-engineered into glucose and
unknown nutrition composition is not injected into the physiology.

`osler_jepa/action_prior.py` compiles route probabilities, active-cell
probabilities, and positive-dose quantiles from the extracted cohort. Randomized
simulator branches and warm-up care can sample this prior instead of the old
uniform hand-written dose grid. Explicit mechanism stress-test protocols remain,
but their doses are clipped to observed support when a prior is supplied.

`osler_jepa/greybox_residual.py` implements a small universal differential
equation candidate around `DKABody`. The residual may correct only glucose,
ketone, bicarbonate, serum potassium, sodium, and creatinine derivatives.
Total-body potassium, volume balance, insulin depots, dose mass balance, and
osmotic injury remain hard-owned by the mechanism and cannot be written by the
network. `train_greybox_residual.py` uses nested patient-group validation and
always emits a candidate-only artifact with no causal or promotion authority.
`train_intervention_jepa.py --greybox-residual ...` can then regenerate
synthetic JEPA branches from `DKABody + residual`, closing the grey-box feedback
loop. Reports must still pass the held-out persistence gate before any checkpoint
can replace `dka_symbolic_jepa_v5.pt`.

`real_world_power_analysis.py` is the promotion reality check. It averages
errors within ICU stay, computes paired stay-level deltas versus persistence,
and estimates the number of stays needed only when the observed candidate is
already better than persistence. A positive delta is reported as wrong-signed,
not underpowered.

`osler_jepa/anchored_residual.py` changes the runtime shape of layer-two factual
forecasting. Persistence becomes the anchor rather than the opponent. A candidate
forecast is converted into a residual, clipped, and shrunk. The residual is
applied only when an active Prolog-derived transition rule supports the same
direction and the candidate source has sufficient agreement. Prolog decides
whether the model may leave the anchor and which direction is explainable; it
does not estimate the numeric effect size. `anchored_residual_hybrid.py` evaluates
this contract on patient-held-out rows.

`counterfactual_shadow_demo.py` reframes JEPA/simulator value away from short
factual forecasting. It emits a research-only what-if contract with explicit
`decision_authority: false`, `clinical_dose_claim_allowed: false`, and
`causal_claim_allowed: false`. The artifact is for explanation, safety shielding,
and planning-simulator experiments; it does not enter live Osler ranking.

`dka_viability_falsification_audit.py` treats each real trajectory that survives
past simulated death as a direct falsification of the simulator's hard viability
mechanism. The maintenance-enriched demo replay produced 11/16 simulated deaths;
all 11 had later real observations. Six were owned by cumulative hyperosmolar
injury and five by potassium-mass failure. Those equations, not the grey-box
residual write set, are the next structural repair targets.

`DKABody` now applies that audit without fitting the 16 replay cases. Serum
potassium follows a pH-corrected equilibrium plus insulin-driven intracellular
shift, while `K_store` remains conserved except for true KCl intake and renal
loss. Cumulative osmotic injury and extreme hyperglycemia are reported burden
signals, not independent terminal death switches. The post-repair audit
(`dka_viability_falsification_audit_v4.json`) reduces simulated deaths from
11/16 to 4/16, removes osmotic-injury deaths, and reduces hypokalemia deaths from
5 to 1. The v3 replay also bounds glucose/diuresis/volume blow-up with effective
distribution volumes and renal/volume-guarded urine flow. Three of the four
remaining deaths are coverage-limited or not falsified by later observations, so
the simulator is better localized but not clinically calibrated.

`dka_numeric_artifact_audit.py` keeps numerical artifacts separate from
physiology. On the maintenance replay it found no remaining `V`, `HCO3`, `Ke`, or
`Cr` floor hits. The remaining notable bound hit is `Na` ceiling, which feeds a
non-terminal osmotic burden with a clipped hourly sodium rate.

`dka_symbolic_jepa_v6_candidate.pt` is a candidate-only integration checkpoint
trained from the repaired simulator plus the candidate grey-box residual. It
improves simulator-held-out counterfactual effect-sign accuracy (`0.9388` at six
steps) and has active 48/48 latent dimensions, but it still loses persistence on
most MIMIC factual targets. It does not replace `dka_symbolic_jepa_v5.pt`.

The full-budget v6 experiment repeats v5's 1,000-scenario, 55-epoch training
budget. It also remains candidate-only. `dka_v5_vs_v6_full_candidate_comparison.json`
records the closing decision: v6 full does not promote over v5 because it still
fails the external persistence gate and is worse than v5 on glucose, HCO3,
osmolality, creatinine, sodium, and anion gap. The runtime/shadow checkpoint
remains `dka_symbolic_jepa_v5.pt`.

Acute viability failures now use reversible severity-by-duration burdens rather
than instant death at the first threshold crossing. Hyperosmolar injury remains a
separate cumulative process. Every training report includes protocol mortality,
causes, response quantiles, and mechanistic direction gates. These are simulator
sanity checks, not clinical mortality calibration.

`DKABody` now samples patient-level weight, renal reserve, insulin sensitivity,
counter-regulatory drive, fluid retention, vascular tone, potassium stores, and
endogenous insulin. Counterfactual branches may begin at presentation or after up
to six hours of prior simulated care. These parameters are broad research priors,
not fitted clinical constants.

The architectural boundary is intentional: JEPA predicts intervention-conditioned
future physiology; Osler validates each final treatment effect as `verified`,
`contradicted`, or `unexplained`, explains the mechanism, and retains safety veto authority.
See `OSLER_JEPA_ROADMAP.md` for the self-learning loop and clinical promotion gates.

The v5 symbolic interface adds four supervised heads: future direction,
Osler transition status, proof-path selection, and intervention rule proposal.
`symbolic_schema.json` is their shared vocabulary. Automated proposals can only
be written to `rules/candidate/`; `rules/active/` is immutable during JEPA runs.
`symbolic_real_test.py` splits MIMIC by patient stay before discovery and held-out
testing. Passing that gate means retrospective reproducibility only, never a
causal claim or automatic promotion into the live symbolic engine.

`osler_jepa/embodied_logic.py` closes the embodied reasoning loop around JEPA.
It converts patient state and interventions into grounded facts, executes the
active Prolog-compatible rules, emits action preconditions and required
co-interventions, validates predicted effect directions, and returns recursive
proof trees. JEPA remains the continuous world model; Prolog remains the final
logical explanation and veto layer.

Differentiable direction constraints are compiled directly from
`rules/active/dka_embodied.pl`. Each trainable `expected/4` rule has a fixed-point
`training_constraint/3` declaration, so training and runtime reasoning cannot
silently drift into separate hand-maintained rule tables.
`temporal_constraint/4` adds an effect window and confidence to the same active
rule source. Constraints and proof labels are inactive outside their declared
window, and confidence weights differentiable penalties.

`dka_causal_evaluation.py` defines six-hour target-trial diagnostics, patient-stay
cross-fitted AIPW, propensity-caliper matched controls, balance, overlap, and
clustered bootstrap intervals. It always emits `causal_claim_allowed: false`.
On the current 12-stay demo, treatment assignment is not aligned to time zero,
concurrent treatment is common, and matched balance remains inadequate.

The live demo can run JEPA as a read-only DKA shadow observer with
`OSLER_JEPA_SHADOW=1`. The observer runs only when the indication and required
state fields match the DKA contract and the symbolic safety gate already allows a
mapped candidate. It writes results under `jepa_shadow`, verifies that the live
recommendation fingerprint did not change, and carries no dosing, ranking, veto,
or causal authority. `OSLER_JEPA_SHADOW_LOG` optionally records JSONL audits.
Later measured physiology can be reconciled with a stored forecast using
`jepa_shadow_outcome.py`. Scoring is allowed only when the observed treatment
start/stop schedule, mean exposure, and elapsed time match the forecast contract.
It reports state-wise JEPA
error versus persistence and changed-state direction accuracy, but never performs
an online weight update, automatic rule promotion, or causal attribution. If a
terminal outcome is supplied, the predicted death risk is scored with a Brier
score; it is not interpreted as calibrated from a single case.

```bash
python jepa_shadow_outcome.py \
  --ledger /private/path/shadow.jsonl \
  --event-id FORECAST_EVENT_ID \
  --future-json future_observation.json \
  --actual-action-json actual_treatment_schedule.json \
  --elapsed-hours 6
```

The action JSON must contain `{"schedule": [{"hours": 0, "action": {...}}]}`.
Shadow ledgers contain physiological observations, remain local, and are ignored
by Git through the repository-wide `*.jsonl` rule.

For patient-grouped evaluation, set a local secret and a local subject key before
reconciliation. The raw subject key is never written; the ledger stores only an
HMAC-SHA256 group hash.

```bash
export OSLER_JEPA_LEDGER_SALT='local-secret-not-in-git'
export OSLER_JEPA_SUBJECT_KEY='local-ehr-subject-key'
python jepa_shadow_outcome.py ...
python jepa_shadow_cohort.py --ledger /private/path/shadow.jsonl
```

The cohort report averages repeated episodes within each subject before running a
patient-cluster bootstrap. Its automated retrospective gate requires at least 30
independent subjects, 50 scored episodes, core-state coverage, changed-state
direction accuracy, no symbolic disagreement, and a 95% confidence interval above
persistence both overall and separately for glucose, potassium, bicarbonate, and
MAP. Passing still does not authorize clinical promotion or online learning.

`real_world_improvement.py` tests episode and propensity weighting, a small
real-world residual adapter, patient-bootstrap ensembles, temperature scaling,
and abstention without adding data. All reported predictions are patient-level
out-of-fold. The generated `dka_real_world_adapter_v1.joblib` is exposed through
`osler_jepa/real_world_adapter.py` only for factual forecasts under an observed
treatment sequence; it is intentionally blocked from causal counterfactual use.
The v2 artifact exposes a per-state selection contract. A learned state requires
at least three of four outer folds to select the same method and lower out-of-fold
MAE than persistence. Unsupported horizons, low ensemble agreement, unstable
state selection, and unvalidated states explicitly abstain to persistence.
See `REAL_WORLD_IMPROVEMENT_FINDINGS.md` for the ablation results.

The June 2026 v4 checkpoint was trained on 1,000 patient scenarios. At six hours
its held-out simulator MAE is 36.56 mg/dL for glucose, 1.53 mmol/L for bicarbonate,
0.187 mmol/L for potassium, 2.78 mOsm/kg for effective osmolality, 3.41 units for
the latent K store, and 0.484 for hyperosmolar injury. Counterfactual effect-sign
accuracy is 96.22%; effective latent rank is 12.484 and the model is not collapsed.

This is still not a clinically validated model. The MIMIC-IV demo was rebuilt with
exact 30-minute treatment grids from `inputevents` plus `emar`/`emar_detail`, six
hours of pre-anchor history, rolling urine output, and route/formulation-aware
insulin. It produced 187 transitions from 12 DKA stays. Active-DKA glucose MAE
improved from v3 `204.55` to v4 `138.46 mg/dL`, but persistence remains better at
`96.65`. V4 beats persistence for active-DKA anion gap and MAP, not for the other
targets. Route-aware DKABody replay improves glucose MAE from `235.32` to `195.45`
but still produces 12/12 simulated deaths. The checkpoint remains research-only.
See `MIMIC_DEMO_FINDINGS.md` and `dka_mimic_demo_v4_evaluation.json`.
