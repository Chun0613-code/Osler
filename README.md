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
├── data/        All knowledge/data files (.json, .npz): drugs_pkpd.json (drug deltas),
│               the organ state-models (heart.json, lung.json, …), and the rest of the
│               legacy knowledge base.
│
├── dka_body.py              Numerical DKA physiology simulator.
├── dka_world_model.py       Action-conditioned JEPA world model + MPC.
├── train_intervention_jepa.py  Branched counterfactual multi-horizon trainer.
├── predict_dka_intervention.py Compare a proposed intervention with no treatment.
├── dka_osler.py             Symbolic shield, mechanism checks, reasoning trace.
├── osler_jepa/              Shared ontology, action schema, curriculum, validator.
├── dka_*.py                 Calibration and real-data validation utilities.
├── SYSTEM_FLOW.md           Current symbolic and numerical JEPA flows.
├── OSLER_JEPA_ROADMAP.md    Target architecture, promotion gates, next build order.
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
  --mimic dka_transitions_6h_demo_v4.parquet
```

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

`real_world_improvement.py` tests episode and propensity weighting, a small
real-world residual adapter, patient-bootstrap ensembles, temperature scaling,
and abstention without adding data. All reported predictions are patient-level
out-of-fold. The generated `dka_real_world_adapter_v1.joblib` is exposed through
`osler_jepa/real_world_adapter.py` only for factual forecasts under an observed
treatment sequence; it is intentionally blocked from causal counterfactual use.
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
