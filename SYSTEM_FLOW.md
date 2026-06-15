# Osler System Flow

## Active Drug-Recommendation Path

1. `demo/demo_app.py` receives free text or structured patient fields.
2. `demo/case_parser.py` parses free text with rules or an LLM. Explicit form
   fields override parsed values.
3. `demo/case_targets.py` resolves the indication to a canonical scenario and
   hand-authored physiological treatment targets.
4. `demo/agent.py` builds an `engine/patient_profile.py` object.
5. `PatientProfile.flags()` derives hypotension, tachycardia, hypoxia,
   hyperkalemia, renal-review, hepatic, and pregnancy flags.
6. `demo/disease_world.py` loads the curated disease perturbations from an organ
   JSON. When no curated disease exists, it inverts the treatment targets to
   create an explicitly inferred fallback model.
7. `engine/reasoning_engine.mechanism_candidates()` canonicalizes duplicate drug
   names and maps target/effect variables through the shared Osler state ontology
   before organ and direction matching against `data/drugs_pkpd.json` effects.
8. Matching effects produce a mechanism score and a readable mechanism chain.
9. `engine/drug_safety_gate.evaluate()` checks FDA-label-derived fields against
   allergies, current medications, patient flags, renal status, and dose-rule
   validation status.
10. `engine/clinical_role.py` assigns scenario appropriateness. This currently
    contains special handling for asthma, albuterol, epinephrine, and anaphylaxis.
11. Candidates are ranked first by clinical-role priority and then by mechanism
    score. A mechanism match alone does not authorize dosing.
12. Optional openFDA enrichment reloads clinical label data and reruns the same
    deterministic recommendation path.
13. `demo/agent.build_graph()` emits patient, disease, perturbation, target, and
    drug nodes for the UI mind map.
14. `demo/llm_client.py` may explain the completed result. Its prompt forbids
    changing the ranking, inventing doses, or introducing unmatched drugs.

`demo/orchestrator.py` may let an LLM choose the order of parser/recommender/wiki
tools, but every clinical decision still comes from the symbolic functions.

## Numerical JEPA-to-Osler Path

The control loop is now explicitly neuro-symbolic and embodied: patient state is
the environment observation, an intervention is the action, JEPA predicts the
continuous future state, and `rules/active/dka_embodied.pl` decides whether the
action is allowed, blocked, or requires a co-intervention. The grounded Prolog
engine returns a proof tree for every conclusion and checks JEPA's predicted
effect direction. `osler_jepa/validator.py` compiles the differentiable training
constraints directly from the human-owned Prolog rule pack.

Before state encoding, `observation_context()` creates a 15-variable value vector,
observed mask, and measurement-age vector. Missing normalized values are imputed
at the population mean but cannot masquerade as measurements because the mask and
age are supplied to the zero-compatible observation encoder. JEPA rollouts consume
the actual interval of each action event and accumulate elapsed time. During
runtime, `PotassiumStoreBelief` predicts the hidden reserve from KCl exposure and
estimated renal loss, then updates it with the serum/pH proxy and JEPA estimate.
Osler and Prolog consume the posterior belief and its provenance, not a fabricated
laboratory value.

The treatment stream contains both continuous dose/rate channels and explicit
start/stop lifecycle channels. MIMIC intervals are converted into aggregate
events that handle carried-in infusions and overlapping administrations without
creating false stops. The event encoder is zero-compatible with older checkpoints.

Active effect rules also carry temporal windows and confidence in
`temporal_constraint/4`. The same metadata controls differentiable loss,
symbolic proof supervision, runtime validation, and deferred checks outside the
effect window.

Simulator terminal events use reversible critical burdens for pH, potassium,
MAP, and glucose. Protocol mortality and response quantiles are emitted as
calibration diagnostics. `dka_causal_evaluation.py` is a separate EHR audit path:
it performs grouped AIPW and matched-control analyses but cannot promote rules or
authorize causal intervention claims.

1. `dka_body.py` defines the continuous physiological state and transition rules.
   Each simulated patient has sampled body size, renal reserve, insulin response,
   stress drive, fluid response, vascular tone, potassium store, and endogenous
   insulin parameters.
2. `osler_jepa/ontology.py` maps symbolic and numerical variable names into a
   versioned canonical state vocabulary and derives clinically readable flags.
3. `osler_jepa/actions.py` represents dose, route, start time, duration, and
   provenance. Its neural encoder makes action effects dependent on elapsed time.
4. `train_intervention_jepa.py` samples a patient state at presentation or after
   up to six hours of prior simulated treatment, then clones it into 11
   branches: no treatment, single interventions, combined protocols, dextrose,
   and random
   actions. Scenario-level splitting keeps all branches of a patient in one split.
5. The state encoder maps 15 physiological values to latent state `z(t)`: glucose,
   pH, bicarbonate, anion gap, potassium, MAP, volume, insulin, sodium, effective
   osmolality, creatinine, urine output, beta-hydroxybutyrate, total-body
   potassium reserve, and cumulative hyperosmolar injury.
6. A history encoder summarizes six hours of dose exposure and action recency. The
   temporal action encoder maps eight continuous channels: IV, rapid-SC, NPH, and
   basal insulin plus fluids, KCl, bicarbonate, and dextrose.
7. The residual predictor estimates `z(t+1)` from `z(t)` and the action embedding.
8. The decoder grounds latent predictions back into readable physiology; a risk
   head predicts whether the next state crosses the simulator viability boundary.
9. `osler_jepa/curriculum.py` progressively emphasizes state grounding, disease
   dynamics, treatment effects, and counterfactual validation.
10. One-step JEPA loss, open-loop 12-step loss, paired counterfactual effect loss,
   reconstruction, death-risk loss, VICReg, and Osler consistency train jointly.
11. `osler_jepa/symbolic.py` converts matched transitions into direction,
   verified/contradicted/unexplained, and proof-path supervision.
12. Four JEPA heads predict future direction, Osler status, proof paths, and a
   structured intervention-effect rule proposal. Contradiction and action-
   contrastive penalties prevent the proposal head from ignoring treatment.
13. `osler_jepa/validator.py` labels decoded effects as verified, contradicted, or
   unexplained and supplies readable mechanism provenance.
14. Validation and test sets report one-step, three-hour, and six-hour MAE, shuffled
   action degradation, counterfactual effect sign accuracy, and latent collapse.
15. `dka_symbolic_jepa_v5.pt` stores the symbolic-grounded route-aware checkpoint.
   Model weights are local generated artifacts and are ignored by Git.
16. `osler_jepa/rule_inducer.py` aggregates repeated structured proposals into
   observational candidate rules. `osler_jepa/rule_sandbox.py` permits automated
   writes only under `rules/candidate/`; active rules remain human-owned.
17. `symbolic_real_test.py` splits MIMIC by stay, performs discovery only on the
   first partition, and tests support and direction accuracy on unseen stays.
   Even a passing candidate remains outside the live engine pending provenance,
   safety regression tests, and human review.
18. `real_world_improvement.py` performs patient-cross-fitted domain adaptation
   without adding data. It tests treatment-episode purity, propensity overlap,
   residual correction, state-wise persistence fallback, calibration, and
   ensemble abstention.
19. `osler_jepa/real_world_adapter.py` may correct a factual forecast only when
   the treatment sequence is observed. It emits uncertainty and an abstention
   flag and explicitly forbids causal intervention claims.
20. `dka_intervention_jepa_v4.pt` stores the earlier route-aware checkpoint. It is ignored
   by Git because model weights are local generated artifacts.
21. `predict_dka_intervention.py` compares a proposed action with no treatment,
   uses the same elapsed-time encoding as training, and returns predicted state
   deltas, risk, and Osler transition validation.
22. During prediction, `dka_osler.shield()` re-evaluates the decoded state every
    30 minutes and may change the next action. Each change emits a readable trace.
23. `dka_osler.verify_mechanisms()` verifies nine learned mechanism directions,
    including dextrose raising glucose. The shield adds dextrose when glucose is
    below 250 mg/dL but ketonemia/acidosis still requires insulin, and withholds
    potassium in renal dysfunction or oliguria, and proactively replaces potassium
    when the latent total-body store is depleted.
24. `mimic_action_history.py`, `dka_transition_extract.py`, and
    `dka_fidelity_extract.py` combine `inputevents` with `emar`/`emar_detail`,
    normalize amounts and rates, create exact 30-minute eight-action grids, and
    retain six hours of treatment history before every anchor.
25. `evaluate_dka_mimic.py` loads an existing checkpoint without retraining and
    compares it with persistence on all windows and an active-DKA subset. It also
    reports missingness and action rates outside simulator training support.
26. The rebuilt MIMIC-IV demo has 12 DKA stays and 187 transitions. It confirms
    exact route-aware extraction works. V4 beats persistence for active-DKA anion
    gap and MAP, but not glucose or the remaining targets. DKABody replay still
   produces 12/12 simulated deaths, so real-data fine-tuning remains blocked.
27. `osler_jepa/shadow.py` can attach the DKA JEPA to the completed live Osler
   bundle in read-only shadow mode. It fingerprints protected recommendation
   fields before and after inference, accepts only symbolic-approved candidates,
   and fails closed on missing state, checkpoint, or inference errors. Shadow
   output cannot rerank drugs, authorize a dose, bypass Prolog, or make a causal
   claim. Enable it with `OSLER_JEPA_SHADOW=1`; optionally set
   `OSLER_JEPA_CHECKPOINT`, `OSLER_JEPA_DEVICE`, and `OSLER_JEPA_SHADOW_LOG`.
28. `osler_jepa/shadow_outcomes.py` closes the observational audit loop. A later
   outcome is scored only when its elapsed time, mean action exposure, and exact
   start/stop schedule match the forecast. It compares JEPA with persistence per measured state and
   records changed-state direction accuracy. Reconciliations are review evidence
   only: online learning, causal claims, and automatic rule promotion stay off.

## Highest-Value Improvements

1. Keep numerical JEPA outside real clinical treatment ranking until it beats
   patient-held-out baselines on a much larger dose-and-time-resolved cohort.
2. Extend the new versioned state ontology to all symbolic drug and disease data,
   then validate every variable reference during startup.
3. Move indication targets and clinical-role rules from Python constants into a
   validated knowledge schema with provenance and tests.
4. Require curated disease models for decision paths; inferred inverted targets
   should remain visualization-only.
5. Separate evidence quality, mechanism fit, safety, and clinical-role scores in
   the UI instead of presenting one implied confidence.
6. Add patient-held-out regression tests for every safety flag and indication.
7. Extend the numerical state/action interface one disease module at a time,
   keeping Osler's veto and explanation authority.
8. Treat unexplained JEPA effects as reviewable rule candidates; never let model
   training silently rewrite Osler's hard safety rules.

See `OSLER_JEPA_ROADMAP.md` for the architecture contract, promotion gates, and
ordered plan for DKA, multi-disease, and real-EHR development.
