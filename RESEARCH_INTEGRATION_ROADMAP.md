# Research Integration Roadmap

This document converts the current research reading list into an executable
Osler-JEPA integration plan. It is not a literature review. A method is kept
only if it can attach to the existing symbolic engine, numerical DKA world model,
or safety-gated self-learning loop.

## Current Anchors

The project already has the interfaces needed for a neuro-symbolic research
program:

| Role | Existing module | Current behavior |
|---|---|---|
| Human-owned symbolic rules | `rules/active/dka_embodied.pl` | Active Prolog-style rules with temporal windows and confidence. |
| Differentiable symbolic constraints | `osler_jepa/validator.py` | Compiles active rules into training penalties and runtime effect checks. |
| Candidate rule discovery | `osler_jepa/rule_inducer.py` | Aggregates JEPA symbolic heads into observational candidate rules. |
| Candidate sandbox | `osler_jepa/rule_sandbox.py` | Allows automated writes only under `rules/candidate/`. |
| Numerical world model | `dka_world_model.py` | Action-conditioned JEPA with partial observations and irregular action intervals. |
| Belief state | `osler_jepa/belief.py` | Predict-update belief over latent total-body potassium reserve. |
| Planning and control | `dka_world_model.py`, `predict_dka_intervention.py`, `dka_osler.py` | MPC-style candidate simulation under Osler veto authority. |

The right integration principle is:

```text
symbolic engine = rule authority and explanation
JEPA / world model = continuous patient dynamics
belief filters = hidden-state inference
candidate sandbox = slow, reviewable self-learning
```

## Strategic Positioning

This roadmap is infrastructure work to run in parallel with credentialed data
access. It is not a substitute for dose/time-resolved real DKA cohorts.

The current bottleneck remains treated real-world dynamics. Prior audits showed
that the Prolog-to-JEPA interface is not the limiting factor: better symbolic
grounding can make rule proposals cleaner, but it cannot promote a checkpoint
from the same small, confounded demo cohort. Therefore:

- LTC and generalized belief filters are the first no-new-data experiments
  because they touch dynamics and hidden-state inference.
- ILP, LNN, NDRE, and DreamCoder are kept as rule-discovery infrastructure to
  harden the system while credentialing proceeds.
- None of these methods can bypass persistence, counterfactual, symbolic, or
  causal-readiness gates.

## Keep: Directly Useful Methods

### Liquid Time-Constant Networks

**Attach to:** `dka_world_model.py`, `train_intervention_jepa.py`

LTCs are the most actionable numerical-model candidate because the project
already has irregular intervals, measurement ages, and partial observations. It
also touches the dynamics path directly, unlike rule-discovery infrastructure.

First implementation target:

- Implemented entry point: `WorldModel(dynamics_cell="ltc")` and
  `train_intervention_jepa.py --dynamics-cell ltc`.
- Apples-to-apples evaluation must lock the current keeper calibration:
  - baseline: PhysioNet presentation/profile priors with the measurement model
    disabled, using `residual_mlp`;
  - candidate: the exact same presentation-only setup, using `ltc`;
  - only the dynamics cell may change.
- Feed it elapsed time, observation mask, measurement age, and action embedding.
- Compare against the presentation-only residual-MLP baseline with the same
  promotion gates: latent rank, counterfactual sign accuracy, action
  sensitivity, changed-only accuracy, and persistence.

Promotion boundary:

- LTC is a candidate dynamics module, not a shortcut around real-data gates.
- If it improves glucose by averaging while rank/counterfactual metrics regress,
  reject it like the full PhysioNet measurement model.

### Generalized Predict-Update Belief Filters

**Attach to:** `osler_jepa/belief.py`, `dka_world_model.py`,
`train_intervention_jepa.py`

The potassium-store belief filter is the current miniature version of Osler's
individualized hidden-state inference. This deserves its own build item because
it is closest to the long-term vision: infer what the patient cannot directly
state from labs, actions, missingness, and physiology.

First implementation target:

- Implemented entry point: `infer_hidden_beliefs(state)` returns typed
  research-only beliefs, and `downstream_observable_gate(...)` validates each
  unmeasured belief by held-out improvement on measurable downstream variables.
- The downstream gate requires a capacity-matched placebo belief. The candidate
  belief must beat both the no-belief baseline and the placebo, so improvement
  cannot be credited merely to added model capacity.
- Candidate hidden states:
  - acid-base buffer reserve,
  - insulin sensitivity / effective insulin action,
  - sodium-water balance,
  - renal reserve / filtration state.
- Each filter must report mean, uncertainty, source, and whether it is measured
  or inferred.
- JEPA may consume belief means and uncertainty, but Osler explanations must
  preserve provenance.

Promotion boundary:

- Hidden-state beliefs are research estimates, not measured labs.
- A belief filter may improve personalization and counterfactual simulation.
- It may not authorize a treatment or override observed patient values.
- A hidden belief is kept only if it improves held-out prediction of measurable
  downstream targets over both baseline and a capacity-matched placebo. Direct
  hidden-state accuracy claims are forbidden because these states are not
  directly observed.

### ILP / FOIL

**Attach to:** `osler_jepa/rule_inducer.py`

The current rule inducer is a thresholded observational aggregator. ILP and FOIL
turn that into a principled relational rule learner:

```text
positive examples: action exposure + observed changed-state direction
negative examples: same action/state pair without the direction
background facts: Prolog patient/action/state facts, time window, cointerventions
output: candidate clauses only, never active rules
```

First implementation target:

- Add a FOIL-style candidate scorer that explicitly tracks positives,
  negatives, cointerventions, and patient-held-out support.
- Emit rules at `level = observational_association` until causal-grade evidence
  exists.
- Preserve the current active-rule conflict check.

Promotion boundary:

- ILP may improve candidate discovery.
- ILP must not write to `rules/active/`.
- ILP must not claim treatment causality from confounded EHR records.
- On the current demo cohort, ILP is expected to produce cleaner candidates, not
  a nonzero promotion count.

### Logical Neural Networks

**Attach to:** `osler_jepa/validator.py`

The validator already does the key LNN move: human-owned logic becomes
differentiable training constraints. LNN should be used to make the constraint
compiler more expressive, not to replace Prolog ownership.

First implementation target:

- Keep `rules/active/dka_embodied.pl` as the single source of truth.
- Add richer differentiable forms for conjunction, negation, confidence, and
  temporal windows.
- Regression-test every compiled constraint against the grounded Prolog proof.

Promotion boundary:

- LNN constraints may shape JEPA training.
- A neuralized rule may not become an active safety rule without human review.
- LNN improves alignment and auditability; it is not expected to solve the
  treated-dynamics bottleneck by itself.

### NDRE / Neural Logic Machines

**Attach to:** JEPA symbolic heads and `osler_jepa/rule_inducer.py`

These methods are useful for extracting lifted, reusable symbolic rules from a
neural model. In Osler, their correct role is not "let the model explain itself"
but "turn repeated model effects into candidate rules with falsification tests."

First implementation target:

- Convert repeated `(action, state, direction, time_window)` proposals into
  lifted rule templates.
- Require negative examples and patient-held-out support before a candidate is
  marked `awaiting_human_review`.
- Store extracted templates in the candidate sandbox with provenance.

Promotion boundary:

- Extracted rules are review evidence.
- They cannot bypass active Prolog, safety gates, or persistence gates.

### DreamCoder

**Attach to:** candidate-rule sandbox and future rule-library growth

DreamCoder is useful as a disciplined version of "self-growing mechanisms":
learn small reusable symbolic abstractions from repeated candidate patterns.

First implementation target:

- Build a research-only rule-library file under `rules/candidate/`.
- Learn abstractions from rejected and passing candidate rules, not from raw
  clinical decisions.
- Prefer reusable mechanism fragments such as "insulin-like action lowers
  glucose in a 0-6h window" over patient-specific clauses.

Promotion boundary:

- DreamCoder may propose reusable candidate abstractions.
- It must not modify active rules, drug ranking, doses, or safety gates.

### Bayesian Program Learning

**Attach to:** small-N rule learning and simulator priors

BPL is not a drop-in module here. Its useful contribution is the design rule for
the 12-stay / 15-stay regime: strong structure beats unstructured curve fitting.

First implementation target:

- Prefer structured priors, explicit latent variables, and small candidate
  programs over high-capacity black-box adapters.
- Use falsification audits to decide which mechanism is wrong before fitting
  constants.

Promotion boundary:

- BPL is a methodology guide, not a runtime dependency.

### Free Energy Principle / Active Inference

**Attach to:** `osler_jepa/belief.py`, `dka_world_model.py`,
`predict_dka_intervention.py`, `dka_osler.py`

This is the best unifying frame for the project:

```text
generative model        = DKABody + JEPA world model
belief inference        = observed state + masks + K-store predict-update
action selection        = candidate intervention schedules under Osler veto
prediction surprise     = shadow outcome reconciliation and learning signal
expected free energy    = risk + uncertainty + distance from target physiology
```

First implementation target:

- Make the planning objective explicit: expected physiologic risk, expected
  uncertainty, and expected distance from symbolic targets.
- Keep it inside the simulator/shadow/advisory loop until real-data gates pass.

Promotion boundary:

- Active inference may frame planning.
- It does not authorize clinical action without Osler and persistence gates.

### TD Learning

**Attach to:** simulator-internal planning only

TD learning is useful only after the world model is validated enough to support
planning experiments.

Allowed:

- Value estimation inside a validated simulator or candidate digital twin.
- Model-based planning diagnostics.

Forbidden:

- Direct off-policy RL on raw EHR trajectories.
- Learning treatment policies from confounded observational assignments.

## Drop Or Treat As Historical Nutrients

These works contain ideas already absorbed into the current design, but they are
not implementation targets:

| Work | Keep as idea | Do not implement as |
|---|---|---|
| TerpreT | Warning that differentiable program induction can be brittle. | Main rule learner. |
| Wiener / Ashby | Feedback, setpoint, homeostasis. | A vague buffer constant without audit support. |
| Samuel checkers | Historical self-play milestone. | EHR RL. |
| McCarthy common sense | Symbolic-AI framing. | A direct medical reasoning method. |
| EPAM | Cognitive history. | Runtime model. |
| Winston structural descriptions | Near-miss/falsification spirit. | Replacement for ILP. |
| Sussman HACKER | Debug-driven repair spirit. | Automated rule rewriting. |
| Holland classifier systems | Evolutionary rule search. | Primary rule induction path. |

## First Build Order

1. **Run one LTC dynamics candidate.**
   Completed as an apples-to-apples presentation-only experiment: residual MLP
   baseline versus LTC, with calibration fixed and only the dynamics cell
   changed. LTC was not collapsed and slightly improved simulator
   MSE/action-sensitivity, but it failed active-DKA glucose against persistence
   and did not improve 6h counterfactual or external symbolic changed-only
   gates. Keep it rejected unless a future treated cohort changes the evidence.

2. **Generalize predict-update belief filters.**
   Extend `osler_jepa/belief.py` from K-store only to additional hidden states
   such as acid-base buffer reserve, insulin sensitivity, sodium-water balance,
   and renal reserve. Keep every belief explicitly marked as inferred, and keep
   it only if it beats both the no-belief baseline and a capacity-matched
   placebo belief on downstream observable targets.

3. **Make the active-inference objective explicit.**
   Express planning as expected risk + uncertainty + symbolic target distance,
   under Osler veto.

4. **Formalize candidate-rule learning as ILP.**
   Extend `osler_jepa/rule_inducer.py` with explicit positives, negatives,
   cointervention facts, and patient-held-out support.

5. **Upgrade validator constraints in an LNN-compatible way.**
   Preserve Prolog as the source of truth while expanding differentiable
   conjunction, negation, temporal windows, and confidence handling.

6. **Add a candidate rule-library growth experiment.**
   Keep DreamCoder-style abstractions under `rules/candidate/`; no live rule
   modification.

## Non-Negotiable Safety Boundaries

- No automatic writes to `rules/active/`.
- No direct clinical RL from observational EHR.
- No causal treatment claims without time-aligned, dose-resolved, balanced
  cohorts.
- No checkpoint promotion from a single metric such as glucose MAE.
- No runtime use of a candidate that fails latent-rank, counterfactual, symbolic,
  or persistence gates.

## Current Decision

The next practical research move is not to add every classic method. The
continuous-time dynamics candidate has been tested and rejected under fixed
calibration, so the remaining no-new-data work is to validate generalized
hidden-state belief filters while credentialed data access proceeds.
ILP/LNN/DreamCoder then harden the candidate-rule loop so that, once larger
dose/time-resolved cohorts
arrive, the system can discover and quarantine symbolic hypotheses cleanly.
Active Inference should be used as the organizing language for belief,
prediction, surprise, and planning, not as permission to bypass Osler's safety
authority.
