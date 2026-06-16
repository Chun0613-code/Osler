# Grey-Box and Safe Hybrid Audit

Date: 2026-06-15

## Existing EHR Action Recovery

The strict MIMIC-IV demo extraction now includes `ingredientevents` maintenance
context in addition to `inputevents` and eMAR:

- 15 usable stays and 252 six-hour transitions
- 399 maintenance events in the strict DKA cohort
- free water, oral intake, enteral nutrition, parenteral nutrition, and explicit
  carbohydrate have separate channels
- calories are not converted to carbohydrate
- unknown nutrition composition remains context-only

The demo contained no safely attributable TPN/enteral carbohydrate mass in these
DKA windows. Replaying the richer trajectories therefore remained 11/16 simulated
deaths with essentially unchanged MAE. This rejects the assumption that the demo
alone contains enough hidden maintenance treatment to explain the residual gap.

## Empirical Action Prior

`dka_action_prior_demo_v1.json` compiles the observed route and dose support.
Randomized simulator branches and warm-up treatment now use empirical active
probabilities and positive-dose quantiles. Insulin route sampling is mutually
exclusive. Supplied protocol doses are clipped to the observed 99th percentile.

This prior is a demo-domain support constraint, not a clinical dosing policy.

## Constrained Grey-Box Residual

The candidate residual ODE has 286 parameters and can correct only concentration
derivatives. It cannot directly modify total-body potassium, fluid volume,
insulin PK depots, administered-dose mass balance, or osmotic injury.

Nested patient-group cross-fit results at six hours:

| State | Mechanism MAE | Grey-box MAE |
| --- | ---: | ---: |
| Glucose | 323.50 | 195.24 |
| Ketone pool | 8.62 | 6.04 |
| HCO3 | 7.43 | 5.83 |
| Serum K | 0.80 | 0.71 |
| Sodium | 8.28 | 4.87 |
| Creatinine | 1.03 | 0.95 |

The first experimental run incorrectly used the outer test fold for early
stopping and was discarded. The reported result uses an inner patient-group
validation split and a sealed outer test fold.

Despite consistent improvement, the artifact remains `candidate_only` and
`promotion_allowed: false` because only 15 stays are available and treatment
confounding remains unresolved.

## Safe Hybrid Contract

The hybrid v2 runtime now reports, per state:

- candidate source and prediction
- selected source and prediction
- fallback source
- ensemble direction agreement
- abstention decision and reason
- requested and supported horizon

Promotion requires at least three of four outer folds to select the same
non-persistence method and lower out-of-fold MAE than persistence. No state passed
both conditions. MAP was the closest candidate, but only two folds selected the
adapter and its all-window safe-hybrid MAE was slightly worse than persistence.
The generated v2 artifact therefore selects persistence for every state.

Overall safe-hybrid versus persistence normalized MAE difference was +0.00398
with 95% bootstrap interval [-0.00231, 0.01212]. Active-DKA hybrid was worse.

## Long-Horizon Check

The factual 6/12/24-hour test found:

- MAP beat persistence at 6h and 12h with 15 stays, but lost at 24h.
- Potassium beat persistence at 12h.
- Anion gap and potassium beat persistence at 24h, but each had only three
  measured targets and cannot support promotion.
- Glucose, sodium, creatinine, and urine output remained worse than persistence.

Longer horizon is therefore useful as an evaluation axis, but it is not evidence
that the current JEPA generally beats persistence.

## Persistence Power Analysis

`real_world_power_analysis.py` uses ICU stay as the unit of evidence. Transition
rows are averaged within stay before computing paired deltas versus persistence.
The report is saved as `dka_persistence_power_analysis_v1.json`.

Key results:

- Base JEPA is wrong-signed overall: normalized stay-level delta +0.2786.
- Ensemble adapter is wrong-signed overall: +0.1208.
- Safe hybrid is effectively persistence and slightly wrong-signed all-window:
  +0.0086.
- Active-only safe hybrid has a tiny favorable mean delta (-0.0045) but would
  require roughly 976 stays at the observed effect/variance.
- Base JEPA MAP is the only clear per-state favorable signal with a plausible
  power number: delta -0.0356, estimated 42 stays required.
- Active-safe glucose shows a small favorable delta (-0.0228), estimated 77 stays
  required, but it did not pass the broader product gate.
- Grey-box residual remains wrong-signed versus persistence on all residual
  states. For glucose, grey-box improved mechanism simulation but still had
  positive normalized delta versus persistence (+0.5064).

This confirms that the current bottleneck is no longer an obvious missing model
hook. In the demo cohort, factual promotion is limited by wrong-signed dense
targets, small stay count, and observational treatment confounding.

## Grey-Box JEPA Feedback Loop

`train_intervention_jepa.py` now accepts `--greybox-residual`. When supplied,
synthetic branches are generated from `DKABody + candidate residual` while the
same action prior and symbolic losses remain active. The checkpoint metadata
records the residual schema and marks it candidate-only.

A smoke candidate was trained with 80 scenarios, sequence length 8, and 3 epochs:
`dka_symbolic_jepa_greybox_smoke_v1.pt`. It proves the closed loop runs end to
end, but it is not a replacement for v5. Its external MIMIC factual proxy still
lost persistence on dense targets, including glucose 353.03 vs 43.50 and sodium
6.26 vs 2.01. The current production/shadow checkpoint therefore remains v5.

## Counterfactual Shadow Contract

`counterfactual_shadow_demo.py` emits `dka_counterfactual_shadow_demo_v1.json`.
It compares fixed research protocols against a no-treatment simulation baseline
inside the candidate grey-box simulator. It explicitly sets:

- `uses_persistence_as_judge: false`
- `decision_authority: false`
- `affects_live_recommendation: false`
- `clinical_dose_claim_allowed: false`
- `causal_claim_allowed: false`

This is the right product framing for the current evidence level: counterfactual
explanation, safety shielding, and planning-simulator research rather than a
short-horizon factual forecaster that claims to beat persistence.

## Persistence-Anchored Residual Gate

`osler_jepa/anchored_residual.py` implements the layer-two architecture that
turns persistence from an opponent into the safety anchor:

```text
selected = persistence + shrink * clipped(candidate - persistence)
```

The residual can be non-zero only when:

- the candidate residual is larger than the per-state stability threshold,
- an active Prolog-derived transition rule supports the same direction,
- the candidate source has sufficient direction agreement,
- the horizon matches the supported contract.

Prolog's role is permission, direction, explanation, and safety. It does not
try to improve numeric magnitude.

Patient-held-out result in `dka_anchored_residual_hybrid_v1.json`:

- `base_jepa` was allowed to leave persistence 0/889 times because it has no
  ensemble agreement signal. This reduces its normalized MAE from 0.5639 back to
  persistence at 0.2670.
- `ensemble_adapter` was allowed to leave persistence only 15/889 times
  (1.69%). Its normalized MAE moved from 0.3959 to 0.2674, essentially the
  persistence safety floor at 0.2670.
- Dense-core normalized MAE moved from persistence 0.2207 to anchored ensemble
  0.2203, a tiny signal but not promotion evidence.
- Most abstentions were due to `no_active_prolog_direction`; this is expected
  because the gate is intentionally narrow and temporal.

This architecture succeeds at preventing self-confident wrong departures. It
does not prove factual superiority over persistence and therefore remains a
research guard, not a promoted forecaster.

## Viability Falsification Audit

`dka_viability_falsification_audit.py` replays real trajectories and treats every
case with later observed measurements after simulated death as a falsification of
the owning viability equation. On `/tmp/dka_maintenance_trajectories.jsonl`, the
audit produced `dka_viability_falsification_audit_v1.json`:

- 16 replay trajectories
- 11 simulated deaths
- 11/11 deaths falsified by later real observations
- Death causes: 6 cumulative hyperosmolar injury, 5 hypokalemia
- Hard-mechanism owners: 6 `osmotic_injury`, 5 `potassium_mass`
- First threshold crossings: 7 hypokalemia, 4 cumulative hyperosmolar injury

This matters because the constrained grey-box residual can write glucose, ketone,
bicarbonate, serum potassium, sodium, and creatinine derivatives only. It is
intentionally forbidden from writing total-body potassium, volume/MAP balance,
insulin depots, dose mass balance, or cumulative osmotic injury. The residual can
therefore improve visible concentrations while still leaving the fatal replay
error in untouched hard mechanisms.

Repair backlog:

- `osmotic_injury`: separate instantaneous osmolality from injury burden, audit
  sodium/free-water balance, and use real-survived high-osmolality stays as
  negative death labels for the injury threshold.
- `potassium_mass`: trace serum potassium collapse into renal loss, insulin and
  acidosis shift, KCl retention, and patient-level total-store initialization.
- `volume_map`: remains on the owner list for future falsifications even though
  this demo replay's first deaths were osmotic and potassium failures.

The next simulator work should repair these owning equations directly instead of
expanding the grey-box residual write set.

## Potassium-First Mechanism Repair

The v1 audit had an important ordering clue:

| Event | Hypokalemia | Cumulative hyperosmolar injury |
| --- | ---: | ---: |
| First threshold crossing | 7 | 4 |
| Final death cause | 5 | 6 |

Potassium crossed first more often, while osmotic injury killed slightly more
often. That makes potassium the likely front-edge failure rather than just a
parallel backlog item.

The repair deliberately does not fit the 16 replay trajectories. Instead,
`DKABody` now encodes sourced physiology:

- serum K is pH-corrected because DKA acidemia can mask total-body depletion;
- insulin shifts K intracellularly but weakens near critical hypokalemia;
- total-body K store changes only through KCl intake and renal K loss;
- cumulative osmotic injury is a reported burden, not an uncalibrated death
  trigger;
- extreme hyperglycemia is also a reported burden, with terminal failure routed
  through acidosis, MAP/volume, potassium, or another sourced organ-failure
  mechanism.

The first post-repair replay artifact is `dka_viability_falsification_audit_v2.json`:

- 16 replay trajectories
- 7 simulated deaths, down from 11
- 7/7 deaths falsified by later real observations
- 0 cumulative hyperosmolar-injury deaths, down from 6
- 1 hypokalemia death, down from 5
- Terminal hard-mechanism owners: 5 `potassium_mass`, 2 `volume_map`
- Remaining death causes: 4 hyperkalemia, 1 hypokalemia, 2 circulatory collapse

This is not a promotion result. It is a localization result. The potassium repair
removed most low-K false deaths and exposed a downstream volume/glucose blow-up
regime. The next owning equations are volume/MAP transfer, glucose
production/clearance under captured treatment, and the remaining potassium
mass/shift behavior. The audit should remain a regression test, but making all
16 cases survive is not the success criterion.

Mechanism-source notes:

- [MSD Manual Professional](https://www.msdmanuals.com/professional/endocrine-and-metabolic-disorders/diabetes-mellitus-and-hypoglycemia/acute-complications-of-diabetes-mellitus)
  notes that DKA hyperglycemia causes osmotic diuresis with fluid/electrolyte
  loss; potassium may be normal or elevated despite a total-body deficit;
  insulin drives potassium into cells; and insulin is held when serum potassium
  is below 3.3 mEq/L.
- NCBI Bookshelf's [Hyperkalemia](https://www.ncbi.nlm.nih.gov/books/NBK470284/)
  and [Hypokalemia](https://www.ncbi.nlm.nih.gov/books/NBK482465/) reviews
  describe predominantly intracellular potassium stores, insulin-driven
  intracellular shift, increased renal K excretion with urine flow, and the
  unreliability of serum potassium as a total body store measurement.

## Glucose/Volume Coupling Repair

The v2 audit exposed a second non-physiologic feedback loop: glucose rose into
the thousands, extracellular volume collapsed toward the numerical floor, and
serum potassium or MAP then became the terminal label. That was not a new
clinical death mechanism; it was a simplified concentration model amplifying
solute inputs through an unrealistically tiny distribution volume.

The v3 repair adds architecture-level safeguards:

- glucose, dextrose, bicarbonate, serum K, and dilution use effective
  distribution volumes rather than `V=1 L` at the lower numerical bound;
- osmotic diuresis is capped by renal perfusion and current volume reserve;
- extreme hyperglycemia remains a burden signal, with terminal failure routed
  through sourced organ-failure mechanisms;
- audit output now flags likely coverage-limited falsifications, such as no
  captured insulin followed by later metabolic improvement, or no captured KCl
  followed by later potassium recovery.

The v3 replay artifact is `dka_viability_falsification_audit_v3.json`:

- 16 replay trajectories
- 4 simulated deaths, down from 11 in v1 and 7 in v2
- 3/4 deaths falsified by later observations
- 2/4 deaths flagged as coverage-limited falsifications
- Death causes: 3 acidosis, 1 hypokalemia
- Hard-mechanism owners: 3 `acid_base`, 1 `potassium_mass`
- Coverage flags: 1 no captured insulin despite later metabolic improvement,
  1 no captured KCl despite later potassium recovery

Fidelity replay moved in the right direction:

| Variable | v2 MAE | v3 MAE |
| --- | ---: | ---: |
| Glucose | 571.74 | 264.39 |
| K | 1.19 | 0.61 |
| MAP | 15.36 | 12.72 |
| Urine output | 937.27 | 198.58 |
| HCO3 | 15.09 | 5.92 |
| Anion gap | 18.64 | 7.53 |

This still does not make JEPA a promoted factual forecaster. It says the
simulator's hard-mechanism failures are now more localized: remaining work is
acid-base calibration on treatment-observed windows, better treatment capture,
and one residual potassium/KCl case.

## Numeric Artifact Audit And Stop Criterion

`dka_numeric_artifact_audit.py` now separates numerical implementation artifacts
from physiology. The audit records each floor/ceiling, whether the clamped state
can enter a denominator or terminal pathway, and whether the bound is hit during
maintenance replay.

`dka_numeric_artifact_audit_v1.json` found:

- no remaining `V.floor`, `HCO3.floor`, `Ke.floor`, or `Cr.floor` hits;
- `Na.ceiling` hit 6 times, but sodium is rate-clipped and only feeds
  non-terminal osmotic burden;
- `osmotic_injury.floor` hits are reporting-floor zeros, not terminal dynamics;
- `counterregulatory_stress` and `renal_perfusion_state` ceiling hits are hidden
  bounded-drive states, not concentration denominators.

The acid-base audit was also tightened. `dka_viability_falsification_audit_v4.json`
keeps the same 4/16 simulated deaths as v3, but 3/4 are now explicitly
coverage-limited:

- 2 acidosis deaths have no captured insulin in a falsified trajectory;
- 1 hypokalemia death has no captured KCl but later potassium recovery;
- the remaining acidosis death has no later observation after simulated death and
  is not a falsification.

That satisfies the demo-data stopping criterion for hard-mechanism repair:
remaining deaths are either coverage-limited or not falsified by later
observations. Further reduction now requires better treatment capture or a larger
cohort, not more hand-tuned equations.

## V6 Candidate Integration

The repaired simulator was fed back into JEPA as a candidate-only integration:

```bash
python train_intervention_jepa.py \
  --scenarios 200 --sequence-length 8 --epochs 8 --batch-size 10 \
  --checkpoint dka_symbolic_jepa_v6_candidate.pt \
  --report dka_symbolic_jepa_v6_candidate_report.json \
  --greybox-residual dka_greybox_residual_candidate_v1.pt \
  --action-prior dka_action_prior_demo_v1.json \
  --mimic dka_transitions_6h_demo_v4.parquet
```

Simulator-held-out candidate metrics:

- six-step counterfactual effect-sign accuracy: `0.9388`
- six-step factual normalized MSE: `0.134703`
- shuffled-action MSE: `0.59499`, so the candidate uses interventions
- Osler verified rate when explainable: `0.9932`
- active latent dimensions: `48/48`

External MIMIC proxy still rejects factual promotion:

- changed-only symbolic factual direction accuracy: `0.6071`
- rules proposed/promoted: `0/0`
- glucose MAE: JEPA `380.96` vs persistence `63.0`
- MAP remains the main factual win: JEPA `11.48` vs persistence `11.74`
- active-DKA anion gap is close: JEPA `3.14` vs persistence `3.25`

Conclusion: v6 is useful as a candidate research artifact for counterfactual and
simulator-side experiments. It does not replace v5 for shadow/advisory runtime,
and it does not change the no-causal-claim boundary.

## Not Executed

eICU/HiRID transfer pretraining was not run because those datasets are not present
in the workspace. Literature-curve calibration was not populated with unsourced
constants. Both can use the same transition/action contracts once a licensed
dataset or an explicitly approved target-curve source is supplied.
