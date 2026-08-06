# eICU Demo DKA Transition Findings

## Scope

This phase turns the eICU demo treatment audit into a DKA-like transition cohort:

```bash
python eicu_dka_transition_extract.py \
  --data-root physionet.org/files/eicu-crd-demo/2.0.1 \
  --output eicu_dka_transitions_6h_demo.parquet \
  --report eicu_dka_transition_report.json
```

The contract remains:

```text
state_t + history_action_grid + future_action_grid + state_t+6h
```

The important correction is that the numeric action grid now uses only
defensible `infusionDrug` administrations. eICU `medication` rows are orders,
not administrations, and `treatment` rows are coarse treatment-presence text.
Those rows are retained only as treatment evidence, not as dose.

This is an observational factual dataset. It is not a causal treatment-effect
estimator and it does not infer hidden dose from future outcomes.

## Cohort

| Metric | Value |
|---|---:|
| DKA-like stays before evaluable-transition filtering | 304 |
| Evaluable stays | 259 |
| 6-hour transitions | 2,762 |
| Active-DKA transitions | 367 |
| Rows with any factual comparison | 1,188 |
| Rows with active-DKA comparison | 233 |

Core target-pair support:

| Target | Paired observations |
|---|---:|
| Glucose | 1,951 |
| Potassium | 637 |
| Bicarbonate | 677 |
| pH | 164 |
| MAP | 2,468 |

## Numeric Action Support

| Action -> target | Stays | Transitions | Active-DKA stays | Active-DKA transitions |
|---|---:|---:|---:|---:|
| Insulin any -> glucose | 27 | 177 | 21 | 34 |
| Insulin IV -> glucose | 27 | 177 | 21 | 34 |
| Rapid/SC insulin -> glucose | 0 | 0 | 0 | 0 |
| Fluids -> MAP | 8 | 32 | 3 | 3 |
| KCl -> potassium | 0 | 0 | 0 | 0 |
| Bicarbonate -> HCO3 | 0 | 0 | 0 | 0 |
| Dextrose -> glucose | 2 | 11 | 1 | 1 |

Using prior MIMIC-demo power thresholds only as a rough reference, eICU demo no
longer clears any numeric action-target stay gate after the order/admin
separation:

| Gate | Passes |
|---|---|
| Fluids -> MAP | false |
| Insulin -> glucose | false |
| KCl -> potassium | false |

This explicitly withdraws the previous stronger claim that the eICU demo cleared
the `fluids -> MAP` and `insulin -> glucose` reference gates. That claim depended
on treating medication orders as administered therapy.

Numeric action extraction quality is now narrow but clean:

| Metric | Value |
|---|---:|
| Normalized dose-grid rows | 1,713 |
| Sources | `eicu_infusiondrug` only |
| Exact timing fraction | 1.0 |
| High-confidence dose fraction | 1.0 |

## Treatment Evidence Layer

The project now keeps treatment presence separate from numeric dose:

| Evidence metric | Value |
|---|---:|
| Treatment evidence rows | 21,782 |
| eICU medication-order rows | 13,035 |
| eICU infusion evidence rows | 2,844 |
| eICU treatment-text rows | 5,903 |
| Active-DKA windows with insulin evidence | 216 |
| Active-DKA windows with insulin evidence but no numeric dose | 181 |

This explains the treatment-recovery audit: many windows look untreated to the
simulator only because the demo lacks dose-resolved administration for insulin.
They are not permission to weaken untreated DKA physiology.

## Factual Proxy Results

### v5 checkpoint

`dka_symbolic_jepa_v5.pt` still loses active-DKA glucose against persistence. It
keeps point wins for bicarbonate, potassium, and MAP:

| Active-DKA target | JEPA MAE | Persistence MAE | Result |
|---|---:|---:|---|
| Glucose | 132.6278 | 92.5628 | loses |
| Bicarbonate | 3.8799 | 4.0555 | wins |
| Potassium | 0.5655 | 0.6061 | wins |
| MAP | 13.6850 | 14.4279 | wins |

### PhysioNet presentation-only candidate

`dka_physionet_presentation_only_candidate.pt` keeps the complementary split:
it wins active-DKA glucose, bicarbonate, and MAP, but loses potassium.

| Active-DKA target | JEPA MAE | Persistence MAE | Result |
|---|---:|---:|---|
| Glucose | 81.1007 | 92.5628 | wins |
| Bicarbonate | 3.9780 | 4.0555 | wins |
| Potassium | 0.7726 | 0.6061 | loses |
| MAP | 13.3696 | 14.4279 | wins |

No checkpoint is promoted. These are point estimates from an observational,
underpowered demo cohort.

## Treatment-Recovery Audit

`dka_treatment_recovery_audit.py` now distinguishes:

- numeric dose captured;
- treatment presence evidence only;
- no insulin evidence.

On active-DKA windows:

| Stratum | Rows | Stays |
|---|---:|---:|
| Numeric insulin dose captured | 35 | 22 |
| Insulin evidence only | 181 | 104 |
| No insulin evidence | 151 | 83 |

The narrow numeric-dose stratum remains physiologically encouraging:

| Target | Rows | Real change/hr | Sim change/hr | Direction agreement | Simulator MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 34 | -24.50 | -13.38 | 79.4% | 94.76 | 150.03 |
| HCO3 | 19 | +0.98 | +1.32 | 89.5% | 5.56 | 5.96 |
| Anion gap | 16 | -1.31 | -0.78 | 62.5% | 5.35 | 8.38 |
| Potassium | 18 | -0.060 | -0.017 | 66.7% | 0.71 | 0.81 |

The treatment-recovery gate still blocks runtime changes:

```text
runtime_change_allowed: false
automatic_parameter_fit_allowed: false
checkpoint_promotion_allowed: false
```

The sourced neutral protocol check still passes: glucose falls `66.07 mg/dL/hr`
under IV insulin 6 U/hr + isotonic fluid + KCl, inside the fixed `50-75 mg/dL/hr`
research boundary.

## Per-Target Ensemble Follow-Up

The patient-held-out per-target selector was rerun on the stricter cohort:

| Metric | Value |
|---|---:|
| Held-out active-DKA stays | 46 |
| Held-out active-DKA target rows | 478 |
| Persistence normalized MAE | 0.3493 |
| v5 normalized MAE | 0.5393 |
| Presentation-only normalized MAE | 0.8132 |
| Per-target ensemble normalized MAE | 0.3458 |
| Patient-bootstrap delta vs persistence | 0.0190 |
| 95% CI | [-0.0215, 0.0639] |

The ensemble remains non-promotable. Its active-DKA normalized MAE is slightly
lower row-wise, but the patient-level bootstrap does not pass and the mean
patient delta is wrong-signed.

Held-out active-DKA glucose remains the useful narrow signal:

| Target | Selected method | JEPA/ensemble MAE | Persistence MAE | Bootstrap delta 95% CI |
|---|---|---:|---:|---|
| Glucose | presentation-only | 66.6420 | 85.7361 | [-29.7683, 0.3322] |
| Bicarbonate | v5 | 4.1580 | 3.8521 | [-0.7890, 0.9319] |
| Potassium | v5 | 0.6367 | 0.5848 | [-0.2255, 0.1747] |
| MAP | presentation-only | 14.7929 | 14.0978 | [-1.0142, 4.3201] |

Even the held-out oracle upper bound is not significant
(`delta = -0.0056`, 95% CI `[-0.0292, 0.0197]`). The bottleneck is still
administration-grade dose support, not selector cleverness.

## Symbolic Real Test

The patient-held-out eICU symbolic test also shrinks after order/admin
separation:

| Metric | Value |
|---|---:|
| Discovery stays | 174 |
| Held-out stays | 85 |
| Discovery records | 16 |
| Held-out records | 4 |
| Factual direction accuracy | 0.5294 |
| Changed-only direction accuracy | 0.5806 |
| Mean proposal confidence | 0.9272 |
| Candidate rules proposed | 8 |
| Retrospectively validated | 2 |
| Active-rule conflicts | 0 |
| Automatically promoted to active rules | 0 |

The validated rules remain candidate-only and require human review. They are
observational associations, not causal claims.

## Interpretation

The eICU demo remains valuable, but its role is now narrower and more honest:

1. It validates the cross-hospital schema and evidence layer.
2. It proves the treatment-capture problem is real: 181 active-DKA windows have
   insulin evidence but no dose-resolved numeric administration.
3. It does not pass numeric power gates after removing medication-order leakage.
4. It still shows a narrow positive physiology signal when numeric insulin dose
   is genuinely captured.
5. Full eICU remains high ROI because the adapter, evidence layer, persistence
   gate, and symbolic sandbox now run end to end.

## Safety Boundary

No clinical, causal, or counterfactual treatment claim is allowed from this
phase. Medication orders and treatment text are never used as numeric dose.
The committed reports contain aggregate metrics only. The local parquet cohort
and candidate-rule files with stay-level identifiers remain ignored.
