# eICU Demo DKA Transition Findings

## Scope

This phase turns the eICU demo treatment coverage audit into an
observed-treatment DKA-like transition cohort:

```bash
python eicu_dka_transition_extract.py \
  --data-root physionet.org/files/eicu-crd-demo/2.0.1 \
  --output eicu_dka_transitions_6h_demo.parquet \
  --report eicu_dka_transition_report.json
```

The extractor emits the same factual contract used by the MIMIC proxy evaluator:

```text
state_t + history_action_grid + future_action_grid + state_t+6h
```

It is an observational factual dataset. It is not a causal treatment-effect
estimator, and it does not infer hidden treatment from future outcomes.

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

## Observed Treatment Support

| Action -> target | Stays | Transitions | Active-DKA stays | Active-DKA transitions |
|---|---:|---:|---:|---:|
| Insulin any -> glucose | 79 | 362 | 56 | 89 |
| Insulin IV -> glucose | 65 | 331 | 49 | 80 |
| Rapid/SC insulin -> glucose | 14 | 21 | 6 | 7 |
| Fluids -> MAP | 152 | 1,345 | 78 | 124 |
| KCl -> potassium | 63 | 139 | 33 | 44 |
| Bicarbonate -> HCO3 | 12 | 28 | 5 | 9 |
| Dextrose -> glucose | 129 | 884 | 75 | 122 |

Using prior MIMIC-demo power thresholds as a rough reference, eICU demo now
clears the stay-count gate for `fluids -> MAP` and barely clears it for
`insulin_any -> glucose`. `KCl -> potassium` remains below the previous
77-stay glucose-style reference after target-pair filtering. eICU effect sizes
may differ, so this is a power proxy, not a proof.

Action extraction quality is high for the demo files:

| Metric | Value |
|---|---:|
| Normalized event rows | 11,868 |
| Exact timing fraction | 0.9304 |
| High-confidence dose fraction | 1.0 |

## Factual Proxy Results

### v5 checkpoint

`dka_symbolic_jepa_v5.pt` still loses active-DKA glucose against persistence, but
it wins active-DKA bicarbonate, potassium, and MAP:

| Active-DKA target | JEPA MAE | Persistence MAE | Result |
|---|---:|---:|---|
| Glucose | 129.7614 | 92.5628 | loses |
| Bicarbonate | 3.6135 | 4.0555 | wins |
| Potassium | 0.5723 | 0.6061 | wins |
| MAP | 13.6500 | 14.4279 | wins |

### PhysioNet presentation-only candidate

`dka_physionet_presentation_only_candidate.pt` shows the opposite split: it wins
active-DKA glucose and MAP, but loses potassium and bicarbonate:

| Active-DKA target | JEPA MAE | Persistence MAE | Result |
|---|---:|---:|---|
| Glucose | 77.3352 | 92.5628 | wins |
| Bicarbonate | 4.0831 | 4.0555 | loses |
| Potassium | 0.7524 | 0.6061 | loses |
| MAP | 13.0738 | 14.4279 | wins |

The split matters. eICU demo contains enough open treatment data to expose real
signals, but it does not justify replacing v5. The two candidate histories win
different physiology slices, so this phase closes with `no_checkpoint_promotion`.

## Per-Target Ensemble Follow-Up

The complementary split suggested a per-target selector. The follow-up script
tests that idea without outcome leakage:

```bash
python eicu_dka_per_target_ensemble.py \
  --cohort eicu_dka_transitions_6h_demo.parquet \
  --v5-checkpoint dka_symbolic_jepa_v5.pt \
  --presentation-checkpoint dka_physionet_presentation_only_candidate.pt \
  --output eicu_dka_per_target_ensemble.json
```

The selector is trained only on discovery active-DKA patient stays, then scored
on held-out patient stays. It selected the expected physiology split:

| Target | Selected method |
|---|---|
| Glucose | presentation-only |
| Bicarbonate | v5 |
| Potassium | v5 |
| MAP | presentation-only |
| Other sparse/slow targets | persistence |

Held-out active-DKA results were more cautious:

| Metric | Value |
|---|---:|
| Held-out active-DKA stays | 46 |
| Held-out active-DKA target rows | 478 |
| Persistence normalized MAE | 0.3493 |
| v5 normalized MAE | 0.5372 |
| Presentation-only normalized MAE | 0.8066 |
| Per-target ensemble normalized MAE | 0.3444 |
| Patient-bootstrap delta vs persistence | 0.0038 |
| 95% CI | [-0.0320, 0.0391] |

The row-weighted normalized MAE is slightly better than persistence, but the
patient-level bootstrap does not pass. This means the ensemble is not robust
enough for promotion on the demo split.

The useful signal is narrower: held-out active-DKA glucose still favors the
presentation-only candidate, but it remains underpowered:

| Target | Selected method | JEPA/ensemble MAE | Persistence MAE | Bootstrap delta 95% CI |
|---|---|---:|---:|---|
| Glucose | presentation-only | 63.3789 | 85.7361 | [-37.8228, 3.5490] |
| Bicarbonate | v5 | 4.2422 | 3.8521 | [-0.8336, 1.0154] |
| Potassium | v5 | 0.6708 | 0.5848 | [-0.1736, 0.2908] |
| MAP | presentation-only | 14.4946 | 14.0978 | [-1.2033, 3.3786] |

Even the held-out oracle upper bound is not significant
(`delta = -0.0149`, 95% CI `[-0.0438, 0.0140]`). That is the key sobriety check:
the complementarity is real enough to study, but eICU demo is still too small
and heterogeneous to prove a deployed selector.

## Symbolic Real Test

The patient-held-out eICU symbolic test correctly labels candidate rules as
coming from the eICU demo cohort. Results:

| Metric | Value |
|---|---:|
| Discovery stays | 174 |
| Held-out stays | 85 |
| Discovery records | 50 |
| Held-out records | 30 |
| Factual direction accuracy | 0.5348 |
| Changed-only direction accuracy | 0.5706 |
| Mean proposal confidence | 0.9022 |
| Candidate rules proposed | 25 |
| Retrospectively validated | 6 |
| Active-rule conflicts | 0 |
| Automatically promoted to active rules | 0 |

The six retrospectively validated rules remain candidate-only and require human
review. They are observational associations, not causal claims.

## Interpretation

This is the first open, cross-hospital, treatment-bearing DKA-like cohort in the
project that is large enough to test the old power bottleneck. It changes the
state of the project:

1. The credentialing wall is no longer the only way to test treated DKA dynamics.
2. eICU demo has enough support for the `fluids -> MAP` and `insulin -> glucose`
   factual proxy gates.
3. Per-target model complementarity appears, but it does not yet pass a
   patient-held-out bootstrap gate.
4. The result is still observational, confounded, and DKA-like rather than
   diagnosis-pure DKA.
5. Full eICU should be high ROI because the adapter, action grid, persistence
   gate, and symbolic sandbox now all run end to end.

## Safety Boundary

No clinical, causal, or counterfactual treatment claim is allowed from this
phase. The committed reports contain aggregate metrics only. The local parquet
cohort and candidate-rule files with stay-level identifiers remain ignored.
