# Synthetic RCT Pretraining Real-Transfer Findings

Date: 2026-07-11

## Boundary

The synthetic RCT checkpoint was trained only to recover simulator truth from
`synthetic_rct_factory.py`. It is fictional data, not patient evidence. This
audit asks one narrow question:

> Does the synthetic-RCT-trained DKA WorldModel transfer to real factual DKA
> forecasting or improve the existing per-target router?

Answer: no.

No clinical, causal, counterfactual, runtime, active-rule, or checkpoint
promotion authority is implied.

## Artifacts

Synthetic training:

- script: `train_synthetic_rct_jepa.py`
- checkpoint: `/private/tmp/osler_synthetic_rct_jepa_candidate.pt`
- synthetic training report: `/private/tmp/osler_synthetic_rct_jepa_candidate_report.json`

Real transfer audits:

- MIMIC-IV full factual eval: `/private/tmp/osler_synthetic_rct_jepa_mimiciv_full_eval.json`
- eICU factual eval: `/private/tmp/osler_synthetic_rct_jepa_eicu_eval.json`
- eICU router gate: `/private/tmp/osler_synthetic_rct_jepa_eicu_router_gate.json`
- MIMIC-IV router robustness: `/private/tmp/osler_synthetic_rct_jepa_mimiciv_router_robustness.json`

## Synthetic Held-Out Result

On 100,000 fictional patients / 600,000 synthetic transitions, the model learned
the generator response surface:

- best validation normalized MSE: `0.0030553`
- factual normalized MSE: `0.0030417`
- shuffled-action normalized MSE: `0.0065140`
- changed-only direction accuracy: `0.94904`
- latent effective rank: `10.342`
- low-rank warning: `false`

Interpretation: the architecture can learn this synthetic response surface.
That is a simulator-recovery result only.

## Real Factual Transfer

### MIMIC-IV full DKA

Single-checkpoint active-DKA MAE:

| target | synthetic checkpoint | persistence |
| --- | ---: | ---: |
| glucose | 167.3607 | 143.3735 |
| bicarbonate | 7.0965 | 3.5697 |
| anion_gap | 8.3091 | 5.0238 |
| potassium | 1.6257 | 0.5951 |
| MAP | 15.6594 | 11.5229 |
| creatinine | 0.9774 | 0.2651 |
| urine_output | 119.3836 | 81.3003 |

Result: the synthetic checkpoint loses to persistence on the real MIMIC-IV DKA
factual task.

### eICU DKA

Single-checkpoint active-DKA MAE:

| target | synthetic checkpoint | persistence |
| --- | ---: | ---: |
| glucose | 189.3611 | 92.5628 |
| bicarbonate | 9.5424 | 4.0555 |
| anion_gap | 12.5438 | 5.0031 |
| potassium | 1.5749 | 0.6061 |
| MAP | 14.0471 | 14.4279 |
| creatinine | 0.9569 | 0.2299 |
| urine_output | 86.9693 | 35.5080 |

MAP is the only point-estimate near-win. It does not survive router validation.

## Router Gate

### eICU

When the synthetic checkpoint is offered as the second model source in the
per-target router:

- active-DKA normalized MAE:
  - persistence: `0.349300`
  - v5: `0.539276`
  - synthetic checkpoint: `0.720866`
  - per-target ensemble: `0.371563`
- ensemble delta vs persistence: `+0.065831`
- bootstrap 95% CI: `[+0.005189, +0.134982]`

Positive delta means worse than persistence. The synthetic source harms the
router.

### MIMIC-IV full

Across seven patient-held-out splits:

- active-DKA beats persistence: `1/7`
- active-DKA significant wins: `0/7`
- median active-DKA delta: `0.0`
- all-window beats persistence: `1/7`
- all-window significant wins: `0/7`

Time-order split:

- active-DKA point delta: `-0.000437`
- bootstrap 95% CI: `[-0.003729, +0.002659]`

This is not robust and does not justify adding the synthetic checkpoint to the
router.

## Conclusion

The synthetic RCT data are useful for:

- estimator and pipeline tests;
- simulator-truth recovery checks;
- power and data-contract experiments;
- checking whether the architecture can represent a known response surface.

They are not useful as a direct pretraining source for the real DKA factual
router in the current form.

The synthetic checkpoint should remain local-only and research-only:

- do not promote it;
- do not use it as a router source;
- do not use it for clinical or causal claims;
- do not treat synthetic oracle recovery as external validation.

The next real improvement path remains real observational forecasting inputs
and validated belief/whole-body observation layers, or externally identified
randomized evidence for causal use.
