# eICU Demo ICU JEPA Pretraining Findings

## Run

The eICU Collaborative Research Database Demo 2.0.1 CSV tables were adapted into
the existing ICU JEPA feature contract used by `physionet2019_pretrain.py`.

- Script: `eicu_demo_pretrain.py`
- Data root: `physionet.org/files/eicu-crd-demo/2.0.1`
- Cache: `eicu_demo_cache.npz`
- Checkpoint: `eicu_demo_icu_jepa.pt`
- Report: `eicu_demo_icu_jepa_report.json`
- Horizon: 6 hours
- Max stay window: first 168 ICU hours
- Transition cap: 30,000 per split
- Epochs: 8

The adapter maps eICU periodic vitals, aperiodic blood pressure, and labs into
the 34 dynamic features already used by the generic ICU JEPA.  This generic
pretraining run does not consume the eICU treatment tables.  The eICU demo has
no PhysioNet Challenge 2019 sepsis label, so the auxiliary binary head uses a
hospital/unit discharge mortality proxy.  That proxy is only an auxiliary demo
signal, not a clinical mortality model.

## Dataset Profile

- Patient table stays: 2,520
- Stays with usable observations: 2,428
- Median hourly sequence length among used stays: 67 hours
- Stay-level mortality proxy positive rate: 0.086491
- All 34 dynamic features had at least some future targets.

Split profile:

| Split | Transitions | Future Target Density | Mortality Proxy Rate |
|---|---:|---:|---:|
| Train | 30,000 | 0.117536 | 0.077900 |
| Validation | 27,393 | 0.122276 | 0.072902 |
| Test | 26,578 | 0.116550 | 0.089096 |

## Main Result

The model trained successfully and the latent space did not collapse.

- Test masked state MSE: `0.567451`
- Mean latent std: `0.220599`
- Minimum latent std: `0.162054`
- Effective rank: `19.851`
- Active latent dimensions: `48 / 48`
- Collapsed: `false`

The auxiliary mortality-proxy head reached:

- Test BCE: `0.296497`
- Rank AUC estimate: `0.763223`

This AUC is useful only as an internal representation sanity check.  It should
not be reported as validated mortality prediction.

## Persistence Comparison

On the held-out test split, the JEPA beat persistence on 10 of 34 dynamic
features:

`O2Sat`, `SBP`, `MAP`, `DBP`, `Resp`, `BaseExcess`, `SaO2`, `Glucose`,
`Magnesium`, `Potassium`.

Selected clinically relevant rows:

| Feature | JEPA MAE | Persistence MAE | Observed Targets | Result |
|---|---:|---:|---:|---|
| Glucose | 33.6122 | 38.8259 | 3,959 | JEPA wins |
| Potassium | 0.3638 | 0.3917 | 1,384 | JEPA wins |
| MAP | 9.6950 | 10.3461 | 13,586 | JEPA wins |
| O2Sat | 1.7884 | 1.8360 | 13,962 | JEPA wins |
| SBP | 13.5895 | 14.2331 | 13,560 | JEPA wins |
| DBP | 8.5263 | 9.1898 | 13,559 | JEPA wins |
| Resp | 2.9475 | 2.9848 | 13,133 | JEPA wins |
| HCO3 | 2.4942 | 2.3982 | 1,409 | persistence wins |
| pH | 0.0644 | 0.0613 | 301 | persistence wins |
| Creatinine | 0.5157 | 0.3718 | 1,249 | persistence wins |
| BUN | 7.8597 | 6.7053 | 1,237 | persistence wins |
| Lactate | 1.6007 | 1.3621 | 114 | persistence wins |
| Platelets | 40.8734 | 33.0069 | 1,083 | persistence wins |
| WBC | 3.7941 | 2.7843 | 1,073 | persistence wins |

## Interpretation

This is a useful real-data smoke result.  The same JEPA architecture can learn
non-collapsed ICU dynamics from the eICU demo and beats persistence on several
dense physiology variables, including glucose, potassium, MAP, and blood
pressure.

It is not yet a strong replacement checkpoint.  Sparse labs and slower-moving
targets still favor persistence, and the eICU demo is much smaller than the full
eICU release.

## Boundary

Allowed use:

- Generic ICU state encoder pretraining experiments.
- Testing the eICU table-to-hourly-state adapter.
- Comparing dense physiology forecasting against persistence.

Disallowed use:

- Replacing the DKA intervention checkpoint.
- Claiming treatment effect, counterfactual, or action-conditioned validity from
  the generic pretraining run.
- Claiming validated sepsis or mortality prediction.

## Treatment Table Audit

`eicu_demo_action_audit.py` confirms that the demo does contain treatment/action
tables that can be mapped into the DKA action contract:

- `medication.csv.gz`: start/stop offsets, dosage text, route.
- `infusiondrug.csv.gz`: time-stamped infusion rates/amounts.
- `treatment.csv.gz`: coarse treatment/procedure text.

The aggregate audit found:

| Scope | Rows | Stays |
|---|---:|---:|
| All mapped DKA-action rows | 24,935 | 1,958 |
| DKA-like stays with any mapped action | 6,755 | 280 |
| DKA-like anchor window, -6h to +24h | 3,635 | 265 |

The DKA-like cohort definition produced 304 stays: 178 lab-only, 38
diagnosis-only, and 88 supported by both diagnosis text/code and labs.

Window-level action coverage:

| Action | Rows | Stays |
|---|---:|---:|
| Fluids | 1,533 | 249 |
| Dextrose | 711 | 208 |
| Insulin IV | 752 | 98 |
| Insulin rapid/SC | 367 | 173 |
| KCl | 155 | 81 |
| Insulin basal/SC | 46 | 41 |
| Bicarbonate | 71 | 33 |

Interpretation:

> This is the first open, cross-hospital dataset in the project that has both
> ICU physiology and DKA-relevant treatment tables. The current output is only an
> aggregate coverage audit, not an action-conditioned model. It makes the next
> build concrete: construct an eICU DKA transition extractor with observed
> treatment grids and run it as a second real treated cohort beside MIMIC demo.

The next useful run is an eICU DKA transition extractor that emits the same
state/action/future-state contract as `dka_transition_extract.py`, followed by a
small action-conditioned real-proxy evaluation. Full eICU should then reuse the
same adapter at larger scale.
