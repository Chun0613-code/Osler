# Synthetic RCT data boundary

`synthetic_rct_factory.py` generates a large, fictional, multi-arm DKA-like RCT for testing estimators, power calculations, transition schemas, and causal-audit code.

It is deliberately not presented as clinical evidence. Every generated row carries `synthetic_rct=true`, `fictional_data=true`, `not_real_patient_data=true`, and `causal_claim_allowed=false`. The metadata and audit report repeat the same boundary.

## What is calibrated

When `--calibration-cohort` points to a local MIMIC transition parquet, the factory uses aggregate moments, quantiles, and observed-rate estimates from the cohort to shape the synthetic presentation and measurement process. It does not copy patient identifiers or emit source rows.

The treatment response is still a transparent simulator prior. It includes bounded, coupled responses for glucose, potassium, bicarbonate, anion gap, pH, sodium, creatinine, MAP, heart rate, lactate, and urine output. The generated oracle columns expose simulator truth so an estimator can be checked honestly.

## What this cannot prove

Passing the randomization-balance audit only shows that the generator randomized its own fictional patients correctly. Recovering an oracle effect only shows that an estimator can recover the simulator's assumptions. Neither result validates a human treatment effect, a clinical policy, a JEPA checkpoint, or an Osler recommendation.

Use real randomized or otherwise identification-valid clinical data for causal claims. Keep the synthetic files out of promotion gates that are meant to establish external or clinical validity.

## Run

```bash
python synthetic_rct_factory.py \
  --n 100000 \
  --calibration-cohort dka_transitions_6h_mimiciv_full_v31_icd.parquet \
  --output /private/tmp/osler_synthetic_rct_dka_100k.parquet \
  --audit /private/tmp/osler_synthetic_rct_dka_100k_audit.json
```
