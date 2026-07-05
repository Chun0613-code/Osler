# MIMIC-IV Waveform Precision Candidate

Status: candidate-only manifest validated. No waveform-derived feature is
validated for factual forecasting yet.

## Why This Exists

The table-based observation layer has reached the point where additional
same-source ICU rows are no longer the main accuracy lever. The remaining
high-precision signal source is MIMIC-IV Waveform: ECG, arterial pressure,
plethysmography, respiration, oxygen saturation, and related high-frequency
signals.

This is a separate modality, so it must enter through its own fail-closed
pipeline rather than being mixed into validated table features.

## Current Implementation

- `osler_jepa/waveform.py`
  - parses WFDB header text;
  - classifies signal labels into coarse physiologic groups;
  - defines candidate waveform feature families;
  - exposes a readiness contract that keeps prediction authority closed.
- `mimiciv_waveform_manifest.py`
  - scans `.hea` files under a waveform root;
  - emits aggregate signal coverage only;
  - does not read raw waveform samples;
  - does not emit row-level records or patient identifiers.
- `osler_jepa/observation_layer.py`
  - reports `mimiciv_waveform_precision_candidate` as candidate-only.

## Candidate Signal Groups

- ECG: beat-to-beat heart rate, RR variability, tachy/brady burden.
- Arterial pressure: beat-to-beat SBP/DBP/MAP, MAP variability, hypotension
  burden, pulse-pressure variability.
- Plethysmography: perfusion/pleth variability and desaturation burden.
- Respiration/capnography: respiratory variability, apnea/low-ventilation
  burden, ETCO2 variability when present.

## Promotion Gate

Before any waveform-derived feature can be used for factual prediction, it must
pass the same discipline as the rest of Osler-JEPA:

- baseline: existing table-based factual forecast;
- candidate: table-based forecast plus waveform-derived features;
- placebo: table-based forecast plus capacity-matched random features;
- validation: patient-heldout plus careunit/time or hospital-heldout where
  available;
- targets: heart rate, MAP/SBP/DBP, oxygen saturation, respiratory rate;
- authority: no causal, counterfactual, clinical, runtime, checkpoint, or active
  rule authority.

## Current Result

The MIMIC-IV Waveform Database local mirror is present at:

```text
physionet.org/files/mimic4wdb/0.1.0
```

Aggregate WFDB header manifest, run 2026-07-05:

| Metric | Count |
|---|---:|
| headers scanned | 4,059 |
| parse errors | 0 |
| ECG coverage | 3,597 |
| respiration coverage | 3,689 |
| pleth coverage | 3,369 |
| arterial pressure coverage | 1,072 |
| central venous pressure coverage | 429 |
| capnography coverage | 4 |

The header parser was also corrected for two WFDB details:

- multi-segment master headers list segment names, not signal labels;
- ECG subchannels such as `ECG #0` are canonicalized to `ECG`.

This means the data source is real and has enough ECG/pleth/respiration coverage
to justify the next engineering step. It still has **no factual prediction
authority** because no raw waveform features or held-out accuracy audit have
been run.

The manifest command is:

```bash
python3 mimiciv_waveform_manifest.py \
  --waveform-root physionet.org/files/mimic4wdb/0.1.0 \
  --output /tmp/mimiciv_waveform_manifest.json
```

## Next Engineering Step

Raw feature extraction has been opened as a bounded candidate pipeline:

- `osler_jepa/waveform_features.py`
  - computes robust channel summaries from bounded waveform windows;
  - emits ECG, arterial pressure, pleth, and respiration candidate features;
  - keeps prediction authority closed.
- `mimiciv_waveform_feature_extract.py`
  - reads local WFDB records through the optional `wfdb` dependency;
  - writes row-level feature candidates only to caller-provided output paths;
  - should use `/tmp` or another non-repository path for smoke outputs.

Bounded feature smoke, run 2026-07-05:

| Metric | Value |
|---|---:|
| records read | 100 |
| read failures | 8 |
| requested window | 60 seconds |
| median seconds read | 20.49 |
| candidate features emitted | 38 |
| ECG feature coverage | 87 / 100 |
| pleth feature coverage | 92 / 100 |
| respiration feature coverage | 97 / 100 |
| arterial pressure feature coverage | 19 / 100 |

The smoke command was:

```bash
.venv/bin/python mimiciv_waveform_feature_extract.py \
  --waveform-root physionet.org/files/mimic4wdb/0.1.0 \
  --seconds 60 \
  --max-records 100 \
  --features-output /tmp/mimiciv_waveform_features_smoke.csv \
  --summary-output /tmp/mimiciv_waveform_features_smoke_summary.json
```

This proves the raw reader and bounded feature path work. It still does **not**
prove that waveform features improve forecasts.

## Accuracy Gate Still Required

Next, run the actual precision audit:

1. ECG/pleth/respiration features first because coverage is highest.
2. Arterial pressure features second because coverage is smaller but clinically
   high value.
3. Compare `table-only forecast` versus `table + waveform features` versus
   capacity-matched placebo.
4. Promote nothing unless patient-heldout plus careunit/time or hospital-heldout
   gates pass.

The current environment does not have the optional `wfdb` Python package
installed by default. It was installed into the local project `.venv` for this
smoke test and is now listed in `requirements.txt`.
