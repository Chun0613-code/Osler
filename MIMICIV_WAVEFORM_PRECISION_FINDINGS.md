# MIMIC-IV Waveform Precision Candidate

Status: candidate-only scaffold. No waveform-derived feature is validated for
factual forecasting yet.

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

No local waveform manifest has been found in the current workspace, and no
accuracy audit has been run. This means waveform is opened as an engineering
track, not as a validated capability.

The correct next step is to place the MIMIC-IV Waveform headers under a local
root and run:

```bash
python3 mimiciv_waveform_manifest.py \
  --waveform-root /path/to/mimic-iv-waveform \
  --output /tmp/mimiciv_waveform_manifest.json
```

If the manifest shows useful ECG/arterial/pleth/respiration coverage, the next
engineering step is high-frequency feature extraction followed by the promotion
gate above.
