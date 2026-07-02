# eICU All-ICU Neuro-Note Findings

Date: 2026-07-02

This pass tests whether the acute-neuro GCS/delirium near-miss was only a power
problem. The bounded acute-neuro cohort had usable GCS and delirium support but
failed the seven-seed gate. This audit scales the same target family to all eICU
ICU stays with timestamped neuro-note observations.

## Cohort

`eicu_neuro_note_all_icu_extract.py` builds an all-ICU event-anchored cohort
from timestamped GCS and delirium/CAM observations. Row-level outputs remain
local-only and are not committed.

- eICU stays scanned: 200,859
- neuro-note events extracted: 3,749,014
- event-covered stays: 147,728
- measurement rows scanned: 389,771,740
- transition rows built: 994,134
- stays: 133,574
- subjects: 100,862
- hospitals: 201

Paired 6h support:

| Target | Paired Rows |
|---|---:|
| `neuro_gcs` | 710,976 |
| `neuro_delirium_present` | 120,208 |

## Leakage Guard

- anchor times are derived from timestamped GCS/delirium observations;
- current `neuro_*_t` observations use only `note_time <= anchor`, with a 24h
  lookback;
- future `neuro_*_tp6` labels use `note_time > anchor`, nearest to
  `anchor + 6h` within +/- 2h;
- same-time nowcasting excludes every `neuro_*` feature;
- future forecasting may use current `neuro_*_t`, because it is known before
  the horizon.

## Gate Result

`eicu_neuro_note_coverage_audit.py` was rerun on only the two supported targets:

- `neuro_gcs`
- `neuro_delirium_present`

No target passed:

| Axis | Eligible Targets | Validated Targets |
|---|---:|---:|
| same-time nowcast | 2 | 0 |
| 6h future forecast | 2 | 0 |
| calibrated interval | 2 | 0 |

Key heldout results:

| Target | Mode | Baseline MAE | Ridge MAE | Interpretation |
|---|---|---:|---:|---|
| `neuro_gcs` | nowcast | 1.804 | 2.117 | ridge beats placebo but loses to baseline |
| `neuro_gcs` | forecast | 0.417 | 0.533 | persistence wins |
| `neuro_delirium_present` | nowcast | 0.110 | 0.198 | ridge beats placebo but loses to baseline |
| `neuro_delirium_present` | forecast | 0.040 | 0.090 | persistence wins |

The all-ICU scale does not rescue the near-miss. It makes the boundary clearer:
the baseline/persistence signal dominates these note-derived targets at the
tested horizon.

## Conclusion

The original hypothesis was:

> GCS/delirium failed in the bounded acute-neuro cohort because the cohort was
> too small.

This all-ICU audit rejects that hypothesis. With 100,862 subjects and hundreds
of thousands of paired rows, GCS and delirium still do not become validated
nowcast or forecast targets.

The stronger rule is:

> Note-derived variables expand what the system can observe, but they do not
> automatically expand what the system can predict.

For GCS and delirium specifically, the useful role is timestamp-valid observed
evidence. If the value was observed before the anchor, Osler-JEPA may display it
as part of the current body state. If it was not observed, the system must leave
it missing. It may not impute it, forecast it, or attach numeric uncertainty
unless a future audit passes.

## Safety Boundary

Allowed:

- timestamp-valid shadow observation of GCS and delirium/CAM evidence;
- capability reporting.

Forbidden:

- same-time imputation of GCS/delirium;
- 6h forecasting of GCS/delirium;
- numeric uncertainty display for GCS/delirium;
- causal treatment-effect claims;
- counterfactual treatment planning;
- clinical recommendation authority;
- runtime treatment authority;
- checkpoint promotion;
- active symbolic-rule promotion.
