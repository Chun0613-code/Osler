# MIMIC-IV Radiology Note Observation Findings

Date: 2026-07-02

This pass tests whether timestamped MIMIC-IV radiology reports can deepen the
whole-body observation layer beyond structured ICU tables. It is a
measurement-depth experiment, not a causal or clinical decision module.

## What Was Built

`mimiciv_radiology_note_observation_extract.py` extracts six structured chest
radiology findings from MIMIC-IV Note v2.2 reports:

- pulmonary edema
- pleural effusion
- consolidation
- atelectasis
- pneumothorax
- cardiomegaly

The extractor uses deterministic phrase and negation rules as a local
stand-in for an LLM-backed structured-note extractor. The interface is designed
so a future LLM extractor can replace the rule extractor while preserving the
same target contract and leakage guards.

## Leakage Guard

The report timestamp policy is fail-closed:

- current report observations use only `note_time <= anchor`, with a 72h
  lookback window;
- future 6h labels use `note_time > anchor`, nearest to `anchor + 6h` within
  +/- 2h;
- `note_time` is `storetime` when available, otherwise `charttime`;
- same-time nowcasting excludes all `rad_*` features to prevent same-report
  leakage;
- future forecasting may use current `rad_*_t` observations because those are
  known before the forecast horizon.

## Extraction Scale

The source cohort is the local MIMIC-IV v3.1 all-ICU 6h observation cohort.
Row-level outputs remain in local scratch storage and are not committed.

- ICU transition rows: 640,164
- selected ICU stays: 90,190
- subjects: 63,307
- admissions with chest radiology reports: 56,063
- chest radiology reports parsed: 317,371
- rows with current chest report context: 406,588
- rows with future chest report labels: 57,976

Future paired label counts:

| Finding | Paired Rows |
|---|---:|
| pleural effusion | 25,483 |
| pneumothorax | 17,904 |
| atelectasis | 12,041 |
| consolidation | 10,051 |
| pulmonary edema | 9,216 |
| cardiomegaly | 3,348 |

## Gate Result

`mimiciv_radiology_note_coverage_audit.py` ran the same aggregate gate family
used by the numeric observation layer:

- seven patient-heldout split seeds;
- first-careunit heldout;
- chronological heldout;
- capacity-matched placebo ridge control;
- persistence/median baseline comparison;
- conformal interval gate for any validated forecast target.

No radiology finding passed the robust validation gate:

| Axis | Eligible Targets | Validated Targets |
|---|---:|---:|
| same-time nowcast | 6 | 0 |
| 6h future forecast | 6 | 0 |
| calibrated interval | 6 | 0 |

## Interpretation

This is a useful negative result. The system can now extract timestamped,
structured imaging findings from free text, but the current structured
physiology feature set does not robustly nowcast or forecast those imaging
findings after patient, careunit, time, and placebo gates.

That means radiology notes add a candidate measurement-depth input stream, not
a validated predictive capability. The correct contract is:

- structured note observations may be indexed and displayed as observed
  evidence when timestamp-valid;
- unobserved radiology findings must remain missing;
- no `rad_*` target may be imputed, forecast, or given numeric uncertainty
  unless a later audit validates that specific target;
- no causal, clinical, runtime, checkpoint-promotion, or active-rule authority
  is granted.

## Safety Boundary

This artifact is aggregate-only. It contains no row-level note text, patient
identifiers, note identifiers, timestamps, or clinical recommendations.

Allowed:

- timestamp-valid shadow observation of extracted note findings;
- capability reporting;
- future replacement of the deterministic extractor with an LLM extractor under
  the same target and timestamp contract.

Forbidden:

- causal treatment-effect claims;
- counterfactual treatment planning;
- clinical recommendation authority;
- runtime treatment authority;
- checkpoint promotion;
- active symbolic-rule promotion;
- treating candidate note targets as validated nowcasts or forecasts.
