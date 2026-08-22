# Shared Whole-Body JEPA 27-Cell Release

This research-only release contains the trained weights and external-validation
contract used by the whole-body personalization runtime on 2026-08-22.

## Contents

- `whole_body_joint_jepa.pt`: frozen shared whole-body JEPA checkpoint.
- `unified_shared_jepa_head_bundle.joblib`: eleven validated target heads using
  the same shared patient representation.
- `runtime_registry.json`: 41-cell fail-closed registry; 27 cells can move and
  14 use persistence.
- `external_validation_report.json`: eICU selection and MIMIC-IV external gate
  results for the converted target heads.
- `MODEL_MANIFEST.json`: file sizes, SHA256 checksums, and safety boundary.

The release performs factual observation/forecast only. It does not support
causal treatment-effect or clinical decision claims.
