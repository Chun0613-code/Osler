# Repository Cleanup Audit

Date: 2026-06-15

This audit covers the 152 already-pending deletions and is intentionally
separate from the JEPA physiology/promotion changes.

## Deleted Groups

- 74 obsolete or disconnected `data/` files
- 65 files under the retired `legacy/` implementation
- 13 tracked Python bytecode files under `__pycache__/`

## Dependency Check

- No current Python source imports `legacy` or reads a path under `legacy/`.
- No current non-document source references the deleted expansion manifests,
  knowledge graph, HPO bridge, or retired disease-mechanism files.
- The live demo disease loader uses only the retained organ files:
  `blood.json`, `brain.json`, `gi.json`, `heart.json`, `lung.json`,
  `pancreas.json`, and `skin.json`.
- The live medication reasoning engine uses the retained `drugs_pkpd.json`.
- Tracked bytecode is generated output and is not a runtime dependency.

## Verification

- `python3 -m unittest discover -p 'test*.py'`: 77 tests passed.
- The strict MIMIC demo extractor completed end to end.
- The demo knowledge loader continued to load the retained drug and disease
  sources during the test suite.

## Decision

The deletion set is safe to commit as repository cleanup. It does not remove
the current Symbolic NEXUS runtime, DKA JEPA implementation, retained demo
knowledge, checkpoints, evaluation reports, or MIMIC extraction code.
