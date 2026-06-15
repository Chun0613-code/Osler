"""Versioned storage boundary for JEPA rule proposals."""

from __future__ import annotations

import json
from pathlib import Path


class RuleSandbox:
    """Allow automated writes only under ``rules/candidate``."""

    REQUIRED_FIELDS = {
        "rule_id", "level", "context", "action", "predicted_transition",
        "confidence", "evidence", "provenance", "status",
    }

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.active = self.root / "active"
        self.candidate = self.root / "candidate"
        self.rejected = self.root / "rejected"
        self.deprecated = self.root / "deprecated"
        for directory in (
            self.active, self.candidate, self.rejected, self.deprecated
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def write_candidates(self, run_id, payload):
        for rule in payload.get("rules", []):
            missing = self.REQUIRED_FIELDS - set(rule)
            if missing:
                raise ValueError(
                    f"candidate {rule.get('rule_id', '<unknown>')} missing {sorted(missing)}"
                )
            if rule["level"] not in {
                "hypothesis", "observational_association",
                "retrospective_validated",
            }:
                raise ValueError("JEPA cannot create active or safety-critical rules")
        destination = (self.candidate / f"{run_id}.json").resolve()
        if destination.parent != self.candidate:
            raise ValueError("candidate output escaped the sandbox")
        destination.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        return destination

    def assert_active_immutable(self, before_snapshot):
        after = {
            path.name: path.read_bytes()
            for path in sorted(self.active.glob("*")) if path.is_file()
        }
        if before_snapshot != after:
            raise RuntimeError("active Osler rules changed during automated proposal run")

    def snapshot_active(self):
        return {
            path.name: path.read_bytes()
            for path in sorted(self.active.glob("*")) if path.is_file()
        }
