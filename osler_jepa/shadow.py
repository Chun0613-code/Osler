"""Fail-closed shadow-mode bridge from live Osler to the DKA JEPA.

Shadow mode observes a completed symbolic recommendation. It cannot add, remove,
rerank, dose, or authorize a candidate. Only candidates already allowed by the
symbolic safety ladder and mapped to the DKA action contract are simulated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from uuid import uuid4


SUPPORTED_INDICATIONS = {"hyperglycemia"}
ALLOWED_SYMBOLIC_DECISIONS = {"ok", "caution", "adjust"}
REQUIRED_DKA_STATE = ("G", "Ke", "HCO3", "MAP")

# Fixed research probes are model inputs, never patient-specific dose advice.
RESEARCH_ACTION_PROBES = {
    "insulin": {"insulin_iv": 4.0},
    "fluids_iv": {"fluids": 250.0},
}

_LAB_ALIASES = {
    "G": ("glucose", "blood_glucose", "blood_sugar", "bg"),
    "pH": ("ph", "pH"),
    "HCO3": ("bicarbonate", "hco3", "hco3_minus"),
    "anion_gap": ("anion_gap", "ag"),
    "Ke": ("potassium", "k", "k_plus"),
    "Na": ("sodium", "na"),
    "osmolality": ("osmolality", "effective_osmolality"),
    "creatinine": ("creatinine", "cr"),
    "urine_output": ("urine_output", "urine_output_ml_hr"),
    "BHB": ("beta_hydroxybutyrate", "bhb"),
}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pick(values: Mapping[str, Any], aliases) -> Optional[float]:
    normalized = {
        str(key).strip().lower().replace("-", "_").replace(" ", "_"): value
        for key, value in values.items()
    }
    for alias in aliases:
        key = str(alias).lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            number = _finite(normalized[key])
            if number is not None:
                return number
    return None


def build_dka_shadow_state(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate live-case fields into a partial DKA observation contract."""
    labs = fields.get("labs") or {}
    vitals = fields.get("vitals") or {}
    state: Dict[str, float] = {}
    provenance: Dict[str, str] = {}

    for model_name, aliases in _LAB_ALIASES.items():
        value = _pick(labs, aliases)
        if value is not None:
            state[model_name] = value
            provenance[model_name] = "measured_lab"

    map_value = _pick(vitals, ("map", "mean_arterial_pressure"))
    if map_value is not None:
        state["MAP"] = map_value
        provenance["MAP"] = "measured_vital"
    else:
        sbp = _pick(vitals, ("sbp", "systolic_bp"))
        dbp = _pick(vitals, ("dbp", "diastolic_bp"))
        if sbp is not None and dbp is not None:
            state["MAP"] = (sbp + 2.0 * dbp) / 3.0
            provenance["MAP"] = "derived_from_sbp_dbp"

    missing = [name for name in REQUIRED_DKA_STATE if name not in state]
    optional_missing = [
        name for name in ("pH", "anion_gap", "Na", "creatinine", "urine_output", "BHB")
        if name not in state
    ]
    return {
        "state": state,
        "provenance": provenance,
        "required_missing": missing,
        "optional_missing": optional_missing,
        "complete_enough": not missing,
    }


def recommendation_fingerprint(result: Mapping[str, Any]) -> str:
    """Hash only the recommendation fields that shadow mode must not change."""
    protected = {
        "target_states": result.get("target_states"),
        "indication": result.get("indication"),
        "candidates": [
            {
                "drug": candidate.get("drug"),
                "clinical_role": candidate.get("clinical_role"),
                "mechanism_score": candidate.get("mechanism_score"),
                "matched_targets": candidate.get("matched_targets"),
                "safety": candidate.get("safety"),
                "dose": candidate.get("dose"),
                "final_answer": candidate.get("final_answer"),
            }
            for candidate in result.get("candidates", [])
        ],
    }
    payload = json.dumps(
        protected, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShadowObserver:
    """Run JEPA beside live Osler without returning decision authority."""

    def __init__(
        self,
        enabled: bool = False,
        checkpoint: str = "dka_symbolic_jepa_v5.pt",
        device: str = "cpu",
        horizon_hours: float = 6.0,
        audit_path: Optional[str] = None,
        runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.enabled = bool(enabled)
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.horizon_hours = float(horizon_hours)
        self.audit_path = Path(audit_path) if audit_path else None
        self._runner = runner
        self._model = None
        self._checkpoint_sha256 = None

    @classmethod
    def from_environment(cls) -> "ShadowObserver":
        enabled = os.environ.get("OSLER_JEPA_SHADOW", "0").strip().lower()
        default_checkpoint = Path(__file__).resolve().parent.parent / (
            "dka_symbolic_jepa_v5.pt"
        )
        return cls(
            enabled=enabled in {"1", "true", "yes", "on"},
            checkpoint=os.environ.get(
                "OSLER_JEPA_CHECKPOINT", str(default_checkpoint)
            ),
            device=os.environ.get("OSLER_JEPA_DEVICE", "cpu"),
            horizon_hours=float(os.environ.get("OSLER_JEPA_SHADOW_HOURS", "6")),
            audit_path=os.environ.get("OSLER_JEPA_SHADOW_LOG") or None,
        )

    def _base_report(self, fingerprint: str) -> Dict[str, Any]:
        if self.enabled and self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = _file_sha256(self.checkpoint)
        return {
            "schema_version": "1.1.0",
            "record_type": "shadow_forecast",
            "event_id": str(uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "shadow",
            "enabled": self.enabled,
            "research_only": True,
            "decision_authority": False,
            "affects_live_recommendation": False,
            "clinical_dose_claim_allowed": False,
            "causal_claim_allowed": False,
            "recommendation_fingerprint_before": fingerprint,
            "recommendation_fingerprint_after": fingerprint,
            "recommendation_integrity_verified": True,
            "checkpoint": {
                "name": self.checkpoint.name,
                "sha256": self._checkpoint_sha256 if self.enabled else None,
                "device": self.device,
            },
        }

    def _run_model(self, state, action, input_warnings):
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint.name}")
        if self._model is None:
            from dka_world_model import load_checkpoint

            self._model, _ = load_checkpoint(self.checkpoint, device=self.device)
        from dka_action_contract import expand_action
        from predict_dka_intervention import compare

        return compare(
            self._model,
            state,
            expand_action(action),
            self.horizon_hours,
            self.device,
            input_warnings=input_warnings,
        )

    def _run(self, state, action, warnings):
        if self._runner is not None:
            return self._runner(
                state=deepcopy(state),
                action=deepcopy(action),
                hours=self.horizon_hours,
                device=self.device,
                input_warnings=list(warnings),
            )
        return self._run_model(state, action, warnings)

    def _summary(self, candidate, action, comparison):
        treated = comparison.get("predicted_intervention_trajectory") or []
        untreated = comparison.get("predicted_no_treatment_trajectory") or []
        treated_risk = treated[-1].get("death_probability") if treated else None
        untreated_risk = untreated[-1].get("death_probability") if untreated else None
        validation = comparison.get("osler_transition_validation") or {}
        prolog = comparison.get("osler_prolog_reasoning") or {}
        final_state = treated[-1].get("state", {}) if treated else {}
        return {
            "candidate": candidate.get("drug"),
            "symbolic_safety_decision": (candidate.get("safety") or {}).get("decision"),
            "research_probe_action": action,
            "probe_is_clinical_dose": False,
            "horizon_hours": self.horizon_hours,
            "predicted_final_state": final_state,
            "effective_action_summary": comparison.get(
                "effective_action_summary", action
            ),
            "effective_action_schedule": comparison.get(
                "osler_dynamic_action_schedule", []
            ),
            "predicted_effect": comparison.get("predicted_effect_at_final_horizon", {}),
            "predicted_risk": {
                "intervention": treated_risk,
                "no_treatment": untreated_risk,
            },
            "transition_validation": validation,
            "prolog_reasoning": prolog,
            "disagreement": bool(
                validation.get("status") == "contradicted"
                or prolog.get("decision") in {"block", "modify"}
            ),
            "outcome_reconciliation_pending": True,
        }

    def _write_audit(self, report: Dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, sort_keys=True, default=str) + "\n")
        except OSError as exc:
            report.setdefault("warnings", []).append(
                f"shadow audit log unavailable: {type(exc).__name__}"
            )

    def observe(
        self,
        canonical_indication: str,
        fields: Mapping[str, Any],
        symbolic_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        before = recommendation_fingerprint(symbolic_result)
        report = self._base_report(before)
        if not self.enabled:
            report["status"] = "disabled"
            return report
        if canonical_indication not in SUPPORTED_INDICATIONS:
            report.update({
                "status": "not_applicable",
                "reason": "no disease-specific JEPA contract for this indication",
                "supported_indications": sorted(SUPPORTED_INDICATIONS),
            })
            self._write_audit(report)
            return report

        state_contract = build_dka_shadow_state(fields)
        report["state_contract"] = state_contract
        if not state_contract["complete_enough"]:
            report.update({
                "status": "insufficient_state",
                "reason": "required DKA observations are missing",
            })
            self._write_audit(report)
            return report

        eligible = []
        skipped = []
        for candidate in symbolic_result.get("candidates", []):
            name = str(candidate.get("drug") or "").strip().lower()
            action = RESEARCH_ACTION_PROBES.get(name)
            decision = (candidate.get("safety") or {}).get("decision")
            if action is None:
                skipped.append({"candidate": name, "reason": "not_mapped_to_dka_action"})
            elif decision not in ALLOWED_SYMBOLIC_DECISIONS:
                skipped.append({
                    "candidate": name,
                    "reason": "not_allowed_by_symbolic_safety_gate",
                    "symbolic_safety_decision": decision,
                })
            else:
                eligible.append((candidate, action))
        report["skipped_candidates"] = skipped
        if not eligible:
            report.update({
                "status": "no_eligible_actions",
                "reason": "no symbolic-approved candidate maps to the DKA action contract",
            })
            self._write_audit(report)
            return report

        warnings = [
            "Shadow probes are fixed model inputs, not clinical doses or recommendations."
        ]
        warnings.extend(
            f"missing optional DKA observation: {name}"
            for name in state_contract["optional_missing"]
        )
        try:
            report["observations"] = [
                self._summary(
                    candidate,
                    action,
                    self._run(state_contract["state"], action, warnings),
                )
                for candidate, action in eligible
            ]
            report["status"] = "observed"
            report["warnings"] = warnings
        except Exception as exc:
            report.update({
                "status": "error",
                "reason": "shadow inference failed closed",
                "error": f"{type(exc).__name__}: {exc}",
            })

        after = recommendation_fingerprint(symbolic_result)
        report["recommendation_fingerprint_after"] = after
        report["recommendation_integrity_verified"] = before == after
        if before != after:
            report.update({
                "status": "integrity_failure",
                "reason": "protected symbolic recommendation changed during shadow inference",
                "observations": [],
            })
        self._write_audit(report)
        return report


def observe_live_recommendation(canonical_indication, fields, symbolic_result):
    """Environment-configured entry point used by the live demo agent."""
    return ShadowObserver.from_environment().observe(
        canonical_indication, fields, symbolic_result
    )
