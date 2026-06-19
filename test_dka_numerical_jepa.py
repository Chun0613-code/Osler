import unittest
from unittest.mock import patch
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from dka_body import DKABody, DKAPatientProfile, OSM_INJURY_DEATH
from dka_osler import shield, shield_full, shield_route_aware
from dka_action_contract import ACTION_INDEX, expand_action
from dka_world_model import (
    A_DIM, DKA_STATE_COMPILER, H_DIM, S_DIM, STATE_KEYS, WorldModel, a2vec,
    load_checkpoint, observation_context, randomized_dka, save_checkpoint, s2vec,
    treatment_history_features,
)
from train_intervention_jepa import (
    generate_branched_dataset,
    simulator_calibration_audit,
)
from osler_jepa.actions import (
    Intervention,
    TREATMENT_EVENT_DIM,
    TemporalActionEncoder,
    treatment_event_features,
)
from osler_jepa.ontology import OSLER_STATE_ONTOLOGY
from osler_jepa.validator import OSLER_DKA_VALIDATOR, compile_transition_rules
from osler_jepa.embodied_logic import OSLER_DKA_PROLOG
from osler_jepa.belief import (
    PotassiumStoreBelief,
    downstream_observable_gate,
    infer_hidden_beliefs,
)
from osler_jepa.rule_sandbox import RuleSandbox
from osler_jepa.rule_inducer import validate_candidates
from osler_jepa.real_world_adapter import RealWorldAdapter
from osler_jepa.causal_evaluation import TargetTrialSpec, evaluate_trial
from osler_jepa.symbolic import RULE_IDS, rule_supervision
from osler_jepa.state_compiler import DKA_SYMBOLIC_FACT_KEYS, ground_dka_facts
from osler_jepa.viability import HomeostaticWorldModelObjective
from osler_jepa.persistence_gate import evaluate_persistence_gate
from osler_jepa.curriculum import STAGES, weights_for_stage
from osler_jepa.action_prior import EmpiricalActionPrior
from osler_jepa.greybox_residual import (
    FORBIDDEN_KEYS, MAX_ABS_RATE, project_residual,
)
from osler_jepa.anchored_residual import (
    AnchoredResidualConfig, AnchoredResidualGate, prolog_direction_gate,
)
from osler_jepa.shadow import (
    ShadowObserver,
    build_dka_shadow_state,
    recommendation_fingerprint,
)
from osler_jepa.shadow_outcomes import (
    load_shadow_forecast,
    pseudonymize_subject,
    reconcile_from_ledger,
    reconcile_shadow_forecast,
)
from osler_jepa.shadow_cohort import (
    evaluate_shadow_cohort,
    evaluate_shadow_group,
    load_reconciliations,
)
from real_world_improvement import episode_features
from real_world_power_analysis import required_stays_for_win
from predict_dka_intervention import compare, predict
from dka_fidelity_replay import init_body
from dka_viability_falsification_audit import (
    classify_death_cause, coverage_artifact_reasons, threshold_excess,
)
from dka_numeric_artifact_audit import BOUND_REGISTRY, DENOMINATOR_GUARDS
from dka_transition_extract import find_dka_onset
from mimic_action_history import (
    action_window_summary, deduplicate_events, normalize_events,
    maintenance_window_summary, normalize_maintenance_events,
    treatment_event_records,
)

sys.path.insert(0, str(Path(__file__).parent / "engine"))
sys.path.insert(0, str(Path(__file__).parent / "demo"))
from reasoning_engine import mechanism_candidates
import case_parser


class NumericalJEPATests(unittest.TestCase):
    @staticmethod
    def _shadow_fields():
        return {
            "vitals": {"sbp": 96, "dbp": 58},
            "labs": {
                "glucose": 480, "potassium": 5.2,
                "bicarbonate": 8, "anion_gap": 24,
                "sodium": 134, "creatinine": 1.4,
            },
        }

    @staticmethod
    def _shadow_result(decision="ok"):
        return {
            "target_states": ["blood glucose elevation down"],
            "indication": "hyperglycemia",
            "candidates": [{
                "drug": "insulin", "mechanism_score": 0.3,
                "clinical_role": {"rank_priority": 0},
                "matched_targets": [],
                "safety": {"decision": decision, "reasons": []},
                "dose": {"patient_specific_allowed": False},
                "final_answer": "symbolic result",
            }],
        }

    @staticmethod
    def _cohort_record(index, subject, improvement=0.5):
        per_state = {
            state: {
                "normalized_improvement_over_persistence": improvement,
                "jepa_normalized_error": 0.25,
                "persistence_normalized_error": 0.25 + improvement,
            }
            for state in ("G", "Ke", "HCO3", "MAP")
        }
        return {
            "record_type": "shadow_outcome_reconciliation",
            "reconciliation_id": f"reconciliation-{index}",
            "forecast_event_id": f"forecast-{index}",
            "candidate": "insulin",
            "checkpoint": {"name": "jepa.pt", "sha256": "checkpoint-sha"},
            "subject_group_hash": subject,
            "status": "scored",
            "summary": {
                "normalized_improvement_over_persistence": improvement,
                "jepa_changed_state_direction_accuracy": 0.8,
            },
            "per_state": per_state,
            "terminal_outcome": {"brier_score": 0.05},
            "forecast_provenance": {"symbolic_disagreement": False},
        }

    def test_dka_parser_and_shadow_state_preserve_measured_provenance(self):
        parsed = case_parser.parse_rules(
            "34F DKA glucose 480, pH 7.08, HCO3 8, anion gap 24, "
            "K 5.2, Na 134, creatinine 1.4, urine output 75, BP 96/58"
        )
        contract = build_dka_shadow_state(parsed)
        self.assertTrue(contract["complete_enough"], contract)
        self.assertAlmostEqual(contract["state"]["MAP"], 70.6667, places=3)
        self.assertEqual(contract["provenance"]["G"], "measured_lab")
        self.assertEqual(contract["provenance"]["MAP"], "derived_from_sbp_dbp")

    def test_shadow_observer_cannot_modify_symbolic_recommendation(self):
        result = self._shadow_result()
        before = recommendation_fingerprint(result)

        def runner(**kwargs):
            self.assertEqual(kwargs["action"], {"insulin_iv": 4.0})
            return {
                "predicted_effect_at_final_horizon": {"G": -80.0},
                "predicted_intervention_trajectory": [
                    {"death_probability": 0.1, "state": {
                        "G": 300.0, "Ke": 4.6, "HCO3": 12.0, "MAP": 74.0,
                    }}
                ],
                "predicted_no_treatment_trajectory": [
                    {"death_probability": 0.2}
                ],
                "osler_transition_validation": {"status": "verified"},
                "osler_prolog_reasoning": {"decision": "allow"},
                "jepa_symbolic_transition": {
                    "transitions": [{"time_window_hours": 6.0}]
                },
                "effective_action_summary": {"insulin_iv": 4.0},
            }

        report = ShadowObserver(enabled=True, runner=runner).observe(
            "hyperglycemia", self._shadow_fields(), result
        )
        self.assertEqual(report["status"], "observed")
        self.assertFalse(report["decision_authority"])
        self.assertFalse(report["affects_live_recommendation"])
        self.assertTrue(report["recommendation_integrity_verified"])
        self.assertEqual(before, recommendation_fingerprint(result))
        self.assertEqual(report["observations"][0]["predicted_effect"]["G"], -80.0)
        self.assertEqual(
            report["observations"][0]["predicted_final_state"]["G"], 300.0
        )

    def test_shadow_outcome_scores_jepa_against_persistence(self):
        forecast = {
            "record_type": "shadow_forecast", "status": "observed",
            "event_id": "forecast-1",
            "state_contract": {"state": {
                "G": 480.0, "Ke": 5.2, "HCO3": 8.0, "MAP": 70.0,
            }},
            "observations": [{
                "candidate": "insulin", "horizon_hours": 6.0,
                "effective_action_summary": {"insulin_iv": 4.0},
                "effective_action_schedule": [{
                    "hours": 0.0, "action": {"insulin_iv": 4.0},
                }],
                "predicted_risk": {"intervention": 0.2},
                "predicted_final_state": {
                    "G": 300.0, "Ke": 4.5, "HCO3": 13.0, "MAP": 76.0,
                },
            }],
        }
        future = {
            "vitals": {"map": 75.0},
            "labs": {
                "glucose": 310.0, "potassium": 4.6, "bicarbonate": 12.5,
            },
            "outcome": {"alive": True},
        }
        record = reconcile_shadow_forecast(
            forecast, future,
            {"schedule": [{
                "hours": 0.0, "action": {"insulin_iv": 4.0},
            }]},
            6.0,
        )
        self.assertEqual(record["status"], "scored")
        self.assertEqual(record["summary"]["winner"], "jepa")
        self.assertGreater(
            record["summary"]["normalized_improvement_over_persistence"], 0
        )
        self.assertFalse(record["online_weight_update_allowed"])
        self.assertFalse(record["automatic_rule_promotion_allowed"])
        self.assertFalse(record["causal_claim_allowed"])
        self.assertEqual(record["terminal_outcome"]["brier_score"], 0.04)

    def test_shadow_outcome_rejects_action_mismatch(self):
        forecast = {
            "record_type": "shadow_forecast", "status": "observed",
            "event_id": "forecast-2",
            "state_contract": {"state": {"G": 480.0}},
            "observations": [{
                "candidate": "insulin", "horizon_hours": 6.0,
                "effective_action_summary": {"insulin_iv": 4.0},
                "effective_action_schedule": [{
                    "hours": 0.0, "action": {"insulin_iv": 4.0},
                }],
                "predicted_final_state": {"G": 300.0},
            }],
        }
        record = reconcile_shadow_forecast(
            forecast,
            {"vitals": {}, "labs": {"glucose": 310.0}},
            {"schedule": [{"hours": 0.0, "action": {}}]},
            6.0,
        )
        self.assertEqual(record["status"], "action_mismatch")
        self.assertNotIn("summary", record)

    def test_shadow_outcome_rejects_same_mean_with_different_timing(self):
        forecast = {
            "record_type": "shadow_forecast", "status": "observed",
            "event_id": "forecast-timing",
            "state_contract": {"state": {"G": 480.0}},
            "observations": [{
                "candidate": "insulin", "horizon_hours": 6.0,
                "effective_action_summary": {"insulin_iv": 4.0},
                "effective_action_schedule": [{
                    "hours": 0.0, "action": {"insulin_iv": 4.0},
                }],
                "predicted_final_state": {"G": 300.0},
            }],
        }
        same_mean_different_timing = {"schedule": [
            {"hours": 0.0, "action": {}},
            {"hours": 3.0, "action": {"insulin_iv": 8.0}},
        ]}
        record = reconcile_shadow_forecast(
            forecast,
            {"labs": {"glucose": 310.0}, "vitals": {}},
            same_mean_different_timing,
            6.0,
        )
        self.assertTrue(record["action_alignment"]["matches"])
        self.assertEqual(record["status"], "action_timing_mismatch")
        self.assertNotIn("summary", record)

    def test_shadow_ledger_round_trip_appends_reconciliation(self):
        forecast = {
            "record_type": "shadow_forecast", "status": "observed",
            "event_id": "forecast-ledger",
            "state_contract": {"state": {"G": 480.0}},
            "observations": [{
                "candidate": "insulin", "horizon_hours": 6.0,
                "effective_action_summary": {"insulin_iv": 4.0},
                "effective_action_schedule": [{
                    "hours": 0.0, "action": {"insulin_iv": 4.0},
                }],
                "predicted_final_state": {"G": 300.0},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "shadow.jsonl"
            ledger.write_text(json.dumps(forecast) + "\n", encoding="utf-8")
            loaded = load_shadow_forecast(ledger, "forecast-ledger")
            self.assertEqual(loaded["event_id"], "forecast-ledger")
            record = reconcile_from_ledger(
                ledger,
                "forecast-ledger",
                {"labs": {"glucose": 310.0}, "vitals": {}},
                {"schedule": [{
                    "hours": 0.0, "action": {"insulin_iv": 4.0},
                }]},
                6.0,
            )
            self.assertEqual(record["status"], "scored")
            lines = ledger.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                json.loads(lines[-1])["record_type"],
                "shadow_outcome_reconciliation",
            )

    def test_subject_pseudonym_is_stable_and_raw_key_is_not_stored(self):
        first = pseudonymize_subject("patient-123", "local-secret")
        second = pseudonymize_subject("patient-123", "local-secret")
        other = pseudonymize_subject("patient-123", "different-secret")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotIn("patient-123", first)
        forecast = {
            "record_type": "shadow_forecast", "status": "observed",
            "event_id": "forecast-private",
            "checkpoint": {"name": "jepa.pt", "sha256": "sha"},
            "state_contract": {"state": {"G": 480.0}},
            "observations": [{
                "candidate": "insulin", "horizon_hours": 6.0,
                "effective_action_summary": {"insulin_iv": 4.0},
                "effective_action_schedule": [{
                    "hours": 0.0, "action": {"insulin_iv": 4.0},
                }],
                "predicted_final_state": {"G": 300.0},
            }],
        }
        record = reconcile_shadow_forecast(
            forecast,
            {"labs": {"glucose": 310.0}, "vitals": {}},
            {"schedule": [{
                "hours": 0.0, "action": {"insulin_iv": 4.0},
            }]},
            6.0,
            subject_key="patient-123",
            subject_salt="local-secret",
        )
        rendered = json.dumps(record)
        self.assertNotIn("patient-123", rendered)
        self.assertEqual(record["subject_group_hash"], first)
        self.assertTrue(record["eligible_for_cohort_evaluation"])

    def test_shadow_cohort_does_not_count_repeated_patient_as_independent(self):
        records = [
            self._cohort_record(index, "same-patient")
            for index in range(100)
        ]
        report = evaluate_shadow_group(records, bootstrap_samples=100)
        self.assertEqual(report["support"]["independent_subjects"], 1)
        self.assertFalse(report["automated_retrospective_gate"]["passes"])
        self.assertIn(
            "too_few_independent_subjects",
            report["automated_retrospective_gate"]["failures"],
        )

    def test_shadow_cohort_gate_can_pass_but_never_promotes_clinically(self):
        records = [
            self._cohort_record(index, f"patient-{index % 35}")
            for index in range(60)
        ]
        report = evaluate_shadow_cohort(records, bootstrap_samples=200)
        group = report["groups"]["checkpoint-sha::insulin"]
        self.assertTrue(group["automated_retrospective_gate"]["passes"], group)
        self.assertGreater(
            group["overall"]["normalized_improvement_over_persistence"]
            ["patient_cluster_bootstrap_95_ci"][0],
            0.0,
        )
        self.assertFalse(group["clinical_promotion_allowed"])
        self.assertFalse(group["causal_claim_allowed"])
        self.assertFalse(group["online_weight_update_allowed"])

    def test_shadow_cohort_rejects_core_state_hidden_by_overall_average(self):
        records = [
            self._cohort_record(index, f"patient-{index % 35}")
            for index in range(60)
        ]
        for record in records:
            potassium = record["per_state"]["Ke"]
            potassium["normalized_improvement_over_persistence"] = -0.2
            potassium["jepa_normalized_error"] = 0.7
            potassium["persistence_normalized_error"] = 0.5
        report = evaluate_shadow_group(records, bootstrap_samples=200)
        self.assertFalse(report["automated_retrospective_gate"]["passes"])
        self.assertIn(
            "core_state_not_better_than_persistence:Ke",
            report["automated_retrospective_gate"]["failures"],
        )

    def test_shadow_cohort_loader_deduplicates_reconciliation(self):
        first = self._cohort_record(1, "patient-a", improvement=0.1)
        second = self._cohort_record(1, "patient-a", improvement=0.7)
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "shadow.jsonl"
            ledger.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            records = load_reconciliations(ledger)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["summary"]["normalized_improvement_over_persistence"],
            0.7,
        )

    def test_shadow_observer_respects_symbolic_safety_block(self):
        def runner(**kwargs):
            self.fail("blocked symbolic candidate must not reach JEPA")

        report = ShadowObserver(enabled=True, runner=runner).observe(
            "hyperglycemia", self._shadow_fields(), self._shadow_result("avoid")
        )
        self.assertEqual(report["status"], "no_eligible_actions")
        self.assertEqual(
            report["skipped_candidates"][0]["reason"],
            "not_allowed_by_symbolic_safety_gate",
        )

    def test_shadow_observer_fails_closed_on_inference_error(self):
        def runner(**kwargs):
            raise RuntimeError("synthetic failure")

        report = ShadowObserver(enabled=True, runner=runner).observe(
            "hyperglycemia", self._shadow_fields(), self._shadow_result()
        )
        self.assertEqual(report["status"], "error")
        self.assertTrue(report["recommendation_integrity_verified"])
        self.assertFalse(report["decision_authority"])

    def test_observation_contract_marks_missing_and_stale_values(self):
        _, mask, age = observation_context(
            {"G": 420.0, "Ke": 4.8}, {"G": 1.5, "Ke": 0.25}
        )
        glucose = STATE_KEYS.index("G")
        potassium = STATE_KEYS.index("Ke")
        bicarbonate = STATE_KEYS.index("HCO3")
        self.assertEqual(mask[glucose], 1.0)
        self.assertEqual(mask[potassium], 1.0)
        self.assertEqual(mask[bicarbonate], 0.0)
        self.assertEqual(age[glucose], 1.5)
        self.assertEqual(age[bicarbonate], 24.0)

    def test_complete_observation_preserves_legacy_encoder_behavior(self):
        model = WorldModel().eval()
        state = torch.randn(3, S_DIM)
        legacy = model.encode_state(state)
        explicit = model.encode_state(
            state, observation_mask=torch.ones_like(state),
            observation_age=torch.zeros_like(state),
        )
        self.assertTrue(torch.allclose(legacy, explicit))

    def test_irregular_rollout_passes_each_time_interval(self):
        model = WorldModel().eval()
        state = torch.zeros(1, S_DIM)
        actions = torch.zeros(1, 3, A_DIM)
        intervals = torch.tensor([[0.25, 0.75, 1.0]])
        with patch.object(model, "predict_latent", wraps=model.predict_latent) as wrapped:
            model.rollout(state, actions, delta_hours=intervals)
        used = [float(call.kwargs["delta_hours"][0]) for call in wrapped.call_args_list]
        elapsed = [float(call.kwargs["elapsed_hours"][0]) for call in wrapped.call_args_list]
        self.assertEqual(used, [0.25, 0.75, 1.0])
        self.assertEqual(elapsed, [0.0, 0.25, 1.0])

    def test_counterfactual_branches_share_irregular_time_grid(self):
        dataset = generate_branched_dataset(2, 4, seed=3)
        self.assertEqual(dataset["time_deltas"].shape, (2, 11, 4))
        self.assertTrue(np.all(
            dataset["time_deltas"][:, :1] == dataset["time_deltas"]
        ))
        self.assertTrue(set(np.unique(dataset["time_deltas"])).issubset(
            {0.0, 0.25, 0.5, 0.75, 1.0}
        ))
        self.assertEqual(
            dataset["action_events"].shape, (2, 11, 4, TREATMENT_EVENT_DIM)
        )

    def test_simulator_audit_reports_mortality_and_response_gates(self):
        dataset = generate_branched_dataset(12, 4, seed=11)
        audit = simulator_calibration_audit(dataset)
        self.assertIn("full_protocol", audit["protocols"])
        self.assertIn(
            "iv_insulin_reduces_glucose_vs_no_treatment",
            audit["mechanistic_direction_gates"],
        )
        self.assertFalse(audit["clinical_calibration_claim_allowed"])

    def test_treatment_event_features_mark_start_and_stop(self):
        sequence = np.zeros((3, A_DIM), dtype=np.float32)
        sequence[0, ACTION_INDEX["insulin_iv"]] = 4.0
        sequence[1, ACTION_INDEX["insulin_iv"]] = 6.0
        events = treatment_event_features(sequence)
        self.assertEqual(events[0, ACTION_INDEX["insulin_iv"]], 1.0)
        self.assertEqual(events[1].sum(), 0.0)
        self.assertEqual(
            events[2, A_DIM + ACTION_INDEX["insulin_iv"]], 1.0
        )

    def test_zero_event_context_preserves_legacy_prediction(self):
        model = WorldModel().eval()
        latent = torch.zeros(1, model.P.latent_dim)
        action = torch.zeros(1, A_DIM)
        legacy = model.predict_latent(latent, action)
        explicit = model.predict_latent(
            latent, action,
            treatment_events=torch.zeros(1, TREATMENT_EVENT_DIM),
        )
        self.assertTrue(torch.allclose(legacy, explicit))

    def test_ltc_dynamics_cell_is_candidate_only_and_time_sensitive(self):
        model = WorldModel(dynamics_cell="ltc").eval()
        self.assertEqual(model.dynamics_cell, "ltc")
        self.assertTrue(model.P.continuous_time)
        state = torch.zeros(2, S_DIM)
        action = torch.zeros(2, A_DIM)
        latent = model.encode_state(state)
        short = model.predict_latent(latent, action, delta_hours=0.25)
        long = model.predict_latent(latent, action, delta_hours=1.0)
        self.assertFalse(torch.allclose(short, long))
        predicted, _, latents = model.rollout(
            state,
            torch.zeros(2, 3, A_DIM),
            delta_hours=torch.tensor([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]]),
        )
        self.assertEqual(predicted.shape, (2, 3, S_DIM))
        self.assertEqual(latents.shape[-1], model.P.latent_dim)

    def test_ltc_checkpoint_roundtrip_preserves_dynamics_cell(self):
        model = WorldModel(dynamics_cell="ltc").eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ltc.pt"
            save_checkpoint(model, path, metadata={
                "model_architecture": {"dynamics_cell": "ltc"}
            })
            loaded, payload = load_checkpoint(path)
        self.assertEqual(loaded.dynamics_cell, "ltc")
        self.assertEqual(payload["dynamics"]["cell"], "ltc")
        self.assertFalse(payload["dynamics"]["runtime_promotion_implied"])

    def test_potassium_store_belief_predicts_and_updates(self):
        state = {
            "G": 480.0, "Ke": 3.4, "pH": 7.1,
            "creatinine": 1.2, "urine_output": 150.0,
        }
        prior = PotassiumStoreBelief.from_state(state)
        predicted = prior.predict({"kcl": 20.0}, state, 1.0)
        posterior = predicted.update(state, model_estimate=90.0)
        self.assertGreater(predicted.mean, prior.mean)
        self.assertLess(posterior.variance, predicted.variance)
        self.assertFalse(posterior.to_dict()["measured"])

    def test_generalized_hidden_beliefs_require_downstream_gate(self):
        state = {
            "G": 480.0, "pH": 7.1, "HCO3": 8.0, "anion_gap": 24.0,
            "Ke": 3.4, "MAP": 70.0, "Na": 132.0,
            "osmolality": 292.0, "creatinine": 1.8,
            "urine_output": 40.0, "I": 6.0,
        }
        beliefs = infer_hidden_beliefs(state)
        self.assertIn("acid_base_buffer_reserve", beliefs)
        self.assertIn("insulin_sensitivity", beliefs)
        for belief in beliefs.values():
            payload = belief.to_dict()
            self.assertFalse(payload["measured"])
            self.assertFalse(
                payload["validation_gate"][
                    "direct_hidden_state_accuracy_claim_allowed"
                ]
            )
            self.assertTrue(payload["validation_gate"]["targets"])

        gate = downstream_observable_gate(
            "insulin_sensitivity",
            baseline_mae={"G": 100.0, "BHB": 2.0},
            candidate_mae={"G": 92.0, "BHB": 1.8},
            targets=("G", "BHB"),
        )
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["clinical_claim_allowed"])

        failed = downstream_observable_gate(
            "insulin_sensitivity",
            baseline_mae={"G": 100.0},
            candidate_mae={"G": 101.0},
            targets=("G",),
        )
        self.assertFalse(failed["passed"])

    def test_prolog_blocks_insulin_during_critical_hypokalemia(self):
        result = OSLER_DKA_PROLOG.evaluate(
            state={"G": 480, "Ke": 2.8, "pH": 7.1},
            proposed_action={"insulin_iv": 6.0},
            applied_action={},
            future={"G": 470.0},
            baseline_future={"G": 480.0},
        )
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["blocked_actions"][0]["action"], "insulin_iv")
        self.assertIn("proofs", result["blocked_actions"][0])

    def test_prolog_is_final_veto_after_symbolic_policy(self):
        safe_action, trace = OSLER_DKA_PROLOG.gate(
            {"G": 480, "Ke": 2.8, "pH": 6.9},
            {"insulin_iv": 4.0, "kcl": 20.0},
        )
        self.assertEqual(safe_action[ACTION_INDEX["insulin_iv"]], 0.0)
        self.assertTrue(any(item["type"] == "prolog_veto" for item in trace))

    def test_prolog_requires_potassium_with_low_k_insulin(self):
        result = OSLER_DKA_PROLOG.evaluate(
            state={"G": 480, "Ke": 3.1, "pH": 7.1},
            proposed_action={"insulin_iv": 6.0},
            applied_action={"insulin_iv": 6.0, "kcl": 10.0},
            future={"G": 430.0, "Ke": 3.2, "K_store": 90.0},
            baseline_future={"G": 480.0, "Ke": 3.1, "K_store": 80.0},
        )
        self.assertEqual(result["decision"], "modify")
        self.assertEqual(
            result["required_cointerventions"][0]["action"], "kcl"
        )
        checks = {check["rule_id"]: check for check in result["effect_checks"]}
        self.assertEqual(checks["iv_insulin_lowers_glucose"]["status"], "verified")
        self.assertEqual(
            checks["kcl_raises_total_body_store"]["status"], "verified"
        )

    def test_prolog_rule_pack_covers_differentiable_validator(self):
        alignment = OSLER_DKA_PROLOG.validator_alignment(OSLER_DKA_VALIDATOR)
        self.assertTrue(alignment["aligned"], alignment)

    def test_differentiable_constraints_compile_from_active_prolog(self):
        compiled = compile_transition_rules()
        self.assertEqual(
            [rule.rule_id for rule in compiled],
            [rule.rule_id for rule in OSLER_DKA_VALIDATOR.rules],
        )
        self.assertTrue(all(
            rule.provenance.endswith("rules/active/dka_embodied.pl")
            for rule in compiled
        ))
        self.assertTrue(all(0.0 < rule.confidence <= 1.0 for rule in compiled))
        self.assertTrue(all(rule.max_hours > rule.min_hours for rule in compiled))

    def test_temporal_constraint_only_penalizes_inside_effect_window(self):
        action = torch.zeros(1, 1, A_DIM)
        action[..., ACTION_INDEX["dextrose"]] = 1.0
        wrong = torch.zeros(1, 1, S_DIM)
        wrong[..., STATE_KEYS.index("G")] = -0.1
        inside = OSLER_DKA_VALIDATOR.consistency_loss(
            wrong, action, horizon_hours=torch.tensor([[1.0]])
        )
        outside = OSLER_DKA_VALIDATOR.consistency_loss(
            wrong, action, horizon_hours=torch.tensor([[3.0]])
        )
        self.assertGreater(float(inside), 0.0)
        self.assertEqual(float(outside), 0.0)

    def test_prolog_defers_effect_check_outside_time_window(self):
        result = OSLER_DKA_PROLOG.evaluate(
            state={"G": 100.0, "Ke": 4.0, "pH": 7.2},
            proposed_action={"dextrose": 10.0},
            applied_action={"dextrose": 10.0},
            future={"G": 120.0},
            baseline_future={"G": 100.0},
            elapsed_hours=3.0,
        )
        self.assertFalse(result["effect_checks"])
        self.assertEqual(
            result["deferred_effect_checks"][0]["status"],
            "outside_time_window",
        )

    def test_neutral_patient_profile_preserves_original_start(self):
        body = DKABody(profile=DKAPatientProfile())
        self.assertAlmostEqual(body.V, 12.0)
        self.assertAlmostEqual(body.MAP, 78.0)

    def test_patient_profiles_change_intervention_response(self):
        resistant = DKABody(profile=DKAPatientProfile(insulin_sensitivity=0.5))
        sensitive = DKABody(profile=DKAPatientProfile(insulin_sensitivity=1.7))
        for body in (resistant, sensitive):
            body.G = 480.0
            body.HCO3 = 8.0
            body.Ket = 11.0
            body.Ke = 5.0
            body.I = 1.0
            body.step([6.0, 500.0, 10.0, 0.0], dt=2.0)
        self.assertLess(sensitive.G, resistant.G)

    def test_state_and_action_contract(self):
        state = randomized_dka(DKABody())
        self.assertEqual(s2vec(state).shape, (S_DIM,))
        self.assertEqual(a2vec([4.0, 500.0, 10.0]).shape, (A_DIM,))
        self.assertEqual(A_DIM, 8)
        self.assertEqual(a2vec([4.0, 500.0, 10.0])[3], 0.0)
        self.assertEqual(a2vec([4.0, 500.0, 10.0])[4], 1.0)
        self.assertTrue(np.isfinite(s2vec(state)).all())

    def test_route_aware_insulin_pk_delays_subcutaneous_effect(self):
        iv = DKABody()
        basal = DKABody()
        iv.step(expand_action({"insulin_iv": 6.0}), dt=0.5)
        basal.step(expand_action({"insulin_basal_sc": 40.0}), dt=0.5)
        self.assertGreater(iv.I, basal.I)
        self.assertGreater(basal.insulin_basal_depot, 0.0)
        start = basal.I
        basal.step(expand_action({}), dt=4.0)
        self.assertGreater(basal.I, start)

    def test_hyperosmolar_injury_is_reported_burden_not_terminal_trigger(self):
        body = DKABody()
        body.G = 800.0
        body.Na = 170.0
        body.step([0, 0, 0, 0, 0], dt=0.5)
        self.assertTrue(body.alive)
        self.assertGreater(body.osmotic_injury, 0.0)
        body.osmotic_injury = OSM_INJURY_DEATH + 1.0
        body._check_death(0.5)
        self.assertTrue(body.alive)
        self.assertIsNone(body.death_cause)

    def test_critical_viability_crossing_requires_sustained_burden(self):
        body = DKABody()
        body.HCO3 = 1.0
        body._check_death(0.1)
        self.assertTrue(body.alive)
        self.assertGreater(body.observe()["critical_burden"], 0.0)
        for _ in range(20):
            body._check_death(0.1)
            if not body.alive:
                break
        self.assertFalse(body.alive)
        self.assertEqual(body.death_cause, "acidosis (pH<6.8)")

    def test_critical_burden_recovers_after_transient_crossing(self):
        body = DKABody()
        body.HCO3 = 1.0
        body._check_death(0.1)
        peak = body.observe()["critical_burden"]
        body.HCO3 = 24.0
        body._check_death(0.5)
        self.assertLess(body.observe()["critical_burden"], peak)
        self.assertTrue(body.alive)

    def test_extreme_hyperglycemia_is_reported_burden_not_terminal_trigger(self):
        body = DKABody()
        body.G = 1600.0
        for _ in range(20):
            body._check_death(0.1)
        self.assertTrue(body.alive)
        self.assertIsNone(body.death_cause)
        self.assertGreater(
            body.critical_burdens["extreme hyperglycemia (G>1400)"],
            0.0,
        )

    def test_kcl_replenishes_total_body_potassium_store(self):
        untreated = DKABody()
        replaced = DKABody()
        untreated.step([0, 0, 0, 0, 0], dt=1.0)
        replaced.step([0, 0, 20, 0, 0], dt=1.0)
        self.assertGreater(replaced.Ki, untreated.Ki)

    def test_insulin_shift_does_not_destroy_total_body_potassium(self):
        treated = DKABody()
        untreated = DKABody()
        treated.step({"insulin_iv": 8.0}, dt=1.0)
        untreated.step({}, dt=1.0)
        self.assertLess(treated.Ke, untreated.Ke)
        self.assertAlmostEqual(treated.Ki, untreated.Ki, delta=2.0)

    def test_acidemia_masks_total_body_potassium_depletion_in_serum(self):
        acidotic = DKABody()
        neutral = DKABody()
        acidotic.HCO3 = 6.0
        neutral.HCO3 = 24.0
        acidotic.Ke = neutral.Ke = 4.2
        acidotic.Ki = neutral.Ki = 100.0
        self.assertGreater(
            acidotic._derivs({})["Ke"],
            neutral._derivs({})["Ke"],
        )

    def test_osmotic_potassium_loss_is_bounded_by_urine_flow(self):
        body = DKABody()
        body.G = 900.0
        initial_store = body.Ki
        body.step({}, dt=1.0)
        self.assertGreater(body.Ki, initial_store - 30.0)
        self.assertLess(body.Ki, initial_store)

    def test_osmotic_diuresis_is_volume_guarded(self):
        body = DKABody()
        body.G = 2000.0
        body.V = 0.25 * body.volume_setpoint
        body.step({}, dt=1.0)
        self.assertLess(body.urine_output_ml_hr, 200.0)
        self.assertGreater(body.V, 0.20 * body.volume_setpoint)

    def test_low_volume_dextrose_uses_effective_distribution_volume(self):
        body = DKABody()
        body.V = 1.0
        body.G = 300.0
        body.step({"dextrose": 20.0}, dt=1.0)
        self.assertLess(body.G, 700.0)

    def test_low_volume_bicarbonate_does_not_create_extreme_alkalemia(self):
        body = DKABody()
        body.V = 1.0
        body.HCO3 = 8.0
        body.step({"bicarbonate": 50.0}, dt=1.0)
        self.assertLess(body.pH, 7.7)

    def test_effective_treatment_resolves_counterregulatory_stress(self):
        body = DKABody()
        initial_stress = body.counterregulatory_stress
        for _ in range(12):
            body.step({"insulin_iv": 6.0, "fluids": 500.0, "kcl": 10.0}, dt=0.5)
        self.assertTrue(body.alive)
        self.assertLess(body.counterregulatory_stress, initial_stress)

    def test_insulin_clears_ketones_and_improves_bicarbonate(self):
        body = DKABody()
        body.G = 500.0
        body.HCO3 = 8.0
        body.Ket = 16.0
        start = body.observe()
        for _ in range(8):
            body.step({"insulin_iv": 6.0, "fluids": 250.0, "kcl": 10.0}, dt=0.5)
        end = body.observe()
        self.assertTrue(body.alive)
        self.assertLess(end["BHB"], start["BHB"])
        self.assertGreater(end["HCO3"], start["HCO3"])
        self.assertGreater(end["pH"], start["pH"])

    def test_fluid_sodium_metadata_changes_sodium_trajectory(self):
        saline = DKABody()
        dextrose_water = DKABody()
        saline.step({"fluids": 500.0, "_fluid_sodium_meq_l": 154.0}, dt=1.0)
        dextrose_water.step({"fluids": 500.0, "_fluid_sodium_meq_l": 0.0}, dt=1.0)
        self.assertGreater(saline.Na, dextrose_water.Na)

    def test_free_water_dilutes_sodium_without_inheriting_iv_sodium(self):
        untreated = DKABody()
        free_water = DKABody()
        untreated.step({}, dt=1.0)
        free_water.step({"_free_water_ml": 500.0}, dt=1.0)
        self.assertLess(free_water.Na, untreated.Na)

    def test_greybox_residual_cannot_write_conserved_pools(self):
        class Residual:
            @staticmethod
            def correction(observation, action):
                return {"G": -10.0, "Ki": -1000.0, "V": 1000.0}

        baseline = DKABody()
        corrected = DKABody(residual_model=Residual())
        baseline.step({}, dt=0.5, substeps=1)
        corrected.step({}, dt=0.5, substeps=1)
        self.assertLess(corrected.G, baseline.G)
        self.assertAlmostEqual(corrected.Ki, baseline.Ki, places=4)
        self.assertAlmostEqual(corrected.V, baseline.V, places=4)
        self.assertIn("Ki", FORBIDDEN_KEYS)

    def test_greybox_projection_enforces_rate_bounds(self):
        projected = project_residual(np.full(6, 1e6, dtype=np.float32))
        for index, value in enumerate(projected.values()):
            self.assertLessEqual(value, float(MAX_ABS_RATE[index]))

    def test_greybox_residual_can_drive_synthetic_jepa_generator(self):
        class Residual:
            @staticmethod
            def correction(observation, action):
                return {"G": -20.0}

        baseline = generate_branched_dataset(1, 1, seed=123)
        corrected = generate_branched_dataset(
            1, 1, seed=123, residual_model=Residual()
        )
        baseline_glucose = (
            baseline["states"][:, :, 1, STATE_KEYS.index("G")] * 200.0 + 250.0
        ).mean()
        corrected_glucose = (
            corrected["states"][:, :, 1, STATE_KEYS.index("G")] * 200.0 + 250.0
        ).mean()
        self.assertLess(corrected_glucose, baseline_glucose)

    def test_viability_audit_maps_fatal_thresholds_to_hard_mechanisms(self):
        self.assertEqual(
            classify_death_cause("cumulative hyperosmolar injury"),
            "osmotic_injury",
        )
        self.assertEqual(
            classify_death_cause("hypokalemia (K<2.5)"),
            "potassium_mass",
        )
        self.assertEqual(
            classify_death_cause("circulatory collapse (MAP<40)"),
            "volume_map",
        )

    def test_viability_audit_measures_threshold_excess(self):
        observation = {
            "G": 250.0,
            "pH": 7.1,
            "Ke": 2.2,
            "MAP": 55.0,
            "osmotic_injury": 12.75,
        }
        self.assertAlmostEqual(
            threshold_excess("hypokalemia (K<2.5)", observation),
            0.3,
        )
        self.assertAlmostEqual(
            threshold_excess("cumulative hyperosmolar injury", observation),
            0.75,
        )

    def test_viability_audit_flags_unobserved_insulin_rescue(self):
        trajectory = {"_cover": {"insulin_iv": 0, "insulin_rapid_sc": 0}}
        reasons = coverage_artifact_reasons(
            trajectory,
            "acidosis (pH<6.8)",
            {"G": 950.0, "HCO3": 1.0, "anion_gap": 50.0},
            [{"t": 1.0, "var": "glucose", "value": 120.0}],
        )
        self.assertIn("no_captured_insulin_but_later_metabolism_improves", reasons)

    def test_numeric_artifact_audit_tracks_volume_denominator_guards(self):
        self.assertEqual(BOUND_REGISTRY["V"]["floor"], 1.0)
        self.assertEqual(
            DENOMINATOR_GUARDS["concentration_volume"]["status"],
            "protected",
        )
        self.assertEqual(
            DENOMINATOR_GUARDS["sodium_balance_volume"]["status"],
            "bounded_not_distribution_protected",
        )

    def test_power_analysis_refuses_wrong_signed_candidate(self):
        self.assertIsNone(required_stays_for_win(mean_delta=0.02, sd_delta=0.1))
        self.assertGreater(required_stays_for_win(mean_delta=-0.02, sd_delta=0.1), 0)

    def test_anchored_residual_requires_prolog_direction(self):
        gate = AnchoredResidualGate(AnchoredResidualConfig(supported_horizon_hours=1.0))
        action = np.zeros(A_DIM, dtype=np.float32)
        decision = gate.select(
            STATE_KEYS.index("G"), 300.0, 250.0, action,
            direction_agreement=1.0,
        )
        self.assertFalse(decision["allowed_to_leave_persistence"])
        self.assertEqual(decision["selected_prediction"], 300.0)
        self.assertIn("no_active_prolog_direction", decision["abstain_reasons"])

    def test_anchored_residual_blocks_prolog_contradiction(self):
        gate = AnchoredResidualGate(AnchoredResidualConfig(supported_horizon_hours=1.0))
        action = np.zeros(A_DIM, dtype=np.float32)
        action[ACTION_INDEX["fluids"]] = 0.5
        direction = prolog_direction_gate(
            STATE_KEYS.index("MAP"), action, horizon_hours=1.0
        )
        self.assertEqual(direction["allowed_sign"], 1)
        decision = gate.select(
            STATE_KEYS.index("MAP"), 70.0, 60.0, action,
            direction_agreement=1.0, horizon_hours=1.0,
        )
        self.assertFalse(decision["allowed_to_leave_persistence"])
        self.assertEqual(decision["selected_prediction"], 70.0)
        self.assertIn("candidate_direction_contradicts_prolog", decision["abstain_reasons"])

    def test_anchored_residual_shrinks_rule_consistent_move(self):
        gate = AnchoredResidualGate(AnchoredResidualConfig(supported_horizon_hours=1.0))
        action = np.zeros(A_DIM, dtype=np.float32)
        action[ACTION_INDEX["fluids"]] = 0.5
        decision = gate.select(
            STATE_KEYS.index("MAP"), 70.0, 90.0, action,
            direction_agreement=1.0, horizon_hours=1.0,
        )
        self.assertTrue(decision["allowed_to_leave_persistence"])
        self.assertGreater(decision["selected_prediction"], 70.0)
        self.assertLess(decision["selected_prediction"], 90.0)
        self.assertEqual(decision["selected_source"], "anchored_residual")

    def test_world_model_is_action_conditioned(self):
        torch.manual_seed(0)
        model = WorldModel().eval()
        state = torch.zeros(1, S_DIM)
        latent = model.E(state)
        no_action = torch.zeros(1, A_DIM)
        insulin = torch.from_numpy(a2vec([6.0, 0.0, 0.0])).unsqueeze(0)
        self.assertFalse(torch.allclose(
            model.predict_latent(latent, no_action),
            model.predict_latent(latent, insulin),
        ))

    def test_world_model_exposes_structured_symbolic_heads(self):
        model = WorldModel().eval()
        latent = model.encode_state(
            torch.zeros(2, S_DIM), torch.zeros(2, H_DIM)
        )
        outputs = model.symbolic_outputs(latent, torch.zeros(2, A_DIM))
        self.assertEqual(outputs["direction_logits"].shape, (2, S_DIM, 3))
        self.assertEqual(outputs["status_logits"].shape, (2, 3))
        self.assertEqual(outputs["proof_logits"].shape, (2, len(RULE_IDS)))
        self.assertEqual(outputs["proposal_logits"].shape, (2, S_DIM, 3))

    def test_current_v5_report_fails_persistence_gate(self):
        report = json.loads(Path("dka_symbolic_jepa_v5_report.json").read_text())
        gate = evaluate_persistence_gate(report)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("glucose" in reason for reason in gate["reasons"]))

    def test_persistence_gate_enables_dynamics_weights_only_after_pass(self):
        metrics = {
            name: {"n": 40, "jepa": 0.8, "persistence": 1.0}
            for name in ("glucose", "potassium", "bicarbonate", "map")
        }
        gate = evaluate_persistence_gate({
            "stays": 40,
            "data_adequacy": {
                "has_exact_dose_and_timing": True,
                "has_explicit_start_stop_events": True,
                "has_pre_anchor_treatment_history": True,
            },
            "mae_active_dka": metrics,
        })
        self.assertTrue(gate["passed"])
        self.assertEqual(weights_for_stage(STAGES[-1], False)["viability"], 0.0)
        self.assertGreater(weights_for_stage(STAGES[-1], True)["viability"], 0.0)

    def test_symbolic_supervision_marks_rule_contradictions(self):
        action = torch.zeros(1, 1, A_DIM)
        action[..., ACTION_INDEX["insulin_iv"]] = 1.0
        effect = torch.zeros(1, 1, S_DIM)
        effect[..., 0] = -0.1
        effect[..., 4] = -0.1
        status, proof = rule_supervision(effect, action)
        self.assertEqual(int(status.item()), 0)
        self.assertGreaterEqual(int(proof.sum()), 2)
        effect[..., 0] = 0.1
        status, _ = rule_supervision(effect, action)
        self.assertEqual(int(status.item()), 1)

    def test_rule_sandbox_rejects_active_rule_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = RuleSandbox(directory)
            rule = {
                "rule_id": "unsafe", "level": "active_safety_critical",
                "context": {}, "action": {}, "predicted_transition": {},
                "confidence": 1.0, "evidence": {}, "provenance": {},
                "status": "active",
            }
            with self.assertRaises(ValueError):
                sandbox.write_candidates("test", {"rules": [rule]})

    def test_rule_gate_rejects_confounded_action_specific_claim(self):
        candidate = {
            "rule_id": "confounded", "active_rule_conflict": None,
            "action": {"name": "insulin_iv"},
            "predicted_transition": {
                "model_variable": "G", "direction": "decrease"
            },
            "evidence": {
                "discovery_factual_direction_accuracy": 1.0,
                "proposal_direction_consistency": 1.0,
                "patient_stay_count": 3,
                "isolated_action_support_count": 0,
            },
            "level": "observational_association", "status": "candidate",
        }
        records = []
        for stay_id in (1, 2, 3):
            exposure = np.zeros(A_DIM, dtype=np.float32)
            exposure[ACTION_INDEX["insulin_iv"]] = 1.0
            exposure[ACTION_INDEX["fluids"]] = 1.0
            directions = np.full(S_DIM, -1, dtype=np.int64)
            directions[0] = 0
            records.append({
                "stay_id": stay_id,
                "action_exposure": exposure,
                "actual_directions": directions,
            })
        result = validate_candidates(
            [candidate], records,
            list(ACTION_INDEX),
            ["G", *[f"state_{index}" for index in range(1, S_DIM)]],
        )[0]
        self.assertFalse(
            result["heldout_test"]["passes_automated_retrospective_gate"]
        )
        self.assertFalse(
            result["heldout_test"]["action_specificity_identifiable"]
        )

    def test_treatment_episode_marks_isolated_action(self):
        sequence = np.zeros((12, A_DIM), dtype=np.float32)
        sequence[:, ACTION_INDEX["fluids"]] = 0.5
        exposure, episode = episode_features(sequence)
        self.assertGreater(exposure[ACTION_INDEX["fluids"]], 0.0)
        self.assertEqual(float(episode[1]), 1.0)
        self.assertEqual(float(episode[2]), 0.0)
        self.assertEqual(float(episode[3]), 1.0)

    def test_real_world_adapter_cannot_claim_causality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.joblib"
            joblib.dump({
                "symbolic_calibration": {
                    "temperature": 2.0,
                    "recommended_min_ensemble_agreement": 0.9,
                },
                "models": {
                    "G": {
                        "target_index": 0, "method": "persistence",
                        "mode": None, "alpha": None, "members": [],
                    },
                },
            }, path)
            adapter = RealWorldAdapter(path)
            state = {
                "G": 300.0, "pH": 7.1, "HCO3": 10.0,
                "anion_gap": 22.0, "Ke": 4.5, "MAP": 70.0,
                "V": 11.0, "I": 1.0,
            }
            result = adapter.apply(
                state,
                np.zeros(S_DIM, dtype=np.float32),
                np.zeros((12, A_DIM), dtype=np.float32),
            )
            self.assertEqual(result["state"]["G"], 300.0)
            self.assertFalse(result["causal_intervention_claim_allowed"])
            self.assertEqual(
                result["state_selection"]["G"]["selected_source"],
                "persistence",
            )
            unsupported = adapter.apply(
                state,
                np.zeros(S_DIM, dtype=np.float32),
                np.zeros((48, A_DIM), dtype=np.float32),
                horizon_hours=24.0,
            )
            self.assertFalse(unsupported["hybrid_contract"]["horizon_supported"])
            self.assertIn(
                "unsupported_horizon",
                unsupported["state_selection"]["G"]["abstain_reasons"],
            )

    def test_aipw_and_matching_recover_synthetic_treatment_effect(self):
        rng = np.random.default_rng(18)
        count = 300
        severity = rng.normal(size=count)
        probability = 1.0 / (1.0 + np.exp(-severity))
        treatment = rng.binomial(1, probability)
        baseline = 300.0 + 30.0 * severity
        change = -40.0 * treatment + 12.0 * severity + rng.normal(0, 5, count)
        frame = pd.DataFrame({
            "stay_id": np.arange(count),
            "glucose_t": baseline,
            "glucose_age_hr": rng.uniform(0, 2, count),
            "glucose_tp6": baseline + change,
            "act_insulin": treatment,
            "act_fluids": rng.binomial(1, 0.2, count),
        })
        report = evaluate_trial(
            frame,
            TargetTrialSpec(
                "synthetic", "act_insulin", "glucose_tp6", "glucose_t"
            ),
            bootstrap_samples=100,
        )
        self.assertTrue(report["available"])
        self.assertLess(report["aipw_ate"]["estimate"], -30.0)
        self.assertGreater(report["aipw_ate"]["estimate"], -50.0)
        self.assertEqual(
            report["propensity_matched_att"]["same_stay_matches"], 0
        )
        self.assertFalse(report["causal_claim_allowed"])

    def test_low_potassium_blocks_insulin(self):
        safe, trace = shield(
            {"Ke": 2.8, "G": 350.0, "pH": 7.1, "MAP": 75.0},
            [6.0, 0.0, 0.0],
        )
        self.assertEqual(safe[0], 0.0)
        self.assertEqual(safe[2], 20.0)
        self.assertTrue(trace)

    def test_hypotension_adds_fluids(self):
        safe, trace = shield(
            {"Ke": 4.2, "G": 300.0, "pH": 7.1, "MAP": 50.0},
            [2.0, 0.0, 0.0],
        )
        self.assertEqual(safe[1], 500.0)
        self.assertTrue(any("MAP" in item for item in trace))

    def test_unresolved_ketoacidosis_adds_dextrose_below_250(self):
        safe, trace = shield_full(
            {
                "Ke": 4.0, "G": 220.0, "pH": 7.2, "HCO3": 14.0,
                "anion_gap": 20.0, "BHB": 4.0, "MAP": 75.0,
            },
            [4.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(safe[0], 4.0)
        self.assertGreaterEqual(safe[4], 5.0)
        self.assertTrue(any("dextrose" in item for item in trace))

    def test_history_features_distinguish_recent_insulin(self):
        no_history = treatment_history_features([])
        recent = treatment_history_features([[6.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(no_history.shape, (H_DIM,))
        self.assertGreater(recent[0], no_history[0])
        self.assertLess(recent[A_DIM], no_history[A_DIM])

    def test_mimic_normalizer_splits_dextrose_fluid_and_preserves_history(self):
        anchor = pd.Timestamp("2026-01-01 12:00:00")
        raw = pd.DataFrame([{
            "subject_id": 1,
            "stay_id": 2,
            "starttime": anchor - pd.Timedelta(hours=1),
            "endtime": anchor + pd.Timedelta(hours=1),
            "label": "Dextrose 5% in Normal Saline",
            "amount": 200.0,
            "uom": "mL",
            "rate": None,
            "rate_uom": None,
            "source": "inputevents",
        }])
        events = normalize_events(raw)
        self.assertEqual(set(events["action"]), {"dextrose", "fluids"})
        summary = action_window_summary(events, anchor)
        self.assertGreater(summary["hist_dextrose_total"], 0.0)
        self.assertGreater(summary["act_fluids_total"], 0.0)
        self.assertIn("future_treatment_event_grid", summary)
        self.assertEqual(summary["future_fluid_sodium_grid"][0], 154.0)
        self.assertEqual(summary["future_action_quality"]["exact_timing_fraction"], 1.0)

    def test_mimic_events_preserve_carried_in_start_and_exact_stop(self):
        anchor = pd.Timestamp("2026-01-01 12:00:00")
        events = pd.DataFrame([{
            "action": "insulin_iv",
            "starttime": anchor - pd.Timedelta(minutes=15),
            "endtime": anchor + pd.Timedelta(minutes=45),
        }])
        records = treatment_event_records(events, anchor, 2.0)
        self.assertEqual(records[0]["event_type"], "start")
        self.assertTrue(records[0]["carried_in"])
        self.assertEqual(records[1]["event_type"], "stop")
        self.assertEqual(records[1]["hour"], 0.75)

    def test_overlapping_infusions_do_not_create_false_stop(self):
        anchor = pd.Timestamp("2026-01-01 12:00:00")
        events = pd.DataFrame([
            {"action": "fluids", "starttime": anchor,
             "endtime": anchor + pd.Timedelta(hours=1)},
            {"action": "fluids", "starttime": anchor + pd.Timedelta(minutes=30),
             "endtime": anchor + pd.Timedelta(hours=1.5)},
        ])
        records = treatment_event_records(events, anchor, 2.0)
        self.assertEqual(
            [(item["event_type"], item["hour"]) for item in records],
            [("start", 0.0), ("stop", 1.5)],
        )

    def test_mimic_normalizer_recognizes_nacl_lr_and_excludes_flushes(self):
        now = pd.Timestamp("2026-01-01 12:00:00")
        raw = pd.DataFrame([
            {"stay_id": 1, "starttime": now, "endtime": now + pd.Timedelta(hours=1),
             "label": "NaCl 0.9%", "amount": 500, "uom": "mL", "source": "inputevents"},
            {"stay_id": 1, "starttime": now, "endtime": now + pd.Timedelta(hours=1),
             "label": "LR", "amount": 500, "uom": "mL", "source": "inputevents"},
            {"stay_id": 1, "starttime": now, "endtime": now,
             "label": "Sodium Chloride 0.9% Flush", "amount": 10, "uom": "mL", "source": "emar"},
        ])
        events = normalize_events(raw)
        self.assertEqual(len(events), 2)
        self.assertEqual(set(events["action"]), {"fluids"})

    def test_mimic_normalizer_spreads_bolus_and_deduplicates_emar(self):
        now = pd.Timestamp("2026-01-01 12:00:00")
        raw = pd.DataFrame([
            {"stay_id": 1, "starttime": now, "endtime": now,
             "label": "Insulin - Glargine", "amount": 20, "uom": "unit",
             "source": "inputevents"},
            {"stay_id": 1, "starttime": now + pd.Timedelta(minutes=5),
             "endtime": now + pd.Timedelta(minutes=5), "label": "Insulin Glargine",
             "amount": 20, "uom": "unit", "source": "emar"},
        ])
        events = deduplicate_events(normalize_events(raw))
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["action"], "insulin_basal_sc")
        self.assertEqual(events.iloc[0]["endtime"] - events.iloc[0]["starttime"],
                         pd.Timedelta(hours=0.5))

    def test_mimic_normalizer_does_not_treat_kcl_carrier_ml_as_meq(self):
        now = pd.Timestamp("2026-01-01 12:00:00")
        raw = pd.DataFrame([{
            "stay_id": 1, "starttime": now,
            "endtime": now + pd.Timedelta(hours=1),
            "label": "KCL (Bolus)", "itemid": 227521,
            "amount": 50, "uom": "mL", "source": "inputevents",
        }])
        self.assertTrue(normalize_events(raw).empty)

    def test_mimic_normalizer_converts_bicarbonate_amp_ml_to_meq(self):
        now = pd.Timestamp("2026-01-01 12:00:00")
        raw = pd.DataFrame([{
            "stay_id": 1, "starttime": now, "endtime": now,
            "label": "Sodium Bicarbonate 8.4%", "itemid": 227533,
            "amount": 50, "uom": "mL", "source": "inputevents",
        }])
        event = normalize_events(raw).iloc[0]
        self.assertEqual(event["action"], "bicarbonate")
        self.assertEqual(event["amount"], 50.0)
        self.assertEqual(event["timing_source"], "default_duration")

    def test_ingredientevents_add_observed_maintenance_without_calorie_guessing(self):
        now = pd.Timestamp("2026-01-01 12:00:00")
        raw = pd.DataFrame([
            {"stay_id": 1, "starttime": now, "endtime": now + pd.Timedelta(hours=1),
             "ingredient_itemid": 227075, "input_label": "Free Water",
             "amount": 200, "uom": "mL", "orderid": 1},
            {"stay_id": 1, "starttime": now, "endtime": now + pd.Timedelta(hours=2),
             "ingredient_itemid": 226221, "input_label": "Glucerna 1.2",
             "amount": 120, "uom": "mL", "orderid": 2},
            {"stay_id": 1, "starttime": now, "endtime": now + pd.Timedelta(hours=1),
             "ingredient_itemid": 226060, "input_label": "Calories",
             "amount": 300, "uom": "kcal", "orderid": 3},
        ])
        events = normalize_maintenance_events(raw)
        self.assertEqual(set(events["maintenance"]), {"free_water", "enteral_nutrition"})
        summary = maintenance_window_summary(events, now)
        self.assertEqual(summary["act_free_water_total"], 200.0)
        self.assertEqual(summary["act_enteral_nutrition_total"], 120.0)
        self.assertEqual(summary["act_carbohydrate_total"], 0.0)

    def test_empirical_action_prior_samples_observed_support(self):
        grid = np.zeros((12, A_DIM), dtype=np.float32)
        grid[:, ACTION_INDEX["fluids"]] = np.linspace(10, 120, 12)
        grid[:4, ACTION_INDEX["insulin_iv"]] = 4.0
        grid[4:8, ACTION_INDEX["insulin_rapid_sc"]] = 8.0
        frame = pd.DataFrame({
            "stay_id": [1, 2],
            "future_action_grid": [json.dumps(grid.tolist()), json.dumps(grid.tolist())],
        })
        prior = EmpiricalActionPrior.fit(frame)
        rng = np.random.default_rng(2)
        for _ in range(50):
            action = prior.sample(rng)
            self.assertLessEqual(int((action[:4] > 0).sum()), 1)
            constrained = prior.constrain(np.full(A_DIM, 1e6))
            self.assertLessEqual(
                constrained[ACTION_INDEX["fluids"]],
                prior.payload["channels"]["fluids"]["positive_quantiles"][-1],
            )
            self.assertEqual(
                constrained[ACTION_INDEX["insulin_intermediate_sc"]], 0.0
            )

    def test_dka_onset_requires_evidence_in_one_temporal_window(self):
        start = pd.Timestamp("2026-01-01 00:00:00")
        measurements = pd.DataFrame([
            {"stay_id": 1, "charttime": start, "var": "glucose", "valuenum": 300},
            {"stay_id": 1, "charttime": start + pd.Timedelta(hours=2),
             "var": "bicarbonate", "valuenum": 12},
            {"stay_id": 1, "charttime": start + pd.Timedelta(hours=3),
             "var": "anion_gap", "valuenum": 20},
            {"stay_id": 2, "charttime": start, "var": "glucose", "valuenum": 300},
            {"stay_id": 2, "charttime": start + pd.Timedelta(hours=8),
             "var": "bicarbonate", "valuenum": 12},
            {"stay_id": 2, "charttime": start + pd.Timedelta(hours=9),
             "var": "anion_gap", "valuenum": 20},
        ])
        onsets = find_dka_onset(measurements, cooccur_hours=4.0)
        self.assertEqual(onsets["stay_id"].tolist(), [1])
        self.assertLessEqual(onsets.iloc[0]["onset_evidence_span_hours"], 4.0)

    def test_fidelity_replay_uses_prior_insulin_and_observed_map(self):
        body = init_body(
            {"glucose": 300, "HCO3": 12, "K": 4.5, "anion_gap": 22, "MAP": 60},
            [{"t": -0.5, "insulin": 4.0}],
        )
        self.assertGreater(body.I, body.profile.endogenous_insulin)
        self.assertAlmostEqual(body.MAP, 60.0, places=4)

    def test_state_ontology_unifies_symbolic_and_dka_names(self):
        self.assertEqual(
            OSLER_STATE_ONTOLOGY.require("G"),
            "metabolic.glucose",
        )
        self.assertEqual(
            OSLER_STATE_ONTOLOGY.require("systemic_BP"),
            "cardiovascular.map",
        )
        self.assertEqual(
            OSLER_STATE_ONTOLOGY.require("hypoxia"),
            "respiratory.hypoxia",
        )
        flags = OSLER_STATE_ONTOLOGY.derive_flags({"Ke": 5.8, "MAP": 55})
        self.assertIn("hyperkalemia", flags)
        self.assertIn("hypotension", flags)

    def test_state_compiler_is_shared_with_prolog_grounding(self):
        state = {
            "G": 480.0,
            "pH": 6.95,
            "HCO3": 8.0,
            "anion_gap": 25.0,
            "Ke": 2.8,
            "MAP": 50.0,
            "V": 11.0,
            "K_store": 60.0,
        }
        compiled = DKA_STATE_COMPILER.compile_mapping(state)
        grounded = ground_dka_facts(state)
        prolog = {
            atom.predicate for atom in OSLER_DKA_PROLOG._state_facts(state)
        }
        self.assertEqual(grounded, frozenset(prolog))
        self.assertIn("critical_hypokalemia", compiled.facts)
        self.assertIn("severe_acidosis", compiled.facts)
        self.assertEqual(
            len(compiled.symbolic_vector), len(DKA_SYMBOLIC_FACT_KEYS)
        )
        self.assertEqual(len(compiled.residual_vector), S_DIM)
        self.assertGreater(compiled.residual_vector[STATE_KEYS.index("G")], 0)
        self.assertLess(compiled.residual_vector[STATE_KEYS.index("pH")], 0)
        tensor_context = DKA_STATE_COMPILER.tensor_context(
            torch.from_numpy(s2vec(state)).reshape(1, -1)
        )[0]
        self.assertTrue(torch.equal(
            tensor_context[:len(DKA_SYMBOLIC_FACT_KEYS)],
            torch.from_numpy(compiled.symbolic_vector),
        ))

    def test_state_compiler_does_not_assert_unobserved_abnormality(self):
        compiled = DKA_STATE_COMPILER.compile_mapping(
            {"G": 500.0, "Ke": 2.5},
            observed=("G",),
        )
        self.assertIn("hyperglycemia", compiled.facts)
        self.assertNotIn("critical_hypokalemia", compiled.facts)
        self.assertEqual(
            compiled.residual_vector[STATE_KEYS.index("Ke")], 0.0
        )

    def test_zero_initialized_compiler_encoder_preserves_legacy_prediction(self):
        model = WorldModel().eval()
        state = torch.from_numpy(np.stack([
            s2vec({"G": 480.0, "Ke": 5.5, "pH": 7.05}),
            s2vec({"G": 90.0, "Ke": 4.0, "pH": 7.4}),
        ])).float()
        self.assertTrue(torch.allclose(model.encode_state(state), model.E(state)))
        self.assertEqual(
            model.uncertainty_logits(model.encode_state(state)).shape,
            state.shape,
        )

    def test_compiler_context_cannot_change_continuous_dynamics(self):
        model = WorldModel().eval()
        state = torch.from_numpy(s2vec({
            "G": 480.0, "Ke": 5.5, "pH": 7.05,
        })).float().reshape(1, -1)
        action = torch.zeros(1, A_DIM)
        before = model.predict_step(state, action)[0]
        with torch.no_grad():
            model.StateCompilerEnc.weight.fill_(0.05)
        after = model.predict_step(state, action)[0]
        self.assertTrue(torch.equal(before, after))

        latent = model.encode_state(state)
        absent = model.symbolic_outputs(
            latent, action,
            compiled_context=torch.zeros(
                1, DKA_STATE_COMPILER.context_dim
            ),
        )["direction_logits"]
        present = model.symbolic_outputs(
            latent, action,
            compiled_context=torch.ones(
                1, DKA_STATE_COMPILER.context_dim
            ),
        )["direction_logits"]
        self.assertFalse(torch.allclose(absent, present))

    def test_viability_reward_is_grounded_in_outcome_not_model_optimism(self):
        objective = HomeostaticWorldModelObjective(DKA_STATE_COMPILER)
        sick = torch.from_numpy(s2vec({
            "G": 500.0, "pH": 6.9, "HCO3": 7.0, "anion_gap": 28.0,
            "Ke": 6.2, "MAP": 48.0, "osmotic_injury": 9.0,
        })).float().reshape(1, 1, 1, -1)
        recovered = torch.from_numpy(s2vec({
            "G": 160.0, "pH": 7.3, "HCO3": 20.0, "anion_gap": 14.0,
            "Ke": 4.2, "MAP": 75.0, "osmotic_injury": 2.0,
        })).float().reshape(1, 1, 1, -1)
        valid = torch.ones(1, 1, 1)
        exact = objective.components(
            sick, sick, torch.zeros_like(sick), valid
        )
        optimistic = objective.components(
            recovered, sick, torch.zeros_like(sick), valid
        )
        self.assertLess(float(exact["world_truth"]), float(optimistic["world_truth"]))
        self.assertLess(float(exact["viability"]), float(optimistic["viability"]))
        self.assertGreater(
            float(objective.grounded_viability_reward(sick, recovered)),
            0.0,
        )

    def test_live_symbolic_matcher_uses_shared_ontology(self):
        drugs = {
            "insulin": {
                "state_effects": [{
                    "organ": "pancreas",
                    "variable": "blood_glucose_elevation",
                    "max_delta": -1.0,
                    "effect_type": "primary",
                }],
            },
        }
        matches = mechanism_candidates(
            [{"organ": "pancreas", "variable": "G", "direction": "low"}],
            drugs,
        )
        self.assertEqual(matches[0]["matched_targets"][0]["canonical_state"],
                         "metabolic.glucose")

    def test_intervention_schema_preserves_route_and_time(self):
        intervention = Intervention(
            name="insulin",
            dose=6.0,
            unit="U/hr",
            route="iv",
            formulation="regular",
            duration_hours=0.5,
            indication="DKA",
        )
        self.assertEqual(intervention.to_dict()["route"], "iv")
        self.assertEqual(intervention.to_dict()["formulation"], "regular")
        self.assertEqual(intervention.to_dict()["indication"], "DKA")

    def test_temporal_action_encoder_uses_elapsed_time(self):
        torch.manual_seed(0)
        encoder = TemporalActionEncoder(A_DIM).eval()
        action = torch.ones(2, A_DIM)
        early = encoder(action, delta_hours=0.5, elapsed_hours=0.0)
        late = encoder(action, delta_hours=0.5, elapsed_hours=6.0)
        self.assertFalse(torch.allclose(early, late))

    def test_osler_penalizes_wrong_intervention_direction(self):
        action = torch.zeros(1, 1, A_DIM)
        action[..., ACTION_INDEX["insulin_iv"]] = 1.0
        valid = torch.ones(1, 1)
        correct = torch.zeros(1, 1, S_DIM)
        correct[..., 0] = -0.2
        correct[..., 4] = -0.1
        wrong = -correct
        correct_loss = OSLER_DKA_VALIDATOR.consistency_loss(correct, action, valid)
        wrong_loss = OSLER_DKA_VALIDATOR.consistency_loss(wrong, action, valid)
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_osler_transition_audit_is_explainable(self):
        audit = OSLER_DKA_VALIDATOR.validate(
            a2vec([6.0, 0.0, 0.0, 0.0, 0.0]),
            {"G": 300.0, "Ke": 4.0},
            {"G": 400.0, "Ke": 4.5},
        )
        self.assertEqual(audit["status"], "verified")
        self.assertTrue(audit["checks"])

    def test_route_aware_shield_preserves_insulin_formulation(self):
        action = expand_action({"insulin_basal_sc": 40.0, "kcl": 0.0})
        safe, trace = shield_route_aware(
            {"Ke": 2.8, "G": 300, "pH": 7.1, "MAP": 75}, action
        )
        self.assertEqual(safe[ACTION_INDEX["insulin_basal_sc"]], 0.0)
        self.assertEqual(safe[ACTION_INDEX["kcl"]], 20.0)
        self.assertTrue(trace)

    def test_osler_uses_hidden_potassium_store_and_osmotic_injury(self):
        safe, trace = shield_full(
            {
                "Ke": 4.1, "K_store": 55.0, "G": 420, "pH": 7.1,
                "MAP": 70, "creatinine": 1.2, "urine_output": 100,
                "osmotic_injury": 9.0,
            },
            [6.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(safe[2], 20.0)
        self.assertEqual(safe[1], 250.0)
        self.assertTrue(any("total-body" in item for item in trace))
        self.assertTrue(any("hyperosmolar" in item for item in trace))

    def test_prediction_uses_elapsed_time_and_returns_osler_validation(self):
        model = WorldModel().eval()
        state = {
            "G": 480.0, "pH": 7.1, "HCO3": 8.0, "anion_gap": 23.0,
            "Ke": 4.2, "MAP": 75.0, "V": 12.0, "I": 1.0,
        }
        with patch.object(model, "predict_latent", wraps=model.predict_latent) as wrapped:
            predict(model, state, [6.0, 500.0, 10.0, 0.0, 0.0], 1.5, "cpu")
        elapsed = [call.kwargs["elapsed_hours"] for call in wrapped.call_args_list]
        self.assertEqual(elapsed, [0.0, 0.5, 1.0])

        result = compare(model, state, [6.0, 500.0, 10.0, 0.0, 0.0], 0.5, "cpu")
        self.assertIn("jepa_symbolic_transition", result)
        self.assertEqual(
            len(result["jepa_symbolic_transition"]["transitions"]), S_DIM
        )
        self.assertIn("osler_transition_validation", result)
        self.assertIn("osler_prolog_reasoning", result)
        self.assertIn(
            "hyperglycemia",
            result["observation_contract"]["compiled_state"]["facts"],
        )
        self.assertEqual(
            result["predicted_intervention_trajectory"][0]["uncertainty"]["status"],
            "unavailable_for_legacy_or_uncalibrated_checkpoint",
        )
        self.assertIn(
            result["osler_prolog_reasoning"]["decision"],
            {"allow", "modify", "block"},
        )
        self.assertIn(
            result["osler_transition_validation"]["status"],
            {"verified", "contradicted", "unexplained"},
        )


if __name__ == "__main__":
    unittest.main()
