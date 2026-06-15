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

from dka_body import DKABody, DKAPatientProfile
from dka_osler import shield, shield_full, shield_route_aware
from dka_action_contract import ACTION_INDEX, expand_action
from dka_world_model import (
    A_DIM, H_DIM, S_DIM, STATE_KEYS, WorldModel, a2vec, observation_context,
    randomized_dka, s2vec, treatment_history_features,
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
from osler_jepa.belief import PotassiumStoreBelief
from osler_jepa.rule_sandbox import RuleSandbox
from osler_jepa.rule_inducer import validate_candidates
from osler_jepa.real_world_adapter import RealWorldAdapter
from osler_jepa.causal_evaluation import TargetTrialSpec, evaluate_trial
from osler_jepa.symbolic import RULE_IDS, rule_supervision
from osler_jepa.shadow import (
    ShadowObserver,
    build_dka_shadow_state,
    recommendation_fingerprint,
)
from osler_jepa.shadow_outcomes import (
    load_shadow_forecast,
    reconcile_from_ledger,
    reconcile_shadow_forecast,
)
from real_world_improvement import episode_features
from predict_dka_intervention import compare, predict
from dka_fidelity_replay import init_body
from mimic_action_history import (
    action_window_summary, deduplicate_events, normalize_events,
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

    def test_hyperosmolar_injury_is_time_dependent(self):
        body = DKABody()
        body.G = 800.0
        body.Na = 170.0
        body.step([0, 0, 0, 0, 0], dt=0.5)
        self.assertTrue(body.alive)
        self.assertGreater(body.osmotic_injury, 0.0)
        for _ in range(12):
            body.step([0, 0, 0, 0, 0], dt=0.5)
            if not body.alive:
                break
        self.assertFalse(body.alive)
        self.assertEqual(body.death_cause, "cumulative hyperosmolar injury")

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

    def test_kcl_replenishes_total_body_potassium_store(self):
        untreated = DKABody()
        replaced = DKABody()
        untreated.step([0, 0, 0, 0, 0], dt=1.0)
        replaced.step([0, 0, 20, 0, 0], dt=1.0)
        self.assertGreater(replaced.Ki, untreated.Ki)

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
            result["osler_prolog_reasoning"]["decision"],
            {"allow", "modify", "block"},
        )
        self.assertIn(
            result["osler_transition_validation"]["status"],
            {"verified", "contradicted", "unexplained"},
        )


if __name__ == "__main__":
    unittest.main()
