from pathlib import Path
import re
import unittest


SCRIPT = (
    Path(__file__).resolve().parent / "demo" / "static" / "forecast-monitoring.js"
).read_text(encoding="utf-8")
MARKUP = (
    Path(__file__).resolve().parent / "demo" / "case_demo.html"
).read_text(encoding="utf-8")
STYLES = (
    Path(__file__).resolve().parent / "demo" / "static" / "forecast-monitoring.css"
).read_text(encoding="utf-8")


def function_body(name, source=SCRIPT):
    """Return the source of a top-level `function name(` declaration.

    Brace matching rather than a regex, so an assertion about what a single
    function does cannot be satisfied by a match somewhere else in the file.
    """
    match = re.search(r"\bfunction\s+" + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise AssertionError("function %s() not found" % name)
    start = source.index("{", match.end() - 1)
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError("unbalanced braces in %s()" % name)


class ForecastMonitoringUiContractTests(unittest.TestCase):
    def test_replay_uses_backend_events_units_and_display_name(self):
        self.assertIn("c.observation_events.map", SCRIPT)
        self.assertIn("c.replay_metadata.variable_units", SCRIPT)
        self.assertIn("c.display_name", SCRIPT)
        self.assertNotIn("forecast_payload", SCRIPT)

    def test_backend_disabled_matching_has_no_ui_tolerance_fallback(self):
        self.assertIn("configured.supported === false", SCRIPT)
        self.assertIn("Awaiting / no configured matching", SCRIPT)
        self.assertNotIn("DEFAULT_MATCH_TOLERANCE", SCRIPT)
        self.assertNotIn("UI default tolerance", SCRIPT)

    def test_degraded_health_gates_every_forecast_request(self):
        self.assertIn("if (!healthy()) throw new Error('Forecast service unavailable')", SCRIPT)
        self.assertIn("if (healthy() && c)", SCRIPT)
        self.assertIn("if (!healthy()) return", SCRIPT)
        self.assertIn("integrity_load", SCRIPT)
        self.assertIn("inference_smoke", SCRIPT)

    def test_fixture_is_explicit_labeled_and_uses_committed_shape(self):
        self.assertIn("data-fm=\"fixture-load\"", SCRIPT)
        self.assertIn("/api/forecast/fixtures", SCRIPT)
        self.assertIn("post_forecast_", SCRIPT)
        self.assertIn("key.slice(-7) === '_prefix'", SCRIPT)
        self.assertIn("Cached research demonstration", SCRIPT)
        self.assertNotIn("/static/forecast-fixtures.json", SCRIPT)

    def test_null_intervals_and_nominal_target_coverage_are_not_misrepresented(self):
        self.assertIn("v === null || v === undefined || v === ''", SCRIPT)
        self.assertIn("Artifact nominal target coverage", SCRIPT)
        self.assertIn("nominal target coverage", SCRIPT)
        self.assertIn("not empirical coverage or accuracy for this replay case", SCRIPT)
        self.assertIn("extend beyond physiological support", SCRIPT)
        self.assertNotIn("90% research interval", SCRIPT)
        self.assertNotIn("Calibrated 89%", SCRIPT)

    def test_forecast_view_has_no_rx_ranking_call_or_accuracy_marketing(self):
        self.assertNotIn("'/api/analyze'", SCRIPT)
        self.assertNotRegex(SCRIPT, r"\b(?:70|80)\s*%")
        self.assertIn("never change drug ranking or dosing", SCRIPT)


class ReasoningGraphLayoutTests(unittest.TestCase):
    def test_reasoning_levels_are_explicit_and_shared_by_layout_and_replay(self):
        levels = function_body("_reasoningLevel", MARKUP)
        self.assertIn("group==='patient')return 0", levels)
        self.assertIn("group==='indication'||group==='flag')return 1", levels)
        self.assertIn("group==='disease')return 2", levels)
        self.assertIn("group==='pathology'||group==='symptom')return 3", levels)
        self.assertIn("group==='target')return 4", levels)
        self.assertIn("return 5", levels)
        self.assertIn("_reasoningLevel(n.group)", function_body("reasoningLayout", MARKUP))
        self.assertIn("_reasoningLevel(n.group)", function_body("buildStages", MARKUP))

    def test_layout_is_deterministic_fixed_and_precomputed_before_replay(self):
        layout = function_body("reasoningLayout", MARKUP)
        render = function_body("renderGraph", MARKUP)
        stage = function_body("_runStage", MARKUP)
        self.assertIn("localeCompare", layout)
        self.assertIn("levelGap=156,rowGap=64", layout)
        self.assertIn("laneGap=160,rowGap=58,blockGap=62", layout)
        self.assertIn("_layout=reasoningLayout(nodes,el.clientWidth<620)", render)
        self.assertIn("fixed:{x:true,y:true},physics:false", render)
        self.assertIn("layout:{improvedLayout:false},physics:false", render)
        self.assertIn("const p=_layout[n.id]", stage)
        self.assertNotIn("barnesHut", render)
        self.assertNotIn("storePositions", render)
        self.assertNotIn("stabilization", render)

    def test_graph_view_expands_and_phone_canvas_has_room_for_two_lanes(self):
        self.assertIn(".split.graph-expanded", STYLES)
        self.assertIn("minmax(300px, .65fr) minmax(0, 1.35fr)", STYLES)
        self.assertRegex(STYLES, r"\.graph-host\s*\{[^}]*height:\s*560px")
        self.assertRegex(
            STYLES,
            r"@media \(max-width: 767px\)[\s\S]*?\.graph-host\s*\{\s*height:\s*780px",
        )


class MedicationEvidenceAccessTests(unittest.TestCase):
    def test_drug_cards_restore_visible_sources_and_live_openfda_state(self):
        cards = function_body("drugCardsHtml", MARKUP)
        evidence = function_body("drugEvidenceHtml", MARKUP)
        self.assertIn("drugEvidenceHtml(c,dm,p)", cards)
        self.assertIn("Sources &amp; evidence", evidence)
        self.assertIn("Live openFDA label loaded", evidence)
        self.assertIn("Live openFDA label unavailable", evidence)
        self.assertIn("Live openFDA not requested", evidence)
        self.assertIn("FDA label · DailyMed SPL", evidence)
        self.assertIn("Mechanism · DrugBank", evidence)
        self.assertIn("Adverse events · Drugs.com", evidence)
        self.assertIn("openfdaRequested", MARKUP)

    def test_pharmacy_links_are_encoded_external_searches_not_live_quotes(self):
        body = function_body("pharmacyLinksHtml", MARKUP)
        self.assertIn("encodeURIComponent(name)", body)
        for vendor in ("CVS", "Walgreens", "Walmart", "GoodRx"):
            self.assertIn("name:'%s'" % vendor, body)
        self.assertIn('target="_blank" rel="noopener noreferrer"', body)
        self.assertIn("illustrative prices", body)
        self.assertIn("not a live quote", body)
        self.assertIn("not a purchase recommendation", body)
        self.assertIn("Access links do not override the Hold safety gate", body)

    def test_rendering_does_not_reorder_or_rescore_candidates(self):
        cards = function_body("drugCardsHtml", MARKUP)
        self.assertIn("res.candidates.map", cards)
        self.assertNotIn(".sort(", cards)
        self.assertNotIn("mechanism_score=", cards)


class ForecastIsUserDrivenTests(unittest.TestCase):
    """Nothing may be forecast before the user asks for it."""

    def test_bootstrap_loads_metadata_only_and_never_forecasts(self):
        body = function_body("bootstrap")
        self.assertIn("loadHealth()", body)
        self.assertIn("loadCases()", body)
        self.assertIn("loadValidation()", body)
        self.assertNotIn("fetchStep", body)
        self.assertNotIn("lastStep", body)
        self.assertNotIn("/api/forecast", body)

    def test_switching_global_view_never_forecasts(self):
        body = function_body("setView")
        self.assertIn("bootstrap()", body)
        self.assertNotIn("fetchStep", body)
        self.assertNotIn("gotoStep", body)
        self.assertNotIn("runSnapshot", body)

    def test_switching_subview_never_forecasts(self):
        body = function_body("setSub")
        self.assertNotIn("fetchStep", body)
        self.assertNotIn("gotoStep", body)
        self.assertNotIn("startReplay", body)

    def test_snapshot_shows_no_forecast_before_an_explicit_run(self):
        body = function_body("renderSnapshot")
        self.assertIn("if (!resp) { root.innerHTML = snapshotIntroHtml(c); return; }", body)
        intro = function_body("snapshotIntroHtml")
        self.assertIn("Run live forecast", intro)
        self.assertIn("No forecast value exists on this page until you run one", intro)
        # the pre-run state may describe the request, never a returned value
        self.assertNotIn("validatedTableHtml", intro)
        self.assertNotIn("chartCardHtml", intro)
        self.assertNotIn("responseFor", intro)

    def test_replay_starts_before_the_first_observation_is_submitted(self):
        render = function_body("renderMonitoring")
        self.assertIn("if (!S.replayStarted) { root.innerHTML = head + replayIntroHtml(c); return; }", render)
        intro = function_body("replayIntroHtml")
        self.assertIn("pre-observation state", intro)
        self.assertIn("withheld — not submitted", intro)
        self.assertNotIn("responseFor", intro)
        bar = function_body("replayBarHtml")
        self.assertIn("Play retrospective replay", bar)
        self.assertIn("none has been submitted to the model", bar)

    def test_reset_returns_to_the_pre_observation_state_without_requesting(self):
        body = function_body("resetReplay")
        self.assertIn("S.replayStarted = false", body)
        self.assertNotIn("fetchStep", body)
        self.assertNotIn("gotoStep", body)

    def test_play_starts_the_clock_without_immediately_jumping_or_requesting(self):
        body = function_body("startReplay")
        self.assertIn("S.replayStarted = true", body)
        self.assertIn("S.step = -1", body)
        self.assertIn("S.replayTime = 0", body)
        self.assertIn("return startPlay()", body)
        self.assertNotIn("gotoStep", body)

    def test_continuous_replay_uses_fixed_anchors_and_no_frame_requests(self):
        play = function_body("startPlay")
        self.assertIn("window.requestAnimationFrame(frame)", play)
        self.assertIn("SPEEDS[S.speedIndex].rate", play)
        self.assertIn("gotoStep(next)", play)
        self.assertNotIn("fetchStep", play)
        self.assertIn("REPLAY_DURATION_MS = 16000", SCRIPT)
        self.assertIn("visual interpolation only", SCRIPT)
        self.assertIn("no intermediate measurement or model inference", SCRIPT)

    def test_pause_seek_and_reset_do_not_request_or_reveal_future(self):
        self.assertNotIn("fetchStep", function_body("stopPlay"))
        seek = function_body("seekReplay")
        self.assertIn("Math.min(replayTime, S.replayFurthestTime)", seek)
        self.assertIn("replayStepAtTime", seek)
        self.assertNotIn("fetchStep", seek)
        rows = function_body("replayRowsHtml")
        self.assertIn("measurement values not in DOM or model prefix", rows)
        self.assertIn("if (i > S.step)", rows)

    def test_replay_stage_has_accessible_transport_and_scrubber(self):
        stage = function_body("replayStageHtml")
        self.assertIn('data-replay-time', stage)
        self.assertIn('data-fm=\"scrub\"', stage)
        self.assertIn('aria-label=\"Seek within elapsed retrospective replay time\"', stage)
        self.assertIn("Retrospective replay — not live monitoring", stage)
        self.assertIn("replayEventsHtml", stage)

    def test_replay_discloses_five_missing_treatment_history_inputs(self):
        stage = function_body("replayStageHtml")
        for field in (
            "hist_fluids",
            "hist_vasopressor",
            "hist_diuretics",
            "hist_renal_replacement",
            "hist_nephrotoxin",
        ):
            self.assertIn(field, stage)
        self.assertIn("supplies zero for these five missing model", stage)
        self.assertIn("fields; zero is missing-input handling", stage)
        self.assertIn("not evidence that no treatment occurred", stage)

    def test_replay_language_never_claims_realtime_detection(self):
        self.assertIn("Retrospective replay — not live monitoring", SCRIPT)
        self.assertIn("visual interpolation only", SCRIPT)
        self.assertIn("no intermediate measurement or model inference", SCRIPT)
        for banned in ("real-time monitoring", "real time monitoring", "deterioration detection"):
            self.assertNotIn(banned, SCRIPT.lower())

    def test_reduced_motion_shortens_replay_and_removes_cursor_effects(self):
        play = function_body("startPlay")
        self.assertIn("var duration = REDUCED ? 2400 : REPLAY_DURATION_MS", play)
        self.assertIn(".rt-point.current .pulse", STYLES)
        self.assertIn("animation: none", STYLES)
        self.assertIn(".rt-cursor", STYLES)


class ForecastRequestProvenanceTests(unittest.TestCase):
    """A result is labelled by where it actually came from."""

    def test_fetch_step_accepts_force_and_only_reuses_a_session_result_without_it(self):
        body = function_body("fetchStep")
        self.assertIn("var force = !!(opts && opts.force)", body)
        self.assertIn("if (!force && S.cache[key]) return S.cache[key]", body)
        self.assertIn("if (!force && S.pending[key]) return S.pending[key]", body)
        self.assertIn("source: 'live'", body)
        self.assertIn("requestedAt: Date.now()", body)
        self.assertIn("prefixRows: payload.trajectory.length", body)
        self.assertIn("forced: force", body)

    def test_every_explicit_run_control_forces_a_live_request(self):
        self.assertIn("fetchStep(c, lastStep(c), { force: true })", function_body("runSnapshot"))
        self.assertIn("'rerun-step'", SCRIPT)
        self.assertIn("gotoStep(S.step, { force: true })", SCRIPT)
        self.assertIn("data-fm=\"rerun-snapshot\"", SCRIPT)
        self.assertIn("data-fm=\"rerun-step\"", SCRIPT)

    def test_three_source_states_are_distinct_and_session_is_never_called_cached(self):
        body = function_body("resultSource")
        self.assertIn("'Cached research demonstration'", body)
        self.assertIn("'Live response'", body)
        self.assertIn("'Previously computed live result'", body)
        # the fixture owns the word "cached"; a session result must not borrow it
        self.assertNotRegex(
            body.replace("'Cached research demonstration'", ""),
            r"[Cc]ached",
        )

    def test_fixture_is_never_loaded_automatically(self):
        # exactly one call site, and it is the explicit click handler
        self.assertEqual(SCRIPT.count("loadFixture();"), 1)
        self.assertIn("else if (action === 'fixture-load') { loadFixture(); }", SCRIPT)
        self.assertNotIn("loadFixture", function_body("bootstrap"))
        self.assertNotIn("loadFixture", function_body("gotoStep"))
        self.assertNotIn("loadFixture", function_body("runSnapshot"))


class WorkflowIsolationTests(unittest.TestCase):
    """Medication Safety and Retrospective Forecast never exchange state."""

    def test_forecast_payload_is_built_from_the_eicu_case_alone(self):
        body = function_body("buildPrefixPayload")
        self.assertIn("patient_id: c.id", body)
        self.assertIn("case_source: c.case_source", body)
        self.assertIn("anchor_hour: anchor", body)
        self.assertIn("trajectory: prefix", body)
        for leaked in ("PATIENTS", "bundle", "candidates", "ranking", "safety", "OslerRx"):
            self.assertNotIn(leaked, body)

    def test_forecast_module_never_reads_rx_state(self):
        self.assertNotIn("PATIENTS", SCRIPT)
        self.assertNotIn("analyze_stream", SCRIPT)
        self.assertNotIn("window.OslerRx.hasResult", SCRIPT)
        # The only Rx touchpoint is the guided demo asking Rx to run its own
        # case. Every reference lives inside startGuided(), and it asks Rx to
        # act — it never reads a safety result back.
        guided = function_body("startGuided")
        self.assertEqual(SCRIPT.count("window.OslerRx"), guided.count("window.OslerRx"))
        self.assertIn("window.OslerRx.runDefaultScenario()", guided)
        self.assertNotIn("=", guided.split("window.OslerRx.runDefaultScenario()")[1].split(";")[0])

    def test_transition_disclosure_is_acknowledged_before_the_workflow_changes(self):
        self.assertIn(
            "The next chapter uses a different deidentified retrospective eICU case and an\n"
            "        independent factual forecast model. No medication-safety result is carried into the forecast.",
            MARKUP,
        )
        self.assertIn('id="transitionAck"', MARKUP)
        commit = function_body("commitTransition")
        self.assertIn("setView('forecast')", commit)
        request = function_body("requestForecastTransition")
        self.assertNotIn("setView(", request)

    def test_global_navigation_names_the_two_independent_workflows(self):
        self.assertIn('data-view="home"', MARKUP)
        self.assertIn(">Medication Safety<", MARKUP)
        self.assertIn(">Retrospective Forecast<", MARKUP)
        self.assertNotIn("Rx &amp; Safety", MARKUP)
        self.assertNotIn("Patient Forecast", MARKUP)
        self.assertNotIn('data-view="monitoring"', MARKUP)
        self.assertIn("Illustrative case scenarios", MARKUP)

    def test_home_view_runs_no_engine(self):
        body = function_body("renderHome")
        self.assertIn("Explore medication safety", body)
        self.assertIn("Explore retrospective forecast", body)
        self.assertIn("Start guided demo", body)
        self.assertNotIn("/api/", body)
        self.assertNotIn("fetchStep", body)


class KeyboardFocusContractTests(unittest.TestCase):
    """Closing overlays never leaves keyboard focus inside hidden content."""

    def test_transition_cancel_restores_focus_and_commit_moves_it_forward(self):
        request = function_body("requestForecastTransition")
        close = function_body("closeTransition")
        commit = function_body("commitTransition")
        self.assertIn("transitionReturnFocus = document.activeElement", request)
        self.assertIn("transitionReturnFocus.focus()", close)
        self.assertIn("closeTransition({ restoreFocus: false })", commit)
        self.assertIn("replayTab.focus()", commit)

    def test_chat_drawer_restores_the_opening_control(self):
        self.assertIn("_drawerFocus=document.activeElement", function_body("openDrawer", MARKUP))
        self.assertIn("_drawerFocus.focus()", function_body("closeDrawer", MARKUP))


class ExplanatoryPresentationTests(unittest.TestCase):
    """The presentation describes the returned document and nothing else."""

    def test_stages_are_conditioned_on_the_actual_response(self):
        body = function_body("stagesFor")
        self.assertIn("if (withPopulation.length)", body)
        self.assertIn("if (withPersonal.length)", body)
        self.assertIn("if (withInterval.length)", body)
        self.assertIn("if (unsupported.length)", body)
        self.assertIn("Target-specific intervals returned where authorized", body)
        self.assertIn("Unsupported cells withheld", body)

    def test_presentation_can_be_skipped_and_replayed_and_respects_reduced_motion(self):
        self.assertIn("prefers-reduced-motion: reduce", SCRIPT)
        self.assertIn("if (REDUCED)", function_body("playStages"))
        self.assertIn("data-fm=\"skip-anim\"", SCRIPT)
        self.assertIn("data-fm=\"replay-presentation\"", SCRIPT)
        self.assertIn("How this forecast was constructed", SCRIPT)

    def test_no_learning_training_or_clinical_language(self):
        self.assertIn(
            "The model is fixed. Forecast outputs update as additional causally available observations are",
            SCRIPT,
        )
        self.assertIn("This is a model-output change, not a treatment effect.", SCRIPT)
        for banned in (
            "algorithm is learning",
            "training live",
            "is training",
            "true hidden physiology",
            "clinical alert",
            "treatment effect of",
        ):
            self.assertNotIn(banned, SCRIPT)

    def test_change_report_is_computed_from_two_real_responses(self):
        body = function_body("diffResponses")
        self.assertIn("if (!prev || !next) return null", body)
        self.assertIn("becameSupported", body)
        self.assertIn("prevTier", body)
        self.assertIn("newTier", body)
        self.assertIn("This cell is now supported at this anchor.", SCRIPT)
        self.assertIn("The trajectory prefix now contains", SCRIPT)
        self.assertIn("relative to the previous replay step.", SCRIPT)

    def test_validation_evidence_keeps_all_seven_statements_verbatim(self):
        body = function_body("renderValidation")
        for statement in (
            "The validation report is based on the complete eICU-CRD 2.0 retrospective dataset.",
            "120 is 12 cells × 10 held-out runs, not 120 patients.",
            "Hospital, care-unit and late/time splits still come from the same database. "
            "This is not independent external validation.",
            "Creatinine@24h records an artifact nominal target coverage of 0.89.",
            "The validation report was submitted together with the model. "
            "It was not independently re-run from the raw data.",
            "The replay case is provenance-labelled eICU CRD Demo 2.0.1. "
            "It is not presented as a held-out accuracy example, and its training or "
            "validation cohort membership is not established.",
        ):
            self.assertIn(statement, body)
        self.assertIn("interval_target_coverage</span> is an artifact", body)
        self.assertIn("nominal target coverage. It is not empirical coverage or accuracy for this replay case.", body)
        self.assertIn("Committed retrospective validation evidence", body)
        self.assertIn("held-out empirical runs passing the 0.87–0.93 acceptance gate", body)
        self.assertIn("minimum ", body)
        self.assertIn("median approximately ", body)
        self.assertIn("maximum ", body)
        self.assertNotIn("was never in a training or validation fold", body)
        self.assertIn("it runs no inference and re-runs no validation", body)
        self.assertNotIn("fetchStep", body)


class ReplayProgressionOverHttpTests(unittest.TestCase):
    """The prefix progression the Replay subview drives, checked end to end."""

    @classmethod
    def setUpClass(cls):
        from demo.demo_app import app

        cls.client = app.test_client()
        cls.case = cls.client.get("/api/monitoring/cases").get_json()["cases"][0]

    def _post_prefix(self, step):
        rows = [
            event["observation"]
            for event in self.case["observation_events"][: step + 1]
        ]
        payload = {
            "patient_id": self.case["id"],
            "case_source": self.case["case_source"],
            "anchor_hour": rows[-1]["hours_since_onset"],
            "trajectory": rows,
        }
        response = self.client.post("/api/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_validated_progression_is_ten_twelve_twelve(self):
        counts = []
        for step in range(3):
            body = self._post_prefix(step)
            self.assertEqual(body["forecasts"][0].get("causal_claim_allowed", False), False)
            counts.append(
                len([f for f in body["forecasts"] if f["tier"] == "validated_artifact"])
            )
        self.assertEqual(counts, [10, 12, 12])

    def test_unsupported_cells_are_fully_null(self):
        body = self._post_prefix(0)
        unsupported = [f for f in body["forecasts"] if f["tier"] == "unsupported"]
        self.assertTrue(unsupported)
        for cell in unsupported:
            self.assertIsNone(cell.get("lower"))
            self.assertIsNone(cell.get("upper"))
            personalized = cell.get("personalized") or {}
            self.assertIsNone(personalized.get("point"))

    def test_intervals_exist_only_on_the_validated_tier(self):
        body = self._post_prefix(2)
        for cell in body["forecasts"]:
            if cell["tier"] != "validated_artifact":
                self.assertIsNone(cell.get("lower"))
                self.assertIsNone(cell.get("upper"))

    def test_an_observation_after_the_anchor_is_still_rejected(self):
        rows = [event["observation"] for event in self.case["observation_events"][:2]]
        payload = {
            "patient_id": self.case["id"],
            "case_source": self.case["case_source"],
            "anchor_hour": rows[0]["hours_since_onset"],
            "trajectory": rows,
        }
        response = self.client.post("/api/forecast", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("after anchor_hour", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
