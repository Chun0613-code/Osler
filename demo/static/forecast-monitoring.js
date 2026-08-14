/* Osler research forecast workflow.
 *
 * Two independent research demonstrations live in this product. This file owns
 * the shell and the second of them:
 *   #view-home        workflow selection · static copy, no model call
 *   #view-forecast    Retrospective Forecast, with three local subviews
 *                       #fx-snapshot     user-requested forecast at an anchor
 *                       #view-monitoring observation-prefix replay
 *                       #fx-validation   committed retrospective evidence
 *
 * Medication Safety (#view-rx) is a different case and a different engine. It
 * is rendered by the inline script in case_demo.html and nothing crosses
 * between them: no case text, patient id, ranking, safety decision or graph
 * state ever enters a forecast request.
 *
 * Hard rules encoded here:
 *   - nothing is forecast until the user asks. bootstrap() loads model health,
 *     the case list and the validation summary, and stops there. Switching a
 *     global view or a subview never issues a forecast;
 *   - every /api/forecast request carries a causal PREFIX of the trajectory
 *     only; anchor_hour is the last prefix observation, and a client-side guard
 *     refuses to send a payload that contains anything after it;
 *   - forecasts are never invented: unsupported cells show no point and no
 *     interval, illustrative cells show no interval, only validated artifacts
 *     show lower/upper;
 *   - a result is labelled by where it came from. A response that just arrived
 *     is a live response; one produced earlier in this browser session is a
 *     previously computed live result; a committed fixture is a cached research
 *     demonstration. The three are never conflated;
 *   - interval-width evidence is labelled as held-out cohort aggregate, never as
 *     shrinkage for this patient;
 *   - a model-health failure or an API error stops forecasting and is shown as
 *     an error; cached fixtures are only ever loaded on an explicit click;
 *   - no diagnosis, causal claim, treatment recommendation or drug-ranking
 *     change.
 *
 * The model is fixed. Forecast outputs update as additional causally available
 * observations are added to the trajectory prefix.
 *
 * Presentation follows the approved design direction: warm ivory surfaces,
 * graphite structure, oxide for action and selected state, teal for research
 * provenance, blue reserved for forecast data. Tier is a filter, not a badge.
 *
 * No external assets: pure DOM + inline SVG so the demo runs with no internet.
 */
(function () {
  'use strict';

  // ── display metadata ───────────────────────────────────────────────────
  // Units describe the eICU demo adapter output. If the backend later ships a
  // `units` map on the case or the forecast response it wins (see unitFor()).
  var DEFAULT_UNITS = {
    creatinine: 'mg/dL', bun: 'mg/dL', urine_output: 'mL/h', map: 'mmHg',
    glucose: 'mg/dL', potassium: 'mEq/L', sodium: 'mEq/L', bicarbonate: 'mEq/L',
    anion_gap: 'mEq/L', heart_rate: 'bpm', respiratory_rate: 'breaths/min',
    o2sat: '%', temperature: '°C'
  };
  var LABELS = {
    creatinine: 'Creatinine', bun: 'BUN', urine_output: 'Urine output', map: 'MAP',
    glucose: 'Glucose', potassium: 'Potassium', sodium: 'Sodium',
    bicarbonate: 'Bicarbonate', anion_gap: 'Anion gap', heart_rate: 'Heart rate',
    respiratory_rate: 'Respiratory rate', o2sat: 'SpO₂', temperature: 'Temperature'
  };
  var DECIMALS = {
    creatinine: 2, bun: 1, urine_output: 0, map: 0, glucose: 0, potassium: 1,
    sodium: 0, bicarbonate: 1, anion_gap: 1, heart_rate: 0, respiratory_rate: 0,
    o2sat: 0, temperature: 1
  };
  var PRIMARY_SERIES = ['creatinine', 'bun', 'urine_output', 'map'];
  var EXTRA_SERIES = ['glucose', 'potassium', 'heart_rate'];
  var SPEEDS = [
    { label: '0.5×', rate: 0.5 },
    { label: '1×', rate: 1 },
    { label: '2×', rate: 2 }
  ];
  var REPLAY_DURATION_MS = 16000;
  var REPLAY_DOMAINS = {
    creatinine: [0, 3], bun: [0, 60], urine_output: [0, 50], map: [40, 110]
  };
  // one ordered explanatory stage every 220 ms — a walk over information that is
  // already on screen, never a wait for it
  var STAGE_MS = 220;

  // design tokens used by the inline SVG charts
  var C = {
    ink: '#171A1C', ink2: '#626762', oxide: '#B64A32', teal: '#27645E',
    blue: '#344F69', divider: '#D1CCC2', dividerSoft: '#E7E3DA',
    surface: '#FCFAF5', nul: '#8B8F89'
  };

  var REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // ── state ──────────────────────────────────────────────────────────────
  var S = {
    booted: false,
    booting: null,
    health: null,
    healthError: null,
    cases: [],
    casesMeta: null,
    casesError: null,
    caseIndex: 0,
    validation: null,
    validationError: null,
    cache: {},          // "<caseId>#<step>" -> forecast response
    meta: {},           // "<caseId>#<step>" -> { source, requestedAt, prefixRows, forced }
    pending: {},        // "<caseId>#<step>" -> promise while in flight
    freshKey: null,     // the one key whose response arrived from the request that just completed
    stepError: null,
    history: [],        // one snapshot row per (step, cell) actually forecast
    sub: 'snapshot',    // snapshot | replay | validation
    caseOpened: false,  // the user has opened the retrospective case
    snapshotRan: false, // a real snapshot forecast has completed at least once
    snapshotBusy: false,
    stageIndex: -1,     // explanatory presentation pointer, -1 = not started
    stageTimers: [],
    replayStarted: false,   // false = pre-observation state, nothing requested
    step: -1,
    playing: false,
    speedIndex: 1,
    playToken: 0,
    replayTime: 0,
    replayFurthestTime: 0,
    replayMaxStep: -1,
    animateData: false, // set for exactly one render after a data change
    fixture: null,      // loaded cached demonstration payloads (explicit click only)
    fixtureNotice: null,
    tier: 'validated',  // which tier the Snapshot grid is filtered to
    showAll: false,     // "Show all 12" on the validated table
    showExtraSeries: false,
    provOpen: false,
    validationRevealed: 0,  // how many evidence groups have been revealed
    validationTimers: [],
    guided: null,       // { chapter: 1 | 2 } while the guided demo is running
    transition: null    // pending workflow change awaiting acknowledgement
  };

  // ── small helpers ──────────────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  /* Strict: a missing/blank/non-numeric field becomes null, never 0. A fail-closed
     `"lower": null` must stay absent — coercing it to 0 would fabricate an interval. */
  function num(v) {
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return null;
    var n = Number(v);
    return isFinite(n) ? n : null;
  }
  function labelFor(t) { return LABELS[t] || String(t || '').replace(/_/g, ' '); }
  function decimalsFor(t) { return DECIMALS[t] == null ? 2 : DECIMALS[t]; }
  function fmtVal(v, target) {
    if (!isNum(v)) return null;
    return v.toFixed(decimalsFor(target));
  }
  function fmtHour(h) {
    if (!isNum(h)) return '—';
    return (Math.round(h * 10) / 10).toFixed(1) + ' h';
  }
  /* The replay anchors are 0.25 h, 6.08 h and 16.33 h. One decimal would round
     the first to 0.3 h and lose the distinction, so the anchor readout keeps the
     precision the source data actually carries. */
  function fmtHourExact(h) {
    if (!isNum(h)) return '—';
    return (Math.round(h * 100) / 100).toFixed(2) + ' h';
  }
  function fmtSigned(v, target) {
    if (!isNum(v)) return '—';
    return (v > 0 ? '+' : '') + v.toFixed(decimalsFor(target));
  }
  function fmtClock(ms) {
    if (!isNum(ms)) return '—';
    var d = new Date(ms);
    var p = function (n) { return String(n).padStart(2, '0'); };
    return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  }
  function el(id) { return document.getElementById(id); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function unitFor(target) {
    var c = activeCase();
    var maps = [
      c && c.replay_metadata && c.replay_metadata.variable_units,
      S.casesMeta && S.casesMeta.units
    ];
    for (var i = 0; i < maps.length; i++) {
      var m = maps[i];
      if (m && typeof m === 'object') {
        var u = m[target];
        if (typeof u === 'string' && u) return u;
        if (u && typeof u === 'object' && typeof u.unit === 'string') return u.unit;
      }
    }
    return DEFAULT_UNITS[target] || '';
  }

  async function getJSON(url, opts) {
    var r;
    try {
      r = await fetch(url, opts);
    } catch (e) {
      throw new Error('Network request failed: ' + (e && e.message ? e.message : url));
    }
    var ct = r.headers.get('content-type') || '';
    var body = null;
    if (ct.indexOf('application/json') >= 0) {
      try { body = await r.json(); } catch (e) { body = null; }
    }
    if (!r.ok) {
      var msg = (body && (body.error || body.message)) || (r.status + ' ' + r.statusText);
      var err = new Error(msg);
      err.status = r.status;
      err.body = body;
      throw err;
    }
    if (body === null) throw new Error('Server did not return JSON (HTTP ' + r.status + ') for ' + url);
    return body;
  }

  // ── case / trajectory accessors ────────────────────────────────────────
  function activeCase() { return S.cases[S.caseIndex] || null; }
  function trajectoryOf(c) {
    if (!c || !Array.isArray(c.observation_events)) return [];
    return c.observation_events.map(function (event) {
      return event && event.observation ? event.observation : {};
    });
  }
  function lastStep(c) { return Math.max(0, trajectoryOf(c).length - 1); }
  function hourAt(c, i) {
    var t = trajectoryOf(c)[i];
    return t ? num(t.hours_since_onset) : null;
  }
  function caseAlias(c, i) {
    if (!c) return 'Demo ICU Patient';
    if (c.display_name) return c.display_name;
    return 'Demo ICU Patient ' + String((i == null ? S.caseIndex : i) + 1).padStart(2, '0');
  }
  function cacheKey(c, step) { return (c && c.id ? c.id : 'case') + '#' + step; }
  function modelVersion() {
    return (S.casesMeta && S.casesMeta.model_version) || (S.health && S.health.model_version) || '—';
  }

  function matchPolicy(c) {
    var meta = (c && c.replay_metadata) || {};
    var configured = meta.forecast_observation_matching;
    if (configured === 'none' || configured === false ||
        (configured && typeof configured === 'object' && configured.supported === false)) {
      return {
        enabled: false,
        source: 'backend replay contract: no configured matching',
        reason: configured && configured.reason ? configured.reason : null
      };
    }
    var tol = configured && typeof configured === 'object'
      ? num(configured.tolerance_hours) : null;
    if (configured && configured.supported === true && tol != null && tol > 0) {
      return { enabled: true, tolerance: tol, source: 'backend replay contract' };
    }
    return {
      enabled: false,
      source: 'backend replay contract: no configured matching'
    };
  }

  // ── health gate ────────────────────────────────────────────────────────
  function healthy() { return !!(S.health && S.health.ready); }
  function integrityHealth() {
    return S.health && S.health.integrity_load ? S.health.integrity_load : null;
  }
  function failedArtifacts() {
    var integrity = integrityHealth();
    if (!integrity) return [];
    return (integrity.checks || []).filter(function (c) { return c.status !== 'loaded'; });
  }

  // ── bootstrap ──────────────────────────────────────────────────────────
  /* Metadata only. No forecast is requested here, so opening the app — or any
     view inside it — never produces a model output the user did not ask for. */
  function bootstrap() {
    if (S.booting) return S.booting;
    S.booting = (async function () {
      await Promise.all([loadHealth(), loadCases(), loadValidation()]);
      S.booted = true;
      renderAll();
    })();
    return S.booting;
  }

  async function loadHealth() {
    S.healthError = null;
    try {
      // /api/health/models answers 503 with a JSON body when degraded.
      var r = await fetch('/api/health/models');
      var ct = r.headers.get('content-type') || '';
      if (ct.indexOf('application/json') < 0) {
        throw new Error('Health endpoint returned HTTP ' + r.status + ' without JSON');
      }
      S.health = await r.json();
    } catch (e) {
      S.health = null;
      S.healthError = e.message || String(e);
    }
  }
  async function loadCases() {
    S.casesError = null;
    try {
      var j = await getJSON('/api/monitoring/cases');
      S.casesMeta = j;
      S.cases = Array.isArray(j.cases) ? j.cases : [];
      if (!S.cases.length) S.casesError = 'The monitoring case list is empty.';
    } catch (e) {
      S.cases = [];
      S.casesError = e.message || String(e);
    }
  }
  async function loadValidation() {
    S.validationError = null;
    try {
      S.validation = await getJSON('/api/forecast/validation-summary');
    } catch (e) {
      S.validation = null;
      S.validationError = e.message || String(e);
    }
  }

  // ── prefix-only forecast request ───────────────────────────────────────
  function buildPrefixPayload(c, step) {
    var traj = trajectoryOf(c);
    var idx = Math.max(0, Math.min(step, traj.length - 1));
    var prefix = traj.slice(0, idx + 1).map(function (row) {
      var copy = {};
      for (var k in row) if (Object.prototype.hasOwnProperty.call(row, k)) copy[k] = row[k];
      return copy;
    });
    var anchor = num(prefix[prefix.length - 1].hours_since_onset);
    if (anchor == null) throw new Error('Observation ' + (idx + 1) + ' has no hours_since_onset.');
    // Guard: the model must never see anything at or after a future time point.
    for (var i = 0; i < prefix.length; i++) {
      var h = num(prefix[i].hours_since_onset);
      if (h == null || h > anchor + 1e-9) {
        throw new Error('Internal guard: refusing to send an observation after anchor_hour.');
      }
    }
    if (prefix.length !== idx + 1 || prefix.length > traj.length) {
      throw new Error('Internal guard: prefix length mismatch.');
    }
    return {
      patient_id: c.id,
      case_source: c.case_source,
      anchor_hour: anchor,
      trajectory: prefix
    };
  }

  /* opts.force issues a real request even when a session result already exists.
     Every explicit "run" or "re-run" control passes it, so the user always gets
     the API call the button promised. */
  async function fetchStep(c, step, opts) {
    var force = !!(opts && opts.force);
    var key = cacheKey(c, step);
    if (!force && S.cache[key]) return S.cache[key];
    if (!healthy()) throw new Error('Forecast service unavailable');
    if (!force && S.pending[key]) return S.pending[key];
    var payload = buildPrefixPayload(c, step);
    var p = getJSON('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (resp) {
      resp._anchor_hour = payload.anchor_hour;
      resp._step = step;
      resp._prefix_rows = payload.trajectory.length;
      S.cache[key] = resp;
      S.meta[key] = {
        source: 'live',
        requestedAt: Date.now(),
        prefixRows: payload.trajectory.length,
        forced: force
      };
      S.freshKey = key;
      delete S.pending[key];
      recordSnapshots(c, step, resp);
      return resp;
    }, function (e) {
      delete S.pending[key];
      throw e;
    });
    S.pending[key] = p;
    return p;
  }

  function responseFor(c, step) {
    if (S.fixture) {
      var f = S.fixture[cacheKey(c, step)] || S.fixture[String(step)];
      if (f) return f;
    }
    return S.cache[cacheKey(c, step)] || null;
  }
  function isPending(c, step) { return !!S.pending[cacheKey(c, step)]; }

  /* Three distinct provenance states. A session result is never called merely
     "cached" — that word belongs to the committed fixture alone. */
  function resultSource(c, step) {
    if (S.fixture && responseFor(c, step)) {
      return { key: 'fixture', label: 'Cached research demonstration', cls: 'na' };
    }
    var key = cacheKey(c, step);
    if (!S.cache[key]) return null;
    var m = S.meta[key] || {};
    if (S.freshKey === key) {
      return { key: 'live', label: 'Live response', cls: 'ok', meta: m };
    }
    return { key: 'session', label: 'Previously computed live result', cls: 'ok', meta: m };
  }

  function sourceLineHtml(c, step) {
    var src = resultSource(c, step);
    if (!src) return '';
    var m = src.meta || {};
    var bits = [];
    if (src.key === 'fixture') {
      bits.push('committed fixture · no model call was made');
    } else {
      bits.push('requested ' + esc(fmtClock(m.requestedAt)));
      if (m.prefixRows != null) bits.push(m.prefixRows + ' prefix row' + (m.prefixRows === 1 ? '' : 's') + ' sent');
      if (m.forced) bits.push('forced live request');
      if (src.key === 'session') bits.push('reused from this browser session');
    }
    return '<div class="source-line"><span class="stat ' + src.cls + '"><i></i>' + esc(src.label) + '</span>' +
      '<span class="d">' + bits.join(' · ') + '</span></div>';
  }

  // ── snapshot history ───────────────────────────────────────────────────
  function cellId(f) { return f.system + '|' + f.target + '|' + f.horizon_hours; }
  function recordSnapshots(c, step, resp) {
    var anchor = num(resp._anchor_hour);
    var caseId = c && c.id ? c.id : 'case';
    S.history = S.history.filter(function (r) {
      return !(r.caseId === caseId && r.step === step);
    });
    (resp.forecasts || []).forEach(function (f) {
      var pers = f.personalized ? num(f.personalized.point) : null;
      var pop = f.population ? num(f.population.point) : null;
      if (f.tier === 'unsupported') return;         // nothing to record: no point exists
      S.history.push({
        caseId: caseId,
        step: step,
        anchorHour: anchor,
        system: f.system,
        target: f.target,
        horizon: num(f.horizon_hours),
        dueHour: anchor != null ? anchor + num(f.horizon_hours) : null,
        tier: f.tier,
        personalized: pers,
        population: pop,
        lower: num(f.lower),
        upper: num(f.upper),
        coverage: num(f.interval_target_coverage),
        cell: cellId(f)
      });
    });
    S.history.sort(function (a, b) {
      return (a.step - b.step) || (a.dueHour - b.dueHour) || a.target.localeCompare(b.target);
    });
  }

  /** Resolve one snapshot against observations the replay has already reached.
   *  Never treats "the nearest observation" as ground truth: without a match
   *  inside the stated tolerance the row stays explicitly unresolved. */
  function resolveSnapshot(row, c, uptoStep) {
    var policy = matchPolicy(c);
    if (!policy.enabled || row.dueHour == null) {
      return { state: 'awaiting', policy: policy };
    }
    var traj = trajectoryOf(c).slice(0, uptoStep + 1);
    var best = null;
    for (var i = 0; i < traj.length; i++) {
      var h = num(traj[i].hours_since_onset);
      var v = num(traj[i][row.target]);
      if (h == null || v == null) continue;
      if (h <= row.anchorHour + 1e-9) continue;            // not a future observation
      var d = Math.abs(h - row.dueHour);
      if (d > policy.tolerance + 1e-9) continue;
      if (!best || d < best.delta) best = { hour: h, value: v, delta: d };
    }
    if (!best) return { state: 'awaiting', policy: policy };
    var out = {
      state: 'observed', policy: policy, hour: best.hour,
      value: best.value, delta: best.delta
    };
    if (isNum(row.lower) && isNum(row.upper)) {
      out.inside = best.value >= row.lower && best.value <= row.upper;
      out.state = out.inside ? 'inside' : 'outside';
    }
    return out;
  }

  function historyFor(c, uptoStep) {
    var caseId = c && c.id ? c.id : 'case';
    return S.history.filter(function (r) {
      return r.caseId === caseId && r.step <= uptoStep;
    }).map(function (r) {
      var res = resolveSnapshot(r, c, uptoStep);
      return { row: r, res: res };
    });
  }

  // ── persistent case context ────────────────────────────────────────────
  function healthStatHtml() {
    if (S.fixture) return '<span class="stat na"><i></i>Not evaluated</span>';
    if (S.healthError) return '<span class="stat bad"><i></i>Health unknown</span>';
    if (!S.health) return '<span class="stat na"><i></i>Checking…</span>';
    var integrity = integrityHealth() || {};
    var inference = S.health.inference_smoke || {};
    var loaded = integrity.loaded_artifacts, expect = integrity.expected_artifacts;
    var ratio = (loaded != null && expect != null) ? loaded + '/' + expect : '';
    if (healthy()) {
      return '<span class="stat ok"><i></i>Healthy <span class="ratio">' + esc(ratio) + '</span></span>' +
        '<button class="disc" type="button" data-fm="toggle-prov">Health detail ⌄</button>';
    }
    return '<span class="stat warn"><i></i>Degraded <span class="ratio">' + esc(ratio) + '</span></span>' +
      '<button class="disc" type="button" data-fm="toggle-prov">Health detail ⌄</button>' +
      '<span class="visually-hidden">integrity ' + esc(integrity.status || 'unknown') +
      ' · inference smoke ' + esc(inference.status || 'unknown') + '</span>';
  }

  /* The result-source fact is a state, not a promise: before any request has
     completed it says so rather than implying a live connection. */
  function sourceStatHtml() {
    if (S.fixture) return '<span class="stat na"><i></i>Cached demonstration</span>';
    var c = activeCase();
    var step = contextStep(c);
    var src = c ? resultSource(c, step) : null;
    if (src) return '<span class="stat ' + src.cls + '"><i></i>' + esc(src.label) + '</span>';
    return '<span class="stat na"><i></i>No forecast requested</span>';
  }

  /* Which anchor the context strip describes: the replay position while the
     replay is running, otherwise the final anchor the Snapshot uses. */
  function contextStep(c) {
    if (!c) return 0;
    if (S.sub === 'replay') return S.replayStarted ? S.step : -1;
    return lastStep(c);
  }

  function caseContextHtml(c) {
    if (!c) return '';
    var meta = c.source_metadata || {};
    var ctx = c.case_context || {};
    var step = contextStep(c);
    var ident = [
      ctx.age ? ctx.age + ' y' : null,
      ctx.gender,
      ctx.unit_type,
      'deidentified retrospective case'
    ].filter(Boolean).join(' · ');
    var visible = (S.sub === 'replay' && !S.replayStarted)
      ? '0 of ' + (lastStep(c) + 1)
      : (step + 1) + ' of ' + (lastStep(c) + 1);
    var anchorFact = (S.sub === 'replay' && !S.replayStarted)
      ? '<div class="fact"><div class="k">Anchor</div><div class="v">not set</div></div>'
      : '<div class="fact"><div class="k">Anchor</div><div class="v">' + esc(fmtHourExact(hourAt(c, step))) + '</div></div>';
    return '<div class="case-head">' +
        '<div class="case-id">' +
          '<div class="case-title">' + esc(caseAlias(c)) + ' · ' + esc(c.title || 'Retrospective trajectory') + '</div>' +
          '<div class="case-meta">' + esc(ident) + ' · ' +
            esc((meta.dataset ? 'eICU CRD Demo' : c.case_source || '') + (meta.version ? ' ' + meta.version : '')) +
            ' · <button class="disc" type="button" data-fm="toggle-prov">Dataset provenance ⌄</button></div>' +
        '</div>' +
        '<div class="case-facts">' +
          '<div class="fact"><div class="k">Model version</div><div class="v mono">' + esc(modelVersion()) + '</div></div>' +
          anchorFact +
          '<div class="fact"><div class="k">Observations visible</div><div class="v">' + esc(visible) + '</div></div>' +
          '<div class="fact sep"><div class="k">Result source</div><div class="v">' + sourceStatHtml() + '</div></div>' +
          '<div class="fact sep"><div class="k">Model health</div><div class="v">' + healthStatHtml() + '</div></div>' +
        '</div>' +
      '</div>' + provenanceHtml(c, Math.max(0, step));
  }

  /* Dataset provenance and artifact verification — collapsed into the case
     header, shown once, never repeated inside a section. */
  function provenanceHtml(c, step) {
    var meta = c.source_metadata || {};
    var ctx = c.case_context || {};
    var admission = ctx.source_admission_label;
    var integrity = integrityHealth() || {};
    var inference = S.health && S.health.inference_smoke ? S.health.inference_smoke : {};
    return '<div class="disc-body" id="provBody" data-open="' + (S.provOpen ? '1' : '0') + '">' +
      '<dl class="kv">' +
        '<dt>Dataset</dt><dd>' + esc(meta.dataset || c.case_source || 'unknown') +
          (meta.version ? ' v' + esc(meta.version) : '') + '</dd>' +
        '<dt>Source</dt><dd class="mono">' + esc(meta.url || 'not supplied') + '</dd>' +
        '<dt>Licence</dt><dd>' + esc(meta.license || 'not supplied') + '</dd>' +
        '<dt>Deidentification</dt><dd>' + esc(meta.deidentification || 'not supplied') + '</dd>' +
        '<dt>Model version</dt><dd class="mono">' + esc(modelVersion()) + '</dd>' +
        '<dt>Current time point</dt><dd>' + esc(fmtHourExact(hourAt(c, step))) +
          ' after ICU admission · observation ' + (step + 1) + ' of ' + (lastStep(c) + 1) + '</dd>' +
        (admission ? '<dt>Source label</dt><dd>' + esc(admission) +
          ' — carried in the source dataset, not a model output</dd>' : '') +
        '<dt>Serialized artifacts loaded</dt><dd class="mono">' +
          esc(integrity.loaded_artifacts != null ? integrity.loaded_artifacts + ' / ' + integrity.expected_artifacts : '—') + '</dd>' +
        '<dt>Inference smoke predictions passed</dt><dd class="mono">' +
          esc(inference.validated_artifacts != null
            ? inference.validated_artifacts + ' / ' + inference.expected_validated_artifacts : '—') + '</dd>' +
        '<dt>Manifest verification</dt><dd>' +
          (S.health && S.health.strict_manifest_verification ? 'strict' : 'not reported') + '</dd>' +
      '</dl>' +
      '<p class="note" style="margin-top:14px">Patient identifiers are not displayed. The source stay key is retained ' +
      'only inside the dataset provenance record. Observation units follow the eICU demo adapter; urine output is an ' +
      'interval-normalised mL/hour rate.</p>' +
      '</div>';
  }

  // ── degraded / error / cached blocks ───────────────────────────────────
  function serviceUnavailableHtml() {
    var failed = failedArtifacts();
    var detail = '';
    if (S.healthError) {
      detail = '<pre>' + esc(S.healthError) + '</pre>';
    } else if (failed.length) {
      detail = '<pre>' + esc(failed.map(function (f) {
        return f.cell + ' → ' + (f.artifact || '?') + ': ' + (f.error || f.status);
      }).join('\n')) + '</pre>';
    } else if (S.health) {
      var integrity = integrityHealth() || {};
      var inference = S.health.inference_smoke || {};
      detail = '<pre>' + esc(
        'integrity_load: ' + (integrity.status || 'unknown') +
        ' (' + (integrity.loaded_artifacts || 0) + '/' + (integrity.expected_artifacts || 0) + ' loaded)\n' +
        'inference_smoke: ' + (inference.status || 'unknown') +
        (inference.error ? '\n' + inference.error : '')
      ) + '</pre>';
    }
    return '<div class="shell"><div class="block">' +
      '<h2>Model degraded — forecasts are withheld</h2>' +
      '<p class="note">Artifact integrity or deterministic inference smoke did not pass, so no forecast is requested. ' +
      'Targets served by the failed artifacts are withheld — they are not shown with wider intervals, and they are not ' +
      'filled from a fallback model. Degraded health never widens an interval. A target is either served by a verified ' +
      'artifact or withheld.</p>' +
      detail + fixtureControlsHtml() + '</div></div>';
  }

  function errorBlockHtml(title, message) {
    return '<div class="shell"><div class="block">' +
      '<h2>' + esc(title) + '</h2>' +
      '<p class="note">No forecast values are shown for this step. Nothing is substituted.</p>' +
      '<pre>' + esc(message) + '</pre>' + fixtureControlsHtml() + '</div></div>';
  }

  /* Cached fixtures are only ever loaded by an explicit click, after a live
     failure. They are never substituted automatically and never blended. */
  function fixtureControlsHtml() {
    if (S.fixture) {
      return '<div class="actions-row">' +
        '<button class="btn" type="button" data-fm="fixture-clear">Run live instead</button></div>';
    }
    return '<div class="actions-row">' +
      '<button class="btn" type="button" data-fm="fixture-load">Load cached research demonstration</button>' +
      '</div><p class="note" style="margin-top:10px">Loads recorded fixture responses if the backend ships them. ' +
      'They are always labelled <b>Cached research demonstration</b> and are never substituted automatically.</p>' +
      (S.fixtureNotice ? '<p class="note">' + esc(S.fixtureNotice) + '</p>' : '');
  }

  function cachedBannerHtml() {
    if (!S.fixture) return '';
    var version = modelVersion();
    return '<div class="shell tight"><div class="notice cached" style="margin:20px 0 0">' +
      '<span class="n">Cached research demonstration</span>' +
      '<span class="d">No model call was made. Every value on this page was captured from a previous run with model ' +
      'version <span class="mono">' + esc(version) + '</span> and is replayed verbatim. Model health is not evaluated ' +
      'in this mode, and the replay controls step through the captured responses rather than requesting new ones. ' +
      'This is not a result computed in this browser session. Refreshing the page leaves this mode.</span>' +
      '<span class="acts"><button class="btn sm" type="button" data-fm="fixture-clear">Run live instead</button></span>' +
      '</div></div>';
  }

  // ── chart ──────────────────────────────────────────────────────────────
  function observationsFor(c, target, uptoStep) {
    var traj = trajectoryOf(c).slice(0, uptoStep + 1);
    var out = [];
    for (var i = 0; i < traj.length; i++) {
      var h = num(traj[i].hours_since_onset);
      var v = num(traj[i][target]);
      if (h != null && v != null) out.push({ hour: h, value: v, age: num(traj[i][target + '_age_hr']) });
    }
    return out;
  }

  function forecastPointsFor(resp, target) {
    if (!resp) return [];
    var anchor = num(resp._anchor_hour);
    if (anchor == null) anchor = num(resp.latest_hour);
    var seen = {};
    var out = [];
    (resp.forecasts || []).forEach(function (f) {
      if (f.target !== target) return;
      if (f.tier === 'unsupported') return;             // no point, nothing to plot
      var h = num(f.horizon_hours);
      var pers = f.personalized ? num(f.personalized.point) : null;
      if (h == null || pers == null) return;
      // one line per horizon: a validated artifact wins over an illustrative duplicate
      var prev = seen[h];
      var rank = f.tier === 'validated_artifact' ? 2 : 1;
      if (prev && prev.rank >= rank) return;
      var item = {
        rank: rank, horizon: h, due: anchor + h, tier: f.tier,
        personalized: pers, population: f.population ? num(f.population.point) : null,
        lower: num(f.lower), upper: num(f.upper)
      };
      seen[h] = item;
      out = out.filter(function (o) { return o.horizon !== h; });
      out.push(item);
    });
    out.sort(function (a, b) { return a.due - b.due; });
    return out;
  }

  function niceTickStep(span) {
    var candidates = [1, 2, 3, 6, 12, 24, 48];
    for (var i = 0; i < candidates.length; i++) {
      if (span / candidates[i] <= 4) return candidates[i];
    }
    return 48;
  }

  /* Observed = solid ink line with filled dots. Forecast = dashed blue with a
     hollow diamond at each due hour. Interval = a thin capped whisker, drawn
     only where the artifact returned one. Anchor = a dashed oxide rule. */
  function chartSvg(opt) {
    var W = 560, H = 186, PL = 46, PR = 14, PT = 14, PB = 30;
    var obs = opt.obs, fcs = opt.forecasts, anchor = opt.anchor;
    var xs = [0], ys = [];
    obs.forEach(function (o) { xs.push(o.hour); ys.push(o.value); });
    fcs.forEach(function (f) {
      xs.push(f.due);
      ys.push(f.personalized);
      if (isNum(f.lower)) ys.push(f.lower);
      if (isNum(f.upper)) ys.push(f.upper);
    });
    if (anchor != null) xs.push(anchor);
    ys = ys.filter(isNum);
    if (!ys.length) return null;
    // Small multiples share one time axis, so the four panels stay comparable.
    var xMin = 0, xMax = isNum(opt.xMax) ? opt.xMax : Math.max.apply(null, xs);
    if (xMax - xMin < 1) xMax = xMin + 1;
    var yMin, yMax;
    if (opt.yDomain && isNum(opt.yDomain[0]) && isNum(opt.yDomain[1])) {
      // A replay step must be comparable with the one before it, so the vertical
      // domain is supplied by the caller and held stable across steps.
      yMin = opt.yDomain[0]; yMax = opt.yDomain[1];
    } else {
      yMin = Math.min.apply(null, ys); yMax = Math.max.apply(null, ys);
    }
    if (yMax - yMin < 1e-6) { yMax = yMin + Math.max(1, Math.abs(yMin) * 0.1); }
    var pad = (yMax - yMin) * 0.16;
    var rawMin = yMin;
    yMin -= pad; yMax += pad;
    // Padding alone must not invent a negative range. If every plotted value is
    // non-negative the axis stops at 0; if the model itself returned a negative
    // bound (it does for urine output) that value is still shown as returned.
    if (rawMin >= 0 && yMin < 0) yMin = 0;
    var X = function (h) { return PL + (h - xMin) / (xMax - xMin) * (W - PL - PR); };
    var Y = function (v) { return PT + (1 - (v - yMin) / (yMax - yMin)) * (H - PT - PB); };

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" preserveAspectRatio="xMidYMid meet">';
    // y axis: three labelled gridlines only
    [yMax, (yMax + yMin) / 2, yMin].forEach(function (v) {
      var y = Y(v);
      s += '<line x1="' + PL + '" y1="' + y.toFixed(1) + '" x2="' + (W - PR) + '" y2="' + y.toFixed(1) +
        '" stroke="' + C.dividerSoft + '" stroke-width="1"/>';
      s += '<text x="' + (PL - 8) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="end" font-size="12" ' +
        'font-family="IBM Plex Mono, monospace" fill="' + C.ink2 + '">' +
        esc(fmtVal(v, opt.target)) + '</text>';
    });
    // x axis ticks — density halves rather than letting labels drop below 11px,
    // and a tick is dropped rather than allowed to collide with the unit label.
    var stepH = niceTickStep(xMax - xMin);
    var xLabel = opt.xLabel || 'h';
    var labelW = xLabel.length * 6.2 + 10;
    s += '<line x1="' + PL + '" y1="' + (H - PB) + '" x2="' + (W - PR) + '" y2="' + (H - PB) +
      '" stroke="' + C.divider + '" stroke-width="1"/>';
    for (var t = 0; t <= xMax + 1e-9; t += stepH) {
      var x = X(t);
      if (x > W - PR - labelW) break;
      s += '<text x="' + x.toFixed(1) + '" y="' + (H - PB + 17) + '" text-anchor="middle" font-size="12" ' +
        'font-family="IBM Plex Mono, monospace" fill="' + C.ink2 + '">' + Math.round(t) + '</text>';
    }
    s += '<text x="' + (W - PR) + '" y="' + (H - PB + 17) + '" text-anchor="end" font-size="12" ' +
      'font-family="IBM Plex Sans, sans-serif" fill="' + C.ink2 + '">' + esc(xLabel) + '</text>';

    // anchor rule
    if (anchor != null) {
      var ax = X(anchor);
      s += '<line x1="' + ax.toFixed(1) + '" y1="' + PT + '" x2="' + ax.toFixed(1) + '" y2="' + (H - PB) +
        '" stroke="' + C.oxide + '" stroke-width="1" stroke-dasharray="3 3"/>';
      s += '<text x="' + (ax + 5).toFixed(1) + '" y="' + (PT + 9) + '" font-size="11" ' +
        'font-family="IBM Plex Sans, sans-serif" fill="' + C.oxide + '">anchor</text>';
    }
    // intervals
    fcs.forEach(function (f) {
      if (!isNum(f.lower) || !isNum(f.upper)) return;
      var x = X(f.due), y1 = Y(f.upper), y2 = Y(f.lower);
      s += '<line x1="' + x.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x.toFixed(1) + '" y2="' + y2.toFixed(1) +
        '" stroke="' + C.ink2 + '" stroke-width="1"/>' +
        '<line x1="' + (x - 4).toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + (x + 4).toFixed(1) + '" y2="' + y1.toFixed(1) +
        '" stroke="' + C.ink2 + '" stroke-width="1"/>' +
        '<line x1="' + (x - 4).toFixed(1) + '" y1="' + y2.toFixed(1) + '" x2="' + (x + 4).toFixed(1) + '" y2="' + y2.toFixed(1) +
        '" stroke="' + C.ink2 + '" stroke-width="1"/>';
    });
    // forecast line: from the last observation through the forecast points
    if (fcs.length) {
      var pts = [];
      if (obs.length) pts.push([X(obs[obs.length - 1].hour), Y(obs[obs.length - 1].value)]);
      fcs.forEach(function (f) { pts.push([X(f.due), Y(f.personalized)]); });
      s += '<polyline fill="none" stroke="' + C.blue + '" stroke-width="1.4" stroke-dasharray="4 3" points="' +
        pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ') + '"/>';
      fcs.forEach(function (f) {
        var x = X(f.due), y = Y(f.personalized), r = 3.6;
        var changed = opt.changed && opt.changed[f.horizon];
        s += '<polygon points="' + x.toFixed(1) + ',' + (y - r).toFixed(1) + ' ' + (x + r).toFixed(1) + ',' + y.toFixed(1) +
          ' ' + x.toFixed(1) + ',' + (y + r).toFixed(1) + ' ' + (x - r).toFixed(1) + ',' + y.toFixed(1) +
          '" fill="' + (changed ? C.oxide : C.surface) + '" stroke="' + (changed ? C.oxide : C.blue) +
          '" stroke-width="1.5"/>';
      });
    }
    // observed
    if (obs.length) {
      s += '<polyline fill="none" stroke="' + C.ink + '" stroke-width="1.6" points="' +
        obs.map(function (o) { return X(o.hour).toFixed(1) + ',' + Y(o.value).toFixed(1); }).join(' ') + '"/>';
      obs.forEach(function (o, i) {
        var isNew = opt.newestHour != null && Math.abs(o.hour - opt.newestHour) < 1e-9;
        s += '<circle cx="' + X(o.hour).toFixed(1) + '" cy="' + Y(o.value).toFixed(1) + '" r="' +
          (isNew ? 4.4 : 3) + '" fill="' + C.ink + '"/>';
        if (isNew) {
          s += '<circle cx="' + X(o.hour).toFixed(1) + '" cy="' + Y(o.value).toFixed(1) +
            '" r="7.5" fill="none" stroke="' + C.oxide + '" stroke-width="1.2"/>';
        }
      });
    }
    return s + '</svg>';
  }

  /* One shared time domain for the whole grid. In the replay it is the whole
     recorded span plus the longest horizon, so the x axis does not move as
     steps are added. The domain is structural — it uses observation times the
     timeline already shows, never a future observation value. */
  function chartDomain(c, resp, uptoStep, series, stable) {
    var xs = [hourAt(c, uptoStep) || 0];
    if (stable) {
      xs.push(hourAt(c, lastStep(c)) || 0);
      var maxH = 0;
      Object.keys(S.cache).forEach(function (k) {
        ((S.cache[k] || {}).forecasts || []).forEach(function (f) {
          var h = num(f.horizon_hours);
          if (h != null && h > maxH) maxH = h;
        });
      });
      xs.push((hourAt(c, lastStep(c)) || 0) + maxH);
    }
    series.forEach(function (t) {
      observationsFor(c, t, uptoStep).forEach(function (o) { xs.push(o.hour); });
      forecastPointsFor(resp, t).forEach(function (f) { xs.push(f.due); });
    });
    var m = Math.max.apply(null, xs.filter(isNum));
    return isNum(m) && m > 0 ? m : null;
  }

  /* Stable vertical domain: the union of the observations the replay has
     reached and every response computed so far for this case. It only grows,
     so a value can never appear to move because the axis moved under it. */
  function stableYDomain(c, target, uptoStep) {
    var caseId = c && c.id ? c.id : 'case';
    var vals = [];
    observationsFor(c, target, uptoStep).forEach(function (o) { vals.push(o.value); });
    Object.keys(S.cache).forEach(function (k) {
      if (k.indexOf(caseId + '#') !== 0) return;
      forecastPointsFor(S.cache[k], target).forEach(function (f) {
        vals.push(f.personalized);
        if (isNum(f.lower)) vals.push(f.lower);
        if (isNum(f.upper)) vals.push(f.upper);
      });
    });
    vals = vals.filter(isNum);
    if (vals.length < 2) return null;
    return [Math.min.apply(null, vals), Math.max.apply(null, vals)];
  }

  function chartCardHtml(c, target, resp, uptoStep, xLabel, xMax, opts) {
    opts = opts || {};
    var obs = observationsFor(c, target, uptoStep);
    var fcs = resp ? forecastPointsFor(resp, target) : [];
    var unit = unitFor(target);
    var anchor = resp ? num(resp._anchor_hour) : hourAt(c, uptoStep);
    var head = '<div class="chart-h"><b>' + esc(labelFor(target)) + '</b><span class="u">' + esc(unit) + '</span>';
    if (obs.length) {
      var last = obs[obs.length - 1];
      head += '<span class="latest">' + esc(fmtVal(last.value, target)) + ' @ ' + esc(fmtHour(last.hour)) + '</span>';
    } else {
      head += '<span class="latest">Not observed</span>';
    }
    head += '</div>';
    var svg = (obs.length || fcs.length)
      ? chartSvg({
          target: target, obs: obs, forecasts: fcs, anchor: anchor, xLabel: xLabel, xMax: xMax,
          yDomain: opts.stable ? stableYDomain(c, target, uptoStep) : null,
          changed: opts.changed && opts.changed[target] ? opts.changed[target] : null,
          newestHour: opts.newestHour
        })
      : null;
    var body = svg || '<div class="chart-none"><b>Not observed</b>No value up to ' + esc(fmtHour(anchor)) +
      '. Nothing is imputed or interpolated. The variable enters the chart at the step where it is first observed.</div>';
    return '<div class="chart">' + head + body + '</div>';
  }

  function chartLegendHtml(withChange) {
    return '<div class="chart-legend">' +
      '<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="' + C.ink +
        '" stroke-width="1.6"/><circle cx="13" cy="5" r="3" fill="' + C.ink + '"/></svg> observed value</span>' +
      '<span><svg width="20" height="12" aria-hidden="true"><polygon points="10,2 14,6 10,10 6,6" fill="' + C.surface +
        '" stroke="' + C.blue + '" stroke-width="1.5"/></svg> personalized point · validated artifact</span>' +
      (withChange
        ? '<span><svg width="20" height="12" aria-hidden="true"><polygon points="10,2 14,6 10,10 6,6" fill="' + C.oxide +
          '" stroke="' + C.oxide + '" stroke-width="1.5"/></svg> changed since the previous replay step</span>'
        : '') +
      '<span><svg width="16" height="14" aria-hidden="true"><line x1="8" y1="1" x2="8" y2="13" stroke="' + C.ink2 +
        '" stroke-width="1"/><line x1="4" y1="1" x2="12" y2="1" stroke="' + C.ink2 + '" stroke-width="1"/>' +
        '<line x1="4" y1="13" x2="12" y2="13" stroke="' + C.ink2 + '" stroke-width="1"/></svg>' +
        ' target-specific research interval · artifact nominal target coverage shown in the table</span>' +
      '<span><svg width="16" height="12" aria-hidden="true"><line x1="8" y1="0" x2="8" y2="12" stroke="' + C.oxide +
        '" stroke-width="1" stroke-dasharray="3 3"/></svg> anchor</span>' +
      '</div>';
  }

  function chartsNoteHtml() {
    return '<p class="note" style="margin-top:12px">Forecast markers are drawn only at their due hour. A point with no ' +
      'interval is drawn as a point, never as a band. Missing values are shown as <i>Not observed</i> and are never ' +
      'interpolated. Raw calibrated statistical intervals are displayed unchanged and may ' +
      'extend beyond physiological support, including below zero; they have no clinical interpretation.</p>';
  }

  // ── forecast tiers ─────────────────────────────────────────────────────
  // counts come from the same deduped lists the panels render, so the tier
  // control and the section headers can never disagree
  function tierCounts(resp) {
    return {
      validated: ofTier(resp, 'validated').length,
      illustrative: ofTier(resp, 'illustrative').length,
      unsupported: ofTier(resp, 'unsupported').length
    };
  }

  function tierBarHtml(resp) {
    var n = tierCounts(resp);
    var mk = function (key, label, count) {
      return '<button class="tier-btn' + (S.tier === key ? ' active' : '') + '" type="button" data-fm="tier" ' +
        'data-tier="' + key + '" aria-pressed="' + (S.tier === key) + '">' + label +
        '<span class="n">' + count + '</span></button>';
    };
    return '<div class="tiers" role="group" aria-label="Forecast tier">' +
      mk('validated', 'Validated', n.validated) +
      mk('illustrative', 'Illustrative', n.illustrative) +
      mk('unsupported', 'Unsupported', n.unsupported) + '</div>';
  }

  function ofTier(resp, tier) {
    var seen = {};
    return ((resp && resp.forecasts) || []).filter(function (f) {
      if (tier === 'validated') return f.tier === 'validated_artifact';
      if (tier === 'illustrative') return f.tier === 'illustrative';
      return f.tier !== 'validated_artifact' && f.tier !== 'illustrative';
    }).filter(function (f) {
      // the same target × horizon can arrive from more than one system; list it once
      var k = f.target + '@' + f.horizon_hours;
      if (seen[k]) return false;
      seen[k] = 1;
      return true;
    }).sort(function (a, b) {
      return a.target.localeCompare(b.target) || (a.horizon_hours - b.horizon_hours);
    });
  }

  /* Collapsed, the grid shows one row per target rather than every horizon of
     whichever target sorts first. Prefer 24 h, else the longest horizon. */
  function onePerTarget(rows) {
    var best = {};
    rows.forEach(function (f) {
      var cur = best[f.target];
      if (!cur) { best[f.target] = f; return; }
      var h = num(f.horizon_hours), ch = num(cur.horizon_hours);
      if (ch === 24) return;
      if (h === 24 || h > ch) best[f.target] = f;
    });
    return Object.keys(best).map(function (k) { return best[k]; })
      .sort(function (a, b) { return a.target.localeCompare(b.target); });
  }

  /* Validated cells only: a lower/upper pair exists solely where the serialized
     patient-specific conformal artifact produced one. */
  function validatedTableHtml(resp, changed) {
    var rows = ofTier(resp, 'validated');
    var shown = S.showAll ? rows : onePerTarget(rows);
    var body = shown.map(function (f) {
      var unit = unitFor(f.target);
      var lo = num(f.lower), hi = num(f.upper);
      var cov = num(f.interval_target_coverage);
      var interval = (isNum(lo) && isNum(hi))
        ? esc(fmtVal(lo, f.target)) + ' – ' + esc(fmtVal(hi, f.target)) + ' ' + esc(unit)
        : '<span class="nullv">null</span>';
      var pers = f.personalized ? num(f.personalized.point) : null;
      var mark = changed && changed[f.target] && changed[f.target][num(f.horizon_hours)];
      return '<tr' + (mark ? ' class="row-changed"' : '') + '>' +
        '<td class="t" data-k="Target">' + esc(labelFor(f.target)) + '</td>' +
        '<td class="num" data-k="Horizon">' + esc(f.horizon_hours) + ' h</td>' +
        '<td class="num" data-k="Point">' + (pers == null ? '<span class="nullv">null</span>' : esc(fmtVal(pers, f.target))) + '</td>' +
        '<td class="num" data-k="Interval">' + interval + '</td>' +
        '<td data-k="Artifact nominal target coverage">' + (cov == null ? '<span class="nullv">n/a</span>'
          : '<span class="stat ok"><i></i><span class="ratio">' + esc(cov.toFixed(2)) + '</span></span>') + '</td>' +
        '</tr>';
    }).join('');
    return '<div class="section-head"><h2 class="h-sec">Validated forecast</h2>' +
        '<span class="meta">' + rows.length + ' targets · nominal target coverage recorded per artifact cell</span>' +
        (rows.length > 4 ? '<button class="disc spacer" type="button" data-fm="toggle-all">' +
          (S.showAll ? 'Show fewer ⌃' : 'Show all ' + rows.length + ' ⌄') + '</button>' : '') +
      '</div>' +
      '<div class="scroll-x"><table class="dtable"><thead><tr><th>Target</th><th>Horizon</th><th>Point</th>' +
      '<th>Target-calibrated research interval</th><th>Artifact nominal target coverage</th></tr></thead><tbody>' +
      (body || '<tr><td colspan="5" class="why">No validated cell in this response.</td></tr>') +
      '</tbody></table></div>' +
      '<p class="note" style="margin-top:12px">The interval comes from the serialized patient-specific conformal ' +
      'artifact. <span class="mono">interval_target_coverage</span> is that artifact\'s nominal target coverage, ' +
      'not empirical coverage or accuracy for this replay case. This is a research interval, not a clinical ' +
      'confidence interval, and it carries no clinical guarantee.</p>';
  }

  function illustrativeHtml(resp) {
    var rows = ofTier(resp, 'illustrative');
    var shown = S.showAll ? rows : onePerTarget(rows).slice(0, 3);
    return '<div class="section-head"><h2 class="h-sec">Illustrative</h2>' +
        '<span class="meta">' + rows.length + ' · no interval, not validated</span></div>' +
      '<ul class="tier-list">' + shown.map(function (f) {
        var pers = f.personalized ? num(f.personalized.point) : null;
        return '<li><span class="t">' + esc(labelFor(f.target)) + ' · ' + esc(f.horizon_hours) + ' h</span>' +
          '<span class="v">' + (pers == null ? 'null' : esc(fmtVal(pers, f.target))) +
          ' <span class="u">' + esc(unitFor(f.target)) + '</span></span></li>';
      }).join('') + '</ul>' +
      (rows.length > shown.length
        ? '<button class="disc" type="button" data-fm="toggle-all">Show ' + (rows.length - shown.length) + ' more ⌄</button>'
        : '');
  }

  function unsupportedRowHtml(resp) {
    var rows = ofTier(resp, 'unsupported');
    return '<div class="count-row"><span class="stat na"><i></i>Unsupported</span>' +
      '<span class="d">' + rows.length + ' targets returned null</span>' +
      '<button class="disc" type="button" data-fm="tier" data-tier="unsupported">Review ⌄</button></div>';
  }

  /* Null is a result: an unsupported cell is rendered as an explicit null with
     a reason, never as a blank or a zero. */
  function unsupportedTableHtml(resp) {
    var rows = ofTier(resp, 'unsupported');
    return '<div class="section-head"><h2 class="h-sec">Unsupported targets</h2>' +
        '<span class="meta">' + rows.length + ' of ' + ((resp && resp.forecasts) || []).length +
        ' · the model returned no value</span></div>' +
      '<p class="note" style="margin-bottom:16px">These targets have no validated artifact and no illustrative ' +
      'fallback, so the model returns nothing. Nothing here is imputed, interpolated or carried forward from a ' +
      'neighbouring target. An empty cell means the question was not answerable with this model on this case.</p>' +
      (rows.length
        ? '<div class="scroll-x"><table class="dtable"><thead><tr><th>Target</th><th>Horizon</th><th>Value</th>' +
          '<th>Why no value</th></tr></thead><tbody>' +
          rows.map(function (f) {
            return '<tr><td class="t" data-k="Target">' + esc(labelFor(f.target)) + '</td>' +
              '<td class="num" data-k="Horizon">' + esc(f.horizon_hours) + ' h</td>' +
              '<td class="nullv" data-k="Value">null</td>' +
              '<td class="why" data-k="Why">' +
              esc(f.note || f.reason || f.status || 'No value was produced for this cell.') +
              '</td></tr>';
          }).join('') + '</tbody></table></div>'
        : '<p class="note">No target returned null at this anchor. Every cell in this response is either a ' +
          'validated artifact or an illustrative point.</p>') +
      '<dl class="defs" style="margin-top:24px">' +
        '<dt>Validated</dt><dd>Serialized artifact with a target-calibrated interval and nominal target coverage recorded per cell.</dd>' +
        '<dt>Illustrative</dt><dd>A point with no interval and no coverage record. Shown for shape only; never charted with a band.</dd>' +
        '<dt>Unsupported</dt><dd>No value returned. Rendered as an explicit null with a reason, never as a blank or a zero.</dd>' +
      '</dl>';
  }

  // ── explanatory stages · "How this forecast was constructed" ───────────
  /* Every stage is conditioned on what the response actually contains. Nothing
     here claims the backend streamed these steps: they describe the returned
     document, in the order the contract resolves it. */
  function stagesFor(resp) {
    if (!resp) return [];
    var validated = ofTier(resp, 'validated');
    var illustrative = ofTier(resp, 'illustrative');
    var unsupported = ofTier(resp, 'unsupported');
    // Every count on this list uses the same deduped target × horizon basis the
    // tier filter and the tables use, so no two numbers on screen can disagree.
    var all = validated.concat(illustrative, unsupported);
    var withPopulation = all.filter(function (f) {
      return f.population && num(f.population.point) != null;
    });
    var withPersonal = all.filter(function (f) {
      var pers = f.personalized ? num(f.personalized.point) : null;
      var pop = f.population ? num(f.population.point) : null;
      return pers != null && pop != null && Math.abs(pers - pop) > 1e-9;
    });
    var withInterval = all.filter(function (f) {
      return isNum(num(f.lower)) && isNum(num(f.upper));
    });
    var rows = resp._prefix_rows;
    var out = [];
    out.push({
      t: 'Causal trajectory prefix accepted',
      d: rows + ' observation row' + (rows === 1 ? '' : 's') + ' up to and including ' +
        fmtHourExact(num(resp._anchor_hour)) + '. Nothing after the anchor was sent.'
    });
    out.push({
      t: 'Authorized target × horizon cells resolved',
      d: (validated.length + illustrative.length + unsupported.length) + ' cells resolved · ' +
        validated.length + ' validated, ' + illustrative.length + ' illustrative, ' +
        unsupported.length + ' unsupported.'
    });
    if (withPopulation.length) {
      out.push({
        t: 'Population points evaluated',
        d: withPopulation.length + ' cell' + (withPopulation.length === 1 ? '' : 's') +
          ' returned a population point.'
      });
    }
    if (withPersonal.length) {
      out.push({
        t: 'Patient-state adjustments applied where available',
        d: withPersonal.length + ' cell' + (withPersonal.length === 1 ? '' : 's') +
          ' returned a personalized point that differs from its population point.'
      });
    }
    if (withInterval.length) {
      out.push({
        t: 'Target-specific intervals returned where authorized',
        d: withInterval.length + ' cell' + (withInterval.length === 1 ? '' : 's') +
          ' carried a lower and upper bound from a serialized conformal artifact.'
      });
    }
    if (unsupported.length) {
      out.push({
        t: 'Unsupported cells withheld',
        d: unsupported.length + ' cell' + (unsupported.length === 1 ? '' : 's') +
          ' returned null. No value was substituted for them.'
      });
    }
    return out;
  }

  function stagesHtml(resp) {
    var stages = stagesFor(resp);
    if (!stages.length) return '';
    var done = S.stageIndex < 0 || S.stageIndex >= stages.length - 1;
    var running = S.stageIndex >= 0 && !done;
    return '<div class="stages-block">' +
      '<div class="section-head"><h2 class="h-sec">How this forecast was constructed</h2>' +
        '<span class="meta">describes the returned document · the backend answered in one request</span>' +
        '<span class="spacer">' +
          (running
            ? '<button class="btn sm" type="button" data-fm="skip-anim">Skip animation</button>'
            : '<button class="btn sm" type="button" data-fm="replay-presentation">Replay presentation</button>') +
        '</span></div>' +
      '<ol class="stages">' + stages.map(function (s, i) {
        var on = S.stageIndex < 0 || i <= S.stageIndex;
        return '<li class="stage' + (on ? ' on' : '') + (i === S.stageIndex ? ' current' : '') + '">' +
          '<span class="n">' + String(i + 1).padStart(2, '0') + '</span>' +
          '<span class="b"><span class="t">' + esc(s.t) + '</span>' +
          '<span class="d">' + esc(s.d) + '</span></span></li>';
      }).join('') + '</ol>' +
      '<p class="note">The model is fixed. Forecast outputs update as additional causally available observations are ' +
      'added to the trajectory prefix.</p></div>';
  }

  function clearStages() {
    S.stageTimers.forEach(clearTimeout);
    S.stageTimers = [];
  }
  function playStages(resp) {
    clearStages();
    var stages = stagesFor(resp);
    if (!stages.length) return;
    if (REDUCED) { S.stageIndex = -1; renderAll(); return; }
    S.stageIndex = 0;
    renderAll();
    for (var i = 1; i < stages.length; i++) {
      (function (idx) {
        S.stageTimers.push(setTimeout(function () {
          S.stageIndex = idx;
          renderAll();
          if (idx === stages.length - 1) {
            S.stageTimers.push(setTimeout(function () { S.stageIndex = -1; renderAll(); }, STAGE_MS));
          }
        }, idx * STAGE_MS));
      })(i);
    }
  }
  function skipStages() { clearStages(); S.stageIndex = -1; renderAll(); }

  // ── what changed between two real responses ────────────────────────────
  /* Both sides come from a completed /api/forecast response. Nothing is
     inferred, and no clinical meaning is attached to a difference. */
  function cellsOf(resp) {
    var map = {};
    ((resp && resp.forecasts) || []).forEach(function (f) {
      var k = f.target + '@' + f.horizon_hours;
      var rank = f.tier === 'validated_artifact' ? 2 : (f.tier === 'illustrative' ? 1 : 0);
      if (map[k] && map[k].rank >= rank) return;
      map[k] = {
        rank: rank, target: f.target, horizon: num(f.horizon_hours), tier: f.tier,
        point: f.personalized ? num(f.personalized.point) : null,
        lower: num(f.lower), upper: num(f.upper)
      };
    });
    return map;
  }

  function diffResponses(prev, next) {
    if (!prev || !next) return null;
    var a = cellsOf(prev), b = cellsOf(next);
    var rows = [];
    Object.keys(b).forEach(function (k) {
      var n = b[k], p = a[k];
      var wasUnsupported = !p || p.tier === 'unsupported' || p.point == null;
      var isSupported = n.tier !== 'unsupported' && n.point != null;
      var pointChanged = p && isNum(p.point) && isNum(n.point) && Math.abs(n.point - p.point) > 1e-9;
      var tierChanged = p && p.tier !== n.tier;
      if (!pointChanged && !tierChanged && !(wasUnsupported && isSupported)) return;
      rows.push({
        target: n.target, horizon: n.horizon,
        prevPoint: p ? p.point : null, newPoint: n.point,
        delta: (p && isNum(p.point) && isNum(n.point)) ? n.point - p.point : null,
        prevTier: p ? p.tier : null, newTier: n.tier,
        becameSupported: !!(wasUnsupported && isSupported),
        prevLower: p ? p.lower : null, prevUpper: p ? p.upper : null,
        newLower: n.lower, newUpper: n.upper
      });
    });
    rows.sort(function (x, y) {
      return (y.becameSupported - x.becameSupported) ||
        x.target.localeCompare(y.target) || (x.horizon - y.horizon);
    });
    return rows;
  }

  /* target -> horizon -> true, for the chart and table highlight */
  function changedMap(rows) {
    var m = {};
    (rows || []).forEach(function (r) {
      if (!m[r.target]) m[r.target] = {};
      m[r.target][r.horizon] = true;
    });
    return m;
  }

  function tierWord(t) {
    if (t === 'validated_artifact') return 'validated';
    if (t === 'illustrative') return 'illustrative';
    if (t === 'unsupported') return 'unsupported';
    return t || 'absent';
  }

  function changeHtml(c, rows, prevStep, step) {
    if (!rows) return '';
    var newlySupported = rows.filter(function (r) { return r.becameSupported; });
    var head = '<div class="section-head"><h2 class="h-sec">What changed</h2>' +
      '<span class="meta">step ' + (prevStep + 1) + ' → step ' + (step + 1) + ' · ' + rows.length +
      ' cell' + (rows.length === 1 ? '' : 's') + '</span></div>';
    var lead = '<ul class="fact-list">' +
      '<li>One new observation became available.</li>' +
      '<li>The trajectory prefix now contains ' + (step + 1) + ' row' + (step === 0 ? '' : 's') + '.</li>' +
      (newlySupported.length
        ? '<li>' + newlySupported.length + ' cell' + (newlySupported.length === 1 ? ' is' : 's are') +
          ' now supported at this anchor.</li>'
        : '') +
      '<li>This is a model-output change, not a treatment effect.</li>' +
      '</ul>';
    if (!rows.length) {
      return '<div class="section change-block">' + head + lead +
        '<p class="note">No forecast cell changed between these two responses.</p></div>';
    }
    var body = rows.map(function (r) {
      var unit = unitFor(r.target);
      var lines = [];
      if (r.becameSupported) {
        lines.push('<span class="chg new">This cell is now supported at this anchor.</span>');
      }
      if (isNum(r.delta)) {
        lines.push('<span class="chg">The personalized point changed by ' +
          esc(fmtSigned(r.delta, r.target)) + ' ' + esc(unit) + ' relative to the previous replay step.</span>');
      }
      var prevInt = (isNum(r.prevLower) && isNum(r.prevUpper))
        ? fmtVal(r.prevLower, r.target) + ' – ' + fmtVal(r.prevUpper, r.target) : null;
      var newInt = (isNum(r.newLower) && isNum(r.newUpper))
        ? fmtVal(r.newLower, r.target) + ' – ' + fmtVal(r.newUpper, r.target) : null;
      return '<tr' + (r.becameSupported ? ' class="row-changed"' : '') + '>' +
        '<td class="t" data-k="Cell">' + esc(labelFor(r.target)) + ' · ' + esc(r.horizon) + ' h</td>' +
        '<td class="num" data-k="Previous point">' +
          (isNum(r.prevPoint) ? esc(fmtVal(r.prevPoint, r.target)) : '<span class="nullv">null</span>') + '</td>' +
        '<td class="num" data-k="New point">' +
          (isNum(r.newPoint) ? esc(fmtVal(r.newPoint, r.target)) : '<span class="nullv">null</span>') + '</td>' +
        '<td class="num" data-k="Delta">' + (isNum(r.delta) ? esc(fmtSigned(r.delta, r.target)) : '—') + '</td>' +
        '<td data-k="Tier">' + esc(tierWord(r.prevTier)) + ' → ' + esc(tierWord(r.newTier)) + '</td>' +
        '<td class="num" data-k="Interval">' +
          (prevInt && newInt ? esc(prevInt) + ' → ' + esc(newInt) : '<span class="nullv">—</span>') + '</td>' +
        '<td class="why" data-k="Statement">' + lines.join(' ') + '</td>' +
        '</tr>';
    }).join('');
    return '<div class="section change-block">' + head + lead +
      '<div class="scroll-x"><table class="dtable"><thead><tr><th>Cell</th><th>Previous point</th>' +
      '<th>New point</th><th>Delta</th><th>Tier</th><th>Interval</th><th>Statement</th></tr></thead>' +
      '<tbody>' + body + '</tbody></table></div>' +
      '<p class="note" style="margin-top:12px">Differences are computed from the two completed API responses only. ' +
      'The model is fixed. Forecast outputs update as additional causally available observations are added to the ' +
      'trajectory prefix. No clinical interpretation is offered and this never changes drug ranking or dosing.</p>' +
      '</div>';
  }

  // ── research signals ───────────────────────────────────────────────────
  /* Descriptive display states only. No clinical thresholds are defined here:
     the states summarise (a) the direction of the patient's own consecutive
     observations, (b) how far the personalized point sits from the population
     point relative to the model's own interval width, and (c) whether a matched
     observation landed inside a previously issued research interval. */
  function computeSignals(c, resp, uptoStep) {
    var out = [];
    var resolved = historyFor(c, uptoStep);
    PRIMARY_SERIES.forEach(function (target) {
      var obs = observationsFor(c, target, uptoStep);
      var unit = unitFor(target);
      if (!obs.length) {
        out.push({
          target: target, state: 'unavailable', label: 'Data unavailable',
          why: 'No observation for this variable up to ' + fmtHour(hourAt(c, uptoStep)) + '.'
        });
        return;
      }
      var reasons = [];
      var state = 'stable', label = 'Stable';

      // (a) direction of the patient's own consecutive observations
      if (obs.length >= 2) {
        var a = obs[obs.length - 2], b = obs[obs.length - 1];
        var diff = b.value - a.value;
        var rel = a.value !== 0 ? Math.abs(diff / a.value) : 0;
        if (rel >= 0.10) {
          reasons.push('consecutive observations ' + (diff > 0 ? 'rising' : 'falling') + ' ' +
            fmtVal(a.value, target) + ' → ' + fmtVal(b.value, target) + ' ' + unit +
            ' between ' + fmtHour(a.hour) + ' and ' + fmtHour(b.hour));
          state = 'watch'; label = 'Watch';
        }
      }

      // (b) personalized point vs population point, scaled by the model's own interval
      (resp ? (resp.forecasts || []) : []).forEach(function (f) {
        if (f.target !== target || f.tier !== 'validated_artifact') return;
        var pop = f.population ? num(f.population.point) : null;
        var pers = f.personalized ? num(f.personalized.point) : null;
        var lo = num(f.lower), hi = num(f.upper);
        if (pop == null || pers == null || !isNum(lo) || !isNum(hi)) return;
        var half = Math.abs(hi - lo) / 2;
        if (half <= 0) return;
        if (Math.abs(pers - pop) / half >= 0.25) {
          reasons.push('personalized +' + f.horizon_hours + ' h point differs from the population point by ' +
            fmtVal(pers - pop, target) + ' ' + unit + ' (≥ 25% of the research interval half-width)');
          if (state === 'stable') { state = 'watch'; label = 'Watch'; }
        }
      });

      // (c) a matched observation fell outside a previously issued research interval
      var outside = resolved.filter(function (r) {
        return r.row.target === target && r.res.state === 'outside';
      });
      if (outside.length) {
        var o = outside[outside.length - 1];
        reasons.push('an observation at ' + fmtHour(o.res.hour) +
          ' fell outside the research interval issued at ' + fmtHour(o.row.anchorHour) +
          ' for +' + o.row.horizon + ' h');
        state = 'signal'; label = 'Research signal';
      }

      if (!reasons.length) {
        reasons.push('no direction change ≥ 10% between the last two observations, ' +
          'no large personalized-vs-population divergence, no interval miss recorded');
      }
      out.push({ target: target, state: state, label: label, why: reasons.join('; ') + '.' });
    });
    return out;
  }

  function signalStat(state) {
    if (state === 'signal') return 'bad';
    if (state === 'watch') return 'warn';
    if (state === 'unavailable') return 'na';
    return 'ok';
  }

  function signalsHtml(c, resp, uptoStep) {
    var sigs = computeSignals(c, resp, uptoStep);
    var active = sigs.filter(function (s) { return s.state === 'watch' || s.state === 'signal'; });
    return '<div class="section-head"><h2 class="h-sec">Research signals</h2>' +
        '<span class="meta">' + active.length + ' at this step</span></div>' +
      sigs.map(function (s) {
        return '<div class="signal"><div class="signal-h"><span class="t">' + esc(labelFor(s.target)) + '</span>' +
          '<span class="stat ' + signalStat(s.state) + '"><i></i>' + esc(s.label) + '</span></div>' +
          '<div class="d">' + esc(s.why) + '</div></div>';
      }).join('') +
      '<p class="note" style="margin-top:14px">Signals are recomputed from scratch at every step. They do not ' +
      'accumulate across the replay. These states are descriptive display labels derived from the observations and ' +
      'model outputs on screen. They are not clinical thresholds, not alerts, and not a severity score. They do not ' +
      'indicate deterioration, name a condition, or suggest any action, and they never change drug ranking or dosing.</p>';
  }

  function matchingNoteHtml(c) {
    var policy = matchPolicy(c);
    if (policy.enabled) {
      return '<p class="note">Outcome matching uses the backend replay contract (±' + esc(policy.tolerance) +
        ' h of the due hour · ' + esc(policy.source) + '). No accuracy verdict is made beyond it.</p>';
    }
    return '<p class="note">Awaiting / no configured matching. The backend replay contract disables observation ' +
      'matching, so no observed value is compared against an issued interval and no accuracy verdict is made.</p>';
  }

  // ── replay toolbar ─────────────────────────────────────────────────────
  function replayStepAtTime(c, replayTime) {
    var found = -1;
    for (var i = 0; i <= lastStep(c); i++) {
      if (hourAt(c, i) <= replayTime + 1e-7) found = i;
    }
    return found;
  }

  function replayPercent(c, replayTime) {
    var end = hourAt(c, lastStep(c)) || 1;
    return Math.max(0, Math.min(100, (replayTime / end) * 100));
  }

  /* Withheld observations appear as hollow ticks. Their timestamps are public
     replay metadata; their measurements and forecast values do not enter the
     DOM until the cursor reaches the corresponding anchor. */
  function replayBarHtml(c) {
    var total = lastStep(c) + 1;
    var started = S.replayStarted;
    var busy = S.step >= 0 && isPending(c, S.step);
    var lastH = hourAt(c, lastStep(c)) || 1;
    var ticks = '';
    for (var i = 0; i < total; i++) {
      var h = hourAt(c, i) || 0;
      var pct = lastH > 0 ? (h / lastH) * 100 : 0;
      var cls = !started || i > S.step ? 'tick withheld'
        : (i === S.step ? 'tick current' : 'tick');
      ticks += '<span class="' + cls + '" style="left:' + pct.toFixed(1) + '%"></span>';
    }
    var withheld = started ? total - 1 - Math.max(-1, S.step) : total;
    var caption = !started
      ? total + ' recorded observations withheld — none has been submitted to the model'
      : (withheld > 0
        ? withheld + ' later observation' + (withheld === 1 ? '' : 's') + ' withheld — not yet in model'
        : 'full prefix posted — no observation withheld');
    var anchorBlock = started
      ? '<div class="replay-anchor"><span class="big">Retrospective replay</span>' +
        '<span class="d">' + (S.step < 0 ? 'before first observation · nothing sent' :
          'last anchor ' + esc(fmtHourExact(hourAt(c, S.step))) + ' · ' + (S.step + 1) +
          ' observation' + (S.step === 0 ? '' : 's') + ' visible · ' + (S.step + 1) +
          ' prefix row' + (S.step === 0 ? '' : 's') + ' sent') + '</span></div>'
      : '<div class="replay-anchor"><span class="big idle">—</span>' +
        '<span class="d">no anchor · 0 of ' + total + ' observations visible · nothing sent to the model</span></div>';
    var transport = started
      ? '<button class="btn sm" type="button" data-fm="prev"' + (S.step < 0 ? ' disabled' : '') + '>◀ Previous</button>' +
        (S.playing
          ? '<button class="btn sm primary" type="button" data-fm="pause">❚❚ Pause</button>'
          : '<button class="btn sm primary" type="button" data-fm="play"' +
            (S.replayTime >= lastH - 1e-7 ? ' disabled' : '') + '>▶ Continue</button>') +
        '<button class="btn sm" type="button" data-fm="next"' + (S.step >= lastStep(c) ? ' disabled' : '') + '>Next ▶</button>' +
        '<span class="speed-wrap">Speed <select class="select" data-fm="speed" aria-label="Replay speed">' +
          SPEEDS.map(function (s, i) {
            return '<option value="' + i + '"' + (i === S.speedIndex ? ' selected' : '') + '>' + esc(s.label) + '</option>';
          }).join('') + '</select></span>' +
        '<button class="btn sm" type="button" data-fm="reset">Reset</button>' +
        '<button class="btn sm" type="button" data-fm="rerun-step"' + (S.step < 0 ? ' disabled' : '') +
          '>Re-run this anchor live</button>' +
        (busy ? '<span class="meta"><span class="spin"></span> requesting…</span>' : '')
      : '<button class="btn sm primary" type="button" data-fm="start-replay"' +
        (healthy() ? '' : ' disabled') + '>▶ Play retrospective replay</button>' +
        (busy ? '<span class="meta"><span class="spin"></span> requesting…</span>' : '');
    return '<div class="replay-bar">' + anchorBlock +
      '<div class="track-wrap"><div class="track">' +
        '<span class="line"></span>' + ticks + '</div>' +
        '<div class="track-labels"><span>0 h</span><span class="cap">' + esc(caption) + '</span>' +
        '<span>' + esc(fmtHour(lastH)) + '</span></div></div>' +
      '<div class="transport">' + transport + '</div></div>';
  }

  function replayTraceSvg(c, step) {
    var end = hourAt(c, lastStep(c)) || 1;
    var x = function (h) { return 92 + (Math.max(0, h) / end) * 858; };
    var rows = '';
    PRIMARY_SERIES.forEach(function (target, row) {
      var top = 28 + row * 74;
      var domain = REPLAY_DOMAINS[target];
      var points = [];
      for (var i = 0; i <= step; i++) {
        var obs = trajectoryOf(c)[i] || {};
        var value = num(obs[target]);
        if (value == null) continue;
        var ratio = Math.max(0, Math.min(1, (value - domain[0]) / (domain[1] - domain[0])));
        points.push({ x: x(hourAt(c, i)), y: top + 43 - ratio * 34, value: value, index: i });
      }
      rows += '<text class="rt-label" x="4" y="' + (top + 17) + '">' + esc(labelFor(target)) + '</text>' +
        '<text class="rt-unit" x="4" y="' + (top + 34) + '">' + esc(unitFor(target)) + '</text>' +
        '<line class="rt-grid" x1="92" y1="' + (top + 43) + '" x2="950" y2="' + (top + 43) + '"></line>';
      if (points.length > 1) {
        rows += '<polyline class="rt-path" points="' + points.map(function (p) {
          return p.x.toFixed(1) + ',' + p.y.toFixed(1);
        }).join(' ') + '"></polyline>';
      }
      points.forEach(function (p) {
        var current = p.index === step ? ' current' : '';
        rows += '<g class="rt-point' + current + '" transform="translate(' + p.x.toFixed(1) + ' ' + p.y.toFixed(1) + ')">' +
          '<circle class="pulse" r="10"></circle><circle r="4"></circle>' +
          '<text x="8" y="-7">' + esc(fmtVal(p.value, target)) + '</text></g>';
      });
    });
    var ticks = '';
    for (var i = 0; i <= lastStep(c); i++) {
      var tickX = x(hourAt(c, i));
      ticks += '<line class="rt-anchor-line" x1="' + tickX.toFixed(1) + '" y1="12" x2="' +
        tickX.toFixed(1) + '" y2="326"></line><text class="rt-hour" x="' + tickX.toFixed(1) +
        '" y="347" text-anchor="middle">' + esc(fmtHourExact(hourAt(c, i))) + '</text>';
    }
    return '<svg viewBox="0 0 980 360" role="img" aria-label="Recorded observation anchors and visual interpolation; no intermediate measurements">' +
      ticks + rows + '<line class="rt-cursor" data-replay-cursor x1="92" y1="8" x2="92" y2="326"></line></svg>';
  }

  function replayRowsHtml(c) {
    var rows = '';
    for (var i = 0; i <= lastStep(c); i++) {
      if (i > S.step) {
        rows += '<li class="future"><span>' + esc(fmtHourExact(hourAt(c, i))) + '</span>' +
          '<b>Withheld</b><small>measurement values not in DOM or model prefix</small></li>';
        continue;
      }
      var o = trajectoryOf(c)[i] || {};
      rows += '<li class="observed' + (i === S.step ? ' current' : '') + '"><span>' +
        esc(fmtHourExact(hourAt(c, i))) + '</span><b>Observation ' + (i + 1) + '</b><small>' +
        'Cr ' + esc(fmtVal(num(o.creatinine), 'creatinine')) + ' · BUN ' + esc(fmtVal(num(o.bun), 'bun')) +
        ' · UO ' + (num(o.urine_output) == null ? 'unavailable' : esc(fmtVal(num(o.urine_output), 'urine_output'))) +
        ' · MAP ' + esc(fmtVal(num(o.map), 'map')) + '</small></li>';
    }
    return rows;
  }

  function replayEventsHtml(c) {
    if (S.step < 0) return '<li><span>Waiting</span><b>No observation or forecast has been revealed.</b></li>';
    var out = '';
    for (var i = 0; i <= S.step; i++) {
      var resp = responseFor(c, i);
      var pending = isPending(c, i);
      var count = resp ? ofTier(resp, 'validated').length : 0;
      out += '<li><span>' + esc(fmtHourExact(hourAt(c, i))) + '</span><b>Observation ' + (i + 1) +
        ' revealed · ' + (i + 1) + '-row prefix ' + (pending ? 'sending…' : 'sent') + '</b><small>' +
        (resp ? count + ' validated artifacts returned · ' + esc((resultSource(c, i) || {}).label || 'result') :
          (S.stepError && i === S.step ? esc(S.stepError) : 'forecast response pending')) + '</small></li>';
    }
    return out;
  }

  function replayStageHtml(c) {
    var end = hourAt(c, lastStep(c)) || 1;
    return '<div class="shell replay-shell"><section class="replay-stage" aria-label="Retrospective replay timeline">' +
      '<div class="replay-stage-head"><div><span class="eyebrow">Current replay time</span>' +
        '<strong class="replay-clock" data-replay-time>0.00 h</strong></div>' +
        '<dl><div><dt>Last observation</dt><dd data-replay-last>none</dd></div>' +
        '<div><dt>Rows sent</dt><dd data-replay-rows>0</dd></div></dl></div>' +
      '<div class="replay-traces" data-replay-traces data-rendered-step="-1">' + replayTraceSvg(c, -1) + '</div>' +
      '<div class="interpolation-note"><b>Retrospective replay — not live monitoring.</b> Lines between recorded ' +
        'anchors are visual interpolation only; there is no intermediate measurement or model inference.</div>' +
      '<div class="input-provenance-note"><b>Input provenance:</b> The public demo rows do not provide ' +
        '<span class="mono">hist_fluids</span>, <span class="mono">hist_vasopressor</span>, ' +
        '<span class="mono">hist_diuretics</span>, <span class="mono">hist_renal_replacement</span>, or ' +
        '<span class="mono">hist_nephrotoxin</span>. The current runtime supplies zero for these five missing model ' +
        'fields; zero is missing-input handling, not evidence that no treatment occurred.</div>' +
      '<label class="scrubber-label" for="replayScrubber"><span>Review elapsed replay time</span>' +
        '<span>future remains withheld</span></label>' +
      '<input id="replayScrubber" class="replay-scrubber" data-fm="scrub" type="range" min="0" max="0" ' +
        'step="any" value="0" aria-label="Seek within elapsed retrospective replay time">' +
      '<div class="replay-stage-grid"><div><h3 class="h-sec">Observation arrivals</h3>' +
        '<ol class="observation-arrivals" data-replay-observations>' + replayRowsHtml(c) + '</ol></div>' +
        '<div><h3 class="h-sec">Replay event log</h3><ol class="replay-events" data-replay-events>' +
        replayEventsHtml(c) + '</ol></div></div>' +
      '<span class="sr-only">Replay ends at ' + esc(fmtHourExact(end)) + '.</span>' +
      '</section><div data-replay-results></div></div>';
  }

  // ── Home ───────────────────────────────────────────────────────────────
  function renderHome() {
    var root = el('view-home');
    if (!root) return;
    root.innerHTML =
      '<div class="shell"><div class="home">' +
        '<div class="home-lead">' +
          '<h1 class="display">Two independent research demonstrations</h1>' +
          '<p class="note">Osler contains two separate research demonstrations. They use different cases, different ' +
          'engines and different evidence. They are not two stages of one patient, and neither reads the other\'s ' +
          'output. Pick the one you want to look at.</p>' +
        '</div>' +
        '<div class="workflow-cards">' +
          '<div class="wcard">' +
            '<div class="wcard-h"><span class="k">01</span><h2>Medication Safety</h2></div>' +
            '<ul class="wcard-list">' +
              '<li>Illustrative case scenarios</li>' +
              '<li>Symbolic medication-safety engine</li>' +
              '<li>Optional live openFDA labels</li>' +
              '<li>Independent of the factual forecast models</li>' +
              '<li>No automated prescribing</li>' +
            '</ul>' +
            '<div class="wcard-foot">' +
              '<button class="btn primary" type="button" data-fm="go-rx">Explore medication safety</button>' +
            '</div>' +
          '</div>' +
          '<div class="wcard">' +
            '<div class="wcard-h"><span class="k">02</span><h2>Retrospective Forecast</h2></div>' +
            '<ul class="wcard-list">' +
              '<li>Deidentified eICU CRD Demo 2.0.1 case</li>' +
              '<li>Real serialized forecast artifacts</li>' +
              '<li>Observation-prefix replay</li>' +
              '<li>Target-specific calibrated intervals</li>' +
              '<li>Not live monitoring</li>' +
            '</ul>' +
            '<div class="wcard-foot">' +
              '<button class="btn primary" type="button" data-fm="go-forecast">Explore retrospective forecast</button>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="home-foot">' +
          '<div><div class="h-sec">Guided demo</div>' +
          '<p class="note">Walks both chapters in order: the medication-safety method first, then the retrospective ' +
          'factual forecast. The change of case and model pathway between them is disclosed and has to be ' +
          'acknowledged.</p></div>' +
          '<button class="btn" type="button" data-fm="start-guided">Start guided demo</button>' +
        '</div>' +
        '<p class="note home-note">This page loads no model. Neither engine is called until you choose a workflow ' +
        'and ask for a result.</p>' +
      '</div></div>';
  }

  // ── Snapshot subview ───────────────────────────────────────────────────
  function snapshotIntroHtml(c) {
    var total = lastStep(c) + 1;
    var first = hourAt(c, 0), last = hourAt(c, lastStep(c));
    return '<div class="shell"><div class="section" style="padding-top:28px">' +
      '<div class="section-head"><h1 class="display">Forecast snapshot</h1>' +
        '<span class="sub">no forecast has been requested yet</span></div>' +
      '<div class="prerun">' +
        '<div class="prerun-main">' +
          '<p class="note">This case carries a recorded retrospective trajectory of <b>' + total +
          ' observations</b> between ' + esc(fmtHourExact(first)) + ' and ' + esc(fmtHourExact(last)) +
          ' after ICU admission. A snapshot posts the full causal prefix — all ' + total +
          ' rows, with <span class="mono">anchor_hour</span> set to ' + esc(fmtHourExact(last)) +
          ' — and renders exactly what the model returns.</p>' +
          '<p class="note">Nothing has been computed. No forecast value exists on this page until you run one.</p>' +
        '</div>' +
        '<dl class="prerun-facts">' +
          '<dt>Final anchor</dt><dd class="mono">' + esc(fmtHourExact(last)) + '</dd>' +
          '<dt>Prefix rows to send</dt><dd class="mono">' + total + '</dd>' +
          '<dt>Model version</dt><dd class="mono">' + esc(modelVersion()) + '</dd>' +
          '<dt>Model health</dt><dd>' + healthStatHtml() + '</dd>' +
        '</dl>' +
      '</div>' +
      '<div class="actions-row">' +
        '<button class="btn primary" type="button" data-fm="run-snapshot"' + (healthy() ? '' : ' disabled') +
          '>Run live forecast</button>' +
        '<button class="btn" type="button" data-fm="sub" data-sub="replay">Open retrospective replay</button>' +
      '</div>' +
      (healthy() ? '' : '<p class="note" style="margin-top:12px">Model health has not passed, so no request is ' +
        'issued and no value is substituted.</p>') +
      '</div></div>';
  }

  function renderSnapshot() {
    var root = el('fx-snapshot');
    if (!root) return;
    var c = activeCase();
    if (!S.booted && !S.casesError) { root.innerHTML = loadingHtml(); return; }
    if (S.casesError || !c) {
      root.innerHTML = errorBlockHtml('Case source unavailable',
        S.casesError || 'No case returned by /api/monitoring/cases');
      return;
    }
    var step = lastStep(c);
    if (!healthy() && !S.fixture) { root.innerHTML = serviceUnavailableHtml(); return; }
    var resp = responseFor(c, step);
    if (S.snapshotBusy && !resp) {
      root.innerHTML = '<div class="shell"><div class="loading"><span class="spin"></span> ' +
        'Posting the full ' + (step + 1) + '-row causal prefix to the forecast model…</div></div>';
      return;
    }
    if (S.stepError && !resp) { root.innerHTML = errorBlockHtml('Forecast request failed', S.stepError); return; }
    if (!resp) { root.innerHTML = snapshotIntroHtml(c); return; }

    var charted = PRIMARY_SERIES.concat(S.showExtraSeries ? EXTRA_SERIES : []);
    var body;
    if (S.tier === 'unsupported') {
      body = '<div class="section">' + unsupportedTableHtml(resp) + '</div>';
    } else if (S.tier === 'illustrative') {
      body = '<div class="section">' + illustrativeHtml(resp) +
        '<p class="note" style="margin-top:16px">An illustrative point has no interval and no coverage record. ' +
        'It is shown for shape only and is never charted with a band.</p></div>';
    } else {
      body = '<div class="split"><div class="main">' + validatedTableHtml(resp) + '</div>' +
        '<aside class="side">' + illustrativeHtml(resp) + unsupportedRowHtml(resp) + '</aside></div>';
    }
    root.innerHTML = cachedBannerHtml() +
      '<div class="shell"><div class="section" style="padding-top:28px">' +
        '<div class="section-head"><h1 class="display">Forecast snapshot</h1>' +
          '<span class="sub">' + charted.length + ' targets charted · anchor ' +
          esc(fmtHourExact(num(resp._anchor_hour))) + '</span>' +
          '<span class="spacer">' + tierBarHtml(resp) + '</span></div>' +
        sourceLineHtml(c, step) +
        '<div class="actions-row" style="margin-bottom:20px">' +
          '<button class="btn" type="button" data-fm="rerun-snapshot"' + (healthy() ? '' : ' disabled') +
            '>Re-run live forecast</button>' +
          '<button class="btn" type="button" data-fm="sub" data-sub="replay">Open retrospective replay</button>' +
        '</div>' +
        stagesHtml(resp) +
        '<div class="charts' + (S.animateData ? ' fx-in' : '') + '">' + (function () {
          var xMax = chartDomain(c, resp, step, charted);
          return charted.map(function (t) {
            return chartCardHtml(c, t, resp, step, 'h', xMax);
          }).join('');
        })() + '</div>' +
        chartLegendHtml(false) + chartsNoteHtml() +
      '</div>' + body + '</div>';
  }

  // ── Replay subview ─────────────────────────────────────────────────────
  function replayIntroHtml(c) {
    var total = lastStep(c) + 1;
    var rows = '';
    for (var i = 0; i < total; i++) {
      rows += '<tr><td class="num" data-k="Step">' + (i + 1) + '</td>' +
        '<td class="num" data-k="Anchor">' + esc(fmtHourExact(hourAt(c, i))) + '</td>' +
        '<td class="num" data-k="Prefix rows">' + (i + 1) + '</td>' +
        '<td class="why" data-k="State">withheld — not submitted</td></tr>';
    }
    return '<div class="shell"><div class="section" style="padding-top:24px">' +
      '<div class="section-head"><h1 class="display">Observation-prefix replay</h1>' +
        '<span class="sub">pre-observation state · nothing has been sent to the model</span></div>' +
      '<div class="prerun">' +
        '<div class="prerun-main">' +
          '<p class="note">The replay begins before the first observation has been submitted. Play moves a continuous ' +
          'retrospective clock across the fixed timeline. Only arrival at a recorded anchor reveals one observation ' +
          'and posts or reuses that exact causal prefix. Later observation values remain withheld.</p>' +
          '<p class="note">The model is fixed. Forecast outputs update as additional causally available observations ' +
          'are added to the trajectory prefix.</p>' +
        '</div>' +
        '<dl class="prerun-facts">' +
          '<dt>Recorded observations</dt><dd class="mono">' + total + '</dd>' +
          '<dt>Observations visible</dt><dd class="mono">0</dd>' +
          '<dt>Requests issued</dt><dd class="mono">0</dd>' +
          '<dt>Model health</dt><dd>' + healthStatHtml() + '</dd>' +
        '</dl>' +
      '</div>' +
      '<div class="scroll-x" style="margin-top:22px"><table class="dtable"><thead><tr><th>Step</th>' +
        '<th>Anchor</th><th>Prefix rows to send</th><th>State</th></tr></thead><tbody>' + rows +
        '</tbody></table></div>' +
      '<p class="note" style="margin-top:12px">No forecast value is present on this page. Use <b>Play retrospective ' +
      'replay</b> to move the clock; the first model request occurs only when the cursor reaches the first recorded anchor.</p>' +
      '</div></div>';
  }

  function replayResultsHtml(c) {
    if (S.step < 0) {
      return '<div class="replay-awaiting"><span class="eyebrow">Forecast state</span>' +
        '<h2 class="h-sec">Awaiting the first recorded observation</h2>' +
        '<p class="note">The cursor is replaying elapsed time. No observation or forecast has been revealed yet.</p></div>';
    }
    var resp = responseFor(c, S.step);
    var busy = isPending(c, S.step);
    var showing = resp, showingStep = S.step, holding = false;
    if (!resp && busy && S.step > 0 && responseFor(c, S.step - 1)) {
      showing = responseFor(c, S.step - 1); showingStep = S.step - 1; holding = true;
    }
    if (!showing && S.stepError) {
      return errorBlockHtml('Forecast request failed at anchor ' + (S.step + 1), S.stepError);
    }
    if (!showing) {
      return '<div class="replay-awaiting"><span class="spin"></span> Posting only the ' + (S.step + 1) +
        '-row causal prefix at the recorded anchor…</div>';
    }
    var prev = (!holding && S.step > 0) ? responseFor(c, S.step - 1) : null;
    var diff = prev ? diffResponses(prev, showing) : null;
    var changed = changedMap(diff);
    var validatedCount = ofTier(showing, 'validated').length;
    var pendingNote = holding
      ? '<div class="notice replay-pending"><span class="n">New recorded observation arrived</span>' +
        '<span class="d">The completed anchor ' + (showingStep + 1) +
        ' result remains visible while the next causal prefix is in flight.</span>' +
        '<span class="acts"><span class="meta"><span class="spin"></span> updating</span></span></div>' : '';
    return pendingNote + '<div class="section replay-forecast-update' + (S.animateData ? ' fx-in' : '') + '">' +
      '<div class="section-head"><h2 class="h-sec">Recorded anchor ' + (showingStep + 1) + ' of ' +
        (lastStep(c) + 1) + '</h2><span class="meta">' + esc(fmtHourExact(num(showing._anchor_hour))) + ' · ' +
        esc(showing._prefix_rows) + ' trajectory row' + (showing._prefix_rows === 1 ? '' : 's') + ' sent · ' +
        validatedCount + ' validated artifact' + (validatedCount === 1 ? '' : 's') + '</span></div>' +
      sourceLineHtml(c, showingStep) + '</div>' +
      '<div class="split wide replay-result-grid"><div class="main"><div class="charts two' +
        (S.animateData ? ' fx-in' : '') + '">' + (function () {
          var xMax = chartDomain(c, showing, showingStep, PRIMARY_SERIES, true);
          return PRIMARY_SERIES.map(function (t) {
            return chartCardHtml(c, t, showing, showingStep, 'h after ICU admission', xMax,
              { stable: true, changed: changed, newestHour: hourAt(c, showingStep) });
          }).join('');
        })() + '</div>' + chartLegendHtml(!!(diff && diff.length)) +
        '<p class="note" style="margin-top:12px">Forecast inference occurs only at the three recorded anchors. ' +
        'The request contains the causal prefix through this anchor and no later observation. Prefix rows sent: <b>' +
        esc(showing._prefix_rows) + '</b>.</p>' + chartsNoteHtml() + matchingNoteHtml(c) + '</div>' +
        '<aside class="side">' + signalsHtml(c, showing, showingStep) + '</aside></div>' +
      (diff ? changeHtml(c, diff, showingStep - 1, showingStep) : '') +
      (S.step >= lastStep(c) && !holding
        ? '<div class="section final-note"><div class="section-head"><h2 class="h-sec">Final retrospective snapshot</h2>' +
          '<span class="meta">full recorded prefix posted</span></div><p class="note">Every recorded observation is ' +
          'now in the causal prefix. This remains a retrospective research replay, not live monitoring, a diagnostic output, or an automated alarm.</p>' +
          '<div class="actions-row"><button class="btn" type="button" data-fm="sub" data-sub="validation">' +
          'Open validation evidence</button><button class="btn" type="button" data-fm="reset">' +
          'Reset to pre-observation state</button></div></div>' : '');
  }

  function updateReplayDom(c, opts) {
    var root = el('view-monitoring');
    if (!root || !S.replayStarted) return;
    opts = opts || {};
    if (opts.anchor) renderForecastContext();
    if (opts.chrome) {
      var toolbar = root.querySelector('[data-replay-toolbar]');
      if (toolbar) toolbar.innerHTML = replayBarHtml(c);
    }
    var time = root.querySelector('[data-replay-time]');
    if (time) time.textContent = S.replayTime.toFixed(2) + ' h';
    var last = root.querySelector('[data-replay-last]');
    if (last) last.textContent = S.step < 0 ? 'none' : fmtHourExact(hourAt(c, S.step));
    var rows = root.querySelector('[data-replay-rows]');
    if (rows) rows.textContent = String(Math.max(0, S.step + 1));
    var cursor = root.querySelector('[data-replay-cursor]');
    if (cursor) {
      var x = 92 + replayPercent(c, S.replayTime) * 8.58;
      cursor.setAttribute('x1', x.toFixed(1)); cursor.setAttribute('x2', x.toFixed(1));
    }
    var scrub = root.querySelector('[data-fm="scrub"]');
    if (scrub) {
      scrub.max = String(Math.max(0.001, S.replayFurthestTime));
      scrub.value = String(Math.min(S.replayTime, S.replayFurthestTime));
    }
    var traces = root.querySelector('[data-replay-traces]');
    if (traces && traces.getAttribute('data-rendered-step') !== String(S.step)) {
      traces.innerHTML = replayTraceSvg(c, S.step);
      traces.setAttribute('data-rendered-step', String(S.step));
      cursor = traces.querySelector('[data-replay-cursor]');
      if (cursor) {
        var cx = 92 + replayPercent(c, S.replayTime) * 8.58;
        cursor.setAttribute('x1', cx.toFixed(1)); cursor.setAttribute('x2', cx.toFixed(1));
      }
    }
    var observations = root.querySelector('[data-replay-observations]');
    if (observations && opts.anchor) observations.innerHTML = replayRowsHtml(c);
    var events = root.querySelector('[data-replay-events]');
    if (events && opts.anchor) events.innerHTML = replayEventsHtml(c);
    var results = root.querySelector('[data-replay-results]');
    if (results && opts.results) results.innerHTML = replayResultsHtml(c);
    S.animateData = false;
  }

  function renderMonitoring() {
    var root = el('view-monitoring');
    if (!root) return;
    if (!S.booted && !S.casesError) { root.innerHTML = loadingHtml(); return; }
    var c = activeCase();
    if (S.casesError || !c) {
      root.innerHTML = errorBlockHtml('Retrospective case unavailable',
        S.casesError || 'No case returned by /api/monitoring/cases');
      return;
    }
    var head = cachedBannerHtml() + '<div data-replay-toolbar>' + replayBarHtml(c) + '</div>';
    if (!healthy() && !S.fixture) { root.innerHTML = head + serviceUnavailableHtml(); return; }
    if (!S.replayStarted) { root.innerHTML = head + replayIntroHtml(c); return; }
    if (!root.querySelector('.replay-stage')) {
      root.innerHTML = head + replayStageHtml(c);
    }
    updateReplayDom(c, { chrome: true, anchor: true, results: true });
  }

  // ── Validation evidence subview ────────────────────────────────────────
  /* Wording is fixed and ships verbatim: these seven statements describe what
     the coverage numbers do and do not establish. Groups reveal in order when
     the view is opened; no number counts up and nothing is re-run. */
  function renderValidation() {
    var root = el('fx-validation');
    if (!root) return;
    var v = S.validation;
    var runs = v && v.cells && v.cells.length ? (v.cells[0].validation_runs || 10) : 10;
    var cellCount = v ? v.serialized_validated_artifact_count : 12;
    var width = (v && v.interval_width_reduction) || {};
    var widthMin = num(width.minimum_percent);
    var widthMedian = num(width.reported_median_percent_approx);
    var widthMax = num(width.maximum_percent);
    var head = 'eICU-CRD 2.0 retrospective · ' + (cellCount * runs) + ' = ' + cellCount + ' cells × ' +
      runs + ' held-out runs · not independent external validation';
    var body;
    if (S.validationError) {
      body = '<pre>' + esc(S.validationError) + '</pre>';
    } else if (!v) {
      body = '<p class="note">Loading validation evidence…</p>';
    } else {
      var grp = function (i) { return S.validationRevealed >= i ? ' shown' : ''; };
      body =
        '<div class="vgroup' + grp(1) + '">' +
        '<h3 class="h-sec" style="margin-bottom:10px">What the coverage numbers do and do not establish</h3>' +
        '<ol class="validation-list">' +
          '<li><span class="n">01</span><span>The validation report is based on the complete eICU-CRD 2.0 retrospective dataset.</span></li>' +
          '<li><span class="n">02</span><span>120 is 12 cells × 10 held-out runs, not 120 patients.</span></li>' +
          '<li><span class="n">03</span><span>Hospital, care-unit and late/time splits still come from the same database. This is not independent external validation.</span></li>' +
          '<li><span class="n">04</span><span><span class="mono">interval_target_coverage</span> is an artifact\'s nominal target coverage. It is not empirical coverage or accuracy for this replay case.</span></li>' +
          '<li><span class="n">05</span><span>Creatinine@24h records an artifact nominal target coverage of 0.89.</span></li>' +
          '<li><span class="n">06</span><span>The validation report was submitted together with the model. It was not independently re-run from the raw data.</span></li>' +
          '<li><span class="n">07</span><span>The replay case is provenance-labelled eICU CRD Demo 2.0.1. It is not presented as a held-out accuracy example, and its training or validation cohort membership is not established.</span></li>' +
        '</ol><p class="note">The committed report records held-out empirical runs passing the 0.87–0.93 acceptance gate. That empirical gate is separate from the API\'s nominal target-coverage field.</p></div>' +
        '<div class="vgroup' + grp(2) + '">' +
        '<h3 class="h-sec" style="margin-bottom:10px">Gates by cell</h3>' +
        '<div class="scroll-x"><table class="dtable"><thead><tr><th>Cell</th><th>Runs</th>' +
        '<th>Patient held-out gates</th><th>External held-out gates</th></tr></thead><tbody>' +
        (v.cells || []).map(function (cell) {
          return '<tr><td class="mono" data-k="Cell">' + esc(cell.cell) + '</td>' +
            '<td class="num" data-k="Runs">' + esc(cell.validation_runs) + '</td>' +
            '<td class="num" data-k="Patient gates">' + esc(cell.patient_heldout_gates) + '</td>' +
            '<td class="num" data-k="External gates">' + esc(cell.external_heldout_gates) + '</td></tr>';
        }).join('') + '</tbody></table></div></div>' +
        '<div class="vgroup' + grp(3) + '">' +
        '<p class="note" style="margin-top:12px">' + esc(cellCount) + ' cells × ' + esc(runs) + ' runs = ' +
        esc(cellCount * runs) + ' evaluations · model <span class="mono">' + esc(v.model_version) + '</span>. ' +
        'Interval-width reduction is a held-out cohort aggregate, never a case-level claim for this patient.</p>' +
        (widthMin != null && widthMedian != null && widthMax != null
          ? '<p class="note">Committed aggregate interval-width reduction across ' + esc(width.validation_runs || 120) +
            ' held-out runs: minimum ' + esc(widthMin.toFixed(1)) + '%, median approximately ' +
            esc(widthMedian.toFixed(1)) + '%, maximum ' + esc(widthMax.toFixed(1)) +
            '%. Aggregate only; not a current-case shrinkage or accuracy claim.</p>' : '') +
        '<p class="note">These values were recorded when the model shipped. Opening this view reads the committed ' +
        'summary; it runs no inference and re-runs no validation.</p></div>';
    }
    root.innerHTML = '<div class="shell"><div class="section" style="padding-top:24px">' +
      '<div class="section-head"><h1 class="display">Committed retrospective validation evidence</h1>' +
        '<span class="sub">' + esc(head) + '</span></div>' +
      '<div class="validation-static">' + body + '</div>' +
      '</div></div>';
  }

  function clearValidationTimers() {
    S.validationTimers.forEach(clearTimeout);
    S.validationTimers = [];
  }
  function revealValidation() {
    clearValidationTimers();
    if (REDUCED) { S.validationRevealed = 3; renderAll(); return; }
    S.validationRevealed = 1;
    renderAll();
    [2, 3].forEach(function (n, i) {
      S.validationTimers.push(setTimeout(function () {
        S.validationRevealed = n; renderAll();
      }, (i + 1) * 180));
    });
  }

  // ── view renderers ─────────────────────────────────────────────────────
  function loadingHtml() {
    return '<div class="shell"><div class="loading"><span class="spin"></span> ' +
      'Loading case source, model health and validation evidence…</div></div>';
  }

  function renderForecastContext() {
    var host = el('fxContext');
    if (!host) return;
    var c = activeCase();
    if (!c) { host.innerHTML = ''; return; }
    host.innerHTML = caseContextHtml(c);
    Array.prototype.forEach.call(document.querySelectorAll('#fxSubnav .subnav-btn'), function (b) {
      var on = b.getAttribute('data-sub') === S.sub;
      b.classList.toggle('active', on);
      b.setAttribute('aria-current', on ? 'true' : 'false');
    });
  }

  function renderAll() {
    var v = document.body.getAttribute('data-view');
    if (v === 'home') { renderHome(); }
    else if (v === 'forecast') {
      renderForecastContext();
      if (S.sub === 'replay') renderMonitoring();
      else if (S.sub === 'validation') renderValidation();
      else renderSnapshot();
    }
    S.animateData = false;
  }

  // ── replay engine ──────────────────────────────────────────────────────
  async function gotoStep(step, opts) {
    var c = activeCase();
    if (!c) return;
    var target = Math.max(0, Math.min(step, lastStep(c)));
    S.step = target;
    S.replayMaxStep = Math.max(S.replayMaxStep, target);
    S.replayTime = hourAt(c, target);
    S.replayFurthestTime = Math.max(S.replayFurthestTime, S.replayTime);
    S.stepError = null;
    updateReplayDom(c, { chrome: true, anchor: true, results: true });
    if (!healthy()) return;
    if (S.fixture && responseFor(c, target)) {
      S.animateData = true;
      updateReplayDom(c, { chrome: true, anchor: true, results: true });
      return;
    }
    try {
      await fetchStep(c, target, opts);
      S.stepError = null;
    } catch (e) {
      S.stepError = e.message || String(e);
      stopPlay();
    }
    if (S.step === target || (opts && opts.force)) {
      S.animateData = true;
      updateReplayDom(c, { chrome: true, anchor: true, results: true });
    }
  }

  function startReplay() {
    var c = activeCase();
    if (!c) return Promise.resolve();
    S.replayStarted = true;
    S.step = -1;
    S.replayMaxStep = -1;
    S.replayTime = 0;
    S.replayFurthestTime = 0;
    S.stepError = null;
    renderAll();
    return startPlay();
  }

  function resetReplay() {
    stopPlay();
    S.replayStarted = false;
    S.step = -1;
    S.replayMaxStep = -1;
    S.replayTime = 0;
    S.replayFurthestTime = 0;
    S.stepError = null;
    S.freshKey = null;
    renderAll();
  }

  function stopPlay() {
    S.playing = false;
    S.playToken++;
  }

  function nextAnchorAfter(c, replayTime) {
    for (var i = 0; i <= lastStep(c); i++) {
      if (hourAt(c, i) > replayTime + 1e-7) return i;
    }
    return -1;
  }

  function seekReplay(c, replayTime) {
    stopPlay();
    S.replayTime = Math.max(0, Math.min(replayTime, S.replayFurthestTime));
    S.step = replayStepAtTime(c, S.replayTime);
    updateReplayDom(c, { chrome: true, anchor: true, results: true });
  }

  function startPlay() {
    var c = activeCase();
    if (!c || S.playing || !healthy() || !S.replayStarted) return;
    var end = hourAt(c, lastStep(c)) || 0;
    if (S.replayTime >= end - 1e-7) return;
    S.playing = true;
    var token = ++S.playToken;
    var previousStamp = null;
    var duration = REDUCED ? 2400 : REPLAY_DURATION_MS;
    updateReplayDom(c, { chrome: true });

    function frame(stamp) {
      if (!S.playing || token !== S.playToken) return;
      if (previousStamp == null) previousStamp = stamp;
      var delta = Math.max(0, Math.min(100, stamp - previousStamp));
      previousStamp = stamp;
      var targetTime = Math.min(end, S.replayTime + delta * SPEEDS[S.speedIndex].rate * end / duration);
      var next = nextAnchorAfter(c, S.replayTime);
      if (next >= 0 && targetTime >= hourAt(c, next) - 1e-7) {
        S.replayTime = hourAt(c, next);
        S.replayFurthestTime = Math.max(S.replayFurthestTime, S.replayTime);
        updateReplayDom(c);
        gotoStep(next).then(function () {
          if (!S.playing || token !== S.playToken || S.stepError) return;
          previousStamp = null;
          window.requestAnimationFrame(frame);
        });
        return;
      }
      S.replayTime = targetTime;
      S.replayFurthestTime = Math.max(S.replayFurthestTime, targetTime);
      updateReplayDom(c);
      if (targetTime >= end - 1e-7) {
        S.playing = false;
        updateReplayDom(c, { chrome: true });
        return;
      }
      window.requestAnimationFrame(frame);
    }
    window.requestAnimationFrame(frame);
  }

  // ── snapshot run ───────────────────────────────────────────────────────
  async function runSnapshot() {
    var c = activeCase();
    S.stepError = null;
    if (healthy() && c) {
      S.snapshotBusy = true;
      clearStages();
      S.stageIndex = -1;
      renderAll();
      try {
        // "Run" and "Re-run" both mean a real API call, never a session replay.
        var resp = await fetchStep(c, lastStep(c), { force: true });
        S.snapshotRan = true;
        S.snapshotBusy = false;
        S.animateData = true;
        renderAll();
        playStages(resp);
      } catch (e) {
        S.snapshotBusy = false;
        S.stepError = e.message || String(e);
        renderAll();
      }
      return;
    }
    renderAll();
  }

  // ── cached fixtures (explicit action only) ─────────────────────────────
  async function loadFixture() {
    S.fixtureNotice = null;
    try {
      var j = await getJSON('/api/forecast/fixtures');
      var map = {};
      Object.keys(j).filter(function (key) {
        return key.indexOf('post_forecast_') === 0 && key.slice(-7) === '_prefix';
      }).forEach(function (key) {
        var entry = j[key] || {};
        var request = entry.request || {};
        var response = entry.response;
        var trajectory = request.trajectory;
        if (!response || !Array.isArray(trajectory) || !trajectory.length || !request.patient_id) return;
        var step = trajectory.length - 1;
        response._anchor_hour = request.anchor_hour;
        response._step = step;
        response._prefix_rows = trajectory.length;
        map[request.patient_id + '#' + step] = response;
      });
      if (Object.keys(map).length) {
        S.fixture = map;
        S.fixtureNotice = null;
        S.replayStarted = true;
        renderAll();
        return;
      }
      throw new Error('fixture document has no forecast prefix responses');
    } catch (e) {
      S.fixtureNotice = 'No cached research demonstration is available from /api/forecast/fixtures. ' +
        (e && e.message ? e.message + '. ' : '') + 'Nothing was substituted.';
      renderAll();
    }
  }

  // ── guided demo ────────────────────────────────────────────────────────
  /* Two chapters over two independent workflows. The external openFDA and LLM
     options are whatever the user left them as — the guided demo never enables
     a network dependency on the user's behalf. */
  function startGuided() {
    S.guided = { chapter: 1 };
    setView('rx');
    if (window.OslerRx && window.OslerRx.runDefaultScenario) {
      window.OslerRx.runDefaultScenario();
    } else if (window.OslerRx && window.OslerRx.openCasePicker) {
      window.OslerRx.openCasePicker();
    }
  }

  var transitionReturnFocus = null;
  function requestForecastTransition() {
    transitionReturnFocus = document.activeElement;
    S.transition = { to: 'forecast' };
    var scrim = el('transitionScrim');
    if (!scrim) { commitTransition(); return; }
    scrim.setAttribute('data-open', '1');
    var ack = el('transitionAck');
    if (ack) ack.focus();
  }
  function closeTransition(opts) {
    var scrim = el('transitionScrim');
    if (scrim) scrim.setAttribute('data-open', '0');
    S.transition = null;
    if (!(opts && opts.restoreFocus === false) && transitionReturnFocus && transitionReturnFocus.focus) {
      transitionReturnFocus.focus();
    }
    transitionReturnFocus = null;
  }
  function commitTransition() {
    closeTransition({ restoreFocus: false });
    if (S.guided) S.guided.chapter = 2;
    setView('forecast');
    setSub('replay');
    var replayTab = document.querySelector('#fxSubnav [data-sub="replay"]');
    if (replayTab) replayTab.focus();
  }

  // ── view switching + events ────────────────────────────────────────────
  /* Switching a view or a subview never issues a forecast. Metadata loads,
     nothing is computed. */
  function setView(name) {
    document.body.setAttribute('data-view', name);
    Array.prototype.forEach.call(document.querySelectorAll('.nav-view'), function (b) {
      var on = b.getAttribute('data-view') === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-current', on ? 'true' : 'false');
    });
    if (name === 'rx') { stopPlay(); return; }
    if (name === 'home') { stopPlay(); renderAll(); return; }
    S.caseOpened = true;
    renderAll();
    bootstrap().then(renderAll, function (e) {
      S.booted = true;
      S.stepError = e && e.message ? e.message : String(e);
      renderAll();
    });
  }

  function setSub(name) {
    S.sub = name;
    document.body.setAttribute('data-sub', name);
    if (name !== 'replay') stopPlay();
    renderAll();
    if (name === 'validation') revealValidation();
  }

  function onClick(e) {
    var navBtn = e.target.closest ? e.target.closest('.nav-view') : null;
    if (navBtn) { setView(navBtn.getAttribute('data-view')); return; }
    var t = e.target.closest ? e.target.closest('[data-fm]') : null;
    if (!t) return;
    var action = t.getAttribute('data-fm');
    if (action === 'sub') { setSub(t.getAttribute('data-sub') || 'snapshot'); }
    else if (action === 'go-rx') { setView('rx'); }
    else if (action === 'go-forecast') { setView('forecast'); setSub('snapshot'); }
    else if (action === 'start-guided') { startGuided(); }
    else if (action === 'run-snapshot' || action === 'rerun-snapshot') { runSnapshot(); }
    else if (action === 'replay-presentation') { playStages(responseFor(activeCase(), lastStep(activeCase()))); }
    else if (action === 'skip-anim') { skipStages(); }
    else if (action === 'start-replay') { startReplay(); }
    else if (action === 'rerun-step' && S.step >= 0) { stopPlay(); gotoStep(S.step, { force: true }); }
    else if (action === 'play') { startPlay(); }
    else if (action === 'pause') { stopPlay(); updateReplayDom(activeCase(), { chrome: true }); }
    else if (action === 'next') { stopPlay(); gotoStep(S.step + 1); }
    else if (action === 'prev') {
      var c = activeCase();
      if (c) seekReplay(c, S.step <= 0 ? 0 : hourAt(c, S.step - 1));
    }
    else if (action === 'reset') { resetReplay(); }
    else if (action === 'tier') { S.tier = t.getAttribute('data-tier') || 'validated'; S.showAll = false; renderAll(); }
    else if (action === 'toggle-all') { S.showAll = !S.showAll; renderAll(); }
    else if (action === 'toggle-prov') { S.provOpen = !S.provOpen; renderAll(); }
    else if (action === 'fixture-load') { loadFixture(); }
    else if (action === 'fixture-clear') { S.fixture = null; S.replayStarted = false; renderAll(); }
  }

  function onChange(e) {
    var t = e.target.closest ? e.target.closest('[data-fm]') : null;
    if (!t) return;
    var action = t.getAttribute('data-fm');
    if (action === 'speed') {
      S.speedIndex = Number(t.value) || 0;
      updateReplayDom(activeCase(), { chrome: true });
    }
    else if (action === 'scrub') { seekReplay(activeCase(), Number(t.value) || 0); }
    else if (action === 'toggle-extra') { S.showExtraSeries = !!t.checked; renderAll(); }
  }

  function onInput(e) {
    var t = e.target.closest ? e.target.closest('[data-fm="scrub"]') : null;
    if (!t) return;
    seekReplay(activeCase(), Number(t.value) || 0);
  }

  document.addEventListener('click', onClick);
  document.addEventListener('change', onChange);
  document.addEventListener('input', onInput);

  var ack = el('transitionAck'), cancel = el('transitionCancel'), close = el('transitionClose');
  if (ack) ack.addEventListener('click', commitTransition);
  if (cancel) cancel.addEventListener('click', closeTransition);
  if (close) close.addEventListener('click', closeTransition);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && S.transition) closeTransition();
  });

  if (!document.body.getAttribute('data-view')) document.body.setAttribute('data-view', 'home');
  document.body.setAttribute('data-sub', S.sub);
  renderHome();

  window.OslerForecastViews = {
    setView: setView,
    setSub: setSub,
    startGuided: startGuided,
    requestForecastTransition: requestForecastTransition,
    state: S
  };
})();
