/* Osler research forecast + remote monitoring replay.
 *
 * Two research-only views on top of the existing Rx demo:
 *   #view-forecast    factual patient-state forecast for the case anchor
 *   #view-monitoring  step-by-step replay of a retrospective observation trajectory
 *
 * Hard rules encoded here:
 *   - every /api/forecast request carries a causal PREFIX of the trajectory only;
 *     anchor_hour is the last prefix observation, and a client-side guard refuses
 *     to send a payload that contains anything after it;
 *   - forecasts are never invented: unsupported cells show no point and no interval,
 *     illustrative cells show no interval, only validated artifacts show lower/upper;
 *   - interval-width evidence is labelled as held-out cohort aggregate, never as
 *     shrinkage for this patient;
 *   - a model-health failure or an API error stops forecasting and is shown as an
 *     error; cached fixtures are only ever loaded on an explicit click and are
 *     labelled "Cached research demonstration";
 *   - no diagnosis, causal claim, treatment recommendation or drug-ranking change.
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
  var MATRIX_TARGETS = ['creatinine', 'bun', 'urine_output', 'map'];
  var MATRIX_HORIZONS = [3, 6, 12, 24, 48];
  var SPEEDS = [
    { label: '0.5×', ms: 5200 },
    { label: '1×', ms: 2600 },
    { label: '2×', ms: 1300 }
  ];

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
    pending: {},        // "<caseId>#<step>" -> true while in flight
    stepError: null,
    history: [],        // one snapshot row per (step, cell) actually forecast
    step: 0,
    playing: false,
    speedIndex: 1,
    playToken: 0,
    fixture: null,      // loaded cached demonstration payloads (explicit click only)
    fixtureNotice: null,
    showAllTiers: false,
    targetFilter: 'all',
    horizonFilter: 'all',
    showExtraSeries: false,
    fcTargetFilter: 'all',
    fcHorizonFilter: 'all',
    fcShowAllTiers: false
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
  function bootstrap() {
    if (S.booting) return S.booting;
    S.booting = (async function () {
      await Promise.all([loadHealth(), loadCases(), loadValidation()]);
      S.booted = true;
      var c = activeCase();
      if (healthy() && c) {
        // A prefetch failure must surface as an error state, never leave the
        // views stuck on a loading placeholder or poison the boot promise.
        await Promise.all([lastStep(c), 0].map(function (step) {
          return fetchStep(c, step).catch(function (e) {
            S.stepError = e.message || String(e);
          });
        }));
      }
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

  async function fetchStep(c, step) {
    var key = cacheKey(c, step);
    if (S.cache[key]) return S.cache[key];
    if (!healthy()) throw new Error('Forecast service unavailable');
    if (S.pending[key]) return S.pending[key];
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

  // ── shared render blocks ───────────────────────────────────────────────
  function provenanceHtml(c, step) {
    if (!c) return '';
    var meta = c.source_metadata || {};
    var ctx = c.case_context || {};
    var anchorNow = hourAt(c, step);
    var alias = caseAlias(c);
    var admission = ctx.source_admission_label;
    return '' +
      '<section class="fm-card fm-prov">' +
        '<div>' +
          '<h2 class="fm-h2">' + esc(alias) + ' · ' + esc(c.title || 'Monitoring trajectory') + '</h2>' +
          '<div class="fm-sub">Deidentified retrospective eICU Demo case · Research replay — not live monitoring.</div>' +
          '<div class="fm-badges fm-row">' +
            '<span class="fm-tag research">Research use only</span>' +
            '<span class="fm-tag research">Deidentified retrospective eICU Demo case</span>' +
            '<span class="fm-tag research">Research replay — not live monitoring</span>' +
            '<span class="fm-tag boundary">No diagnosis / no treatment recommendation</span>' +
            '<span class="fm-tag boundary">No causal or counterfactual claim</span>' +
          '</div>' +
          '<dl class="fm-kv">' +
            '<dt>Dataset</dt><dd>' + esc(meta.dataset || c.case_source || 'unknown') +
              (meta.version ? ' v' + esc(meta.version) : '') + '</dd>' +
            '<dt>Source</dt><dd>' + (meta.url
                ? '<span class="fm-mono">' + esc(meta.url) + '</span>' : 'not supplied') + '</dd>' +
            '<dt>Licence</dt><dd>' + esc(meta.license || 'not supplied') + '</dd>' +
            '<dt>Deidentification</dt><dd>' + esc(meta.deidentification || 'not supplied') + '</dd>' +
            '<dt>Model version</dt><dd class="fm-mono">' +
              esc((S.casesMeta && S.casesMeta.model_version) || (S.health && S.health.model_version) || '—') + '</dd>' +
            '<dt>Case context</dt><dd>' +
              esc([ctx.age ? ctx.age + ' y' : null, ctx.gender, ctx.unit_type]
                .filter(Boolean).join(' · ') || 'not supplied') + '</dd>' +
            '<dt>Current time point</dt><dd><b>' + esc(fmtHour(anchorNow)) +
              '</b> after ICU admission · observation ' + (step + 1) + ' of ' +
              (lastStep(c) + 1) + '</dd>' +
          '</dl>' +
          (admission
            ? '<div class="fm-src-label" style="margin-top:9px"><b>Historical admission label recorded in the source data:</b> ' +
              esc(admission) +
              '<span class="fm-cav">This is a label carried in the source dataset, not a model output and not a diagnosis produced by this system.</span></div>'
            : '') +
          '<div class="fm-note">Patient identifiers are not displayed. The source stay key is retained only inside the dataset provenance record. ' +
            'Observation units follow the eICU demo adapter; urine output is an interval-normalised mL/hour rate.</div>' +
        '</div>' +
        '<div class="fm-health">' + healthBlockHtml() + '</div>' +
      '</section>';
  }

  function healthBlockHtml() {
    if (S.healthError) {
      return '<span class="fm-tag bad">Model health unknown</span>' +
        '<div class="fm-note">' + esc(S.healthError) + '</div>';
    }
    if (!S.health) return '<span class="fm-tag boundary">Checking model health…</span>';
    var failed = failedArtifacts();
    var integrity = integrityHealth();
    var inference = S.health.inference_smoke || {};
    if (healthy()) {
      return '<b>Model health</b><span class="fm-tag ok">Healthy</span>' +
        '<div class="fm-note">' + esc(integrity.loaded_artifacts) + ' / ' + esc(integrity.expected_artifacts) +
        ' serialized artifacts loaded · ' + esc(inference.validated_artifacts) + ' / ' +
        esc(inference.expected_validated_artifacts) + ' inference smoke predictions passed' +
        (S.health.strict_manifest_verification ? ' · strict manifest verification' : '') +
        '</div>';
    }
    return '<b>Model health</b><span class="fm-tag bad">Degraded</span>' +
      '<div class="fm-note">integrity ' + esc(integrity ? integrity.status : 'unknown') + ' · inference smoke ' +
      esc(inference.status || 'unknown') + ' · ' + failed.length + ' artifact load failures</div>';
  }

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
    return '<section class="fm-card fm-error">' +
      '<h2 class="fm-h2">Forecast service unavailable</h2>' +
      '<div class="fm-sub">Artifact integrity or deterministic inference smoke did not pass, so no forecast is requested. ' +
      'No substitute, estimated or mock values are shown.</div>' +
      (S.health ? '<div class="fm-note">Health status: <b>' + esc(S.health.status) + '</b></div>' : '') +
      detail + fixtureControlsHtml() + '</section>';
  }

  function errorCardHtml(title, message) {
    return '<section class="fm-card fm-error">' +
      '<h2 class="fm-h2">' + esc(title) + '</h2>' +
      '<div class="fm-sub">No forecast values are shown for this step. Nothing is substituted.</div>' +
      '<pre>' + esc(message) + '</pre>' + fixtureControlsHtml() + '</section>';
  }

  function fixtureControlsHtml() {
    if (S.fixture) {
      return '<div class="fm-row" style="margin-top:10px">' +
        '<span class="fm-tag cached">Cached research demonstration</span>' +
        '<button class="fm-btn" data-fm="fixture-clear">Clear cached demonstration</button></div>';
    }
    return '<div class="fm-row" style="margin-top:10px">' +
      '<button class="fm-btn" data-fm="fixture-load">Load cached research demonstration</button>' +
      '<span class="fm-note" style="margin:0">Loads recorded fixture responses if the backend ships them. ' +
      'They are always labelled <b>Cached research demonstration</b> and are never substituted automatically.</span>' +
      (S.fixtureNotice ? '<div class="fm-note">' + esc(S.fixtureNotice) + '</div>' : '') + '</div>';
  }

  function cachedBannerHtml() {
    if (!S.fixture) return '';
    return '<section class="fm-card fm-cached-banner">' +
      '<span class="fm-tag cached">Cached research demonstration</span>' +
      '<div class="fm-note" style="margin-top:6px">These values come from recorded fixture responses, ' +
      'not from a live model call in this session.</div></section>';
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
      if (span / candidates[i] <= 6) return candidates[i];
    }
    return 48;
  }

  function chartSvg(opt) {
    var W = 640, H = 178, PL = 50, PR = 18, PT = 12, PB = 28;
    var obs = opt.obs, fcs = opt.forecasts, anchor = opt.anchor;
    var xs = [0], ys = [];
    obs.forEach(function (o) { xs.push(o.hour); ys.push(o.value); });
    fcs.forEach(function (f) {
      xs.push(f.due);
      ys.push(f.personalized);
      if (isNum(f.population)) ys.push(f.population);
      if (isNum(f.lower)) ys.push(f.lower);
      if (isNum(f.upper)) ys.push(f.upper);
    });
    if (anchor != null) xs.push(anchor);
    ys = ys.filter(isNum);
    if (!ys.length) return null;
    var xMin = 0, xMax = Math.max.apply(null, xs);
    if (xMax - xMin < 1) xMax = xMin + 1;
    var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
    if (yMax - yMin < 1e-6) { yMax = yMin + Math.max(1, Math.abs(yMin) * 0.1); }
    var pad = (yMax - yMin) * 0.14;
    var rawMin = yMin;
    yMin -= pad; yMax += pad;
    // Padding alone must not invent a negative range. If every plotted value is
    // non-negative the axis stops at 0; if the model itself returned a negative
    // bound (it does for urine output) that value is still shown as returned.
    if (rawMin >= 0 && yMin < 0) yMin = 0;
    var sx = function (h) { return PL + (h - xMin) / (xMax - xMin) * (W - PL - PR); };
    var sy = function (v) { return PT + (yMax - v) / (yMax - yMin) * (H - PT - PB); };
    var d = decimalsFor(opt.target);
    var g = [];

    // y grid
    [0, 0.5, 1].forEach(function (t) {
      var v = yMin + (yMax - yMin) * t, y = sy(v);
      g.push('<line x1="' + PL + '" y1="' + y.toFixed(1) + '" x2="' + (W - PR) + '" y2="' + y.toFixed(1) +
        '" stroke="#E2E7F0" stroke-width="1"/>');
      g.push('<text x="' + (PL - 6) + '" y="' + (y + 3.5).toFixed(1) + '" text-anchor="end" font-size="10" fill="#8C9BB5">' +
        esc(v.toFixed(d)) + '</text>');
    });
    // x axis
    var step = niceTickStep(xMax - xMin);
    g.push('<line x1="' + PL + '" y1="' + (H - PB) + '" x2="' + (W - PR) + '" y2="' + (H - PB) +
      '" stroke="#C5CEDB" stroke-width="1"/>');
    for (var t = 0; t <= xMax + 1e-9; t += step) {
      var x = sx(t);
      g.push('<line x1="' + x.toFixed(1) + '" y1="' + (H - PB) + '" x2="' + x.toFixed(1) + '" y2="' + (H - PB + 4) +
        '" stroke="#C5CEDB" stroke-width="1"/>');
      g.push('<text x="' + x.toFixed(1) + '" y="' + (H - PB + 16) + '" text-anchor="middle" font-size="10" fill="#8C9BB5">' +
        t + '</text>');
    }
    g.push('<text x="' + (W - PR) + '" y="' + (H - 3) + '" text-anchor="end" font-size="9.5" fill="#8C9BB5">' +
      'hours after ICU admission</text>');

    // anchor marker
    if (anchor != null) {
      var ax = sx(anchor);
      g.push('<line x1="' + ax.toFixed(1) + '" y1="' + PT + '" x2="' + ax.toFixed(1) + '" y2="' + (H - PB) +
        '" stroke="#E8724A" stroke-width="1.2" stroke-dasharray="3 3"/>');
      g.push('<text x="' + (ax + 4).toFixed(1) + '" y="' + (PT + 9) + '" font-size="9.5" fill="#E8724A" font-weight="600">anchor</text>');
    }

    // observed (solid)
    if (obs.length) {
      var pts = obs.map(function (o) { return sx(o.hour).toFixed(1) + ',' + sy(o.value).toFixed(1); }).join(' ');
      if (obs.length > 1) {
        g.push('<polyline points="' + pts + '" fill="none" stroke="#0F2B5B" stroke-width="2.1" ' +
          'stroke-linejoin="round" stroke-linecap="round"/>');
      }
      obs.forEach(function (o) {
        g.push('<circle cx="' + sx(o.hour).toFixed(1) + '" cy="' + sy(o.value).toFixed(1) +
          '" r="3.4" fill="#0F2B5B"/>');
      });
      var last = obs[obs.length - 1];
      // dashed link from the last observed value into the forecast points
      var persPts = fcs.filter(function (f) { return isNum(f.personalized); })
        .map(function (f) { return sx(f.due).toFixed(1) + ',' + sy(f.personalized).toFixed(1); });
      if (persPts.length) {
        g.push('<polyline points="' + sx(last.hour).toFixed(1) + ',' + sy(last.value).toFixed(1) + ' ' +
          persPts.join(' ') + '" fill="none" stroke="#0F2B5B" stroke-width="1.3" ' +
          'stroke-dasharray="4 4" opacity="0.55"/>');
      }
    }

    // forecasts
    fcs.forEach(function (f) {
      var x = sx(f.due);
      if (isNum(f.lower) && isNum(f.upper)) {
        var yl = sy(f.lower), yu = sy(f.upper);
        g.push('<line x1="' + x.toFixed(1) + '" y1="' + yu.toFixed(1) + '" x2="' + x.toFixed(1) + '" y2="' + yl.toFixed(1) +
          '" stroke="#0F2B5B" stroke-width="1.6" opacity="0.42"/>');
        g.push('<line x1="' + (x - 4).toFixed(1) + '" y1="' + yu.toFixed(1) + '" x2="' + (x + 4).toFixed(1) + '" y2="' + yu.toFixed(1) +
          '" stroke="#0F2B5B" stroke-width="1.6" opacity="0.42"/>');
        g.push('<line x1="' + (x - 4).toFixed(1) + '" y1="' + yl.toFixed(1) + '" x2="' + (x + 4).toFixed(1) + '" y2="' + yl.toFixed(1) +
          '" stroke="#0F2B5B" stroke-width="1.6" opacity="0.42"/>');
      }
      if (isNum(f.population)) {
        // population POINT only: never drawn as a band, because it has no interval
        var yp = sy(f.population);
        g.push('<line x1="' + (x - 5).toFixed(1) + '" y1="' + yp.toFixed(1) + '" x2="' + (x + 5).toFixed(1) + '" y2="' + yp.toFixed(1) +
          '" stroke="#0D9488" stroke-width="2"/>');
      }
      var y = sy(f.personalized), r = 4.2;
      if (f.tier === 'validated_artifact') {
        g.push('<polygon points="' + [
          x.toFixed(1) + ',' + (y - r).toFixed(1), (x + r).toFixed(1) + ',' + y.toFixed(1),
          x.toFixed(1) + ',' + (y + r).toFixed(1), (x - r).toFixed(1) + ',' + y.toFixed(1)
        ].join(' ') + '" fill="#FFFFFF" stroke="#0F2B5B" stroke-width="2"/>');
      } else {
        g.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="3.6" fill="#FFFFFF" ' +
          'stroke="#8C9BB5" stroke-width="1.6" stroke-dasharray="2 1.6"/>');
      }
    });

    return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
      esc(labelFor(opt.target) + ' trajectory and forecasts') + '">' + g.join('') + '</svg>';
  }

  function chartCardHtml(c, target, resp, uptoStep) {
    var obs = observationsFor(c, target, uptoStep);
    var fcs = resp ? forecastPointsFor(resp, target) : [];
    var unit = unitFor(target);
    var anchor = resp ? num(resp._anchor_hour) : hourAt(c, uptoStep);
    var head = '<div class="fm-chart-h"><b>' + esc(labelFor(target)) + '</b><span>' + esc(unit) + '</span>';
    if (obs.length) {
      var last = obs[obs.length - 1];
      head += '<span class="fm-latest">latest ' + esc(fmtVal(last.value, target)) + ' ' + esc(unit) +
        ' @ ' + esc(fmtHour(last.hour)) + '</span>';
    } else {
      head += '<span class="fm-latest">Not observed</span>';
    }
    head += '</div>';
    var svg = (obs.length || fcs.length) ? chartSvg({
      target: target, obs: obs, forecasts: fcs, anchor: anchor
    }) : null;
    var body = svg || '<div class="fm-nodata">Not observed — no value for this variable up to ' +
      esc(fmtHour(anchor)) + '. Nothing is imputed or interpolated.</div>';
    return '<div class="fm-chart">' + head + body + '</div>';
  }

  function chartsHtml(c, resp, uptoStep) {
    var series = PRIMARY_SERIES.concat(S.showExtraSeries ? EXTRA_SERIES : []);
    return '<section class="fm-card">' +
      '<div class="fm-row"><h3 class="fm-h" style="margin:0">Patient trajectory · observed and forecast</h3>' +
      '<label class="fm-note" style="margin:0 0 0 auto"><input type="checkbox" data-fm="toggle-extra"' +
        (S.showExtraSeries ? ' checked' : '') + '> show glucose · potassium · heart rate</label></div>' +
      '<div class="fm-charts">' +
        series.map(function (t) { return chartCardHtml(c, t, resp, uptoStep); }).join('') +
      '</div>' +
      '<div class="fm-legend">' +
        '<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="#0F2B5B" stroke-width="2.1"/>' +
          '<circle cx="13" cy="5" r="3.2" fill="#0F2B5B"/></svg> observed value</span>' +
        '<span><svg width="18" height="12"><polygon points="9,2 15,6 9,10 3,6" fill="#fff" stroke="#0F2B5B" stroke-width="2"/></svg>' +
          ' personalized point (validated artifact)</span>' +
        '<span><svg width="18" height="14"><line x1="9" y1="1" x2="9" y2="13" stroke="#0F2B5B" stroke-width="1.6" opacity=".42"/>' +
          '<line x1="5" y1="1" x2="13" y2="1" stroke="#0F2B5B" stroke-width="1.6" opacity=".42"/>' +
          '<line x1="5" y1="13" x2="13" y2="13" stroke="#0F2B5B" stroke-width="1.6" opacity=".42"/></svg>' +
          ' target-specific calibrated research interval</span>' +
        '<span><svg width="18" height="10"><circle cx="9" cy="5" r="3.6" fill="#fff" stroke="#8C9BB5" stroke-width="1.6" stroke-dasharray="2 1.6"/></svg>' +
          ' illustrative point (no interval)</span>' +
        '<span><svg width="18" height="10"><line x1="3" y1="5" x2="15" y2="5" stroke="#0D9488" stroke-width="2"/></svg>' +
          ' population point (no interval)</span>' +
      '</div>' +
      '<div class="fm-note">Solid line = observations available up to the anchor. Forecast markers are drawn only at their due hour. ' +
      'A population point is drawn as a point, never as a band, because it carries no interval. Missing values are shown as ' +
      '<i>Not observed</i> and are never interpolated. Raw calibrated statistical intervals are displayed unchanged and may ' +
      'extend beyond physiological support, including below zero; they have no clinical interpretation.</div>' +
      '</section>';
  }

  // ── forecast cards ─────────────────────────────────────────────────────
  function tierMeta(tier) {
    if (tier === 'validated_artifact') {
      return { cls: 'validated', label: 'Validated serialized artifact' };
    }
    if (tier === 'illustrative') {
      return { cls: 'illustrative', label: 'Illustrative belief output' };
    }
    return { cls: 'unsupported', label: 'Model abstained' };
  }

  function forecastCardHtml(f, c, resp, uptoStep) {
    var meta = tierMeta(f.tier);
    var unit = unitFor(f.target);
    var anchor = num(resp._anchor_hour);
    var due = anchor != null ? anchor + num(f.horizon_hours) : null;
    var h = '<article class="fm-fc ' + meta.cls + '">' +
      '<div class="fm-fc-top">' +
        '<span class="fm-fc-name">' + esc(labelFor(f.target)) + '</span>' +
        '<span class="fm-fc-h">+' + esc(f.horizon_hours) + ' h</span>' +
        '<span class="fm-fc-tier">' + esc(meta.label) + '</span>' +
      '</div>' +
      '<div class="fm-note" style="margin-top:4px">' + esc(f.system) + ' system · due at ' +
        esc(fmtHour(due)) + ' after ICU admission</div>';

    if (f.tier === 'unsupported') {
      h += '<div class="fm-fc-msg">No point forecast and no interval are produced for this cell. ' +
        'The model abstained rather than emitting a value.' +
        (f.reason ? '<br>Reason: ' + esc(f.reason) : '') +
        (f.status ? '<br>Status: <span class="fm-mono">' + esc(f.status) + '</span>' : '') +
        (f.interval_status ? '<br>Interval status: <span class="fm-mono">' + esc(f.interval_status) + '</span>' : '') +
        '</div></article>';
      return h;
    }

    var pop = f.population ? num(f.population.point) : null;
    var pers = f.personalized ? num(f.personalized.point) : null;
    h += '<div class="fm-fc-nums">' +
      '<div class="fm-fc-num pop"><div class="k">Population point</div><div class="v">' +
        (pop == null ? '<small>not produced</small>' : esc(fmtVal(pop, f.target)) + '<small>' + esc(unit) + '</small>') +
      '</div></div>' +
      '<div class="fm-fc-num"><div class="k">Personalized point</div><div class="v">' +
        (pers == null ? '<small>not produced</small>' : esc(fmtVal(pers, f.target)) + '<small>' + esc(unit) + '</small>') +
      '</div></div>';
    if (f.personalized && isNum(num(f.personalized.delta_vs_population))) {
      var dv = num(f.personalized.delta_vs_population);
      h += '<div class="fm-fc-num"><div class="k">Δ vs population</div><div class="v">' +
        (dv >= 0 ? '+' : '') + esc(fmtVal(dv, f.target)) + '<small>' + esc(unit) + '</small></div></div>';
    }
    h += '</div>';

    if (f.tier === 'validated_artifact') {
      if (isNum(num(f.lower)) && isNum(num(f.upper))) {
        var cov = num(f.interval_target_coverage);
        h += '<div class="fm-fc-int"><b>' + esc(fmtVal(num(f.lower), f.target)) + ' – ' +
          esc(fmtVal(num(f.upper), f.target)) + ' ' + esc(unit) + '</b><br>' +
          (cov != null ? 'Calibrated ' + esc((cov * 100).toFixed(0)) + '% ' : 'Target-specific calibrated ') +
          '<b>research</b> interval from the serialized patient-specific conformal artifact. ' +
          'This is not a clinical confidence interval and carries no clinical guarantee.' +
          (cov != null ? '<br><span class="fm-note" style="margin:0">Target coverage reported by the artifact: ' +
            esc(cov.toFixed(4)) + '</span>' : '') +
          (f.note ? '<br><span class="fm-note" style="margin:0">' + esc(f.note) + '</span>' : '') +
          '</div>';
      } else {
        h += '<div class="fm-fc-msg">This validated cell returned no lower/upper bound in this response. ' +
          'No interval is displayed and none is inferred.</div>';
      }
    } else {
      h += '<div class="fm-fc-msg"><b>No calibrated interval.</b> This is a belief-derived research illustration; ' +
        'lower and upper bounds are not produced for this cell and are not estimated here.' +
        (f.interval_status ? '<br>Interval status: <span class="fm-mono">' + esc(f.interval_status) + '</span>' : '') +
        '</div>';
    }

    h += '<div class="fm-fc-src">Status <span class="fm-mono">' + esc(f.status || '—') + '</span>' +
      (f.population && f.population.source ? ' · population <span class="fm-mono">' + esc(f.population.source) + '</span>' : '') +
      (f.personalized && f.personalized.source ? ' · personalized <span class="fm-mono">' + esc(f.personalized.source) + '</span>' : '') +
      '</div>';

    h += outcomeHtml(f, c, resp, uptoStep, due, unit);
    return h + '</article>';
  }

  function outcomeHtml(f, c, resp, uptoStep, due, unit) {
    var row = {
      target: f.target, anchorHour: num(resp._anchor_hour), dueHour: due,
      lower: num(f.lower), upper: num(f.upper)
    };
    var res = resolveSnapshot(row, c, uptoStep);
    if (res.state === 'awaiting') {
      return '<div class="fm-fc-outcome awaiting">' +
        (res.policy && res.policy.enabled
          ? 'Awaiting a matching future observation (±' + esc(res.policy.tolerance) + ' h of the due hour · ' + esc(res.policy.source) + ').'
          : 'Awaiting / no configured matching. The backend replay contract disables observation matching.') +
        ' No accuracy verdict is made.</div>';
    }
    var stamp = 'Observed ' + esc(fmtVal(res.value, f.target)) + ' ' + esc(unit) + ' at ' + esc(fmtHour(res.hour)) +
      ' (Δ ' + esc((Math.round(res.delta * 100) / 100).toFixed(2)) + ' h from the due hour, matched within ±' +
      esc(res.policy.tolerance) + ' h · ' + esc(res.policy.source) + ').';
    if (res.state === 'inside') {
      return '<div class="fm-fc-outcome inside">Observed value was inside the research interval.<br>' + stamp + '</div>';
    }
    if (res.state === 'outside') {
      return '<div class="fm-fc-outcome outside">Observed value was outside the research interval.<br>' + stamp + '</div>';
    }
    return '<div class="fm-fc-outcome plain">' + stamp +
      '<br>No calibrated interval exists for this cell, so no interval comparison is made.</div>';
  }

  function filterForecasts(resp, targetFilter, horizonFilter, showAll) {
    var list = (resp.forecasts || []).slice();
    if (!showAll) list = list.filter(function (f) { return f.tier === 'validated_artifact'; });
    if (targetFilter !== 'all') list = list.filter(function (f) { return f.target === targetFilter; });
    if (horizonFilter !== 'all') {
      list = list.filter(function (f) { return String(f.horizon_hours) === String(horizonFilter); });
    }
    var order = { validated_artifact: 0, illustrative: 1, unsupported: 2 };
    list.sort(function (a, b) {
      return (order[a.tier] - order[b.tier]) ||
        a.target.localeCompare(b.target) || (a.horizon_hours - b.horizon_hours);
    });
    return list;
  }

  function filterBarHtml(resp, prefix, targetFilter, horizonFilter, showAll, shown) {
    var targets = {}, horizons = {};
    (resp.forecasts || []).forEach(function (f) {
      targets[f.target] = true; horizons[f.horizon_hours] = true;
    });
    var tOpts = ['<option value="all">All targets</option>'].concat(
      Object.keys(targets).sort().map(function (t) {
        return '<option value="' + esc(t) + '"' + (t === targetFilter ? ' selected' : '') + '>' + esc(labelFor(t)) + '</option>';
      })).join('');
    var hOpts = ['<option value="all">All horizons</option>'].concat(
      Object.keys(horizons).map(Number).sort(function (a, b) { return a - b; }).map(function (h) {
        return '<option value="' + h + '"' + (String(h) === String(horizonFilter) ? ' selected' : '') + '>+' + h + ' h</option>';
      })).join('');
    return '<div class="fm-filters">' +
      '<select class="fm-select" data-fm="' + prefix + '-target">' + tOpts + '</select>' +
      '<select class="fm-select" data-fm="' + prefix + '-horizon">' + hOpts + '</select>' +
      '<label><input type="checkbox" data-fm="' + prefix + '-alltiers"' + (showAll ? ' checked' : '') +
        '> include illustrative and abstained cells</label>' +
      '<span class="fm-count">' + shown + ' shown</span></div>';
  }

  // ── evidence matrix + validation summary ───────────────────────────────
  function validationHtml(resp) {
    if (S.validationError) {
      return errorCardHtml('Validation evidence unavailable', S.validationError);
    }
    var v = S.validation;
    if (!v) return '<section class="fm-card"><h3 class="fm-h">Validation evidence</h3>' +
      '<div class="fm-empty">Loading…</div></section>';
    var r = v.interval_width_reduction || {};
    var gates = {};
    (v.cells || []).forEach(function (cell) { gates[cell.cell] = cell; });
    var liveStatus = {};
    if (resp) {
      (resp.forecasts || []).forEach(function (f) {
        var k = f.target + '@' + f.horizon_hours + 'h';
        if (f.tier === 'validated_artifact' || !liveStatus[k]) liveStatus[k] = f;
      });
    }
    var rows = MATRIX_TARGETS.map(function (t) {
      return '<tr><td class="rowh">' + esc(labelFor(t)) + '</td>' +
        MATRIX_HORIZONS.map(function (h) {
          var k = t + '@' + h + 'h';
          var g = gates[k];
          if (!g) return '<td><span class="fm-cell n">—</span></td>';
          var live = liveStatus[k];
          var sub = 'patient ' + esc(g.patient_heldout_gates) + ' · external ' + esc(g.external_heldout_gates);
          if (live && live.tier !== 'validated_artifact') {
            return '<td><span class="fm-cell abstain">abstained<small>' +
              esc(live.status || live.tier) + '</small></span></td>';
          }
          return '<td><span class="fm-cell v">validated<small>' + sub + '</small></span></td>';
        }).join('') + '</tr>';
    }).join('');

    return '<section class="fm-card">' +
      '<h3 class="fm-h">Validation evidence · held-out cohort aggregate</h3>' +
      '<div class="fm-sub"><b>' + esc(v.serialized_validated_artifact_count) +
        '</b> serialized validated target × horizon cells · <b>' + esc(r.validation_runs) +
        '</b> held-out validation runs · model <span class="fm-mono">' + esc(v.model_version) + '</span></div>' +
      '<div class="fm-stats">' +
        '<div class="fm-stat"><div class="k">Minimum</div><div class="v">≈' + esc(r.minimum_percent) +
          '%</div><div class="c">interval-width reduction · held-out cohort aggregate</div></div>' +
        '<div class="fm-stat"><div class="k">Median</div><div class="v">≈' +
          esc(r.reported_median_percent_approx != null ? r.reported_median_percent_approx : r.median_percent) +
          '%</div><div class="c">interval-width reduction · held-out cohort aggregate</div></div>' +
        '<div class="fm-stat"><div class="k">Maximum</div><div class="v">≈' + esc(r.maximum_percent) +
          '%</div><div class="c">interval-width reduction · held-out cohort aggregate</div></div>' +
      '</div>' +
      '<div class="fm-note" style="font-size:.79rem;color:var(--text-secondary)">' +
        '<b>These are held-out cohort aggregate results, not real-time interval shrinkage for this individual case.</b> ' +
        'Every number above summarises ' + esc(r.validation_runs) + ' held-out validation runs across the ' +
        esc(v.serialized_validated_artifact_count) + ' serialized cells. They say nothing about how much any interval ' +
        'narrowed for the patient currently on screen. They are not a measure of predictive accuracy, ' +
        'of diagnostic performance, or of clinical performance.' +
        (r.scope ? ' Backend scope flag: <span class="fm-mono">' + esc(r.scope) + '</span>.' : '') +
        (r.case_level_claim_allowed === false ? ' Case-level claims are explicitly disallowed by the backend contract.' : '') +
      '</div>' +
      '<hr class="fm-hr">' +
      '<h3 class="fm-h">12-cell evidence matrix</h3>' +
      '<div class="fm-matrix-wrap"><table class="fm-matrix"><thead><tr><th class="rowh">Target</th>' +
        MATRIX_HORIZONS.map(function (h) { return '<th>+' + h + ' h</th>'; }).join('') +
      '</tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<div class="fm-note">“validated” = an exact serialized target × horizon artifact exists and passed its patient / external held-out gates. ' +
      '“abstained” = the artifact exists but this specific request has no usable current value, so the runtime failed closed for this step. ' +
      '“—” = no serialized artifact; the cell is outside the validated set.</div>' +
      '</section>';
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

  function signalsHtml(c, resp, uptoStep) {
    var sigs = computeSignals(c, resp, uptoStep);
    return '<section class="fm-card tight">' +
      '<h3 class="fm-h">Research signals</h3>' +
      sigs.map(function (s) {
        return '<div class="fm-sig"><span class="nm">' + esc(labelFor(s.target)) + '</span>' +
          '<span class="fm-state ' + esc(s.state) + '">' + esc(s.label) + '</span>' +
          '<span class="why">' + esc(s.why) + '</span></div>';
      }).join('') +
      '<div class="fm-note">These four states are descriptive display labels derived from the observations and model outputs on screen. ' +
      'They are not clinical thresholds, not alerts, and not a severity score. They do not indicate deterioration, ' +
      'name a condition, or suggest any action, and they never change drug ranking or dosing.</div>' +
      '</section>';
  }

  // ── history table ──────────────────────────────────────────────────────
  function historyHtml(c, uptoStep) {
    var rows = historyFor(c, uptoStep);
    if (!rows.length) {
      return '<section class="fm-card"><h3 class="fm-h">Forecast history · issued snapshots</h3>' +
        '<div class="fm-empty">No forecast snapshots yet.</div></section>';
    }
    var body = rows.slice().reverse().map(function (r) {
      var row = r.row, res = r.res, unit = unitFor(row.target);
      var outcome, cls;
      if (res.state === 'inside') { outcome = 'Inside research interval'; cls = 'inside'; }
      else if (res.state === 'outside') { outcome = 'Outside research interval'; cls = 'outside'; }
      else if (res.state === 'observed') { outcome = 'Observed ' + fmtVal(res.value, row.target) + ' ' + unit + ' · no interval to compare'; cls = 'awaiting'; }
      else { outcome = 'Awaiting / no configured matching'; cls = 'awaiting'; }
      return '<tr class="tier-' + esc(row.tier) + '">' +
        '<td class="num">' + esc(row.step + 1) + '</td>' +
        '<td class="num">' + esc(fmtHour(row.anchorHour)) + '</td>' +
        '<td>' + esc(labelFor(row.target)) + '</td>' +
        '<td class="num">+' + esc(row.horizon) + ' h</td>' +
        '<td class="num">' + esc(fmtHour(row.dueHour)) + '</td>' +
        '<td class="num">' + esc(fmtVal(row.personalized, row.target) || '—') + ' ' + esc(unit) + '</td>' +
        '<td class="num">' + (isNum(row.lower) && isNum(row.upper)
          ? esc(fmtVal(row.lower, row.target)) + ' – ' + esc(fmtVal(row.upper, row.target))
          : '<span style="color:var(--text-muted)">no calibrated interval</span>') + '</td>' +
        '<td><span class="fm-pillmini ' + (row.tier === 'validated_artifact' ? 'validated' : 'illustrative') + '">' +
          esc(row.tier === 'validated_artifact' ? 'validated' : 'illustrative') + '</span></td>' +
        '<td><span class="fm-pillmini ' + cls + '">' + esc(outcome) + '</span></td>' +
        '</tr>';
    }).join('');
    var pol = matchPolicy(c);
    return '<section class="fm-card">' +
      '<h3 class="fm-h">Forecast history · issued snapshots</h3>' +
      '<div class="fm-hist-wrap"><table class="fm-hist"><thead><tr>' +
        '<th>Step</th><th>Anchor</th><th>Target</th><th>Horizon</th><th>Due hour</th>' +
        '<th>Personalized</th><th>Research interval</th><th>Tier</th><th>Later observation</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table></div>' +
      '<div class="fm-note">Each row is the snapshot issued at that replay step, from the prefix available at that time. ' +
      (pol.enabled
        ? 'A later observation is only paired with a snapshot when it falls within ±' + esc(pol.tolerance) +
          ' h of the due hour (' + esc(pol.source) + '); otherwise the row stays unresolved. '
        : 'The backend replay contract has no configured observation matching, so rows stay unresolved. ') +
      'No row is scored as correct or incorrect, and the nearest available observation is never treated as ground truth.</div>' +
      '</section>';
  }

  // ── Rx handoff ─────────────────────────────────────────────────────────
  function handoffHtml() {
    return '<section class="fm-card fm-handoff">' +
      '<h3 class="fm-h">Independent medication-safety view</h3>' +
      '<div class="fm-notice">Forecasts provide factual patient-state context only. They do not estimate treatment effects ' +
      'and do not automatically change drug ranking or dosing.</div>' +
      '<button class="fm-btn solid" data-fm="open-rx">Open independent Rx &amp; Safety view</button>' +
      '<div class="fm-note">The Rx view runs the symbolic pharmacology engine on its own case input. ' +
      'Nothing from this forecast is written into it, no drug is preselected, and no ranking is altered.</div>' +
      '</section>';
  }

  // ── replay controls ────────────────────────────────────────────────────
  function replayHtml(c) {
    var total = lastStep(c) + 1;
    var busy = !!S.pending[cacheKey(c, S.step)];
    var pct = total > 1 ? (S.step / (total - 1)) * 100 : 100;
    var anchor = hourAt(c, S.step);
    return '<section class="fm-card tight">' +
      '<h3 class="fm-h">Replay control · <span class="fm-step-no">step ' + (S.step + 1) + ' / ' + total + '</span></h3>' +
      '<div class="fm-clock">' + esc(fmtHour(anchor)) + '<small>anchor — hours after ICU admission</small></div>' +
      '<div class="fm-track"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
      '<div class="fm-ticks"><span>0 h</span><span>' + esc(fmtHour(hourAt(c, lastStep(c)))) + '</span></div>' +
      '<div class="fm-transport" style="margin-top:11px">' +
        '<button class="fm-btn" data-fm="reset" title="Reset">⏮ Reset</button>' +
        '<button class="fm-btn" data-fm="prev"' + (S.step <= 0 ? ' disabled' : '') + ' title="Previous observation">◀ Prev</button>' +
        (S.playing
          ? '<button class="fm-btn solid" data-fm="pause">⏸ Pause</button>'
          : '<button class="fm-btn solid" data-fm="play"' + (S.step >= lastStep(c) ? ' disabled' : '') + '>▶ Play</button>') +
        '<button class="fm-btn" data-fm="next"' + (S.step >= lastStep(c) ? ' disabled' : '') + ' title="Next observation">Next ▶</button>' +
      '</div>' +
      '<div class="fm-row" style="margin-top:9px">' +
        '<label class="fm-note" style="margin:0">Speed</label>' +
        '<select class="fm-select" data-fm="speed">' + SPEEDS.map(function (s, i) {
          return '<option value="' + i + '"' + (i === S.speedIndex ? ' selected' : '') + '>' + esc(s.label) + '</option>';
        }).join('') + '</select>' +
        (busy ? '<span class="fm-note" style="margin:0"><span class="fm-loading"></span> requesting forecast…</span>' : '') +
      '</div>' +
      '<div class="fm-note">Each step posts only the trajectory prefix up to and including the current observation, ' +
      'with <span class="fm-mono">anchor_hour</span> set to that observation’s time. Later observations are held in the browser ' +
      'for playback and are never sent to the model.</div>' +
      '</section>';
  }

  function currentObservationsHtml(c, resp) {
    var row = trajectoryOf(c)[S.step] || {};
    var series = PRIMARY_SERIES.concat(EXTRA_SERIES);
    var items = series.map(function (t) {
      var v = num(row[t]);
      var age = num(row[t + '_age_hr']);
      if (v == null) {
        return '<div class="fm-obs-item na"><div class="n">' + esc(labelFor(t)) + '</div>' +
          '<div class="v">Not observed</div></div>';
      }
      return '<div class="fm-obs-item"><div class="n">' + esc(labelFor(t)) + '</div>' +
        '<div class="v">' + esc(fmtVal(v, t)) + '<span class="u">' + esc(unitFor(t)) + '</span></div>' +
        (age != null ? '<div class="age">carried forward ' + esc(fmtHour(age)) + '</div>' : '') + '</div>';
    }).join('');
    return '<section class="fm-card tight">' +
      '<h3 class="fm-h">Current observation · ' + esc(fmtHour(hourAt(c, S.step))) + '</h3>' +
      '<div class="fm-obs">' + items + '</div>' +
      '<div class="fm-note">Values marked <i>Not observed</i> have no measurement in the source data at or before this time point. ' +
      'Nothing is imputed. “Carried forward” is the age of the most recent backward-looking measurement, as recorded by the eICU adapter.' +
      (resp ? ' Prefix rows sent to the model at this step: <b>' + esc(resp._prefix_rows) + '</b>.' : '') +
      '</div></section>';
  }

  // ── view renderers ─────────────────────────────────────────────────────
  function renderMonitoring() {
    var root = el('view-monitoring');
    if (!root) return;
    if (!S.booted && !S.casesError) { root.innerHTML = loadingHtml(); return; }
    var c = activeCase();
    if (S.casesError || !c) {
      root.innerHTML = errorCardHtml('Monitoring cases unavailable', S.casesError || 'No case returned by /api/monitoring/cases');
      return;
    }
    var head = provenanceHtml(c, S.step) + cachedBannerHtml();
    if (!healthy() && !S.fixture) { root.innerHTML = head + serviceUnavailableHtml(); return; }
    var resp = responseFor(c, S.step);
    var body;
    if (!resp && S.stepError) {
      body = errorCardHtml('Forecast request failed at step ' + (S.step + 1), S.stepError);
    } else if (!resp) {
      body = '<section class="fm-card"><div class="fm-empty"><span class="fm-loading"></span> ' +
        'Requesting the forecast for the trajectory prefix…</div></section>';
    } else {
      body = '';
    }
    var shown = resp ? filterForecasts(resp, S.targetFilter, S.horizonFilter, S.showAllTiers) : [];
    root.innerHTML = head +
      '<div class="fm-dash">' +
        '<div class="fm-col">' + replayHtml(c) + currentObservationsHtml(c, resp) + '</div>' +
        '<div class="fm-col">' + body + (resp ? chartsHtml(c, resp, S.step) : '') + '</div>' +
        '<div class="fm-col fm-col-right">' +
          (resp ? signalsHtml(c, resp, S.step) : '') +
          (resp ? '<section class="fm-card tight"><h3 class="fm-h">Forecast at this anchor</h3>' +
            filterBarHtml(resp, 'mon', S.targetFilter, S.horizonFilter, S.showAllTiers, shown.length) +
            (shown.length ? shown.map(function (f) { return forecastCardHtml(f, c, resp, S.step); }).join('')
              : '<div class="fm-empty">No cell matches this filter.</div>') +
            '</section>' : '') +
        '</div>' +
      '</div>' +
      validationHtml(resp) + historyHtml(c, S.step) + handoffHtml();
  }

  function renderForecast() {
    var root = el('view-forecast');
    if (!root) return;
    if (!S.booted && !S.casesError) { root.innerHTML = loadingHtml(); return; }
    var c = activeCase();
    if (S.casesError || !c) {
      root.innerHTML = errorCardHtml('Case source unavailable', S.casesError || 'No case returned by /api/monitoring/cases');
      return;
    }
    var step = lastStep(c);
    var head = provenanceHtml(c, step) + cachedBannerHtml();
    if (!healthy() && !S.fixture) { root.innerHTML = head + serviceUnavailableHtml(); return; }
    var resp = responseFor(c, step);
    if (S.stepError && !resp) {
      root.innerHTML = head + errorCardHtml('Forecast request failed', S.stepError);
      return;
    }
    if (!resp) {
      root.innerHTML = head + '<section class="fm-card"><div class="fm-empty"><span class="fm-loading"></span> ' +
        'Requesting the forecast for the full case trajectory…</div></section>';
      return;
    }
    var shown = filterForecasts(resp, S.fcTargetFilter, S.fcHorizonFilter, S.fcShowAllTiers);
    root.innerHTML = head +
      chartsHtml(c, resp, step) +
      '<section class="fm-card">' +
        '<h3 class="fm-h">Patient-individualised forecast at ' + esc(fmtHour(num(resp._anchor_hour))) + '</h3>' +
        '<div class="fm-sub">Population point, personalized point and — for serialized validated cells only — a ' +
        'target-specific calibrated research interval. Factual patient-state output: no diagnosis, no treatment effect, no recommendation.</div>' +
        filterBarHtml(resp, 'fc', S.fcTargetFilter, S.fcHorizonFilter, S.fcShowAllTiers, shown.length) +
        '<div class="fm-charts" style="grid-template-columns:1fr 1fr">' +
          (shown.length
            ? shown.map(function (f) { return forecastCardHtml(f, c, resp, step); }).join('')
            : '<div class="fm-empty">No cell matches this filter.</div>') +
        '</div>' +
      '</section>' +
      validationHtml(resp) + handoffHtml();
  }

  function loadingHtml() {
    return '<section class="fm-card"><div class="fm-empty"><span class="fm-loading"></span> ' +
      'Loading case source, model health and validation evidence…</div></section>';
  }

  function renderAll() {
    var v = document.body.getAttribute('data-view');
    if (v === 'monitoring') renderMonitoring();
    else if (v === 'forecast') renderForecast();
  }

  // ── replay engine ──────────────────────────────────────────────────────
  async function gotoStep(step, opts) {
    var c = activeCase();
    if (!c) return;
    var target = Math.max(0, Math.min(step, lastStep(c)));
    S.step = target;
    S.stepError = null;
    renderAll();
    if (!healthy()) return;
    if (S.fixture && responseFor(c, target)) { renderAll(); return; }
    try {
      await fetchStep(c, target);
      S.stepError = null;
    } catch (e) {
      S.stepError = e.message || String(e);
      stopPlay();
    }
    if (S.step === target || (opts && opts.force)) renderAll();
  }

  function stopPlay() {
    S.playing = false;
    S.playToken++;
  }

  async function startPlay() {
    var c = activeCase();
    if (!c || S.playing || !healthy()) return;
    S.playing = true;
    var token = ++S.playToken;
    renderAll();
    while (S.playing && token === S.playToken) {
      if (S.step >= lastStep(c)) break;
      await gotoStep(S.step + 1);
      if (!S.playing || token !== S.playToken || S.stepError) break;
      await sleep(SPEEDS[S.speedIndex].ms);
    }
    if (token === S.playToken) { S.playing = false; renderAll(); }
  }

  // ── cached fixtures (explicit action only) ─────────────────────────────
  async function loadFixture() {
    S.fixtureNotice = null;
    try {
      var j = await getJSON('/api/forecast/fixtures');
      var map = {};
      Object.keys(j).filter(function (key) {
        return key.indexOf('post_forecast_') === 0;
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

  // ── view switching + events ────────────────────────────────────────────
  function setView(name) {
    document.body.setAttribute('data-view', name);
    Array.prototype.forEach.call(document.querySelectorAll('.nav-view'), function (b) {
      b.classList.toggle('active', b.getAttribute('data-view') === name);
    });
    if (name === 'rx') { stopPlay(); return; }
    renderAll();
    bootstrap().then(function () {
      var c = activeCase();
      if (!c || !healthy()) { renderAll(); return; }
      if (name === 'monitoring') return gotoStep(S.step, { force: true });
      if (!responseFor(c, lastStep(c))) {
        return fetchStep(c, lastStep(c)).then(function () { S.stepError = null; renderAll(); },
          function (e) { S.stepError = e.message || String(e); renderAll(); });
      }
      renderAll();
    }, function (e) {
      S.booted = true;
      S.stepError = e && e.message ? e.message : String(e);
      renderAll();
    });
  }

  function onClick(e) {
    var navBtn = e.target.closest ? e.target.closest('.nav-view') : null;
    if (navBtn) { setView(navBtn.getAttribute('data-view')); return; }
    var t = e.target.closest ? e.target.closest('[data-fm]') : null;
    if (!t) return;
    var action = t.getAttribute('data-fm');
    if (action === 'play') { startPlay(); }
    else if (action === 'pause') { stopPlay(); renderAll(); }
    else if (action === 'next') { stopPlay(); gotoStep(S.step + 1); }
    else if (action === 'prev') { stopPlay(); gotoStep(S.step - 1); }
    else if (action === 'reset') { stopPlay(); gotoStep(0); }
    else if (action === 'open-rx') { setView('rx'); }
    else if (action === 'fixture-load') { loadFixture(); }
    else if (action === 'fixture-clear') { S.fixture = null; renderAll(); }
  }

  function onChange(e) {
    var t = e.target.closest ? e.target.closest('[data-fm]') : null;
    if (!t) return;
    var action = t.getAttribute('data-fm');
    if (action === 'speed') { S.speedIndex = Number(t.value) || 0; renderAll(); }
    else if (action === 'toggle-extra') { S.showExtraSeries = !!t.checked; renderAll(); }
    else if (action === 'mon-target') { S.targetFilter = t.value; renderAll(); }
    else if (action === 'mon-horizon') { S.horizonFilter = t.value; renderAll(); }
    else if (action === 'mon-alltiers') { S.showAllTiers = !!t.checked; renderAll(); }
    else if (action === 'fc-target') { S.fcTargetFilter = t.value; renderAll(); }
    else if (action === 'fc-horizon') { S.fcHorizonFilter = t.value; renderAll(); }
    else if (action === 'fc-alltiers') { S.fcShowAllTiers = !!t.checked; renderAll(); }
  }

  document.addEventListener('click', onClick);
  document.addEventListener('change', onChange);
  if (!document.body.getAttribute('data-view')) document.body.setAttribute('data-view', 'rx');

  window.OslerForecastViews = { setView: setView, state: S };
})();
