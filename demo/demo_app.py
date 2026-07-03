"""
demo_app.py — Drug-recommendation agent demo (for clinicians).

  import a case (free text or form)  ─▶  agent parses + symbolic engine recommends
  ─▶ mind-map reasoning graph (case ▸ disease ▸ targets ▸ drugs)  ─▶ chat to ask why.

The symbolic engine (reasoning_engine) makes every recommendation; the LLM only
parses the case and explains the result. Knowledge sources: drugs_pkpd.json (drug
deltas), the organ JSONs via disease_world (local disease "wiki"), and optional
live openFDA labels via agent.enrich_openfda.

Run:  py -m pip install -r requirements.txt
      py demo_app.py   →  http://127.0.0.1:5000
Chat/agent-parse use an API key entered in the top-right box (or OPENAI_API_KEY env).
Without a key it still runs: rule-based parsing + reasoning graph (chat degraded).
"""
from __future__ import annotations
import sys
import os
import json
import urllib.parse
from pathlib import Path

# This demo (demo/) reuses the pharmacology engine in engine/ and data in data/.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "demo", _ROOT / "engine", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines → os.environ. Lets you keep
    Photon sandbox creds in demo/.env instead of re-exporting them in every shell. Runs
    before photon_client is imported (it reads os.environ at import time). An already-set
    real env var wins over the file (setdefault)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).parent / ".env")

from flask import Flask, request, jsonify, Response, redirect

import reasoning_engine as RE
import case_targets
import case_parser
import agent
import llm_client
import prescription
import photon_client
import photon_oauth
import transcribe

_HERE = Path(__file__).parent
app = Flask(__name__)
# 8 MB default: voice case-note clips (m4a/AAC ~1 MB/min) would 413 at the old
# 1 MB cap. JSON routes stay tiny regardless.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_REQUEST_BYTES", "8388608"))


def _allowed_origins() -> set[str]:
    raw = os.environ.get("CORS_ORIGINS", "")
    return {origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()}


@app.after_request
def add_public_headers(response):
    """Allow an optional separately hosted Expo web build to call this API."""
    origin = (request.headers.get("Origin") or "").rstrip("/")
    allowed = _allowed_origins()
    if origin and ("*" in allowed or origin in allowed):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.route("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "service": "oslian-demo",
        "photon": photon_client.env_label() if photon_client.is_enabled() else "mock",
    })

DRUGS_PKPD = RE._load("drugs_pkpd.json")["drugs"]


def _load_optional(name: str) -> dict:
    p = _HERE / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace")).get("drugs", {})
    except Exception:
        return {}


CLINICAL = _load_optional("demo_clinical_data.json")
SAMPLE_CASES = json.loads((_HERE / "sample_cases.json").read_text(encoding="utf-8"))["cases"]

# patient_id -> last analyze bundle, used to ground chat.
_CACHE: dict[str, dict] = {}


@app.route("/")
def index():
    # The Photon OAuth redirect lands here (the redirect_uri is the bare whitelisted origin
    # http://127.0.0.1:5000). When Auth0 appends ?code&state, finish the login; otherwise
    # serve the normal demo page.
    if request.args.get("code") and request.args.get("state"):
        return _handle_oauth_callback()
    return Response((_HERE / "case_demo.html").read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/api/cases")
def api_cases():
    return jsonify({"cases": SAMPLE_CASES, "indications": case_targets.list_indications(),
                    "env_llm": llm_client.available()})


@app.route("/api/voice/transcribe", methods=["POST"])
def api_voice_transcribe():
    """Voice case-note dictation: multipart audio → transcript text. This only turns
    speech into text; the clinician reviews/edits it before Analyze, and nothing is
    analyzed or prescribed from voice here. Provider lives in demo/transcribe.py."""
    if not transcribe.transcribe_available():
        return jsonify({"error": "Voice transcription is not configured "
                                 "(set GEMINI_API_KEY in demo/.env)."}), 503
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "no audio (send multipart field 'audio')"}), 400
    fmt = (request.form.get("format") or "").strip().lstrip(".") or "m4a"
    audio_bytes = f.read()
    if not audio_bytes:
        return jsonify({"error": "empty audio"}), 400
    try:
        text = transcribe.transcribe_audio(audio_bytes, fmt)
    except Exception as e:  # noqa: BLE001 — normalize provider errors to one shape
        return jsonify({"error": f"transcription failed: {type(e).__name__}: {e}"}), 502
    return jsonify({"text": text})


def _indication_label(canon: str) -> str:
    """Human label for a canonical indication key (same lookup agent.py uses)."""
    return next((i["label"] for i in case_targets.list_indications()
                 if i["value"] == canon), canon)


@app.route("/api/parse_case", methods=["POST"])
def api_parse_case():
    """Structure a dictated / free-text note into fields the clinician CONFIRMS before
    Analyze — the second half of voice case-note entry (transcribe → structure → confirm).

    This is a *preview* of the exact parse the Analyze pipeline runs server-side: it reuses
    the same case_parser (with the same provider) and the same PatientProfile.flags() the
    engine acts on, so the captured chips and amber vitals flags shown for confirmation
    match what Analyze will see. Nothing is analyzed, ranked, or prescribed here."""
    d = request.get_json(force=True) or {}
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text (send {\"text\": ...})"}), 400
    # Pass the caller's key/provider straight through, identical to /api/analyze, so the
    # preview and the real analyze parse the note the same way (LLM if a key is available,
    # else deterministic rules). Set LLM_PROVIDER=gemini in demo/.env to use the same
    # Gemini key as transcription for both.
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    fields = case_parser.parse(text, api_key, provider)
    parser_used = fields.pop("_parser", "rules")
    patient = agent.build_patient(fields)
    indication = fields.get("indication") or ""
    canon = case_targets.resolve(indication)
    return jsonify({
        "fields": fields,
        "parser": parser_used,
        "indication": indication,
        "indication_label": _indication_label(canon) if canon else indication,
        "indication_known": bool(canon),
        # Same amber safety flags the engine raises (hypotension, tachycardia, …).
        "flags": patient.flags(),
        "missing_core": patient.missing_core(),
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    d = request.get_json(force=True) or {}
    out = agent.analyze(d.get("fields") or {}, DRUGS_PKPD, dict(CLINICAL),
                        text=d.get("text") or "", api_key=(d.get("api_key") or "").strip() or None,
                        provider=d.get("provider") or None, use_openfda=bool(d.get("use_openfda")))
    if "error" in out:
        return jsonify(out), 400
    pid = d.get("patient_id") or out["indication"]
    out["patient_id"] = pid
    _CACHE[pid] = out
    return jsonify(out)


@app.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    """NDJSON stream: one JSON object per line, one per workflow step, ending with
    {type:final, bundle}. Lets the UI render the agent's steps live (Claude-Code style)."""
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id") or "p"
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    # deterministic fixed pipeline, or LLM tool-calling orchestration
    if d.get("orchestrator") == "llm":
        import orchestrator
        stream_fn = orchestrator.analyze_stream_llm
    else:
        stream_fn = agent.analyze_stream

    def gen():
        trace = []
        try:
            for ev in stream_fn(
                    d.get("fields") or {}, DRUGS_PKPD, dict(CLINICAL),
                    text=d.get("text") or "", api_key=api_key,
                    provider=provider, use_openfda=bool(d.get("use_openfda"))):
                if ev["type"] == "step" and not ev.get("running"):
                    trace.append({"icon": ev["icon"], "title": ev["title"], "detail": ev.get("detail", "")})
                if ev["type"] == "final":
                    ev["bundle"]["trace"] = trace
                    ev["bundle"]["patient_id"] = pid
                    _CACHE[pid] = ev["bundle"]
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # never break the stream silently
            yield json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson")


_SYSTEM = (
    "You are the explanation assistant for a clinician-facing drug-recommendation system. "
    "Your ONLY job is to explain, in clear clinical language, the recommendation that the "
    "symbolic reasoning engine has ALREADY computed (provided below).\n"
    "Rules:\n"
    "1) Do not make independent medical decisions, change the ranking, or suggest drugs not "
    "in the candidate list.\n"
    "2) Explain only from the provided data: disease perturbations (the local world model), "
    "mechanism_chain, matched_targets, clinical_role, safety, final_answer.\n"
    "3) Never invent a dose; only cite dose.verbatim when present, and note a clinician must "
    "confirm it.\n"
    "4) If asked about a drug not in the list, say it was not matched and why "
    "(mechanism mismatch / not in the drug knowledge base).\n"
    "5) delta magnitudes are estimates; direction is more reliable. Decision support only — "
    "a licensed clinician makes the final call.\n"
    "Answer concisely, in the user's language, citing specific drugs and target variables."
)


def _ground(bundle: dict) -> str:
    res = bundle["result"]
    dm = bundle.get("disease_model", {})
    p = res["patient"]
    lines = [f"Indication: {res.get('indication_label', res.get('indication'))}",
             f"Treatment targets: {', '.join(res.get('target_states', []))}",
             f"Disease world-model ({dm.get('source')}): {dm.get('disease')}",
             "  perturbations: " + ", ".join(
                 f"{pp['variable']} {'↑' if pp['direction'] == 'high' else '↓'}"
                 for pp in dm.get("perturbations", [])),
             "  symptoms: " + ", ".join(dm.get("symptoms", [])),
             f"Patient: age={p.get('age')}, renal={p.get('renal_label')}, "
             f"allergies={p.get('allergies')}, meds={p.get('meds')}, flags={p.get('flags')}",
             f"Mode: {'mechanism-only' if res.get('mechanism_only') else 'mechanism + label safety gate'}",
             "", "Ranked candidates:"]
    for i, c in enumerate(res.get("candidates", []), 1):
        dose = c.get("dose", {})
        dose_txt = dose.get("verbatim") if dose.get("patient_specific_allowed") else None
        reasons = "; ".join(r.get("message", "") for r in c.get("safety", {}).get("reasons", []))
        lines.append(
            f"{i}. {c['drug']} [{c['clinical_role'].get('label')}] score={c.get('mechanism_score')}\n"
            f"   mechanism: {c.get('mechanism_chain')}\n"
            f"   matched: {[m['target'] + '/' + m['effect_type'] for m in c.get('matched_targets', [])]}\n"
            f"   safety: {c['safety'].get('decision')} ({reasons})\n"
            f"   dose: {dose_txt or 'not shown'}\n"
            f"   conclusion: {c.get('final_answer')}")
    return "\n".join(lines)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id")
    messages = d.get("messages") or []
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    # Prefer grounding sent by the client (survives server restarts); fall back to cache.
    bundle = None
    if d.get("result"):
        bundle = {"result": d["result"], "disease_model": d.get("disease_model") or {}}
    elif pid in _CACHE:
        bundle = _CACHE[pid]
    if not bundle:
        return jsonify({"reply": "Please analyze a case first so I can explain its result.",
                        "ok": False})
    system = _SYSTEM + "\n\n=== CURRENT CASE RESULT ===\n" + _ground(bundle)
    ok, text = llm_client.chat(system, messages, api_key=api_key, provider=provider)
    return jsonify({"reply": text, "ok": ok})


# ── e-prescribing (Photon sandbox / mock) ─────────────────────────────────────
# The symbolic engine recommended the drug; here a clinician reviews, edits the sig,
# picks a pharmacy and SIGNS — in Photon's certified workflow, or a local mock when no
# sandbox credentials are set. Prescription state lives in prescription._RX (in-memory,
# like _CACHE): a demo session starts fresh.


@app.route("/api/prescribe/start", methods=["POST"])
def api_prescribe_start():
    """Begin an Rx for {patient_id, drug} from the cached analysis bundle. Returns the
    embed path the mobile WebView opens to review/sign. Safety gate may 400."""
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id")
    bundle = _CACHE.get(pid)
    if not bundle:
        return jsonify({"error": "Analyze this case first (no cached result on server)."}), 400
    try:
        rec = prescription.start(pid, d.get("drug") or "", bundle)
    except prescription.RxError as e:
        return jsonify({"error": e.message}), 400
    resp = {"rx_id": rec["rx_id"], "mode": rec["mode"],
            "embed_path": f"/prescribe-embed?rx_id={rec['rx_id']}"}
    if rec.get("photon_prescribe_url"):
        resp["photon_url"] = rec["photon_prescribe_url"]  # open in the system browser
    return jsonify(resp)


@app.route("/prescribe-embed")
def prescribe_embed():
    """The review/sign page rendered inside the mobile WebView (mock or Photon Elements)."""
    return Response((_HERE / "prescribe_embed.html").read_text(encoding="utf-8"),
                    mimetype="text/html")


@app.route("/api/prescribe/context")
def api_prescribe_context():
    """Everything the embed page needs to render (drug, prefilled sig, mode, Photon config)."""
    try:
        return jsonify(prescription.context(request.args.get("rx_id") or ""))
    except prescription.RxError as e:
        return jsonify({"error": e.message}), 404


@app.route("/api/prescribe/complete", methods=["POST"])
def api_prescribe_complete():
    """Clinician signed & sent: finalize the prescription record."""
    d = request.get_json(force=True) or {}
    try:
        rec = prescription.complete(
            d.get("rx_id") or "",
            sig=d.get("sig") or "",
            dispense_quantity=d.get("dispense_quantity"),
            dispense_unit=d.get("dispense_unit"),
            days_supply=d.get("days_supply"),
            refills=d.get("refills") or 0,
            pharmacy=d.get("pharmacy"),
            photon_prescription_id=d.get("photon_prescription_id"),
            photon_order_id=d.get("photon_order_id"))
    except prescription.RxError as e:
        return jsonify({"error": e.message}), 400
    return jsonify(rec)


@app.route("/api/photon/webhook", methods=["POST"])
def api_photon_webhook():
    """Photon Order/Prescription events advance an Rx's status. Production must verify
    the webhook signature; the sandbox demo accepts and best-effort matches by order id."""
    event = request.get_json(force=True, silent=True) or {}
    updated = prescription.apply_webhook(event)
    return jsonify({"ok": True, "matched": bool(updated)})


@app.route("/api/prescriptions")
def api_prescriptions():
    """List a patient's prescriptions (newest first) for the Prescriptions tab."""
    pid = request.args.get("patient_id") or ""
    return jsonify({"prescriptions": prescription.list_for(pid),
                    "photon": photon_client.is_enabled(),
                    "env": photon_client.env_label()})


# ── provider OAuth (native Sign & Send) ───────────────────────────────────────
# A clinician logs in ONCE so the app can sign prescriptions under their authorized
# identity (the M2M token can't — it lacks write:prescription). The whole dance runs
# here on the backend; the device only ever learns connected:true/false. See photon_oauth.

def _deep_link(url: str, **params) -> str:
    """Append query params to an app deep link (e.g. oslianrx://photon-connected)."""
    parts = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(parts.query))
    q.update(params)
    return urllib.parse.urlunparse(parts._replace(query=urllib.parse.urlencode(q)))


@app.route("/api/photon/oauth/start")
def api_oauth_start():
    """Return the Neutron authorize URL the app opens in the system browser. `return` is
    the app deep link we bounce back to after the callback."""
    if not photon_oauth.is_enabled():
        return jsonify({"error": "Photon SPA client id is not configured."}), 400
    return_url = request.args.get("return") or "oslianrx://photon-connected"
    try:
        return jsonify({"authorize_url": photon_oauth.build_authorize_url(return_url)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


def _handle_oauth_callback():
    """Exchange the authorization code for a provider token, then 302 back into the app via
    its deep link (carrying ?photon=connected|error). Shared by the root route (the redirect
    target — a bare whitelisted origin) and the explicit /api/photon/oauth/callback path."""
    state = request.args.get("state") or ""
    pend = photon_oauth.consume_state(state)
    if not pend:
        # No matching state → can't safely bounce to an app; show a plain page.
        return Response("<h3>Photon login: unknown or expired session. Please retry "
                        "from the app.</h3>", mimetype="text/html", status=400)
    return_url = pend["return_url"]
    err = request.args.get("error")
    if err:
        desc = request.args.get("error_description") or err
        return redirect(_deep_link(return_url, photon="error", msg=desc))
    code = request.args.get("code") or ""
    try:
        photon_oauth.exchange_code(code, pend["verifier"])
    except Exception as e:  # noqa: BLE001
        print(f"[oauth] token exchange FAILED: {e}", flush=True)
        return redirect(_deep_link(return_url, photon="error", msg=str(e)))
    st = photon_oauth.status()
    print(f"[oauth] CONNECTED provider={st.get('provider')} "
          f"can_prescribe={st.get('can_prescribe')} scopes={st.get('scopes')!r}", flush=True)
    return redirect(_deep_link(return_url, photon="connected"))


@app.route("/api/photon/oauth/callback")
def api_oauth_callback():
    """Neutron redirects here after login (when the full path is whitelisted)."""
    return _handle_oauth_callback()


@app.route("/api/photon/oauth/status")
def api_oauth_status():
    """Is a provider connected (for the Settings UI)?"""
    return jsonify(photon_oauth.status())


@app.route("/api/photon/oauth/disconnect", methods=["POST"])
def api_oauth_disconnect():
    photon_oauth.disconnect()
    return jsonify({"ok": True})


# ── native prescribe (options + Sign & Send) ──────────────────────────────────
@app.route("/api/prescribe/options")
def api_prescribe_options():
    """Catalog choices + prefilled defaults for the native review screen."""
    try:
        return jsonify(prescription.options(request.args.get("rx_id") or ""))
    except prescription.RxError as e:
        return jsonify({"error": e.message}), 404


@app.route("/api/prescribe/sign", methods=["POST"])
def api_prescribe_sign():
    """Provider's in-app Sign & Send: createPrescription → createOrder under the provider
    token. 401 if no provider is connected; 502 if Photon rejects the write."""
    token = photon_oauth.provider_token()
    if not token:
        return jsonify({"error": "Connect Photon in Settings first (no provider signed in)."}), 401
    d = request.get_json(force=True) or {}
    try:
        rec = prescription.sign(
            d.get("rx_id") or "", provider_token=token,
            treatment_id=d.get("treatment_id") or "",
            sig=d.get("sig") or "",
            dispense_quantity=d.get("dispense_quantity") or 0,
            dispense_unit=d.get("dispense_unit") or "Each",
            days_supply=d.get("days_supply") or 0,
            refills=d.get("refills") or 0)
    except prescription.RxError as e:
        print(f"[sign] gate/input error: {e.message}", flush=True)
        return jsonify({"error": e.message}), 400
    except RuntimeError as e:  # Photon GraphQL/transport error
        print(f"[sign] Photon error: {e}", flush=True)
        return jsonify({"error": str(e)}), 502
    print(f"[sign] OK drug={rec.get('drug')} rx={rec.get('photon_prescription_id')} "
          f"order={rec.get('photon_order_id')}", flush=True)
    return jsonify(rec)


if __name__ == "__main__":
    print(f"[demo] drugs={len(DRUGS_PKPD)} clinical={len(CLINICAL)} cases={len(SAMPLE_CASES)} "
          f"env_llm={'on' if llm_client.available() else 'off'} "
          f"photon={'on (' + photon_client.env_label() + ')' if photon_client.is_enabled() else 'mock'} "
          f"provider_oauth={'ready' if photon_oauth.is_enabled() else 'off'}")
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )
