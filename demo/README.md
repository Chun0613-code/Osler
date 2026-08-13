# Osler Research Demo

A research-only demo with two separate paths: a symbolic candidate-ranking
illustration and a factual patient-state forecast API. The forecast API never
changes the candidate ranking.

It reuses the project's existing **symbolic pharmacology engine** (`reasoning_engine.py`
in the project root) unchanged. The candidate ranking, safety gating and graph are
computed by that engine. An LLM is used **only**
to (a) parse messy free-text case notes and (b) explain the result in chat. The LLM never
makes a medical decision.

---

## ⚠️ Accuracy — read this first

This is a **methodology / architecture demo, NOT a clinically validated tool.**

- The drug `delta` values (`drugs_pkpd.json`) and the disease perturbations (organ JSONs)
  are **hand-authored estimates** marked `review_status: unreviewed`.
- `demo_clinical_data.json` is **illustrative, hand-written label text — not real FDA labels.**
- Directions and magnitudes are illustrative. They are not evidence of diagnosis,
  efficacy, treatment effect, or a patient-specific prescription.
- **The AI agent does NOT make it more medically accurate.** Accuracy comes from the
  *knowledge base*, not the LLM. To make it production-grade you must:
  1. use **"fetch live openFDA labels"** only for drug-label and safety evidence;
  2. have clinicians validate/anchor the `delta` knowledge base to guidelines;
  3. align the variable vocabulary between drugs and disease models.

No diagnosis, clinical claim, causal claim, treatment recommendation, or
automated prescribing is authorized.

---

## Factual forecast API

The backend exposes:

- `POST /api/forecast`
- `GET /api/forecast/capabilities`
- `GET /api/forecast/validation-summary`
- `GET /api/health/models`
- `GET /api/monitoring/cases`

Every forecast item has one tier:

- `validated_artifact`: one of 12 exact serialized target×horizon artifacts;
  only these cells can return calibrated `lower` and `upper` values.
- `illustrative`: belief-derived research illustration with no interval.
- `unsupported`: fail-closed abstention with no point or interval.

The interval-width summary is held-out cohort aggregate evidence (120 runs,
approximately 1.1%–34.5%, median approximately 11.8%). It is never presented as
real-time shrinkage for an individual case.

---

## Run

From the project root:

```bat
py -m pip install -r requirements.txt
py demo\demo_app.py
```

Open http://127.0.0.1:5000 .

**Chat / free-text parsing** need an LLM key — paste it in the **top-right box**
(OpenAI by default, or pick Gemini). Without a key the app still runs fully: rule-based
case parsing + the symbolic reasoning graph; only the chat falls back to a notice.

Optional env instead of the box:
```bat
set OPENAI_API_KEY=sk-...
:: or:  set GEMINI_API_KEY=...   &&   set LLM_PROVIDER=gemini
```

---

## How to use

1. **Import a case** (left): click a preset, or paste a free-text note like
   `64M crushing chest pain, acute coronary syndrome. BP 88/54, HR 112, eGFR 72.`
   then **Analyze case**.
2. **Reasoning graph** (right): `patient ▸ disease (real perturbations + symptoms) ▸
   treatment targets ▸ drugs` colored by safety verdict. Click a drug node to jump to its card.
3. **Chat** (center): ask "Why is aspirin first?", "Why is nitroglycerin avoided?",
   "What is the disease doing to the body?". Answers are grounded in the engine output.
4. **Patients** (left): each analyzed case becomes a switchable patient with its own
   graph and chat history.

---

## What is computed vs. what uses AI

| Step | Engine | Uses LLM? |
|---|---|---|
| Drug match, score, ranking, safety gate, dose gating | `reasoning_engine` + `drug_safety_gate` + `clinical_role` (symbolic) | ❌ never |
| Disease world-model (perturbations, symptoms) | `disease_world` reads the organ JSONs | ❌ never |
| Reasoning mind-map | `agent.build_graph` | ❌ never |
| Parse a **free-text** case → fields + indication | `case_parser` | ✅ if key set (else regex rules) |
| "Why this drug?" chat | `llm_client` | ✅ if key set (else fallback notice) |

A preset case sends structured fields, so it triggers no LLM.

## Rebuild the public monitoring case

Download the four public eICU CRD Demo 2.0.1 tables (`patient`, `lab`,
`vitalPeriodic`, and `intakeOutput`) from PhysioNet, then run:

```bat
py demo\eicu_demo_adapter.py --data-dir PATH_TO_TABLES --stay-id 2677807 --anchor-offset-minutes 980
```

The adapter uses backward-only observation lookup and canonical interval-aware
urine-output normalization. Only the small converted JSON is committed; the
source CSV files are not.

---

## Files

| File | Role |
|---|---|
| `demo_app.py` | Flask backend for symbolic and factual forecast routes |
| `forecast_api.py` | Framework-independent forecast HTTP contract |
| `eicu_demo_adapter.py` | Public eICU demo table converter with anchor-time guard |
| `monitoring_cases.json` | Small deidentified public monitoring case |
| `agent.py` | Orchestrator: parse → recommend → disease world-model → build graph; optional openFDA |
| `case_parser.py` | Free-text case → structured fields (LLM or deterministic rules) |
| `case_targets.py` | Indication → physiological treatment targets (15 indications) |
| `disease_world.py` | Local "wiki": indication → real disease perturbations + symptoms from organ JSONs |
| `llm_client.py` | Provider-agnostic LLM (OpenAI / Gemini), per-request key, graceful no-key fallback |
| `case_demo.html` | Single-page UI: patient roster · import · chat · mind-map graph |
| `sample_cases.json` | 5 preset cases |
| `demo_clinical_data.json` | Illustrative (non-clinical) label data so the safety gate + dose path demonstrate |

See `../SYSTEM_FLOW.md` for the current symbolic and JEPA-to-Osler execution paths.
