/**
 * API client for the Osler·Rx demo backend (demo/demo_app.py, Flask, port 5000).
 *
 * Development defaults to the Mac's localhost. Release builds default to the
 * public demo API; EXPO_PUBLIC_API_BASE can override either environment.
 */

// 127.0.0.1 (not "localhost"): macOS AirPlay Receiver squats on *:5000 and iOS
// resolves localhost to ::1 first, which would hit AirPlay instead of Flask.
export const API_BASE =
  process.env.EXPO_PUBLIC_API_BASE ??
  (__DEV__ ? 'http://127.0.0.1:5000' : 'https://demo.oslian.com');

// ── Types mirroring demo/agent.py bundle ─────────────────────────────────

export interface SampleCase {
  id: string;
  title: string;
  indication: string;
  age?: number;
  sex?: string;
  weight_kg?: number;
  egfr?: number;
  hepatic_status?: string;
  allergies?: string[];
  current_medications?: string[];
  conditions?: string[];
  symptoms?: string[];
  vitals?: Record<string, number>;
  labs?: Record<string, number>;
}

export interface Indication {
  value: string;
  label: string;
}

export interface CasesResponse {
  cases: SampleCase[];
  indications: Indication[];
  env_llm: boolean;
}

export interface MatchedTarget {
  target: string;
  effect_type: string;
}

/** Provenance for a drug claim (openFDA/DailyMed SPL record). */
export interface Evidence {
  source?: string; // e.g. "openFDA / DailyMed (FDA SPL)"
  set_id?: string; // DailyMed SPL setid → direct drugInfo link
  retrieved_at?: string;
}

/** FAERS adverse-event signal (FDA Adverse Event Reporting System). */
export interface FaersSignal {
  event: string;
  report_count: number;
  source?: string;
  confidence?: string;
}

export interface DrugCandidate {
  drug: string;
  clinical_role: { label?: string; role?: string };
  mechanism_score?: number;
  mechanism_chain?: string;
  matched_targets?: MatchedTarget[];
  safety: { decision: string; reasons?: { message?: string }[] };
  dose?: { verbatim?: string; patient_specific_allowed?: boolean };
  final_answer?: string;
  rationale?: string;
  evidence?: Evidence[];
  faers_signals?: FaersSignal[];
  indication_support?: string;
  label_status?: string;
}

export interface AnalyzeResult {
  patient: {
    age?: number;
    sex?: string;
    renal_label?: string;
    allergies?: string[];
    meds?: string[];
    flags?: string[];
  };
  indication?: string;
  indication_label?: string;
  target_states?: string[];
  candidates: DrugCandidate[];
  mechanism_only?: boolean;
  _disclaimer?: string;
}

export interface Perturbation {
  variable: string;
  direction: 'high' | 'low';
  cause?: string;
}

export interface DiseaseModel {
  source?: string;
  source_file?: string;
  disease?: string;
  description?: string;
  perturbations: Perturbation[];
  symptoms: string[];
}

export interface GraphNode {
  id: string;
  label: string;
  group: string;
  title?: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
  dashes?: boolean;
  color?: string;
}

export interface TraceStep {
  icon: string;
  title: string;
  detail?: string;
}

export interface Bundle {
  indication: string;
  parser: string;
  fields: Record<string, unknown>;
  result: AnalyzeResult;
  disease_model: DiseaseModel;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  openfda_loaded?: string[] | null;
  trace: TraceStep[];
  patient_id: string;
}

export interface LlmSettings {
  apiKey?: string;
  provider?: 'openai' | 'gemini';
}

// ── Endpoints ────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) {
    throw new Error((j as { error?: string }).error ?? `HTTP ${r.status}`);
  }
  return j as T;
}

export async function getCases(): Promise<CasesResponse> {
  const r = await fetch(`${API_BASE}/api/cases`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as CasesResponse;
}

export interface AnalyzeRequest {
  fields?: Record<string, unknown>;
  text?: string;
  patient_id?: string;
  use_openfda?: boolean;
  llm?: LlmSettings;
}

export async function analyze(req: AnalyzeRequest): Promise<Bundle> {
  return post<Bundle>('/api/analyze', {
    fields: req.fields ?? {},
    text: req.text ?? '',
    patient_id: req.patient_id,
    use_openfda: !!req.use_openfda,
    api_key: req.llm?.apiKey ?? '',
    provider: req.llm?.provider,
  });
}


// ── voice case-note dictation ────────────────────────────────────────────
// Record a clip (expo-audio) → upload here → transcript. The clinician reviews
// the text in the case note before Analyze; voice never analyzes or prescribes.

/** Upload a recorded audio file URI to /api/voice/transcribe → transcript text.
 * Multipart FormData — do NOT set Content-Type; fetch adds the boundary itself. */
export async function transcribeVoice(uri: string, format: 'm4a' = 'm4a'): Promise<string> {
  const form = new FormData();
  // React Native FormData file part: { uri, name, type }.
  form.append('audio', { uri, name: `dictation.${format}`, type: 'audio/m4a' } as unknown as Blob);
  form.append('format', format);
  const r = await fetch(`${API_BASE}/api/voice/transcribe`, { method: 'POST', body: form });
  const j = await r.json();
  if (!r.ok) throw new Error((j as { error?: string }).error ?? `HTTP ${r.status}`);
  return ((j as { text?: string }).text ?? '').trim();
}

// ── structure a dictated / free-text note (voice stage 3B) ────────────────
// After a transcript lands, /api/parse_case previews the SAME parse Analyze runs
// server-side: structured fields for the clinician to confirm + the engine's amber
// vitals flags. Nothing is analyzed or prescribed here — every value is confirmed first.

/** A caution flag PatientProfile.flags() raised (mirrors engine/patient_profile.py). */
export interface CaseFlag {
  flag: string; // e.g. "hypotension", "tachycardia"
  detail: string; // e.g. "SBP 88 < 90"
  keywords?: string[];
}

/** Structured fields extracted from the note (same shape agent.build_patient expects). */
export interface ParsedCaseFields {
  age?: number | null;
  sex?: string | null;
  weight_kg?: number | null;
  egfr?: number | null;
  hepatic_status?: string;
  allergies?: string[];
  current_medications?: string[];
  symptoms?: string[];
  indication?: string;
  vitals?: Record<string, number>;
  labs?: Record<string, number>;
}

export interface ParseCaseResult {
  fields: ParsedCaseFields;
  parser: string; // "rules" | "llm"
  indication: string;
  indication_label: string;
  /** false → not one of the engine's mappable indications; edit before Analyze. */
  indication_known: boolean;
  flags: CaseFlag[];
  missing_core: string[];
}

/** Structure a case note into confirmable fields + the engine's vitals flags.
 * Sends the same api_key/provider as analyze() so the preview matches Analyze. */
export async function parseCase(text: string, llm?: LlmSettings): Promise<ParseCaseResult> {
  return post<ParseCaseResult>('/api/parse_case', {
    text,
    api_key: llm?.apiKey ?? '',
    provider: llm?.provider,
  });
}

// ── e-prescribing (Photon sandbox / mock) ────────────────────────────────
// The symbolic engine recommends a drug; the clinician reviews & signs the actual
// prescription in a WebView (Photon's certified UI, or a local mock). Photon sends it
// to the pharmacy — we never transmit to a pharmacy directly.

export interface PrescribeStartResponse {
  rx_id: string;
  mode: 'mock' | 'photon';
  /** mock mode: open at `${API_BASE}${embed_path}` in a WebView */
  embed_path: string;
  /** photon mode: Photon hosted prescribe page — open in the SYSTEM browser */
  photon_url?: string;
}

export interface Prescription {
  rx_id: string;
  patient_id: string;
  drug: string;
  clinical_role?: string | null;
  safety_decision?: string;
  sig?: string;
  dispense_quantity?: number | null;
  dispense_unit?: string | null;
  days_supply?: number | null;
  refills?: number;
  pharmacy?: { id: string; name: string } | null;
  status: 'draft' | 'sent' | 'filled' | 'error';
  mode?: 'mock' | 'photon';
  est_price?: number | null;
  source_rationale?: string | null;
}

export interface PrescriptionsResponse {
  prescriptions: Prescription[];
  photon: boolean;
  env: string;
}

/** Begin an Rx for a recommended drug. Throws (HTTP 400) if the safety gate blocks it. */
export async function startPrescription(
  patientId: string,
  drug: string,
): Promise<PrescribeStartResponse> {
  return post<PrescribeStartResponse>('/api/prescribe/start', {
    patient_id: patientId,
    drug,
  });
}

export async function getPrescriptions(
  patientId: string,
): Promise<PrescriptionsResponse> {
  const r = await fetch(
    `${API_BASE}/api/prescriptions?patient_id=${encodeURIComponent(patientId)}`,
  );
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as PrescriptionsResponse;
}

/** Mark an Rx submitted — called after the provider returns from Photon's hosted page. */
export async function completePrescription(rxId: string): Promise<Prescription> {
  return post<Prescription>('/api/prescribe/complete', { rx_id: rxId, refills: 0 });
}

// ── provider OAuth (native Sign & Send) ──────────────────────────────────────
// A clinician logs in once (system browser, PKCE) so the backend can sign prescriptions
// under their authorized identity. The whole flow runs on the backend; the device only
// learns connected:true/false and triggers the browser hop.

export interface PhotonStatus {
  connected: boolean;
  /** false when PHOTON_SPA_CLIENT_ID isn't configured — Connect is unavailable */
  enabled: boolean;
  env: string;
  provider: { name?: string | null; email?: string | null } | null;
}

/** Is a provider signed in? */
export async function getPhotonStatus(): Promise<PhotonStatus> {
  const r = await fetch(`${API_BASE}/api/photon/oauth/status`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as PhotonStatus;
}

/** Ask the backend for the Neutron authorize URL to open in the system browser.
 * `returnUrl` is the app deep link Neutron bounces back to after login. */
export async function getPhotonAuthorizeUrl(returnUrl: string): Promise<string> {
  const r = await fetch(
    `${API_BASE}/api/photon/oauth/start?return=${encodeURIComponent(returnUrl)}`,
  );
  const j = (await r.json()) as { authorize_url?: string; error?: string };
  if (!r.ok || !j.authorize_url) throw new Error(j.error ?? `HTTP ${r.status}`);
  return j.authorize_url;
}

export async function disconnectPhoton(): Promise<void> {
  await post('/api/photon/oauth/disconnect', {});
}

// ── native prescribe options + sign ──────────────────────────────────────────

export interface MedOption {
  treatment_id: string;
  name: string;
  strength?: string | null;
  form?: string | null;
}

export interface PrescribeDefaults {
  treatment_id: string | null;
  sig: string;
  dispense_quantity: number;
  dispense_unit: string;
  days_supply: number;
  refills: number;
}

export interface PrescribeOptions {
  rx_id: string;
  drug: string;
  clinical_role?: string | null;
  rationale?: string | null;
  candidates: MedOption[];
  default: PrescribeDefaults;
}

/** Catalog choices + prefilled defaults for the native review screen. */
export async function getPrescribeOptions(rxId: string): Promise<PrescribeOptions> {
  const r = await fetch(
    `${API_BASE}/api/prescribe/options?rx_id=${encodeURIComponent(rxId)}`,
  );
  const j = await r.json();
  if (!r.ok) throw new Error((j as { error?: string }).error ?? `HTTP ${r.status}`);
  return j as PrescribeOptions;
}

export interface SignRequest {
  rx_id: string;
  treatment_id: string;
  sig: string;
  dispense_quantity: number;
  dispense_unit: string;
  days_supply: number;
  refills: number;
}

/** Provider Sign & Send → createPrescription→createOrder on the backend.
 * Throws 401 (connect Photon first) / 502 (Photon rejected) with a message. */
export async function signPrescription(req: SignRequest): Promise<Prescription> {
  return post<Prescription>('/api/prescribe/sign', req);
}

/**
 * v2 (not used yet): live streaming of /api/analyze_stream (NDJSON, one JSON
 * object per line: {type:"step"|"error"|"final"}). Implement with Expo SDK 52+
 * `expo/fetch`, whose Response.body is a ReadableStream:
 *
 *   import { fetch as expoFetch } from 'expo/fetch';
 *   const res = await expoFetch(`${API_BASE}/api/analyze_stream`, {...});
 *   const reader = res.body.getReader();
 *   // accumulate chunks, split on '\n', JSON.parse each complete line.
 *
 * Do NOT add react-native-sse — expo/fetch covers this natively.
 */
