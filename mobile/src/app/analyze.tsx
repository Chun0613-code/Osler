/**
 * Analyze screen — preset cases, structured case fields, compact notes,
 * openFDA toggle, and a navy agent-trace overlay that plays placeholder
 * steps while /api/analyze runs, then the real trace.
 */
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  analyze,
  API_BASE,
  getCases,
  parseCase,
  type Indication,
  type ParseCaseResult,
  type SampleCase,
  type TraceStep,
} from '@/api/osler';
import CapturedFields, { capturedKeys, type CapturedKey } from '@/components/CapturedFields';
import CasePresetChips from '@/components/CasePresetChips';
import VoiceDictation from '@/components/VoiceDictation';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

const PLACEHOLDER_STEPS: TraceStep[] = [
  { icon: '', title: 'Parsing case…' },
  { icon: '', title: 'Mapping indication → treatment targets…' },
  { icon: '', title: 'Symbolic engine ranking drugs…' },
  { icon: '', title: 'Loading disease world-model…' },
];

const PLACEHOLDER_MS = 900;
const REAL_STEP_MS = 250;
const SEX_OPTIONS = ['M', 'F', 'Other'] as const;

type ExternalPayload = Record<string, unknown>;
type SexOption = '' | (typeof SEX_OPTIONS)[number];
type ManualCaseFields = {
  indication: string;
  age: string;
  sex: SexOption;
  weightKg: string;
  egfr: string;
  allergies: string;
  meds: string;
};

const EMPTY_MANUAL_FIELDS: ManualCaseFields = {
  indication: '',
  age: '',
  sex: '',
  weightKg: '',
  egfr: '',
  allergies: '',
  meds: '',
};

function asObject(value: unknown): ExternalPayload | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as ExternalPayload)
    : null;
}

function firstString(value: unknown): string | undefined {
  if (Array.isArray(value)) return firstString(value[0]);
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function toStringArray(value: unknown): string[] {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === 'string') return item;
        const obj = asObject(item);
        return firstString(obj?.name) ?? firstString(obj?.text) ?? firstString(obj?.display);
      })
      .filter((item): item is string => !!item);
  }
  if (typeof value === 'string') {
    return value.split(/[;,]/).map((item) => item.trim()).filter(Boolean);
  }
  return [];
}

function splitListText(value: string): string[] | undefined {
  const items = value.split(/[;,]/).map((item) => item.trim()).filter(Boolean);
  return items.length ? items : undefined;
}

function mapRecord(value: unknown): Record<string, number> | undefined {
  const obj = asObject(value);
  if (!obj) return undefined;
  const out: Record<string, number> = {};
  Object.entries(obj).forEach(([key, raw]) => {
    const n = toNumber(raw);
    if (n != null) out[key] = n;
  });
  return Object.keys(out).length ? out : undefined;
}

function ageFromBirthDate(value: unknown): number | undefined {
  const text = firstString(value);
  if (!text) return undefined;
  const birth = new Date(text);
  if (Number.isNaN(birth.getTime())) return undefined;
  const now = new Date();
  let age = now.getFullYear() - birth.getFullYear();
  const beforeBirthday =
    now.getMonth() < birth.getMonth() ||
    (now.getMonth() === birth.getMonth() && now.getDate() < birth.getDate());
  if (beforeBirthday) age -= 1;
  return age >= 0 && age < 130 ? age : undefined;
}

function displayFromCode(value: unknown): string | undefined {
  const obj = asObject(value);
  if (!obj) return firstString(value);
  const coding = Array.isArray(obj.coding) ? asObject(obj.coding[0]) : null;
  return firstString(obj.text) ?? firstString(coding?.display) ?? firstString(coding?.code);
}

function normalizeObservationName(name: string): string {
  const key = name.trim().toLowerCase();
  if (key.includes('systolic')) return 'sbp';
  if (key.includes('diastolic')) return 'dbp';
  if (key.includes('heart rate') || key === 'hr') return 'heart_rate';
  if (key.includes('oxygen') || key.includes('spo2') || key.includes('sat')) return 'spo2';
  if (key.includes('temperature') || key === 'temp') return 'temp';
  if (key.includes('potassium')) return 'potassium';
  if (key.includes('creatinine')) return 'creatinine';
  if (key.includes('glucose')) return 'glucose';
  return key.replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
}

function extractFhirBundle(payload: ExternalPayload): Partial<SampleCase> | null {
  if (payload.resourceType !== 'Bundle' || !Array.isArray(payload.entry)) return null;
  const resources = payload.entry
    .map((entry) => asObject(asObject(entry)?.resource))
    .filter((resource): resource is ExternalPayload => !!resource);
  const patient = resources.find((resource) => resource.resourceType === 'Patient');
  const conditions = resources
    .filter((resource) => resource.resourceType === 'Condition')
    .map((resource) => displayFromCode(resource.code))
    .filter((item): item is string => !!item);
  const allergies = resources
    .filter((resource) => resource.resourceType === 'AllergyIntolerance')
    .map((resource) => displayFromCode(resource.code))
    .filter((item): item is string => !!item);
  const meds = resources
    .filter((resource) => resource.resourceType === 'MedicationStatement' || resource.resourceType === 'MedicationRequest')
    .map((resource) => displayFromCode(resource.medicationCodeableConcept))
    .filter((item): item is string => !!item);
  const vitals: Record<string, number> = {};
  const labs: Record<string, number> = {};

  resources
    .filter((resource) => resource.resourceType === 'Observation')
    .forEach((resource) => {
      const name = displayFromCode(resource.code);
      const value = toNumber(asObject(resource.valueQuantity)?.value ?? resource.valueInteger ?? resource.valueDecimal);
      if (!name || value == null) return;
      const category = JSON.stringify(resource.category ?? '').toLowerCase();
      const bucket = category.includes('laboratory') ? labs : vitals;
      bucket[normalizeObservationName(name)] = value;
    });

  const patientObj = asObject(patient);
  const name = Array.isArray(patientObj?.name) ? asObject(patientObj.name[0]) : null;
  const given = Array.isArray(name?.given) ? firstString(name.given[0]) : undefined;
  const family = firstString(name?.family);
  const titleName = [given, family].filter(Boolean).join(' ');

  return {
    id: firstString(patientObj?.id) ?? firstString(payload.id),
    title: titleName || undefined,
    age: ageFromBirthDate(patientObj?.birthDate),
    sex: firstString(patientObj?.gender)?.slice(0, 1).toUpperCase(),
    indication: conditions[0],
    conditions,
    allergies,
    current_medications: meds,
    vitals: Object.keys(vitals).length ? vitals : undefined,
    labs: Object.keys(labs).length ? labs : undefined,
  };
}

function decodePayload(raw: string): ExternalPayload {
  const candidates = [raw, decodeURIComponent(raw)];
  if (typeof globalThis.atob === 'function') {
    const padded = raw.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(raw.length / 4) * 4, '=');
    try {
      candidates.push(globalThis.atob(padded));
    } catch {
      // ignore non-base64 payloads
    }
  }
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      const obj = asObject(parsed);
      if (obj) return obj;
    } catch {
      // keep trying candidate formats
    }
  }
  throw new Error('Patient payload must be JSON.');
}

function normalizePatientPayload(payload: ExternalPayload): SampleCase {
  const fhir = extractFhirBundle(payload);
  const root = asObject(payload.case) ?? asObject(payload.patientData) ?? payload;
  const patient = asObject(root.patient) ?? root;
  const fhirOrEmpty = fhir ?? {};
  const id =
    firstString(root.id) ??
    firstString(patient.id) ??
    firstString(patient.mrn) ??
    firstString(patient.external_id) ??
    fhirOrEmpty.id ??
    `external-${Date.now()}`;
  const age =
    toNumber(root.age) ??
    toNumber(patient.age) ??
    ageFromBirthDate(root.birthDate ?? patient.birthDate ?? patient.dob) ??
    fhirOrEmpty.age;
  const sex =
    firstString(root.sex) ??
    firstString(patient.sex) ??
    firstString(root.gender) ??
    firstString(patient.gender) ??
    fhirOrEmpty.sex;
  const indication =
    firstString(root.indication) ??
    firstString(root.diagnosis) ??
    firstString(root.chief_complaint) ??
    firstString(root.reason) ??
    firstString(root.problem) ??
    fhirOrEmpty.indication ??
    '';
  const fallbackTitle =
    [age != null ? `${age}${sex ? sex.slice(0, 1).toUpperCase() : ''}` : null, indication]
      .filter(Boolean)
      .join(' · ') || 'Imported patient';
  const title =
    firstString(root.title) ||
    firstString(patient.title) ||
    firstString(patient.name) ||
    fhirOrEmpty.title ||
    fallbackTitle;
  const renal = asObject(root.renal);

  return {
    id,
    title,
    indication,
    age,
    sex,
    weight_kg: toNumber(root.weight_kg ?? root.weightKg ?? patient.weight_kg ?? patient.weightKg),
    egfr: toNumber(root.egfr ?? root.eGFR ?? renal?.egfr ?? patient.egfr),
    hepatic_status: firstString(root.hepatic_status ?? root.hepaticStatus ?? patient.hepatic_status),
    allergies: toStringArray(root.allergies ?? patient.allergies).concat(fhirOrEmpty.allergies ?? []),
    current_medications: toStringArray(root.current_medications ?? root.medications ?? root.meds ?? patient.medications).concat(
      fhirOrEmpty.current_medications ?? [],
    ),
    conditions: toStringArray(root.conditions ?? root.problems ?? patient.conditions).concat(fhirOrEmpty.conditions ?? []),
    symptoms: toStringArray(root.symptoms ?? root.complaints ?? patient.symptoms),
    vitals: mapRecord(root.vitals ?? root.vitalSigns ?? patient.vitals) ?? fhirOrEmpty.vitals,
    labs: mapRecord(root.labs ?? root.laboratory ?? patient.labs) ?? fhirOrEmpty.labs,
  };
}

function caseToManualFields(c: SampleCase): ManualCaseFields {
  const sex = c.sex?.trim();
  const normalizedSex: SexOption =
    sex === 'M' || sex === 'F' || sex === 'Other' ? sex : sex ? 'Other' : '';

  return {
    indication: c.indication ?? '',
    age: c.age != null ? String(c.age) : '',
    sex: normalizedSex,
    weightKg: c.weight_kg != null ? String(c.weight_kg) : '',
    egfr: c.egfr != null ? String(c.egfr) : '',
    allergies: c.allergies?.join(', ') ?? '',
    meds: c.current_medications?.join(', ') ?? '',
  };
}

function buildManualCaseFields(fields: ManualCaseFields): Partial<SampleCase> {
  const out: Partial<SampleCase> = {};
  const indication = fields.indication.trim();
  if (indication) out.indication = indication;
  const age = toNumber(fields.age);
  if (age != null) out.age = age;
  if (fields.sex) out.sex = fields.sex;
  const weight = toNumber(fields.weightKg);
  if (weight != null) out.weight_kg = weight;
  const egfr = toNumber(fields.egfr);
  if (egfr != null) out.egfr = egfr;
  const allergies = splitListText(fields.allergies);
  if (allergies) out.allergies = allergies;
  const meds = splitListText(fields.meds);
  if (meds) out.current_medications = meds;
  return out;
}

function titleFromManualFields(fields: Partial<SampleCase>): string | null {
  const ageSex =
    fields.age != null || fields.sex
      ? [fields.age != null ? String(fields.age) : null, fields.sex].filter(Boolean).join('')
      : null;
  const title = [ageSex, fields.indication].filter(Boolean).join(' · ');
  return title || null;
}

// Team policy (2026-07-02, agreed with Chun): the clinical fields needed for correct dosing
// must be filled before Analyze — even ones the symbolic engine treats as optional:
// indication, age, sex, weight, eGFR (precise dose + 未成年/minor + sex distinctions).
// Allergies and current meds are NOT required — an empty field just means "none" (per Alphonse).
const REQUIRED_FIELDS: { key: keyof SampleCase; label: string }[] = [
  { key: 'indication', label: 'Indication' },
  { key: 'age', label: 'Age' },
  { key: 'sex', label: 'Sex' },
  { key: 'weight_kg', label: 'Weight' },
  { key: 'egfr', label: 'eGFR' },
];

function missingRequiredFields(fields: Record<string, unknown>): string[] {
  const filled = (v: unknown) =>
    v != null &&
    !(typeof v === 'string' && !v.trim()) &&
    !(Array.isArray(v) && v.length === 0);
  return REQUIRED_FIELDS.filter((f) => !filled(fields[f.key])).map((f) => f.label);
}

function summarizeCase(c: SampleCase): string {
  const parts: string[] = [];
  if (c.age != null || c.sex) parts.push([c.age, c.sex].filter((x) => x != null).join(''));
  parts.push(`Indication: ${c.indication}`);
  if (c.weight_kg != null) parts.push(`${c.weight_kg} kg`);
  if (c.egfr != null) parts.push(`eGFR ${c.egfr}`);
  if (c.hepatic_status) parts.push(`Hepatic: ${c.hepatic_status}`);
  if (c.allergies?.length) parts.push(`Allergies: ${c.allergies.join(', ')}`);
  if (c.current_medications?.length) parts.push(`Meds: ${c.current_medications.join(', ')}`);
  if (c.conditions?.length) parts.push(`Conditions: ${c.conditions.join(', ')}`);
  if (c.symptoms?.length) parts.push(`Symptoms: ${c.symptoms.join(', ')}`);
  if (c.vitals && Object.keys(c.vitals).length)
    parts.push(
      `Vitals: ${Object.entries(c.vitals).map(([k, v]) => `${k} ${v}`).join(', ')}`,
    );
  if (c.labs && Object.keys(c.labs).length)
    parts.push(
      `Labs: ${Object.entries(c.labs).map(([k, v]) => `${k} ${v}`).join(', ')}`,
    );
  return parts.join(' · ');
}

export default function AnalyzeScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ payload?: string; patient?: string; auto?: string }>();
  const { patients, addPatient, llm } = useApp();

  const [cases, setCases] = useState<SampleCase[]>([]);
  const [indications, setIndications] = useState<Indication[]>([]);
  const [bannerVisible, setBannerVisible] = useState(false);
  const [selected, setSelected] = useState<SampleCase | null>(null);
  const [importedCase, setImportedCase] = useState<SampleCase | null>(null);
  const [manualFields, setManualFields] = useState<ManualCaseFields>(EMPTY_MANUAL_FIELDS);
  const [text, setText] = useState('');
  const [useOpenFda, setUseOpenFda] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // overlay state
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [busy, setBusy] = useState(false);

  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const mountedRef = useRef(true);
  const lastPayloadRef = useRef<string | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    (async () => {
      try {
        const res = await getCases();
        if (!mountedRef.current) return;
        setCases(res.cases);
        setIndications(res.indications ?? []);
        setBannerVisible(false);
      } catch {
        if (mountedRef.current) setBannerVisible(true);
      }
    })();
    return () => {
      mountedRef.current = false;
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
    };
  }, []);

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  const onSelectPreset = useCallback((c: SampleCase) => {
    setImportedCase(null);
    if (selected?.id === c.id) {
      setSelected(null);
      return;
    }
    setSelected(c);
    setManualFields(caseToManualFields(c));
  }, [selected?.id]);

  const importPayload = useCallback((raw: string): SampleCase => {
    const imported = normalizePatientPayload(decodePayload(raw));
    setImportedCase(imported);
    setSelected(null);
    setManualFields(EMPTY_MANUAL_FIELDS);
    setError(null);
    return imported;
  }, []);

  const setManualField = useCallback(
    (key: keyof ManualCaseFields, value: string) => {
      setManualFields((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  // ── voice stage 3B: structure the dictated note → confirmable fields + flags ──
  const [parsed, setParsed] = useState<ParseCaseResult | null>(null);
  const [parsing, setParsing] = useState(false);
  const [confirmed, setConfirmed] = useState<Set<CapturedKey>>(new Set());
  const textRef = useRef(text);
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  // Structure the note via /api/parse_case — the same parse Analyze runs server-side,
  // surfaced for confirmation. Never analyzes or prescribes; the clinician confirms.
  const runParse = useCallback(
    async (note: string) => {
      const trimmed = note.trim();
      if (!trimmed) return;
      setParsing(true);
      setError(null);
      try {
        const res = await parseCase(trimmed, llm);
        if (!mountedRef.current) return;
        setParsed(res);
        setConfirmed(new Set());
      } catch (e) {
        if (mountedRef.current) {
          setError(e instanceof Error ? e.message : 'Could not structure the note');
        }
      } finally {
        if (mountedRef.current) setParsing(false);
      }
    },
    [llm],
  );

  // Voice transcript → append to the case note, then structure it for confirmation.
  const handleTranscript = useCallback(
    (t: string) => {
      const prev = textRef.current.trim();
      const next = prev ? `${prev} ${t}` : t;
      setText(next);
      void runParse(next);
    },
    [runParse],
  );

  // Confirm one captured value into the Case-details form (voice is never silently adopted).
  const confirmField = useCallback(
    (key: CapturedKey) => {
      const f = parsed?.fields;
      if (!f) return;
      switch (key) {
        case 'indication':
          setManualField('indication', f.indication ?? '');
          break;
        case 'age':
          setManualField('age', f.age != null ? String(f.age) : '');
          break;
        case 'sex':
          setManualField('sex', f.sex === 'M' || f.sex === 'F' ? f.sex : f.sex ? 'Other' : '');
          break;
        case 'egfr':
          setManualField('egfr', f.egfr != null ? String(f.egfr) : '');
          break;
        case 'weight_kg':
          setManualField('weightKg', f.weight_kg != null ? String(f.weight_kg) : '');
          break;
        case 'allergies':
          setManualField('allergies', (f.allergies ?? []).join(', '));
          break;
        case 'meds':
          setManualField('meds', (f.current_medications ?? []).join(', '));
          break;
      }
      setConfirmed((prev) => new Set(prev).add(key));
    },
    [parsed, setManualField],
  );

  // Confirm exactly the chips CapturedFields renders (shared capturedKeys → no drift).
  const confirmAllFields = useCallback(() => {
    if (!parsed) return;
    capturedKeys(parsed).forEach(confirmField);
  }, [parsed, confirmField]);

  // Pick a known indication when the dictated one didn't map → fill the form + confirm.
  const pickIndication = useCallback(
    (value: string) => {
      setManualField('indication', value);
      setConfirmed((prev) => new Set(prev).add('indication'));
    },
    [setManualField],
  );

  // Merge preset/imported source fields with the manual form → the fields sent to Analyze.
  // Shared by the required-field gate (onAnalyze) and the request itself (runAnalyze).
  const collectAnalyzeInput = useCallback(
    (overrideCase?: SampleCase) => {
      const sourceCase = overrideCase ?? importedCase ?? selected;
      const { id, title: sourceTitle, ...sourceFields } = sourceCase ?? ({} as SampleCase);
      void id;
      void sourceTitle;
      const manualCaseFields = overrideCase ? {} : buildManualCaseFields(manualFields);
      const fields = {
        ...(sourceCase ? (sourceFields as unknown as Record<string, unknown>) : {}),
        ...(manualCaseFields as Record<string, unknown>),
      };
      return { sourceCase, manualCaseFields, fields };
    },
    [importedCase, selected, manualFields],
  );

  const runAnalyze = useCallback(async (overrideCase?: SampleCase) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setSteps([]);
    setOverlayVisible(true);

    // Play placeholder steps one-by-one while the request runs.
    PLACEHOLDER_STEPS.forEach((step, i) => {
      const t = setTimeout(() => {
        if (mountedRef.current) setSteps((prev) => [...prev, step]);
      }, i * PLACEHOLDER_MS);
      timersRef.current.push(t);
    });

    const { sourceCase, manualCaseFields, fields } = collectAnalyzeInput(overrideCase);

    try {
      const bundle = await analyze({
        fields,
        text,
        patient_id: sourceCase?.id ?? `case-${patients.length + 1}`,
        use_openfda: useOpenFda,
        llm,
      });
      if (!mountedRef.current) return;

      // Replace placeholders with the real trace, revealed rapidly.
      clearTimers();
      setSteps([]);
      const trace = bundle.trace ?? [];
      trace.forEach((step, i) => {
        const t = setTimeout(() => {
          if (mountedRef.current) setSteps((prev) => [...prev, step]);
        }, i * REAL_STEP_MS);
        timersRef.current.push(t);
      });

      const title =
        sourceCase?.title ??
        titleFromManualFields(manualCaseFields) ??
        (text.trim() ? text.trim().slice(0, 40) : null) ??
        bundle.result.indication_label ??
        'Case';

      const done = setTimeout(() => {
        if (!mountedRef.current) return;
        addPatient(bundle, title);
        setOverlayVisible(false);
        setBusy(false);
        router.push('/reasoning');
      }, trace.length * REAL_STEP_MS + 400);
      timersRef.current.push(done);
    } catch (e) {
      clearTimers();
      if (!mountedRef.current) return;
      setOverlayVisible(false);
      setBusy(false);
      setError(e instanceof Error ? e.message : 'Analyze failed');
    }
  }, [busy, collectAnalyzeInput, text, useOpenFda, llm, patients.length, addPatient, router, clearTimers]);

  // #3 soft guard: if the note was structured but some captured values are still
  // unconfirmed, nudge the clinician to review first — non-blocking ("Analyze anyway"
  // proceeds). Skipped for imported/preset cases (overrideCase), which have no voice chips.
  const onAnalyze = useCallback(
    (overrideCase?: SampleCase) => {
      if (busy) return;
      // Hard gate (team policy): every case field must be filled before Analyze — precise
      // dose (weight/eGFR/age), minor & sex distinctions. Blocks; not a soft nudge.
      const { fields } = collectAnalyzeInput(overrideCase);
      const missing = missingRequiredFields(fields);
      if (missing.length > 0) {
        setError(
          `Complete the required fields before analyzing — missing: ${missing.join(', ')}. ` +
            'Precise dose, minor & sex distinctions need indication, age, sex, weight and eGFR.',
        );
        return;
      }
      setError(null);
      const pending =
        parsed && !overrideCase ? capturedKeys(parsed).filter((k) => !confirmed.has(k)) : [];
      if (pending.length > 0) {
        Alert.alert(
          'Unconfirmed values',
          `${pending.length} captured value${pending.length > 1 ? 's are' : ' is'} not confirmed yet. Review before analyzing?`,
          [
            { text: 'Review', style: 'cancel' },
            {
              text: 'Analyze anyway',
              style: 'destructive',
              onPress: () => void runAnalyze(overrideCase),
            },
          ],
        );
        return;
      }
      void runAnalyze(overrideCase);
    },
    [busy, collectAnalyzeInput, parsed, confirmed, runAnalyze],
  );

  useEffect(() => {
    const raw = firstString(params.payload) ?? firstString(params.patient);
    if (!raw || raw === lastPayloadRef.current) return;
    lastPayloadRef.current = raw;
    try {
      const imported = importPayload(raw);
      const auto = firstString(params.auto);
      if (auto === '1' || auto === 'true') {
        setTimeout(() => {
          void onAnalyze(imported);
        }, 80);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not import patient payload.');
    }
  }, [params.payload, params.patient, params.auto, importPayload, onAnalyze]);

  return (
    <KeyboardAvoidingView
      style={styles.screen}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled">
        {bannerVisible && (
          <View style={styles.banner}>
            <Text style={styles.bannerText}>
              Backend not reachable at {API_BASE} — start demo/demo_app.py
            </Text>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Dismiss warning"
              onPress={() => setBannerVisible(false)}
              hitSlop={12}
              style={styles.bannerClose}>
              <Text style={styles.bannerCloseText}>✕</Text>
            </Pressable>
          </View>
        )}

        <Text style={styles.sectionLabel}>SAMPLE CASES</Text>
        <CasePresetChips
          cases={cases}
          selectedId={selected?.id ?? null}
          onSelect={onSelectPreset}
        />
        {selected && (
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>{selected.title}</Text>
            <Text style={styles.summaryText}>{summarizeCase(selected)}</Text>
          </View>
        )}

        <Text style={styles.sectionLabel}>CASE DETAILS</Text>
        <Text style={styles.requiredHint}>
          Indication, age, sex, weight &amp; eGFR required — for precise dose, minor &amp; sex
          distinctions. Allergies &amp; meds optional (empty = none).
        </Text>
        <View style={styles.formCard}>
          <View style={styles.fieldBlock}>
            <Text style={styles.fieldLabel}>Indication</Text>
            <TextInput
              style={styles.fieldInput}
              value={manualFields.indication}
              onChangeText={(value) => setManualField('indication', value)}
              placeholder="acute coronary syndrome"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
            />
          </View>

          <View style={styles.fieldRow}>
            <View style={styles.ageField}>
              <Text style={styles.fieldLabel}>Age</Text>
              <TextInput
                style={styles.fieldInput}
                value={manualFields.age}
                onChangeText={(value) => setManualField('age', value)}
                keyboardType="number-pad"
                placeholder="64"
                placeholderTextColor={colors.textMuted}
              />
            </View>
            <View style={styles.sexField}>
              <Text style={styles.fieldLabel}>Sex</Text>
              <View style={styles.sexPicker}>
                {SEX_OPTIONS.map((option) => {
                  const active = manualFields.sex === option;
                  return (
                    <Pressable
                      key={option}
                      accessibilityRole="button"
                      accessibilityState={{ selected: active }}
                      onPress={() => setManualField('sex', active ? '' : option)}
                      style={({ pressed }) => [
                        styles.sexOption,
                        active && styles.sexOptionActive,
                        pressed && { opacity: 0.82 },
                      ]}>
                      <Text style={[styles.sexOptionText, active && styles.sexOptionTextActive]}>
                        {option}
                      </Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>
          </View>

          <View style={styles.fieldRow}>
            <View style={styles.compactField}>
              <Text style={styles.fieldLabel}>Weight kg</Text>
              <TextInput
                style={styles.fieldInput}
                value={manualFields.weightKg}
                onChangeText={(value) => setManualField('weightKg', value)}
                keyboardType="decimal-pad"
                placeholder="82"
                placeholderTextColor={colors.textMuted}
              />
            </View>
            <View style={styles.compactField}>
              <Text style={styles.fieldLabel}>eGFR</Text>
              <TextInput
                style={styles.fieldInput}
                value={manualFields.egfr}
                onChangeText={(value) => setManualField('egfr', value)}
                keyboardType="decimal-pad"
                placeholder="72"
                placeholderTextColor={colors.textMuted}
              />
            </View>
          </View>

          <View style={styles.fieldBlock}>
            <Text style={styles.fieldLabel}>Allergies</Text>
            <TextInput
              style={styles.fieldInput}
              value={manualFields.allergies}
              onChangeText={(value) => setManualField('allergies', value)}
              placeholder="aspirin, penicillin"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
            />
          </View>

          <View style={styles.fieldBlock}>
            <Text style={styles.fieldLabel}>Current meds</Text>
            <TextInput
              style={styles.fieldInput}
              value={manualFields.meds}
              onChangeText={(value) => setManualField('meds', value)}
              placeholder="metoprolol, warfarin"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
            />
          </View>
        </View>

        <Text style={styles.sectionLabel}>CASE NOTE</Text>
        <VoiceDictation onTranscript={handleTranscript} />
        <TextInput
          style={styles.textarea}
          multiline
          textAlignVertical="top"
          value={text}
          onChangeText={setText}
          placeholder="BP 88/54, HR 112, crushing chest pain."
          placeholderTextColor={colors.textMuted}
        />

        {/* Structure a typed or edited note into confirmable fields (voice auto-structures). */}
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Structure note into fields"
          onPress={() => void runParse(text)}
          disabled={parsing || !text.trim()}
          style={({ pressed }) => [
            styles.structureBtn,
            (pressed || parsing || !text.trim()) && { opacity: 0.6 },
          ]}>
          {parsing ? (
            <ActivityIndicator size="small" color={colors.accent} />
          ) : (
            <Text style={styles.structureBtnText}>Structure note</Text>
          )}
        </Pressable>

        {parsed && (
          <CapturedFields
            parsed={parsed}
            confirmed={confirmed}
            onConfirm={confirmField}
            onConfirmAll={confirmAllFields}
            indicationOptions={indications}
            onPickIndication={pickIndication}
          />
        )}

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>Fetch live openFDA labels</Text>
          <Switch
            value={useOpenFda}
            onValueChange={setUseOpenFda}
            trackColor={{ false: colors.silverLight, true: colors.accent }}
            thumbColor="#FFFFFF"
          />
        </View>

        {error && <Text style={styles.errorText}>{error}</Text>}

        <Pressable
          accessibilityRole="button"
          onPress={() => onAnalyze()}
          disabled={busy}
          style={({ pressed }) => [
            styles.primaryBtn,
            (pressed || busy) && { opacity: 0.7 },
          ]}>
          <Text style={styles.primaryBtnText}>Analyze case →</Text>
        </Pressable>
      </ScrollView>

      <Modal visible={overlayVisible} transparent animationType="fade">
        <View style={styles.overlayBackdrop}>
          <View style={styles.traceCard}>
            <View style={styles.traceHeader}>
              <Text style={styles.traceHeaderText}>AGENT WORKFLOW</Text>
              <ActivityIndicator size="small" color={colors.traceText} />
            </View>
            <ScrollView style={styles.traceBody}>
              {steps.map((s, i) => (
                <View key={`${i}-${s.title}`} style={styles.traceStep}>
                  <Text style={styles.traceIcon}>•</Text>
                  <View style={styles.traceStepBody}>
                    <Text style={styles.traceTitle}>{s.title}</Text>
                    {!!s.detail && <Text style={styles.traceDetail}>{s.detail}</Text>}
                  </View>
                </View>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: 40,
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FEF3C7',
    borderWidth: 1,
    borderColor: colors.amber,
    borderRadius: radius.sm,
    padding: spacing.md,
    marginBottom: spacing.lg,
    gap: spacing.sm,
  },
  bannerText: {
    flex: 1,
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.amber,
  },
  bannerClose: {
    minWidth: 28,
    minHeight: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bannerCloseText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.amber,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.4,
    color: colors.textMuted,
    textTransform: 'uppercase',
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  requiredHint: {
    fontFamily: fonts.body,
    fontSize: 11.5,
    lineHeight: 16,
    color: colors.textMuted,
    marginTop: -4,
    marginBottom: spacing.sm,
  },
  formCard: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.card,
    padding: spacing.md,
    gap: spacing.sm,
    ...shadow.sm,
  },
  fieldBlock: {
    gap: 6,
  },
  fieldRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  ageField: {
    width: 86,
    gap: 6,
  },
  sexField: {
    flex: 1,
    gap: 6,
  },
  compactField: {
    flex: 1,
    gap: 6,
  },
  fieldLabel: {
    fontFamily: fonts.heading,
    fontSize: 10.5,
    letterSpacing: 0,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  fieldInput: {
    minHeight: 44,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: 10,
    fontFamily: fonts.body,
    fontSize: 13.5,
    color: colors.text,
  },
  sexPicker: {
    flexDirection: 'row',
    gap: 6,
  },
  sexOption: {
    flex: 1,
    minHeight: 44,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    backgroundColor: colors.bg,
    borderRadius: radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sexOptionActive: {
    borderColor: colors.accent,
    backgroundColor: colors.accent,
  },
  sexOptionText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13,
    color: colors.textSecondary,
  },
  sexOptionTextActive: {
    color: '#FFFFFF',
  },
  summaryCard: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.card,
    padding: spacing.md,
    marginTop: spacing.md,
    ...shadow.sm,
  },
  summaryTitle: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13.5,
    color: colors.accent,
    marginBottom: 4,
  },
  summaryText: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  textarea: {
    minHeight: 82,
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    padding: spacing.md,
    fontFamily: fonts.body,
    fontSize: 13.5,
    lineHeight: 20,
    color: colors.text,
  },
  structureBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 40,
    marginTop: spacing.sm,
    marginBottom: spacing.sm,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    backgroundColor: colors.bgCard,
  },
  structureBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13,
    color: colors.accent,
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 44,
    marginTop: spacing.lg,
  },
  switchLabel: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13.5,
    color: colors.textSecondary,
  },
  errorText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.red,
    marginTop: spacing.md,
  },
  primaryBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    minHeight: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.lg,
    ...shadow.md,
  },
  primaryBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14.5,
    color: '#FFFFFF',
  },
  overlayBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(15,43,91,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  traceCard: {
    width: '100%',
    maxHeight: '70%',
    backgroundColor: colors.traceBg,
    borderRadius: radius.card,
    padding: spacing.lg,
    ...shadow.md,
  },
  traceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  traceHeaderText: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.6,
    color: colors.traceMuted,
  },
  traceBody: {
    flexGrow: 0,
  },
  traceStep: {
    flexDirection: 'row',
    gap: 9,
    paddingVertical: 5,
    alignItems: 'flex-start',
  },
  traceIcon: {
    fontSize: 15,
    lineHeight: 20,
  },
  traceStepBody: {
    flex: 1,
  },
  traceTitle: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13.5,
    color: '#FFFFFF',
  },
  traceDetail: {
    fontFamily: fonts.heading,
    fontSize: 12,
    color: colors.traceMuted,
    marginTop: 1,
  },
});
