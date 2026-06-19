/**
 * Prescribe — the review/sign step. Pushed from a DrugCard's "Prescribe" CTA (not a tab).
 * We ask the backend to start an Rx for {patientId, drug}; then branch on mode:
 *   • photon + provider connected → NATIVE one-screen Sign & Send (review the engine's
 *     pick, edit sig/qty/days/refills, sign). createPrescription/createOrder run on the
 *     backend under the provider's token. No browser hop.
 *   • photon + NOT connected → prompt the clinician to connect Photon in Settings first.
 *   • mock mode → the local review/sign form in a WebView that postMessages back.
 */
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';

import {
  API_BASE,
  getPhotonStatus,
  getPrescribeOptions,
  signPrescription,
  startPrescription,
  type MedOption,
  type PrescribeOptions,
} from '@/api/osler';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

type Phase = 'loading' | 'mock' | 'needConnect' | 'review' | 'error';

export default function PrescribeScreen() {
  const router = useRouter();
  const { drug, patientId } = useLocalSearchParams<{ drug: string; patientId: string }>();

  const [phase, setPhase] = useState<Phase>('loading');
  const [error, setError] = useState<string | null>(null);
  const [mockUrl, setMockUrl] = useState<string | null>(null);

  // Native review state.
  const [rxId, setRxId] = useState<string>('');
  const [options, setOptions] = useState<PrescribeOptions | null>(null);
  const [treatmentId, setTreatmentId] = useState<string>('');
  const [sig, setSig] = useState('');
  const [qty, setQty] = useState('30');
  const [unit, setUnit] = useState('Each');
  const [days, setDays] = useState('30');
  const [refills, setRefills] = useState('0');
  const [signing, setSigning] = useState(false);

  const busy = useRef(false);

  // Runs each time the screen gains focus. /prescribe is a persistent tab route
  // (href:null), so a plain useEffect would fire only on first mount and stale-cache the
  // result. The busy ref prevents re-entry while the async flow is in progress.
  useFocusEffect(
    useCallback(() => {
      if (busy.current) return;
      busy.current = true;
      setPhase('loading');
      setError(null);
      setMockUrl(null);
      (async () => {
        try {
          const res = await startPrescription(String(patientId), String(drug));
          setRxId(res.rx_id);
          if (res.mode !== 'photon') {
            setMockUrl(`${API_BASE}${res.embed_path}`); // mock → embedded form
            setPhase('mock');
            return;
          }
          // photon mode: a provider must be signed in to sign natively.
          const status = await getPhotonStatus();
          if (!status.connected) {
            setPhase('needConnect');
            return;
          }
          const opts = await getPrescribeOptions(res.rx_id);
          setOptions(opts);
          setTreatmentId(opts.default.treatment_id ?? opts.candidates[0]?.treatment_id ?? '');
          setSig(opts.default.sig);
          setQty(String(opts.default.dispense_quantity));
          setUnit(opts.default.dispense_unit);
          setDays(String(opts.default.days_supply));
          setRefills(String(opts.default.refills));
          setPhase('review');
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Could not start prescription.');
          setPhase('error');
        } finally {
          busy.current = false;
        }
      })();
    }, [drug, patientId]),
  );

  const onMockMessage = (e: WebViewMessageEvent) => {
    let msg: { type?: string } = {};
    try {
      msg = JSON.parse(e.nativeEvent.data);
    } catch {
      return;
    }
    if (msg.type === 'done') router.replace('/prescriptions');
    else if (msg.type === 'cancel') router.back();
  };

  const onSign = async () => {
    if (!treatmentId) {
      setError('Pick a medication to prescribe.');
      return;
    }
    const q = Number(qty);
    const d = Number(days);
    if (!sig.trim()) {
      setError('Enter the sig (patient instructions).');
      return;
    }
    if (!Number.isFinite(q) || q <= 0 || !Number.isFinite(d) || d <= 0) {
      setError('Quantity and days supply must be positive numbers.');
      return;
    }
    setError(null);
    setSigning(true);
    try {
      await signPrescription({
        rx_id: rxId,
        treatment_id: treatmentId,
        sig: sig.trim(),
        dispense_quantity: q,
        dispense_unit: unit.trim() || 'Each',
        days_supply: d,
        refills: Number(refills) || 0,
      });
      router.replace('/prescriptions');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not sign the prescription.');
    } finally {
      setSigning(false);
    }
  };

  // ── mock mode: embedded WebView form ─────────────────────────────────────
  if (phase === 'mock' && mockUrl) {
    return (
      <View style={styles.screen}>
        <WebView
          source={{ uri: mockUrl }}
          onMessage={onMockMessage}
          originWhitelist={['*']}
          startInLoadingState
          domStorageEnabled
          style={styles.web}
        />
      </View>
    );
  }

  // ── hard error (couldn't start) ──────────────────────────────────────────
  if (phase === 'error') {
    return (
      <View style={styles.center}>
        <Text style={styles.errTitle}>Cannot prescribe</Text>
        <Text style={styles.errText}>{error}</Text>
      </View>
    );
  }

  // ── photon mode, provider not connected ──────────────────────────────────
  if (phase === 'needConnect') {
    return (
      <View style={styles.center}>
        <Text style={styles.errTitle}>Connect Photon to sign</Text>
        <Text style={styles.errText}>
          Signing a prescription requires a logged-in prescriber. Connect your Neutron
          account once in Settings, then return here.
        </Text>
        <Pressable
          accessibilityRole="button"
          onPress={() => router.replace('/settings')}
          style={({ pressed }) => [styles.primaryBtn, pressed && { opacity: 0.85 }]}>
          <Text style={styles.primaryBtnText}>Go to Settings</Text>
        </Pressable>
      </View>
    );
  }

  // ── loading ──────────────────────────────────────────────────────────────
  if (phase === 'loading' || !options) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.loading}>Preparing the prescription…</Text>
      </View>
    );
  }

  // ── native review / Sign & Send ──────────────────────────────────────────
  return (
    <KeyboardAvoidingView
      style={styles.screen}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={styles.drug}>{options.drug}</Text>
        {options.clinical_role ? (
          <Text style={styles.role}>{options.clinical_role}</Text>
        ) : null}
        {options.rationale ? <Text style={styles.rationale}>{options.rationale}</Text> : null}

        <Text style={styles.fieldLabel}>MEDICATION</Text>
        <View style={styles.card}>
          {options.candidates.length === 0 ? (
            <Text style={styles.note}>No catalog match found for {options.drug}.</Text>
          ) : (
            options.candidates.map((m: MedOption) => {
              const active = m.treatment_id === treatmentId;
              return (
                <Pressable
                  key={m.treatment_id}
                  accessibilityRole="radio"
                  accessibilityState={{ selected: active }}
                  onPress={() => setTreatmentId(m.treatment_id)}
                  style={({ pressed }) => [
                    styles.medRow,
                    active && styles.medRowActive,
                    pressed && { opacity: 0.9 },
                  ]}>
                  <View style={[styles.radio, active && styles.radioActive]}>
                    {active ? <View style={styles.radioDot} /> : null}
                  </View>
                  <Text style={[styles.medName, active && styles.medNameActive]} numberOfLines={2}>
                    {m.name}
                  </Text>
                </Pressable>
              );
            })
          )}
        </View>

        <Text style={styles.fieldLabel}>SIG (PATIENT INSTRUCTIONS)</Text>
        <TextInput
          style={[styles.input, styles.sigInput]}
          value={sig}
          onChangeText={setSig}
          placeholder="e.g. Take 1 tablet by mouth once daily"
          placeholderTextColor={colors.textMuted}
          multiline
        />

        <View style={styles.row}>
          <View style={styles.col}>
            <Text style={styles.fieldLabel}>QUANTITY</Text>
            <TextInput
              style={styles.input}
              value={qty}
              onChangeText={setQty}
              keyboardType="numeric"
              placeholderTextColor={colors.textMuted}
            />
          </View>
          <View style={styles.col}>
            <Text style={styles.fieldLabel}>UNIT</Text>
            <TextInput
              style={styles.input}
              value={unit}
              onChangeText={setUnit}
              autoCapitalize="words"
              placeholderTextColor={colors.textMuted}
            />
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.col}>
            <Text style={styles.fieldLabel}>DAYS SUPPLY</Text>
            <TextInput
              style={styles.input}
              value={days}
              onChangeText={setDays}
              keyboardType="numeric"
              placeholderTextColor={colors.textMuted}
            />
          </View>
          <View style={styles.col}>
            <Text style={styles.fieldLabel}>REFILLS</Text>
            <TextInput
              style={styles.input}
              value={refills}
              onChangeText={setRefills}
              keyboardType="numeric"
              placeholderTextColor={colors.textMuted}
            />
          </View>
        </View>

        {error ? <Text style={styles.errorNote}>{error}</Text> : null}

        <Pressable
          accessibilityRole="button"
          disabled={signing}
          onPress={onSign}
          style={({ pressed }) => [styles.signBtn, pressed && { opacity: 0.85 }]}>
          {signing ? (
            <ActivityIndicator color="#FFFFFF" />
          ) : (
            <Text style={styles.signBtnText}>Sign &amp; Send</Text>
          )}
        </Pressable>

        <Text style={styles.disclaimer}>
          Signs under your connected Neutron prescriber identity and creates a real order
          in the Photon sandbox (no medication is dispensed). You are the prescriber of
          record.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  web: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.lg, paddingBottom: 48 },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.bg,
    padding: spacing.xl,
    gap: spacing.sm,
  },
  loading: { fontFamily: fonts.body, fontSize: 13.5, color: colors.textMuted },
  errTitle: { fontFamily: fonts.headingBold, fontSize: 18, color: colors.text, textAlign: 'center' },
  errText: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textSecondary,
    textAlign: 'center',
  },
  primaryBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 28,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.md,
  },
  primaryBtnText: { fontFamily: fonts.bodySemiBold, fontSize: 14, color: '#FFFFFF' },

  drug: { fontFamily: fonts.headingBold, fontSize: 22, color: colors.text },
  role: { fontFamily: fonts.bodyMedium, fontSize: 13, color: colors.accent, marginTop: 2 },
  rationale: {
    fontFamily: fonts.body,
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
    marginTop: spacing.sm,
  },
  fieldLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.4,
    color: colors.textMuted,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.card,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    padding: spacing.sm,
    gap: spacing.xs,
    ...shadow.sm,
  },
  medRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.sm,
    borderRadius: radius.sm,
  },
  medRowActive: { backgroundColor: colors.accentLight },
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: colors.borderSolid,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioActive: { borderColor: colors.accent },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: colors.accent },
  medName: { flex: 1, fontFamily: fonts.body, fontSize: 13.5, color: colors.textSecondary },
  medNameActive: { color: colors.text, fontFamily: fonts.bodyMedium },
  note: {
    fontFamily: fonts.body,
    fontSize: 13,
    lineHeight: 19,
    color: colors.textMuted,
    padding: spacing.sm,
  },

  input: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    paddingHorizontal: 14,
    minHeight: 44,
    paddingVertical: 0,
  },
  sigInput: { minHeight: 66, paddingTop: 12, textAlignVertical: 'top' },
  row: { flexDirection: 'row', gap: spacing.md },
  col: { flex: 1 },

  errorNote: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.red,
    marginTop: spacing.md,
  },
  signBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    minHeight: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.lg,
  },
  signBtnText: { fontFamily: fonts.bodySemiBold, fontSize: 15, color: '#FFFFFF' },
  disclaimer: {
    fontFamily: fonts.body,
    fontSize: 11.5,
    lineHeight: 17,
    color: colors.textMuted,
    marginTop: spacing.md,
    textAlign: 'center',
  },
});
