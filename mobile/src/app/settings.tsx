/**
 * Settings — LLM provider/key (persisted via expo-secure-store through
 * AppContext.setLlm), backend info, and an About blurb.
 */
import { useFocusEffect } from 'expo-router';
import * as Linking from 'expo-linking';
import * as WebBrowser from 'expo-web-browser';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  API_BASE,
  disconnectPhoton,
  getPhotonAuthorizeUrl,
  getPhotonStatus,
  type LlmSettings,
  type PhotonStatus,
} from '@/api/osler';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

type Provider = NonNullable<LlmSettings['provider']>;

const PROVIDERS: { value: Provider; label: string }[] = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Gemini' },
];

export default function SettingsScreen() {
  const { llm, setLlm } = useApp();
  const [provider, setProvider] = useState<Provider>(llm.provider ?? 'openai');
  const [apiKey, setApiKey] = useState(llm.apiKey ?? '');
  const [saved, setSaved] = useState(false);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Photon e-prescribing connection ──────────────────────────────────────
  const [photon, setPhoton] = useState<PhotonStatus | null>(null);
  const [photonBusy, setPhotonBusy] = useState(false);
  const [photonError, setPhotonError] = useState<string | null>(null);

  const refreshPhoton = useCallback(async () => {
    try {
      setPhoton(await getPhotonStatus());
    } catch {
      setPhoton(null); // backend unreachable — treat as unknown, hide the card
    }
  }, []);

  // Refresh on focus so returning from the OAuth browser reflects the new state.
  useFocusEffect(
    useCallback(() => {
      refreshPhoton();
    }, [refreshPhoton]),
  );

  const connectPhoton = async () => {
    setPhotonError(null);
    setPhotonBusy(true);
    try {
      // Deep link Neutron bounces back to; the backend appends ?photon=connected|error.
      const returnUrl = Linking.createURL('photon-connected');
      const authorizeUrl = await getPhotonAuthorizeUrl(returnUrl);
      const result = await WebBrowser.openAuthSessionAsync(authorizeUrl, returnUrl);
      if (result.type === 'success' && result.url) {
        const { queryParams } = Linking.parse(result.url);
        if (queryParams?.photon === 'error') {
          setPhotonError(String(queryParams.msg ?? 'Photon login failed.'));
        }
      }
      // 'cancel'/'dismiss' → user backed out; status refresh below reflects reality.
      await refreshPhoton();
    } catch (e) {
      setPhotonError(e instanceof Error ? e.message : 'Could not start Photon login.');
    } finally {
      setPhotonBusy(false);
    }
  };

  const handleDisconnect = async () => {
    setPhotonBusy(true);
    setPhotonError(null);
    try {
      await disconnectPhoton();
    } catch {
      // non-fatal
    } finally {
      await refreshPhoton();
      setPhotonBusy(false);
    }
  };

  // Sync local form once the persisted settings finish loading.
  useEffect(() => {
    setProvider(llm.provider ?? 'openai');
    setApiKey(llm.apiKey ?? '');
  }, [llm.provider, llm.apiKey]);

  useEffect(
    () => () => {
      if (savedTimer.current) clearTimeout(savedTimer.current);
    },
    [],
  );

  const save = async () => {
    const trimmed = apiKey.trim();
    await setLlm({ apiKey: trimmed || undefined, provider });
    setSaved(true);
    if (savedTimer.current) clearTimeout(savedTimer.current);
    savedTimer.current = setTimeout(() => setSaved(false), 2000);
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        style={styles.flex}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled">
        {/* ── LLM ─────────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>LLM (FOR CHAT & FREE-TEXT PARSING)</Text>
        <View style={styles.card}>
          <View style={styles.toggleRow}>
            {PROVIDERS.map((p) => {
              const active = provider === p.value;
              return (
                <Pressable
                  key={p.value}
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  onPress={() => setProvider(p.value)}
                  style={({ pressed }) => [
                    styles.toggle,
                    active && styles.toggleActive,
                    pressed && { opacity: 0.8 },
                  ]}>
                  <Text style={[styles.toggleText, active && styles.toggleTextActive]}>
                    {p.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
          <TextInput
            style={styles.input}
            value={apiKey}
            onChangeText={setApiKey}
            placeholder="sk-..."
            placeholderTextColor={colors.textMuted}
            secureTextEntry
            autoCapitalize="none"
            autoCorrect={false}
            accessibilityLabel="API key"
          />
          <View style={styles.saveRow}>
            <Pressable
              accessibilityRole="button"
              onPress={save}
              style={({ pressed }) => [styles.saveButton, pressed && { opacity: 0.85 }]}>
              <Text style={styles.saveButtonText}>Save</Text>
            </Pressable>
            {saved && <Text style={styles.savedText}>Saved ✓</Text>}
          </View>
          <Text style={styles.note}>
            Stored securely on-device (expo-secure-store). Without a key the demo
            still works: rule-based parsing + symbolic engine; only chat falls back
            to a notice.
          </Text>
        </View>

        {/* ── Photon (e-prescribing) ──────────────────────────── */}
        {photon?.enabled && (
          <>
            <Text style={styles.sectionLabel}>PHOTON · E-PRESCRIBING ({photon.env})</Text>
            <View style={styles.card}>
              <View style={styles.statusRow}>
                <View
                  style={[
                    styles.statusDot,
                    { backgroundColor: photon.connected ? colors.green : colors.textMuted },
                  ]}
                />
                <Text style={styles.statusText}>
                  {photon.connected
                    ? `Connected${photon.provider?.name ? ` · ${photon.provider.name}` : ''}`
                    : 'Not connected'}
                </Text>
              </View>
              {photon.connected && photon.provider?.email ? (
                <Text style={styles.note}>{photon.provider.email}</Text>
              ) : null}

              {photon.connected ? (
                <Pressable
                  accessibilityRole="button"
                  disabled={photonBusy}
                  onPress={handleDisconnect}
                  style={({ pressed }) => [styles.outlineButton, pressed && { opacity: 0.85 }]}>
                  {photonBusy ? (
                    <ActivityIndicator color={colors.accent} />
                  ) : (
                    <Text style={styles.outlineButtonText}>Disconnect</Text>
                  )}
                </Pressable>
              ) : (
                <Pressable
                  accessibilityRole="button"
                  disabled={photonBusy}
                  onPress={connectPhoton}
                  style={({ pressed }) => [
                    styles.saveButton,
                    styles.selfStart,
                    pressed && { opacity: 0.85 },
                  ]}>
                  {photonBusy ? (
                    <ActivityIndicator color="#FFFFFF" />
                  ) : (
                    <Text style={styles.saveButtonText}>Connect Photon</Text>
                  )}
                </Pressable>
              )}

              {photonError ? <Text style={styles.errorNote}>{photonError}</Text> : null}
              <Text style={styles.note}>
                Sign in once with your Neutron prescriber account so the symbolic
                engine&apos;s picks can be signed &amp; sent natively. Login opens in your
                system browser (embedded login is blocked).
              </Text>
            </View>
          </>
        )}

        {/* ── Backend ─────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>BACKEND</Text>
        <View style={styles.card}>
          <Text style={styles.apiBase}>{API_BASE}</Text>
          <Text style={styles.note}>
            iOS simulator shares the Mac&apos;s localhost. Start the backend with:{' '}
            <Text style={styles.code}>python demo/demo_app.py</Text> (from the Osler
            repo root, branch demo-rx). For a physical device set{' '}
            <Text style={styles.code}>EXPO_PUBLIC_API_BASE</Text> to your Mac&apos;s
            LAN IP.
          </Text>
        </View>

        {/* ── About ───────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>ABOUT</Text>
        <View style={styles.card}>
          <Image
            source={require('../../assets/images/oslian-mark.png')}
            resizeMode="contain"
            style={styles.aboutLogo}
          />
          <Text style={styles.note}>
            Oslian·Rx — clinician-facing drug recommendation demo. The symbolic
            engine makes every recommendation; the LLM only parses cases and
            explains results. Decision support only — not a clinically validated
            tool.
          </Text>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  content: {
    padding: spacing.lg,
    paddingBottom: 48,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.6,
    color: colors.textMuted,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.card,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    padding: spacing.lg,
    gap: spacing.md,
    ...shadow.sm,
  },
  toggleRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  toggle: {
    flex: 1,
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleActive: {
    backgroundColor: colors.accent,
  },
  toggleText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.accent,
  },
  toggleTextActive: {
    color: '#FFFFFF',
  },
  input: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    paddingHorizontal: 14,
    minHeight: 44,
    paddingVertical: 0,
  },
  saveRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  saveButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 28,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  selfStart: { alignSelf: 'flex-start' },
  saveButtonText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  savedText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.green,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 4.5,
  },
  statusText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.text,
  },
  outlineButton: {
    borderWidth: 1,
    borderColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 28,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-start',
  },
  outlineButtonText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.accent,
  },
  errorNote: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.red,
  },
  note: {
    fontFamily: fonts.body,
    fontSize: 12,
    lineHeight: 18,
    color: colors.textSecondary,
  },
  code: {
    fontFamily: fonts.bodyMedium,
    backgroundColor: colors.accentLight,
    color: colors.accent,
  },
  apiBase: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.text,
  },
  aboutLogo: {
    width: 52,
    height: 52,
    marginBottom: spacing.xs,
  },
});
