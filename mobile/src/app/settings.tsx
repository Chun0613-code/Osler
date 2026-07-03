/**
 * Settings — Photon e-prescribing connection and an About blurb.
 */
import { useFocusEffect } from 'expo-router';
import * as Linking from 'expo-linking';
import * as WebBrowser from 'expo-web-browser';
import React, { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import {
  disconnectPhoton,
  getPhotonAuthorizeUrl,
  getPhotonStatus,
  type PhotonStatus,
} from '@/api/osler';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

export default function SettingsScreen() {

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

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        style={styles.flex}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled">
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

        {/* ── About ───────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>ABOUT</Text>
        <View style={styles.card}>
          <Image
            source={require('../../assets/images/oslian-mark.png')}
            resizeMode="contain"
            style={styles.aboutLogo}
          />
          <Text style={styles.note}>
            Oslian·Rx — clinician-facing drug recommendation demo. Every
            recommendation comes from the symbolic engine, not a language model.
            Decision support only — not a clinically validated tool.
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
  aboutLogo: {
    width: 52,
    height: 52,
    marginBottom: spacing.xs,
  },
});
