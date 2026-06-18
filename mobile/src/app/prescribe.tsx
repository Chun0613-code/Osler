/**
 * Prescribe — the review/sign step. Pushed from a DrugCard's "Prescribe" CTA (not a tab).
 * We ask the backend to start an Rx for {patientId, drug}; then:
 *   • photon mode → open Photon's hosted prescribe page in the SYSTEM browser (Auth0/Google
 *     login is blocked inside an embedded WebView), then return to the Rx tab.
 *   • mock mode  → a local review/sign form in a WebView that postMessages back.
 */
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import * as WebBrowser from 'expo-web-browser';
import React, { useCallback, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';

import { API_BASE, completePrescription, startPrescription } from '@/api/osler';
import { colors, fonts, spacing } from '@/theme/tokens';

export default function PrescribeScreen() {
  const router = useRouter();
  const { drug, patientId } = useLocalSearchParams<{ drug: string; patientId: string }>();
  const [url, setUrl] = useState<string | null>(null); // mock mode: embed page for the WebView
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);

  // Runs each time the screen gains focus. /prescribe is a persistent tab route
  // (href:null), so a plain useEffect would fire only on first mount and stale-cache the
  // result. The busy ref prevents re-entry while the async flow — including the system
  // browser — is in progress.
  useFocusEffect(
    useCallback(() => {
      if (busy.current) return;
      busy.current = true;
      setUrl(null);
      setError(null);
      (async () => {
        try {
          const res = await startPrescription(String(patientId), String(drug));
          if (res.mode === 'photon' && res.photon_url) {
            // Photon's hosted prescribe page opens in the SYSTEM browser — Auth0/Google
            // login works there (an embedded WebView is blocked). The synced patient is
            // pre-bound. No postMessage back, so on return we mark it submitted and show
            // it in the Rx tab; the live order status arrives later via webhook.
            await WebBrowser.openBrowserAsync(res.photon_url);
            try {
              await completePrescription(res.rx_id);
            } catch {
              // leave as draft if the mark-submitted call fails
            }
            router.replace('/prescriptions');
          } else {
            setUrl(`${API_BASE}${res.embed_path}`); // mock → embedded form in a WebView
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Could not start prescription.');
        } finally {
          busy.current = false;
        }
      })();
    }, [drug, patientId, router]),
  );

  const onMessage = (e: WebViewMessageEvent) => {
    let msg: { type?: string } = {};
    try {
      msg = JSON.parse(e.nativeEvent.data);
    } catch {
      return;
    }
    if (msg.type === 'done') router.replace('/prescriptions');
    else if (msg.type === 'cancel') router.back();
  };

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errTitle}>Cannot prescribe</Text>
        <Text style={styles.errText}>{error}</Text>
      </View>
    );
  }

  if (!url) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.loading}>Opening the secure prescribe page…</Text>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <WebView
        source={{ uri: url }}
        onMessage={onMessage}
        originWhitelist={['*']}
        startInLoadingState
        // The embedded Photon/Auth0 (provider) login needs cookies, DOM storage and
        // in-WebView popups to keep its session and complete the OAuth redirect.
        sharedCookiesEnabled
        thirdPartyCookiesEnabled
        domStorageEnabled
        javaScriptCanOpenWindowsAutomatically
        setSupportMultipleWindows={false}
        style={styles.web}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  web: { flex: 1, backgroundColor: colors.bg },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.bg,
    padding: spacing.xl,
    gap: spacing.sm,
  },
  loading: { fontFamily: fonts.body, fontSize: 13.5, color: colors.textMuted },
  errTitle: { fontFamily: fonts.headingBold, fontSize: 18, color: colors.text },
  errText: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textSecondary,
    textAlign: 'center',
  },
});
