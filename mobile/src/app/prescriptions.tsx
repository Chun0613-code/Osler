/**
 * Prescriptions — e-prescribing history for the current patient. Reads from the backend
 * prescription store, refreshing on focus so a freshly-signed Rx appears after returning
 * from the prescribe WebView.
 */
import { useFocusEffect } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { FlatList, RefreshControl, StyleSheet, Text, View } from 'react-native';

import { getPrescriptions, type Prescription } from '@/api/osler';
import PrescriptionCard from '@/components/PrescriptionCard';
import { useApp } from '@/state/AppContext';
import { colors, fonts, spacing } from '@/theme/tokens';

export default function PrescriptionsScreen() {
  const { current } = useApp();
  const [items, setItems] = useState<Prescription[]>([]);
  const [photon, setPhoton] = useState(false);
  const [env, setEnv] = useState('sandbox');
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!current) {
      setItems([]);
      return;
    }
    setLoading(true);
    try {
      const res = await getPrescriptions(current.id);
      setItems(res.prescriptions);
      setPhoton(res.photon);
      setEnv(res.env);
    } catch {
      // demo-grade: keep the prior list on a transient error
    } finally {
      setLoading(false);
    }
  }, [current]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  if (!current) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyTitle}>No patient selected</Text>
        <Text style={styles.emptyText}>
          Analyze a case, then tap “Prescribe” on a drug in the Reasoning tab.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <View style={styles.banner}>
        <Text style={styles.bannerText}>
          {photon ? `Photon · ${env}` : 'Mock mode (no pharmacy transmission)'} · {current.title}
        </Text>
      </View>
      <FlatList
        data={items}
        keyExtractor={(it) => it.rx_id}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={loading} onRefresh={load} tintColor={colors.accent} />
        }
        renderItem={({ item }) => <PrescriptionCard rx={item} />}
        ListEmptyComponent={
          <Text style={styles.none}>
            No prescriptions yet. Open a drug in Reasoning and tap “Prescribe”.
          </Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  banner: {
    backgroundColor: colors.bgCard,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.borderSolid,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  bannerText: { fontFamily: fonts.bodyMedium, fontSize: 12, color: colors.textMuted },
  list: { padding: spacing.md },
  none: {
    fontFamily: fonts.body,
    fontSize: 13.5,
    lineHeight: 20,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: spacing.xl,
    paddingHorizontal: spacing.xl,
  },
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.bg,
    padding: spacing.xl,
    gap: spacing.sm,
  },
  emptyTitle: { fontFamily: fonts.headingBold, fontSize: 19, color: colors.text },
  emptyText: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textSecondary,
    textAlign: 'center',
  },
});
