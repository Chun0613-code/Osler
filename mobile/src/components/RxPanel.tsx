/**
 * RxPanel — the current patient's e-prescribing history, shown as the "Rx"
 * segment inside Reasoning (Rx is a per-patient detail, not a standalone tab).
 * Loads on mount and on patient change; pull-to-refresh. Transmission status
 * (mock vs. actually sent) is surfaced so the clinician knows whether an Rx
 * really went to a pharmacy. The patient identity lives in the Reasoning
 * summary strip above, so it is not repeated here.
 */
import React, { useCallback, useEffect, useState } from 'react';
import { FlatList, RefreshControl, StyleSheet, Text } from 'react-native';

import { getPrescriptions, type Prescription } from '@/api/osler';
import PrescriptionCard from '@/components/PrescriptionCard';
import { useApp } from '@/state/AppContext';
import { colors, fonts, spacing } from '@/theme/tokens';

export default function RxPanel() {
  const { current } = useApp();
  const [items, setItems] = useState<Prescription[]>([]);
  const [transmits, setTransmits] = useState(false); // true = real pharmacy transmission (Photon)
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
      setTransmits(res.photon);
    } catch {
      // demo-grade: keep the prior list on a transient error
    } finally {
      setLoading(false);
    }
  }, [current]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <FlatList
      data={items}
      keyExtractor={(it) => it.rx_id}
      contentContainerStyle={styles.list}
      refreshControl={
        <RefreshControl refreshing={loading} onRefresh={load} tintColor={colors.accent} />
      }
      ListHeaderComponent={
        !transmits && items.length > 0 ? (
          <Text style={styles.mockNote}>Mock mode — not sent to a pharmacy.</Text>
        ) : null
      }
      renderItem={({ item }) => <PrescriptionCard rx={item} />}
      ListEmptyComponent={
        <Text style={styles.none}>
          No prescriptions yet — prescribe a drug from the Drugs tab.
        </Text>
      }
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: spacing.md },
  mockNote: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12,
    color: colors.amber,
    marginBottom: spacing.sm,
    paddingHorizontal: spacing.xs,
  },
  none: {
    fontFamily: fonts.body,
    fontSize: 13.5,
    lineHeight: 20,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: spacing.xl,
    paddingHorizontal: spacing.xl,
  },
});
