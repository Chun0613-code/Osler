/**
 * Prescriptions — e-prescribing history for the current patient. Reads from the backend
 * prescription store, refreshing on focus so a freshly-signed Rx appears after returning
 * from the prescribe WebView.
 */
import { useFocusEffect } from 'expo-router';
import React, { useCallback, useState } from 'react';
import {
  FlatList,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { getPrescriptions, type Prescription } from '@/api/osler';
import PrescriptionCard from '@/components/PrescriptionCard';
import { useApp, type Patient } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

export default function PrescriptionsScreen() {
  const { patients, current, currentId, selectPatient } = useApp();
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
      <PatientSelector
        patients={patients}
        currentId={currentId}
        onSelect={selectPatient}
      />
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

function PatientSelector({
  patients,
  currentId,
  onSelect,
}: {
  patients: Patient[];
  currentId: string | null;
  onSelect: (id: string) => void;
}) {
  if (patients.length <= 1) return null;

  return (
    <View style={styles.selector}>
      <Text style={styles.sectionLabel}>RX FOR</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.selectorScroll}>
        {patients.map((patient, index) => {
          const active = patient.id === currentId;
          const drugCount = patient.bundle.result.candidates.length;
          return (
            <Pressable
              key={patient.id}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              onPress={() => onSelect(patient.id)}
              style={({ pressed }) => [
                styles.patientPill,
                active && styles.patientPillActive,
                pressed && { opacity: 0.88 },
              ]}>
              <View style={[styles.patientIndex, active && styles.patientIndexActive]}>
                <Text style={[styles.patientIndexText, active && styles.patientIndexTextActive]}>
                  {index + 1}
                </Text>
              </View>
              <View style={styles.patientPillBody}>
                <Text
                  style={[styles.patientTitle, active && styles.patientTitleActive]}
                  numberOfLines={1}>
                  {patient.title}
                </Text>
                <Text
                  style={[styles.patientMeta, active && styles.patientMetaActive]}
                  numberOfLines={1}>
                  {drugCount} drug{drugCount === 1 ? '' : 's'}
                </Text>
              </View>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  selector: {
    backgroundColor: colors.bg,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.md,
    gap: spacing.xs,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  selectorScroll: {
    gap: spacing.sm,
    paddingRight: spacing.lg,
  },
  patientPill: {
    width: 246,
    minHeight: 68,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.card,
    paddingHorizontal: 12,
    paddingVertical: 10,
    ...shadow.sm,
  },
  patientPillActive: {
    borderColor: colors.accent,
    backgroundColor: colors.accent,
  },
  patientIndex: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accentLight,
  },
  patientIndexActive: {
    backgroundColor: '#FFFFFF',
  },
  patientIndexText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13,
    color: colors.accent,
  },
  patientIndexTextActive: {
    color: colors.accent,
  },
  patientPillBody: {
    flex: 1,
    minWidth: 0,
  },
  patientTitle: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13.5,
    color: colors.accent,
  },
  patientTitleActive: {
    color: '#FFFFFF',
  },
  patientMeta: {
    fontFamily: fonts.body,
    fontSize: 11.5,
    color: colors.textMuted,
    marginTop: 2,
  },
  patientMetaActive: {
    color: '#DCE6F7',
  },
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
