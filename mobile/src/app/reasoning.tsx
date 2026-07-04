/**
 * Reasoning screen — answer-first: a Drugs | Rx segmented control over the
 * current patient's analysis bundle, with the low-frequency reference views
 * (reasoning graph, disease model) demoted to secondary links below it.
 *
 * Tapping a drug node in the Graph posts back through GraphWebView →
 * setHighlightDrug(context) → this screen auto-switches to Drugs, scrolls to
 * the card and pulses a highlight for ~2.5s.
 */
import Ionicons from '@expo/vector-icons/Ionicons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import type { DrugCandidate } from '@/api/osler';
import DiseaseModelPanel from '@/components/DiseaseModel';
import DrugCard from '@/components/DrugCard';
import GraphWebView from '@/components/GraphWebView';
import RxPanel from '@/components/RxPanel';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

type Segment = 'drugs' | 'graph' | 'disease' | 'rx';

// Answer-first: the segmented control carries the two views clinicians act on;
// the reference views live in a quieter tinted-chip row (toggle back to Drugs).
const PRIMARY_SEGMENTS: { key: Segment; label: string }[] = [
  { key: 'drugs', label: 'Drugs' },
  { key: 'rx', label: 'Rx' },
];

const REFERENCE_VIEWS: {
  key: Segment;
  label: string;
  icon: React.ComponentProps<typeof Ionicons>['name'];
}[] = [
  { key: 'graph', label: 'Reasoning graph', icon: 'git-network-outline' },
  { key: 'disease', label: 'Disease model', icon: 'pulse-outline' },
];

export default function ReasoningScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ view?: string; t?: string }>();
  const { current, highlightDrug, setHighlightDrug } = useApp();
  const [segment, setSegment] = useState<Segment>(params.view === 'rx' ? 'rx' : 'drugs');
  const [pulseDrug, setPulseDrug] = useState<string | null>(null);
  const listRef = useRef<FlatList<DrugCandidate>>(null);

  const bundle = current?.bundle ?? null;
  const candidates = useMemo(
    () => bundle?.result?.candidates ?? [],
    [bundle],
  );

  // Graph node tap (or any setHighlightDrug) → jump to Drugs, scroll, pulse.
  useEffect(() => {
    if (!highlightDrug) return;
    const target = highlightDrug.toLowerCase();
    setSegment('drugs');
    setPulseDrug(target);

    const index = candidates.findIndex(
      (c) => c.drug?.toLowerCase() === target,
    );
    // Wait a tick so the Drugs FlatList is mounted before scrolling.
    const scrollTimer = setTimeout(() => {
      if (index >= 0) {
        listRef.current?.scrollToIndex({
          index,
          animated: true,
          viewPosition: 0.1,
        });
      }
    }, 250);

    const clearTimer = setTimeout(() => {
      setPulseDrug(null);
      setHighlightDrug(null);
    }, 2500);

    return () => {
      clearTimeout(scrollTimer);
      clearTimeout(clearTimer);
    };
  }, [highlightDrug, candidates, setHighlightDrug]);

  // Land on the Rx segment when returning from a just-signed prescription.
  useEffect(() => {
    if (params.view === 'rx') setSegment('rx');
  }, [params.view, params.t]);

  if (!bundle) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyTitle}>No analysis yet</Text>
        <Text style={styles.emptyText}>Analyze a case first</Text>
        <Pressable
          style={styles.emptyBtn}
          onPress={() => router.push('/analyze')}
          accessibilityRole="button">
          <Text style={styles.emptyBtnText}>Go to Analyze</Text>
        </Pressable>
      </View>
    );
  }

  const patient = bundle.result?.patient ?? {};
  const flags = patient.flags ?? [];
  const demo = [
    patient.age != null ? `${patient.age} y` : null,
    patient.sex || null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <View style={styles.screen}>
      {/* Patient summary strip — one line (title truncates, age/sex always show) */}
      <View style={styles.summary}>
        <View style={styles.summaryTextRow}>
          <Text style={styles.summaryTitle} numberOfLines={1}>
            {bundle.result?.indication_label ?? bundle.indication}
          </Text>
          {!!demo && <Text style={styles.summaryDemo}>{demo}</Text>}
        </View>
        {flags.length > 0 && (
          <View style={styles.flagWrap}>
            {flags.map((f, i) => (
              <View key={`${f}-${i}`} style={styles.flagPill}>
                <Text style={styles.flagText}>{f}</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Segmented control — the views clinicians act on */}
      <View style={styles.segmentBar}>
        {PRIMARY_SEGMENTS.map((s) => {
          const active = segment === s.key;
          return (
            <Pressable
              key={s.key}
              onPress={() => setSegment(s.key)}
              style={[styles.segmentBtn, active && styles.segmentBtnActive]}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}>
              <Text
                style={[
                  styles.segmentLabel,
                  active && styles.segmentLabelActive,
                ]}>
                {s.label}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* Reference views, demoted — tinted chips; tap toggles in, tap again returns to Drugs */}
      <View style={styles.referenceRow}>
        {REFERENCE_VIEWS.map((s) => {
          const active = segment === s.key;
          return (
            <Pressable
              key={s.key}
              onPress={() => setSegment(active ? 'drugs' : s.key)}
              hitSlop={6}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              style={({ pressed }) => [
                styles.referenceChip,
                active && styles.referenceChipActive,
                pressed && { opacity: 0.82 },
              ]}>
              <Ionicons name={s.icon} size={14} color={active ? '#FFFFFF' : colors.accent} />
              <Text
                style={[
                  styles.referenceChipText,
                  active && styles.referenceChipTextActive,
                ]}>
                {s.label}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* Content */}
      {segment === 'drugs' && (
        <FlatList
          ref={listRef}
          data={candidates}
          keyExtractor={(item, i) => `${item.drug}-${i}`}
          contentContainerStyle={styles.listContent}
          onScrollToIndexFailed={(info) => {
            setTimeout(() => {
              listRef.current?.scrollToIndex({
                index: info.index,
                animated: true,
                viewPosition: 0.1,
              });
            }, 300);
          }}
          renderItem={({ item, index }) => (
            <DrugCard
              candidate={item}
              rank={index + 1}
              highlighted={pulseDrug === item.drug?.toLowerCase()}
              mechanismOnly={!!bundle?.result?.mechanism_only}
              disease={bundle?.disease_model}
              onPrescribe={(cand) =>
                router.push({
                  pathname: '/prescribe',
                  params: { drug: cand.drug, patientId: current?.id ?? '' },
                })
              }
            />
          )}
          ListEmptyComponent={
            <Text style={styles.noData}>
              No drug candidates in this analysis.
            </Text>
          }
          ListFooterComponent={
            bundle?.result?._disclaimer ? (
              <Text style={styles.disclaimer}>{bundle.result._disclaimer}</Text>
            ) : null
          }
        />
      )}

      {segment === 'graph' && (
        <View style={styles.graphWrap}>
          <GraphWebView
            graph={bundle.graph}
            onDrugTap={(drug) => setHighlightDrug(drug)}
          />
        </View>
      )}

      {segment === 'disease' && (
        <ScrollView contentContainerStyle={styles.listContent}>
          <DiseaseModelPanel model={bundle.disease_model} />
        </ScrollView>
      )}

      {segment === 'rx' && <RxPanel />}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  // Empty state
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.bg,
    padding: spacing.xl,
    gap: spacing.sm,
  },
  emptyTitle: {
    fontFamily: fonts.headingBold,
    fontSize: 19,
    color: colors.text,
  },
  emptyText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.textSecondary,
  },
  emptyBtn: {
    marginTop: spacing.md,
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.xl,
    paddingVertical: 12,
  },
  emptyBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  // Summary strip
  summary: {
    backgroundColor: colors.bgCard,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.borderSolid,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    gap: 6,
  },
  summaryTextRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: spacing.sm,
  },
  summaryTitle: {
    flexShrink: 1,
    fontFamily: fonts.heading,
    fontSize: 14.5,
    color: colors.text,
  },
  summaryDemo: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textMuted,
  },
  flagWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
  },
  flagPill: {
    backgroundColor: 'rgba(220,38,38,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(220,38,38,0.25)',
    borderRadius: radius.pill,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  flagText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 11,
    color: colors.red,
  },
  // Segmented control (mirrors web .reasoning-tabs)
  segmentBar: {
    flexDirection: 'row',
    backgroundColor: colors.bgWarm,
    borderRadius: radius.sm + 2,
    marginHorizontal: spacing.md,
    marginTop: spacing.md,
    padding: 3,
    gap: 3,
  },
  // Demoted reference views — tinted chips (CasePresetChips language, one tier
  // below the segmented control: smaller type, no shadow, tinted fill)
  referenceRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  referenceChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    minHeight: 34,
    paddingHorizontal: spacing.md,
    paddingVertical: 7,
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
  },
  referenceChipActive: {
    backgroundColor: colors.accent,
  },
  referenceChipText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 12.5,
    color: colors.accent,
  },
  referenceChipTextActive: {
    color: '#FFFFFF',
  },
  segmentBtn: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
    borderRadius: radius.sm,
  },
  segmentBtnActive: {
    backgroundColor: colors.bgCard,
    ...shadow.sm,
  },
  segmentLabel: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13.5,
    color: colors.textSecondary,
  },
  segmentLabelActive: {
    fontFamily: fonts.bodySemiBold,
    color: colors.accent,
  },
  // Content
  listContent: {
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.xl,
  },
  noData: {
    fontFamily: fonts.body,
    fontSize: 13.5,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: spacing.xl,
  },
  disclaimer: {
    fontFamily: fonts.body,
    fontSize: 11.5,
    lineHeight: 17,
    color: colors.textMuted,
    fontStyle: 'italic',
    marginTop: spacing.sm,
    marginBottom: spacing.xl,
  },
  graphWrap: {
    flex: 1,
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    borderRadius: radius.card,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderSolid,
    backgroundColor: colors.bg,
  },
});
