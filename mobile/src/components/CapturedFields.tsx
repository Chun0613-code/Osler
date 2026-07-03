/**
 * CapturedFields — the "structure & confirm" step of voice case-note entry (stage 3B).
 *
 * After a dictated note is transcribed, /api/parse_case previews the SAME parse the
 * Analyze pipeline runs server-side. This card surfaces it for the clinician:
 *   • CAPTURED — structured value chips (indication / age / sex / eGFR / …). Each starts
 *     "pending" (amber, dashed); tapping it CONFIRMS the value into the Case-details form.
 *   • VITALS — the engine's own amber caution flags (hypotension, tachycardia, …), read
 *     only. They come straight from PatientProfile.flags(), so what you see here is what
 *     the safety layer will act on.
 *
 * Safety: nothing is adopted automatically and nothing is prescribed from voice — the
 * clinician confirms every value, then reviews the filled form before Analyze.
 */
import React, { useMemo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { ParseCaseResult } from '@/api/osler';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

/** Fields the clinician can confirm into the Case-details form (mirrors ManualCaseFields). */
export type CapturedKey =
  | 'indication'
  | 'age'
  | 'sex'
  | 'egfr'
  | 'weight_kg'
  | 'allergies'
  | 'meds';

interface Chip {
  key: CapturedKey;
  label: string;
}

const FLAG_LABEL: Record<string, string> = {
  hypotension: 'Hypotension',
  tachycardia: 'Tachycardia',
  hypoxia: 'Hypoxia',
  hyperkalemia: 'Hyperkalemia',
  renal_dose_review: 'Renal dose review',
  hepatic_impairment: 'Hepatic impairment',
  pregnancy: 'Pregnancy',
};

function humanizeFlag(flag: string): string {
  return (
    FLAG_LABEL[flag] ??
    flag.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
  );
}

/** Which fields the parser captured — a chip is shown, and can be confirmed, for each.
 * Exported so analyze.tsx's "Confirm all" confirms exactly the chips rendered here. */
export function capturedKeys(parsed: ParseCaseResult): CapturedKey[] {
  const f = parsed.fields;
  const keys: CapturedKey[] = [];
  if (f.indication) keys.push('indication');
  if (f.age != null) keys.push('age');
  if (f.sex) keys.push('sex');
  if (f.egfr != null) keys.push('egfr');
  if (f.weight_kg != null) keys.push('weight_kg');
  if (f.allergies?.length) keys.push('allergies');
  if (f.current_medications?.length) keys.push('meds');
  return keys;
}

function chipLabel(key: CapturedKey, f: ParseCaseResult['fields']): string {
  switch (key) {
    case 'indication':
      return `Dx · ${f.indication}`;
    case 'age':
      return `Age ${f.age}`;
    case 'sex':
      return `Sex ${f.sex}`;
    case 'egfr':
      return `eGFR ${f.egfr}`;
    case 'weight_kg':
      return `${f.weight_kg} kg`;
    case 'allergies':
      return `Allergies · ${(f.allergies ?? []).join(', ')}`;
    case 'meds':
      return `Meds · ${(f.current_medications ?? []).join(', ')}`;
  }
}

export default function CapturedFields({
  parsed,
  confirmed,
  onConfirm,
  onConfirmAll,
}: {
  parsed: ParseCaseResult;
  confirmed: Set<CapturedKey>;
  onConfirm: (key: CapturedKey) => void;
  onConfirmAll: () => void;
}) {
  const chips = useMemo(
    () => capturedKeys(parsed).map((key) => ({ key, label: chipLabel(key, parsed.fields) })),
    [parsed],
  );
  const flags = parsed.flags ?? [];
  const pendingKeys = chips.filter((c) => !confirmed.has(c.key)).map((c) => c.key);
  const indicationIssue = !parsed.fields.indication
    ? 'empty'
    : !parsed.indication_known
      ? 'unmappable'
      : null;

  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <View style={styles.headerLeft}>
          <Text style={styles.sectionLabel}>CAPTURED</Text>
          <View style={styles.parserBadge}>
            <Text style={styles.parserBadgeText}>
              {parsed.parser === 'llm' ? 'AI' : 'RULES'}
            </Text>
          </View>
        </View>
        {pendingKeys.length > 0 && (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Confirm all captured values"
            onPress={onConfirmAll}
            hitSlop={8}
            style={({ pressed }) => [styles.confirmAll, pressed && { opacity: 0.7 }]}>
            <Text style={styles.confirmAllText}>Confirm all</Text>
          </Pressable>
        )}
      </View>

      {chips.length > 0 ? (
        <>
          <Text style={styles.subtitle}>
            Tap a value to confirm — nothing is adopted automatically.
          </Text>
          <View style={styles.chipWrap}>
            {chips.map((c) => {
              const isConfirmed = confirmed.has(c.key);
              return (
                <Pressable
                  key={c.key}
                  accessibilityRole="button"
                  accessibilityLabel={`${isConfirmed ? 'Confirmed' : 'Confirm'} ${c.label}`}
                  accessibilityState={{ selected: isConfirmed }}
                  onPress={() => onConfirm(c.key)}
                  disabled={isConfirmed}
                  style={({ pressed }) => [
                    styles.chip,
                    isConfirmed ? styles.chipConfirmed : styles.chipPending,
                    pressed && !isConfirmed && { opacity: 0.75 },
                  ]}>
                  <Text
                    style={[
                      styles.chipText,
                      isConfirmed ? styles.chipTextConfirmed : styles.chipTextPending,
                    ]}>
                    {isConfirmed ? '✓ ' : ''}
                    {c.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </>
      ) : (
        <Text style={styles.subtitle}>No structured values captured from this note.</Text>
      )}

      {flags.length > 0 && (
        <View style={styles.vitalsBlock}>
          <Text style={styles.vitalsLabel}>VITALS · AUTO-FLAGGED BY THE ENGINE</Text>
          {flags.map((fl, i) => (
            <View key={`${fl.flag}-${i}`} style={styles.flagRow}>
              <Text style={styles.flagIcon}>⚠️</Text>
              <Text style={styles.flagText}>
                <Text style={styles.flagName}>{humanizeFlag(fl.flag)}</Text>
                {fl.detail ? `  ·  ${fl.detail}` : ''}
              </Text>
            </View>
          ))}
        </View>
      )}

      {indicationIssue && (
        <View style={styles.indicationNote}>
          <Text style={styles.indicationNoteText}>
            {indicationIssue === 'empty'
              ? 'No indication detected — add one in Case details so the engine can map treatment targets.'
              : `“${parsed.fields.indication}” isn’t a mappable indication yet — edit it in Case details before Analyze.`}
          </Text>
        </View>
      )}

      <Text style={styles.footer}>
        Nothing is prescribed from voice alone — review every value, then Analyze.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.card,
    padding: spacing.md,
    marginBottom: spacing.sm,
    gap: spacing.sm,
    ...shadow.sm,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.4,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  parserBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: radius.pill,
    backgroundColor: colors.accentLight,
  },
  parserBadgeText: {
    fontFamily: fonts.heading,
    fontSize: 8.5,
    letterSpacing: 0.8,
    color: colors.accent,
  },
  confirmAll: {
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.orange,
    backgroundColor: '#FFF7F4',
  },
  confirmAllText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 12,
    color: colors.orange,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 12,
    lineHeight: 17,
    color: colors.textSecondary,
  },
  chipWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    minHeight: 34,
    paddingHorizontal: spacing.md,
    paddingVertical: 7,
    borderRadius: radius.pill,
    justifyContent: 'center',
  },
  chipPending: {
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: colors.amber,
    backgroundColor: '#FFFBF3',
  },
  chipConfirmed: {
    borderWidth: 1,
    borderColor: colors.accent,
    backgroundColor: colors.accent,
  },
  chipText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 12.5,
  },
  chipTextPending: { color: colors.amber },
  chipTextConfirmed: { color: '#FFFFFF' },
  vitalsBlock: {
    gap: 6,
    marginTop: 2,
  },
  vitalsLabel: {
    fontFamily: fonts.heading,
    fontSize: 10,
    letterSpacing: 1.1,
    color: colors.amber,
    textTransform: 'uppercase',
  },
  flagRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#FEF3C7',
    borderWidth: 1,
    borderColor: colors.amber,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: 9,
  },
  flagIcon: { fontSize: 13 },
  flagText: {
    flex: 1,
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: '#92400E',
  },
  flagName: { fontFamily: fonts.bodySemiBold },
  indicationNote: {
    backgroundColor: '#FFF7ED',
    borderLeftWidth: 3,
    borderLeftColor: colors.orange,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: 9,
  },
  indicationNoteText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12,
    lineHeight: 17,
    color: '#9A3412',
  },
  footer: {
    fontFamily: fonts.body,
    fontSize: 11,
    lineHeight: 15,
    color: colors.textMuted,
    marginTop: 2,
  },
});
