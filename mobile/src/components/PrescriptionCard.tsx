/**
 * PrescriptionCard — one e-prescription: drug, sig, dispense, pharmacy, status.
 * Left border + badge colored by status (draft/sent/filled/error).
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import type { Prescription } from '@/api/osler';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

const STATUS_COLOR: Record<string, string> = {
  draft: colors.silver,
  sent: colors.green,
  filled: colors.teal,
  error: colors.red,
};

export default function PrescriptionCard({ rx }: { rx: Prescription }) {
  const color = STATUS_COLOR[rx.status] ?? colors.gray;
  const qty =
    rx.dispense_quantity != null
      ? `Qty ${rx.dispense_quantity}${rx.dispense_unit ? ` ${rx.dispense_unit}` : ''}`
      : null;

  return (
    <View style={[styles.card, { borderLeftColor: color }]}>
      <View style={styles.head}>
        <Text style={styles.drug} numberOfLines={1}>
          {rx.drug}
        </Text>
        <View style={[styles.badge, { backgroundColor: color }]}>
          <Text style={styles.badgeText}>{rx.status}</Text>
        </View>
      </View>

      {!!rx.clinical_role && <Text style={styles.role}>{rx.clinical_role}</Text>}
      {!!rx.sig && <Text style={styles.sig}>{rx.sig}</Text>}

      <View style={styles.metaRow}>
        {!!qty && <Text style={styles.meta}>{qty}</Text>}
        {rx.days_supply != null && <Text style={styles.meta}>{rx.days_supply} d</Text>}
        {!!rx.refills && rx.refills > 0 && <Text style={styles.meta}>Refills {rx.refills}</Text>}
      </View>

      {!!rx.pharmacy?.name && <Text style={styles.pharmacy}>{rx.pharmacy.name}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.card,
    borderLeftWidth: 4,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderSolid,
    padding: spacing.lg,
    marginBottom: spacing.md,
    ...shadow.sm,
  },
  head: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  drug: {
    flex: 1,
    fontFamily: fonts.headingBold,
    fontSize: 17,
    color: colors.text,
    textTransform: 'capitalize',
  },
  badge: { borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 3 },
  badgeText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 10,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    color: '#FFFFFF',
  },
  role: { fontFamily: fonts.bodyMedium, fontSize: 12.5, color: colors.textMuted, marginTop: 2 },
  sig: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 20,
    color: colors.text,
    marginTop: spacing.sm,
  },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, marginTop: spacing.sm },
  meta: { fontFamily: fonts.bodyMedium, fontSize: 12.5, color: colors.textSecondary },
  pharmacy: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.teal,
    marginTop: spacing.sm,
  },
});
