/**
 * VoiceDictation — tap-to-dictate control for the Analyze case note.
 *
 * Idle: a mic button. Recording: navy "Listening…" card with a timer, a Stop
 * button, and an animated waveform (per the product mockup). Transcribing: a
 * spinner. The transcript is handed up via `onTranscript`; the clinician reviews
 * and edits it in the case note before Analyze — nothing is adopted or prescribed
 * from voice alone.
 *
 * v1 is record-then-transcribe (the transcript appears when you stop), not live
 * word-by-word streaming — matching the confirmed MVP. The waveform is decorative.
 */
import React, { useEffect, useRef } from 'react';
import {
  ActivityIndicator,
  Animated,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { useVoiceCapture } from '@/hooks/useVoiceCapture';
import { colors, fonts, radius, spacing } from '@/theme/tokens';

function fmtTime(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

const BAR_COUNT = 13;

function Waveform() {
  // Decorative "listening" pulse — a row of bars looping at staggered phases.
  const bars = useRef(
    Array.from({ length: BAR_COUNT }, () => new Animated.Value(0.35)),
  ).current;

  useEffect(() => {
    const loops = bars.map((v, i) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(v, {
            toValue: 1,
            duration: 260 + (i % 5) * 80,
            delay: i * 45,
            useNativeDriver: true,
          }),
          Animated.timing(v, {
            toValue: 0.35,
            duration: 260 + (i % 5) * 80,
            useNativeDriver: true,
          }),
        ]),
      ),
    );
    loops.forEach((l) => l.start());
    return () => loops.forEach((l) => l.stop());
  }, [bars]);

  return (
    <View style={styles.waveform}>
      {bars.map((v, i) => (
        <Animated.View
          key={i}
          style={[
            styles.waveBar,
            {
              backgroundColor: i % 3 === 0 ? colors.orange : '#6E86AE',
              transform: [{ scaleY: v }],
            },
          ]}
        />
      ))}
    </View>
  );
}

export default function VoiceDictation({
  onTranscript,
}: {
  onTranscript: (text: string) => void;
}) {
  const voice = useVoiceCapture(onTranscript);

  if (voice.status === 'idle') {
    return (
      <View style={styles.idleWrap}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Dictate case note"
          onPress={voice.start}
          style={({ pressed }) => [styles.micBtn, pressed && { opacity: 0.85 }]}>
          <Text style={styles.micText}>Dictate case note</Text>
        </Pressable>
        {!!voice.error && <Text style={styles.error}>{voice.error}</Text>}
      </View>
    );
  }

  const recording = voice.status === 'recording';
  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <View style={styles.headerLeft}>
          <View style={styles.recDot} />
          <Text style={styles.headerText}>{recording ? 'Listening…' : 'Transcribing…'}</Text>
        </View>
        {recording && <Text style={styles.timer}>{fmtTime(voice.duration)}</Text>}
      </View>

      {recording ? (
        <View style={styles.recRow}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Stop recording"
            onPress={voice.stop}
            style={({ pressed }) => [styles.stopBtn, pressed && { opacity: 0.85 }]}>
            <View style={styles.stopSquare} />
          </Pressable>
          <Waveform />
        </View>
      ) : (
        <View style={styles.transcribingRow}>
          <ActivityIndicator color="#FFFFFF" />
          <Text style={styles.transcribingText}>Converting speech to text…</Text>
        </View>
      )}

      <Text style={styles.hint}>
        You can edit the transcript before Analyze — nothing is prescribed from voice.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  idleWrap: { marginBottom: spacing.sm },
  micBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    minHeight: 46,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.orange,
    backgroundColor: '#FFF7F4',
  },
  micText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.orange,
  },
  error: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: colors.red,
    marginTop: 6,
  },
  card: {
    backgroundColor: colors.accent,
    borderRadius: radius.card,
    padding: spacing.md,
    marginBottom: spacing.sm,
    gap: spacing.md,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  recDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.orange,
  },
  headerText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  timer: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.traceMuted,
    fontVariant: ['tabular-nums'],
  },
  recRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  stopBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.orange,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopSquare: {
    width: 15,
    height: 15,
    borderRadius: 3,
    backgroundColor: '#FFFFFF',
  },
  waveform: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    height: 40,
    gap: 4,
  },
  waveBar: {
    flex: 1,
    height: 34,
    borderRadius: 3,
  },
  transcribingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: 6,
  },
  transcribingText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13.5,
    color: '#FFFFFF',
  },
  hint: {
    fontFamily: fonts.body,
    fontSize: 11.5,
    lineHeight: 16,
    color: colors.traceMuted,
  },
});
