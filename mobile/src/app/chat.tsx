/**
 * Chat screen — ask the agent about the CURRENT patient's result.
 * Mirrors the web demo's center column: trace card up top, user/AI bubbles,
 * suggested-question chips, pinned composer.
 */
import Ionicons from '@expo/vector-icons/Ionicons';
import { Link, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { chat, type ChatMessage } from '@/api/osler';
import MessageBubble from '@/components/MessageBubble';
import TraceCard from '@/components/TraceCard';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

function TypingBubble() {
  const pulse = useRef(new Animated.Value(0.35)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 500, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0.35, duration: 500, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [pulse]);
  return (
    <View style={styles.typingBubble}>
      <Animated.Text style={[styles.typingDots, { opacity: pulse }]}>…</Animated.Text>
    </View>
  );
}

export default function ChatScreen() {
  const router = useRouter();
  const { patients, current, currentId, selectPatient, appendChat, llm } = useApp();
  const [input, setInput] = useState('');
  const [sendingForId, setSendingForId] = useState<string | null>(null);
  const [showSettingsHint, setShowSettingsHint] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const listRef = useRef<FlatList<ChatMessage>>(null);

  const sending = sendingForId === current?.id;
  const isSendingAny = sendingForId !== null;
  const chatLen = current?.chat.length ?? 0;
  useEffect(() => {
    if (chatLen > 0) {
      // small delay so the new row has rendered before we scroll
      const t = setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 80);
      return () => clearTimeout(t);
    }
    return undefined;
  }, [chatLen, sending]);

  const suggestions = useMemo(() => {
    if (!current) return [];
    const cands = current.bundle.result?.candidates ?? [];
    const out: string[] = [];
    const first = cands[0];
    if (first) out.push(`Why is ${first.drug} first?`);
    const avoided = cands.find((c) => {
      const d = (c.safety?.decision ?? '').toLowerCase();
      return d === 'avoid' || d === 'block';
    });
    if (avoided) out.push(`Why is ${avoided.drug} avoided?`);
    out.push('What is the disease doing to the body?');
    return out;
  }, [current]);

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || !current || isSendingAny) return;
      const patient = current;
      setInput('');
      setSendingForId(patient.id);
      const userMsg: ChatMessage = { role: 'user', content };
      appendChat(patient.id, userMsg);
      try {
        const res = await chat(
          patient.id,
          [...patient.chat, userMsg],
          patient.bundle,
          llm,
        );
        appendChat(patient.id, { role: 'assistant', content: res.reply });
        if (res.ok === false) setShowSettingsHint(true);
      } catch (e) {
        appendChat(patient.id, {
          role: 'assistant',
          content: `Could not reach the backend (${e instanceof Error ? e.message : 'network error'}). Is \`python demo/demo_app.py\` running?`,
        });
      } finally {
        setSendingForId(null);
      }
    },
    [current, isSendingAny, appendChat, llm],
  );

  if (!current) {
    return (
      <View style={styles.empty}>
        <Ionicons name="chatbubble-ellipses-outline" size={40} color={colors.silverLight} />
        <Text style={styles.emptyText}>
          Analyze a case first so I can explain its result.
        </Text>
        <Pressable
          accessibilityRole="button"
          onPress={() => router.push('/analyze')}
          style={({ pressed }) => [styles.emptyButton, pressed && { opacity: 0.85 }]}>
          <Text style={styles.emptyButtonText}>Go to Analyze</Text>
        </Pressable>
      </View>
    );
  }

  const showChips = current.chat.length === 0 && !sending;

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}>
      {!llm.apiKey && !bannerDismissed && (
        <View style={styles.banner}>
          <Text style={styles.bannerText}>
            No LLM key set — chat will return a fallback notice. Add one in{' '}
            <Link href="/settings" style={styles.bannerLink}>
              Settings
            </Link>
            .
          </Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Dismiss"
            onPress={() => setBannerDismissed(true)}
            hitSlop={12}
            style={styles.bannerClose}>
            <Ionicons name="close" size={16} color={colors.textSecondary} />
          </Pressable>
        </View>
      )}

      <FlatList
        ref={listRef}
        data={current.chat}
        keyExtractor={(_, i) => String(i)}
        renderItem={({ item }) => <MessageBubble message={item} />}
        contentContainerStyle={styles.listContent}
        keyboardShouldPersistTaps="handled"
        ListHeaderComponent={
          <View style={styles.listHeader}>
            <View style={styles.patientSelector}>
              <Text style={styles.sectionLabel}>CHAT WITH</Text>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.patientScroll}
                keyboardShouldPersistTaps="handled">
                {patients.map((patient, index) => {
                  const active = patient.id === currentId;
                  const result = patient.bundle.result;
                  const drugCount = result.candidates.length;
                  const unreadHint = patient.chat.length > 0 ? `${patient.chat.length}` : 'New';
                  return (
                    <Pressable
                      key={patient.id}
                      accessibilityRole="button"
                      accessibilityState={{ selected: active, disabled: isSendingAny && !active }}
                      disabled={isSendingAny && !active}
                      onPress={() => selectPatient(patient.id)}
                      style={({ pressed }) => [
                        styles.patientPill,
                        active && styles.patientPillActive,
                        isSendingAny && !active && { opacity: 0.55 },
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
                          {drugCount} drug{drugCount === 1 ? '' : 's'} · {unreadHint}
                        </Text>
                      </View>
                    </Pressable>
                  );
                })}
              </ScrollView>
            </View>
            <TraceCard steps={current.bundle.trace} />
            {showChips && (
              <View style={styles.chips}>
                {suggestions.map((q) => (
                  <Pressable
                    key={q}
                    accessibilityRole="button"
                    onPress={() => send(q)}
                    style={({ pressed }) => [styles.chip, pressed && { opacity: 0.8 }]}>
                    <Text style={styles.chipText}>{q}</Text>
                  </Pressable>
                ))}
              </View>
            )}
          </View>
        }
        ListFooterComponent={
          <View>
            {sending && <TypingBubble />}
            {showSettingsHint && (
              <Link href="/settings" style={styles.hint}>
                Add an API key in Settings to enable full chat →
              </Link>
            )}
          </View>
        }
      />

      <View style={styles.composer}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder={`Ask about ${current.title}…`}
          placeholderTextColor={colors.textMuted}
          returnKeyType="send"
          onSubmitEditing={() => send(input)}
          editable={!isSendingAny}
          multiline={false}
        />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Send message"
          onPress={() => send(input)}
          disabled={isSendingAny || !input.trim()}
          style={({ pressed }) => [
            styles.sendButton,
            (isSendingAny || !input.trim()) && { opacity: 0.5 },
            pressed && { opacity: 0.8 },
          ]}>
          <Ionicons name="send" size={18} color="#FFFFFF" />
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.lg,
    padding: spacing.xl,
    backgroundColor: colors.bg,
  },
  emptyText: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 21,
    color: colors.textSecondary,
    textAlign: 'center',
    maxWidth: 280,
  },
  emptyButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 24,
    minHeight: 44,
    justifyContent: 'center',
  },
  emptyButtonText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.accentLight,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  bannerText: {
    flex: 1,
    fontFamily: fonts.body,
    fontSize: 12,
    lineHeight: 17,
    color: colors.textSecondary,
  },
  bannerLink: {
    fontFamily: fonts.bodySemiBold,
    color: colors.accent,
    textDecorationLine: 'underline',
  },
  bannerClose: {
    minWidth: 28,
    minHeight: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  listContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  listHeader: {
    marginBottom: spacing.sm,
    gap: spacing.md,
  },
  patientSelector: {
    gap: spacing.xs,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  patientScroll: {
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
  chips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 7,
  },
  chip: {
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 12,
    minHeight: 44,
    justifyContent: 'center',
  },
  chipText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: colors.accent,
  },
  typingBubble: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    borderBottomRightRadius: 12,
    borderBottomLeftRadius: 0,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginVertical: 6,
    alignSelf: 'flex-start',
    ...shadow.sm,
  },
  typingDots: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 18,
    color: colors.textMuted,
    letterSpacing: 2,
  },
  hint: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.accent,
    textDecorationLine: 'underline',
    marginTop: spacing.xs,
    marginLeft: 2,
  },
  composer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.borderSolid,
    backgroundColor: colors.bgCard,
  },
  input: {
    flex: 1,
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.pill,
    paddingHorizontal: 16,
    paddingVertical: 0,
    minHeight: 44,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
