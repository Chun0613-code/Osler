/**
 * useVoiceCapture — tap-to-dictate voice input for the Analyze case note.
 *
 * Records a clip with expo-audio, uploads it to /api/voice/transcribe, and hands
 * the transcript back via `onTranscript`. It does NOT run Analyze or adopt any
 * value — the clinician reviews/edits the text in the case note first. Tap to
 * start, tap to stop; auto-stops at 60s.
 *
 * (Ported from the Glossa mobile `useVoiceInput` hook. On the iOS simulator the
 * native on-device recognizer can't init, so we record here and transcribe in the
 * cloud — recording works on the simulator.)
 */
import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio';
import { useCallback, useEffect, useRef, useState } from 'react';

import { transcribeVoice } from '@/api/osler';

const MAX_DURATION_S = 60;

export type VoiceStatus = 'idle' | 'recording' | 'transcribing';

export interface UseVoiceCaptureReturn {
  status: VoiceStatus;
  /** seconds recorded so far (0 when idle) */
  duration: number;
  /** start recording (asks mic permission on first use) */
  start: () => Promise<void>;
  /** stop + transcribe; the transcript arrives via onTranscript */
  stop: () => void;
  /** discard the current recording without transcribing */
  cancel: () => void;
  error: string | null;
}

export function useVoiceCapture(
  onTranscript: (text: string) => void,
): UseVoiceCaptureReturn {
  const [status, setStatus] = useState<VoiceStatus>('idle');
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cancelledRef = useRef(false);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    cancelledRef.current = false;
    try {
      const perm = await requestRecordingPermissionsAsync();
      if (!perm.granted) {
        setError('Microphone permission denied');
        return;
      }
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await recorder.prepareToRecordAsync();
      recorder.record();
      setStatus('recording');
      setDuration(0);
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
    } catch {
      setError('Could not start recording');
      setStatus('idle');
    }
  }, [recorder]);

  const finishRecording = useCallback(
    async (doTranscribe: boolean) => {
      clearTimer();
      try {
        await recorder.stop();
      } catch {
        /* already stopped */
      }
      setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true }).catch(() => {});
      const uri = recorder.uri;
      if (!doTranscribe || cancelledRef.current || !uri) {
        setStatus('idle');
        setDuration(0);
        return;
      }
      setStatus('transcribing');
      try {
        const text = await transcribeVoice(uri, 'm4a');
        if (text) onTranscript(text);
        else setError('No speech detected');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Transcription failed');
      } finally {
        setStatus('idle');
        setDuration(0);
      }
    },
    [recorder, onTranscript, clearTimer],
  );

  const stop = useCallback(() => {
    cancelledRef.current = false;
    void finishRecording(true);
  }, [finishRecording]);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    void finishRecording(false);
  }, [finishRecording]);

  // auto-stop at the cap
  useEffect(() => {
    if (status === 'recording' && duration >= MAX_DURATION_S) stop();
  }, [status, duration, stop]);

  // cleanup on unmount — guard every recorder access; at teardown the native
  // shared object may already be released and reading it would throw.
  useEffect(() => {
    return () => {
      clearTimer();
      try {
        if (recorder.isRecording) recorder.stop().catch(() => {});
      } catch {
        /* recorder's native object already gone — nothing to stop */
      }
    };
  }, [recorder, clearTimer]);

  return { status, duration, start, stop, cancel, error };
}
