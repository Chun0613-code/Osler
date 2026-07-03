"""
transcribe.py — speech-to-text for voice case-note dictation.

The mobile app records a short clip (expo-audio) and uploads it; this turns that
audio into text, which the clinician reviews before it ever reaches the symbolic
engine. It never prescribes or decides anything — it only produces a transcript.

v1 provider = Gemini (google-genai, same SDK/key as llm_client). Provider is
abstracted (VOICE_STT_PROVIDER) so Deepgram (streaming + medical vocabulary) can
be swapped in with a small change later.

PHI note: cloud transcription sends patient audio to Google. Acceptable for the
sandbox demo (no real patients). For production, route to a HIPAA-BAA provider or
use on-device recognition on a real device.

Key: GEMINI_API_KEY (shared with llm_client). transcribe_available() lets the
route surface a friendly "voice not configured" message instead of a 500.
"""
from __future__ import annotations

import os
from typing import Optional

# Client-declared format → MIME. Gemini accepts a range of audio containers;
# expo-audio's HIGH_QUALITY preset yields .m4a (AAC in an MP4 container) on iOS.
_MIME = {
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "aac": "audio/aac",
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
    "caf": "audio/mp4",
}

# Verbatim clinical dictation — numbers/vitals/drug names must survive exactly,
# because they flow into the same safety gate as typed input (a mis-heard "15"
# vs "50" changes dosing/gating). Downstream the app still highlights every
# number/drug for one-tap confirmation; nothing is adopted silently.
_PROMPT = (
    "You are transcribing a clinician's spoken patient case note. "
    "Transcribe verbatim, in English. Preserve numbers, vital signs, drug names "
    "and dosages exactly as spoken (e.g. 'BP 88 over 54', 'heart rate 112', "
    "'eGFR 72', 'aspirin 81 milligrams'). Do not paraphrase, summarize, or add "
    "clinical interpretation. Output ONLY the transcript text — no preamble, no "
    "quotation marks, no commentary."
)

# Model names drift; try current ones in order (env override wins). Mirrors
# llm_client._GEMINI_MODELS so both stay on the same working model.
_GEMINI_MODELS = [
    m for m in [
        os.environ.get("GEMINI_MODEL"),
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash-001",
    ] if m
]


def _provider() -> str:
    return (os.environ.get("VOICE_STT_PROVIDER") or "gemini").strip().lower()


def transcribe_available() -> bool:
    """Whether voice input can be transcribed with the current config."""
    p = _provider()
    if p == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if p == "deepgram":
        return bool(os.environ.get("DEEPGRAM_API_KEY"))
    return False


def transcribe_audio(audio_bytes: bytes, audio_format: str = "m4a",
                     api_key: Optional[str] = None) -> str:
    """Transcribe recorded audio → text. Raises on hard failure so the route can
    turn it into a 4xx/5xx with a message."""
    if not audio_bytes:
        raise ValueError("empty audio")
    p = _provider()
    if p == "gemini":
        return _gemini_transcribe(audio_bytes, audio_format, api_key)
    if p == "deepgram":
        return _deepgram_transcribe(audio_bytes, audio_format)
    raise RuntimeError(f"unknown VOICE_STT_PROVIDER: {p}")


def _gemini_transcribe(audio_bytes: bytes, audio_format: str,
                       api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    fmt = (audio_format or "m4a").lower().lstrip(".")
    mime = _MIME.get(fmt, "audio/mp4")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)

    last: Optional[Exception] = None
    for model_name in _GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[audio_part, _PROMPT],
            )
            return (resp.text or "").strip()
        except Exception as e:  # try next model on not-found, else surface
            last = e
            s = str(e).lower()
            if "not_found" in s or "not found" in s or "404" in s:
                continue
            raise
    raise last or RuntimeError("no Gemini model available")


def _deepgram_transcribe(audio_bytes: bytes, audio_format: str) -> str:
    # Deferred: Deepgram medical (nova-2-medical) gives better drug/dose accuracy
    # and supports live streaming. Swap-in point kept intentionally small.
    raise NotImplementedError(
        "Deepgram STT not wired yet — set VOICE_STT_PROVIDER=gemini"
    )
