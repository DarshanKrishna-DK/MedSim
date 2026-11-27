"""Gemini TTS synthesis: Google Gemini generateContent with AUDIO response modality.

Provides:
  - synthesize_speech(text, voice, model, effort) -> (wav_bytes, "audio/wav")
  - concat_audio_chunks(chunks) -> bytes
  - pick_voice_for_role(role_or_gender, case_id, lang) -> str
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import sys
import wave
from typing import Optional

import httpx

log = logging.getLogger("medsim.gemini_tts")

DEFAULT_TTS_MODEL = "gemini-3.8-flash-lite-tts"
FALLBACK_TTS_MODEL = "gemini-2.5-flash-preview-tts"

MALE_VOICES = [
    "Puck", "Charon", "Fenrir", "Orus", "Iapetus",
    "Enceladus", "Algenib", "Rasalgethi", "Alnilam", "Schedar", "Achird"
]
FEMALE_VOICES = [
    "Aoede", "Kore", "Leda", "Zephyr", "Callirrhoe",
    "Autonoe", "Umbriel", "Algieba", "Despina", "Erinome", "Sulafat"
]


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key and "unittest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ:
        # Check .env.local if present
        env_path = os.path.join(os.path.dirname(__file__), ".env.local")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        key = line.strip().split("=", 1)[1].strip()
                        break
                    if line.startswith("GOOGLE_API_KEY=") and not key:
                        key = line.strip().split("=", 1)[1].strip()
    return key


def pick_voice_for_role(role_or_gender: str, case_id: Optional[str] = None, lang: Optional[str] = "en") -> str:
    """Map role or gender to a valid Gemini prebuilt voice name."""
    r = (role_or_gender or "nurse").lower().strip()
    cid = (case_id or "").lower().strip()
    if cid in ("viva-examiner", "viva") or r in ("viva", "examiner"):
        return os.environ.get("GEMINI_VOICE_VIVA", "").strip() or os.environ.get("MURF_VIVA_VOICE", "").strip() or "Charon"
    if r in ("nurse", "asha"):
        return os.environ.get("GEMINI_VOICE_NURSE", "").strip() or os.environ.get("MURF_NURSE_VOICE", "").strip() or "Aoede"
    if r in ("pharmacist", "rx"):
        return os.environ.get("GEMINI_VOICE_PHARMACIST", "").strip() or "Kore"
    if r in ("receptionist", "reception") or cid == "receptionist":
        return os.environ.get("GEMINI_VOICE_RECEPTIONIST", "").strip() or "Aoede"
    if cid == "teacher" or r == "teacher":
        return os.environ.get("GEMINI_VOICE_TEACHER", "").strip() or "Charon"

    is_female = r in ("f", "female", "woman")
    pool = FEMALE_VOICES if is_female else MALE_VOICES
    env_override = os.environ.get("GEMINI_VOICE_FEMALE" if is_female else "GEMINI_VOICE_MALE", "").strip()
    if env_override:
        return env_override
    if case_id:
        idx = int(hashlib.md5(f"{case_id}-{r}".encode()).hexdigest(), 16) % len(pool)
        return pool[idx]
    return pool[0]


def wrap_pcm_in_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw 16-bit mono PCM frames in a standard RIFF WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)
    return buf.getvalue()


def concat_audio_chunks(chunks: list[bytes]) -> bytes:
    """Combine multiple WAV chunks into a single valid RIFF WAV file."""
    if not chunks:
        return b""
    frames: list[bytes] = []
    framerate = 24000
    nchannels = 1
    sampwidth = 2
    for chunk in chunks:
        if not chunk:
            continue
        if chunk.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(chunk), "rb") as r:
                    framerate = r.getframerate()
                    nchannels = r.getnchannels()
                    sampwidth = r.getsampwidth()
                    frames.append(r.readframes(r.getnframes()))
            except Exception as e:
                log.warning("concat_audio_chunks: unparseable WAV chunk: %s", e)
                frames.append(chunk)
        else:
            frames.append(chunk)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


async def synthesize_speech(
    text: str,
    voice: str = "Aoede",
    model: str | None = None,
    effort: str = "low",
) -> tuple[bytes, str]:
    """Synthesize text to speech using Google Gemini generateContent with AUDIO response modality.

    Returns:
        tuple[bytes, str]: (wav_bytes, "audio/wav")
    """
    key = _get_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    primary_model = model or os.environ.get("GEMINI_TTS_MODEL", DEFAULT_TTS_MODEL)
    models_to_try = [primary_model]
    if primary_model != FALLBACK_TTS_MODEL:
        models_to_try.append(FALLBACK_TTS_MODEL)

    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for cur_model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent"
            headers = {
                "x-goog-api-key": key,
                "Content-Type": "application/json",
            }
            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": voice,
                            }
                        }
                    },
                },
            }
            try:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code != 200:
                    log.warning(
                        "Gemini TTS request to %s failed (%s): %s",
                        cur_model, res.status_code, res.text[:200]
                    )
                    continue

                res_data = res.json()
                candidates = res_data.get("candidates") or []
                if not candidates:
                    log.warning("Gemini TTS %s returned no candidates", cur_model)
                    continue

                part = candidates[0].get("content", {}).get("parts", [{}])[0]
                inline_data = part.get("inlineData")
                if not inline_data or not inline_data.get("data"):
                    log.warning("Gemini TTS %s returned empty inlineData", cur_model)
                    continue

                raw_bytes = base64.b64decode(inline_data["data"])
                mime_type = inline_data.get("mimeType", "").lower()

                # If RIFF WAV, validate header
                if ("wav" in mime_type or raw_bytes.startswith(b"RIFF")):
                    if raw_bytes.startswith(b"RIFF"):
                        return raw_bytes, "audio/wav"
                    # Wrap if corrupted/missing header
                    return wrap_pcm_in_wav(raw_bytes, sample_rate=24000), "audio/wav"

                # If headerless PCM (e.g. audio/L16;codec=pcm;rate=24000)
                rate = 24000
                if "rate=16000" in mime_type:
                    rate = 16000
                elif "rate=48000" in mime_type:
                    rate = 48000
                wav_bytes = wrap_pcm_in_wav(raw_bytes, sample_rate=rate)
                return wav_bytes, "audio/wav"

            except Exception as e:
                log.warning("Gemini TTS %s error: %s", cur_model, e)
                last_error = e
                continue

    raise RuntimeError(f"Gemini TTS failed across all models: {last_error or 'no audio returned'}")
