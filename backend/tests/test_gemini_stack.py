"""Dedicated tests for 100% Google Gemini stack components:
1. LLM reasoning effort clamping (strictly low/medium, never high/xhigh)
2. Gemini TTS audio framing and concatenation (valid RIFF WAV)
3. Gemini Live WebSocket configuration builder (omits thinking/affective/proactivity on gemini-3.8-live)
4. Gemini Observer WebSocket token guard and frame relay
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import wave
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

import gemini_live  # noqa: E402
import gemini_tts  # noqa: E402
import llm_provider  # noqa: E402
import observer  # noqa: E402
import server  # noqa: E402


class EffortClampingTests(unittest.TestCase):
    """Reasoning effort must ALWAYS be capped at low or medium — NEVER high or xhigh."""

    def test_high_and_xhigh_clamped_to_medium(self):
        self.assertEqual(llm_provider.clamp_effort("high"), "medium")
        self.assertEqual(llm_provider.clamp_effort("HIGH"), "medium")
        self.assertEqual(llm_provider.clamp_effort("xhigh"), "medium")
        self.assertEqual(llm_provider.clamp_effort("XHIGH"), "medium")
        self.assertEqual(llm_provider.clamp_effort("max"), "medium")

    def test_low_and_medium_preserved(self):
        self.assertEqual(llm_provider.clamp_effort("low"), "low")
        self.assertEqual(llm_provider.clamp_effort("LOW"), "low")
        self.assertEqual(llm_provider.clamp_effort("medium"), "medium")
        self.assertEqual(llm_provider.clamp_effort("MEDIUM"), "medium")

    def test_fallback_and_invalid_inputs(self):
        self.assertEqual(llm_provider.clamp_effort(None), "low")
        self.assertEqual(llm_provider.clamp_effort(""), "low")
        self.assertEqual(llm_provider.clamp_effort("ultra"), "low")
        self.assertEqual(llm_provider.clamp_effort(None, default="medium"), "medium")
        self.assertEqual(llm_provider.clamp_effort("bogus", default="medium"), "medium")

    def test_forbidden_values_never_returned(self):
        test_inputs = [
            "high", "xhigh", "HIGH", "XHIGH", "ultra", "extreme", "maximum",
            "super_high", "o1_high", "o3_high", "very_high",
        ]
        for inp in test_inputs:
            out = llm_provider.clamp_effort(inp)
            self.assertNotIn(out, ("high", "xhigh", "max", "ultra", "extreme"))
            self.assertIn(out, ("low", "medium"))


class GeminiTtsAudioTests(unittest.TestCase):
    """Gemini TTS generates RIFF WAV audio; helper functions must produce standard WAV headers."""

    def test_wrap_pcm_in_wav(self):
        # 100 samples of 16-bit mono PCM (200 bytes)
        raw_pcm = b"\x00\x00" * 100
        wav_bytes = gemini_tts.wrap_pcm_in_wav(raw_pcm, sample_rate=24000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVEfmt ", wav_bytes)

        # Inspect using python's wave module
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getnframes(), 100)
            self.assertEqual(w.readframes(100), raw_pcm)

    def test_concat_audio_chunks(self):
        # Create two separate WAV chunks
        pcm1 = b"\x01\x00" * 50
        pcm2 = b"\x02\x00" * 70
        chunk1 = gemini_tts.wrap_pcm_in_wav(pcm1, sample_rate=24000)
        chunk2 = gemini_tts.wrap_pcm_in_wav(pcm2, sample_rate=24000)

        combined = gemini_tts.concat_audio_chunks([chunk1, chunk2])
        self.assertTrue(combined.startswith(b"RIFF"))

        with wave.open(io.BytesIO(combined), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getnframes(), 120)
            self.assertEqual(w.readframes(120), pcm1 + pcm2)

    def test_concat_empty_and_single_chunk(self):
        self.assertEqual(gemini_tts.concat_audio_chunks([]), b"")
        pcm = b"\x05\x00" * 10
        chunk = gemini_tts.wrap_pcm_in_wav(pcm, sample_rate=24000)
        out = gemini_tts.concat_audio_chunks([chunk])
        self.assertTrue(out.startswith(b"RIFF"))


class GeminiLiveConfigTests(unittest.TestCase):
    """gemini-3.8-live config builder must strictly omit thinking_config,
    enable_affective_dialog, and proactivity.
    """

    def test_live_config_omits_thinking_and_proactivity_on_live_model(self):
        config = gemini_live.build_live_connect_config(
            model="gemini-3.8-live",
            voice_name="Charon",
            system_prompt="You are a senior physician examiner.",
            agent_kind="viva-examiner",
        )
        self.assertIsNone(config.thinking_config)
        self.assertFalse(getattr(config, "enable_affective_dialog", False))
        self.assertIsNone(getattr(config, "proactivity", None))

        # Validate voice configuration
        speech_cfg = config.speech_config
        self.assertIsNotNone(speech_cfg)
        self.assertEqual(
            speech_cfg.voice_config.prebuilt_voice_config.voice_name,
            "Charon",
        )

        # Validate system instructions
        self.assertIsNotNone(config.system_instruction)
        self.assertEqual(
            config.system_instruction.parts[0].text,
            "You are a senior physician examiner.",
        )


class ObserverWebSocketRelayTests(unittest.TestCase):
    """Observer WebSocket relay (/voice/observer/ws) token protection and frame relay."""

    def setUp(self):
        observer.reset_state()
        self.client = TestClient(server.app)

    def tearDown(self):
        observer.reset_state()

    def test_rejects_invalid_token(self):
        with self.assertRaises((WebSocketDisconnect, Exception)):
            with self.client.websocket_connect(
                "/voice/observer/ws?channel=test-bad&uid=1001&token=bad-token"
            ) as ws:
                ws.receive_json()

    def test_connect_valid_session_and_relay(self):
        channel = "ch-gemini-relay-01"
        student_uid = 1001
        agent_uid = 900001
        observer.register_session(channel, student_uid, "case-01", agent_uid)

        # Mint student token
        student_token = observer.mint_token(channel, student_uid)

        # Connect student
        with self.client.websocket_connect(
            f"/voice/observer/ws?channel={channel}&uid={student_uid}&token={student_token}&role=student"
        ) as ws_student:
            snap = ws_student.receive_json()
            self.assertEqual(snap["type"], "presence")
            self.assertEqual(snap["action"], "SNAPSHOT")
            self.assertEqual(snap["channel"], channel)

            # Test ping/pong
            ws_student.send_text(json.dumps({"type": "ping"}))
            pong = ws_student.receive_json()
            self.assertEqual(pong["type"], "pong")

            # Test medsim.whisper ack
            ws_student.send_text(json.dumps({"type": "medsim.whisper", "text": "Listen to heart sounds"}))
            ack = ws_student.receive_json()
            self.assertEqual(ack["type"], "medsim.ack")
            self.assertEqual(ack["for"], "medsim.whisper")


if __name__ == "__main__":
    unittest.main()
