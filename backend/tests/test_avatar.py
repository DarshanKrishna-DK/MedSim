"""Avatar configuration and option of /voice/start for Gemini stack. No network."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import voice  # noqa: E402

ENV = {
    "GEMINI_API_KEY": "fake-gemini-key",
}
AV = ["AVATAR_VENDOR", "AVATAR_API_KEY", "AVATAR_ID", "AVATAR_QUALITY"]
HDR = {"Origin": "http://localhost:5173"}


class AvatarTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = {k: os.environ.get(k) for k in [*ENV, *AV]}
        os.environ.update(ENV)
        for k in AV:
            os.environ.pop(k, None)
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _start(self, case: str = "teacher", **extra):
        return self.client.post("/voice/start", headers=HDR, json={
            "caseId": case, "systemPrompt": "x", "initialLine": "y", **extra})

    def _enable(self, vendor: str = "anam") -> None:
        os.environ.update({"AVATAR_VENDOR": vendor, "AVATAR_API_KEY": "av-key-zz", "AVATAR_ID": "face1"})

    def test_off_without_env(self) -> None:
        self.assertIsNone(voice.avatar_config())
        req = voice.VoiceStartRequest(caseId="teacher", systemPrompt="x", initialLine="y", avatar=True)
        self.assertFalse(voice.want_avatar(req))
        r = self._start(avatar=True)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNone(r.json().get("avatarUid"))

    def test_config_parsing(self) -> None:
        self._enable("anam")
        cfg = voice.avatar_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["vendor"], "anam")
        self.assertEqual(cfg["api_key"], "av-key-zz")
        self.assertEqual(cfg["avatar_id"], "face1")
        self.assertEqual(cfg["quality"], "high")

    def test_on_for_teacher_by_default(self) -> None:
        self._enable()
        req = voice.VoiceStartRequest(caseId="teacher", systemPrompt="x", initialLine="y")
        self.assertTrue(voice.want_avatar(req))
        r = self._start("teacher")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["avatarUid"], voice.AVATAR_UID)
        self.assertNotIn("av-key-zz", r.text)

    def test_patient_needs_explicit_request(self) -> None:
        self._enable()
        req1 = voice.VoiceStartRequest(caseId="case1", systemPrompt="x", initialLine="y")
        self.assertFalse(voice.want_avatar(req1))
        r1 = self._start("case1")
        self.assertIsNone(r1.json().get("avatarUid"))

        req2 = voice.VoiceStartRequest(caseId="case1", systemPrompt="x", initialLine="y", avatar=True)
        self.assertTrue(voice.want_avatar(req2))
        r2 = self._start("case1", avatar=True)
        self.assertEqual(r2.json()["avatarUid"], voice.AVATAR_UID)

        req3 = voice.VoiceStartRequest(caseId="teacher", systemPrompt="x", initialLine="y", avatar=False)
        self.assertFalse(voice.want_avatar(req3))
        r3 = self._start("teacher", avatar=False)
        self.assertIsNone(r3.json().get("avatarUid"))


if __name__ == "__main__":
    unittest.main()
