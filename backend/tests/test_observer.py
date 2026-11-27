"""Live faculty observer: invite/join minting, expiry, single-channel codes and
token verification. No network."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from fastapi.testclient import TestClient  # noqa: E402

import observer  # noqa: E402
import server  # noqa: E402

HDR = {"Origin": "http://localhost:5173"}


class ObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        observer.reset_state()
        observer.register_session("medsim-im-001-abcd1234", 123456, "im-001", 900001)
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        observer.reset_state()

    def _invite(self, channel: str = "medsim-im-001-abcd1234", case_id: str = "im-001"):
        return self.client.post("/voice/observer/invite", json={"channel": channel, "caseId": case_id}, headers=HDR)

    def test_invite_returns_code_url_and_link(self):
        r = self._invite()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn(".", body["code"])
        self.assertEqual(body["inviteUrl"], f"http://localhost:5173/?observe={body['code']}")
        self.assertGreaterEqual(body["link"]["uid"], observer.LINK_UID_MIN)
        # Codes are random: two invites differ.
        self.assertNotEqual(body["code"], self._invite().json()["code"])

    def test_invite_requires_registered_channel(self):
        self.assertEqual(self._invite(channel="medsim-made-up-0000").status_code, 404)
        self.assertEqual(self._invite(case_id="other-case").status_code, 400)

    def test_invite_needs_trusted_origin(self):
        r = self.client.post(
            "/voice/observer/invite",
            json={"channel": "medsim-im-001-abcd1234"},
            headers={"Origin": "https://evil.example.com"},
        )
        self.assertEqual(r.status_code, 401)

    def test_join_mints_audience_token_for_that_channel_only(self):
        code = self._invite().json()["code"]
        r = self.client.post("/voice/observer/join", json={"code": code}, headers=HDR)
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["channel"], "medsim-im-001-abcd1234")
        self.assertEqual(j["role"], "audience")
        self.assertEqual(j["studentUid"], 123456)
        self.assertEqual(j["agentUid"], 900001)
        self.assertNotIn(j["uid"], (123456, 900001, j["linkUid"]))

        # Verify token validity
        self.assertTrue(observer.verify_token(j["token"], j["channel"], j["uid"]))
        # False for wrong channel or wrong uid
        self.assertFalse(observer.verify_token(j["token"], "other-channel", j["uid"]))
        self.assertFalse(observer.verify_token(j["token"], j["channel"], 999999))

    def test_link_token_is_valid(self):
        link = self._invite().json()["link"]
        self.assertTrue(observer.verify_token(link["token"], "medsim-im-001-abcd1234", link["uid"]))

    def test_bad_and_forged_codes_rejected(self):
        code = self._invite().json()["code"]
        raw, _, _tag = code.partition(".")
        for bad in ("nope", f"{raw}.0000000000", f"x{raw}.{_tag}", code + "x"):
            r = self.client.post("/voice/observer/join", json={"code": bad}, headers=HDR)
            self.assertEqual(r.status_code, 404, bad)

    def test_codes_expire(self):
        code = self._invite().json()["code"]
        later = observer._now() + observer.INVITE_TTL_S + 1
        self.assertIsNone(observer.lookup_invite(code, now=later))
        with mock.patch.object(observer, "_now", return_value=later):
            r = self.client.post("/voice/observer/join", json={"code": code}, headers=HDR)
        self.assertEqual(r.status_code, 404)

    def test_invite_and_join_caps(self):
        for _ in range(observer.MAX_INVITES_PER_CHANNEL - 1):
            self.assertEqual(self._invite().status_code, 200)
        code = self._invite().json()["code"]
        self.assertEqual(self._invite().status_code, 429)
        for _ in range(observer.MAX_JOINS_PER_CODE):
            self.assertEqual(self.client.post("/voice/observer/join", json={"code": code}, headers=HDR).status_code, 200)
        self.assertEqual(self.client.post("/voice/observer/join", json={"code": code}, headers=HDR).status_code, 429)


class VoiceStartRegistersTests(unittest.TestCase):
    """A channel created by /voice/start can be invited to."""

    ENV = {
        "GEMINI_API_KEY": "fake-gemini-key",
    }

    def setUp(self) -> None:
        self._old = {k: os.environ.get(k) for k in self.ENV}
        os.environ.update(self.ENV)
        observer.reset_state()
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        observer.reset_state()
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_started_channel_is_invitable(self):
        r = self.client.post("/voice/start", headers=HDR, json={
            "caseId": "im-001", "systemPrompt": "x", "initialLine": "y", "gender": "M"})
        self.assertEqual(r.status_code, 200, r.text)
        s = r.json()
        inv = self.client.post("/voice/observer/invite", headers=HDR, json={"channel": s["channel"], "caseId": "im-001"})
        self.assertEqual(inv.status_code, 200, inv.text)
        j = self.client.post("/voice/observer/join", headers=HDR, json={"code": inv.json()["code"]}).json()
        self.assertEqual(j["studentUid"], s["uid"])
        self.assertEqual(j["channel"], s["channel"])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
