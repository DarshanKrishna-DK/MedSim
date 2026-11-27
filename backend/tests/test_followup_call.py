"""Asha follow-up call: registry, public /call-api routes, rate limit, 4 minute
guard and summary coercion. Agora REST and the LLM are mocked."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402

ENV = {
    "GEMINI_API_KEY": "fake-gemini-key",
    "AGORA_APP_ID": "0" * 32,
    "AGORA_APP_CERTIFICATE": "1" * 32,
    "AGORA_CUSTOMER_ID": "cust",
    "AGORA_CUSTOMER_SECRET": "secret",
    "MURF_API_KEY": "murf-test",
    "SARVAM_API_KEY": "sarvam-test",
}

SNAP = {
    "caseId": "im-001",
    "patientName": "Ravi Kumar",
    "age": 52,
    "gender": "M",
    "chiefComplaint": "cough and fever for 5 days",
    "prescriptions": [{"name": "Azithromycin", "dose": "500 mg once daily", "duration": "3 days"}],
    "reason": "check fever settled",
    "lang": "en",
}


class RegistryTest(unittest.TestCase):
    def test_register_and_get(self):
        from integrations import followup_calls as reg

        tok = reg.register_call(SNAP)
        self.assertGreaterEqual(len(tok), 16)
        got = reg.get_call(tok)
        self.assertEqual(got["patientName"], "Ravi Kumar")
        got["patientName"] = "changed"  # copies, not the stored dict
        self.assertEqual(reg.get_call(tok)["patientName"], "Ravi Kumar")
        self.assertIsNone(reg.get_call("nope"))

    def test_ttl_and_cap(self):
        from integrations import followup_calls as reg

        tok = reg.register_call(SNAP)
        with mock.patch.object(reg.time, "time", return_value=time.time() + reg.TTL_SECONDS + 5):
            self.assertIsNone(reg.get_call(tok))
        with mock.patch.object(reg, "MAX_ENTRIES", 3):
            toks = [reg.register_call(SNAP) for _ in range(5)]
            self.assertIsNone(reg.get_call(toks[0]))
            self.assertIsNotNone(reg.get_call(toks[-1]))


class CoerceTest(unittest.TestCase):
    def test_coerces_messy_llm_output(self):
        from followup_call import coerce_summary

        raw = "<think>hmm</think>```json\n" + json.dumps({
            "adherence": "PARTIAL",
            "sideEffects": "mild nausea",
            "redFlags": ["none"],
            "patientMood": "a bit tired",
            "summary": " ".join(["word"] * 90),
            "followUpNeeded": "no",
        }) + "\n```"
        out = coerce_summary(raw)
        self.assertEqual(out["adherence"], "partial")
        self.assertEqual(out["sideEffects"], ["mild nausea"])
        self.assertEqual(out["redFlags"], [])
        self.assertFalse(out["followUpNeeded"])
        self.assertLessEqual(len(out["summary"].split()), 60)

    def test_red_flag_forces_follow_up_and_bad_values_default(self):
        from followup_call import coerce_summary

        out = coerce_summary('{"adherence": "excellent", "redFlags": ["chest pain"], "followUpNeeded": false}')
        self.assertEqual(out["adherence"], "unknown")
        self.assertTrue(out["followUpNeeded"])
        self.assertEqual(coerce_summary("not json at all")["adherence"], "unknown")
        self.assertEqual(set(coerce_summary({})), {"adherence", "sideEffects", "redFlags", "patientMood", "summary", "followUpNeeded"})


class PersonaTest(unittest.TestCase):
    def test_persona_has_rules_meds_and_language(self):
        from followup_call import build_persona, greeting_for

        p = build_persona(SNAP)
        self.assertIn("Azithromycin", p)
        self.assertIn("108", p)
        self.assertIn("Never change", p)
        self.assertIn("ONE question", p)
        self.assertIn("at most two short sentences", p)
        self.assertIn("never restart your earlier sentence", p)
        self.assertIn("Never say you did not understand", p)
        self.assertIn("Tele-MANAS on 14416", p)  # red-flag safety intact
        self.assertLessEqual(len(greeting_for(SNAP).split()), 22)
        hi = build_persona({**SNAP, "lang": "hi"})
        self.assertIn("Devanagari", hi)
        self.assertIn("Ravi Kumar", greeting_for(SNAP))
        self.assertIn("आशा", greeting_for({**SNAP, "lang": "hi"}))
        child = build_persona({**SNAP, "age": 6})
        self.assertIn("parent", child)


class EndpointTest(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, ENV)
        self._env.start()
        os.environ.pop("PATIENT_LLM_PROVIDER", None)
        import followup_call
        import gemini_live
        import server
        import voice
        from integrations import followup_calls as reg

        self.fc, self.voice, self.reg, self.gemini_live = followup_call, voice, reg, gemini_live
        followup_call._history.clear()
        self._guard = mock.patch.object(followup_call, "_schedule_guard", lambda call, lang: None)
        self._guard.start()
        from fastapi.testclient import TestClient

        self.client = TestClient(server.app)
        self.token = reg.register_call(SNAP)

    def tearDown(self):
        self._guard.stop()
        self._env.stop()

    def test_info_is_public_and_404_for_unknown(self):
        # No Origin header at all, like a phone opening the link.
        r = self.client.get(f"/call-api/{self.token}")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["patientName"], "Ravi Kumar")
        self.assertEqual(d["prescriptions"][0]["name"], "Azithromycin")
        self.assertEqual(d["callsLeftToday"], 3)
        self.assertNotIn("chiefComplaint", d)
        self.assertEqual(self.client.get("/call-api/unknown-token-123").status_code, 404)
        self.assertEqual(self.client.get("/call-api/bad$tok").status_code, 404)

    def test_start_builds_nurse_agent_and_stop(self):
        r = self.client.post(f"/call-api/{self.token}/start", headers={"Origin": "https://phone.example.com"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        for k in ("appId", "channel", "uid", "token", "agentUid", "agentId", "callId"):
            self.assertIn(k, d)
        self.assertEqual(d["maxSeconds"], 240)
        self.assertEqual(d["callsLeftToday"], 2)
        self.assertTrue(d["channel"].startswith("asha-followup-call-"))
        prep = self.gemini_live.get_prepared_session(d["channel"])
        self.assertIsNotNone(prep)
        self.assertEqual(prep.agent_kind, "followup")
        self.assertIn("Nurse Asha", prep.initial_line)
        self.assertIn("Nurse Asha", prep.system_prompt)
        # Stop accepts a sendBeacon text/plain body.
        r = self.client.post(
            f"/call-api/{self.token}/stop", content=json.dumps({"agentId": d["agentId"]}),
            headers={"Content-Type": "text/plain"},
        )
        self.assertEqual(r.json(), {"ok": True, "stopped": 1})
        # A foreign agent id is not stopped through this link.
        r = self.client.post(f"/call-api/{self.token}/stop", json={"agentId": "someone-else"})
        self.assertEqual(r.json()["stopped"], 0)

    def test_rate_limit_three_per_day(self):
        for i in range(3):
            self.assertEqual(self.client.post(f"/call-api/{self.token}/start").status_code, 200, i)
        r = self.client.post(f"/call-api/{self.token}/start")
        self.assertEqual(r.status_code, 429)

    def test_failed_join_does_not_count(self):
        with mock.patch.object(self.voice, "start_agent_session", side_effect=self.voice.HTTPException(502, "x")):
            self.assertEqual(self.client.post(f"/call-api/{self.token}/start").status_code, 502)
        self.assertEqual(self.fc.calls_left(self.token), 3)

    def test_voice_start_unchanged(self):
        r = self.client.post("/voice/start", json={"caseId": "im-001", "systemPrompt": "p", "initialLine": "hi", "gender": "F"},
                             headers={"Origin": "http://localhost:5173"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["channel"].startswith("medsim-im-001-"))
        # The window rolls after 24 hours.
        with mock.patch.object(self.fc.time, "time", return_value=time.time() + self.fc.DAY_S + 1):
            self.assertEqual(self.fc.calls_left(self.token), 3)

    def test_failed_join_does_not_count(self):
        with mock.patch.object(self.voice, "start_agent_session", side_effect=self.voice.HTTPException(502, "x")):
            self.assertEqual(self.client.post(f"/call-api/{self.token}/start").status_code, 502)
        self.assertEqual(self.fc.calls_left(self.token), 3)

    def test_summary_flow_and_doctor_route(self):
        # No call started yet: the public summary route refuses (not a free LLM).
        r = self.client.post(f"/call-api/{self.token}/summary", json={"transcript": []})
        self.assertEqual(r.status_code, 409)
        self.client.post(f"/call-api/{self.token}/start")
        llm_reply = {"content": json.dumps({
            "adherence": "good", "sideEffects": [], "redFlags": [], "patientMood": "relieved",
            "summary": "Fever settled, taking azithromycin daily.", "followUpNeeded": False,
        })}

        async def fake_chat(messages, **kw):
            self.assertTrue(kw.get("json_mode"))
            self.assertIn("I feel better", messages[1]["content"])
            return llm_reply, "sarvam"

        with mock.patch.object(self.fc, "chat", fake_chat):
            r = self.client.post(f"/call-api/{self.token}/summary", json={"transcript": [
                {"role": "assistant", "content": "Namaste Ravi, how are you feeling?"},
                {"role": "user", "content": "I feel better, the fever is gone."},
                {"role": "user", "content": ""},
            ]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["summary"]["adherence"], "good")
        # Only one summary per call.
        self.assertEqual(self.client.post(f"/call-api/{self.token}/summary", json={"transcript": []}).status_code, 409)

        # Doctor side is under the /agent/ origin guard.
        path = f"/agent/followup-call/{self.token}/summaries"
        self.assertEqual(self.client.get(path, headers={"Origin": "https://evil.example.com"}).status_code, 401)
        r = self.client.get(path, headers={"Origin": "http://localhost:5173"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["caseId"], "im-001")
        self.assertEqual(len(d["calls"]), 1)
        self.assertEqual(d["calls"][0]["summary"]["patientMood"], "relieved")
        self.assertEqual(len(d["calls"][0]["transcript"]), 2)
        self.assertEqual(self.client.get("/agent/followup-call/unknown-token-1/summaries",
                                         headers={"Origin": "http://localhost:5173"}).status_code, 404)

    def test_summary_without_patient_speech_skips_llm_and_llm_failure_falls_back(self):
        self.client.post(f"/call-api/{self.token}/start")

        async def boom(*a, **kw):
            raise AssertionError("LLM must not be called")

        with mock.patch.object(self.fc, "chat", boom):
            r = self.client.post(f"/call-api/{self.token}/summary", json={"transcript": [{"role": "assistant", "content": "Hello?"}]})
        self.assertEqual(r.json()["summary"]["adherence"], "unknown")
        self.assertTrue(r.json()["summary"]["followUpNeeded"])

        self.client.post(f"/call-api/{self.token}/start")

        async def fail(*a, **kw):
            raise RuntimeError("down")

        with mock.patch.object(self.fc, "chat", fail):
            r = self.client.post(f"/call-api/{self.token}/summary", json={"transcript": [{"role": "user", "content": "chest pain"}]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["summary"]["followUpNeeded"])


class GuardTest(unittest.TestCase):
    def test_guard_wraps_up_then_leaves(self):
        import followup_call as fc

        events: list[str] = []

        async def speak(agent_id, text, **kw):
            events.append(f"speak:{agent_id}")

        async def leave(agent_id):
            events.append(f"leave:{agent_id}")
            return True

        async def no_sleep(_s):
            return None

        call = {"agentId": "a1", "endedAt": None}
        with mock.patch.object(fc.voice, "speak_line", speak), mock.patch.object(fc.voice, "leave_agent", leave), \
                mock.patch.object(fc.asyncio, "sleep", no_sleep):
            asyncio.run(fc._guard(call, "hi"))
        self.assertEqual(events, ["speak:a1", "leave:a1"])
        self.assertIsNotNone(call["endedAt"])

    def test_guard_noop_when_call_already_ended(self):
        import followup_call as fc

        events: list[str] = []

        async def leave(agent_id):
            events.append("leave")
            return True

        async def no_sleep(_s):
            return None

        call = {"agentId": "a2", "endedAt": 123}
        with mock.patch.object(fc.voice, "leave_agent", leave), mock.patch.object(fc.asyncio, "sleep", no_sleep):
            asyncio.run(fc._guard(call, "en"))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
