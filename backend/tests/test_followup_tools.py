"""Asha acting during the follow-up call: tools run inside our custom LLM endpoint. No network."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import followup_call  # noqa: E402
import followup_tools  # noqa: E402
import gemini_live  # noqa: E402
import llm_proxy  # noqa: E402
import server  # noqa: E402
import voice  # noqa: E402
from integrations import call_reminders, fhir, followup_calls, gcal_script, notify  # noqa: E402

ENV = {
    "GEMINI_API_KEY": "fake-gemini-key",
    "AGORA_APP_ID": "0" * 32, "AGORA_APP_CERTIFICATE": "1" * 32,
    "AGORA_CUSTOMER_ID": "cust", "AGORA_CUSTOMER_SECRET": "secret",
    "MURF_API_KEY": "murf-test", "SARVAM_API_KEY": "sarvam-test",
    "DOCTOR_ALERT_DEST": "tg:12345",
}
SNAP = {
    "caseId": "im-001", "patientName": "Ravi Kumar", "age": 52, "gender": "M",
    "chiefComplaint": "cough and fever", "reason": "check fever settled", "lang": "en",
    "prescriptions": [{"name": "Azithromycin", "dose": "500 mg", "duration": "3 days"}],
    "contact": "ravi@example.com",
}
SSE = b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'


def tool_msg(name: str, args, cid: str = "call_1") -> dict:
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": raw}}]}


def text_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def sse_text(body: str) -> str:
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for ch in json.loads(line[6:])["choices"]:
                out.append(ch["delta"].get("content") or "")
    return "".join(out)


class Base(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, ENV)
        self._env.start()
        for k in ("GROQ_API_KEY", "PATIENT_LLM_PROVIDER", "PUBLIC_URL"):
            os.environ.pop(k, None)
        followup_call._history.clear()
        followup_tools._orphans.clear()
        self.replies: list = []  # queued provider answers: dict message or int status
        self.seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            self.seen.append(body)
            if body.get("stream"):  # patient pass-through path
                return httpx.Response(200, stream=httpx.ByteStream(SSE), headers={"content-type": "text/event-stream"})
            nxt = self.replies.pop(0) if self.replies else text_msg("ok")
            if isinstance(nxt, int):
                return httpx.Response(nxt, json={"error": "boom"})
            return httpx.Response(200, json={"choices": [{"message": nxt}]})

        self._cf = mock.patch.object(llm_proxy, "_client_factory",
                                     lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self._cf.start()
        self.sent: list[tuple] = []

        async def fake_send(channel, to, message, *a, **k):
            self.sent.append((channel, to, message))
            return {"ok": True, "channel": "telegram"}

        self.flags: list[tuple] = []

        async def fake_flag(case, text):
            self.flags.append((case, text))
            return {"ok": True, "resourceType": "Flag", "id": "f1"}

        self._p = [mock.patch.object(notify, "send_to_patient", fake_send),
                   mock.patch.object(fhir, "record_flag", fake_flag)]
        for p in self._p:
            p.start()
        self.token = followup_calls.register_call(SNAP)
        self.call = {"callId": "c1", "agentId": "ag1", "startedAt": 1, "endedAt": None,
                     "summary": None, "transcript": None, "actions": followup_call.new_actions()}
        followup_call._hist(self.token)["calls"].append(self.call)
        self.meta = {"kind": "followup", "callToken": self.token, "callId": "c1"}
        self.client = TestClient(server.app)

    def tearDown(self):
        for p in self._p:
            p.stop()
        self._cf.stop()
        self._env.stop()

    def turn(self, user: str, stream: bool = True, tok: str | None = None):
        tok = tok or llm_proxy.mint_token(self.meta)
        body = {"stream": stream, "messages": [
            {"role": "system", "content": "You are Nurse Asha"}, {"role": "user", "content": user}]}
        return self.client.post(llm_proxy.PROXY_PATH, json=body, headers={"Authorization": f"Bearer {tok}"})


class ToolTurnTests(Base):
    def test_round_trip_streams_filler_then_final_text(self):
        self.replies = [tool_msg("escalate_red_flag", {"symptom": "chest pain", "severity": "emergency"}),
                        text_msg("I have alerted your doctor. Please call 108 now.")]
        r = self.turn("I have chest pain since morning")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
        spoken = sse_text(r.text)
        self.assertTrue(spoken.startswith("One moment, I'm letting your doctor know."))
        self.assertIn("I have alerted your doctor. Please call 108 now.", spoken)
        # Provider got the tools and, in round two, the tool result.
        self.assertEqual({t["function"]["name"] for t in self.seen[0]["tools"]},
                         {"escalate_red_flag", "reschedule_call", "log_symptom"})
        self.assertEqual(self.seen[1]["messages"][-1]["role"], "tool")
        self.assertEqual(self.sent, [("auto", "tg:12345",
                                      "RED FLAG from Ravi Kumar's follow-up call: chest pain. Asha advised emergency care / 108.")])
        self.assertEqual(len(self.flags), 1)
        self.assertEqual(self.flags[0][0]["id"], "im-001")
        rf = self.call["actions"]["redFlags"]
        self.assertEqual([(f["symptom"], f["severity"], f["doctorNotified"]) for f in rf], [("chest pain", "emergency", True)])
        # Doctor route shows the live action before any summary exists.
        d = self.client.get(f"/agent/followup-call/{self.token}/summaries", headers={"Origin": "http://localhost:5173"}).json()
        self.assertEqual(d["calls"][0]["actions"]["redFlags"][0]["symptom"], "chest pain")

    def test_plain_answer_without_tools(self):
        self.replies = [text_msg("Glad you feel better. Are you taking the azithromycin?")]
        r = self.turn("I feel better")
        self.assertEqual(sse_text(r.text), "Glad you feel better. Are you taking the azithromycin?")
        self.assertEqual(self.sent, [])

    def test_non_streaming_returns_completion_json(self):
        self.replies = [tool_msg("log_symptom", {"symptom": "mild nausea", "adherence": "partial"}),
                        text_msg("Thanks, I have noted the nausea.")]
        r = self.turn("I feel a bit sick and missed two doses", stream=False)
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "Thanks, I have noted the nausea.")
        self.assertEqual(self.call["actions"]["symptoms"][0]["symptom"], "mild nausea")
        self.assertEqual(self.call["actions"]["symptoms"][0]["adherence"], "partial")

    def test_escalation_dedupes_within_a_call(self):
        esc = {"symptom": "Chest pain", "severity": "urgent"}
        self.replies = [tool_msg("escalate_red_flag", esc), text_msg("Doctor alerted."),
                        tool_msg("escalate_red_flag", {"symptom": "chest  pain!", "severity": "emergency"}),
                        text_msg("Your doctor already knows.")]
        self.turn("chest pain")
        r2 = self.turn("it still hurts in my chest")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.flags), 1)
        self.assertEqual(len(self.call["actions"]["redFlags"]), 1)
        self.assertIn("Your doctor already knows.", sse_text(r2.text))
        # Second turn told the model what was already done.
        notes = [m["content"] for m in self.seen[2]["messages"] if m["role"] == "system"]
        self.assertTrue(any("already alerted about: Chest pain" in n for n in notes))

    def test_alert_falls_back_to_patient_contact(self):
        os.environ.pop("DOCTOR_ALERT_DEST")
        self.replies = [tool_msg("escalate_red_flag", {"symptom": "fainted", "severity": "emergency"}), text_msg("ok")]
        self.turn("I fainted")
        self.assertEqual(self.sent[0][1], "ravi@example.com")

    def test_reschedule_books_event_and_reminder(self):
        book = mock.AsyncMock(return_value={"ok": True, "htmlLink": "https://cal/x"})
        sched = mock.Mock(return_value=["2026-11-22T10:00:00+05:30"])
        self.replies = [tool_msg("reschedule_call", {"when_text": "tomorrow", "in_days": 1}),
                        text_msg("Done, I will call you tomorrow.")]
        with mock.patch.object(gcal_script, "book_event", book), \
                mock.patch.object(call_reminders, "schedule", sched), \
                mock.patch.object(notify, "absolute_url", lambda p: "https://medsim.test" + p):
            r = self.turn("Can you call me tomorrow instead?")
            again = asyncio.run(followup_tools.reschedule_call(followup_tools._ctx(self.meta),
                                                               {"when_text": "tomorrow", "in_days": 1}))
        self.assertIn("One moment, I'm moving your call.", sse_text(r.text))
        self.assertIn("Done, I will call you tomorrow.", sse_text(r.text))
        book.assert_awaited_once()
        self.assertIn(f"https://medsim.test/call/{self.token}", book.await_args.args[1])
        self.assertEqual(book.await_args.kwargs["guest_email"], "ravi@example.com")
        sched.assert_called_once()
        self.assertEqual(sched.call_args.args[:3], ("ravi@example.com", "Ravi Kumar", f"https://medsim.test/call/{self.token}"))
        rs = self.call["actions"]["reschedules"]
        self.assertEqual(len(rs), 1)
        self.assertTrue(rs[0]["booked"])
        self.assertTrue(rs[0]["startText"])
        tool_result = json.loads(self.seen[1]["messages"][-1]["content"])
        self.assertEqual(tool_result["startText"], rs[0]["startText"])
        self.assertTrue(again["alreadyBooked"])

    def test_reschedule_in_minutes_uses_start_time(self):
        book = mock.AsyncMock(return_value=None)
        with mock.patch.object(gcal_script, "book_event", book), \
                mock.patch.object(call_reminders, "schedule", mock.Mock(return_value=[])):
            res = asyncio.run(followup_tools.reschedule_call(
                followup_tools._ctx(self.meta), {"when_text": "in half an hour", "starts_in_minutes": "30"}))
        self.assertTrue(res["ok"])
        self.assertFalse(res["calendarBooked"])
        self.assertEqual(self.call["actions"]["reschedules"][0]["slot"], "m30")

    def test_malformed_args_do_not_crash_and_speak_fallback(self):
        self.replies = [tool_msg("reschedule_call", "{not json"), 500]
        r = self.turn("call me later")
        self.assertEqual(r.status_code, 200)
        self.assertIn(followup_tools.FALLBACK["en"], sse_text(r.text))
        self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(self.call["actions"]["reschedules"], [])
        # A garbled escalation still alerts the doctor, using the patient's words.
        self.replies = [tool_msg("escalate_red_flag", "{{{"), 500]
        r = self.turn("my lips are swelling up")
        self.assertIn(followup_tools.EMERGENCY_FALLBACK["en"], sse_text(r.text))
        self.assertEqual(self.call["actions"]["redFlags"][0]["symptom"], "my lips are swelling up")

    def test_unknown_tool_and_endless_tool_calls_end_with_safe_line(self):
        self.replies = [tool_msg("delete_everything", {}), tool_msg("log_symptom", {"symptom": "cough"})]
        r = self.turn("hello")
        self.assertIn(followup_tools.FALLBACK["en"], sse_text(r.text))
        self.assertEqual(len(self.seen), 2)  # max 2 rounds

    def test_provider_down_speaks_fallback(self):
        self.replies = [500, 500, 500]
        r = self.turn("hello")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sse_text(r.text), followup_tools.FALLBACK["en"])

    def test_followup_turn_addressing_asha_is_not_silenced(self):
        self.replies = [text_msg("Yes, I am listening.")]
        r = self.turn("Asha, I have a question")
        self.assertEqual(sse_text(r.text), "Yes, I am listening.")

    def test_summary_keeps_live_red_flags(self):
        self.call["actions"]["redFlags"].append({"symptom": "chest pain", "severity": "emergency"})

        async def fake_chat(*a, **k):
            return {"content": json.dumps({"adherence": "good", "redFlags": [], "summary": "fine"})}, "sarvam"

        with mock.patch.object(followup_call, "chat", fake_chat), \
                mock.patch.object(followup_call.voice, "leave_agent", mock.AsyncMock(return_value=True)):
            r = self.client.post(f"/call-api/{self.token}/summary", json={
                "callId": "c1", "transcript": [{"role": "user", "content": "chest pain"}]})
        s = r.json()["summary"]
        self.assertEqual(s["redFlags"], ["chest pain"])
        self.assertTrue(s["followUpNeeded"])


class PatientSessionTests(Base):
    def test_patient_tokens_pass_through_unchanged(self):
        with mock.patch.object(followup_tools, "handle", side_effect=AssertionError("tools path used")):
            r = self.turn("I have a cough", tok=llm_proxy.mint_token())
        self.assertEqual(r.content, SSE)
        self.assertNotIn("tools", self.seen[0])
        self.assertIsNone(llm_proxy.token_meta(llm_proxy.mint_token()))

    def test_patient_nurse_address_still_silenced(self):
        r = self.turn("Nurse, order a CBC", tok=llm_proxy.mint_token())
        self.assertEqual(sse_text(r.text), "")
        self.assertEqual(self.seen, [])


class WiringTests(Base):
    def test_followup_start_tags_proxy_token_and_persona_acts(self):
        with mock.patch.object(followup_call, "_schedule_guard", lambda call, lang: None):
            r = self.client.post(f"/call-api/{self.token}/start")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        call_id = data["callId"]
        prep = gemini_live.get_prepared_session(data["channel"])
        self.assertIsNotNone(prep)
        self.assertEqual(prep.agent_kind, "followup")
        self.assertEqual(prep.llm_token_meta, {"kind": "followup", "callToken": self.token, "callId": call_id})
        self.assertIn("escalate_red_flag", prep.system_prompt)
        self.assertIn("reschedule_call", prep.system_prompt)
        tools = gemini_live._build_tools("followup")
        self.assertTrue(len(tools) > 0)
        live = followup_call.find_call(self.token, call_id)
        self.assertEqual(live["actions"], followup_call.new_actions())
        # The patient's page never sees the contact on file.
        self.assertNotIn("contact", self.client.get(f"/call-api/{self.token}").json())

    def test_persona_without_proxy_promises_no_actions(self):
        self.assertNotIn("escalate_red_flag", followup_call.build_persona(SNAP))
        self.assertIn("escalate_red_flag", followup_call.build_persona(SNAP, acting=True))

    def test_record_flag_posts_fhir_flag(self):
        posted: list[tuple[str, dict]] = []

        async def fake_post(rtype, body, extra=None):
            posted.append((rtype, body))
            return {"ok": True, "resourceType": rtype, "id": "p1" if rtype == "Patient" else "f1"}

        for p in self._p:
            p.stop()
        self._p = []
        fhir._patient_cache.pop("im-001", None)
        with mock.patch.object(fhir, "_post", fake_post):
            res = asyncio.run(fhir.record_flag({"id": "im-001", "name": "Ravi Kumar", "age": 52, "gender": "M"}, "RED FLAG x"))
        self.assertEqual(res["resourceType"], "Flag")
        flag = posted[-1][1]
        self.assertEqual(flag["status"], "active")
        self.assertEqual(flag["code"]["text"], "RED FLAG x")
        self.assertEqual(flag["subject"]["reference"], "Patient/p1")


if __name__ == "__main__":
    unittest.main()
