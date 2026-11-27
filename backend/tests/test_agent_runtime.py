"""Autonomous voice agents: runtime routing, toolsets, event bus, SSE shape. No network."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import agent_events  # noqa: E402
import agent_runtime  # noqa: E402
import agent_toolsets  # noqa: E402
import assignments as asg  # noqa: E402
import followup_tools  # noqa: E402
import gemini_live  # noqa: E402
import llm_proxy  # noqa: E402
import server  # noqa: E402
import voice  # noqa: E402
from integrations import gcal_script, notify  # noqa: E402
from integrations.calendar_invite import IST  # noqa: E402

ENV = {
    "GEMINI_API_KEY": "fake-gemini-key",
    "AGORA_APP_ID": "0" * 32, "AGORA_APP_CERTIFICATE": "1" * 32,
    "AGORA_CUSTOMER_ID": "cust", "AGORA_CUSTOMER_SECRET": "secret",
    "MURF_API_KEY": "murf-test", "SARVAM_API_KEY": "sarvam-test",
    "ASSIGNMENTS_FILE": "off",
}
SSE = b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
DEV = {"Origin": "http://localhost:5173"}
TEACHER_EMAIL = "teacher@school.test"


def tool_msg(*calls) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}


def text_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def sse_chunks(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def sse_text(body: str) -> str:
    return "".join(ch["delta"].get("content") or "" for c in sse_chunks(body) for ch in c["choices"])


class Base(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, ENV)
        self._env.start()
        for k in ("GROQ_API_KEY", "PATIENT_LLM_PROVIDER", "PUBLIC_URL", "GCAL_SCRIPT_URL", "GCAL_SCRIPT_SECRET"):
            os.environ.pop(k, None)
        agent_runtime.reset()
        agent_events.clear()
        asg.reset_store()
        self.replies: list = []
        self.seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            self.seen.append(body)
            if body.get("stream"):  # patient pass-through
                return httpx.Response(200, stream=httpx.ByteStream(SSE), headers={"content-type": "text/event-stream"})
            nxt = self.replies.pop(0) if self.replies else text_msg("ok")
            if isinstance(nxt, int):
                return httpx.Response(nxt, json={"error": "boom"})
            return httpx.Response(200, json={"choices": [{"message": nxt}]})

        self._cf = mock.patch.object(llm_proxy, "_client_factory",
                                     lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self._cf.start()
        self.client = TestClient(server.app)
        self.channel = "medsim-test-0001"

    def tearDown(self):
        self._cf.stop()
        self._env.stop()

    def meta(self, kind: str, ctx: dict | None = None) -> dict:
        return {"kind": kind, "ctx": agent_runtime.clean_ctx(kind, ctx or {}), "channel": self.channel, "lang": "en"}

    def turn(self, tok: str, user: str, stream: bool = True):
        body = {"stream": stream, "messages": [{"role": "system", "content": "persona"}, {"role": "user", "content": user}]}
        return self.client.post(llm_proxy.PROXY_PATH, json=body, headers={"Authorization": f"Bearer {tok}"})

    def events(self, after: int = 0) -> list[dict]:
        return agent_events.since(self.channel, after)["events"]


# ---------------------------------------------------------------- routing

class RoutingTests(Base):
    def test_registered_kinds(self):
        for k in ("receptionist", "teacher", "viva-examiner", "followup"):
            self.assertTrue(agent_runtime.is_registered(k), k)
        self.assertFalse(agent_runtime.is_registered("im-001"))
        self.assertFalse(agent_runtime.is_registered(None))

    def test_patient_token_is_byte_for_byte_pass_through(self):
        with mock.patch.object(agent_runtime, "handle", side_effect=AssertionError("runtime used")):
            r = self.turn(llm_proxy.mint_token(), "I have a cough")
        self.assertEqual(r.content, SSE)
        self.assertNotIn("tools", self.seen[0])
        self.assertTrue(self.seen[0]["stream"])

    def test_unknown_kind_falls_back_to_pass_through(self):
        r = self.turn(llm_proxy.mint_token({"kind": "nope"}), "hello")
        self.assertEqual(r.content, SSE)

    def test_followup_still_routes_to_followup_tools(self):
        called = mock.AsyncMock(return_value=llm_proxy.completion_json("x", "m", "hi"))
        with mock.patch.object(followup_tools, "handle", called):
            r = self.turn(llm_proxy.mint_token({"kind": "followup", "callToken": "t", "callId": "c"}), "hello", stream=False)
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "hi")
        called.assert_awaited_once()

    def test_receptionist_token_gets_tools_and_addendum(self):
        self.replies = [text_msg("Namaste! How can I help?")]
        r = self.turn(llm_proxy.mint_token(self.meta("receptionist")), "hello")
        self.assertEqual(sse_text(r.text), "Namaste! How can I help?")
        names = {t["function"]["name"] for t in self.seen[0]["tools"]}
        self.assertEqual(names, {"get_assignment_status", "get_my_scores", "email_report_to_teacher", "list_assignments",
                                 "book_study_session", "send_to_my_phone"})
        sysmsgs = [m["content"] for m in self.seen[0]["messages"] if m["role"] == "system"]
        self.assertEqual(sysmsgs[0], "persona")
        self.assertIn("TOOLS:", sysmsgs[1])
        self.assertEqual(self.seen[0]["messages"][-1]["role"], "user")


class HumanVoiceTests(unittest.TestCase):
    BAD = ("understand", "catch that", "didn't catch", "sorry", "error", "could not finish")

    def test_fallbacks_are_short_and_human(self):
        tables = [agent_runtime.DEFAULT_FALLBACK] + [agent_runtime.get(k).fallback
                                                     for k in ("receptionist", "teacher", "viva-examiner")]
        for table in tables:
            for line in table.values():
                self.assertLessEqual(len(line.split()), 16, line)
                for bad in self.BAD:
                    self.assertNotIn(bad, line.lower(), line)

    def test_every_addendum_carries_voice_style(self):
        tctx = {"lang": "en", "ctx": {}, "state": {"actions": []}}
        for kind in ("receptionist", "teacher", "viva-examiner"):
            text = agent_runtime._addendum(agent_runtime.get(kind), tctx)
            self.assertTrue(text.startswith(agent_runtime.VOICE_STYLE))
            self.assertIn("never say you did not understand", text)
            self.assertIn("ONE helpful action", text)

    def test_addenda_have_autonomy_guidance(self):
        r = agent_toolsets.RECEPTION_ADDENDUM
        for kw in ("INTENT, NOT KEYWORDS", "send this to sir", "block time to study cardiology tomorrow",
                   "text me that", "how'm I doing", "remind me", "PROACTIVE BUT BOUNDED"):
            self.assertIn(kw, r)
        t = agent_toolsets.TEACHER_ADDENDUM
        for kw in ("draw_on_board", "start_quiz", "give the class some cases", "homework for Friday",
                   "struggling", "PROACTIVE BUT BOUNDED"):
            self.assertIn(kw, t)
        v = agent_toolsets.VIVA_ADDENDUM
        for kw in ("record_answer_grade without being asked", "Adapt", "That's enough", "I'm done",
                   "end_viva at once", "PROACTIVE BUT BOUNDED"):
            self.assertIn(kw, v)
        for text in (r, t, v, agent_runtime.VOICE_STYLE):
            self.assertNotIn(chr(0x2014), text)
            self.assertNotIn(chr(0x2013), text)


# ---------------------------------------------------------------- SSE + fallbacks

class SseTests(Base):
    def test_sse_shape_filler_then_final(self):
        os.environ.update(GCAL_SCRIPT_URL="https://script.test", GCAL_SCRIPT_SECRET="s")
        book = mock.AsyncMock(return_value={"ok": True})
        self.replies = [tool_msg(("book_study_session", {"topic": "cardiology", "when_text": "tomorrow evening"})),
                        text_msg("Done, tomorrow at six.")]
        with mock.patch.object(gcal_script, "book_event", book):
            r = self.turn(llm_proxy.mint_token(self.meta("receptionist")), "I'm free tomorrow evening")
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
        chunks = sse_chunks(r.text)
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        for c in chunks:
            self.assertEqual(c["object"], "chat.completion.chunk")
            self.assertTrue(c["id"].startswith("chatcmpl-"))
        spoken = sse_text(r.text)
        self.assertTrue(spoken.startswith("Achha, let me block that in your calendar."))
        self.assertTrue(spoken.endswith("Done, tomorrow at six."))
        self.assertEqual(self.seen[1]["messages"][-1]["role"], "tool")

    def test_non_streaming_completion(self):
        self.replies = [text_msg("Hi there")]
        r = self.turn(llm_proxy.mint_token(self.meta("teacher")), "hi", stream=False)
        self.assertEqual(r.json()["object"], "chat.completion")
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "Hi there")

    def test_provider_down_speaks_fallback(self):
        self.replies = [500, 500, 500]
        r = self.turn(llm_proxy.mint_token(self.meta("receptionist")), "hello")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sse_text(r.text), agent_runtime.DEFAULT_FALLBACK["en"])

    def test_tool_crash_is_reported_to_model_not_user(self):
        async def boom(*a, **k):
            raise RuntimeError("calendar exploded")

        self.replies = [tool_msg(("schedule_revision", {"topic": "renal", "when_text": "sunday"})),
                        text_msg("Sorry, I could not book that.")]
        with mock.patch.object(agent_toolsets, "book_calendar", boom):
            r = self.turn(llm_proxy.mint_token(self.meta("teacher")), "book revision sunday")
        self.assertIn("Sorry, I could not book that.", sse_text(r.text))
        res = json.loads(self.seen[1]["messages"][-1]["content"])
        self.assertEqual(res, {"ok": False, "error": "RuntimeError"})
        ev = [e for e in self.events() if e["type"] == "tool"]
        self.assertFalse(ev[0]["data"]["ok"])

    def test_malformed_args_and_unknown_tool(self):
        self.replies = [tool_msg(("draw_on_board", "{nope"), ("rm_rf", {})), text_msg("Let me try again.")]
        r = self.turn(llm_proxy.mint_token(self.meta("teacher")), "draw")
        self.assertIn("Let me try again.", sse_text(r.text))
        results = [json.loads(m["content"]) for m in self.seen[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(results[0]["error"], "malformed arguments")
        self.assertIn("unknown tool", results[1]["error"])
        self.assertFalse([e for e in self.events() if e["type"] == "whiteboard"])

    def test_rounds_exhausted_gets_closing_answer_then_fallback(self):
        self.replies = [tool_msg(("draw_on_board", {"topic": "A"})), tool_msg(("draw_on_board", {"topic": "B"})),
                        text_msg("Both are on the board.")]
        r = self.turn(llm_proxy.mint_token(self.meta("teacher")), "draw two")
        self.assertIn("Both are on the board.", sse_text(r.text))
        self.assertNotIn("tools", self.seen[2])  # closing call has no tools
        self.replies = [tool_msg(("draw_on_board", {"topic": "C"})), tool_msg(("draw_on_board", {"topic": "D"})), 500]
        r = self.turn(llm_proxy.mint_token(self.meta("teacher")), "draw two more")
        self.assertEqual(sse_text(r.text).split("board. ")[-1], agent_runtime.DEFAULT_FALLBACK["en"])


# ---------------------------------------------------------------- receptionist

class ReceptionTests(Base):
    def setUp(self):
        super().setUp()
        a = asg.create_assignment(asg.AssignmentIn(title="Cardio week", items=[asg.ItemIn(kind="case", caseId="cv-001")],
                                                   teacherName="Dr. Rao", teacherEmail=TEACHER_EMAIL))
        self.code = a["code"]
        self.ctx = {"assignmentCode": self.code, "knownCodes": [self.code], "profile": {"name": "Asha Rani", "university": "BMC"},
                    "contact": "+919876543210", "evil": "dropped"}

    def run_tool(self, name, args, ctx=None):
        meta = self.meta("receptionist", self.ctx if ctx is None else ctx)
        ts = agent_runtime.get("receptionist")
        return asyncio.run(agent_runtime.run_tool(ts, agent_runtime.tool_ctx(meta), name, args))

    def test_ctx_whitelist(self):
        m = self.meta("receptionist", self.ctx)
        self.assertNotIn("evil", m["ctx"])
        self.assertEqual(m["ctx"]["assignmentCode"], self.code)

    def test_email_goes_only_to_assignment_teacher(self):
        sent = []

        async def fake_email(to, subject, html, text):
            sent.append(to)
            return {"ok": True, "remaining": 90}

        with mock.patch.object(gcal_script, "send_email", fake_email):
            res = self.run_tool("email_report_to_teacher", {"to": "attacker@evil.test", "note": "hi"})
            again = self.run_tool("email_report_to_teacher", {})
        self.assertTrue(res["ok"])
        self.assertEqual(sent, [TEACHER_EMAIL])  # deduped: one email
        self.assertTrue(again["alreadyDone"])
        self.assertEqual(res["teacherName"], "Dr. Rao")
        self.assertIn("report_emailed", [e["type"] for e in self.events()])

    def test_email_refuses_unknown_assignment(self):
        other = asg.create_assignment(asg.AssignmentIn(title="Other class", items=[asg.ItemIn(kind="case", caseId="cv-002")],
                                                       teacherEmail="x@school.test"))
        with mock.patch.object(gcal_script, "send_email", mock.AsyncMock(side_effect=AssertionError("sent"))):
            res = self.run_tool("email_report_to_teacher", {"code": other["code"]})
        self.assertFalse(res["ok"])

    def test_status_lookup(self):
        res = self.run_tool("get_assignment_status", {})
        self.assertTrue(res["ok"])
        self.assertEqual(res["title"], "Cardio week")
        self.assertEqual(res["total"], 1)

    def test_book_study_session_calendar(self):
        os.environ.update(GCAL_SCRIPT_URL="https://script.test", GCAL_SCRIPT_SECRET="s")
        book = mock.AsyncMock(return_value={"ok": True, "htmlLink": "https://cal/x"})
        with mock.patch.object(gcal_script, "book_event", book):
            res = self.run_tool("book_study_session", {"topic": "cardiology", "when_text": "tomorrow evening", "minutes": 60})
        self.assertTrue(res["ok"])
        self.assertTrue(res["calendarBooked"])
        title, _desc, start = book.await_args.args[:3]
        self.assertEqual(title, "Study: cardiology")
        self.assertEqual(start.hour, 18)
        self.assertEqual(book.await_args.kwargs["minutes"], 60)
        cal = [e for e in self.events() if e["type"] == "calendar"]
        self.assertTrue(cal[0]["data"]["booked"])

    def test_book_without_calendar_reports_not_connected(self):
        res = self.run_tool("book_study_session", {"topic": "renal", "when_text": "friday 7 pm"})
        self.assertFalse(res["ok"])
        self.assertFalse(res["calendarBooked"])
        self.assertIn("startText", res)

    def test_send_to_my_phone_uses_ctx_contact_only(self):
        sent = []

        async def fake_send(channel, to, message, *a, **k):
            sent.append((channel, to, message))
            return {"ok": True, "channel": "whatsapp"}

        with mock.patch.object(notify, "send_to_patient", fake_send):
            res = self.run_tool("send_to_my_phone", {"text": "Revise cardiology at 6 pm", "to": "+911111111111"})
            dup = self.run_tool("send_to_my_phone", {"text": "Revise cardiology at 6 pm"})
            none = self.run_tool("send_to_my_phone", {"text": "hello"}, ctx={**self.ctx, "contact": "not a phone"})
        self.assertTrue(res["ok"])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1], "+919876543210")
        self.assertIn("Revise cardiology at 6 pm", sent[0][2])
        self.assertTrue(dup["alreadyDone"])
        self.assertFalse(none["ok"])

    def test_chained_book_and_text_in_one_turn(self):
        os.environ.update(GCAL_SCRIPT_URL="https://script.test", GCAL_SCRIPT_SECRET="s")
        sent = []

        async def fake_send(channel, to, message, *a, **k):
            sent.append(to)
            return {"ok": True, "channel": "whatsapp"}

        self.replies = [tool_msg(("book_study_session", {"topic": "cardiology", "when_text": "tomorrow evening"})),
                        tool_msg(("send_to_my_phone", {"text": "Cardiology revision tomorrow 6 to 7 pm"})),
                        text_msg("Booked and texted.")]
        with mock.patch.object(gcal_script, "book_event", mock.AsyncMock(return_value={"ok": True})), \
                mock.patch.object(notify, "send_to_patient", fake_send):
            r = self.turn(llm_proxy.mint_token(self.meta("receptionist", self.ctx)),
                          "Priya, I'm free tomorrow evening, can you block an hour to revise cardiology and text me the plan?")
        self.assertTrue(sse_text(r.text).endswith("Booked and texted."))
        self.assertEqual(sent, ["+919876543210"])
        tools = [e["data"]["tool"] for e in self.events() if e["type"] == "tool"]
        self.assertEqual(tools, ["book_study_session", "send_to_my_phone"])
        # The next turn is told what was already done.
        self.replies = [text_msg("Anything else?")]
        self.turn(llm_proxy.mint_token(self.meta("receptionist", self.ctx)), "thanks")
        notes = [m["content"] for m in self.seen[-1]["messages"] if m["role"] == "system"]
        self.assertTrue(any("Actions already taken" in n and "Sent to" in n for n in notes))


# ---------------------------------------------------------------- teacher

CATALOG = {"cardiology": [["cv-001", "Ravi, 54M: chest pain"], ["cv-002", "Meena, 61F: breathless"]],
           "neurology": [["neu-001", "Arun, 30M: headache"]]}


class TeacherTests(Base):
    def run_tool(self, name, args, ctx=None):
        meta = self.meta("teacher", ctx if ctx is not None else {"teacherEmail": TEACHER_EMAIL, "catalog": CATALOG})
        return asyncio.run(agent_runtime.run_tool(agent_runtime.get("teacher"), agent_runtime.tool_ctx(meta), name, args))

    def test_draw_on_board_event(self):
        res = self.run_tool("draw_on_board", {"topic": "ECG basics", "preset": "ECG Normal"})
        self.assertTrue(res["ok"])
        wb = [e for e in self.events() if e["type"] == "whiteboard"]
        self.assertEqual(wb[0]["data"], {"topic": "ECG basics", "preset": "ecg-normal"})
        res = self.run_tool("draw_on_board", {"topic": "Kidney", "preset": "made-up"})
        self.assertIsNone(res["preset"])
        self.assertTrue(self.run_tool("draw_on_board", {"topic": "ECG basics"})["alreadyDone"])

    def test_start_quiz_event(self):
        res = self.run_tool("start_quiz", {"question": "Which wave is atrial depolarisation?", "options": ["P", "QRS", "T"],
                                           "answer": 0, "why": "P wave"})
        self.assertTrue(res["ok"])
        q = [e for e in self.events() if e["type"] == "quiz"][0]["data"]
        self.assertEqual(q["answerIndex"], 0)
        self.assertEqual(q["options"], ["P", "QRS", "T"])
        bad = self.run_tool("start_quiz", {"question": "x?", "options": ["only one"], "answer": 0})
        self.assertFalse(bad["ok"])

    def test_assign_practice_creates_assignment_for_ctx_teacher(self):
        res = self.run_tool("assign_practice", {"specialty": "Cardiology", "due_text": "next week"})
        self.assertTrue(res["ok"])
        a = asg.get_assignment(res["code"])
        self.assertEqual(a["teacherEmail"], TEACHER_EMAIL)
        self.assertEqual([i["caseId"] for i in a["items"]], ["cv-001", "cv-002"])
        ev = [e for e in self.events() if e["type"] == "assignment"][0]["data"]
        self.assertEqual(ev["code"], res["code"])
        self.assertEqual(ev["link"], f"/?assignment={res['code']}")

    def test_assign_practice_checks_case_ids_and_defaults_email(self):
        bad = self.run_tool("assign_practice", {"case_ids": ["zz-999"]})
        self.assertFalse(bad["ok"])
        res = self.run_tool("assign_practice", {"case_ids": ["neu-001"], "in_days": 3},
                            ctx={"teacherEmail": "not-an-email", "catalog": CATALOG})
        self.assertTrue(res["ok"])
        self.assertEqual(asg.get_assignment(res["code"])["teacherEmail"], asg.DEFAULT_TEACHER_EMAIL)

    def test_schedule_revision(self):
        os.environ.update(GCAL_SCRIPT_URL="https://script.test", GCAL_SCRIPT_SECRET="s")
        book = mock.AsyncMock(return_value={"ok": True})
        with mock.patch.object(gcal_script, "book_event", book):
            res = self.run_tool("schedule_revision", {"topic": "renal physiology", "when_text": "sunday morning"})
        self.assertTrue(res["calendarBooked"])
        self.assertEqual(book.await_args.args[2].hour, 9)
        self.assertEqual(book.await_args.args[2].weekday(), 6)


# ---------------------------------------------------------------- viva

PLAN = [{"q": "Define shock", "difficulty": 1}, {"q": "Compensated vs decompensated?", "difficulty": 2},
        {"q": "Vasopressor choice in septic shock?", "difficulty": 3}]


class VivaTests(Base):
    def run_tool(self, name, args, ctx=None):
        meta = self.meta("viva-examiner", ctx if ctx is not None else {"subject": "Medicine", "year": "3", "plan": PLAN})
        return asyncio.run(agent_runtime.run_tool(agent_runtime.get("viva-examiner"), agent_runtime.tool_ctx(meta), name, args))

    def test_grade_event_and_running_average(self):
        r1 = self.run_tool("record_answer_grade", {"question": "Define shock", "score": 8, "key_points_hit": ["hypoperfusion"],
                                                   "missed": ["lactate"]})
        self.assertEqual(r1["runningAverage"], 8)
        r2 = self.run_tool("record_answer_grade", {"question": "Define  shock!", "score": 6})  # re-mark, no double count
        self.assertEqual(r2["recorded"], 1)
        self.assertEqual(self.run_tool("record_answer_grade", {"question": "x", "score": 42})["recorded"], 2)  # clamped
        grades = [e["data"] for e in self.events() if e["type"] == "viva_grade"]
        self.assertEqual(grades[0]["hit"], ["hypoperfusion"])
        self.assertEqual(grades[-1]["score"], 10)

    def test_next_question_adapts_and_end_viva_once(self):
        q = self.run_tool("next_question", {"difficulty": "hard"})
        self.assertEqual(q["index"], 2)
        q = self.run_tool("next_question", {"difficulty": "hard"})  # none left at 3: nearest
        self.assertEqual(q["difficulty"], 2)
        self.assertEqual([e["data"]["index"] for e in self.events() if e["type"] == "viva_question"], [2, 1])
        end = self.run_tool("end_viva", {"summary": "Solid basics"})
        self.assertTrue(end["ended"])
        self.assertTrue(self.run_tool("end_viva", {})["alreadyEnded"])
        self.assertFalse(self.run_tool("next_question", {"difficulty": "easy"})["ok"])
        self.assertEqual(len([e for e in self.events() if e["type"] == "viva_end"]), 1)

    def test_next_question_without_plan(self):
        q = self.run_tool("next_question", {"difficulty": "easy"}, ctx={"subject": "Pharmacology"})
        self.assertIsNone(q["question"])
        self.assertIn("Pharmacology", q["guidance"])

    def test_viva_has_no_fillers(self):
        self.replies = [tool_msg(("record_answer_grade", {"question": "Define shock", "score": 7})),
                        text_msg("Good. Now, tell me about compensated shock.")]
        r = self.turn(llm_proxy.mint_token(self.meta("viva-examiner", {"plan": PLAN})), "Shock is hypoperfusion")
        self.assertEqual(sse_text(r.text), "Good. Now, tell me about compensated shock.")


# ---------------------------------------------------------------- event bus

class EventBusTests(Base):
    def test_ordering_cursor_and_limits(self):
        for i in range(5):
            agent_events.publish(self.channel, "t", {"i": i})
        page = agent_events.since(self.channel, 0, limit=2)
        self.assertEqual([e["data"]["i"] for e in page["events"]], [0, 1])
        page = agent_events.since(self.channel, page["next"])
        self.assertEqual([e["seq"] for e in page["events"]], [3, 4, 5])
        self.assertEqual(agent_events.since(self.channel, 5)["events"], [])
        self.assertEqual(agent_events.since(self.channel, 99)["next"], 5)  # stale cursor resets
        for i in range(agent_events.MAX_EVENTS + 10):
            agent_events.publish(self.channel, "t", {"i": i})
        allev = agent_events.since(self.channel, 0, limit=10_000)
        self.assertEqual(len(allev["events"]), agent_events.MAX_PAGE)
        self.assertEqual(allev["events"][0]["seq"], 5 + 10 + 1)  # oldest dropped
        self.assertIsNone(agent_events.publish("bad channel!", "t"))
        self.assertEqual(agent_events.since("other", 0), {"events": [], "next": 0})

    def test_channel_cap(self):
        with mock.patch.object(agent_events, "MAX_CHANNELS", 3):
            for i in range(5):
                agent_events.publish(f"ch-{i}", "t")
            self.assertEqual(list(agent_events._channels), ["ch-2", "ch-3", "ch-4"])

    def test_endpoint_is_guarded(self):
        agent_events.publish(self.channel, "quiz", {"q": 1})
        r = self.client.get(f"/agent/agent-events/{self.channel}?after=0", headers=DEV)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["events"][0]["type"], "quiz")
        self.assertEqual(self.client.get(f"/agent/agent-events/{self.channel}").status_code, 401)


# ---------------------------------------------------------------- session wiring

class WiringTests(Base):
    def start(self, body: dict):
        return self.client.post("/voice/start", json=body, headers=DEV)

    def test_receptionist_start_tags_token_with_ctx(self):
        r = self.start({"caseId": "receptionist", "systemPrompt": "Priya", "initialLine": "Namaste", "gender": "F",
                        "agentContext": {"assignmentCode": "ABC234", "knownCodes": ["ABC234"], "junk": 1}})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["toolsEnabled"])
        self.assertEqual(data["mode"], "gemini_live")
        prep = gemini_live.get_prepared_session(data["channel"])
        self.assertIsNotNone(prep)
        self.assertEqual(prep.agent_kind, "receptionist")
        self.assertEqual(prep.agent_ctx, {"assignmentCode": "ABC234", "knownCodes": ["ABC234"]})
        # /voice/context merges new ctx
        c = self.client.post("/voice/context", json={"agentId": data["channel"], "agentContext": {"contact": "+919876543210"}},
                             headers=DEV)
        self.assertEqual(c.json(), {"ok": True, "ctxUpdated": True})
        self.assertEqual(prep.agent_ctx["contact"], "+919876543210")
        self.assertEqual(prep.agent_ctx["assignmentCode"], "ABC234")

    def test_teacher_start_tools_enabled(self):
        r = self.start({"caseId": "teacher", "systemPrompt": "Iyer", "initialLine": "Hello", "gender": "F"})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["toolsEnabled"])
        prep = gemini_live.get_prepared_session(data["channel"])
        self.assertIsNotNone(prep)
        self.assertEqual(prep.agent_kind, "teacher")

    def test_patient_start_unchanged(self):
        r = self.start({"caseId": "im-001", "systemPrompt": "p", "initialLine": "hi", "gender": "M",
                        "agentContext": {"x": 1}})
        self.assertEqual(r.status_code, 200)
        prep = gemini_live.get_prepared_session(r.json()["channel"])
        self.assertIsNotNone(prep)
        self.assertIsNone(prep.agent_kind)

    def test_oversized_ctx_rejected(self):
        r = self.start({"caseId": "viva-examiner", "systemPrompt": "p", "initialLine": "hi", "gender": "M",
                        "agentContext": {"plan": [{"q": "x" * 400, "difficulty": 1}] * 40}})
        self.assertEqual(r.status_code, 422)


class WhenTests(unittest.TestCase):
    NOW = datetime(2026, 11, 21, 10, 0, tzinfo=IST)  # a Saturday

    def test_words(self):
        rw = agent_toolsets.resolve_when
        self.assertEqual(rw("tomorrow evening", now=self.NOW), datetime(2026, 11, 22, 18, 0, tzinfo=IST))
        self.assertEqual(rw("friday 7 pm", now=self.NOW), datetime(2026, 11, 27, 19, 0, tzinfo=IST))
        self.assertEqual(rw("monday morning", now=self.NOW), datetime(2026, 11, 23, 9, 0, tzinfo=IST))
        self.assertEqual(rw("in 3 days", now=self.NOW).day, 24)
        self.assertEqual(rw("this morning", now=self.NOW), datetime(2026, 11, 22, 9, 0, tzinfo=IST))  # past -> next day
        self.assertEqual(rw("whenever", in_days=2, hour=15, now=self.NOW), datetime(2026, 11, 23, 15, 0, tzinfo=IST))


if __name__ == "__main__":
    unittest.main()
