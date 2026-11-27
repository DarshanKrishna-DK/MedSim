"""Invite an observer by name and Asha phones them: Twilio request shape (Url,
never Twiml), TwiML shape, X-Twilio-Signature, the speech turn loop (LLM and
Murf mocked), stored messages, caps and the non-patient command regex.
No network: Twilio, Telegram, Murf and the LLM are all mocked."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qsl, urlencode

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import observer  # noqa: E402
import observer_call as oc  # noqa: E402
import server  # noqa: E402
from llm_proxy import addressed_to_nurse, addressed_to_observer_call  # noqa: E402

HDR = {"Origin": "http://localhost:5173"}
PUB = "https://medsim.example.com"
ENV = {
    "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
    "TWILIO_AUTH_TOKEN": "twilio-test-token",
    "TWILIO_VOICE_FROM": "+17372508034",
    "GEMINI_API_KEY": "fake-gemini-key",
    "MURF_API_KEY": "murf-test",
    "SARVAM_API_KEY": "sarvam-test",
    "TELEGRAM_BOT_TOKEN": "",
    "TWILIO_SMS_FROM": "",
    "TWILIO_WHATSAPP_CONTENT_SID": "",
    "PUBLIC_URL": PUB,
}


def make_invite() -> str:
    code = observer._new_code()
    now = time.time()
    observer._invites[code] = observer.Invite(
        code=code, channel="ch-1", case_id="im-001", student_uid=1, link_uid=3_000_001, agent_uid=900001,
        created=now, expires=now + observer.INVITE_TTL_S,
    )
    return code


def body(code: str, **over):
    b = {
        "contact": {"name": "Darshan Kumar", "relation": "teacher", "phone": "9876543210", "telegram": "@darshan_k"},
        "inviteUrl": f"http://localhost:5173/?observe={code}",
        "student": {"name": "Ananya Rao", "university": "RGUHS", "gender": "female"},
        "case": {"specialty": "cardiology", "specialtyLabel": "Cardiology", "chiefComplaint": "chest pain on exertion",
                 "caseId": "im-001"},
        "available": False,
    }
    b.update(over)
    return b


def sign(url: str, params: dict[str, str]) -> str:
    data = url + "".join(k + v for k, v in sorted(params.items()))
    return base64.b64encode(hmac.new(ENV["TWILIO_AUTH_TOKEN"].encode(), data.encode(), hashlib.sha1).digest()).decode()


class Base(unittest.TestCase):
    def setUp(self):
        oc.reset_state()
        observer.reset_state()
        self.env = mock.patch.dict(os.environ, ENV)
        self.env.start()
        self.twilio_reqs: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            self.twilio_reqs.append(req)
            return httpx.Response(201, json={"sid": "CA123", "status": "queued"})

        self.patches = [
            mock.patch.object(oc, "_client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))),
            mock.patch.object(oc.notify, "send_to_patient",
                              mock.AsyncMock(return_value={"ok": True, "channel": "telegram", "to": "Telegram chat"})),
            mock.patch("rx_audio.synthesize", mock.AsyncMock(return_value=(b"ID3mp3", "en-IN-anisha", "en-IN"))),
        ]
        for p in self.patches:
            p.start()
        self.chat = mock.AsyncMock(return_value=({"content": json.dumps({"reply": "Sure.", "message_for_student": "",
                                                                          "end_call": False})}, "sarvam"))
        self.chat_patch = mock.patch.object(oc, "chat", self.chat)
        self.chat_patch.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.chat_patch.stop()
        for p in self.patches:
            p.stop()
        self.env.stop()
        oc.reset_state()

    def start_call(self, **over) -> dict:
        r = self.client.post("/agent/observer/call", json=body(make_invite(), **over), headers=HDR)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def hook(self, token: str, leaf: str, params: dict[str, str], signature: str | None = None):
        url = f"{PUB}/rx/obs-call/{token}/{leaf}"
        sig = sign(url, params) if signature is None else signature
        return self.client.post(f"/rx/obs-call/{token}/{leaf}", content=urlencode(params),
                                headers={"Content-Type": "application/x-www-form-urlencoded", "X-Twilio-Signature": sig})


class CallStartTests(Base):
    def test_twilio_request_uses_url_not_twiml(self):
        res = self.start_call()
        self.assertEqual(res["status"], "queued")
        self.assertEqual(len(self.twilio_reqs), 1)
        req = self.twilio_reqs[0]
        self.assertTrue(str(req.url).endswith(f"/Accounts/{ENV['TWILIO_ACCOUNT_SID']}/Calls.json"))
        form = parse_qsl(req.content.decode())
        keys = [k for k, _ in form]
        d = dict(form)
        self.assertNotIn("Twiml", keys)
        self.assertEqual(d["Url"], f"{PUB}/rx/obs-call/{res['token']}/twiml")
        self.assertEqual(d["Method"], "POST")
        self.assertEqual(d["From"], "+17372508034")
        self.assertEqual(d["To"], "+919876543210")
        self.assertEqual(d["StatusCallback"], f"{PUB}/rx/obs-call/{res['token']}/status")
        self.assertIn("completed", [v for k, v in form if k == "StatusCallbackEvent"])
        self.assertTrue(req.headers["authorization"].startswith("Basic "))

    def test_link_sent_on_telegram_with_public_url(self):
        res = self.start_call()
        send = oc.notify.send_to_patient
        ch, dest, msg = send.await_args.args[:3]
        self.assertEqual((ch, dest), ("telegram", "@darshan_k"))
        self.assertIn("Dr. Ananya Rao invited you to observe a live MedSim consultation (Cardiology). Join: "
                      f"{PUB}/?observe=", msg)
        self.assertNotIn("localhost", msg)
        self.assertNotIn("\u2014", msg)
        self.assertEqual(res["spoken"], "Calling Darshan now. I've sent the link on Telegram.")
        self.assertEqual(res["contact"]["phone"], "+91******3210")

    def test_invalid_or_expired_invite_rejected(self):
        r = self.client.post("/agent/observer/call", json=body("bogus.code"), headers=HDR)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.twilio_reqs, [])

    def test_bad_phone_rejected(self):
        b = body(make_invite())
        b["contact"]["phone"] = "12"
        self.assertEqual(self.client.post("/agent/observer/call", json=b, headers=HDR).status_code, 400)

    def test_agent_route_is_origin_guarded(self):
        r = self.client.post("/agent/observer/call", json=body(make_invite()))
        self.assertEqual(r.status_code, 401)

    def test_per_contact_cap(self):
        for _ in range(oc.PER_CONTACT_PER_HOUR):
            self.start_call()
        r = self.client.post("/agent/observer/call", json=body(make_invite()), headers=HDR)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(len(self.twilio_reqs), oc.PER_CONTACT_PER_HOUR)

    def test_global_cap(self):
        with mock.patch.object(oc, "GLOBAL_PER_DAY", 2):
            self.start_call()
            b = body(make_invite())
            b["contact"]["phone"] = "+919811111111"
            self.assertEqual(self.client.post("/agent/observer/call", json=b, headers=HDR).status_code, 200)
            b["contact"]["phone"] = "+919822222222"
            b["inviteUrl"] = f"http://localhost:5173/?observe={make_invite()}"
            self.assertEqual(self.client.post("/agent/observer/call", json=b, headers=HDR).status_code, 429)

    def test_twilio_error_surfaces(self):
        def handler(req):
            return httpx.Response(400, json={"message": "The number is unverified. Trial accounts cannot call unverified numbers"})

        with mock.patch.object(oc, "_client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))):
            res = self.start_call()
        self.assertEqual(res["status"], "failed")
        self.assertIn("unverified", res["error"])
        self.assertEqual(res["spoken"], "I couldn't place the call to Darshan, but I've sent the link on Telegram.")

    def test_no_public_url_means_no_call(self):
        with mock.patch.dict(os.environ, {"PUBLIC_URL": ""}), mock.patch("llm_proxy._state", {"url": None, "verified": False}):
            res = self.start_call()
        self.assertEqual(res["status"], "failed")
        self.assertIn("public URL", res["error"])
        self.assertEqual(self.twilio_reqs, [])


class TwimlTests(Base):
    def test_twiml_shape(self):
        tok = self.start_call()["token"]
        r = self.hook(tok, "twiml", {"CallSid": "CA123", "CallStatus": "in-progress"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("application/xml", r.headers["content-type"])
        x = r.text
        # Barge-in: the opener <Play> is nested INSIDE <Gather>, so speech interrupts it.
        gather = (f'<Gather input="speech" language="en-IN" speechTimeout="auto" actionOnEmptyResult="true" '
                  f'hints="{oc.SPEECH_HINTS}" action="{PUB}/rx/obs-call/{tok}/turn" method="POST"><Play>{PUB}/rx/audio/')
        self.assertTrue(x.startswith('<?xml version="1.0" encoding="UTF-8"?><Response>' + gather), x)
        self.assertRegex(x, r"<Gather [^>]*><Play>[^<]+</Play></Gather><Play>[^<]+</Play></Response>$")
        self.assertNotIn("<Pause", x)
        status = self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()
        self.assertEqual(status["status"], "in-progress")
        opener = status["transcript"][0]["text"]
        self.assertEqual(
            opener,
            "Hi Darshan, it's Asha from MedSim. Ananya invited you, as her teacher, to watch a live Cardiology consult. "
            "Link's on Telegram, shall I add it to your calendar too?",
        )
        self.assertLessEqual(len(re.findall(r"[.!?](?:\s|$)", opener)), 3)
        self.assertEqual(oc.schedule_intent("yes please", status["transcript"]), "now")  # the opener offers booking

    def test_say_fallback_when_murf_fails(self):
        with mock.patch("rx_audio.synthesize", mock.AsyncMock(side_effect=RuntimeError("murf down"))):
            tok = self.start_call()["token"]
        x = self.hook(tok, "twiml", {"CallSid": "CA123"}).text
        self.assertIn('method="POST"><Say language="en-IN">Hi Darshan', x)  # Say also nested for barge-in

    def test_signature_required_and_checked(self):
        os.environ["TWILIO_STRICT_SIGNATURE"] = "1"  # removed by self.env.stop() in tearDown
        tok = self.start_call()["token"]
        self.assertEqual(self.hook(tok, "twiml", {"CallSid": "CA123"}, signature="").status_code, 403)
        self.assertEqual(self.hook(tok, "twiml", {"CallSid": "CA123"}, signature="AAAA").status_code, 403)
        # Signed for different params: rejected.
        bad = sign(f"{PUB}/rx/obs-call/{tok}/twiml", {"CallSid": "CA999"})
        self.assertEqual(self.hook(tok, "twiml", {"CallSid": "CA123"}, signature=bad).status_code, 403)

    def test_signature_algorithm(self):
        params = [("To", "+1800"), ("CallSid", "CA1"), ("From", "+1234")]
        expect = base64.b64encode(hmac.new(b"tok", b"https://x.test/hookCallSidCA1From+1234To+1800",
                                           hashlib.sha1).digest()).decode()
        self.assertEqual(oc.twilio_signature("tok", "https://x.test/hook", params), expect)

    def test_unknown_token_404(self):
        self.assertEqual(self.hook("A" * 24, "twiml", {}).status_code, 404)

    def test_status_callback(self):
        tok = self.start_call()["token"]
        self.assertEqual(self.hook(tok, "status", {"CallStatus": "ringing"}).status_code, 204)
        self.assertEqual(self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()["status"], "ringing")
        self.hook(tok, "status", {"CallStatus": "completed"})
        self.hook(tok, "status", {"CallStatus": "in-progress"})  # late event never revives a finished call
        self.assertEqual(self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()["status"], "completed")


class TurnTests(Base):
    def reply(self, reply, message="", end=False):
        self.chat.return_value = ({"content": json.dumps({"reply": reply, "message_for_student": message,
                                                          "end_call": end})}, "sarvam")

    def test_role_answer_uses_context_only(self):
        tok = self.start_call()["token"]
        self.reply("As her teacher you can follow the transcript and chart live, and send private notes. Anything else?")
        r = self.hook(tok, "turn", {"SpeechResult": "What can I do as an observer?", "Confidence": "0.9"})
        self.assertEqual(r.status_code, 200)
        self.assertRegex(r.text, rf'<Gather [^>]*method="POST"><Play>{re.escape(PUB)}/rx/audio/[^<]+</Play></Gather>')
        system = self.chat.await_args.args[0][0]["content"]
        self.assertIn("Do not repeat information you already gave", system)
        self.assertIn("Never say you did not understand", system)
        self.assertIn("link bhej do", system)
        self.assertNotIn("End with a short question like 'Anything else?'", system)
        self.assertIn(oc.ROLE_TEXT["teacher"], system)
        self.assertIn("chest pain on exertion", system)
        self.assertIn("Pronouns: she/her", system)
        self.assertIn("not available to talk", system)
        self.assertIn("listen only", system)
        self.assertNotIn("\u2014", system)
        self.assertTrue(self.chat.await_args.kwargs["json_mode"])
        tr = self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()["transcript"]
        self.assertEqual(tr[-2], {"who": "observer", "text": "What can I do as an observer?", "at": tr[-2]["at"]})
        self.assertEqual(tr[-1]["who"], "asha")
        self.assertEqual(tr[-1]["text"], "As her teacher you can follow the transcript and chart live, and send private notes.")

    def test_link_request_answers_without_llm(self):
        tok = self.start_call()["token"]
        for said in ("send me the link", "can you resend the link", "link bhej do"):
            r = self.hook(tok, "turn", {"SpeechResult": said})
            self.assertIn("<Gather", r.text)
            tr = self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()["transcript"]
            self.assertIn("Telegram", tr[-1]["text"])
            self.assertIn("calendar invite", tr[-1]["text"])
        self.chat.assert_not_awaited()
        # "yes" after the link offer books it (deterministic, not the model).
        tr = self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()["transcript"]
        self.assertEqual(oc.schedule_intent("haan", tr), "now")

    def test_family_role_text(self):
        b = body(make_invite())
        b["contact"]["relation"] = "family"
        tok = self.client.post("/agent/observer/call", json=b, headers=HDR).json()["token"]
        self.hook(tok, "turn", {"SpeechResult": "Can I talk to her?"})
        system = self.chat.await_args.args[0][0]["content"]
        self.assertIn(oc.ROLE_TEXT["family"], system)
        self.assertNotIn(oc.ROLE_TEXT["teacher"], system)

    def test_not_available_and_message_stored(self):
        tok = self.start_call()["token"]
        self.reply("She is with a patient right now, but I can pass on a message. What should I tell her?")
        self.hook(tok, "turn", {"SpeechResult": "Can I speak to Ananya?"})
        self.assertEqual(self.client.get(f"/agent/observer/call/{tok}/events", headers=HDR).json()["messages"], [])
        self.reply("Done, I will tell her. Anything else?", message="All the best, I will join in five minutes")
        self.hook(tok, "turn", {"SpeechResult": "Tell her all the best, I will join in five minutes"})
        ev = self.client.get(f"/agent/observer/call/{tok}/events", headers=HDR).json()
        self.assertEqual(len(ev["messages"]), 1)
        self.assertEqual(ev["messages"][0]["text"], "All the best, I will join in five minutes")
        self.assertEqual(ev["messages"][0]["from"], "Darshan Kumar")
        after = self.client.get(f"/agent/observer/call/{tok}/events?after=1", headers=HDR).json()
        self.assertEqual(after["messages"], [])

    def test_goodbye_hangs_up_without_llm(self):
        tok = self.start_call()["token"]
        r = self.hook(tok, "turn", {"SpeechResult": "No, thank you. Bye."})
        self.assertIn("<Hangup/>", r.text)
        self.assertNotIn("<Gather", r.text)
        self.chat.assert_not_awaited()

    def test_llm_end_call_hangs_up(self):
        tok = self.start_call()["token"]
        self.reply("You're welcome. Goodbye!", end=True)
        r = self.hook(tok, "turn", {"SpeechResult": "Alright that is everything I needed to know"})
        self.assertIn("<Hangup/>", r.text)

    def test_silence_twice_hangs_up(self):
        tok = self.start_call()["token"]
        first = self.hook(tok, "turn", {"SpeechResult": ""})
        self.assertIn("<Gather", first.text)
        second = self.hook(tok, "turn", {})
        self.assertIn("<Hangup/>", second.text)
        self.assertNotIn("<Gather", second.text)

    def test_turn_cap(self):
        tok = self.start_call()["token"]
        for i in range(oc.MAX_TURNS - 1):
            self.assertIn("<Gather", self.hook(tok, "turn", {"SpeechResult": f"Question number {i}?"}).text)
        last = self.hook(tok, "turn", {"SpeechResult": "One more question?"})
        self.assertIn("<Hangup/>", last.text)
        self.assertEqual(self.chat.await_count, oc.MAX_TURNS)

    def test_llm_timeout_falls_back_to_canned_line(self):
        tok = self.start_call()["token"]
        self.chat.side_effect = asyncio.TimeoutError()
        r = self.hook(tok, "turn", {"SpeechResult": "How do I join?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("<Gather", r.text)
        tr = self.client.get(f"/agent/observer/call/{tok}", headers=HDR).json()["transcript"]
        self.assertEqual(tr[-1]["text"], oc.CANNED_TROUBLE)

    def test_reply_is_cleaned(self):
        out = oc._parse_reply('<think>x</think>{"reply": "Open the link \u2014 no app **needed**.", "end_call": false}')
        self.assertEqual(out["reply"], "Open the link, no app needed.")
        self.assertEqual(oc._parse_reply("Plain text answer.")["reply"], "Plain text answer.")

    def test_reply_word_cap_keeps_whole_sentences(self):
        long = ("Sure, you just open the link on your phone browser. It's listen only so nobody hears you at all. "
                "You can also send her a quick tip that pops up on her screen while she works.")
        out = oc._parse_reply(json.dumps({"reply": long}))["reply"]
        self.assertLessEqual(len(out.split()), oc.MAX_REPLY_WORDS)
        self.assertEqual(out, "Sure, you just open the link on your phone browser. It's listen only so nobody hears you at all.")
        one = oc.cap_reply(" ".join(["word"] * 40))
        self.assertEqual(len(one.split()), oc.MAX_REPLY_WORDS)
        self.assertTrue(one.endswith("."))

    def test_no_trailing_anything_else(self):
        for tail in ("Anything else?", "Is there anything else I can help you with?", "Do you have any other questions?"):
            self.assertEqual(oc.cap_reply(f"Got it, I'll tell her. {tail}"), "Got it, I'll tell her.")
        self.assertEqual(oc.cap_reply("Anything else?"), "Anything else?")  # never blank the reply
        for line in (oc.CANNED_TROUBLE, oc.REPROMPT, oc.goodbye_text({"student": {"name": "Ananya"},
                                                                         "contact": {"name": "Darshan"}})):
            self.assertNotIn("anything else", line.lower())

    def test_hangup_twiml_unchanged(self):
        x = oc.twiml_hangup("Bye.", "https://p/a.mp3", "See you.", None)
        self.assertEqual(x, '<?xml version="1.0" encoding="UTF-8"?><Response><Play>https://p/a.mp3</Play>'
                            '<Say language="en-IN">See you.</Say><Hangup/></Response>')

    def test_mismatch_allowed_without_strict_mode(self):
        tok = self.start_call()["token"]
        self.assertEqual(self.hook(tok, "twiml", {"CallSid": "CA123"}, signature="AAAA").status_code, 200)

    def test_turn_signature_checked(self):
        os.environ["TWILIO_STRICT_SIGNATURE"] = "1"  # removed by self.env.stop() in tearDown
        tok = self.start_call()["token"]
        r = self.hook(tok, "turn", {"SpeechResult": "hi"}, signature=sign(f"{PUB}/rx/obs-call/{tok}/turn", {"SpeechResult": "x"}))
        self.assertEqual(r.status_code, 403)
        self.chat.assert_not_awaited()


class HelperTests(unittest.TestCase):
    def test_goodbye_detection(self):
        for t in ("No", "no thanks", "Thank you, bye", "That's all", "nothing, thank you so much", "bye bye"):
            self.assertTrue(oc.is_goodbye(t), t)
        for t in ("How do I join?", "okay", "Thanks, what time does it end", "Can I talk to her", ""):
            self.assertFalse(oc.is_goodbye(t), t)

    def test_pronouns(self):
        self.assertEqual(oc.pronouns("male"), ("he", "his"))
        self.assertEqual(oc.pronouns("other"), ("they", "their"))


class ObserverCommandRegexTests(unittest.TestCase):
    def user(self, text):
        return [{"role": "system", "content": "You are Ravi, a 52-year-old"}, {"role": "user", "content": text}]

    def test_observer_commands_silence_the_patient(self):
        for t in ("Invite Darshan", "invite Darshan to observe", "Okay, call Darshan to watch",
                  "please ask Darshan Kumar to join", "Can you invite Priya now", "Asha, ask Darshan to join",
                  "ring Meera to observe."):
            self.assertTrue(addressed_to_observer_call(t) or addressed_to_nurse(self.user(t)), t)
            self.assertTrue(addressed_to_nurse(self.user(t)), t)

    def test_patient_lines_still_reach_the_patient(self):
        for t in ("Call me if it gets worse", "Invite your husband in", "Ask your mother to join us",
                  "Can you call the lab results up", "I will invite you back next week",
                  "Did you call anyone?", "My friend invited me to watch a movie"):
            self.assertFalse(addressed_to_nurse(self.user(t)), t)


if __name__ == "__main__":
    unittest.main()


class ScheduleIntentTests(unittest.TestCase):
    """Booking must not depend on the model: speech recognition garbles words."""

    OFFER = [{"who": "asha", "text": "I can put the joining link straight into your calendar, shall I?"}]

    def test_garbled_and_affirmative_requests_book(self):
        self.assertEqual(oc.schedule_intent("SP added to my calendar", self.OFFER), "now")
        self.assertEqual(oc.schedule_intent("yes please", self.OFFER), "now")
        self.assertEqual(oc.schedule_intent("send me a calendar invite", []), "now")

    def test_time_phrases(self):
        self.assertEqual(oc.schedule_intent("can we have the meeting after 5 minutes?", []), "after 5 minutes")
        self.assertEqual(oc.schedule_intent("schedule it for tomorrow 6 pm", []), "tomorrow 6 pm")

    def test_questions_do_not_book(self):
        self.assertIsNone(oc.schedule_intent("does the invite have a link?", []))
        self.assertIsNone(oc.schedule_intent("Which specialty is this?", self.OFFER))
        self.assertIsNone(oc.schedule_intent("no questions", self.OFFER))

    def test_link_intent(self):
        for t in ("send me the link", "resend link please", "link bhej do", "I didn't get the link", "where is the link"):
            self.assertTrue(oc.link_intent(t), t)
        for t in ("how do I join", "does the invite have a link?", "what time is it"):
            self.assertFalse(oc.link_intent(t), t)
