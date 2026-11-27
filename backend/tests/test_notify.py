"""Patient notifications: Twilio shape, SMTP, normalisation, masking, rate limit, audio router. No network."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from integrations import notify  # noqa: E402

TW = {"TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "tok", "TWILIO_SMS_FROM": "+15005550006",
      "TWILIO_WHATSAPP_CONTENT_SID": "", "TWILIO_WHATSAPP_FROM": ""}
SMTP = {"SMTP_USER": "doc@gmail.com", "SMTP_PASSWORD": "pw"}


def run(coro):
    return asyncio.run(coro)


class Base(unittest.TestCase):
    def setUp(self):
        notify._sends.clear()
        notify._audio.clear()
        keys = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM", "TWILIO_SMS_FROM",
                "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"]
        p = mock.patch.dict(os.environ, {k: "" for k in keys})
        p.start()
        self.addCleanup(p.stop)

    def twilio(self, status=201, body=None):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(status, json=body or {"sid": "SM123"})

        p = mock.patch.object(notify, "_client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        p.start()
        self.addCleanup(p.stop)
        return seen


class TwilioTests(Base):
    def test_whatsapp_shape(self):
        seen = self.twilio()
        with mock.patch.dict(os.environ, TW):
            r = run(notify.send_to_patient("whatsapp", "9876543210", "hi", "https://x.test/rx/audio/a.mp3"))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["sid"], "SM123")
        self.assertEqual(r["to"], "+91******3210")
        req = seen[0]
        self.assertIn("/Accounts/ACtest/Messages.json", str(req.url))
        self.assertTrue(req.headers["authorization"].startswith("Basic "))
        f = parse_qs(req.content.decode())
        self.assertEqual(f["From"], ["whatsapp:+14155238886"])
        self.assertEqual(f["To"], ["whatsapp:+919876543210"])
        self.assertEqual(f["MediaUrl"], ["https://x.test/rx/audio/a.mp3"])

    def test_whatsapp_no_media_for_http(self):
        seen = self.twilio()
        with mock.patch.dict(os.environ, TW):
            run(notify.send_to_patient("whatsapp", "+919876543210", "hi", "http://x.test/a.mp3"))
        self.assertNotIn("MediaUrl", parse_qs(seen[0].content.decode()))

    def test_sms_no_media(self):
        seen = self.twilio()
        with mock.patch.dict(os.environ, TW):
            r = run(notify.send_to_patient("sms", "+919876543210", "hi", "https://x.test/a.mp3"))
        self.assertTrue(r["ok"])
        f = parse_qs(seen[0].content.decode())
        self.assertNotIn("MediaUrl", f)
        self.assertEqual(f["From"], ["+15005550006"])
        self.assertEqual(f["To"], ["+919876543210"])

    def test_twilio_error(self):
        self.twilio(400, {"message": "bad"})
        with mock.patch.dict(os.environ, TW):
            r = run(notify.send_to_patient("whatsapp", "+919876543210", "hi"))
        self.assertFalse(r["ok"])
        self.assertIn("400", r["error"])

    def test_unconfigured(self):
        r = run(notify.send_to_patient("whatsapp", "+919876543210", "hi"))
        self.assertFalse(r["ok"])
        self.assertIn("not configured", r["error"])
        r = run(notify.send_to_patient("email", "a@b.com", "hi"))
        self.assertFalse(r["ok"])
        self.assertIn("not configured", r["error"])

    def test_auto_picks(self):
        seen = self.twilio()
        with mock.patch.dict(os.environ, TW):
            r = run(notify.send_to_patient("auto", "9876543210", "hi"))
        self.assertEqual(r["channel"], "whatsapp")
        self.assertTrue(seen)
        with mock.patch.dict(os.environ, SMTP), mock.patch.object(notify.smtplib, "SMTP_SSL"):
            r = run(notify.send_to_patient("auto", "d@gmail.com", "hi"))
        self.assertEqual(r["channel"], "email")
        self.assertTrue(r["ok"])

    def test_invalid_destinations(self):
        with mock.patch.dict(os.environ, TW):
            self.assertFalse(run(notify.send_to_patient("whatsapp", "12345", "hi"))["ok"])
        with mock.patch.dict(os.environ, SMTP):
            self.assertFalse(run(notify.send_to_patient("email", "nope", "hi"))["ok"])

    def test_rate_limit(self):
        self.twilio()
        with mock.patch.dict(os.environ, TW):
            res = [run(notify.send_to_patient("whatsapp", "+919876543210", "hi")) for _ in range(6)]
        self.assertTrue(all(r["ok"] for r in res[:5]))
        self.assertFalse(res[5]["ok"])
        self.assertIn("Rate limit", res[5]["error"])


class EmailTests(Base):
    def test_email_sends_attachments(self):
        with mock.patch.dict(os.environ, SMTP), mock.patch.object(notify.smtplib, "SMTP_SSL") as ssl_cls:
            r = run(notify.send_to_patient("email", "d@gmail.com", "body", None, b"RIFFWAVDATA", b"BEGIN:VCALENDAR"))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["to"], "d***@gmail.com")
        self.assertTrue(r["messageId"])
        ssl_cls.assert_called_once()
        self.assertEqual(ssl_cls.call_args.args[:2], ("smtp.gmail.com", 465))
        smtp = ssl_cls.return_value.__enter__.return_value
        smtp.login.assert_called_once_with("doc@gmail.com", "pw")
        msg = smtp.send_message.call_args.args[0]
        names = [p.get_filename() for p in msg.iter_attachments()]
        self.assertEqual(names, ["prescription.wav", "follow-up.ics"])
        self.assertEqual(msg["To"], "d@gmail.com")

    def test_email_failure_is_contained(self):
        with mock.patch.dict(os.environ, SMTP), mock.patch.object(notify.smtplib, "SMTP_SSL", side_effect=OSError("x")):
            r = run(notify.send_to_patient("email", "d@gmail.com", "body"))
        self.assertFalse(r["ok"])
        self.assertNotIn("pw", str(r))


class HelperTests(Base):
    def test_normalize_and_mask(self):
        self.assertEqual(notify.normalize_phone("98765 43210"), "+919876543210")
        self.assertEqual(notify.normalize_phone("+14155238886"), "+14155238886")
        self.assertIsNone(notify.normalize_phone("+123"))
        self.assertIsNone(notify.normalize_phone("abc"))
        self.assertEqual(notify.mask("+919876543210"), "+91******3210")
        self.assertEqual(notify.mask("dev@gmail.com"), "d***@gmail.com")

    def test_configured_channels(self):
        clean = {"TELEGRAM_BOT_TOKEN": "", "TWILIO_ACCOUNT_SID": "", "TWILIO_AUTH_TOKEN": "", "SMTP_USER": "", "SMTP_PASSWORD": ""}
        with mock.patch.dict(os.environ, clean):
            self.assertEqual(notify.configured_channels(), [])
        with mock.patch.dict(os.environ, {**clean, **TW, **SMTP}):
            self.assertEqual(notify.configured_channels(), ["whatsapp", "sms", "email"])

    def test_compose(self):
        meds = [{"name": "Paracetamol", "dose": "500 mg", "duration": "3 days"}]
        t = notify.compose_message("Asha", "Dr. Rao", meds, {"startText": "Tue 25 Nov 10:00 am", "reason": "review"},
                                   "https://x/a.mp3", "https://x/c", "hi")
        self.assertIn(notify.DISCLAIMER, t)
        self.assertIn("https://x/a.mp3", t)
        self.assertIn("Tue 25 Nov 10:00 am", t)
        self.assertNotIn(chr(0x2014), t)
        self.assertLessEqual(len(t), 1200)
        long = notify.compose_message("A", "B", [{"name": "x" * 70, "dose": "y" * 70, "duration": "z"}] * 30, None, None, None, "en")
        self.assertLessEqual(len(long), 1200)
        self.assertTrue(long.endswith(notify.DISCLAIMER))

    def test_no_public_url(self):
        with mock.patch.object(notify, "public_url", return_value=None):
            self.assertIsNone(notify.absolute_url("/rx/audio/a.mp3"))
        with mock.patch.object(notify, "public_url", return_value="https://t.example/"):
            self.assertEqual(notify.absolute_url("/p"), "https://t.example/p")


class AudioRouterTests(Base):
    def client(self):
        app = FastAPI()
        app.include_router(notify.router)
        return TestClient(app)

    def test_roundtrip_and_404(self):
        path = notify.store_audio(b"ID3abc")
        self.assertRegex(path, r"^/rx/audio/[A-Za-z0-9_-]{20,}\.mp3$")
        c = self.client()
        r = c.get(path)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/mpeg")
        self.assertEqual(r.content, b"ID3abc")
        self.assertEqual(c.get("/rx/audio/nope.mp3").status_code, 404)

    def test_ttl_and_cap(self):
        path = notify.store_audio(b"x")
        token = path.split("/")[-1][:-4]
        notify._audio[token] = (time.time() - notify.AUDIO_TTL_S - 5, b"x")
        self.assertEqual(self.client().get(path).status_code, 404)
        for _ in range(notify.AUDIO_MAX + 20):
            notify.store_audio(b"y")
        self.assertLessEqual(len(notify._audio), notify.AUDIO_MAX)


if __name__ == "__main__":
    unittest.main()
