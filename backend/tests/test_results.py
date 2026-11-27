"""Consultation result to Telegram: text, PDF, router validation, rate limit, bundle shape. No network."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import results  # noqa: E402
from integrations import notify  # noqa: E402
from pdf_report import build_report_pdf  # noqa: E402

EM = "—"


def payload(**over):
    body = {
        "to": "telegram",
        "profile": {"name": "Ananya Rao", "university": "RGUHS"},
        "case": {"id": "gm-01", "title": "Ravi Kumar, 45M", "specialty": "General medicine",
                 "chiefComplaint": "Fever for 3 days " + EM + " with chills"},
        "evaluation": {
            "score": 72.4, "verdict": "Good",
            "domain_scores": [{"name": "History", "score": 8, "max": 10}, {"name": "Management", "score": 55, "max": 100}],
            "strengths": ["Asked about <travel> & contacts", "Ordered a malaria smear early", "Clear safety-net advice", "Fourth"],
            "improvements": ["Check allergies before prescribing " + EM + " always", "Examine the abdomen"],
            "narrative": "Solid consultation with a focused history.",
            "diagnosis": {"chosen": "Malaria", "correct": "Malaria"},
        },
        "lang": "en",
        "audio": False,
    }
    body.update(over)
    return body


def tg_factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TextTests(unittest.TestCase):
    def test_format(self):
        txt = results.compose_result_html(results.ResultRequest(**payload()))
        self.assertTrue(txt.startswith("<b>MedSim consultation result</b>"))
        self.assertIn("<b>Student:</b> Ananya Rao (RGUHS)", txt)
        self.assertIn("<b>Score:</b> 72/100, Good", txt)
        self.assertIn("Malaria, correct", txt)
        self.assertIn("&lt;travel&gt; &amp; contacts", txt)
        self.assertNotIn("Fourth", txt)  # top 3 only
        self.assertIn("• Examine the abdomen", txt)
        self.assertIn("<i>", txt)
        self.assertIn("training simulation", txt)
        self.assertNotIn(EM, txt)
        self.assertLessEqual(len(txt), results.MAX_TEXT)

    def test_long_input_is_capped(self):
        p = payload()
        p["evaluation"]["strengths"] = ["x" * 3000] * 10
        p["evaluation"]["improvements"] = ["y" * 3000] * 10
        txt = results.compose_result_html(results.ResultRequest(**p))
        self.assertLessEqual(len(txt), results.MAX_TEXT)

    def test_wrong_diagnosis(self):
        p = payload()
        p["evaluation"]["diagnosis"] = {"chosen": "Dengue", "correct": "Malaria"}
        txt = results.compose_result_html(results.ResultRequest(**p))
        self.assertIn("Dengue, not the case diagnosis (Malaria)", txt)

    def test_score_from_domains_when_missing(self):
        p = payload()
        p["evaluation"]["score"] = None
        txt = results.compose_result_html(results.ResultRequest(**p))
        self.assertIn("68/100", txt)  # mean of 80 and 55, rounded

    def test_voice_script(self):
        s = results.voice_script(results.ResultRequest(**payload()))
        self.assertIn("Hello Ananya", s)
        self.assertIn("72 out of 100", s)
        self.assertLess(len(s.split()), 80)
        self.assertNotIn(EM, s)


class PdfTests(unittest.TestCase):
    def test_pdf_bytes(self):
        pdf = results.make_pdf(results.ResultRequest(**payload()))
        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"Ananya Rao", pdf)
        self.assertIn(b"SIMULATION", pdf)
        self.assertNotIn(EM.encode("utf-8"), pdf)
        self.assertNotIn(b"\x97", pdf)  # WinAnsi em dash
        start = int(pdf.rsplit(b"startxref\n", 1)[1].split(b"\n")[0])
        self.assertEqual(pdf[start:start + 4], b"xref")

    def test_escapes_parens_and_non_latin(self):
        pdf = build_report_pdf("Dr (A) \\ B", "", "Case", "", "", 50, "Pass", [("X", 50)], ["नम"], [])
        self.assertIn(b"Dr \\(A\\) \\\\ B", pdf)


class RouterTests(unittest.TestCase):
    def setUp(self):
        results._hits.clear()
        notify._tg_chats.clear()
        notify._tg_offset["next"] = None
        self.calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            method = req.url.path.rsplit("/", 1)[-1]
            self.calls.append((method, req))
            if method == "getUpdates":
                return httpx.Response(200, json={"ok": True, "result": [
                    {"update_id": 5, "message": {"chat": {"id": 222, "username": "DrDemo"}, "text": "/start"}}]})
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

        env = mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:abc", "NOTIFY_ALLOWED_DESTS": ""})
        env.start()
        self.addCleanup(env.stop)
        f = mock.patch.object(notify, "_client_factory", tg_factory(handler))
        f.start()
        self.addCleanup(f.stop)
        app = FastAPI()
        app.include_router(results.router)
        self.client = TestClient(app)

    def test_sends_text_pdf_audio(self):
        async def fake_audio(req):
            return b"ID3fake", "en"

        with mock.patch.object(results, "make_audio", fake_audio):
            r = self.client.post("/agent/results/send", json=payload(audio=True))
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["resolvedTo"], "tg:222")
        self.assertTrue(data["parts"]["text"]["ok"])
        self.assertTrue(data["parts"]["document"]["ok"])
        self.assertTrue(data["parts"]["audio"]["ok"])
        self.assertEqual([m for m, _ in self.calls], ["getUpdates", "sendMessage", "sendDocument", "sendAudio"])
        msg = json.loads(self.calls[1][1].content)
        self.assertEqual(msg["parse_mode"], "HTML")
        self.assertEqual(msg["chat_id"], 222)
        self.assertIn("<b>MedSim consultation result</b>", msg["text"])
        doc = self.calls[2][1].content
        self.assertIn(b'filename="medsim-report-gm-01.pdf"', doc)
        self.assertIn(b"application/pdf", doc)
        self.assertIn(b"%PDF-1.4", doc)
        self.assertIn(b"audio/wav", self.calls[3][1].content)

    def test_default_destination_and_audio_skipped(self):
        data = self.client.post("/agent/results/send", json=payload(to=None)).json()
        self.assertTrue(data["ok"], data)
        self.assertTrue(data["parts"]["audio"]["skipped"])
        self.assertNotIn("sendAudio", [m for m, _ in self.calls])

    def test_rejects_non_telegram(self):
        for bad in ("+919876543210", "a@b.com", "@x", "tg:abc'"):
            r = self.client.post("/agent/results/send", json=payload(to=bad))
            self.assertEqual(r.status_code, 400, bad)
        r = self.client.post("/agent/results/send", json={"to": "telegram"})
        self.assertEqual(r.status_code, 422)

    def test_not_configured(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}):
            r = self.client.post("/agent/results/send", json=payload())
        self.assertEqual(r.status_code, 503)

    def test_allowed_list(self):
        with mock.patch.dict(os.environ, {"NOTIFY_ALLOWED_DESTS": "@someoneelse"}):
            r = self.client.post("/agent/results/send", json=payload())
        self.assertEqual(r.status_code, 403)

    def test_rate_limit(self):
        for _ in range(results.RATE_MAX):
            self.assertEqual(self.client.post("/agent/results/send", json=payload(to="@DrDemo")).status_code, 200)
        self.assertEqual(self.client.post("/agent/results/send", json=payload(to="@drdemo")).status_code, 429)
        self.assertEqual(self.client.post("/agent/results/send", json=payload(to="telegram")).status_code, 200)

    def test_no_chat_yet(self):
        def empty(req):
            return httpx.Response(200, json={"ok": True, "result": []})

        with mock.patch.object(notify, "_client_factory", tg_factory(empty)):
            data = self.client.post("/agent/results/send", json=payload()).json()
        self.assertFalse(data["ok"])
        self.assertIn("press Start", data["error"])

    def test_part_failure_reported(self):
        def handler(req):
            method = req.url.path.rsplit("/", 1)[-1]
            if method == "getUpdates":
                return httpx.Response(200, json={"ok": True, "result": [
                    {"update_id": 1, "message": {"chat": {"id": 7}, "text": "/start"}}]})
            if method == "sendDocument":
                return httpx.Response(400, json={"ok": False, "description": "Bad Request: file too big"})
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        with mock.patch.object(notify, "_client_factory", tg_factory(handler)):
            data = self.client.post("/agent/results/send", json=payload()).json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["parts"]["document"]["ok"])
        self.assertIn("file too big", data["parts"]["document"]["error"])


class AudioTests(unittest.TestCase):
    def test_translation_failure_falls_back_to_english(self):
        import rx_audio

        seen = {}

        async def boom(script, lang):
            raise RuntimeError("x")

        async def synth(script, lang):
            seen["lang"] = lang
            return b"mp3", "v", "en-IN"

        with mock.patch.object(results, "_translate", boom), mock.patch.object(rx_audio, "synthesize", synth):
            mp3, lang = asyncio.run(results.make_audio(results.ResultRequest(**payload(lang="hi"))))
        self.assertEqual(mp3, b"mp3")
        self.assertEqual(lang, "en")
        self.assertEqual(seen["lang"], "en")

    def test_murf_failure_never_raises(self):
        import rx_audio

        async def synth(script, lang):
            raise HTTPException(status_code=500, detail="MURF_API_KEY missing")

        with mock.patch.object(rx_audio, "synthesize", synth):
            mp3, note = asyncio.run(results.make_audio(results.ResultRequest(**payload())))
        self.assertIsNone(mp3)
        self.assertIn("MURF", note)


if __name__ == "__main__":
    unittest.main()
