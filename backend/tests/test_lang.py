"""Hindi / Kannada support: locale -> Murf voice mapping, prompt language blocks,
filler phrases, greeting translation and the /voice/start + /voice/tts bodies. No network."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import coach  # noqa: E402
import gemini_live  # noqa: E402
import gemini_tts  # noqa: E402
import lang  # noqa: E402
import nurse  # noqa: E402
import server  # noqa: E402
import teacher  # noqa: E402
import voice  # noqa: E402

SECRETS = {
    "GEMINI_API_KEY": "fake-gemini-key",
    "AGORA_APP_ID": "a" * 32,
    "AGORA_APP_CERTIFICATE": "b" * 32,
    "AGORA_CUSTOMER_ID": "cust-id-zz",
    "AGORA_CUSTOMER_SECRET": "cust-secret-zz",
    "MURF_API_KEY": "murf-key-zz",
    "SARVAM_API_KEY": "sarvam-key-zz",
}
HDR = {"Origin": "http://localhost:5173"}

# Voices verified (2026-11-21) to list the locale in Murf's supportedLocales.
HI_OK = {
    "en-IN-aarav", "en-IN-nikhil", "en-IN-samar", "en-IN-abhinav", "en-IN-eashwar",
    "en-IN-priya", "en-IN-pooja", "en-IN-anusha", "en-IN-isha", "en-IN-arohi", "en-IN-alia", "en-IN-anisha",
}
KN_OK = {
    "en-IN-nikhil", "en-IN-samar", "hi-IN-chirag", "hi-IN-pranav",
    "en-IN-pooja", "hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "kn-IN-harshitha",
}


TA_OK = {
    "en-IN-samar", "hi-IN-chirag", "hi-IN-pranav", "ta-IN-karthikeyan", "en-IN-pooja", "hi-IN-poonam",
    "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "en-IN-anisha",
}
TE_OK = {
    "en-IN-samar", "hi-IN-chirag", "hi-IN-pranav", "en-IN-pooja", "en-IN-alia", "hi-IN-poonam", "hi-IN-neeta",
    "hi-IN-jaya", "hi-IN-rachna", "hi-IN-ayushi", "en-IN-anisha",
}
MR_OK = {
    "en-IN-samar", "en-IN-abhinav", "hi-IN-chirag", "hi-IN-pranav", "mr-IN-prathamesh", "hi-IN-poonam",
    "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "mr-IN-prajakta", "en-IN-anisha",
}
BN_OK = {
    "hi-IN-chirag", "hi-IN-pranav", "bn-IN-subhankar", "hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya",
    "hi-IN-rachna", "bn-IN-debarati", "en-IN-anisha",
}
NEW_LANGS = {"ta": TA_OK, "te": TE_OK, "mr": MR_OK, "bn": BN_OK}
SAMPLE_TEXT = {"ta": "வணக்கம் டாக்டர்", "te": "నమస్కారం డాక్టర్", "mr": "नमस्कार डॉक्टर", "bn": "নমস্কার ডাক্তারবাবু"}


class LangUnitTests(unittest.TestCase):
    def test_norm_and_locale(self) -> None:
        self.assertEqual(lang.norm_lang("hi-IN"), "hi")
        self.assertEqual(lang.norm_lang("KN"), "kn")
        self.assertEqual(lang.norm_lang("en-IN"), "en")
        self.assertEqual(lang.norm_lang("fr"), "en")
        self.assertEqual(lang.norm_lang(None), "en")
        self.assertEqual(lang.locale_for("kn"), "kn-IN")
        self.assertEqual(lang.locale_for(None), "en-IN")

    def test_every_voice_supports_its_locale(self) -> None:
        for lg, ok in (("hi", HI_OK), ("kn", KN_OK)):
            for _g, pool in lang.PATIENT_VOICE_POOLS[lg].items():
                for v in pool:
                    self.assertIn(v, ok, f"{v} not verified for {lg}")
            self.assertIn(lang.nurse_voice(lg), ok)

    def test_new_languages_voices_and_locales(self) -> None:
        self.assertEqual(lang.LANGS, ("en", "hi", "kn", "ta", "te", "mr", "bn"))
        for lg, ok in NEW_LANGS.items():
            self.assertEqual(lang.locale_for(f"{lg}-IN"), f"{lg}-IN")
            self.assertEqual(lang.norm_lang(lg.upper()), lg)
            for g in ("M", "F"):
                self.assertTrue(lang.PATIENT_VOICE_POOLS[lg][g], f"{lg} {g} pool empty")
                for v in lang.PATIENT_VOICE_POOLS[lg][g]:
                    self.assertIn(v, ok, f"{v} not verified for {lg}")
            self.assertIn(lang.nurse_voice(lg), ok)
            self.assertEqual(len(lang.fillers(lg)), len(lang.fillers("hi")))
            for p in lang.fillers(lg):
                self.assertTrue(lang.has_script(p, lg), p)
            self.assertTrue(lang.has_script(lang.failure_message(lg), lg))
            self.assertTrue(lang.has_script(SAMPLE_TEXT[lg], lg))
            self.assertFalse(lang.has_script("hello", lg))
        self.assertEqual(lang.nurse_voice("mr"), "mr-IN-prajakta")
        self.assertEqual(lang.nurse_voice("bn"), "bn-IN-debarati")
        self.assertFalse(lang.has_script("வணக்கம்", "hi"))

    def test_new_language_blocks(self) -> None:
        for lg, script in (("ta", "Tamil script"), ("te", "Telugu script"), ("mr", "Devanagari script"), ("bn", "Bengali script")):
            b = lang.language_block(lg, female=True)
            self.assertIn(script, b)
            self.assertIn("BP, sugar, test", b)
            self.assertIn("Always reply in", b)
            self.assertIn("MIXED LANGUAGES", b)
            self.assertIn("JSON", lang.language_block(lg, role="coach"))
            self.assertIn("lesson", lang.language_block(lg, role="teacher"))
        self.assertIn("feminine Marathi", lang.language_block("mr", female=True))
        self.assertIn("masculine Marathi", lang.language_block("mr", female=False))
        self.assertNotIn("Hindi verb", lang.language_block("ta"))
        self.assertIn("Tamil", teacher.lesson_system("student", "ta"))

    def test_new_language_nurse_and_coach_text(self) -> None:
        class St:
            new_tests: list = []
            new_rx: list = []
            follow_up = None
            tests: dict = {}
            meds: dict = {}

        for lg in NEW_LANGS:
            self.assertTrue(lang.has_script(nurse._fallback_confirmation(St(), [], lg), lg))
            req = coach.CoachCheckRequest(
                event=coach.CoachEvent(kind="prescription", detail="Amoxicillin"),
                patient=coach.CoachPatient(allergies="penicillin"),
                lang=lg,
            )
            a = coach.rule_alert(req)
            self.assertTrue(lang.has_script(a.title, lg), lg)
            self.assertTrue(lang.has_script(a.explanation, lg), lg)

    def test_voice_stable_and_gendered(self) -> None:
        for lg in lang.LANGS:
            a = lang.pick_patient_voice(lg, "F", "case-1")
            self.assertEqual(a, lang.pick_patient_voice(lg, "F", "case-1"))
            self.assertIn(a, lang.PATIENT_VOICE_POOLS[lg]["F"])
            self.assertIn(lang.pick_patient_voice(lg, "M", "case-1"), lang.PATIENT_VOICE_POOLS[lg]["M"])

    def test_fillers_per_language(self) -> None:
        self.assertIn("हाँ डॉक्टर...", lang.fillers("hi"))
        self.assertIn("एक मिनट...", lang.fillers("hi"))
        self.assertIn("ಹೌದು ಡಾಕ್ಟರ್...", lang.fillers("kn"))
        self.assertIn("ಒಂದು ನಿಮಿಷ...", lang.fillers("kn"))
        self.assertIn("Haan, doctor...", lang.fillers("en"))
        for p in lang.fillers("hi"):
            self.assertTrue(lang.has_script(p, "hi"), p)
        for p in lang.fillers("kn"):
            self.assertTrue(lang.has_script(p, "kn"), p)

    def test_language_blocks(self) -> None:
        self.assertIn("MIXED LANGUAGES", lang.language_block("en"))
        self.assertIn("Indian English", lang.language_block("en"))
        hi = lang.language_block("hi", female=True)
        self.assertIn("Devanagari", hi)
        self.assertIn("feminine Hindi", hi)
        self.assertIn("BP, sugar, test", hi)
        self.assertIn("Always reply in Hindi", hi)
        self.assertIn("masculine Hindi", lang.language_block("hi", female=False))
        kn = lang.language_block("kn")
        self.assertIn("Kannada script", kn)
        self.assertNotIn("feminine", kn)
        self.assertIn("JSON", lang.language_block("hi", role="coach"))
        self.assertIn("lesson", lang.language_block("kn", role="teacher"))
        self.assertTrue(lang.with_language("X", "en").startswith("X\n\nMIXED LANGUAGES"))
        self.assertTrue(lang.with_language("X", "hi", role="nurse").startswith("X\n\n"))

    def test_failure_message_per_language(self) -> None:
        self.assertTrue(lang.has_script(lang.failure_message("hi"), "hi"))
        self.assertTrue(lang.has_script(lang.failure_message("kn"), "kn"))
        self.assertEqual(lang.failure_message("en"), "Hmm, sorry, say that once more?")

    def test_failure_and_retry_lines_sound_human(self) -> None:
        doctor = ("doctor", "डॉक्टर", "ಡಾಕ್ಟರ್", "டாக்டர்", "డాక్టర్", "ডাক্তার")
        robotic = ("didn't", "did not", "catch that", "error", "system", "सुन नहीं")
        for lg in lang.LANGS:
            lines = [lang.failure_message(lg), *lang.NATURAL_RETRY_LINES[lg]]
            self.assertGreaterEqual(len(lang.NATURAL_RETRY_LINES[lg]), 3, lg)
            for line in lines:
                low = line.lower()
                self.assertFalse([w for w in doctor if w in low], f"{lg}: {line}")
                self.assertFalse([w for w in robotic if w in low], f"{lg}: {line}")
                self.assertLessEqual(len(line), 80, f"{lg}: {line}")
                if lg != "en":
                    self.assertTrue(lang.has_script(line, lg), f"{lg}: {line}")
            self.assertIn(lang.natural_retry_line(lg), lang.NATURAL_RETRY_LINES[lg])

    def test_voice_for_nurse(self) -> None:
        old = os.environ.get("MURF_NURSE_VOICE")
        os.environ["MURF_NURSE_VOICE"] = "custom-voice"
        try:
            self.assertEqual(voice._voice_for("nurse"), "custom-voice")
        finally:
            if old is None:
                os.environ.pop("MURF_NURSE_VOICE", None)
            else:
                os.environ["MURF_NURSE_VOICE"] = old

    def test_rule_alerts_localised(self) -> None:
        req = coach.CoachCheckRequest(
            event=coach.CoachEvent(kind="prescription", detail="Amoxicillin"),
            patient=coach.CoachPatient(allergies="penicillin"),
            lang="hi",
        )
        a = coach.rule_alert(req)
        self.assertIsNotNone(a)
        self.assertTrue(lang.has_script(a.explanation, "hi"))
        req.lang = "kn"
        self.assertTrue(lang.has_script(coach.rule_alert(req).title, "kn"))
        req.lang = None
        self.assertEqual(coach.rule_alert(req).title, "Allergy conflict")

    def test_lesson_prompt_and_cache_key(self) -> None:
        self.assertNotIn("LANGUAGE:", teacher.lesson_system("student", "en"))
        self.assertIn("Kannada", teacher.lesson_system("student", "kn"))
        self.assertIn("feminine Hindi", teacher.lesson_system("intern", "hi"))
        self.assertNotEqual(teacher.cache_key("ECG", "student", "hi"), teacher.cache_key("ECG", "student"))
        self.assertEqual(teacher.cache_key("ECG", "student", "en"), teacher.cache_key("ECG", "student"))

    def test_nurse_fallback_confirmation_localised(self) -> None:
        class St:
            new_tests: list = []
            new_rx: list = []
            follow_up = None
            tests: dict = {}
            meds: dict = {}

        self.assertTrue(lang.has_script(nurse._fallback_confirmation(St(), [], "hi"), "hi"))
        self.assertTrue(lang.has_script(nurse._fallback_confirmation(St(), [], "kn"), "kn"))


class VoiceLocaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = {k: os.environ.get(k) for k in [*SECRETS, "PATIENT_LLM_PROVIDER", "VOICE_ASR", "PUBLIC_URL"]}
        os.environ.update(SECRETS)
        os.environ["PATIENT_LLM_PROVIDER"] = "sarvam"
        os.environ.pop("VOICE_ASR", None)
        os.environ.pop("PUBLIC_URL", None)
        self.tts_calls: list[dict] = []

        FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        async def fake_synth(text, voice="Aoede", **kw):
            self.tts_calls.append({"text": text, "voice": voice, **kw})
            return FAKE_WAV, "audio/wav"

        self._orig_synth = gemini_tts.synthesize_speech
        gemini_tts.synthesize_speech = fake_synth

        self._orig_tr = lang.translate_line

        async def fake_translate(text, lg):
            return {"hi": "मुझे सीने में दर्द है", "kn": "ನನಗೆ ಎದೆ ನೋವು ಇದೆ"}.get(lang.norm_lang(lg), text)

        lang.translate_line = fake_translate
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        gemini_tts.synthesize_speech = self._orig_synth
        lang.translate_line = self._orig_tr
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _start(self, locale, gender: str = "F"):
        body = {"caseId": "c-77", "systemPrompt": "sp", "initialLine": "Chest pain", "gender": gender}
        if locale:
            body["locale"] = locale
        r = self.client.post("/voice/start", headers=HDR, json=body)
        self.assertEqual(r.status_code, 200, r.text)
        channel = r.json()["channel"]
        return gemini_live.get_prepared_session(channel)

    def test_start_hindi(self) -> None:
        p = self._start("hi-IN")
        self.assertEqual(p.locale, "hi")
        self.assertEqual(p.initial_line, "मुझे सीने में दर्द है")

    def test_start_kannada(self) -> None:
        for g in ("M", "F"):
            p = self._start("kn-IN", g)
            self.assertEqual(p.locale, "kn")
            self.assertEqual(p.initial_line, "ನನಗೆ ಎದೆ ನೋವು ಇದೆ")

    def test_start_new_languages(self) -> None:
        for lg, ok in NEW_LANGS.items():
            for g in ("M", "F"):
                p = self._start(f"{lg}-IN", g)
                self.assertEqual(p.locale, lg)
                self.assertTrue(lang.has_script(lang.failure_message(lg), lg))

    def test_tts_new_language_nurse(self) -> None:
        for lg, ok in NEW_LANGS.items():
            body = {"text": SAMPLE_TEXT[lg], "voice": "nurse", "locale": f"{lg}-IN"}
            r = self.client.post("/voice/tts", headers=HDR, json=body)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.headers["content-type"], "audio/wav")

    def test_start_english_unchanged(self) -> None:
        p = self._start(None)
        self.assertEqual(p.locale, "en")
        self.assertEqual(p.initial_line, "Chest pain")

    def test_tts_locale_and_voice(self) -> None:
        r = self.client.post("/voice/tts", headers=HDR, json={"text": "ठीक है डॉक्टर", "voice": "nurse", "locale": "hi-IN"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertEqual(self.tts_calls[-1]["voice"], "Aoede")

        r = self.client.post("/voice/tts", headers=HDR, json={"text": "ಸರಿ ಡಾಕ್ಟರ್", "voice": "nurse", "locale": "kn-IN"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertEqual(self.tts_calls[-1]["voice"], "Aoede")

        self.client.post("/voice/tts", headers=HDR, json={"text": "Okay doctor"})
        self.assertEqual(self.tts_calls[-1]["voice"], "Aoede")


if __name__ == "__main__":
    unittest.main()
