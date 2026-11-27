"""Tests for the viva voce exam routes (/agent/viva/*). LLM mocked, no network."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import viva  # noqa: E402
import voice  # noqa: E402

HDR = {"Origin": "http://localhost:5173"}
SECRET = "sarvam-key-viva-zz"
EM = chr(0x2014)  # built at runtime so the source stays dash-free


def _q(i: int, difficulty=2, **over) -> dict:
    out = {
        "id": f"q{i}",
        "q": f"Question {i}: What is the mechanism of drug number {i}?",
        "keyPoints": [f"Blocks receptor {i}", f"Effect {i}"],
        "difficulty": difficulty,
        "followUps": ["Why?", "Name two side effects."],
    }
    out.update(over)
    return out


def _plan_json(n: int = 8, **over) -> str:
    qs = [_q(i, difficulty=1 if i <= 2 else (3 if i > n - 2 else 2)) for i in range(1, n + 1)]
    out = {"questions": qs}
    out.update(over)
    return json.dumps(out)


class FakeChat:
    """Returns the queued contents in order (last one repeats)."""

    def __init__(self, *contents: str, exc: Exception | None = None):
        self.contents, self.exc = list(contents), exc
        self.calls: list[list[dict]] = []
        self.kw: dict = {}

    async def __call__(self, messages, **kw):
        self.calls.append(messages)
        self.kw = kw
        if self.exc:
            raise self.exc
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return {"role": "assistant", "content": content}, "sarvam"


TRANSCRIPT = [
    {"role": "examiner", "text": "Good morning. Shall we begin?"},
    {"role": "student", "text": "Yes sir."},
    {"role": "examiner", "text": "What is the mechanism of atropine?"},
    {"role": "student", "text": "Atropine is a competitive antagonist at muscarinic receptors."},
    {"role": "examiner", "text": "Hmm. Name two side effects."},
    {"role": "student", "text": "Dry mouth and, um, tachycardia sir."},
    {"role": "examiner", "text": "Next question. Why is neostigmine used in myasthenia?"},
    {"role": "student", "text": "I don't know sir."},
]


class VivaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_chat = viva.chat
        self._old_key = os.environ.get("SARVAM_API_KEY")
        os.environ["SARVAM_API_KEY"] = SECRET
        viva._plan_cache.clear()
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        viva.chat = self._orig_chat
        if self._old_key is None:
            os.environ.pop("SARVAM_API_KEY", None)
        else:
            os.environ["SARVAM_API_KEY"] = self._old_key

    def _post(self, path: str, body: dict, fake: FakeChat | None, status: int = 200):
        if fake is not None:
            viva.chat = fake
        r = self.client.post(path, headers=HDR, json=body)
        self.assertEqual(r.status_code, status, r.text)
        self.assertNotIn(SECRET, r.text)
        return r.json()

    # ── plan ──────────────────────────────────────────────

    def test_validate_plan_normalises_shape(self) -> None:
        raw = {"questions": [
            _q(1, difficulty="easy", q="Q1. Define a drug" + EM + "any drug?"),
            _q(2, difficulty=7),
            _q(3, difficulty="3", keyPoints="Point A; Point B"),
            _q(4, keyPoints=[]),  # dropped: no key points
            _q(5, q="Question 2: What is the mechanism of drug number 2?"),  # duplicate after numbering strip
            {"question": "Why is atropine used in organophosphate poisoning?", "key_points": ["Blocks muscarinic effects"],
             "follow_ups": "Dose?\nEndpoint?"},
        ]}
        out = viva.validate_plan(raw, 10, 1)
        self.assertEqual([q["id"] for q in out], ["q1", "q2", "q3", "q4"])
        self.assertEqual(out[0]["q"], "Define a drug, any drug?")
        self.assertEqual(out[0]["difficulty"], 1)
        self.assertEqual(out[1]["difficulty"], 3)
        self.assertEqual(out[2]["keyPoints"], ["Point A", "Point B"])
        self.assertEqual(out[3]["followUps"], ["Dose?", "Endpoint?"])
        for q in out:
            self.assertNotIn(EM, json.dumps(q, ensure_ascii=False))
        with self.assertRaises(ValueError):
            viva.validate_plan(raw, 10, 8)
        with self.assertRaises(ValueError):
            viva.validate_plan({"questions": "nope"})

    def test_plan_route_validates_and_caches(self) -> None:
        fake = FakeChat("Sure! ```json\n" + _plan_json(9) + "\n```")
        body = {"subject": "Pharmacology", "topic": "Autonomic drugs", "year": "2nd year MBBS", "lang": "en"}
        out = self._post("/agent/viva/plan", body, fake)
        self.assertEqual(len(out["questions"]), 8)  # count defaults to 8
        self.assertFalse(out["cached"])
        self.assertTrue(fake.kw.get("json_mode"))
        self.assertIn("Autonomic drugs", fake.calls[0][0]["content"])
        again = self._post("/agent/viva/plan", body, fake)
        self.assertTrue(again["cached"])
        self.assertEqual(len(fake.calls), 1)
        self._post("/agent/viva/plan", {**body, "fresh": True}, fake)
        self.assertEqual(len(fake.calls), 2)
        # Language is part of the cache key.
        self._post("/agent/viva/plan", {**body, "lang": "hi"}, fake)
        self.assertEqual(len(fake.calls), 3)

    def test_plan_route_repairs_once_then_accepts_short_plan(self) -> None:
        fake = FakeChat("not json at all", _plan_json(7))
        out = self._post("/agent/viva/plan", {"subject": "Physiology", "topic": "Cardiac cycle", "count": 10}, fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(out["questions"]), 7)
        self.assertIn("not usable", fake.calls[1][-1]["content"])

    def test_plan_route_fails_cleanly_on_garbage(self) -> None:
        fake = FakeChat('{"questions": []}')
        self._post("/agent/viva/plan", {"subject": "Anatomy", "topic": "Brachial plexus"}, fake, status=502)
        self.assertEqual(len(fake.calls), 2)
        self._post("/agent/viva/plan", {"subject": "Anatomy", "topic": "x", "count": 12}, None, status=422)

    # ── grade ─────────────────────────────────────────────

    def _grade_body(self, **over) -> dict:
        plan = viva.validate_plan(json.loads(_plan_json(8)), 8, 8)
        body = {"subject": "Pharmacology", "topic": "Autonomic drugs", "year": "2nd year MBBS",
                "plan": plan, "transcript": TRANSCRIPT, "lang": "en"}
        body.update(over)
        return body

    def _grade_model(self, **over) -> str:
        out = {
            "perQuestion": [
                {"id": "q1", "verdict": "correct", "expected": "Muscarinic antagonist" + EM + "competitive.",
                 "said": "competitive antagonist at muscarinic receptors", "note": "Good."},
                {"id": "q2", "verdict": "partial", "said": "Dry mouth and tachycardia sir", "note": "Two named."},
                {"id": "q3", "verdict": "correct", "said": "Neostigmine increases acetylcholine", "note": "Invented."},
                {"id": "q4", "verdict": "Incorrect", "said": "I don't know sir.", "note": "Did not know."},
                {"id": "q5", "verdict": "not_asked", "said": "anything", "note": ""},
            ],
            "strengths": ["Clear on receptors"],
            "gaps": "Cholinesterase inhibitors\nMyasthenia",
            "recommendedReading": [
                {"book": "KD Tripathi, Essentials of Medical Pharmacology", "chapter": "Anticholinergic drugs"},
                "Goodman and Gilman, Cholinergic agonists, pp. 120-135",
                "KD Tripathi, Essentials of Medical Pharmacology: Anticholinergic drugs",
            ],
            "examinerComment": "Sound basics. Revise anticholinesterases.",
            "score": 99,
        }
        out.update(over)
        return json.dumps(out)

    def test_grade_verifies_quotes_and_computes_score(self) -> None:
        fake = FakeChat(self._grade_model())
        out = self._post("/agent/viva/grade", self._grade_body(), fake)
        self.assertTrue(fake.kw.get("json_mode"))
        user = fake.calls[0][1]["content"]
        self.assertIn("STUDENT: Atropine is a competitive antagonist", user)
        self.assertIn("EXAMINER: What is the mechanism of atropine?", user)
        items = {i["id"]: i for i in out["perQuestion"]}
        self.assertEqual(len(items), 8)
        # Exact quote kept.
        self.assertEqual(items["q1"]["verdict"], "correct")
        self.assertEqual(items["q1"]["said"], "competitive antagonist at muscarinic receptors")
        self.assertNotIn(EM, items["q1"]["expected"])
        # Close paraphrase swapped for the verbatim student line.
        self.assertEqual(items["q2"]["said"], "Dry mouth and, um, tachycardia sir.")
        # Invented quote dropped; a "correct" without evidence becomes partial.
        self.assertEqual(items["q3"]["said"], "")
        self.assertEqual(items["q3"]["verdict"], "partial")
        self.assertEqual(items["q4"]["verdict"], "incorrect")
        # not_asked normalised, quote cleared; missing ids are not asked with plan key points.
        self.assertEqual(items["q5"]["verdict"], "not asked")
        self.assertEqual(items["q5"]["said"], "")
        self.assertEqual(items["q8"]["verdict"], "not asked")
        self.assertEqual(items["q8"]["expected"], "Blocks receptor 8; Effect 8")
        # Score is ours, not the model's 99: q1 d1 correct, q2 d1 partial, q3 d2 partial, q4 d2 incorrect.
        self.assertEqual(out["asked"], 4)
        self.assertEqual(out["total"], 8)
        self.assertEqual(out["score"], round(100 * (1 + 0.5 + 1 + 0) / (1 + 1 + 2 + 2)))
        self.assertEqual(out["gaps"], ["Cholinesterase inhibitors", "Myasthenia"])
        reading = out["recommendedReading"]
        self.assertEqual(reading[0], "KD Tripathi, Essentials of Medical Pharmacology: Anticholinergic drugs")
        self.assertEqual(len(reading), 2)  # duplicate dropped
        self.assertNotRegex(" ".join(reading), r"\bpp?\.?\s*\d")

    def test_grade_without_answers_skips_model(self) -> None:
        fake = FakeChat("should not be called")
        body = self._grade_body(transcript=[{"role": "examiner", "text": "Good morning."}])
        out = self._post("/agent/viva/grade", body, fake)
        self.assertEqual(fake.calls, [])
        self.assertEqual(out["score"], 0)
        self.assertEqual(out["asked"], 0)
        self.assertTrue(all(i["verdict"] == "not asked" for i in out["perQuestion"]))
        self.assertTrue(out["recommendedReading"][0].startswith("KD Tripathi"))

    def test_grade_malformed_model_reply_is_502(self) -> None:
        self._post("/agent/viva/grade", self._grade_body(), FakeChat("I think they did well."), status=502)
        self._post("/agent/viva/grade", self._grade_body(plan=[]), None, status=422)

    def test_grade_hindi_quotes_and_reading_fallback(self) -> None:
        transcript = [
            {"role": "examiner", "text": "एट्रोपिन का mechanism क्या है?"},
            {"role": "user", "text": "यह muscarinic receptor को block करता है।"},
        ]
        model = self._grade_model(perQuestion=[
            {"id": "q1", "verdict": "correct", "said": "muscarinic receptor को block करता है", "note": "ok"},
        ], recommendedReading=[])
        fake = FakeChat(model)
        out = self._post("/agent/viva/grade", self._grade_body(transcript=transcript, lang="hi",
                                                              subject="Pharmacology"), fake)
        self.assertIn("spoken in Hindi", fake.calls[0][0]["content"])
        q1 = out["perQuestion"][0]
        self.assertEqual(q1["verdict"], "correct")
        self.assertEqual(q1["said"], "muscarinic receptor को block करता है")
        self.assertTrue(out["recommendedReading"][0].startswith("KD Tripathi"))

    def test_score_and_reading_helpers(self) -> None:
        plan = [{"id": "q1", "difficulty": 3}, {"id": "q2", "difficulty": 1}]
        self.assertEqual(viva.score_items([], plan), (0, 0))
        self.assertEqual(viva.score_items([{"id": "q1", "verdict": "correct"}, {"id": "q2", "verdict": "incorrect"}], plan), (75, 2))
        self.assertEqual(viva.subject_key("Obstetrics and Gynaecology"), "obg")
        self.assertEqual(viva.subject_key("Forensic Medicine and Toxicology"), "forensic-medicine")
        self.assertEqual(viva.subject_key("Paediatrics"), "paediatrics")
        self.assertEqual(viva.clean_reading(["Guyton, Cardiac cycle, page 112"], "physiology"), ["Guyton, Cardiac cycle"])

    # ── typed fallback ────────────────────────────────────

    def test_ask_returns_clean_reply(self) -> None:
        fake = FakeChat("<think>plan</think>**Hmm.** And the mechanism" + EM + "why?")
        body = {"system": "You are the examiner.", "messages": [{"role": "user", "content": "It blocks receptors."}]}
        out = self._post("/agent/viva/ask", body, fake)
        self.assertEqual(out["reply"], "Hmm. And the mechanism, why?")
        self.assertEqual(fake.calls[0][0], {"role": "system", "content": "You are the examiner."})
        self._post("/agent/viva/ask", body, FakeChat("   "), status=502)

    # ── voice hooks ───────────────────────────────────────

    def test_examiner_voice_and_fillers(self) -> None:
        v = voice._voice_for("M", viva.VIVA_CASE_ID, "en")
        self.assertEqual(v, viva.EXAMINER_VOICE)
        self.assertEqual(voice._voice_for("M", viva.VIVA_CASE_ID, "kn"), viva.EXAMINER_VOICE)
        old = os.environ.get("GEMINI_VOICE_VIVA")
        try:
            os.environ["GEMINI_VOICE_VIVA"] = "Puck"
            self.assertEqual(voice._voice_for("M", viva.VIVA_CASE_ID, "en"), "Puck")
        finally:
            if old is not None:
                os.environ["GEMINI_VOICE_VIVA"] = old
            else:
                os.environ.pop("GEMINI_VOICE_VIVA", None)


if __name__ == "__main__":
    unittest.main()
