"""Tests for /voice/start (mocked Agora), nurse tools, and debrief parsing. No network."""

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

import debrief  # noqa: E402
import gemini_live  # noqa: E402
import gemini_tts  # noqa: E402
import nurse  # noqa: E402
import server  # noqa: E402
import voice  # noqa: E402

SECRETS = {
    "GEMINI_API_KEY": "fake-gemini-key-voice-nurse",
}
HDR = {"Origin": "http://localhost:5173"}


class VoiceStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = {k: os.environ.get(k) for k in [*SECRETS, "GEMINI_LIVE_MODEL"]}
        os.environ.update(SECRETS)
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_start_builds_gemini_live_session_and_hides_keys(self) -> None:
        r = self.client.post("/voice/start", headers=HDR, json={
            "caseId": "case 1", "systemPrompt": "You are Raj.", "initialLine": "Hi doc", "gender": "F"
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["mode"], "gemini_live")
        self.assertEqual(data["model"], "gemini-3.8-live")
        self.assertTrue(data["channel"].startswith("medsim-case1-"))
        self.assertTrue(100000 <= data["uid"] <= 999999)
        self.assertEqual(data["sampleRateIn"], 16000)
        self.assertEqual(data["sampleRateOut"], 24000)
        self.assertIn("/voice/live/", data["wsUrl"])
        self.assertIn("/voice/observer/ws", data["observerWsUrl"])
        self.assertTrue(data["token"])
        self.assertTrue(data["observerToken"])
        for secret in SECRETS.values():
            self.assertNotIn(secret, r.text)

        prep = gemini_live.get_prepared_session(data["channel"])
        self.assertIsNotNone(prep)
        self.assertEqual(prep.initial_line, "Hi doc")
        self.assertEqual(prep.system_prompt, "You are Raj.")
        self.assertIn(prep.voice, gemini_tts.FEMALE_VOICES)

    def test_start_voice_selection(self) -> None:
        self.assertEqual(voice._voice_for("F", "case-x"), voice._voice_for("F", "case-x"))
        self.assertEqual(voice._voice_for("nurse"), "Aoede")

    def test_missing_config_is_503_without_values(self) -> None:
        os.environ.pop("GEMINI_API_KEY", None)
        r = self.client.post("/voice/start", headers=HDR, json={
            "caseId": "c", "systemPrompt": "x", "initialLine": "y", "gender": "M"
        })
        self.assertEqual(r.status_code, 503)
        self.assertIn("GEMINI_API_KEY", r.text)

    def test_stop_is_idempotent(self) -> None:
        r = self.client.post("/voice/stop", headers=HDR, json={"agentId": "gone"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_voice_tts_success(self) -> None:
        async def fake_synth(text, voice="Aoede"):
            return (gemini_tts.wrap_pcm_in_wav(b"\x00\x00" * 50, 24000), "audio/wav")

        orig_synth = gemini_tts.synthesize_speech
        gemini_tts.synthesize_speech = fake_synth
        try:
            r = self.client.post("/voice/tts", headers=HDR, json={
                "text": "Hello doctor", "voice": "nurse"
            })
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get("content-type"), "audio/wav")
            self.assertTrue(r.content.startswith(b"RIFF"))
        finally:
            gemini_tts.synthesize_speech = orig_synth

    def test_voice_tts_error_handling(self) -> None:
        async def fake_fail(text, voice="Aoede"):
            raise RuntimeError("Upstream TTS failure")

        orig_synth = gemini_tts.synthesize_speech
        gemini_tts.synthesize_speech = fake_fail
        try:
            r = self.client.post("/voice/tts", headers=HDR, json={
                "text": "Hello doctor", "voice": "nurse"
            })
            self.assertEqual(r.status_code, 502)
            self.assertIn("Gemini TTS failed", r.text)
        finally:
            gemini_tts.synthesize_speech = orig_synth

    def test_stop_is_idempotent(self) -> None:
        r = self.client.post("/voice/stop", headers=HDR, json={"agentId": "gone"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


def _req(**over) -> nurse.NurseActRequest:
    base = {
        "caseId": "c", "utterance": "x",
        "patient": {"name": "A", "allergies": "Penicillin (rash)",
                    "anamnesis": [{"question": "Current medications?", "answer": "warfarin"}],
                    "testResults": [{"testId": "cbc", "result": "WBC 14", "abnormal": True}]},
        "catalog": {"tests": [{"id": "cbc", "name": "Complete Blood Count", "category": "lab"},
                              {"id": "crp", "name": "C-Reactive Protein", "category": "lab"}],
                    "medications": [{"id": "amoxicillin", "name": "Amoxicillin", "dosing": "500mg TID"},
                                    {"id": "ibuprofen", "name": "Ibuprofen"},
                                    {"id": "azithromycin", "name": "Azithromycin"}]},
        "alreadyOrdered": [], "alreadyPrescribed": [],
    }
    base.update(over)
    return nurse.NurseActRequest(**base)


class NurseToolTests(unittest.TestCase):
    def test_substitute_never_inherits_the_original_drugs_dose(self) -> None:
        # Doctor ordered amoxicillin 500 mg TDS (held for allergy); the nurse
        # offers azithromycin. It must NOT be given "500 mg three times a day".
        req = _req(
            utterance="Start amoxicillin 500 mg three times a day",
            patient={"name": "A", "allergies": "Penicillin (rash)", "anamnesis": [], "testResults": []},
            catalog={"tests": [], "medications": [
                {"id": "amoxicillin", "name": "Amoxicillin", "dosing": "500mg TID"},
                {"id": "azithromycin-500", "name": "Azithromycin 500mg", "dosing": "500 mg once daily for 3 days"},
            ]},
        )
        st = nurse.NurseState(req)
        self.assertTrue(st.prescribe("amoxicillin", "500 mg three times a day", "5 days")["refused"])
        sub = st.prescribe("azithromycin-500", "500 mg three times a day", "5 days")
        self.assertTrue(sub["ok"])
        self.assertTrue(sub.get("substitute"))
        self.assertEqual(sub["dose"], "500 mg once daily for 3 days")

    def test_named_drug_keeps_the_doctors_dose(self) -> None:
        req = _req(utterance="Start aspirin 75 mg once daily",
                   catalog={"tests": [], "medications": [{"id": "aspirin-100", "name": "Aspirin 100mg", "dosing": "100 mg OD"}]},
                   patient={"name": "A", "allergies": "", "anamnesis": [], "testResults": []})
        st = nurse.NurseState(req)
        rx = st.prescribe("aspirin-100", "75 mg once daily", "ongoing")
        self.assertEqual(rx["dose"], "75 mg once daily")
        self.assertNotIn("substitute", rx)

    def test_find_in_catalog(self) -> None:
        st = nurse.NurseState(_req())
        self.assertEqual(st.find_in_catalog("blood count", "test")["matches"][0]["id"], "cbc")
        self.assertEqual(st.find_in_catalog("amoxicillin", "medication")["matches"][0]["id"], "amoxicillin")
        self.assertFalse(st.find_in_catalog("zzz", "test")["ok"])

    def test_order_test_uses_case_result_or_default(self) -> None:
        st = nurse.NurseState(_req())
        a = st.order_test("cbc")
        self.assertEqual((a["result"], a["abnormal"]), ("WBC 14", True))
        b = st.order_test("crp")
        self.assertEqual((b["result"], b["abnormal"]), ("Within normal limits", False))
        self.assertFalse(st.order_test("nope")["ok"])
        self.assertEqual(st.new_tests, ["cbc", "crp"])
        self.assertTrue(st.order_test("cbc")["alreadyOrdered"])

    def test_allergy_refuses_prescribe(self) -> None:
        st = nurse.NurseState(_req())
        chk = st.check_allergies_and_interactions("amoxicillin")
        self.assertFalse(chk["safe"])
        res = st.prescribe("amoxicillin", "500mg", "5d")
        self.assertFalse(res["ok"])
        self.assertTrue(res["refused"])
        self.assertEqual(st.new_rx, [])
        self.assertTrue(st.prescribe("azithromycin", "500mg", "3d")["ok"])
        self.assertEqual(st.new_rx[0]["medicationId"], "azithromycin")

    def test_interaction_warning_from_anamnesis(self) -> None:
        st = nurse.NurseState(_req(patient={"allergies": "none", "anamnesis": [
            {"question": "Current medications?", "answer": "warfarin"}]}))
        chk = st.check_allergies_and_interactions("ibuprofen")
        self.assertTrue(chk["safe"])
        self.assertTrue(chk["warnings"])

    def test_follow_up(self) -> None:
        st = nurse.NurseState(_req())
        self.assertTrue(st.schedule_follow_up(7, "recheck")["ok"])
        self.assertEqual(st.follow_up, {"inDays": 7, "reason": "recheck"})
        self.assertFalse(st.schedule_follow_up("soon", "x")["ok"])

    def test_unknown_tool_and_bad_args(self) -> None:
        st = nurse.NurseState(_req())
        self.assertFalse(st.execute("rm_rf", {})["ok"])
        self.assertFalse(st.execute("order_test", {"wrong": 1})["ok"])


class DevOriginTests(unittest.TestCase):
    def test_any_local_port_is_trusted_but_foreign_origins_are_not(self) -> None:
        self.assertTrue(server._is_dev_origin("http://localhost:5181"))
        self.assertTrue(server._is_dev_origin("http://127.0.0.1:5190"))
        self.assertFalse(server._is_dev_origin("http://evil.example.com"))
        self.assertFalse(server._is_dev_origin("http://localhost.evil.com:5173"))
        self.assertFalse(server._is_dev_origin("https://localhost:5173.evil.com"))


class DebriefTests(unittest.TestCase):
    def test_rubric_accepts_list_or_authored_object(self) -> None:
        domains, criteria = debrief.split_rubric(["History", "Plan"])
        self.assertEqual(domains, ["History", "Plan"])
        self.assertEqual(criteria, [])
        authored = {
            "data_gathering": [{"criterion_id": "dg-01", "label": "Onset and character", "weight": 2, "evidence": "Asks when it started"}],
            "clinical_management": [{"criterion_id": "cm-01", "label": "Avoids amoxicillin", "weight": 3, "evidence": "Penicillin allergy"}],
            "interpersonal": [],
            "safety_netting": {"required_elements": ["return if breathless"], "weight": 1},
            "global_rating": "borderline-regression",
        }
        domains, criteria = debrief.split_rubric(authored)
        self.assertEqual(domains, list(debrief.DEFAULT_RUBRIC))
        self.assertEqual(len(criteria), 3)
        self.assertTrue(criteria[1].startswith("[clinical_management] Avoids amoxicillin"))
        # The request model no longer rejects the object form (previously a 422).
        req = debrief.DebriefRequest(caseId="im-005", rubric=authored)
        self.assertIsInstance(req.rubric, dict)
        self.assertEqual(debrief.split_rubric(None)[0], list(debrief.DEFAULT_RUBRIC))

    def test_coerce_and_clamp(self) -> None:
        raw = debrief.parse_json_object(
            '<think>x</think>```json\n{"score":130,"domains":[{"name":"A","score":"80","notes":"n"}],"summary":"s"}\n```')
        out = debrief.coerce_debrief(raw, ["A"])
        self.assertEqual(out.score, 100)
        self.assertEqual(out.domains[0].score, 80)

    def test_string_lists_are_split_into_lines_not_characters(self) -> None:
        raw = {
            "score": 70,
            "domains": [{"name": "History", "score": 70, "notes": "ok"}],
            "strengths": "- Asked about allergies\n- Clear explanation",
            "improvements": "Check vitals earlier",
        }
        out = debrief.coerce_debrief(raw, ["History"])
        self.assertEqual(out.strengths, ["Asked about allergies", "Clear explanation"])
        self.assertEqual(out.improvements, ["Check vitals earlier"])

    def test_no_domains_raises(self) -> None:
        with self.assertRaises(ValueError):
            debrief.coerce_debrief({"score": 5}, [])


if __name__ == "__main__":
    unittest.main()
