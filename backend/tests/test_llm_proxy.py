"""Tests for the patient LLM proxy + /voice/start wiring. No network."""

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

import llm_proxy  # noqa: E402
import server  # noqa: E402
import voice  # noqa: E402

KEYS = {
    "GEMINI_API_KEY": "fake-gemini-key",
    "AGORA_APP_ID": "a" * 32, "AGORA_APP_CERTIFICATE": "b" * 32,
    "AGORA_CUSTOMER_ID": "cid", "AGORA_CUSTOMER_SECRET": "csec",
    "MURF_API_KEY": "murf-zz", "SARVAM_API_KEY": "sarvam-secret-zz",
}
class _Chunks(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        mid = len(self.data) // 2
        yield self.data[:mid]
        yield self.data[mid:]


SSE = b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self._old = {k: os.environ.get(k) for k in [*KEYS, "PUBLIC_URL", "PATIENT_LLM_PROVIDER"]}
        os.environ.update(KEYS)
        os.environ["PATIENT_LLM_PROVIDER"] = "sarvam"
        os.environ.pop("PUBLIC_URL", None)
        self.seen: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            if "agora.io" in req.url.host:
                return httpx.Response(200, json={"agent_id": "ag1"})
            self.seen.append(req)
            body = json.loads(req.content)
            if body.get("stream"):
                return httpx.Response(200, stream=_Chunks(SSE), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        self._orig = (llm_proxy._client_factory, voice._client_factory)
        mk = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: E731
        llm_proxy._client_factory = mk
        voice._client_factory = mk
        self.c = TestClient(server.app)
        self.tok = llm_proxy.mint_token()
        self.auth = {"Authorization": f"Bearer {self.tok}"}

    def tearDown(self):
        llm_proxy._client_factory, voice._client_factory = self._orig
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def post(self, body, headers=None):
        return self.c.post(llm_proxy.PROXY_PATH, json=body, headers=headers if headers is not None else self.auth)

    def test_blank_user_content_replaced(self):
        msgs = [
            {"role": "system", "content": "you are a patient"},
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": [{"type": "text", "text": ""}]},
        ]
        r = self.post({"model": "placeholder", "messages": msgs, "stream": False})
        self.assertEqual(r.status_code, 200)
        sent = json.loads(self.seen[-1].content)
        self.assertEqual(sent["model"], "sarvam-105b-conversations")
        users = [m for m in sent["messages"] if m["role"] == "user"]
        self.assertEqual([m["content"] for m in users], [llm_proxy.PLACEHOLDER])
        self.assertFalse([m for m in sent["messages"] if m["role"] == "assistant"])
        self.assertEqual(self.seen[-1].headers["authorization"], "Bearer sarvam-secret-zz")

    def test_auth_required(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self.post(body, headers={}).status_code, 401)
        self.assertEqual(self.post(body, headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertFalse(self.seen)

    def test_stream_passthrough(self):
        r = self.post({"messages": [{"role": "user", "content": ""}], "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(r.content, SSE)
        self.assertNotIn("sarvam-secret-zz", r.text)

    def _fail_with(self, handler):
        calls = []

        def h(req):
            calls.append(req)
            return handler(req)
        llm_proxy._client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(h))
        old_pause = llm_proxy.RETRY_PAUSE_S
        llm_proxy.RETRY_PAUSE_S = 0
        self.addCleanup(setattr, llm_proxy, "RETRY_PAUSE_S", old_pause)
        return calls

    @staticmethod
    def _sse_text(r) -> str:
        out = []
        for line in r.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:])["choices"][0]["delta"].get("content") or "")
        return "".join(out)

    def test_upstream_error_becomes_natural_reply(self):
        import lang
        calls = self._fail_with(lambda req: httpx.Response(400, json={"detail": "bad sarvam-secret-zz"}))
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        r = self.post({"messages": [{"role": "user", "content": "x"}], "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
        self.assertIn(self._sse_text(r), lang.NATURAL_RETRY_LINES["en"])
        self.assertNotIn("sarvam-secret-zz", r.text)
        self.assertEqual(len(calls), 1)  # 4xx is not retried on the same provider

    def test_all_providers_fail_natural_reply_after_retries(self):
        import lang
        os.environ["GROQ_API_KEY"] = "groq-zz"
        self.addCleanup(os.environ.pop, "GROQ_API_KEY", None)

        def boom(req):
            if "groq" in req.url.host:
                raise httpx.ConnectTimeout("slow", request=req)
            return httpx.Response(503, text="overloaded")
        calls = self._fail_with(boom)
        r = self.post({"messages": [{"role": "user", "content": "since when?"}], "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertIn(self._sse_text(r), lang.NATURAL_RETRY_LINES["en"])
        hosts = [c.url.host for c in calls]
        self.assertEqual(sum("sarvam" in h for h in hosts), 2)  # one quick same-provider retry
        self.assertEqual(sum("groq" in h for h in hosts), 2)

        r = self.post({"messages": [{"role": "user", "content": "x"}], "stream": False})
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["choices"][0]["message"]["content"], lang.NATURAL_RETRY_LINES["en"])

    def test_retry_same_provider_recovers(self):
        n = {"i": 0}

        def flaky(req):
            n["i"] += 1
            if n["i"] == 1:
                return httpx.Response(502, text="bad gateway")
            return httpx.Response(200, stream=_Chunks(SSE), headers={"content-type": "text/event-stream"})
        calls = self._fail_with(flaky)
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        r = self.post({"messages": [{"role": "user", "content": "any pain?"}], "stream": True})
        self.assertEqual(r.content, SSE)
        self.assertEqual(len(calls), 2)

    def test_stream_error_before_first_token_falls_back(self):
        import lang

        class _Broken(httpx.AsyncByteStream):
            async def __aiter__(self):
                raise httpx.ReadTimeout("stalled")
                yield b""  # pragma: no cover
        self._fail_with(lambda req: httpx.Response(200, stream=_Broken(),
                                                   headers={"content-type": "text/event-stream"}))
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        msgs = [{"role": "system", "content": "You are Ravi. LANGUAGE: Speak natural, everyday spoken Hindi."},
                {"role": "user", "content": "क्या हुआ?"}]
        r = self.post({"messages": msgs, "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertIn(self._sse_text(r), lang.NATURAL_RETRY_LINES["hi"])

    def test_conversation_lang(self):
        sysmsg = {"role": "system", "content": "LANGUAGE: Speak natural, everyday spoken Marathi, ..."}
        self.assertEqual(llm_proxy.conversation_lang([sysmsg, {"role": "user", "content": "काय झालं?"}]), "mr")
        self.assertEqual(llm_proxy.conversation_lang([{"role": "user", "content": "ಏನಾಯ್ತು?"}]), "kn")
        self.assertEqual(llm_proxy.conversation_lang([{"role": "user", "content": "kya hua"}]), "en")
        self.assertEqual(llm_proxy.conversation_lang([sysmsg, {"role": "user", "content": ""}]), "mr")

    def test_nurse_turn_still_silent_when_providers_down(self):
        calls = self._fail_with(lambda req: httpx.Response(500, text="down"))
        r = self.post({"messages": [{"role": "user", "content": "Nurse, order a CBC"}], "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self._sse_text(r), "")
        self.assertFalse(calls)

    def test_voice_start_proxy_vs_direct(self):
        req = {"caseId": "c1", "systemPrompt": "sp", "initialLine": "hi", "gender": "M"}
        hdr = {"Origin": "http://localhost:5173"}
        r = self.c.post("/voice/start", json=req, headers=hdr)
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["mode"], "gemini_live")
        self.assertEqual(d["model"], "gemini-3.8-live")
        self.assertTrue(d["wsUrl"].startswith("/voice/live/"))
        self.assertTrue(d["token"])
        self.assertEqual(d["sampleRateIn"], 16000)
        self.assertEqual(d["sampleRateOut"], 24000)

    def test_effort_clamping_low_medium_only(self):
        import llm_provider
        self.assertEqual(llm_provider.clamp_effort("high"), "medium")
        self.assertEqual(llm_provider.clamp_effort("xhigh"), "medium")
        self.assertEqual(llm_provider.clamp_effort("HIGH"), "medium")
        self.assertEqual(llm_provider.clamp_effort("low"), "low")
        self.assertEqual(llm_provider.clamp_effort("medium"), "medium")
        self.assertEqual(llm_provider.clamp_effort(None), "low")


if __name__ == "__main__":
    unittest.main()
