"""The Gemini Live session tracks runtime token and turn metrics over WebSockets and /voice/metrics."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from fastapi.testclient import TestClient  # noqa: E402

import gemini_live  # noqa: E402
import server  # noqa: E402
import voice  # noqa: E402

HDR = {"Origin": "http://localhost:5173"}


class JoinMetricsTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_metrics_endpoint_overall_and_per_agent(self):
        # Overall metrics
        r = self.client.get("/voice/metrics", headers=HDR)
        self.assertEqual(r.status_code, 200)
        self.assertIn("active_sessions", r.json())

        # Unknown agent metrics returns zeroes
        r2 = self.client.get("/voice/metrics?agentId=unknown-channel", headers=HDR)
        self.assertEqual(r2.status_code, 200)
        d = r2.json()
        self.assertEqual(d["prompt_tokens"], 0)
        self.assertEqual(d["candidates_tokens"], 0)
        self.assertEqual(d["total_tokens"], 0)

    def test_prepared_session_metrics(self):
        channel = "test-metrics-chan"
        tok = gemini_live.register_prepared_session(
            channel=channel,
            case_id="comm-001",
            system_prompt="prompt",
            initial_line="hello",
            voice="Aoede",
        )
        self.assertTrue(tok)
        m = gemini_live.get_session_metrics(channel)
        self.assertEqual(m["prompt_tokens"], 0)
        self.assertEqual(m["candidates_tokens"], 0)
        self.assertEqual(m["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
