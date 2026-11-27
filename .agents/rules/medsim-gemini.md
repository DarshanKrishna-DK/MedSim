---
name: medsim-gemini-rules
description: Enforces MedSim-Gemini architecture, 100% Google Gemini stack, effort caps, and branch isolation.
---

# MedSim Gemini Governance & Architecture Rules

1. **100% Google Gemini Stack**:
   - Real-Time Live API: `gemini-3.8-live`
   - TTS: `gemini-3.8-flash-lite-tts`
   - AI Brain / OSCE / Debrief: `gemini-3.5-flash-lite` / `gemini-3.8-flash`
   - No Murf, Sarvam, Agora, Groq, or Anthropic.

2. **Inference Effort Cap**:
   - Always cap reasoning effort at `low` or `medium`. **NEVER** use `high` or `xhigh`.

3. **Strict Branch Isolation**:
   - Always work on `MedSim-Gemini`. **NEVER** commit to `main` or merge `MedSim-Gemini` into `main`.

4. **Cloud Run Deployment**:
   - `--timeout=3600 --session-affinity --no-use-http2 --port=8080`.
