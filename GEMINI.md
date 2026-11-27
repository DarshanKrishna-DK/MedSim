# MedSim - Google Gemini Stack Architecture & Governance

MedSim is a cutting-edge medical simulation and training platform running on a **100% Google Gemini Stack**.

## Core Technology Stack

- **Real-Time Voice / Live Interaction**: Google Gemini Live API (`gemini-3.8-live`) with low-latency bidirectional WebSockets.
- **Text-to-Speech (TTS)**: Gemini TTS (`gemini-3.8-flash-lite-tts`) for natural clinical narration and patient voicing.
- **AI Brain / Reasoning / OSCE / Debrief**: Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) and Gemini 3.8 Flash (`gemini-3.8-flash`).
- **Inference Effort Control**: Always cap reasoning effort at `low` or `medium` — **NEVER** `high` or `xhigh`.
- **Single Store Rule**: One unified state store for simulation state, patient vitals, and evaluation metrics.

## Strict Branch Isolation Policy

- **`MedSim-Gemini` Branch Isolation**: All work must be performed exclusively on the `MedSim-Gemini` branch.
- **NEVER** commit directly to `main`.
- **NEVER** merge `MedSim-Gemini` into `main` or any other branch without explicit governance signoff.

## Prohibited Providers & Frameworks

- **No Third-Party AI/Voice Providers**: Murf, Sarvam, Agora, Groq, Anthropic, OpenAI, or other legacy third-party providers are strictly prohibited in the active Gemini stack.

## Cloud Run Deployment Configuration

- **Target Service**: `medsim-gemini` on Google Cloud Run.
- **Flags & Parameters**: `--timeout=3600 --session-affinity --no-use-http2 --min-instances=0 --max-instances=3 --memory=1Gi --cpu=1 --allow-unauthenticated`.
- **Port**: Port 8080 (`PORT=8080`).
