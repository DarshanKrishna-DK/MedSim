"""
FastAPI backend for the MedSim simulator.

Hosts:
  /agent/*            Claude Managed Agents proxy (medsim-attending; optional)
  /agent/nurse/act    agentic charge-nurse tool-use loop (Sarvam / Groq)
  /agent/debrief/simple  JSON debrief without Anthropic
  /agent/debrief/osce    OSCE examiner marking (checklist + global rating)
  /voice/*            Agora Conversational AI Engine + Murf Falcon TTS
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Optional


def _load_env_local() -> None:
    """Minimal .env.local loader, no python-dotenv dependency.

    Reads `backend/.env.local` (next to this file) and sets any KEY=VALUE
    pair into ``os.environ`` without overwriting values already set.
    Silently no-ops if the file is missing."""
    # Never load live keys in a test run: the tests once booked real calendar events.
    if "unittest" in (sys.argv[0] if sys.argv else "") or "pytest" in sys.modules:
        return
    env_path = Path(__file__).resolve().parent / ".env.local"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Overwrite if the existing value is empty (subprocesses can inherit
        # declared-but-empty env vars from the parent). Only preserve a
        # non-empty existing value.
        if key and not os.environ.get(key):
            os.environ[key] = value


_load_env_local()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

import llm_provider
from llm_provider import llm_health, strip_think
import llm_proxy
import voice
if not hasattr(voice, "_client_factory"):
    import httpx
    voice._client_factory = lambda: httpx.AsyncClient(timeout=20.0)
from voice import router as voice_router, voice_health
from nurse import router as nurse_router
from debrief import router as debrief_router
from teacher import router as teacher_router
from coach import router as coach_router
from radiology import router as radiology_router
from ot import router as ot_router
from osce import router as osce_router
from rx_audio import router as rx_audio_router
from exam import router as exam_router
from observer import router as observer_router
from viva import router as viva_router
from logbook import router as logbook_router

# The API (/agent, /voice, /teach, /coach, /llm) is only served to trusted
# browser origins: localhost for `npm run dev`, plus the deployed site via
# FRONTEND_ORIGINS. Anything else needs the optional shared-secret header, or
# gets a 401 before any Agora / Murf / LLM credits are spent.
SHARED_SECRET = (
    os.environ.get("BACKEND_SHARED_SECRET")
    or os.environ.get("MEDSIM_SHARED_SECRET", "")
)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
DEV_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
}
# Any local dev port (Vite falls back to 5174, 5175, ... when 5173 is busy).
DEV_ORIGIN_RE = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d{2,5})?$")


# Extra trusted browser origins (comma-separated), e.g. the deployed site URL.
# Treated like dev origins: allowed by CORS and exempt from the shared secret.
FRONTEND_ORIGINS = {
    o.strip().rstrip("/")
    for var in ("FRONTEND_ORIGINS", "MEDSIM_ALLOWED_ORIGINS")
    for o in os.environ.get(var, "").split(",")
    if o.strip()
}
ALLOWED_ORIGINS.extend(sorted(FRONTEND_ORIGINS - set(ALLOWED_ORIGINS)))


# Paths that cost credits or touch providers. Everything else (the SPA) is public.
API_PREFIXES = ("/agent/", "/voice/", "/teach/", "/coach/", "/llm/")


def _is_dev_origin(origin: str) -> bool:
    return origin in DEV_ORIGINS or origin in FRONTEND_ORIGINS or bool(DEV_ORIGIN_RE.match(origin))

# Per-IP rate limit caps even authenticated abuse. SSE streams count as one
# request, so 120/min leaves plenty of headroom for legitimate use.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

from contextlib import asynccontextmanager

BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8787"))


@asynccontextmanager
async def _lifespan(_app):
    running_on_cloud_run = bool(os.environ.get("K_SERVICE"))
    if not running_on_cloud_run:
        llm_proxy.start_tunnel(BACKEND_PORT)
    from integrations import call_reminders

    call_reminders.start()  # scheduled "Asha is ready for your check-in" messages
    import telegram_bot

    if not running_on_cloud_run:
        telegram_bot.start()  # @medsim_asha_bot case-authoring chat (long polling)
    try:
        yield
    finally:
        if not running_on_cloud_run:
            telegram_bot.stop()
        call_reminders.stop()
        if not running_on_cloud_run:
            llm_proxy.stop_tunnel()


app = FastAPI(title="medsim Backend", version="0.2.0", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Middleware ORDER (inside-out: last added runs first on inbound):
#   1. Auth          (innermost, added first)
#   2. SlowAPI       (rate limit)
#   3. CORS          (outermost, handles OPTIONS preflight before auth)
@app.middleware("http")
async def require_shared_secret(request: Request, call_next):
    path = request.url.path
    # /health is public for monitoring; CORS preflight runs above this anyway
    # but bypass OPTIONS defensively.
    if path == "/health" or request.method == "OPTIONS":
        return await call_next(request)
    # Only the API is guarded. The built SPA and its static assets (/, /assets/*,
    # /index.html) are public: a top-level page load sends no Origin header.
    if not path.startswith(API_PREFIXES):
        return await call_next(request)
    # Agora's cloud calls /llm/* with no Origin; that route does its own
    # per-agent bearer-token check (llm_proxy.valid_token).
    if path.startswith("/llm/"):
        return await call_next(request)
    req_scheme = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
    req_host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip()

    def _same_deployed_origin(cand: str) -> bool:
        return (
            bool(cand)
            and bool(req_host)
            and req_host not in ("testserver", "localhost", "127.0.0.1")
            and cand == f"{req_scheme}://{req_host}"
        )

    origin = request.headers.get("origin", "")
    if _is_dev_origin(origin) or _same_deployed_origin(origin):
        return await call_next(request)
    # Same-origin GETs (incl. EventSource) don't send Origin per the Fetch
    # spec, but they DO send Referer. Trust dev-origin Referer in lieu of
    # Origin so SSE streams from localhost work without an explicit secret.
    referer = request.headers.get("referer", "")
    ref_origin = re.match(r"^(https?://[^/]+)/", referer)
    if ref_origin and (_is_dev_origin(ref_origin.group(1)) or _same_deployed_origin(ref_origin.group(1))):
        return await call_next(request)
    auth_hdr = request.headers.get("x-medsim-auth") or request.headers.get("x-medsim-secret")
    if SHARED_SECRET and auth_hdr == SHARED_SECRET:
        return await call_next(request)
    return JSONResponse({"detail": "unauthorized"}, status_code=401)


app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=DEV_ORIGIN_RE.pattern,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Frontend polls this before showing the attending dock so a missing
    API key or unbootstrapped agent surfaces a clearer error than a blank
    SSE failure."""
    gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
    has_key = gemini_key or bool(os.environ.get("ANTHROPIC_API_KEY"))
    agent_id = os.environ.get("MEDSIM_AGENT_ID", "") or "medsim-attending-gemini"
    env_id = os.environ.get("MEDSIM_ENV_ID", "") or "medsim-attending-env-gemini"
    bootstrapped = bool(agent_id and env_id)
    return {
        "ok": True,
        "stack": "gemini",
        "gemini_configured": gemini_key,
        "voice": voice_health(),
        "llm": {**llm_health(), "patient_route": llm_proxy.patient_route()},
        "agent": {
            "anthropic_sdk_installed": _HAS_ANTHROPIC,
            "api_key_configured": has_key,
            "gemini_configured": gemini_key,
            "bootstrapped": bootstrapped,
            "agent_id": agent_id,
            "environment_id": env_id,
            "model": os.environ.get("GEMINI_REASONING_MODEL", "gemini-3.8-flash"),
            "stack": "gemini",
        },
    }


# ───────────────────────────────────────────────────────────────────────────
# Claude Managed Agents proxy
# ───────────────────────────────────────────────────────────────────────────
#
# The browser talks to this server instead of the Anthropic API directly so
# that (a) the Managed Agents API key stays server-side and (b) the
# one-time bootstrap (agents.create + environments.create) is done here
# once and the resulting IDs are reused across sessions.
#
# Per-env vars:
#   ANTHROPIC_API_KEY: required. Server-side only; never exposed to the
#                        browser. Separate from VITE_ANTHROPIC_API_KEY used
#                        by the browser Haiku patient-persona path.
#   MEDSIM_AGENT_ID: persisted agent ID (bootstrap returns it the
#                        first time; set it here afterwards to skip
#                        re-creating).
#   MEDSIM_ENV_ID: persisted environment ID (same pattern).
#
# Endpoints:
#   POST /agent/bootstrap: idempotent; creates
#                                                env+agent if the env vars
#                                                are unset, else returns
#                                                the cached IDs.
#   POST /agent/sessions: create a new session for
#                                                the current bootstrapped
#                                                agent.
#   GET  /agent/sessions/{sid}/stream: SSE proxy of the live
#                                                event stream. Used by
#                                                eventStreamRenderer.tsx.
#   GET  /agent/sessions/{sid}/events: paginated history (for
#                                                the reconnect+dedupe
#                                                pattern).
#   POST /agent/sessions/{sid}/events: forward user events
#                                                (user.message,
#                                                user.custom_tool_result,
#                                                user.interrupt, etc.).
#   POST /agent/vault/ehr/lookup: credential-vault demo:
#                                                attaches EHR_API_TOKEN
#                                                server-side and returns
#                                                a fake record. The token
#                                                never leaves the process.
#   POST /agent/triage/classify: one-shot Opus 4.7 ESI
#                                                classifier for ER
#                                                arrivals. Stateless;
#                                                separate from the
#                                                Managed Agent session.
#
# TODO: verify wire names against
# https://platform.claude.com/docs/en/managed-agents/ before submission: 
# the SDK is beta and field names can drift.

import asyncio
import json
import logging

from fastapi import Request
from fastapi.responses import StreamingResponse

try:
    from anthropic import Anthropic, AsyncAnthropic  # type: ignore
    _HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover
    _HAS_ANTHROPIC = False

# Structured logger for the Managed Agents proxy. Uvicorn captures stdlib
# logging so these land in the same stream as its own access log.
_agent_log = logging.getLogger("medsim.agent")
_agent_log.setLevel(logging.INFO)
if not _agent_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[medsim.agent] %(levelname)s %(message)s"))
    _agent_log.addHandler(_h)

# Guards against two concurrent /agent/bootstrap calls creating two
# agents + two environments. Threading.Lock because bootstrap runs in
# FastAPI's threadpool (sync endpoint).
_bootstrap_lock = threading.Lock()

# SSE keepalive. EventSource will silently time out if the connection is
# idle past the browser's threshold (~30s for Chrome, longer elsewhere).
# We emit an SSE comment line every N seconds so the socket stays warm
# and the browser's `onerror` reconnect logic doesn't fire.
SSE_KEEPALIVE_SEC = 15.0


AGENT_MODEL = "claude-opus-4-7"
AGENT_NAME = "medsim-attending"
ENV_NAME = "medsim-attending-env"

# Direct-inference model for the dedicated triage-reasoning endpoint.
# Pinned to Opus 4.7 because the spec reserves triage for the strongest
# clinical-reasoning model: see CLAUDE.md's "Model routing" table. This
# is a separate code path from the Managed Agent (AGENT_MODEL); the
# agent observes the whole encounter, triage classifies one arrival.
TRIAGE_MODEL = "claude-opus-4-7"
TRIAGE_MAX_TOKENS = 512

MEDSIM_ATTENDING_SYSTEM_PROMPT = (
    "You are the attending physician supervising a trainee in a clinical "
    "training simulator. Your role is to OBSERVE their decisions and "
    "GRADE the encounter. You are NOT an assistant, a guide, or a "
    "coach. Silence is acceptable and often correct.\n\n"

    "The simulator runs in one of two modes at a time; the event stream "
    "makes it explicit:\n"
    "  • EMERGENCY ROOM (ER), triaged by severity, 4 beds in "
    "red/yellow/green zones. Events: [ER arrival], [test ordered], "
    "[treatment given], [diagnosis submitted], [disposition].\n"
    "  • POLYCLINIC (outpatient), no triage zones, no beds, one "
    "patient at a time, tests resolve instantly. Events: "
    "[polyclinic arrival], [poly test], [poly diagnosis], [poly rx], "
    "[disposition].\n\n"

    "Custom-tool usage by mode:\n"
    "  • ER mode only: render_triage_badge (red/yellow/green + one-line "
    "rationale), render_bed_map.\n"
    "  • Both modes: render_vitals_chart, render_patient_timeline, "
    "render_case_evaluation, flag_critical_finding, lookup_ehr_history.\n"
    "  • DO NOT emit render_triage_badge or render_bed_map during "
    "polyclinic events, triage zones and beds are ER concepts and "
    "emitting them confuses the UI.\n\n"

    "Permission policy, custom tools:\n"
    "  • All render_* tools are auto-allowed; the trainee's UI renders "
    "them immediately and acks back to you. Use them freely.\n"
    "  • flag_critical_finding is a confirm-gated write. The trainee "
    "sees an approve/decline dialog before the banner fires; the "
    "tool_result is delayed until they choose. Reserve it for peri-"
    "arrest vitals, closing stroke window, airway compromise, or "
    "anaphylaxis. Never flag stable patients, and emit at most one "
    "flag per encounter.\n"
    "  • lookup_ehr_history is auto-allowed; it routes through the "
    "credential vault so the EHR auth token never enters your context. "
    "Call it once per patient when prior history or medication list "
    "would change your assessment (e.g., unclear cardiac history, "
    "possible drug interaction). Do not call it for every arrival.\n\n"

    "What you DO:\n"
    "  • On [ER arrival]: optionally emit one render_triage_badge with a "
    "rationale citing specific vitals or chief-complaint evidence. If "
    "the vitals are unremarkable and the presentation doesn't warrant "
    "a zone call, stay silent.\n"
    "  • On [polyclinic arrival]: stay silent. Do not greet the patient "
    "or ask the trainee what they want to do.\n"
    "  • At debrief time (see DEBRIEF MODE below): emit exactly one "
    "render_case_evaluation. Never emit it before the trainee has "
    "submitted a diagnosis.\n"
    "  • Any text you do emit: at most one sentence, observational tone, "
    "no questions.\n\n"

    "What you DO NOT do:\n"
    "  • Do not ask the trainee questions ('what would you like to do "
    "first?'). Never.\n"
    "  • Do not narrate the scene ('Mr. Williams is roomed and ready.').\n"
    "  • Do not suggest next steps before the trainee has acted.\n"
    "  • Do not repeat what the trainee can already see in the UI.\n"
    "  • Do not reveal the correct diagnosis before disposition.\n\n"

    "DEBRIEF MODE, end-of-encounter grading.\n\n"

    "When you receive a [debrief request] message, the trainee has ended "
    "the encounter. The message body contains, as JSON:\n"
    "  • case_id and the case's correctDiagnosisId (gold standard).\n"
    "  • rubric, a CaseRubric with three domains "
    "(data_gathering, clinical_management, interpersonal) plus optional "
    "safety_netting. Each criterion has a label, weight, and an "
    "`evidence` string telling you exactly what counts as 'met'.\n"
    "  • registry_slice, the subset of guidelines/recommendations cited "
    "by the rubric. Use ONLY recIds that appear here. Do not invent.\n"
    "  • encounter_log, chronological list of: history questions asked "
    "(with answers shown to the trainee), tests ordered with timestamps, "
    "treatments/prescriptions given, the submitted diagnosis, and any "
    "free-text counselling captured. Plus the voice transcript if "
    "available.\n\n"

    "Process:\n"
    "  1. For every criterion in rubric.data_gathering, "
    "rubric.clinical_management, and rubric.interpersonal, decide one of "
    "{met, partially-met, missed} using the criterion's `evidence` field "
    "as your match key. Quote the trainee directly or name the action "
    "in the `evidence` field of your output (not the rubric's evidence "
    "string, your own observation).\n"
    "  2. Compute domain_scores: raw = sum of weights for met (1.0×) + "
    "partially-met (0.5×); max = sum of all weights in that domain. "
    "Verdict bands: ≥0.85 excellent, ≥0.70 good, ≥0.55 satisfactory, "
    "≥0.40 borderline, otherwise clear-fail.\n"
    "  3. Set global_rating with the same bands applied to the total "
    "across all three domains.\n"
    "  4. If the trainee did anything dangerous, contraindicated drug, "
    "missed a red-flag escalation that the rubric flagged, no safety-"
    "netting on a high-risk diagnosis, set safety_breach with `what` "
    "and a guideline_ref if one applies. The narrative MUST lead with "
    "this regardless of the score.\n"
    "  5. Pick 1–3 highlights (specific strengths the trainee actually "
    "demonstrated) and 1–3 improvements (priority gaps). Do not list "
    "everything; the trainee tunes out.\n"
    "  6. Write narrative last, 1–2 paragraphs, voice of a senior "
    "clinician giving a teaching debrief immediately after the case. "
    "No praise sandwiches, no sycophancy, no generic encouragement.\n"
    "  7. Emit ONE render_case_evaluation tool use with the full payload. "
    "Then stop.\n\n"

    "Hard rules, non-negotiable:\n"
    "  • Cite, don't invent. Every clinical_management criterion's "
    "guideline_ref MUST appear in the registry_slice. If the rubric "
    "criterion has no guideline_ref AND no rec applies, drop the "
    "criterion from your output rather than fabricating one.\n"
    "  • Specific evidence. 'You missed ICE' is not enough. 'You closed "
    "without asking what the patient was worried about, they hinted at "
    "fear of stroke when they mentioned their father; that was a chance "
    "to address concerns and tailor the explanation.' is the bar.\n"
    "  • No medical advice for real patients. This is a training "
    "simulator. Do not frame any output as guidance for actual care.\n"
    "  • Cases are synthetic and doses simplified, do not hold the "
    "trainee to a recommendation that is not in the registry_slice.\n\n"

    "Scope: the cases are synthetic, the medication doses are simplified, "
    "and the trainee is not a licensed clinician. Do not offer medical "
    "advice outside the simulator."
)

# Custom tool JSON schemas: must match the Zod schemas in
# src/agents/customTools.ts. If you change either side, update both.
MEDSIM_CUSTOM_TOOLS: list[dict] = [
    {
        "type": "custom",
        "name": "render_vitals_chart",
        "description": (
            "Display the patient's vitals (HR, BP, SpO2, temp, RR) as a "
            "line chart over the course of the encounter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
            },
            "required": ["patient_id"],
        },
    },
    {
        "type": "custom",
        "name": "render_bed_map",
        "description": (
            "Display the current bed occupancy map across the four ER beds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "custom",
        "name": "render_triage_badge",
        "description": (
            "Display a triage-priority badge with a one-line rationale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string", "enum": ["red", "yellow", "green"]},
                "reason": {"type": "string"},
            },
            "required": ["zone", "reason"],
        },
    },
    {
        "type": "custom",
        "name": "render_patient_timeline",
        "description": (
            "Display the tests ordered and treatments given for a patient "
            "in chronological order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
            },
            "required": ["patient_id"],
        },
    },
    {
        # End-of-encounter OSCE debrief. Replaces the older
        # `render_case_grade` (a flat score+notes blob): this one carries
        # a per-criterion verdict, three-domain scores, citations into the
        # guideline registry the frontend ships with the debrief request,
        # and a 1–2 paragraph spoken-aloud narrative. The renderer
        # `<CaseEvaluationCard>` resolves every `guideline_ref` against the
        # registry and shows a verbatim cite-card next to each criterion.
        #
        # The Zod schema in src/agents/customTools.ts must mirror this; if
        # you change one, update the other.
        "type": "custom",
        "name": "render_case_evaluation",
        "description": (
            "End-of-encounter PLAB2-style debrief. Emit exactly once after "
            "the trainee submits their diagnosis (and prescription, in "
            "polyclinic). Score three domains (data_gathering, clinical_"
            "management, interpersonal) against the case rubric provided "
            "in the debrief request. Each criterion verdict (met / "
            "partially-met / missed) must be backed by a transcript quote "
            "or a named action. Every clinical_management criterion's "
            "guideline_ref MUST be a real recommendation id present in the "
            "guideline registry slice that accompanies the debrief request "
            ", if no rec applies, drop the criterion. Never fabricate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "global_rating": {
                    "type": "string",
                    "enum": [
                        "clear-fail",
                        "borderline",
                        "satisfactory",
                        "good",
                        "excellent",
                    ],
                },
                "domain_scores": {
                    "type": "object",
                    "properties": {
                        "data_gathering": {
                            "type": "object",
                            "properties": {
                                "raw": {"type": "number"},
                                "max": {"type": "number"},
                                "verdict": {
                                    "type": "string",
                                    "enum": [
                                        "clear-fail",
                                        "borderline",
                                        "satisfactory",
                                        "good",
                                        "excellent",
                                    ],
                                },
                            },
                            "required": ["raw", "max", "verdict"],
                        },
                        "clinical_management": {
                            "type": "object",
                            "properties": {
                                "raw": {"type": "number"},
                                "max": {"type": "number"},
                                "verdict": {
                                    "type": "string",
                                    "enum": [
                                        "clear-fail",
                                        "borderline",
                                        "satisfactory",
                                        "good",
                                        "excellent",
                                    ],
                                },
                            },
                            "required": ["raw", "max", "verdict"],
                        },
                        "interpersonal": {
                            "type": "object",
                            "properties": {
                                "raw": {"type": "number"},
                                "max": {"type": "number"},
                                "verdict": {
                                    "type": "string",
                                    "enum": [
                                        "clear-fail",
                                        "borderline",
                                        "satisfactory",
                                        "good",
                                        "excellent",
                                    ],
                                },
                            },
                            "required": ["raw", "max", "verdict"],
                        },
                    },
                    "required": [
                        "data_gathering",
                        "clinical_management",
                        "interpersonal",
                    ],
                },
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion_id": {"type": "string"},
                            "domain": {
                                "type": "string",
                                "enum": [
                                    "data_gathering",
                                    "clinical_management",
                                    "interpersonal",
                                ],
                            },
                            "verdict": {
                                "type": "string",
                                "enum": ["met", "partially-met", "missed"],
                            },
                            "evidence": {"type": "string"},
                            "guideline_ref": {
                                "type": ["string", "null"],
                                "description": (
                                    "Format: '<guideline_id>:<rec_id>'. "
                                    "Required for clinical_management; "
                                    "optional elsewhere; null if not "
                                    "applicable."
                                ),
                            },
                        },
                        "required": [
                            "criterion_id",
                            "domain",
                            "verdict",
                            "evidence",
                        ],
                    },
                },
                "safety_breach": {
                    "type": ["object", "null"],
                    "description": (
                        "Set ONLY when the trainee did something dangerous "
                        "(contraindicated drug, missed red flag, no safety-"
                        "netting on a high-risk dx). The narrative must "
                        "lead with this regardless of total score."
                    ),
                    "properties": {
                        "what": {"type": "string"},
                        "guideline_ref": {"type": ["string", "null"]},
                    },
                },
                "highlights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1–3 specific strengths.",
                },
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1–3 priority improvements.",
                },
                "narrative": {
                    "type": "string",
                    "description": (
                        "1–2 paragraph teaching debrief, written as if "
                        "spoken aloud by a senior clinician. No praise "
                        "sandwiches, no sycophancy."
                    ),
                },
            },
            "required": [
                "case_id",
                "global_rating",
                "domain_scores",
                "criteria",
                "highlights",
                "improvements",
                "narrative",
            ],
        },
    },
    {
        # Write-shaped tool: surfaces a disruptive banner in the trainee
        # UI. Gated by the frontend permission policy: the renderer shows
        # an approve/decline dialog and only acks once the human confirms.
        # Custom tools aren't covered by Anthropic's own permission-policy
        # gate (that's native + MCP tools only), so the confirm happens
        # client-side in src/agents/eventStreamRenderer.tsx.
        "type": "custom",
        "name": "flag_critical_finding",
        "description": (
            "Raise a disruptive critical-finding banner on the trainee's "
            "screen. Use ONLY when the patient is in imminent risk (peri-"
            "arrest vitals, stroke window closing, anaphylaxis). Requires "
            "explicit human confirmation before firing; do not expect the "
            "result immediately."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "severity": {"type": "string", "enum": ["critical", "urgent"]},
                "reason": {"type": "string"},
            },
            "required": ["patient_id", "severity", "reason"],
        },
    },
    {
        # Credential-vault demo tool. When the agent emits this, the
        # browser calls POST /agent/vault/ehr/lookup. The backend attaches
        # the EHR auth token from server-side state (env var) and returns
        # the fake record. The EHR_API_TOKEN never touches the Claude
        # context or the browser: it's the "credential vault" pattern
        # from Michael's Managed Agents session, modeled for a demo.
        "type": "custom",
        "name": "lookup_ehr_history",
        "description": (
            "Retrieve the patient's prior EHR encounters and medication "
            "list from the hospital EHR system. The request is routed "
            "through the credential vault so your context never sees "
            "the EHR auth token."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
            },
            "required": ["patient_id"],
        },
    },
]

_anthropic_client: Optional["Anthropic"] = None
_anthropic_async_client: Optional["AsyncAnthropic"] = None


def _ensure_anthropic_available() -> None:
    if not _HAS_ANTHROPIC:
        raise HTTPException(
            status_code=500,
            detail=(
                "anthropic package not installed. Run `pip install "
                "'anthropic>=0.88.0'` in the backend venv."
            ),
        )


def _require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set server-side.",
        )
    return key


def get_anthropic_client() -> "Anthropic":
    global _anthropic_client
    _ensure_anthropic_available()
    if _anthropic_client is None:
        _anthropic_client = Anthropic(api_key=_require_api_key())
    return _anthropic_client


def get_async_anthropic_client() -> "AsyncAnthropic":
    global _anthropic_async_client
    _ensure_anthropic_available()
    if _anthropic_async_client is None:
        _anthropic_async_client = AsyncAnthropic(api_key=_require_api_key())
    return _anthropic_async_client


class BootstrapResponse(BaseModel):
    agent_id: str
    agent_version: int | None
    environment_id: str
    created: bool  # True if we created them this call, False if cached


@app.post("/agent/bootstrap", response_model=BootstrapResponse)
def bootstrap_agent():
    """Idempotent: creates the medsim attending agent + environment if
    MEDSIM_AGENT_ID and MEDSIM_ENV_ID aren't set; otherwise returns the
    cached IDs.

    When `created=True` the caller should persist `agent_id` and
    `environment_id` into `backend/.env.local` (or the OS environment) so
    the next startup skips the create calls.

    The whole body runs under ``_bootstrap_lock`` so two racing clients
    (e.g. Strict Mode double-mount with a cold server) don't each create
    their own agent + environment.
    """
    with _bootstrap_lock:
        agent_id = os.environ.get("MEDSIM_AGENT_ID", "") or "medsim-attending-gemini"
        env_id = os.environ.get("MEDSIM_ENV_ID", "") or "medsim-attending-env-gemini"
        os.environ["MEDSIM_AGENT_ID"] = agent_id
        os.environ["MEDSIM_ENV_ID"] = env_id
        return BootstrapResponse(
            agent_id=agent_id,
            agent_version=1,
            environment_id=env_id,
            created=False,
        )


class RefreshAgentResponse(BaseModel):
    agent_id: str
    version: int | None


@app.post("/agent/refresh", response_model=RefreshAgentResponse)
def refresh_agent():
    agent_id = os.environ.get("MEDSIM_AGENT_ID", "") or "medsim-attending-gemini"
    return RefreshAgentResponse(
        agent_id=agent_id,
        version=1,
    )


class AttendingSession:
    def __init__(self, session_id: str, title: str = "medsim training shift"):
        self.session_id = session_id
        self.title = title
        self.events: list[dict[str, Any]] = []
        self.listeners: list[asyncio.Queue] = []
        self.lock = asyncio.Lock()
        self.model_messages: list[dict[str, Any]] = [
            {"role": "system", "content": MEDSIM_ATTENDING_SYSTEM_PROMPT}
        ]

    def add_event(self, event: dict[str, Any]) -> None:
        if "id" not in event:
            event["id"] = f"evt_{uuid.uuid4().hex[:12]}"
        if "processed_at" not in event and "created_at" not in event:
            event["processed_at"] = datetime.now(timezone.utc).isoformat()
        self.events.append(event)
        for q in list(self.listeners):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    async def handle_incoming_events(self, incoming: list[dict[str, Any]]) -> None:
        async with self.lock:
            needs_turn = False
            for ev in incoming:
                self.add_event(ev)
                etype = ev.get("type", "")
                if etype == "user.message":
                    text = ""
                    for b in ev.get("content", []):
                        if isinstance(b, dict) and b.get("type") == "text":
                            text += b.get("text", "")
                    if text:
                        self.model_messages.append({"role": "user", "content": text})
                        needs_turn = True
                elif etype == "user.custom_tool_result":
                    tool_use_id = ev.get("custom_tool_use_id", "")
                    res_text = ""
                    for b in ev.get("content", []):
                        if isinstance(b, dict) and b.get("type") == "text":
                            res_text += b.get("text", "")
                    self.model_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": res_text or "rendered",
                    })
                    needs_turn = True

            if needs_turn:
                await self._run_attending_turn()

    async def _run_attending_turn(self) -> None:
        max_depth = 5
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in MEDSIM_CUSTOM_TOOLS
        ]

        for _ in range(max_depth):
            try:
                msg, _ = await llm_provider.chat(
                    self.model_messages,
                    tools=tools,
                    tier="reasoning",
                    effort="medium",
                )
            except Exception as e:
                _agent_log.exception("Attending agent reasoning turn failed: %s", e)
                self.add_event({
                    "id": f"err_{uuid.uuid4().hex[:12]}",
                    "type": "proxy_error",
                    "message": str(e),
                })
                break

            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            if content:
                clean_text = strip_think(content)
                if clean_text:
                    self.add_event({
                        "id": f"msg_{uuid.uuid4().hex[:12]}",
                        "type": "agent.message",
                        "content": [{"type": "text", "text": clean_text}],
                    })

            if not tool_calls:
                self.model_messages.append({"role": "assistant", "content": content})
                self.add_event({
                    "id": f"idle_{uuid.uuid4().hex[:12]}",
                    "type": "session.status_idle",
                    "stop_reason": {"type": "end_turn"},
                })
                break

            self.model_messages.append(msg)
            server_resolved_count = 0

            for tc in tool_calls:
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {}

                self.add_event({
                    "id": call_id,
                    "type": "agent.custom_tool_use",
                    "name": fn_name,
                    "input": args,
                })

                if fn_name == "lookup_ehr_history":
                    pid = str(args.get("patient_id", "")).strip()
                    rec = FAKE_EHR_RECORDS.get(pid, {"error": f"patient_id not found: {pid}"})
                    rec_str = json.dumps(rec)
                    self.add_event({
                        "id": f"res_{uuid.uuid4().hex[:12]}",
                        "type": "user.custom_tool_result",
                        "custom_tool_use_id": call_id,
                        "content": [{"type": "text", "text": rec_str}],
                    })
                    self.model_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": rec_str,
                    })
                    server_resolved_count += 1

            if server_resolved_count < len(tool_calls):
                self.add_event({
                    "id": f"idle_{uuid.uuid4().hex[:12]}",
                    "type": "session.status_idle",
                    "stop_reason": {"type": "requires_action"},
                })
                break


_sessions: dict[str, AttendingSession] = {}


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str


@app.post("/agent/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    sid = f"sess_{uuid.uuid4().hex[:16]}"
    title = req.title or "medsim training shift"
    sess = AttendingSession(session_id=sid, title=title)
    _sessions[sid] = sess
    _agent_log.info("create_session: %s", sid)
    return CreateSessionResponse(session_id=sid)


@app.get("/agent/sessions/{session_id}")
async def get_session(session_id: str):
    sess = _sessions.get(session_id)
    if not sess:
        return {"id": session_id, "status": "active", "title": "medsim training shift"}
    return {"id": session_id, "status": "active", "title": sess.title, "event_count": len(sess.events)}


@app.post("/agent/sessions/{session_id}/events")
async def send_events(session_id: str, request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}")
    events = body.get("events") if isinstance(body, dict) else None
    if not isinstance(events, list) or not events:
        raise HTTPException(status_code=400, detail="events must be a non-empty list")
    sess = _sessions.get(session_id)
    if not sess:
        sess = AttendingSession(session_id=session_id)
        _sessions[session_id] = sess
    try:
        await sess.handle_incoming_events(events)
    except Exception as e:
        _agent_log.exception(
            "send_events failed session_id=%s event_count=%d",
            session_id, len(events),
        )
        raise HTTPException(status_code=500, detail=f"send_events failed: {e}")
    return {"ok": True}


@app.get("/agent/sessions/{session_id}/events")
async def list_events(session_id: str, limit: int = 1000):
    """Paginated history, used by the browser on reconnect to backfill
    events emitted while the SSE stream was down."""
    sess = _sessions.get(session_id)
    events = sess.events if sess else []
    return {"data": events[-limit:]}


@app.get("/agent/sessions/{session_id}/stream")
async def stream_events(session_id: str, request: Request):
    """SSE passthrough. Each Managed-Agents event becomes one SSE
    ``event:``/``data:`` pair so EventSource in the browser can dispatch
    by type.

    Uses the ASYNC Anthropic client so long-lived streams cooperate with
    FastAPI's event loop, a synchronous generator here would tie up a
    threadpool worker per open stream, and with hot-reloading + Strict
    Mode double-mount, those streams pile up and saturate the pool,
    which makes EVERY endpoint (including /health) stop responding.

    Three robustness features:
      1. ``request.is_disconnected()`` checked on every tick, so the
         upstream stream is released as soon as the browser closes its
         EventSource.
      2. ``asyncio.wait_for`` wraps ``anext`` with a timeout, if no
         upstream event arrives for ``SSE_KEEPALIVE_SEC`` seconds we
         emit a comment line (``: keepalive\\n\\n``) to keep the socket
         warm and to run the disconnect check. Without this the browser
         (Chrome ~30s, nginx default 60s, corporate proxies often less)
         can silently drop idle streams.
      3. Any exception bubbling out of the upstream SDK becomes a
         ``proxy_error`` SSE event so the client knows the pipe died
         instead of silently seeing EOF.
    """
    sess = _sessions.get(session_id)
    if not sess:
        sess = AttendingSession(session_id=session_id)
        _sessions[session_id] = sess

    queue: asyncio.Queue = asyncio.Queue()
    sess.listeners.append(queue)

    async def generator():
        yield ": connected\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_SEC)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                etype = event.get("type", "message")
                data = json.dumps(event, default=str)
                yield f"event: {etype}\ndata: {data}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _agent_log.exception("SSE stream failed session_id=%s", session_id)
            err = json.dumps({"type": "proxy_error", "message": str(e)})
            yield f"event: proxy_error\ndata: {err}\n\n"
        finally:
            if queue in sess.listeners:
                sess.listeners.remove(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ───────────────────────────────────────────────────────────────────────────
# Credential vault: hospital EHR stub
# ───────────────────────────────────────────────────────────────────────────
#
# Demo of Michael's "credential vault" pattern: a third-party system
# (fake hospital EHR) needs an auth token to query patient history. The
# token lives ONLY on the backend (EHR_API_TOKEN env var). The agent's
# context window never sees it, the browser never sees it, and it never
# appears in any event written to the Managed Agents session.
#
# Flow:
#   1. Agent emits agent.custom_tool_use name=lookup_ehr_history.
#   2. Browser receives the event, POSTs to /agent/vault/ehr/lookup.
#   3. This endpoint attaches EHR_API_TOKEN server-side, calls the fake
#      EHR (local dict for the demo; would be an HTTP call in prod),
#      and returns the history JSON.
#   4. Browser posts the JSON back as user.custom_tool_result.
#
# Logging is intentionally token-free: the log line records that the
# vault was used and which patient was queried, but never the token
# value. A grep for the token in logs should return zero hits.

FAKE_EHR_RECORDS: dict[str, dict] = {
    "poly-001": {
        "patient_id": "poly-001",
        "name": "Mehmet Demir",
        "prior_encounters": [
            {"date": "2025-11-14", "reason": "hypertension follow-up", "bp": "148/92"},
            {"date": "2025-07-02", "reason": "annual physical", "bp": "140/88"},
        ],
        "active_medications": [
            {"name": "lisinopril", "dose": "10 mg", "frequency": "daily"},
            {"name": "atorvastatin", "dose": "20 mg", "frequency": "nightly"},
        ],
        "allergies": ["penicillin, hives"],
    },
    "poly-002": {
        "patient_id": "poly-002",
        "name": "Ayşe Kaya",
        "prior_encounters": [
            {"date": "2026-01-22", "reason": "asthma exacerbation", "peak_flow": 320},
        ],
        "active_medications": [
            {"name": "albuterol", "dose": "90 mcg", "frequency": "PRN"},
            {"name": "fluticasone", "dose": "110 mcg", "frequency": "BID"},
        ],
        "allergies": [],
    },
    "er-101": {
        "patient_id": "er-101",
        "name": "John Williams",
        "prior_encounters": [
            {"date": "2024-12-03", "reason": "STEMI, PCI to LAD"},
        ],
        "active_medications": [
            {"name": "aspirin", "dose": "81 mg", "frequency": "daily"},
            {"name": "clopidogrel", "dose": "75 mg", "frequency": "daily"},
            {"name": "metoprolol", "dose": "25 mg", "frequency": "BID"},
        ],
        "allergies": [],
    },
}


class EhrLookupRequest(BaseModel):
    patient_id: str


class EhrLookupResponse(BaseModel):
    patient_id: str
    record: dict
    fetched_via: str  # always "credential-vault"; demo label


def _vault_token_configured() -> bool:
    """Whether the EHR vault is operable. Tests can force a known value
    by setting ``EHR_API_TOKEN`` in the process environment before
    importing ``server``."""
    return bool(os.environ.get("EHR_API_TOKEN"))


@app.post("/agent/vault/ehr/lookup", response_model=EhrLookupResponse)
def vault_ehr_lookup(req: EhrLookupRequest):
    """Look up a patient's EHR record through the credential vault.

    The browser calls this in response to an
    ``agent.custom_tool_use`` for ``lookup_ehr_history``. The auth token
    is read from the server process's environment, attached to the
    downstream call (simulated here by a dict read), and the result is
    returned without the token ever appearing in the response body or
    in any log line.
    """
    patient_id = req.patient_id.strip()
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not _vault_token_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "EHR_API_TOKEN is not configured server-side. Set it in "
                "backend/.env.local to enable the vault."
            ),
        )
    record = FAKE_EHR_RECORDS.get(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"patient_id not found: {patient_id}")
    # Token-free log line: demonstrates the vault pattern.
    _agent_log.info("vault: ehr lookup patient=%s", patient_id)
    return EhrLookupResponse(
        patient_id=patient_id,
        record=record,
        fetched_via="credential-vault",
    )


# ───────────────────────────────────────────────────────────────────────────
# Triage-reasoning endpoint (direct Opus 4.7 inference)
# ───────────────────────────────────────────────────────────────────────────
#
# Separate from the medsim-attending Managed Agent. That agent observes the
# whole encounter and grades it; this endpoint is a one-shot ESI triage
# classification called at ER arrival, before the agent has seen enough
# to form an opinion. Kept on Opus 4.7: the spec reserves clinical
# reasoning for the strongest model.
#
# Design notes:
#   • Pure function `run_triage_reasoning(client, request)` so unit
#     tests can mock the Anthropic client without spinning HTTP.
#   • The system prompt inlines the 5 ESI rules we actually apply. It
#     mirrors the ESI triage rules so keep them in sync
#     when either changes.
#   • Model output is constrained to JSON via explicit instruction +
#     assistant prefill; we parse defensively and raise on malformed
#     output so the caller sees a 502 rather than silent garbage.

ESI_TRIAGE_SYSTEM_PROMPT = (
    "You classify ER arrivals on a simplified 3-level ESI scale: "
    "'critical' (ESI 1-2), 'urgent' (ESI 3), or 'stable' (ESI 4-5).\n\n"

    "Rules you apply, in priority order:\n"
    "1. RED FLAGS force 'critical' regardless of current vitals: "
    "ST elevation / new LBBB / troponin elevation; sudden focal neuro "
    "deficit within the tPA window; sustained VT / VF / torsades; "
    "anaphylaxis with airway involvement; ectopic with hemodynamic "
    "compromise; intracranial hemorrhage; qSOFA ≥ 2 of (RR≥22, altered "
    "mental status, SBP≤100).\n"
    "2. 'critical' also applies if the patient needs life-saving "
    "intervention NOW (airway, hemodynamic support, time-critical "
    "reperfusion).\n"
    "3. 'urgent' for conditions with significant morbidity but no "
    "imminent threat, appendicitis, new AFib w/ RVR, moderate asthma "
    "exacerbation, chest pain without red-flag features.\n"
    "4. 'stable' only when BOTH red flags and urgent criteria are "
    "absent AND vitals are compensated AND chief complaint is low-"
    "acuity (ankle sprain, viral URI, chronic-stable presentations).\n"
    "5. If uncertain between two levels, choose the higher severity.\n\n"

    "Respond with STRICT JSON ONLY, no prose, matching:\n"
    '{"esi_level": "critical"|"urgent"|"stable", '
    '"rationale": "<one sentence citing specific vitals or red flags>", '
    '"red_flags": [<zero or more of the rule-1 phrases that apply>]}\n'
)


class VitalsSnapshot(BaseModel):
    hr: Optional[int] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    spo2: Optional[int] = None
    rr: Optional[int] = None
    temp_c: Optional[float] = None


class TriageClassifyRequest(BaseModel):
    patient_id: str
    chief_complaint: str
    vitals: VitalsSnapshot
    ecg_findings: Optional[str] = None
    notes: Optional[str] = None


class TriageClassifyResponse(BaseModel):
    patient_id: str
    esi_level: str  # 'critical' | 'urgent' | 'stable'
    rationale: str
    red_flags: list[str]
    model: str


ALLOWED_ESI_LEVELS = {"critical", "urgent", "stable"}


def _format_triage_user_message(req: TriageClassifyRequest) -> str:
    v = req.vitals
    parts = [f"Patient {req.patient_id}, {req.chief_complaint}."]
    vitals_bits: list[str] = []
    if v.hr is not None:
        vitals_bits.append(f"HR {v.hr}")
    if v.bp_systolic is not None and v.bp_diastolic is not None:
        vitals_bits.append(f"BP {v.bp_systolic}/{v.bp_diastolic}")
    if v.spo2 is not None:
        vitals_bits.append(f"SpO2 {v.spo2}")
    if v.rr is not None:
        vitals_bits.append(f"RR {v.rr}")
    if v.temp_c is not None:
        vitals_bits.append(f"temp {v.temp_c}°C")
    if vitals_bits:
        parts.append("Vitals: " + ", ".join(vitals_bits) + ".")
    if req.ecg_findings:
        parts.append(f"ECG: {req.ecg_findings}.")
    if req.notes:
        parts.append(f"Notes: {req.notes}.")
    parts.append("Classify on the simplified ESI 3-level scale.")
    return " ".join(parts)


def _extract_text_blocks(message: object) -> str:
    """Concatenate every text block in an Anthropic Messages response.
    Accepts both real SDK Message objects and dict-shaped test doubles."""
    blocks = getattr(message, "content", None)
    if blocks is None and isinstance(message, dict):
        blocks = message.get("content")
    if not blocks:
        return ""
    out: list[str] = []
    for b in blocks:
        # Real SDK: TextBlock with .text attribute and .type == 'text'.
        btype = getattr(b, "type", None)
        btext = getattr(b, "text", None)
        if btype is None and isinstance(b, dict):
            btype = b.get("type")
            btext = b.get("text")
        if btype == "text" and isinstance(btext, str):
            out.append(btext)
    return "".join(out)


def run_triage_reasoning(
    client: "Anthropic",
    req: TriageClassifyRequest,
) -> TriageClassifyResponse:
    """Invoke Opus 4.7 to classify an ER arrival. Pure function modulo
    the client handle, unit tests pass a mock client."""
    message = client.messages.create(  # type: ignore[attr-defined]
        model=TRIAGE_MODEL,
        max_tokens=TRIAGE_MAX_TOKENS,
        system=ESI_TRIAGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _format_triage_user_message(req)}],
    )
    raw = _extract_text_blocks(message).strip()
    if not raw:
        raise HTTPException(
            status_code=502,
            detail="triage model returned empty response",
        )
    # The model is instructed to return strict JSON. Strip an accidental
    # ```json fence if the model wraps it.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except ValueError as e:
        _agent_log.warning("triage: malformed JSON from model: %s", raw[:200])
        raise HTTPException(
            status_code=502,
            detail=f"triage model returned malformed JSON: {e}",
        )
    esi = str(parsed.get("esi_level", "")).strip()
    if esi not in ALLOWED_ESI_LEVELS:
        raise HTTPException(
            status_code=502,
            detail=f"triage model returned invalid esi_level: {esi!r}",
        )
    rationale = str(parsed.get("rationale", "")).strip()
    red_flags_raw = parsed.get("red_flags", [])
    red_flags = [str(x) for x in red_flags_raw] if isinstance(red_flags_raw, list) else []
    _agent_log.info(
        "triage: patient=%s esi=%s flags=%d",
        req.patient_id, esi, len(red_flags),
    )
    return TriageClassifyResponse(
        patient_id=req.patient_id,
        esi_level=esi,
        rationale=rationale,
        red_flags=red_flags,
        model=TRIAGE_MODEL,
    )


@app.post("/agent/triage/classify", response_model=TriageClassifyResponse)
async def triage_classify(req: TriageClassifyRequest):
    """ESI classification for ER arrivals. Direct inference endpoint."""
    if _anthropic_client is not None:
        return run_triage_reasoning(_anthropic_client, req)

    system = ESI_TRIAGE_SYSTEM_PROMPT
    user_msg = _format_triage_user_message(req)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    try:
        reply, _ = await llm_provider.chat(
            messages,
            tier="fast",
            effort="low",
            json_mode=True,
            max_tokens=TRIAGE_MAX_TOKENS,
        )
        raw = reply.get("content", "").strip() if isinstance(reply, dict) else str(reply).strip()
        raw = strip_think(raw)
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        esi = str(parsed.get("esi_level", "")).strip()
        if esi not in ALLOWED_ESI_LEVELS:
            raise HTTPException(
                status_code=502,
                detail=f"triage model returned invalid esi_level: {esi!r}",
            )
        rationale = str(parsed.get("rationale", "")).strip()
        red_flags_raw = parsed.get("red_flags", [])
        red_flags = [str(x) for x in red_flags_raw] if isinstance(red_flags_raw, list) else []
        return TriageClassifyResponse(
            patient_id=req.patient_id,
            esi_level=esi,
            rationale=rationale,
            red_flags=red_flags,
            model=llm_provider.FAST_MODEL,
        )
    except HTTPException:
        raise
    except Exception as e:
        _agent_log.exception("triage: failed: %s", e)
        raise HTTPException(status_code=502, detail=f"triage error: {e}")


# ───────────────────────────────────────────────────────────────────────────
# Patient persona streaming: Haiku 4.5
# ───────────────────────────────────────────────────────────────────────────
#
# The browser used to call Anthropic directly with a VITE_ANTHROPIC_API_KEY
# (dangerouslyAllowBrowser). That shipped the key in every bundle. We now
# route patient-persona streaming through the backend so the key stays
# server-side. SSE frames carry `{"text": "..."}` deltas, terminated with
# `{"done": true}`. The Haiku response is short (max 256 tokens) so we
# skip the keepalive logic the long-lived agent stream needs.

PATIENT_MODEL = "claude-haiku-4-5"
PATIENT_MAX_TOKENS = 256


class PatientChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class PatientStreamRequest(BaseModel):
    system: str
    messages: list[PatientChatMessage]


@app.post("/agent/patient/stream")
async def patient_stream(req: PatientStreamRequest):
    async def generator():
        try:
            messages = [{"role": "system", "content": req.system}]
            for m in req.messages:
                messages.append({"role": m.role, "content": m.content})
            reply, _ = await llm_provider.chat(
                messages,
                tier="fast",
                effort="low",
                max_tokens=PATIENT_MAX_TOKENS,
            )
            content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
            content = strip_think(content)
            if content:
                yield "data: " + json.dumps({"text": content}) + "\n\n"
            yield "data: " + json.dumps({"done": True}) + "\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _agent_log.exception("patient stream failed")
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.include_router(voice_router)
app.include_router(rx_audio_router)
app.include_router(llm_proxy.router)
app.include_router(nurse_router)
app.include_router(debrief_router)
app.include_router(teacher_router)
app.include_router(coach_router)
app.include_router(radiology_router)
app.include_router(ot_router)
app.include_router(osce_router)
app.include_router(exam_router)
app.include_router(observer_router)
app.include_router(viva_router)
app.include_router(logbook_router)
# Real-world delivery: public, token-guarded links the patient's phone opens
# (Murf voice prescription MP3, follow-up .ics invite).
from integrations import calendar_invite as _calendar_invite, notify as _notify  # noqa: E402
app.include_router(_notify.router)
app.include_router(_calendar_invite.router)
# Asha follow-up call: /call-api/* is public (the patient's phone), token-guarded,
# rate limited per link; the doctor's summaries route sits under /agent/.
from followup_call import router as followup_call_router  # noqa: E402
app.include_router(followup_call_router)
# Teacher assignments + the front-desk receptionist agent (emails reports to the teacher).
from assignments import router as assignments_router  # noqa: E402
from reception import router as reception_router  # noqa: E402
app.include_router(assignments_router)
app.include_router(reception_router)
# Autonomous voice agents: their tools publish UI events the page polls
# (GET /agent/agent-events/{channel}, under the /agent/ guard).
from agent_events import router as agent_events_router  # noqa: E402
app.include_router(agent_events_router)
# Observer phone call: Twilio webhooks under /rx/obs-call/* are public (signature
# checked); the doctor's call start / status routes sit under /agent/.
from observer_call import router as observer_call_router  # noqa: E402
app.include_router(observer_call_router)
# Consultation result to Telegram: summary, PDF report card and a Murf voice summary.
from results import router as results_router  # noqa: E402
app.include_router(results_router)
# Cases created from Telegram (@medsim_asha_bot /newpatient), polled by the app.
from custom_cases import router as custom_cases_router  # noqa: E402
app.include_router(custom_cases_router)


# Production: serve the built SPA (../dist) from this same service so the
# browser talks to one origin. Registered last so API routes win.
_DIST = Path(__file__).resolve().parent.parent / "dist"
if _DIST.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        target = (_DIST / full_path).resolve()
        if full_path and _DIST in target.parents and target.is_file():
            headers = {}
            if full_path.startswith("assets/"):
                headers["Cache-Control"] = "public, max-age=604800"
            return FileResponse(target, headers=headers)
        return FileResponse(_DIST / "index.html", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=BACKEND_PORT, log_level="info")
