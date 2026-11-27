"""Real-time voice: Google Gemini Live API WebSockets + Gemini TTS (RIFF WAV).

Endpoints:
  GET  /voice/config   health and active voice transport status
  POST /voice/start    prepare Gemini Live session, mint tokens
  POST /voice/stop     close live session (idempotent)
  POST /voice/speak    speak a line into active session
  POST /voice/context  update session prompt or agent context
  POST /voice/update   alias for /voice/context
  GET  /voice/metrics  realtime token/turn metrics
  GET  /voice/status   active session stats
  POST /voice/tts      Gemini TTS RIFF WAV audio
  WS   /voice/live/{channel}  WebSocket bridge to Google Gemini Live API
"""

from __future__ import annotations

import copy
import logging
import os
import secrets
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, model_validator

import gemini_live
import gemini_tts
import lang as langmod
import observer
import viva

log = logging.getLogger("medsim.voice")
router = APIRouter()
router.include_router(gemini_live.router)

AGENT_UID = 900001
AVATAR_UID = 900002
TOKEN_TTL_S = 7200 * 2
AGENT_IDLE_TIMEOUT_S = 45

DEFAULT_VOICES = {"M": "Puck", "F": "Aoede", "nurse": "Aoede"}


def _require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        raise HTTPException(status_code=503, detail=f"Voice not configured: missing {', '.join(missing)}")
    return {n: os.environ[n].strip() for n in names}


def voice_health() -> dict[str, Any]:
    key = gemini_tts._get_api_key()
    return {
        "transport": "gemini_live",
        "gemini_configured": bool(key),
        "gemini_live_configured": bool(key),
        "gemini_tts_configured": bool(key),
        "model": os.environ.get("GEMINI_LIVE_MODEL", gemini_live.DEFAULT_LIVE_MODEL),
        "tts_model": os.environ.get("GEMINI_TTS_MODEL", gemini_tts.DEFAULT_TTS_MODEL),
        "active_sessions": gemini_live.active_session_count(),
    }


@router.get("/voice/config")
def voice_config() -> dict[str, Any]:
    return voice_health()


RECEPTIONIST_CASE_ID = "receptionist"


def _voice_for(kind: str, case_id: str | None = None, lang: str | None = "en") -> str:
    """Gemini voice selection."""
    return gemini_tts.pick_voice_for_role(kind, case_id, lang)


def avatar_config() -> Optional[dict[str, str]]:
    vendor = os.environ.get("AVATAR_VENDOR", "").strip().lower()
    key = os.environ.get("AVATAR_API_KEY", "").strip()
    if not vendor or not key:
        return None
    return {
        "vendor": vendor,
        "api_key": key,
        "avatar_id": os.environ.get("AVATAR_ID", "").strip(),
        "quality": os.environ.get("AVATAR_QUALITY", "").strip() or "high",
    }


def want_avatar(req: "VoiceStartRequest") -> bool:
    if avatar_config() is None:
        return False
    if req.avatar is None:
        return req.caseId in {"teacher"}
    return req.avatar


class VoiceStartRequest(BaseModel):
    caseId: str
    systemPrompt: str
    initialLine: str
    gender: Literal["M", "F"] = "M"
    locale: Optional[str] = None
    avatar: Optional[bool] = None
    agentContext: Optional[dict[str, Any]] = None


AUTONOMOUS_CASES = ("receptionist", "teacher", viva.VIVA_CASE_ID)


class VoiceStartResponse(BaseModel):
    ok: bool = True
    appId: str = "gemini-live"
    app_id: str = "gemini-live"
    channel: str = ""
    uid: int = 0
    token: str = ""
    agentUid: int = AGENT_UID
    agent_uid: int = AGENT_UID
    agentId: str = ""
    agent_id: str = ""
    mode: str = "gemini_live"
    model: str = "gemini-3.8-live"
    wsUrl: str = ""
    ws_url: str = ""
    observerWsUrl: str = ""
    observer_ws_url: str = ""
    observerToken: str = ""
    observer_token: str = ""
    sampleRateIn: int = 16000
    sample_rate_in: int = 16000
    sampleRateOut: int = 24000
    sample_rate_out: int = 24000
    avatarUid: Optional[int] = None
    avatar_uid: Optional[int] = None
    avatarError: Optional[str] = None
    avatar_error: Optional[str] = None
    toolsEnabled: Optional[bool] = True
    tools_enabled: Optional[bool] = True

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        pairs = [
            ("appId", "app_id"),
            ("agentUid", "agent_uid"),
            ("agentId", "agent_id"),
            ("wsUrl", "ws_url"),
            ("observerWsUrl", "observer_ws_url"),
            ("observerToken", "observer_token"),
            ("sampleRateIn", "sample_rate_in"),
            ("sampleRateOut", "sample_rate_out"),
            ("avatarUid", "avatar_uid"),
            ("avatarError", "avatar_error"),
            ("toolsEnabled", "tools_enabled"),
        ]
        for camel, snake in pairs:
            if camel in d and snake not in d:
                d[snake] = d[camel]
            elif snake in d and camel not in d:
                d[camel] = d[snake]
        return d


async def start_agent_session(
    req: VoiceStartRequest,
    *,
    voice_kind: Optional[str] = None,
    channel_prefix: str = "medsim",
    idle_timeout: int = AGENT_IDLE_TIMEOUT_S,
    translate_greeting: bool = True,
    llm_token_meta: Optional[dict[str, Any]] = None,
    agent_kind: Optional[str] = None,
    agent_ctx: Optional[dict[str, Any]] = None,
) -> VoiceStartResponse:
    """Prepare a Gemini Live session, mint tokens, and return connection details."""
    key = gemini_tts._get_api_key()
    if not key:
        raise HTTPException(status_code=503, detail="Voice not configured: missing GEMINI_API_KEY")

    safe_case = "".join(ch for ch in req.caseId if ch.isalnum() or ch in "-_")[:24] or "case"
    channel = f"{channel_prefix}-{safe_case}-{secrets.token_hex(4)}"

    lang = langmod.norm_lang(req.locale)
    voice_name = gemini_tts.pick_voice_for_role(voice_kind or req.gender, req.caseId, lang)

    user_uid = secrets.randbelow(900000) + 100000

    # Translate opening greeting if required
    initial_line = req.initialLine
    if translate_greeting and lang != "en":
        try:
            initial_line = await langmod.translate_line(initial_line, req.locale)
        except Exception:
            pass

    # Register prepared live session
    live_token = gemini_live.register_prepared_session(
        channel=channel,
        case_id=req.caseId,
        system_prompt=req.systemPrompt,
        initial_line=initial_line,
        voice=voice_name,
        locale=lang,
        agent_kind=agent_kind or (req.caseId if req.caseId in AUTONOMOUS_CASES else None),
        agent_ctx=agent_ctx or req.agentContext,
        llm_token_meta=llm_token_meta,
    )

    # Register with observer hub
    observer.register_session(channel, user_uid, req.caseId, AGENT_UID)
    obs_token = observer.mint_token(channel, user_uid)

    ws_url = f"/voice/live/{channel}?token={live_token}"
    obs_ws_url = f"/voice/observer/ws?channel={channel}&uid={user_uid}&token={obs_token}"
    model = os.environ.get("GEMINI_LIVE_MODEL", gemini_live.DEFAULT_LIVE_MODEL)

    has_avatar = want_avatar(req)

    return VoiceStartResponse(
        ok=True,
        appId="gemini-live",
        app_id="gemini-live",
        channel=channel,
        uid=user_uid,
        token=live_token,
        agentUid=AGENT_UID,
        agent_uid=AGENT_UID,
        agentId=channel,
        agent_id=channel,
        mode="gemini_live",
        model=model,
        wsUrl=ws_url,
        ws_url=ws_url,
        observerWsUrl=obs_ws_url,
        observer_ws_url=obs_ws_url,
        observerToken=obs_token,
        observer_token=obs_token,
        sampleRateIn=16000,
        sample_rate_in=16000,
        sampleRateOut=24000,
        sample_rate_out=24000,
        avatarUid=AVATAR_UID if has_avatar else None,
        avatar_uid=AVATAR_UID if has_avatar else None,
        toolsEnabled=True,
        tools_enabled=True,
    )


@router.post("/voice/start", response_model=VoiceStartResponse, response_model_exclude_none=True)
async def voice_start(req: VoiceStartRequest):
    if req.caseId in AUTONOMOUS_CASES:
        import agent_runtime

        try:
            ctx = agent_runtime.clean_ctx(req.caseId, req.agentContext or {})
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return await start_agent_session(req, agent_kind=req.caseId, agent_ctx=ctx)
    return await start_agent_session(req)


async def leave_agent(agent_id: str) -> bool:
    """Stop an agent session. Idempotent."""
    return await gemini_live.stop_session(agent_id)


async def speak_line(agent_id: str, text: str, priority: str = "INTERRUPT", interruptable: bool = True) -> None:
    """Make the agent speak a line."""
    await gemini_live.speak_in_session(agent_id, text, priority, interruptable)


class AgentRef(BaseModel):
    agentId: str


@router.post("/voice/stop")
async def voice_stop(request: Request):
    try:
        req = AgentRef.model_validate_json(await request.body())
    except Exception:
        raise HTTPException(status_code=422, detail="body must be JSON {agentId}")
    stopped = await gemini_live.stop_session(req.agentId)
    return {"ok": True, "stopped": stopped}


class VoiceSpeakRequest(BaseModel):
    agentId: str
    text: str
    priority: Literal["INTERRUPT", "APPEND", "IGNORE"] = "INTERRUPT"
    interruptable: bool = True


@router.post("/voice/speak")
async def voice_speak(req: VoiceSpeakRequest):
    await gemini_live.speak_in_session(req.agentId, req.text, req.priority, req.interruptable)
    return {"ok": True}


class VoiceContextRequest(BaseModel):
    agentId: str
    systemPrompt: Optional[str] = None
    agentContext: Optional[dict[str, Any]] = None


def update_agent_ctx(agent_id: str, patch: dict[str, Any]) -> bool:
    """Merge a validated ctx patch into session context."""
    import asyncio
    asyncio.create_task(gemini_live.update_session_context(agent_id, agent_ctx=patch))
    return True


@router.post("/voice/context")
@router.post("/voice/update")
async def voice_context(req: VoiceContextRequest):
    updated = await gemini_live.update_session_context(
        req.agentId,
        system_prompt=req.systemPrompt,
        agent_ctx=req.agentContext,
    )
    return {"ok": True, "ctxUpdated": updated}


@router.get("/voice/metrics")
async def voice_metrics(agentId: Optional[str] = None):
    if agentId:
        return gemini_live.get_session_metrics(agentId)
    return {"active_sessions": gemini_live.active_session_count()}


@router.get("/voice/status")
async def voice_status():
    return {
        "ok": True,
        "mode": "gemini_live",
        "active_sessions": gemini_live.active_session_count(),
    }


class VoiceTtsRequest(BaseModel):
    text: str
    voice: Literal["nurse", "M", "F"] = "nurse"
    locale: Optional[str] = None
    caseId: Optional[str] = None
    gender: Optional[Literal["M", "F"]] = None


def tts_voice_and_locale(req: VoiceTtsRequest) -> tuple[str, str]:
    lang = langmod.norm_lang(req.locale)
    role = req.gender or (req.voice if req.voice in ("M", "F") else "nurse")
    voice_name = gemini_tts.pick_voice_for_role(role, req.caseId, lang)
    locale = langmod.locale_for(lang)
    return voice_name, locale


@router.post("/voice/tts")
async def voice_tts(req: VoiceTtsRequest):
    voice_name, _locale = tts_voice_and_locale(req)
    try:
        wav_bytes, _mime = await gemini_tts.synthesize_speech(req.text, voice=voice_name)
    except Exception as e:
        log.warning("Voice TTS error: %s", e)
        raise HTTPException(status_code=502, detail=f"Gemini TTS failed: {type(e).__name__}: {e}")
    return Response(content=wav_bytes, media_type="audio/wav", headers={"Cache-Control": "no-store"})
