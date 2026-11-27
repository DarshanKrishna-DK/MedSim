"""Google Gemini Live API WebSocket Bridge for MedSim.

Replaces Agora RTC/RTM and ConvoAI with direct bidirectional streaming over WebSockets:
- Client -> Server: 16kHz mono 16-bit PCM (send_realtime_input)
- Server -> Client: 24kHz mono 16-bit PCM from Gemini Live
- Realtime transcriptions, barge-in interruptions, tools, and observer relays.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

import gemini_tts
import observer

log = logging.getLogger("medsim.gemini_live")
router = APIRouter()

DEFAULT_LIVE_MODEL = "gemini-3.8-live"
FALLBACK_LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"

_TOKEN_SECRET = secrets.token_bytes(32)


def _sign(raw: str) -> str:
    return hmac.new(_TOKEN_SECRET, raw.encode(), hashlib.sha256).hexdigest()[:16]


def mint_live_token(channel: str, ttl: int = 7200) -> str:
    expiry = int(time.time()) + ttl
    raw = f"{channel}:{expiry}"
    sig = _sign(raw)
    return f"{expiry}.{sig}"


def verify_live_token(token: str, channel: str) -> bool:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return False
        expiry = int(parts[0])
        sig = parts[1]
        if time.time() > expiry:
            return False
        raw = f"{channel}:{expiry}"
        return hmac.compare_digest(sig, _sign(raw))
    except Exception:
        return False


@dataclass
class PreparedSession:
    channel: str
    case_id: str
    system_prompt: str
    initial_line: str
    voice: str
    locale: str = "en"
    agent_kind: Optional[str] = None
    agent_ctx: Optional[dict[str, Any]] = None
    llm_token_meta: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    active_session: Optional["ActiveLiveSession"] = None


_prepared_sessions: dict[str, PreparedSession] = {}
_sessions_lock = asyncio.Lock()


def register_prepared_session(
    channel: str,
    case_id: str,
    system_prompt: str,
    initial_line: str,
    voice: str,
    locale: str = "en",
    agent_kind: Optional[str] = None,
    agent_ctx: Optional[dict[str, Any]] = None,
    llm_token_meta: Optional[dict[str, Any]] = None,
) -> str:
    """Register a prepared session and return the live auth token."""
    token = mint_live_token(channel)
    prep = PreparedSession(
        channel=channel,
        case_id=case_id,
        system_prompt=system_prompt,
        initial_line=initial_line,
        voice=voice,
        locale=locale,
        agent_kind=agent_kind,
        agent_ctx=agent_ctx,
        llm_token_meta=llm_token_meta,
    )
    _prepared_sessions[channel] = prep
    return token


def get_prepared_session(channel: str) -> Optional[PreparedSession]:
    return _prepared_sessions.get(channel)


def _get_api_key() -> str:
    return gemini_tts._get_api_key()


def _build_tools(agent_kind: Optional[str] = None) -> list[types.Tool]:
    """Build function declarations for the Gemini Live session."""
    fds: list[types.FunctionDeclaration] = [
        types.FunctionDeclaration(
            name="leave_agent",
            description="End the consultation or call gracefully.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "reason": types.Schema(
                        type=types.Type.STRING,
                        description="Reason for ending the consultation",
                    )
                },
            ),
        ),
        types.FunctionDeclaration(
            name="speak_line",
            description="Speak a direct statement or instruction to the student.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text": types.Schema(
                        type=types.Type.STRING,
                        description="Direct speech content",
                    )
                },
                required=["text"],
            ),
        ),
        types.FunctionDeclaration(
            name="update_context",
            description="Record notes, vitals, or clinical progress in the chart.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "notes": types.Schema(
                        type=types.Type.STRING,
                        description="Summary of update or findings",
                    )
                },
                required=["notes"],
            ),
        ),
    ]

    # Additional autonomous tools if applicable
    raw_tools: list[dict[str, Any]] = []
    try:
        if agent_kind == "receptionist":
            import agent_toolsets
            raw_tools = agent_toolsets.RECEPTION_TOOLS
        elif agent_kind == "teacher":
            import agent_toolsets
            raw_tools = agent_toolsets.TEACHER_TOOLS
        elif agent_kind in ("viva-examiner", "viva"):
            import agent_toolsets
            raw_tools = agent_toolsets.VIVA_TOOLS
        elif agent_kind == "followup":
            import followup_tools
            raw_tools = followup_tools.TOOLS
    except Exception as e:
        log.warning("Error importing toolset for %s: %s", agent_kind, e)

    core_names = {"leave_agent", "speak_line", "update_context"}
    for rt in raw_tools:
        fn = rt.get("function", {})
        name = fn.get("name")
        if not name or name in core_names:
            continue
        try:
            fd = types.FunctionDeclaration.model_validate({
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
            fds.append(fd)
        except Exception as e:
            log.warning("Could not convert tool %s: %s", name, e)

    return [types.Tool(function_declarations=fds)]


def build_live_connect_config(
    model: str = DEFAULT_LIVE_MODEL,
    voice_name: Optional[str] = "Aoede",
    system_prompt: Optional[str] = None,
    agent_kind: Optional[str] = None,
) -> types.LiveConnectConfig:
    """Construct LiveConnectConfig adhering strictly to Gemini Live model rules.
    Omits thinking_config, enable_affective_dialog, and proactivity on gemini-3.8-live.
    """
    cfg_kwargs: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name or "Aoede"
                )
            )
        ),
        "input_audio_transcription": types.AudioTranscriptionConfig(),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        "context_window_compression": types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
        "tools": _build_tools(agent_kind),
    }
    if system_prompt:
        cfg_kwargs["system_instruction"] = types.Content(
            parts=[types.Part.from_text(text=system_prompt)]
        )

    # Never include thinking_config, affective_dialog or proactivity on gemini-3.8-live
    if model != "gemini-3.8-live" and ("extended-thinking" in model or "3.1-flash-live-preview" in model):
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")

    return types.LiveConnectConfig(**cfg_kwargs)


class ActiveLiveSession:
    """Manages one live session bridging Browser WebSocket <-> Gemini Live API."""

    def __init__(self, prep: PreparedSession, websocket: WebSocket):
        self.prep = prep
        self.websocket = websocket
        self.channel = prep.channel
        self.closed = False
        self.turn_id = 1
        self.gemini_session: Any = None
        self.metrics: dict[str, Any] = {
            "prompt_tokens": 0,
            "candidates_tokens": 0,
            "total_tokens": 0,
        }
        self.current_state = "idle"

    async def _send_state(self, state: str) -> None:
        if self.current_state != state and not self.closed:
            self.current_state = state
            try:
                await self.websocket.send_text(json.dumps({"type": "state", "state": state}))
            except Exception:
                pass

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute client/agent tools."""
        log.info("Executing tool %s on channel %s with args: %s", name, self.channel, args)
        if name == "leave_agent":
            asyncio.create_task(self.close(reason="Agent left"))
            return {"status": "success", "action": "leave", "reason": args.get("reason", "Call ended")}

        if name == "speak_line":
            return {"status": "success", "spoken": args.get("text", "")}

        if name == "update_context":
            return {"status": "success", "updated": True}

        # Autonomous toolsets
        kind = self.prep.agent_kind
        tctx = dict(self.prep.agent_ctx or {})
        try:
            if kind == "receptionist":
                import agent_toolsets
                return agent_toolsets.reception_exec(name, args, tctx)
            elif kind == "teacher":
                import agent_toolsets
                return agent_toolsets.teacher_exec(name, args, tctx)
            elif kind in ("viva-examiner", "viva"):
                import agent_toolsets
                return agent_toolsets.viva_exec(name, args, tctx)
            elif kind == "followup":
                import followup_tools
                meta = dict(self.prep.llm_token_meta or {})
                return followup_tools.run_tool(meta, name, args, [])
        except Exception as e:
            log.warning("Tool %s execution failed: %s", name, e)
            return {"error": str(e)}

        return {"status": "unknown_tool", "name": name}

    async def speak(self, text: str, priority: str = "INTERRUPT", interruptable: bool = True) -> None:
        """Inject speech text into the live session."""
        if self.gemini_session and not self.closed:
            try:
                await self.gemini_session.send_realtime_input(text=text)
            except Exception as e:
                log.warning("Failed injecting speak text into %s: %s", self.channel, e)

    async def update_context(self, system_prompt: Optional[str] = None, agent_ctx: Optional[dict] = None) -> None:
        """Update system prompt or autonomous context."""
        if system_prompt:
            self.prep.system_prompt = system_prompt
        if agent_ctx:
            merged = {**(self.prep.agent_ctx or {}), **agent_ctx}
            self.prep.agent_ctx = merged
        # Best effort prompt injection
        if system_prompt and self.gemini_session and not self.closed:
            try:
                await self.gemini_session.send_realtime_input(
                    text=f"[SYSTEM UPDATE]: {system_prompt}"
                )
            except Exception as e:
                log.warning("Failed updating context on %s: %s", self.channel, e)

    async def run(self) -> None:
        """Run the bidirectional bridging loop."""
        api_key = _get_api_key()
        if not api_key:
            await self.websocket.send_text(
                json.dumps({"object": "message.error", "message": "GEMINI_API_KEY is not configured"})
            )
            await self.websocket.close(code=1011, reason="Missing API key")
            return

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

        primary_model = os.environ.get("GEMINI_LIVE_MODEL", DEFAULT_LIVE_MODEL)
        models_to_try = [primary_model]
        if primary_model != FALLBACK_LIVE_MODEL:
            models_to_try.append(FALLBACK_LIVE_MODEL)

        last_err: Exception | None = None
        connected = False

        for cur_model in models_to_try:
            config = build_live_connect_config(
                model=cur_model,
                voice_name=self.prep.voice or "Aoede",
                system_prompt=self.prep.system_prompt,
                agent_kind=self.prep.agent_kind,
            )

            try:
                log.info("Connecting channel %s to Gemini Live model %s...", self.channel, cur_model)
                async with client.aio.live.connect(model=cur_model, config=config) as session:
                    self.gemini_session = session
                    connected = True
                    log.info("Channel %s successfully connected to %s", self.channel, cur_model)

                    # Send opening greeting if provided
                    if self.prep.initial_line:
                        try:
                            await session.send_realtime_input(
                                text=f"Say your opening line aloud to start: \"{self.prep.initial_line}\""
                            )
                        except Exception as e:
                            log.warning("Failed sending opening greeting: %s", e)

                    await self._send_state("listening")

                    # Run reader & writer tasks concurrently
                    task_client_to_gemini = asyncio.create_task(self._loop_client_to_gemini())
                    task_gemini_to_client = asyncio.create_task(self._loop_gemini_to_client())

                    done, pending = await asyncio.wait(
                        [task_client_to_gemini, task_gemini_to_client],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()
                    break

            except Exception as e:
                log.warning("Gemini Live connection with %s failed: %s", cur_model, e)
                last_err = e
                continue

        if not connected:
            log.error("All Gemini Live models failed for channel %s: %s", self.channel, last_err)
            try:
                await self.websocket.send_text(
                    json.dumps({
                        "object": "message.error",
                        "message": f"Could not connect to voice service: {last_err}",
                    })
                )
                await self.websocket.close(code=1011, reason="Gemini Live failed")
            except Exception:
                pass

    async def _loop_client_to_gemini(self) -> None:
        """Read from Browser WebSocket and send realtime inputs to Gemini Live."""
        try:
            while not self.closed:
                msg = await self.websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    pcm_data = msg["bytes"]
                    # Send 16kHz mono PCM
                    await self.gemini_session.send_realtime_input(
                        audio=types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
                    )
                    # Broadcast to observer hub (student 16k PCM = prefix 0x01)
                    await observer.broadcast_student_audio(self.channel, pcm_data)

                elif "text" in msg and msg["text"]:
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        continue

                    mtype = payload.get("type") or payload.get("object")
                    if mtype == "ping":
                        await self.websocket.send_text(json.dumps({"type": "pong"}))
                    elif mtype == "audio_end":
                        await self.gemini_session.send_realtime_input(audio_stream_end=True)
                    elif mtype in ("text", "speak") and payload.get("text"):
                        await self.gemini_session.send_realtime_input(text=payload["text"])

        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("Client->Gemini loop error on %s: %s", self.channel, e)
        finally:
            self.closed = True

    async def _loop_gemini_to_client(self) -> None:
        """Receive messages from Gemini Live and stream to Browser WebSocket + Observer."""
        try:
            while not self.closed:
                async for msg in self.gemini_session.receive():
                    if self.closed:
                        break

                    # Handle tool calls from Gemini
                    if msg.tool_call and msg.tool_call.function_calls:
                        responses: list[types.FunctionResponse] = []
                        for fc in msg.tool_call.function_calls:
                            res = await self._execute_tool(fc.name, fc.args or {})
                            responses.append(
                                types.FunctionResponse(name=fc.name, id=fc.id, response={"result": res})
                            )
                        if responses and not self.closed:
                            await self.gemini_session.send_tool_response(function_responses=responses)

                    # Handle usage metadata (Live API uses response_token_count or candidates_token_count)
                    if msg.usage_metadata:
                        um = msg.usage_metadata
                        prompt_tok = getattr(um, "prompt_token_count", 0) or 0
                        cand_tok = (
                            getattr(um, "candidates_token_count", None)
                            or getattr(um, "response_token_count", None)
                            or 0
                        )
                        total_tok = getattr(um, "total_token_count", None) or (prompt_tok + cand_tok)
                        self.metrics["prompt_tokens"] = prompt_tok
                        self.metrics["candidates_tokens"] = cand_tok
                        self.metrics["total_tokens"] = total_tok
                        metrics_env = {"object": "message.metrics", **self.metrics}
                        await self.websocket.send_text(json.dumps(metrics_env))
                        await observer.broadcast_event(self.channel, metrics_env)

                    # Handle server content (audio PCM, transcripts, barge-in)
                    if msg.server_content:
                        sc = msg.server_content

                        if sc.interrupted:
                            int_env = {"object": "message.interrupt", "turn_id": self.turn_id}
                            await self.websocket.send_text(json.dumps(int_env))
                            await observer.broadcast_event(self.channel, int_env)
                            await self._send_state("listening")

                        if sc.model_turn:
                            await self._send_state("speaking")
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    pcm_chunk = part.inline_data.data
                                    # Send raw 24kHz Int16LE PCM binary frame
                                    await self.websocket.send_bytes(pcm_chunk)
                                    # Broadcast to observer hub (agent 24k PCM = prefix 0x02)
                                    await observer.broadcast_agent_audio(self.channel, pcm_chunk)

                        if sc.input_transcription and sc.input_transcription.text:
                            tx_env = {
                                "object": "user.transcription",
                                "text": sc.input_transcription.text,
                                "final": bool(sc.turn_complete),
                                "turn_id": self.turn_id,
                                "turn_status": 1,
                            }
                            await self.websocket.send_text(json.dumps(tx_env))
                            await observer.broadcast_transcript(self.channel, tx_env)

                        if sc.output_transcription and sc.output_transcription.text:
                            tx_env = {
                                "object": "assistant.transcription",
                                "text": sc.output_transcription.text,
                                "final": bool(sc.turn_complete),
                                "turn_id": self.turn_id,
                                "turn_status": 1,
                            }
                            await self.websocket.send_text(json.dumps(tx_env))
                            await observer.broadcast_transcript(self.channel, tx_env)

                        if sc.turn_complete:
                            self.turn_id += 1
                            await self._send_state("listening")
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("Gemini->Client loop error on %s: %s", self.channel, e)
        finally:
            self.closed = True

    async def close(self, reason: str = "Session closed") -> None:
        """Close the active live session."""
        if not self.closed:
            self.closed = True
            try:
                await self.websocket.send_text(json.dumps({"type": "closed", "reason": reason}))
                await self.websocket.close(code=1000, reason=reason)
            except Exception:
                pass


# ------------------------------------------------------------------ session management


_active_sessions: dict[str, ActiveLiveSession] = {}


def active_session_count() -> int:
    return len(_active_sessions)


def get_active_session(channel: str) -> Optional[ActiveLiveSession]:
    return _active_sessions.get(channel)


async def stop_session(channel_or_agent_id: str) -> bool:
    """Stop an active session idempotently."""
    sess = _active_sessions.pop(channel_or_agent_id, None)
    if sess:
        await sess.close(reason="Stopped by API call")
        return True
    return False


async def speak_in_session(
    channel_or_agent_id: str,
    text: str,
    priority: str = "INTERRUPT",
    interruptable: bool = True,
) -> bool:
    """Inject speech into an active session."""
    sess = _active_sessions.get(channel_or_agent_id)
    if sess:
        await sess.speak(text, priority=priority, interruptable=interruptable)
        return True
    return False


async def update_session_context(
    channel_or_agent_id: str,
    system_prompt: Optional[str] = None,
    agent_ctx: Optional[dict[str, Any]] = None,
) -> bool:
    """Update context on an active or prepared session."""
    sess = _active_sessions.get(channel_or_agent_id)
    if sess:
        await sess.update_context(system_prompt=system_prompt, agent_ctx=agent_ctx)
        return True
    prep = _prepared_sessions.get(channel_or_agent_id)
    if prep:
        if system_prompt:
            prep.system_prompt = system_prompt
        if agent_ctx:
            prep.agent_ctx = {**(prep.agent_ctx or {}), **agent_ctx}
        return True
    return False


def get_session_metrics(channel_or_agent_id: str) -> dict[str, Any]:
    sess = _active_sessions.get(channel_or_agent_id)
    if sess:
        return sess.metrics
    return {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}


# ------------------------------------------------------------------ websocket endpoint


@router.websocket("/voice/live/{channel}")
async def live_websocket(websocket: WebSocket, channel: str, token: str = Query(...)):
    """FastAPI WebSocket endpoint for Gemini Live consultation audio."""
    if not verify_live_token(token, channel):
        # Also allow if channel matches prepared session
        prep = _prepared_sessions.get(channel)
        if prep is None:
            await websocket.close(code=4403, reason="Forbidden: invalid live token")
            return

    prep = _prepared_sessions.get(channel)
    if not prep:
        # Create minimal fallback prepared session if missing
        prep = PreparedSession(
            channel=channel,
            case_id="default",
            system_prompt="You are a clinical training simulation patient.",
            initial_line="Hello doctor.",
            voice="Aoede",
        )
        _prepared_sessions[channel] = prep

    await websocket.accept()
    log.info("Accepted Gemini Live WebSocket connection on channel %s", channel)

    session = ActiveLiveSession(prep, websocket)
    _active_sessions[channel] = session
    prep.active_session = session

    try:
        await session.run()
    finally:
        _active_sessions.pop(channel, None)
        log.info("Gemini Live WebSocket finished for channel %s", channel)
