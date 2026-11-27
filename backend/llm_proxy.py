"""Patient-LLM proxy for Agora Conversational AI (custom LLM).

Agora's engine calls an OpenAI-compatible endpoint with `Authorization: Bearer <llm.api_key>`.
Background noise makes it send user turns with empty content, which Sarvam rejects
with HTTP 400. This proxy sanitizes the messages, injects the real provider key
server-side and streams the upstream SSE bytes back unchanged.

Agora docs (custom LLM): endpoint must accept OpenAI chat-completions requests
(`stream` defaults to true) and answer with `text/event-stream` chunks ending in
`data: [DONE]`. `llm.vendor` accepts `custom` per the /join reference.

Also here: per-agent bearer tokens and the optional cloudflared quick tunnel
that gives Agora's cloud an HTTPS URL to reach this backend.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import lang as langmod
from llm_provider import configured_providers

log = logging.getLogger("medsim.llmproxy")
router = APIRouter()

PLACEHOLDER = "(inaudible, background noise)"
TOKEN_TTL_S = 4 * 3600
PROXY_PATH = "/llm/patient/chat/completions"

_ALLOWED_FIELDS = (
    "messages", "stream", "max_tokens", "max_completion_tokens", "temperature", "top_p",
    "tools", "tool_choice", "stop", "response_format", "frequency_penalty", "presence_penalty",
)

# ---------------------------------------------------------------- tokens

_tokens: dict[str, float] = {}
# token -> what the agent is, e.g. {"kind": "followup", "callToken": ..., "callId": ...}.
# Tokens without meta are patient sessions (plain pass-through).
_token_meta: dict[str, dict[str, Any]] = {}
_tokens_lock = threading.Lock()


def _token_secret() -> bytes:
    raw = (os.environ.get("MEDSIM_PROXY_SECRET") or os.environ.get("AGORA_APP_CERTIFICATE") or "").strip()
    return raw.encode() or _FALLBACK_SECRET


_FALLBACK_SECRET = secrets.token_bytes(32)


def _sign(body: str) -> str:
    import base64
    import hashlib

    mac = hmac.new(_token_secret(), body.encode(), hashlib.sha256).digest()[:18]
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def _signed_ok(candidate: str, now: float) -> bool:
    """Stateless check, so a token minted before a redeploy or on another
    instance still works (Agora keeps calling with it for the whole session)."""
    try:
        exp_s, nonce, sig = candidate.split(".", 2)
        return float(exp_s) >= now and hmac.compare_digest(sig, _sign(f"{exp_s}.{nonce}"))
    except (ValueError, TypeError):
        return False


def mint_token(meta: Optional[dict[str, Any]] = None) -> str:
    now = time.time()
    body = f"{int(now + TOKEN_TTL_S)}.{secrets.token_urlsafe(16)}"
    tok = f"{body}.{_sign(body)}"
    with _tokens_lock:
        for t in [t for t, exp in _tokens.items() if exp < now]:
            del _tokens[t]
            _token_meta.pop(t, None)
        _tokens[tok] = now + TOKEN_TTL_S
        if meta:
            _token_meta[tok] = dict(meta)
    return tok


def token_meta(candidate: str) -> Optional[dict[str, Any]]:
    """Meta registered for a valid token, else None (patient session or unknown)."""
    if not valid_token(candidate):
        return None
    with _tokens_lock:
        meta = _token_meta.get(candidate)
    return dict(meta) if meta else None


# Agora agent id -> its LLM bearer token, so /voice/context can update the ctx
# a tool-enabled agent's tools read (agent_runtime).
_agent_tokens: dict[str, str] = {}


def bind_agent(agent_id: str, token: str) -> None:
    if not agent_id or not token:
        return
    with _tokens_lock:
        for a in [a for a, t in _agent_tokens.items() if t not in _tokens]:
            del _agent_tokens[a]
        _agent_tokens[agent_id] = token


def agent_token(agent_id: str) -> Optional[str]:
    with _tokens_lock:
        tok = _agent_tokens.get(agent_id)
        return tok if tok and tok in _tokens else None


def update_token_ctx(agent_id: str, ctx: dict[str, Any]) -> bool:
    """Replace meta["ctx"] for the agent's token. False when the agent has no tagged token."""
    with _tokens_lock:
        tok = _agent_tokens.get(agent_id)
        meta = _token_meta.get(tok or "")
        if not tok or tok not in _tokens or meta is None or "ctx" not in meta:
            return False
        meta["ctx"] = dict(ctx)
        return True


def valid_token(candidate: str) -> bool:
    if not candidate:
        return False
    now = time.time()
    ok = False
    with _tokens_lock:
        for t, exp in list(_tokens.items()):
            # compare against every token (no early exit) to stay constant-time-ish
            if hmac.compare_digest(t.encode(), candidate.encode()) and exp >= now:
                ok = True
    return ok or _signed_ok(candidate, now)


# ---------------------------------------------------------------- sanitizing


def _text_of(content: Any) -> tuple[str, bool]:
    """Return (joined text, has_non_text_parts)."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        texts: list[str] = []
        other = False
        for p in content:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict) and p.get("type") in (None, "text") and isinstance(p.get("text"), str):
                texts.append(p["text"])
            else:
                other = True
        return " ".join(t for t in texts if t.strip()), other
    return str(content), False


def sanitize_messages(messages: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool", "developer"):
            continue
        text, _other = _text_of(m.get("content"))
        text = text.strip()
        if role == "user":
            out.append({**{k: v for k, v in m.items() if k != "content"}, "content": text or PLACEHOLDER})
        elif role == "assistant":
            if not text and not m.get("tool_calls"):
                continue  # Sarvam rejects empty assistant content
            msg = {k: v for k, v in m.items() if k != "content"}
            msg["content"] = text if text else None
            out.append(msg)
        elif role == "tool":
            out.append({**{k: v for k, v in m.items() if k != "content"}, "content": text or "(no result)"})
        else:  # system / developer
            if text:
                out.append({**{k: v for k, v in m.items() if k != "content"}, "content": text})
    # merge adjacent plain user messages (left behind when an empty assistant turn was dropped)
    merged: list[dict[str, Any]] = []
    for m in out:
        if merged and m["role"] == "user" and merged[-1]["role"] == "user":
            prev = merged[-1]["content"]
            cur = m["content"]
            if prev == PLACEHOLDER:
                merged[-1]["content"] = cur
            elif cur != PLACEHOLDER:
                merged[-1]["content"] = f"{prev}\n{cur}"
            continue
        merged.append(m)
    return merged


def build_upstream_body(payload: dict[str, Any], cfg: dict[str, str]) -> dict[str, Any]:
    body = {k: payload[k] for k in _ALLOWED_FIELDS if k in payload and payload[k] is not None}
    body["messages"] = sanitize_messages(payload.get("messages"))
    body["model"] = cfg["model"]  # never trust the client's model
    if "reasoning_effort" in payload:
        from llm_provider import clamp_effort
        body["reasoning_effort"] = clamp_effort(payload["reasoning_effort"], default="low")
    elif cfg.get("provider") == "gemini" or "generativelanguage" in cfg.get("url", ""):
        from llm_provider import clamp_effort
        body["reasoning_effort"] = clamp_effort(cfg.get("reasoning_effort"), default="low")
    return body


class PatientChatRequest(BaseModel):
    system: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 400
    tier: str = "fast"
    effort: str = "low"


@router.post("/agent/patient/chat")
async def patient_chat(req: PatientChatRequest):
    """Direct route for patient chat turns via Google Gemini."""
    from llm_provider import chat, clamp_effort

    msgs: list[dict[str, Any]] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.extend(sanitize_messages(req.messages))
    effort = clamp_effort(req.effort, default="low")
    msg, provider = await chat(
        msgs,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        tier=req.tier if req.tier in ("fast", "reasoning") else "fast",
        effort=effort,
    )
    return {"reply": msg.get("content") or "", "role": "assistant", "provider": provider}



# ---------------------------------------------------------------- nurse-addressed turns

# The doctor talking to Nurse Asha ("Nurse, order a CBC", "Asha, how am I doing?")
# is handled by the nurse agent in the browser. The patient must stay silent, so
# those turns get an empty reply here instead of going to the LLM.
_NURSE_ADDRESS = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|and|right|alright|hello|hi|hey|accha|achha)\W+){0,2}"
    r"(?:nurse|sister|staff nurse|a{1,2}sha|नर्स|आशा|सिस्टर|दीदी|ನರ್ಸ್|ಆಶಾ|ಸಿಸ್ಟರ್|ನರ್ಸ|நர்ஸ்|ஆஷா|சிஸ்டர்|నర్స్|ఆశా|సిస్టర్|নার্স|আশা|সিস্টার)(?![a-z])",
    re.IGNORECASE,
)
# Same lead-in, but only the unambiguous titles (used when the patient is named Asha).
_NURSE_TITLE = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|and|right|alright|hello|hi|hey|accha|achha)\W+){0,2}"
    r"(?:nurse|sister|staff nurse|नर्स|सिस्टर|ನರ್ಸ್|ಸಿಸ್ಟರ್|நர்ஸ்|నర్స్|নার্স)(?![a-z])",
    re.IGNORECASE,
)
_NURSE_ANYWHERE = re.compile(r"\b(?:nurse a{1,2}sha|sister a{1,2}sha|how am i doing,? a{1,2}sha)\b", re.IGNORECASE)
# How src/voice/patientPersona.ts names the patient: "You are Asha, a 34-year-old ...",
# "You are a human patient named Asha, ..." or "You are the mother of Asha, a 6-year-old ...".
_PATIENT_IS_ASHA = re.compile(
    r"\byou are (?:a human patient named |the \w+ of )?a{1,2}sha\b|\byour name is a{1,2}sha\b", re.IGNORECASE)


# "Invite Darshan to observe", "call Darshan to watch", "ask Darshan to join": the
# doctor asking Asha to phone an observer (src/observer/contacts.ts parses the same
# lines). A bare "invite <name>" counts on its own; "call/ask <name>" needs the
# "to observe/watch/join" tail, so "call me if it gets worse" still reaches the patient.
_OBS_NAME = (
    r"(?!(?:you|your|me|us|him|her|them|the|a|an|my|our|someone|somebody|back)\b)[a-z][a-z'.-]*"
    r"(?:\s+(?!to\b|please\b|now\b)[a-z][a-z'.-]*)?"
)
_OBSERVER_CMD = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|and|right|alright|hello|hi|hey|accha|achha|please|can you|could you|would you)\W+){0,3}"
    r"(?:(?:nurse|sister|a{1,2}sha)\W+)?(?:(?:please|can you|could you|would you)\W+){0,2}"
    r"(?:(?:invite|add|bring|loop in)\s+" + _OBS_NAME + r"(?:\s+(?:as|to|for|into|in|please|now|too)\b.*)?\W*$"
    r"|(?:ask|call|ring|phone|get|have)\s+" + _OBS_NAME
    + r"\b.*\b(?:observ\w*|watch\w*|join\w*|sit in|listen\w*|session|consultation)\b)",
    re.IGNORECASE,
)


def addressed_to_observer_call(text: str) -> bool:
    return bool(_OBSERVER_CMD.search((text or "").strip()))


def _patient_named_asha(messages: list[dict[str, Any]]) -> bool:
    first = next((m for m in messages if m.get("role") in ("system", "developer")), None)
    return bool(first and _PATIENT_IS_ASHA.search(str(first.get("content") or "")))


def addressed_to_nurse(messages: list[dict[str, Any]]) -> bool:
    """True when the newest message is the doctor speaking to the nurse."""
    if not messages or messages[-1].get("role") != "user":
        return False
    text = str(messages[-1].get("content") or "").strip()
    if not text:
        return False
    if _NURSE_ANYWHERE.search(text) or addressed_to_observer_call(text):
        return True
    # A patient (or child) called Asha: a bare leading "Asha" is the doctor talking to them.
    lead = _NURSE_TITLE if _patient_named_asha(messages) else _NURSE_ADDRESS
    return bool(lead.search(text))


def sse_chunk(cid: str, model: str, delta: dict[str, Any], finish: Optional[str] = None) -> str:
    """One OpenAI chat.completion.chunk SSE event (same shape as _silent_reply's)."""
    c = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"


def completion_json(cid: str, model: str, text: str) -> JSONResponse:
    """Non-streaming OpenAI chat.completion with one assistant message."""
    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    })


def _silent_reply(stream: bool, model: str) -> Response:
    """An OpenAI-shaped empty assistant turn: Agora speaks nothing."""
    cid = "chatcmpl-nurse-" + secrets.token_hex(6)
    now = int(time.time())
    if not stream:
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": now, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        })
    chunks = [
        {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return Response(content=body, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- natural fallback

RETRY_BUDGET_S = 3.5  # one same-provider retry; pause + this stays under 4 s
RETRY_PAUSE_S = 0.3

_LANG_NAMES = {"Hindi": "hi", "Kannada": "kn", "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Bengali": "bn"}
_SPOKEN_LANG = re.compile(r"everyday spoken (" + "|".join(_LANG_NAMES) + r")\b")


def conversation_lang(messages: Any) -> str:
    """Best guess at the language the patient should answer in: the script of the
    doctor's newest turn, else the LANGUAGE block in the system prompt, else en."""
    msgs = sanitize_messages(messages)
    system = next((str(m.get("content") or "") for m in msgs if m["role"] in ("system", "developer")), "")
    hit = _SPOKEN_LANG.search(system)
    prompt_lang = _LANG_NAMES[hit.group(1)] if hit else "en"
    last_user = next((str(m.get("content") or "") for m in reversed(msgs) if m["role"] == "user"), "")
    for lg in ("kn", "ta", "te", "bn", "hi"):
        if langmod.has_script(last_user, lg):
            # Hindi and Marathi share Devanagari: trust the prompt to split them.
            return "mr" if lg == "hi" and prompt_lang == "mr" else lg
    return prompt_lang


def natural_reply(stream: bool, model: str, text: str) -> Response:
    """An OpenAI-shaped assistant turn carrying `text` (SSE or JSON)."""
    cid = "chatcmpl-fallback-" + secrets.token_hex(6)
    if not stream:
        return completion_json(cid, model, text)
    body = (sse_chunk(cid, model, {"role": "assistant", "content": text})
            + sse_chunk(cid, model, {}, "stop") + "data: [DONE]\n\n")
    return Response(content=body, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _transient_exc(e: Exception) -> bool:
    return isinstance(e, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


# ---------------------------------------------------------------- endpoint


def _client_factory() -> httpx.AsyncClient:
    """Indirection so tests can inject a mock transport."""
    return httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))


def _err(status: int, message: str, etype: str = "upstream_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": etype, "code": status}}, status_code=status
    )


def _bearer(request: Request) -> str:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else ""


@router.post(PROXY_PATH)
async def patient_chat_completions(request: Request):
    bearer = _bearer(request)
    if not valid_token(bearer):
        return _err(401, "invalid or missing bearer token", "invalid_request_error")
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err(400, "body must be JSON", "invalid_request_error")
    if not isinstance(payload, dict):
        return _err(400, "body must be a JSON object", "invalid_request_error")
    cfgs = configured_providers()
    if not cfgs:
        return _err(503, "no LLM provider configured", "server_error")
    meta = token_meta(bearer)
    if meta:
        # Tool-enabled agents (Asha's follow-up call, Priya, Dr. Iyer, the viva
        # examiner): tools run here, server-side (Agora custom-LLM tool pattern).
        import agent_runtime

        if agent_runtime.is_registered(meta.get("kind")):
            return await agent_runtime.handle(payload, meta, cfgs[0]["model"])
    if addressed_to_nurse(sanitize_messages(payload.get("messages"))):
        log.info("doctor addressed the nurse; patient stays silent")
        return _silent_reply(payload.get("stream", True) is not False, cfgs[0]["model"])

    stream = payload.get("stream", True) is not False
    client = _client_factory()
    for cfg in cfgs:
        body = build_upstream_body(payload, cfg)
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        for attempt in range(2):
            retry = attempt == 1
            try:
                # The same-provider retry is a quick one: it may add at most RETRY_BUDGET_S.
                req = client.build_request(
                    "POST", cfg["url"], json=body, headers=headers,
                    timeout=httpx.Timeout(RETRY_BUDGET_S) if retry else httpx.USE_CLIENT_DEFAULT,
                )
                resp = await client.send(req, stream=True)
            except httpx.HTTPError as e:
                log.warning("%s request failed%s: %s", cfg["name"], " (retry)" if retry else "", type(e).__name__)
                if not retry and _transient_exc(e):
                    await asyncio.sleep(RETRY_PAUSE_S)
                    continue
                break
            if resp.status_code >= 400:
                raw = (await resp.aread()).decode("utf-8", "replace")
                await resp.aclose()
                log.warning("%s HTTP %s%s: %s", cfg["name"], resp.status_code, " (retry)" if retry else "",
                            raw.replace(cfg["api_key"], "***")[:200])
                if not retry and resp.status_code >= 500:
                    await asyncio.sleep(RETRY_PAUSE_S)
                    continue
                break
            ctype = resp.headers.get("content-type", "application/json")
            if not body.get("stream"):
                data = await resp.aread()
                await resp.aclose()
                await client.aclose()
                return Response(content=data, status_code=resp.status_code, media_type=ctype.split(";")[0])

            # Read the first bytes before committing to this provider: an error or
            # empty stream here can still fall through to a retry or the next one.
            it = resp.aiter_raw()
            try:
                first = await it.__anext__()
            except (httpx.HTTPError, StopAsyncIteration) as e:
                await resp.aclose()
                log.warning("%s stream failed before the first token%s: %s", cfg["name"],
                            " (retry)" if retry else "", type(e).__name__)
                if not retry and not isinstance(e, StopAsyncIteration) and _transient_exc(e):
                    await asyncio.sleep(RETRY_PAUSE_S)
                    continue
                break

            async def gen(first=first, it=it, resp=resp, client=client, name=cfg["name"]):
                try:
                    yield first
                    async for chunk in it:
                        yield chunk
                except httpx.HTTPError as e:
                    log.warning("%s stream broke mid-reply: %s", name, type(e).__name__)
                    yield b"data: [DONE]\n\n"
                finally:
                    await resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                gen(),
                status_code=resp.status_code,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
    await client.aclose()
    # Every provider failed. An HTTP error here makes Agora speak its canned
    # failure_message; a short in-character line keeps the patient human.
    line = langmod.natural_retry_line(conversation_lang(payload.get("messages")))
    log.warning("all LLM providers failed; patient answers with a natural retry line")
    return natural_reply(stream, cfgs[0]["model"], line)


# ---------------------------------------------------------------- public URL / tunnel

_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_state: dict[str, Any] = {"url": None, "verified": False, "proc": None}
_state_lock = threading.Lock()


def _find_cloudflared() -> Optional[str]:
    found = shutil.which("cloudflared")
    if found:
        return found
    for p in (r"C:\Program Files (x86)\cloudflared\cloudflared.exe", r"C:\Program Files\cloudflared\cloudflared.exe"):
        if os.path.exists(p):
            return p
    return None


def public_url() -> Optional[str]:
    """HTTPS base URL Agora can reach, or None (then the patient LLM is called directly)."""
    env = os.environ.get("PUBLIC_URL", "").strip().rstrip("/")
    if env.startswith("https://"):
        return env
    with _state_lock:
        if _state["url"] and _state["verified"]:
            return _state["url"]
    return None


def patient_route() -> str:
    url = public_url()
    if not url:
        return "direct"
    return f"proxy via {urlparse(url).netloc}"


def _verify_tunnel(url: str, timeout_s: float = 300.0) -> None:
    # Local DNS for a fresh trycloudflare host can take minutes to resolve on
    # some networks (seen 2026-11-21: ~70 s). Keep probing in the background;
    # /voice/start uses the direct route until the tunnel is verified.
    # cloudflared prints the URL before its DNS record exists; probing at once
    # can poison the resolver's negative cache, so wait a moment first.
    time.sleep(5.0)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=5.0)
            if r.status_code == 200:
                with _state_lock:
                    if _state["url"] == url:
                        _state["verified"] = True
                log.info("tunnel verified: %s", urlparse(url).netloc)
                return
            log.info("tunnel health probe: HTTP %s", r.status_code)
        except httpx.HTTPError as e:
            log.info("tunnel health probe: %s", type(e).__name__)
        time.sleep(3.0)
    log.warning("tunnel %s never answered /health; staying on direct route", urlparse(url).netloc)


def _reader(proc: "subprocess.Popen[str]") -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        m = _TUNNEL_RE.search(line)
        if m and not _state["url"]:
            with _state_lock:
                _state["url"] = m.group(0)
            log.info("tunnel url: %s", urlparse(m.group(0)).netloc)
            threading.Thread(target=_verify_tunnel, args=(m.group(0),), daemon=True).start()


def start_tunnel(port: int) -> bool:
    mode = os.environ.get("PATIENT_LLM_TUNNEL", "").strip().lower()
    if mode not in ("cloudflared", "auto") or os.environ.get("PUBLIC_URL", "").strip():
        return False
    exe = _find_cloudflared()
    if not exe:
        log.warning("PATIENT_LLM_TUNNEL set but cloudflared not found; using direct route")
        return False
    proc = subprocess.Popen(
        [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _state["proc"] = proc
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    return True


def stop_tunnel() -> None:
    proc = _state.get("proc")
    with _state_lock:
        _state.update(url=None, verified=False, proc=None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
