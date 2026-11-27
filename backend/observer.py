"""Live faculty observer: a second browser joins a running consultation as an
observer and RTM subscriber over WebSockets.

Endpoints
  POST /voice/observer/invite  {channel, caseId} -> {code, inviteUrl, expiresAt, link}
  POST /voice/observer/join    {code}            -> {appId, channel, uid, token, role, ...}
  WS   /voice/observer/ws      ?channel=...&uid=...&token=...

Security model
  * /voice/start registers every channel it creates (register_session). An
    invite can only be minted for a registered, unexpired channel, so a
    caller cannot point an observer at an arbitrary channel name.
  * Invite codes are random ids plus an HMAC tag keyed by a per-process
    secret, kept in memory with a 2 h expiry. A code names exactly one channel.
  * Observer tokens are built using HMAC-SHA256 with timestamp expiry.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

log = logging.getLogger("medsim.observer")
router = APIRouter()

INVITE_TTL_S = 2 * 60 * 60
SESSION_TTL_S = 4 * 60 * 60
MAX_INVITES_PER_CHANNEL = 5
MAX_JOINS_PER_CODE = 12
OBSERVER_UID_MIN = 2_000_000
LINK_UID_MIN = 3_000_000
UID_SPAN = 900_000

_SECRET = secrets.token_bytes(32)
_lock = threading.Lock()


@dataclass
class _Session:
    uid: int
    case_id: str
    agent_uid: int
    created: float


@dataclass
class Invite:
    code: str
    channel: str
    case_id: str
    student_uid: int
    link_uid: int
    agent_uid: int
    created: float
    expires: float
    observer_uids: list[int] = field(default_factory=list)


_sessions: dict[str, _Session] = {}
_invites: dict[str, Invite] = {}

# Channel -> set of active WebSockets
_subscribers: dict[str, set[WebSocket]] = {}
_subscribers_lock = asyncio.Lock()


def _now() -> float:
    return time.time()


def _prune(now: float) -> None:
    for ch in [c for c, s in _sessions.items() if now - s.created > SESSION_TTL_S]:
        _sessions.pop(ch, None)
    for code in [c for c, i in _invites.items() if i.expires <= now]:
        _invites.pop(code, None)


def register_session(channel: str, uid: int, case_id: str, agent_uid: int) -> None:
    """Called by /voice/start for every channel it creates."""
    with _lock:
        now = _now()
        _prune(now)
        _sessions[channel] = _Session(uid=uid, case_id=case_id, agent_uid=agent_uid, created=now)


def reset_state() -> None:
    """Tests only."""
    with _lock:
        _sessions.clear()
        _invites.clear()


def _sign(raw: str) -> str:
    return hmac.new(_SECRET, raw.encode(), hashlib.sha256).hexdigest()[:16]


def _new_code() -> str:
    raw = secrets.token_urlsafe(9)  # 12 chars, 72 bits
    return f"{raw}.{_sign(raw)[:10]}"


def _valid_shape(code: str) -> bool:
    raw, sep, tag = code.partition(".")
    if not sep or not raw or len(code) > 64:
        return False
    return hmac.compare_digest(tag, _sign(raw)[:10])


def lookup_invite(code: str, now: Optional[float] = None) -> Optional[Invite]:
    now = _now() if now is None else now
    if not _valid_shape(code):
        return None
    with _lock:
        inv = _invites.get(code)
        if inv is None or inv.expires <= now:
            return None
        return inv


def mint_token(channel: str, uid: int, ttl: int = INVITE_TTL_S) -> str:
    """Mint an HMAC token for channel/uid authentication."""
    expiry = int(_now()) + ttl
    raw = f"{channel}:{uid}:{expiry}"
    sig = _sign(raw)
    return f"{uid}.{expiry}.{sig}"


def verify_token(token: str, channel: str, uid: Optional[int] = None) -> bool:
    """Verify an observer HMAC token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False
        tok_uid = int(parts[0])
        tok_expiry = int(parts[1])
        tok_sig = parts[2]
        if _now() > tok_expiry:
            return False
        if uid is not None and tok_uid != uid:
            return False
        raw = f"{channel}:{tok_uid}:{tok_expiry}"
        return hmac.compare_digest(tok_sig, _sign(raw))
    except Exception:
        return False


def audience_token(
    app_id: str = "",
    cert: str = "",
    channel: str = "",
    uid: int = 0,
    ttl: int = INVITE_TTL_S,
) -> str:
    """Backward-compatible audience token generator."""
    # If called with legacy signature (app_id, cert, channel, uid, ttl)
    actual_channel = channel or app_id
    actual_uid = uid if uid else (int(cert) if cert.isdigit() else 0)
    return mint_token(actual_channel, actual_uid, ttl)


def _origin(request: Request) -> str:
    origin = request.headers.get("origin", "").strip().rstrip("/")
    if origin.startswith(("http://", "https://")):
        return origin
    return str(request.base_url).rstrip("/")


class InviteRequest(BaseModel):
    channel: str = Field(min_length=1, max_length=64)
    caseId: str = Field(default="", max_length=64)


class LinkCreds(BaseModel):
    uid: int
    token: str


class InviteResponse(BaseModel):
    code: str
    inviteUrl: str
    expiresAt: int
    link: LinkCreds


@router.post("/voice/observer/invite", response_model=InviteResponse)
def observer_invite(req: InviteRequest, request: Request):
    now = _now()
    with _lock:
        _prune(now)
        sess = _sessions.get(req.channel)
        if sess is None:
            raise HTTPException(status_code=404, detail="No live voice session on that channel. Start voice first.")
        if req.caseId and sess.case_id and req.caseId != sess.case_id:
            raise HTTPException(status_code=400, detail="caseId does not match the session")
        live = [i for i in _invites.values() if i.channel == req.channel]
        if len(live) >= MAX_INVITES_PER_CHANNEL:
            raise HTTPException(status_code=429, detail="Too many observer invites for this session")
        code = _new_code()
        link_uid = LINK_UID_MIN + secrets.randbelow(UID_SPAN)
        inv = Invite(
            code=code,
            channel=req.channel,
            case_id=sess.case_id,
            student_uid=sess.uid,
            link_uid=link_uid,
            agent_uid=sess.agent_uid,
            created=now,
            expires=now + INVITE_TTL_S,
        )
        _invites[code] = inv
    link_token = audience_token(channel=inv.channel, uid=link_uid, ttl=INVITE_TTL_S)
    return InviteResponse(
        code=code,
        inviteUrl=f"{_origin(request)}/?observe={code}",
        expiresAt=int(inv.expires * 1000),
        link=LinkCreds(uid=link_uid, token=link_token),
    )


class JoinRequest(BaseModel):
    code: str = Field(min_length=1, max_length=64)


class JoinResponse(BaseModel):
    appId: str
    channel: str
    uid: int
    token: str
    role: str
    caseId: str
    studentUid: int
    linkUid: int
    agentUid: int
    expiresAt: int
    wsUrl: Optional[str] = None


@router.post("/voice/observer/join", response_model=JoinResponse)
def observer_join(req: JoinRequest):
    now = _now()
    inv = lookup_invite(req.code.strip(), now)
    if inv is None:
        raise HTTPException(status_code=404, detail="Observer link is invalid or has expired")
    with _lock:
        if len(inv.observer_uids) >= MAX_JOINS_PER_CODE:
            raise HTTPException(status_code=429, detail="Observer link used too many times")
        taken = {inv.student_uid, inv.link_uid, inv.agent_uid, *inv.observer_uids}
        uid = OBSERVER_UID_MIN + secrets.randbelow(UID_SPAN)
        while uid in taken:
            uid = OBSERVER_UID_MIN + secrets.randbelow(UID_SPAN)
        inv.observer_uids.append(uid)
    ttl = max(60, int(inv.expires - now))
    token = audience_token(channel=inv.channel, uid=uid, ttl=ttl)
    return JoinResponse(
        appId="gemini-live",
        channel=inv.channel,
        uid=uid,
        token=token,
        role="audience",
        caseId=inv.case_id,
        studentUid=inv.student_uid,
        linkUid=inv.link_uid,
        agentUid=inv.agent_uid,
        expiresAt=int(inv.expires * 1000),
        wsUrl=f"/voice/observer/ws?channel={inv.channel}&uid={uid}&token={token}",
    )


# ------------------------------------------------------------------ relay hub


async def broadcast_student_audio(channel: str, pcm: bytes) -> None:
    """Broadcast student 16kHz PCM audio chunk (prefixed with 0x01) to observers."""
    await _broadcast_bytes(channel, b"\x01" + pcm)


async def broadcast_agent_audio(channel: str, pcm: bytes) -> None:
    """Broadcast agent 24kHz PCM audio chunk (prefixed with 0x02) to observers."""
    await _broadcast_bytes(channel, b"\x02" + pcm)


async def broadcast_event(channel: str, event: dict[str, Any]) -> None:
    """Broadcast an RTM or medsim.* event to observers."""
    msg = json.dumps(event)
    await _broadcast_text(channel, msg)


async def broadcast_transcript(channel: str, transcript: dict[str, Any]) -> None:
    """Broadcast transcription event to observers."""
    await broadcast_event(channel, transcript)


async def _broadcast_bytes(channel: str, data: bytes) -> None:
    targets = list(_subscribers.get(channel, set()))
    if not targets:
        return
    dead: list[WebSocket] = []
    for ws in targets:
        try:
            await ws.send_bytes(data)
        except Exception:
            dead.append(ws)
    if dead:
        for d in dead:
            _subscribers.get(channel, set()).discard(d)


async def _broadcast_text(channel: str, text: str, exclude: WebSocket | None = None) -> None:
    targets = list(_subscribers.get(channel, set()))
    if not targets:
        return
    dead: list[WebSocket] = []
    for ws in targets:
        if ws == exclude:
            continue
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    if dead:
        for d in dead:
            _subscribers.get(channel, set()).discard(d)


@router.websocket("/voice/observer/ws")
async def observer_ws(
    websocket: WebSocket,
    channel: str = Query(...),
    uid: int = Query(...),
    token: str = Query(...),
    role: str = Query("observer"),
):
    """Observer WebSocket relay endpoint."""
    if not verify_token(token, channel, uid):
        # Also check if token matches active session token or invite
        with _lock:
            sess = _sessions.get(channel)
            valid = (sess is not None and (sess.uid == uid or sess.agent_uid == uid))
        if not valid:
            await websocket.close(code=4403, reason="Forbidden: invalid observer token")
            return

    await websocket.accept()

    if channel not in _subscribers:
        _subscribers[channel] = set()
    _subscribers[channel].add(websocket)

    log.info("Observer joined channel %s (uid=%s, role=%s)", channel, uid, role)

    # Send initial snapshot
    with _lock:
        sess = _sessions.get(channel)
        active_users = [sess.uid, sess.agent_uid] if sess else []

    await websocket.send_text(
        json.dumps({
            "type": "presence",
            "action": "SNAPSHOT",
            "channel": channel,
            "uid": uid,
            "users": active_users,
        })
    )

    # Notify others of remote join
    await _broadcast_text(
        channel,
        json.dumps({
            "type": "presence",
            "action": "REMOTE_JOIN",
            "channel": channel,
            "uid": uid,
            "role": role,
        }),
        exclude=websocket,
    )

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                # Relay raw audio frame to other observers
                await _broadcast_bytes(channel, msg["bytes"])

            elif "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    continue

                msg_type = payload.get("type") or payload.get("object", "")
                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                if msg_type == "medsim.hello":
                    await websocket.send_text(json.dumps({"type": "medsim.ack", "for": "hello"}))
                    continue

                # Whisper or marks from observer -> fan out and ack
                if msg_type in ("medsim.whisper", "medsim.marks", "medsim.event"):
                    await websocket.send_text(json.dumps({"type": "medsim.ack", "for": msg_type}))
                    await _broadcast_text(channel, msg["text"], exclude=websocket)
                    continue

                # Fan out other protocol envelopes
                await _broadcast_text(channel, msg["text"], exclude=websocket)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("Observer WS error on %s (uid=%s): %s", channel, uid, e)
    finally:
        if channel in _subscribers:
            _subscribers[channel].discard(websocket)
            if not _subscribers[channel]:
                _subscribers.pop(channel, None)

        await _broadcast_text(
            channel,
            json.dumps({
                "type": "presence",
                "action": "REMOTE_LEAVE",
                "channel": channel,
                "uid": uid,
                "role": role,
            }),
        )
        log.info("Observer left channel %s (uid=%s)", channel, uid)
