"""Patient notifications: prescription + Murf voice-prescription link + follow-up date.

Channels (all free tier, all env-gated, none ever raises):
  whatsapp  Twilio WhatsApp sandbox   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
  sms       Twilio trial SMS          TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_FROM
  email     SMTP over SSL (Gmail)     SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

Also hosts the MP3 in memory (24h, 200 entries) at GET /rx/audio/{token}.mp3. Those paths are
public on purpose (the patient phone opens them); the token is unguessable.
Phone numbers, emails and secrets are never logged unmasked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from llm_proxy import public_url

log = logging.getLogger("medsim.notify")
router = APIRouter()

AUDIO_TTL_S = 24 * 3600
AUDIO_MAX = 200
RATE_MAX = 5
RATE_WINDOW_S = 3600
SMTP_TIMEOUT_S = 10
MAX_BODY = 1200
DISCLAIMER = "This is a training simulation, not medical advice."
DEFAULT_WA_FROM = "+14155238886"
TWILIO_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

_lock = threading.Lock()
_audio: dict[str, tuple[float, bytes]] = {}
_sends: dict[str, list[float]] = {}


# ------------------------------------------------------------------ audio hosting


def store_audio(audio_bytes: bytes) -> str:
    """Keep the audio (WAV or MP3) in memory and return its public path '/rx/audio/<token>.(wav|mp3)'."""
    now = time.time()
    token = secrets.token_urlsafe(16)
    b = bytes(audio_bytes)
    ext = "wav" if b.startswith(b"RIFF") else "mp3"
    with _lock:
        for k in [k for k, (t, _) in _audio.items() if now - t > AUDIO_TTL_S]:
            del _audio[k]
        while len(_audio) >= AUDIO_MAX:
            del _audio[min(_audio, key=lambda k: _audio[k][0])]
        _audio[token] = (now, b)
    return f"/rx/audio/{token}.{ext}"


@router.get("/rx/audio/{token}.mp3")
@router.get("/rx/audio/{token}.wav")
async def rx_audio_file(token: str) -> Response:
    with _lock:
        item = _audio.get(token)
        if item and time.time() - item[0] > AUDIO_TTL_S:
            del _audio[token]
            item = None
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    media_type = "audio/wav" if item[1].startswith(b"RIFF") else "audio/mpeg"
    return Response(content=item[1], media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})


def absolute_url(path: str) -> Optional[str]:
    """Absolute link for a path, or None when no public base URL is reachable."""
    base = public_url()
    return f"{base.rstrip('/')}{path}" if base else None


# ------------------------------------------------------------------ message


def compose_message(
    patient_name: str,
    doctor_name: str,
    meds: list[dict],
    follow_up: Optional[dict],
    audio_url: Optional[str],
    calendar_url: Optional[str],
    lang: str = "en",
) -> str:
    items = []
    for i, m in enumerate(meds or [], 1):
        bits = [str(m.get("name", "")).strip(), str(m.get("dose", "")).strip(), str(m.get("duration", "")).strip()]
        items.append(f"{i}) " + ", ".join(b for b in bits if b))
    parts = [f"MedSim Clinic: Hello {(patient_name or 'there').strip()}."]
    parts.append(f"Your prescription from {(doctor_name or 'your doctor').strip()}: " + ("; ".join(items) or "none") + ".")
    if audio_url:
        label = "Listen in your language" if (lang or "en") != "en" else "Listen to your prescription"
        parts.append(f"{label}: {audio_url}.")
    if follow_up and (follow_up.get("startText") or follow_up.get("reason")):
        when = str(follow_up.get("startText") or "soon").strip()
        reason = str(follow_up.get("reason") or "").strip()
        parts.append(f"Follow-up: {when}" + (f" ({reason})" if reason else "") + ".")
        if calendar_url:
            parts.append(f"Add to calendar: {calendar_url}.")
    text = " ".join(parts)
    room = MAX_BODY - len(DISCLAIMER) - 1
    if len(text) > room:
        text = text[: room - 3].rstrip() + "..."
    return f"{text} {DISCLAIMER}".replace(chr(0x2014), "-")


# ------------------------------------------------------------------ validation / masking

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_phone(raw: str) -> Optional[str]:
    """E.164 or None. 10 bare digits are assumed to be Indian (+91)."""
    s = re.sub(r"[\s\-().]", "", (raw or "").strip())
    if s.startswith("whatsapp:"):
        s = s[9:]
    if s.startswith("00"):
        s = "+" + s[2:]
    if s.startswith("+"):
        return s if re.fullmatch(r"\+\d{8,15}", s) else None
    if re.fullmatch(r"\d{10}", s):
        return "+91" + s
    return None


def is_email(raw: str) -> bool:
    return bool(_EMAIL_RE.match((raw or "").strip()))


def mask(dest: str) -> str:
    d = (dest or "").strip()
    if is_telegram(d):
        return "Telegram chat"
    if "@" in d:
        user, _, dom = d.partition("@")
        return f"{user[:1]}***@{dom}"
    if len(d) > 6:
        return d[:3] + "*" * (len(d) - 7) + d[-4:]
    return "***"


# ------------------------------------------------------------------ config


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _twilio_ready() -> bool:
    return bool(_env("TWILIO_ACCOUNT_SID") and _env("TWILIO_AUTH_TOKEN"))


def _smtp_ready() -> bool:
    return bool(_env("SMTP_USER") and _env("SMTP_PASSWORD"))


def _telegram_ready() -> bool:
    return bool(_env("TELEGRAM_BOT_TOKEN"))


def is_telegram(dest: str) -> bool:
    """'telegram' (the latest chat that pressed Start), '@username', or 'tg:<chat id>'."""
    d = (dest or "").strip().lower()
    return d in ("telegram", "tg") or d.startswith("tg:") or bool(re.fullmatch(r"@[a-z0-9_]{4,32}", d))


def configured_channels() -> list[str]:
    out = []
    if _telegram_ready():
        out.append("telegram")
    if _twilio_ready():
        out.append("whatsapp")
        if _env("TWILIO_SMS_FROM"):
            out.append("sms")
    if _smtp_ready():
        out.append("email")
    return out


def _client_factory() -> httpx.AsyncClient:
    """Indirection so tests can inject a mock transport."""
    return httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0))


_GLOBAL_KEY = "*"


def _max_per_hour() -> int:
    try:
        return max(1, int(_env("NOTIFY_MAX_PER_HOUR") or 30))
    except ValueError:
        return 30


def _norm_dest(raw: str) -> str:
    """Comparable form of a destination: E.164 phone, lower-case email, tg:<id> or 'telegram'."""
    d = (raw or "").strip()
    return normalize_phone(d) or d.lower()


def dest_allowed(key: str) -> bool:
    """NOTIFY_ALLOWED_DESTS (comma list) limits where messages may go; unset means anywhere.
    'telegram' in the list allows every Telegram chat ('telegram', '@user', tg:<id>)."""
    raw = _env("NOTIFY_ALLOWED_DESTS")
    if not raw:
        return True
    allowed = {_norm_dest(x) for x in raw.split(",") if x.strip()}
    k = _norm_dest(key)
    if is_telegram(k):
        return "telegram" in allowed or k in allowed
    return k in allowed


def _rate_ok(key: str) -> bool:
    """Per-destination cap (RATE_MAX an hour), a global cap across all destinations
    (NOTIFY_MAX_PER_HOUR, default 30) and the optional NOTIFY_ALLOWED_DESTS list."""
    if not dest_allowed(key):
        return False
    now = time.time()
    with _lock:
        hits = [t for t in _sends.get(key, []) if now - t < RATE_WINDOW_S]
        total = [t for t in _sends.get(_GLOBAL_KEY, []) if now - t < RATE_WINDOW_S]
        if len(hits) >= RATE_MAX or len(total) >= _max_per_hour():
            _sends[key] = hits
            _sends[_GLOBAL_KEY] = total
            return False
        hits.append(now)
        total.append(now)
        _sends[key] = hits
        _sends[_GLOBAL_KEY] = total
    return True


def _rate_error(key: str) -> str:
    if not dest_allowed(key):
        return "This destination is not on the allowed list (NOTIFY_ALLOWED_DESTS)"
    return f"Rate limit: at most {RATE_MAX} messages per destination per hour ({_max_per_hour()} in total)"


def _fail(channel: str, to: str, error: str) -> dict[str, Any]:
    return {"ok": False, "channel": channel, "to": mask(to), "error": error}


# ------------------------------------------------------------------ channels


async def _send_twilio(channel: str, number: str, message: str, audio_url: Optional[str]) -> dict[str, Any]:
    sid, token = _env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN")
    if not (sid and token):
        return _fail(channel, number, "Twilio is not configured (set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)")
    if channel == "whatsapp":
        frm = "whatsapp:" + (_env("TWILIO_WHATSAPP_FROM") or DEFAULT_WA_FROM).removeprefix("whatsapp:")
        content_sid = _env("TWILIO_WHATSAPP_CONTENT_SID")
        if content_sid:
            # Trial / business-initiated sends: WhatsApp only allows a pre-approved
            # template here (Twilio trial: the "Appointment Reminders" template).
            form = {"From": frm, "To": "whatsapp:" + number, "ContentSid": content_sid}
        else:
            form = {"From": frm, "To": "whatsapp:" + number, "Body": message}
            if audio_url and audio_url.startswith("https://"):
                form["MediaUrl"] = audio_url
    else:
        frm = _env("TWILIO_SMS_FROM")
        if not frm:
            return _fail(channel, number, "SMS is not configured (set TWILIO_SMS_FROM)")
        form = {"From": frm, "To": number, "Body": message}
    try:
        async with _client_factory() as c:
            r = await c.post(TWILIO_URL.format(sid=sid), data=form, auth=(sid, token))
    except httpx.HTTPError as e:
        return _fail(channel, number, f"Twilio request failed: {type(e).__name__}")
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        data = {}
    if r.status_code >= 300:
        msg = str(data.get("message") or "")[:200] if isinstance(data, dict) else ""
        return _fail(channel, number, f"Twilio error {r.status_code}" + (f": {msg}" if msg else ""))
    out = {"ok": True, "channel": channel, "to": mask(number), "sid": data.get("sid") if isinstance(data, dict) else None}
    if "ContentSid" in form:
        out["template"] = True
    return out


def _smtp_send(msg: EmailMessage) -> None:
    host = _env("SMTP_HOST") or "smtp.gmail.com"
    port = int(_env("SMTP_PORT") or 465)
    with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_S, context=ssl.create_default_context()) as s:
        s.login(_env("SMTP_USER"), _env("SMTP_PASSWORD"))
        s.send_message(msg)


async def _send_email(addr: str, message: str, mp3: Optional[bytes], ics: Optional[bytes]) -> dict[str, Any]:
    if not _smtp_ready():
        return _fail("email", addr, "Email is not configured (set SMTP_USER and SMTP_PASSWORD)")
    msg = EmailMessage()
    msg["Subject"] = "Your MedSim prescription"
    msg["From"] = _env("SMTP_FROM") or _env("SMTP_USER")
    msg["To"] = addr
    msg["Message-ID"] = f"<{secrets.token_hex(12)}@medsim.local>"
    msg.set_content(message)
    if mp3:
        subtype = "wav" if mp3.startswith(b"RIFF") else "mpeg"
        fname = f"prescription.{subtype}"
        msg.add_attachment(mp3, maintype="audio", subtype=subtype, filename=fname)
    if ics:
        msg.add_attachment(ics, maintype="text", subtype="calendar", filename="follow-up.ics")
    try:
        await asyncio.wait_for(asyncio.to_thread(_smtp_send, msg), timeout=SMTP_TIMEOUT_S + 5)
    except Exception as e:  # noqa: BLE001
        return _fail("email", addr, f"Email send failed: {type(e).__name__}")
    return {"ok": True, "channel": "email", "to": mask(addr), "messageId": msg["Message-ID"]}


# ------------------------------------------------------------------ telegram

_TG_API = "https://api.telegram.org"
_tg_chats: dict[str, int] = {}


def _tg_url(method: str) -> str:
    return f"{_TG_API}/bot{_env('TELEGRAM_BOT_TOKEN')}/{method}"


async def telegram_bot_info() -> Optional[dict[str, Any]]:
    """{username, link} for the configured bot, or None."""
    if not _telegram_ready():
        return None
    try:
        async with _client_factory() as c:
            r = await c.get(_tg_url("getMe"))
        u = (r.json() or {}).get("result") or {}
        name = u.get("username")
        return {"username": name, "link": f"https://t.me/{name}?start=medsim"} if name else None
    except Exception as e:  # noqa: BLE001
        log.warning("telegram getMe failed: %s", type(e).__name__)
        return None


# Next getUpdates offset (last update_id + 1): only new updates are scanned, and
# chats already seen stay in _tg_chats ('@username' -> id, 'telegram' -> latest chat).
_tg_offset: dict[str, Optional[int]] = {"next": None}


async def _tg_scan(c: httpx.AsyncClient) -> None:
    params: dict[str, Any] = {"limit": 100, "timeout": 0}
    if _tg_offset["next"] is not None:
        params["offset"] = _tg_offset["next"]
    r = await c.get(_tg_url("getUpdates"), params=params)
    updates = (r.json() or {}).get("result") or []
    top: Optional[int] = None
    for u in updates:
        if not isinstance(u, dict):
            continue
        uid = u.get("update_id")
        if isinstance(uid, int) and (top is None or uid > top):
            top = uid
        msg = u.get("message") or u.get("edited_message") or {}
        chat = msg.get("chat") or {}
        if not chat.get("id"):
            continue
        if chat.get("username"):
            _tg_chats["@" + str(chat["username"]).lower()] = chat["id"]
        _tg_chats["telegram"] = chat["id"]
    if top is not None:
        _tg_offset["next"] = top + 1


async def _tg_resolve_chat(c: httpx.AsyncClient, dest: str) -> Optional[int]:
    d = dest.strip().lower()
    if d.startswith("tg:"):
        try:
            return int(d[3:])
        except ValueError:
            return None
    if d == "tg":
        d = "telegram"
    # The bot poller (telegram_bot.py) consumes getUpdates, so ask its cache first.
    try:
        import telegram_bot

        hit = telegram_bot.known_chat(d)
        if hit or telegram_bot.polling_active():
            return hit or _tg_chats.get(d)
    except ImportError:
        pass
    try:
        await _tg_scan(c)
    except Exception as e:  # noqa: BLE001
        log.warning("telegram getUpdates failed: %s", type(e).__name__)
    return _tg_chats.get(d)


async def resolve_telegram(dest: str) -> Optional[str]:
    """'telegram' / '@user' / 'tg:<id>' -> 'tg:<chat id>', or None when unknown. Never raises."""
    if not (_telegram_ready() and is_telegram(dest)):
        return None
    try:
        async with _client_factory() as c:
            chat_id = await _tg_resolve_chat(c, dest)
    except Exception as e:  # noqa: BLE001
        log.warning("telegram resolve failed: %s", type(e).__name__)
        return None
    return f"tg:{chat_id}" if chat_id else None


_SAFE_URL = re.compile(r"https://[^\s\"'<>]+")


def _tg_link(url_escaped: str, label: str, original: str) -> str:
    """An <a> tag only for a clean https:// URL; anything else stays plain (escaped) text."""
    import html as _html

    url = _html.unescape(url_escaped)
    if not _SAFE_URL.fullmatch(url):
        return original
    return f'<a href="{_html.escape(url, quote=True)}">{label}</a>'


def telegram_html(message: str, audio_attached: bool) -> str:
    """Telegram version of the message: links become short tappable labels, and the
    audio link is dropped when the MP3 itself is attached."""
    import html as _html

    text = _html.escape(message, quote=False)
    if audio_attached:
        text = re.sub(r"\s*(Listen to your prescription|Listen in your language): \S+?\.(?=\s|$)", "", text)
    else:
        text = re.sub(r"(Listen to your prescription|Listen in your language): (\S+?)\.(?=\s|$)",
                      lambda m: _tg_link(m.group(2), m.group(1), m.group(0)[:-1]) + ".", text)
    text = re.sub(r"Add to calendar: (\S+?)\.(?=\s|$)",
                  lambda m: _tg_link(m.group(1), "Add to Google Calendar", m.group(0)[:-1]) + ".", text)
    text = re.sub(r"(Join your Asha follow-up call|Asha call): (\S+?)\.?(?=\s|$)",
                  lambda m: _tg_link(m.group(2), "Join your Asha follow-up call", m.group(0)), text)
    return text


async def _send_telegram(dest: str, message: str, mp3: Optional[bytes], ics: Optional[bytes]) -> dict[str, Any]:
    async with _client_factory() as c:
        chat_id = await _tg_resolve_chat(c, dest)
        if not chat_id:
            return _fail("telegram", dest, "No Telegram chat yet: open the bot link on the phone and press Start, then retry")
        resolved = f"tg:{chat_id}"
        # Rate-limited per real chat: 'telegram' and '@user' never share one bucket across people.
        if not _rate_ok(resolved):
            return _fail("telegram", dest, _rate_error(resolved))
        r = await c.post(_tg_url("sendMessage"), json={"chat_id": chat_id, "text": telegram_html(message, bool(mp3)),
                                                        "parse_mode": "HTML", "disable_web_page_preview": True})
        body = r.json() if r.content else {}
        if not body.get("ok"):
            return _fail("telegram", dest, f"Telegram error: {str(body.get('description') or r.status_code)[:120]}")
        sent = {"ok": True, "channel": "telegram", "to": mask(dest), "messageId": str(body["result"]["message_id"]),
                "resolvedTo": resolved}
        if mp3:
            try:
                subtype = "audio/wav" if mp3.startswith(b"RIFF") else "audio/mpeg"
                fname = "prescription.wav" if mp3.startswith(b"RIFF") else "prescription.mp3"
                a = await c.post(_tg_url("sendAudio"), data={"chat_id": str(chat_id), "title": "Your prescription (MedSim)",
                                                              "performer": "MedSim Clinic"},
                                 files={"audio": (fname, mp3, subtype)})
                sent["audioSent"] = bool(a.content and a.json().get("ok"))
            except Exception as e:  # noqa: BLE001
                log.warning("telegram sendAudio failed: %s", type(e).__name__)
                sent["audioSent"] = False
        if ics:
            try:
                d = await c.post(_tg_url("sendDocument"), data={"chat_id": str(chat_id), "caption": "Follow-up: add to your calendar"},
                                 files={"document": ("follow-up.ics", ics, "text/calendar")})
                sent["inviteSent"] = bool(d.content and d.json().get("ok"))
            except Exception as e:  # noqa: BLE001
                log.warning("telegram sendDocument failed: %s", type(e).__name__)
                sent["inviteSent"] = False
        return sent


async def _tg_file(c: httpx.AsyncClient, method: str, field: str, chat_id: int, filename: str, data: bytes,
                   mime: str, extra: dict[str, str]) -> dict[str, Any]:
    """One sendDocument / sendAudio upload; {ok} or {ok: False, error}. Never raises."""
    try:
        r = await c.post(_tg_url(method), data={"chat_id": str(chat_id), **extra},
                         files={field: (filename, data, mime)})
        body = r.json() if r.content else {}
    except Exception as e:  # noqa: BLE001
        log.warning("telegram %s failed: %s", method, type(e).__name__)
        return {"ok": False, "error": f"Telegram request failed: {type(e).__name__}"}
    if not body.get("ok"):
        return {"ok": False, "error": f"Telegram error: {str(body.get('description') or r.status_code)[:120]}"}
    return {"ok": True, "messageId": str((body.get("result") or {}).get("message_id", ""))}


async def send_telegram_bundle(
    dest: str,
    html_text: str,
    document: Optional[tuple[str, bytes, str, str]] = None,
    audio: Optional[tuple[str, bytes, str]] = None,
) -> dict[str, Any]:
    """Telegram HTML message, then an optional document (filename, bytes, mime, caption) and an
    optional MP3 (filename, bytes, title). Per-part results under 'parts'. The caller owns rate
    limiting; NOTIFY_ALLOWED_DESTS is still honoured. Never raises."""
    d = (dest or "telegram").strip()
    if not _telegram_ready():
        return _fail("telegram", d, "Telegram is not configured (set TELEGRAM_BOT_TOKEN)")
    if not is_telegram(d):
        return _fail("telegram", d, "Not a Telegram destination (use 'telegram', '@username' or 'tg:<chat id>')")
    if not dest_allowed(d):
        return _fail("telegram", d, _rate_error(d))
    try:
        async with _client_factory() as c:
            chat_id = await _tg_resolve_chat(c, d)
            if not chat_id:
                return _fail("telegram", d, "No Telegram chat yet: open the bot link on the phone and press Start, then retry")
            parts: dict[str, Any] = {}
            try:
                r = await c.post(_tg_url("sendMessage"), json={"chat_id": chat_id, "text": html_text,
                                                                "parse_mode": "HTML", "disable_web_page_preview": True})
                body = r.json() if r.content else {}
                parts["text"] = ({"ok": True, "messageId": str(body["result"]["message_id"])} if body.get("ok")
                                 else {"ok": False, "error": f"Telegram error: {str(body.get('description') or r.status_code)[:120]}"})
            except Exception as e:  # noqa: BLE001
                parts["text"] = {"ok": False, "error": f"Telegram request failed: {type(e).__name__}"}
            if document:
                name, data, mime, caption = document
                parts["document"] = await _tg_file(c, "sendDocument", "document", chat_id, name, data, mime,
                                                   {"caption": caption[:1000]} if caption else {})
            if audio:
                name, data, title = audio
                mime = "audio/wav" if (name.endswith(".wav") or data.startswith(b"RIFF")) else "audio/mpeg"
                parts["audio"] = await _tg_file(c, "sendAudio", "audio", chat_id, name, data, mime,
                                                {"title": title[:64], "performer": "MedSim"})
    except Exception as e:  # noqa: BLE001
        return _fail("telegram", d, f"Unexpected error: {type(e).__name__}")
    ok = bool(parts.get("text", {}).get("ok"))
    log.info("notify telegram bundle ok=%s parts=%s", ok, {k: v.get("ok") for k, v in parts.items()})
    return {"ok": ok, "channel": "telegram", "to": mask(d), "resolvedTo": f"tg:{chat_id}", "parts": parts}


# ------------------------------------------------------------------ public entry


async def send_to_patient(
    channel: str,
    to: str,
    message: str,
    audio_url: Optional[str] = None,
    mp3: Optional[bytes] = None,
    ics: Optional[bytes] = None,
) -> dict[str, Any]:
    ch = (channel or "auto").strip().lower()
    dest = (to or "").strip()
    try:
        phone = normalize_phone(dest)
        email_ok = is_email(dest)
        if ch == "auto" and is_telegram(dest) and _telegram_ready():
            ch = "telegram"
        if ch == "telegram":
            if not _telegram_ready():
                return _fail(ch, dest, "Telegram is not configured (set TELEGRAM_BOT_TOKEN)")
            tdest = dest if is_telegram(dest) else "telegram"
            if not dest_allowed(tdest):
                return _fail(ch, tdest, _rate_error(tdest))
            res = await _send_telegram(tdest, message, mp3, ics)
            log.info("notify telegram ok=%s", res.get("ok"))
            return res
        if ch == "auto":
            if phone and _twilio_ready():
                ch = "whatsapp"
            elif email_ok and _smtp_ready():
                ch = "email"
            else:
                have = ", ".join(configured_channels()) or "none"
                return _fail("auto", dest, f"No configured channel fits this destination (configured: {have})")
        if ch not in ("whatsapp", "sms", "email"):
            return _fail(ch, dest, "Unknown channel (use telegram, whatsapp, sms, email or auto)")
        if ch == "email":
            if not email_ok:
                return _fail(ch, dest, "Invalid email address")
            key, target = dest.lower(), dest
        else:
            if not phone:
                return _fail(ch, dest, "Invalid phone number (use E.164 like +919876543210)")
            key, target = phone, phone
        if not _rate_ok(key):
            return _fail(ch, target, _rate_error(key))
        if ch == "email":
            res = await _send_email(target, message, mp3, ics)
        else:
            res = await _send_twilio(ch, target, message, audio_url)
        log.info("notify %s to %s ok=%s", ch, mask(target), res.get("ok"))
        return res
    except Exception as e:  # noqa: BLE001
        return _fail(ch, dest, f"Unexpected error: {type(e).__name__}")
