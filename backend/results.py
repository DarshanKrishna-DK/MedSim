"""Consultation result delivery: POST /agent/results/send.

Sends the student's debrief to Telegram (the existing notify channel): a Telegram HTML
summary, a one-page PDF report card (pdf_report.py) and a short Gemini TTS voice summary in the
session language (best effort). Rate limited per destination; NOTIFY_ALLOWED_DESTS applies.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import lang as langmod
from integrations import notify
from pdf_report import build_report_pdf

log = logging.getLogger("medsim.results")
router = APIRouter()

RATE_MAX = 10
RATE_WINDOW_S = 3600
MAX_TEXT = 1500
AUDIO_TIMEOUT_S = 30
DISCLAIMER = "This is a training simulation, not medical advice."

_lock = threading.Lock()
_hits: dict[str, list[float]] = {}


class Profile(BaseModel):
    name: str = Field("", max_length=80)
    university: str = Field("", max_length=120)


class CaseInfo(BaseModel):
    id: str = Field("", max_length=80)
    title: str = Field("", max_length=160)
    patientName: str = Field("", max_length=80)
    specialty: str = Field("", max_length=80)
    chiefComplaint: str = Field("", max_length=300)


class Domain(BaseModel):
    name: str = Field(..., max_length=80)
    score: float = 0
    max: float = 100


class Diagnosis(BaseModel):
    chosen: Optional[str] = Field(None, max_length=160)
    correct: Optional[str] = Field(None, max_length=160)
    isCorrect: Optional[bool] = None


class Evaluation(BaseModel):
    score: Optional[float] = None
    verdict: str = Field("", max_length=40)
    domain_scores: list[Domain] = Field(default_factory=list, max_length=8)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)
    narrative: str = Field("", max_length=3000)
    diagnosis: Optional[Diagnosis] = None
    prescriptions: list[str] = Field(default_factory=list, max_length=20)
    tests: list[str] = Field(default_factory=list, max_length=40)


class ResultRequest(BaseModel):
    to: Optional[str] = Field(None, max_length=80)
    profile: Profile = Field(default_factory=Profile)
    case: CaseInfo = Field(default_factory=CaseInfo)
    evaluation: Evaluation
    lang: str = "en"
    audio: bool = True


# ------------------------------------------------------------------ text


def _clean(s: Any, limit: int = 200) -> str:
    t = " ".join(str(s or "").replace("—", "-").replace("–", "-").split())
    return t if len(t) <= limit else t[: limit - 3].rstrip() + "..."


def _case_title(req: ResultRequest) -> str:
    return _clean(req.case.title or req.case.patientName or req.case.id or "Outpatient case", 120)


def _score(req: ResultRequest) -> Optional[int]:
    s = req.evaluation.score
    if s is None and req.evaluation.domain_scores:
        parts = [d.score / d.max * 100 for d in req.evaluation.domain_scores if d.max > 0]
        s = sum(parts) / len(parts) if parts else None
    return None if s is None else int(round(max(0.0, min(100.0, s))))


def _dx(req: ResultRequest) -> tuple[Optional[bool], str]:
    d = req.evaluation.diagnosis
    if not d or not (d.chosen or d.correct):
        return None, ""
    ok = d.isCorrect if d.isCorrect is not None else (
        bool(d.chosen and d.correct and d.chosen.strip().lower() == d.correct.strip().lower()))
    if not d.chosen:
        return False, f"none submitted (case: {_clean(d.correct, 80)})"
    if ok:
        return True, f"{_clean(d.chosen, 80)}, correct"
    return False, f"{_clean(d.chosen, 80)}, not the case diagnosis" + (f" ({_clean(d.correct, 80)})" if d.correct else "")


def compose_result_html(req: ResultRequest, max_len: int = MAX_TEXT) -> str:
    """Telegram HTML summary (<b>, <i>, bullet lines), every value escaped, at most max_len chars."""
    e = lambda s: html.escape(s, quote=False)  # noqa: E731
    p, ev = req.profile, req.evaluation
    score = _score(req)
    _, dx = _dx(req)
    head = ["<b>MedSim consultation result</b>"]
    student = _clean(p.name or "Student", 80) + (f" ({_clean(p.university, 80)})" if p.university else "")
    head.append(f"<b>Student:</b> {e(student)}")
    case = _case_title(req) + (f", {_clean(req.case.specialty, 60)}" if req.case.specialty else "")
    head.append(f"<b>Case:</b> {e(case)}")
    if req.case.chiefComplaint:
        head.append(f"<i>{e(_clean(req.case.chiefComplaint, 160))}</i>")
    sc = f"{score}/100" if score is not None else "not scored"
    head.append(f"<b>Score:</b> {e(sc)}" + (f", {e(_clean(ev.verdict, 40))}" if ev.verdict else ""))
    if dx:
        head.append(f"<b>Diagnosis:</b> {e(dx)}")
    tail = f"<i>{DISCLAIMER}</i>"
    strengths = [_clean(s, 180) for s in ev.strengths if str(s).strip()][:3]
    improve = [_clean(s, 180) for s in ev.improvements if str(s).strip()][:3]

    def render(cut: int) -> str:
        body = list(head)
        for title, items in (("What went well", strengths), ("What to work on", improve)):
            if items:
                body.append("")
                body.append(f"<b>{title}</b>")
                body += [f"• {e(_clean(i, cut))}" for i in items]
        return "\n".join(body + ["", tail])

    for cut in (180, 120, 80, 50):
        out = render(cut)
        if len(out) <= max_len:
            return out
    return render(40)[:max_len]


def voice_script(req: ResultRequest) -> str:
    """About 20 seconds of plain English for Gemini TTS."""
    first = _clean(req.profile.name or "Doctor", 60).split(" ")[0]
    score = _score(req)
    ok, _ = _dx(req)
    parts = [f"Hello {first}. Here is your MedSim result for {_case_title(req)}."]
    if score is not None:
        parts.append(f"You scored {score} out of 100" + (f", {_clean(req.evaluation.verdict, 40)}." if req.evaluation.verdict else "."))
    if ok is True:
        parts.append("Your diagnosis was correct.")
    elif ok is False:
        parts.append("Your diagnosis did not match the case.")
    if req.evaluation.strengths:
        parts.append(f"What went well: {_clean(req.evaluation.strengths[0], 140).rstrip('.')}.")
    if req.evaluation.improvements:
        parts.append(f"Work on: {_clean(req.evaluation.improvements[0], 140).rstrip('.')}.")
    parts.append("This is a training simulation. Well done for practising.")
    return " ".join(parts)


async def _translate(script: str, lang: str) -> Optional[str]:
    from llm_provider import chat, strip_think

    name = langmod.LANG_NAMES.get(lang, lang)
    msgs = [{"role": "system", "content": f"Translate the user's text into natural spoken {name} in its native script. "
                                          "Output only the translation."},
            {"role": "user", "content": script}]
    msg, _ = await asyncio.wait_for(chat(msgs, max_tokens=500, temperature=0.2), timeout=15)
    out = strip_think(msg.get("content") or "").strip()
    return out if out and langmod.has_script(out, lang) else None


async def make_audio(req: ResultRequest) -> tuple[Optional[bytes], str]:
    """(mp3, lang used) or (None, reason). Never raises."""
    lang = langmod.norm_lang(req.lang)
    script = voice_script(req)
    try:
        if lang != "en":
            translated = None
            try:
                translated = await _translate(script, lang)
            except Exception as e:  # noqa: BLE001
                log.info("result voice translation failed: %s", type(e).__name__)
            if translated:
                script = translated
            else:
                lang = "en"
        import rx_audio

        wav, _, _ = await asyncio.wait_for(rx_audio.synthesize(script, lang), timeout=AUDIO_TIMEOUT_S)
        return (wav or None), lang
    except HTTPException as e:
        return None, str(e.detail)[:160]
    except Exception as e:  # noqa: BLE001
        return None, f"Voice summary failed: {type(e).__name__}"


def make_pdf(req: ResultRequest) -> bytes:
    ev = req.evaluation
    _, dx = _dx(req)
    domains = [(_clean(d.name, 48), (d.score / d.max * 100) if d.max > 0 else 0) for d in ev.domain_scores]
    return build_report_pdf(
        student=_clean(req.profile.name or "Student", 80),
        university=_clean(req.profile.university, 100),
        case_title=_case_title(req),
        specialty=_clean(req.case.specialty, 60),
        complaint=_clean(req.case.chiefComplaint, 200),
        score=_score(req),
        verdict=_clean(ev.verdict, 40),
        domains=domains,
        strengths=[_clean(s, 300) for s in ev.strengths][:5],
        improvements=[_clean(s, 300) for s in ev.improvements][:5],
        narrative=_clean(ev.narrative, 900),
        diagnosis_line=dx,
        date_text=datetime.now().strftime("%d %b %Y, %H:%M"),
    )


# ------------------------------------------------------------------ route


def _rate_ok(key: str) -> bool:
    now = time.time()
    with _lock:
        hits = [t for t in _hits.get(key, []) if now - t < RATE_WINDOW_S]
        if len(hits) >= RATE_MAX:
            _hits[key] = hits
            return False
        hits.append(now)
        _hits[key] = hits
    return True


@router.post("/agent/results/send")
async def send_result(req: ResultRequest) -> dict[str, Any]:
    dest = (req.to or "").strip() or "telegram"
    if not notify.is_telegram(dest) or (dest.lower().startswith("tg:") and not re.fullmatch(r"tg:-?\d{1,20}", dest.lower())):
        raise HTTPException(status_code=400, detail="Results go to Telegram: use 'telegram', '@username' or 'tg:<chat id>'")
    if not notify._telegram_ready():
        raise HTTPException(status_code=503, detail="Telegram is not configured (set TELEGRAM_BOT_TOKEN)")
    if not notify.dest_allowed(dest):
        raise HTTPException(status_code=403, detail="This destination is not on the allowed list (NOTIFY_ALLOWED_DESTS)")
    key = dest.lower()
    if not _rate_ok(key):
        raise HTTPException(status_code=429, detail=f"Rate limit: at most {RATE_MAX} results per destination per hour")

    text = compose_result_html(req)
    try:
        pdf: Optional[bytes] = make_pdf(req)
    except Exception as e:  # noqa: BLE001
        log.warning("report card PDF failed: %s", type(e).__name__)
        pdf = None
    audio_bytes, audio_note = (await make_audio(req)) if req.audio else (None, "not requested")

    slug = "".join(ch if ch.isalnum() else "-" for ch in (req.case.id or "case"))[:40].strip("-") or "case"
    document = (f"medsim-report-{slug}.pdf", pdf, "application/pdf", "MedSim report card (training simulation)") if pdf else None
    audio = ("medsim-result.wav", audio_bytes, "MedSim result summary") if audio_bytes else None
    res = await notify.send_telegram_bundle(dest, text, document, audio)

    parts = dict(res.get("parts") or {})
    if not pdf:
        parts["document"] = {"ok": False, "error": "PDF generation failed"}
    if not audio_bytes:
        parts["audio"] = {"ok": False, "skipped": True, "error": audio_note}
    elif "audio" in parts:
        parts["audio"]["lang"] = audio_note
    out = {"ok": bool(res.get("ok")), "channel": "telegram", "to": res.get("to") or notify.mask(dest), "parts": parts}
    if res.get("resolvedTo"):
        out["resolvedTo"] = res["resolvedTo"]
    if not res.get("ok"):
        out["error"] = res.get("error") or (parts.get("text") or {}).get("error") or "Send failed"
    return out
