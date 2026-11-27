"""Referral letter pipeline for the nurse tool `refer_patient`.

FHIR ServiceRequest (category referral) + SBAR letter (LLM, with a deterministic
fallback) + Google Doc/PDF via the Apps Script bridge + a Gemini TTS voice handover,
optionally delivered to a consultant (REFERRAL_DEST). Every step degrades on its
own; nothing here raises. Every artifact is labelled as a simulation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

log = logging.getLogger("medsim.referral")

LABEL = "SIMULATION, synthetic patient"
URGENCIES = ("routine", "urgent", "emergency")

SBAR_SYSTEM = (
    "You write a concise clinical referral in SBAR form for a medical training simulation. "
    "Use only the facts given, never invent results. Plain short sentences, no markdown, "
    "no dashes used as punctuation (use commas or full stops). Reply with JSON only: "
    '{"situation":"...","background":"...","assessment":"...","recommendation":"...",'
    '"voice":"..."}. Each SBAR field is at most 45 words. "voice" is a spoken handover of the '
    "same content for a phone call, 45 to 55 words, natural speech, numbers as words where easy, "
    "and it must begin with: This is a simulation, synthetic patient."
)


def clean_text(text: Any) -> str:
    """No em or en dashes in anything we publish."""
    return re.sub("\\s*[\\u2014\\u2013]\\s*", ", ", str(text or "")).strip()


def normalise_urgency(value: Any) -> str:
    u = str(value or "routine").strip().lower()
    return u if u in URGENCIES else "routine"


def _facts(ctx: dict[str, Any]) -> str:
    return json.dumps({
        "patient": f"{ctx.get('name', 'Patient')}, {ctx.get('age', '?')} years, {ctx.get('gender', '')}",
        "referToSpecialty": ctx.get("specialty"),
        "urgency": ctx.get("urgency"),
        "reasonForReferral": ctx.get("reason"),
        "chiefComplaint": ctx.get("complaint"),
        "workingDiagnosis": ctx.get("diagnosis") or "not stated",
        "vitals": ctx.get("vitals") or {},
        "allergies": ctx.get("allergies") or "not recorded",
        "abnormalResultsSeen": ctx.get("abnormal") or [],
        "testsOrdered": ctx.get("tests") or [],
        "prescriptions": ctx.get("prescriptions") or [],
    })


def _vitals_text(v: dict[str, Any]) -> str:
    keys = (("hr", "HR"), ("bp", "BP"), ("spo2", "SpO2"), ("temp", "Temp"), ("rr", "RR"))
    return ", ".join(f"{label} {v[k]}" for k, label in keys if v.get(k) not in (None, ""))


def fallback_sbar(ctx: dict[str, Any]) -> dict[str, str]:
    name, spec, urg = ctx.get("name", "The patient"), ctx.get("specialty", "the specialist"), ctx.get("urgency", "routine")
    age = ctx.get("age")
    who = f"{name}, {int(age)} years" if isinstance(age, (int, float)) else name
    complaint = ctx.get("complaint") or "a presenting complaint"
    vit = _vitals_text(ctx.get("vitals") or {})
    abn = ", ".join(ctx.get("abnormal") or []) or "no abnormal results yet"
    rx = ", ".join(ctx.get("prescriptions") or []) or "none so far"
    dx = ctx.get("diagnosis") or "not yet established"
    sbar = {
        "situation": f"{who} is referred to {spec} ({urg}). Reason: {ctx.get('reason') or complaint}.",
        "background": f"Presented with {complaint}. Vitals: {vit or 'not recorded'}. Allergies: {ctx.get('allergies') or 'not recorded'}. Current prescriptions: {rx}.",
        "assessment": f"Working diagnosis: {dx}. Abnormal results: {abn}.",
        "recommendation": f"Please review this patient {'as an emergency' if urg == 'emergency' else 'urgently' if urg == 'urgent' else 'at the next routine slot'} and advise on further management.",
    }
    sbar["voice"] = (f"This is a simulation, synthetic patient. Handover for {spec}, {urg}. {who}, with {complaint}. "
                     f"Working diagnosis, {dx}. Abnormal results, {abn}. Currently on {rx}. "
                     f"Please review and advise.")
    return {k: clean_text(v) for k, v in sbar.items()}


def _parse(text: str) -> Optional[dict[str, Any]]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


async def build_sbar(ctx: dict[str, Any]) -> dict[str, str]:
    from llm_provider import chat, strip_think

    base = fallback_sbar(ctx)
    try:
        msg, _ = await asyncio.wait_for(chat(
            [{"role": "system", "content": SBAR_SYSTEM}, {"role": "user", "content": _facts(ctx)}],
            json_mode=True, max_tokens=700, temperature=0.2), timeout=25.0)
        obj = _parse(strip_think(msg.get("content") or "")) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("SBAR generation failed: %s", type(e).__name__)
        obj = {}
    out = dict(base)
    for k in ("situation", "background", "assessment", "recommendation", "voice"):
        v = clean_text(obj.get(k))
        if v:
            out[k] = v
    if "simulation" not in out["voice"].lower():
        out["voice"] = f"This is a simulation, synthetic patient. {out['voice']}"
    return out


def letter_text(ctx: dict[str, Any], sbar: dict[str, str]) -> str:
    return "\n".join([
        LABEL,
        f"Referral: {ctx.get('name', 'Patient')} to {ctx.get('specialty', '')} ({ctx.get('urgency', 'routine')})",
        f"Situation: {sbar['situation']}",
        f"Background: {sbar['background']}",
        f"Assessment: {sbar['assessment']}",
        f"Recommendation: {sbar['recommendation']}",
        LABEL,
    ])


async def _doc(ctx: dict[str, Any], sbar: dict[str, str]) -> Optional[dict]:
    from integrations import gcal_script

    try:
        return await asyncio.wait_for(gcal_script.create_referral({
            "patient": ctx.get("name", "Patient"), "specialty": ctx.get("specialty", ""),
            "urgency": ctx.get("urgency", "routine"), "reason": ctx.get("reason", ""),
            "situation": sbar["situation"], "background": sbar["background"],
            "assessment": sbar["assessment"], "recommendation": sbar["recommendation"], "label": LABEL,
        }), timeout=30.0)
    except Exception as e:  # noqa: BLE001
        log.warning("referral doc failed: %s", type(e).__name__)
        return None


async def _voice(script: str) -> tuple[Optional[bytes], Optional[str]]:
    """Gemini TTS handover. Returns (wav, public audio url) or (None, None)."""
    from integrations import notify
    import rx_audio

    try:
        audio, _voice_id, _locale = await asyncio.wait_for(rx_audio.synthesize(script, "en"), timeout=40.0)
        return audio, notify.absolute_url(notify.store_audio(audio))
    except Exception as e:  # noqa: BLE001
        log.warning("referral voice failed: %s", type(e).__name__)
        return None, None


async def _fhir(ctx: dict[str, Any], sbar: dict[str, str]) -> Optional[dict]:
    from integrations import fhir

    try:
        pid = await fhir.ensure_patient(ctx.get("caseId", ""), ctx.get("name", "Synthetic Patient"),
                                        ctx.get("age"), ctx.get("gender", ""))
        if not pid:
            return None
        return await asyncio.wait_for(fhir.record_referral(
            pid, ctx.get("specialty", ""), ctx.get("urgency", "routine"), ctx.get("reason", ""),
            ctx.get("diagnosis") or None, letter_text(ctx, sbar)), timeout=10.0)
    except Exception as e:  # noqa: BLE001
        log.warning("referral FHIR failed: %s", type(e).__name__)
        return None


async def _deliver(ctx: dict[str, Any], sbar: dict[str, str], pdf_url: Optional[str],
                   mp3: Optional[bytes], audio_url: Optional[str]) -> Optional[dict]:
    dest = (os.environ.get("REFERRAL_DEST") or "").strip()
    if not dest:
        return None
    from integrations import notify

    message = (f"{LABEL}. Referral to {ctx.get('specialty')} ({ctx.get('urgency')}) for {ctx.get('name')}. "
               f"{sbar['situation']} {sbar['assessment']}"
               + (f" Letter: {pdf_url}" if pdf_url else "") + (f" Voice handover: {audio_url}" if audio_url else ""))
    channel = "telegram" if notify.is_telegram(dest) else "auto"
    try:
        out = await asyncio.wait_for(notify.send_to_patient(channel, dest, clean_text(message), audio_url, mp3, None), timeout=30.0)
    except Exception as e:  # noqa: BLE001
        log.warning("referral delivery failed: %s", type(e).__name__)
        return {"ok": False, "error": "delivery failed"}
    return {"ok": bool(out.get("ok")), "channel": out.get("channel"), "error": out.get("error")}


async def make_referral(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run the whole pipeline. ctx keys: caseId, name, age, gender, complaint, vitals, allergies,
    specialty, urgency, reason, diagnosis, abnormal[], tests[], prescriptions[]."""
    ctx = dict(ctx)
    ctx["urgency"] = normalise_urgency(ctx.get("urgency"))
    ctx["reason"] = clean_text(ctx.get("reason"))[:300]
    sbar = await build_sbar(ctx)
    fhir_rec, doc, voice = await asyncio.gather(_fhir(ctx, sbar), _doc(ctx, sbar), _voice(sbar["voice"]))
    mp3, audio_url = voice
    pdf_url = (doc or {}).get("pdfUrl") if (doc or {}).get("ok") else None
    doc_url = (doc or {}).get("docUrl") if (doc or {}).get("ok") else None
    delivered = await _deliver(ctx, sbar, pdf_url, mp3, audio_url)
    res: dict[str, Any] = {
        "ok": True, "specialty": ctx.get("specialty"), "urgency": ctx["urgency"], "label": LABEL,
        "sbar": {k: sbar[k] for k in ("situation", "background", "assessment", "recommendation")},
        "letter": letter_text(ctx, sbar), "voiceScript": sbar["voice"],
        "pdfUrl": pdf_url, "docUrl": doc_url, "audioUrl": audio_url, "fhir": fhir_rec,
    }
    if delivered is not None:
        res["delivered"] = delivered
    return res
