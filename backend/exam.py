"""Physical examination: POST /agent/exam/perform.

The trainee examines one body system (by voice or a click). Findings come from
the OpenAI-compatible provider powered by Gemini in JSON mode at
low temperature, are validated and reconciled against the documented vitals,
and are cached per (caseId, system) so a repeat exam of the same patient gives
the same findings. The "vitals" system is deterministic and never calls the
LLM. Findings stay in English (chart notes in Indian hospitals are written in
English); `lang` is accepted for symmetry with the other routes.

Synthetic training data, not a claim of clinical accuracy. Keys are never
logged or returned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_provider import chat, strip_think

log = logging.getLogger("medsim.exam")
router = APIRouter()

EXAM_TIMEOUT_S = 15.0
CACHE_MAX = 512
EM_DASH = chr(0x2014)

SYSTEMS = (
    "general", "vitals", "cardio", "resp", "abdomen", "neuro", "msk",
    "skin", "ent", "eye", "obgyn", "paeds",
)

SYSTEM_ALIASES = {
    "cvs": "cardio", "cardiac": "cardio", "cardiovascular": "cardio", "heart": "cardio",
    "rs": "resp", "respiratory": "resp", "chest": "resp", "lungs": "resp", "lung": "resp",
    "abdominal": "abdomen", "abdo": "abdomen", "gi": "abdomen", "pa": "abdomen", "per abdomen": "abdomen",
    "cns": "neuro", "neurological": "neuro", "nervous": "neuro",
    "musculoskeletal": "msk", "joints": "msk", "locomotor": "msk",
    "derm": "skin", "dermatological": "skin",
    "throat": "ent", "ear": "ent", "nose": "ent",
    "eyes": "eye", "ophthalmic": "eye",
    "gynae": "obgyn", "obstetric": "obgyn", "gyn": "obgyn", "pv": "obgyn",
    "paediatric": "paeds", "pediatric": "paeds", "peds": "paeds",
    "ge": "general", "general examination": "general",
    "vital signs": "vitals", "vital": "vitals",
}

SYSTEM_LABELS = {
    "general": "general examination",
    "vitals": "vital signs",
    "cardio": "cardiovascular system",
    "resp": "respiratory system",
    "abdomen": "abdomen",
    "neuro": "nervous system",
    "msk": "musculoskeletal system",
    "skin": "skin",
    "ent": "ear, nose and throat",
    "eye": "eyes",
    "obgyn": "obstetric and gynaecological examination",
    "paeds": "paediatric examination (growth, hydration, development)",
}

HEART_SOUNDS = ("normal-heart", "murmur-systolic", "murmur-diastolic", "s3", "pericardial-rub")
LUNG_SOUNDS = ("normal-breath", "wheeze", "crackles", "bronchial", "absent")
SOUNDS = HEART_SOUNDS + LUNG_SOUNDS


def norm_system(value: Any) -> Optional[str]:
    v = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if v in SYSTEMS:
        return v
    return SYSTEM_ALIASES.get(v)


# ── request / response models ──────────────────────────────────────────────


class QA(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = ""
    answer: str = ""


class TestRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    testId: str = ""
    name: str = ""
    result: str = ""
    abnormal: bool = False


class ExamPatient(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    age: Optional[float] = None
    gender: str = ""
    chiefComplaint: str = ""
    anamnesis: list[QA] = Field(default_factory=list)
    vitals: dict[str, Any] = Field(default_factory=dict)
    testResults: list[TestRow] = Field(default_factory=list)


class ExamRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    caseId: str = ""
    system: str
    maneuver: Optional[str] = None
    patient: ExamPatient = Field(default_factory=ExamPatient)
    correctDiagnosis: str = ""
    lang: Optional[str] = None

    @field_validator("system")
    @classmethod
    def _valid_system(cls, v: str) -> str:
        s = norm_system(v)
        if not s:
            raise ValueError(f"system must be one of {', '.join(SYSTEMS)}")
        return s


class Finding(BaseModel):
    label: str
    value: str
    abnormal: bool = False


class ExamResponse(BaseModel):
    system: str
    findings: list[Finding]
    summary: str
    sound: Optional[str] = None


# ── text helpers ───────────────────────────────────────────────────────────


def _clean(text: Any) -> str:
    s = re.sub(r"\s*" + EM_DASH + r"\s*", ", ", str(text or "")).replace(chr(0x2013), "-")
    return re.sub(r"\s+", " ", s).strip()


def _cap(text: Any, chars: int) -> str:
    s = _clean(text)
    return s if len(s) <= chars else s[: chars - 3].rstrip(" ,;:") + "..."


def _sentences(text: Any, n: int = 2, words: int = 50) -> str:
    s = _clean(text)
    parts = re.split(r"(?<=[.!?])\s+", s)
    out = " ".join(parts[:n]).strip()
    w = out.split(" ")
    if len(w) > words:
        out = " ".join(w[:words]).rstrip(",;:") + "."
    return out


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v or "").strip().lower() in ("true", "yes", "1", "abnormal", "y")


def parse_model_json(text: str) -> Optional[dict[str, Any]]:
    text = strip_think(text or "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return v if isinstance(v, dict) else None


# ── vitals (deterministic) ─────────────────────────────────────────────────


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN guard


def _bp(v: Any) -> tuple[Optional[float], Optional[float]]:
    m = re.match(r"\s*(\d{2,3})\s*/\s*(\d{2,3})", str(v or ""))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def _ranges(age: Optional[float]) -> dict[str, tuple[float, float]]:
    """Normal (low, high) by age band. Rough textbook ranges, synthetic sim."""
    a = age if age is not None else 30
    if a < 1:
        return {"hr": (100, 160), "rr": (30, 60)}
    if a < 5:
        return {"hr": (80, 140), "rr": (22, 40)}
    if a < 12:
        return {"hr": (70, 120), "rr": (18, 30)}
    return {"hr": (60, 100), "rr": (12, 20)}


def vitals_findings(vitals: dict[str, Any], age: Optional[float]) -> list[Finding]:
    rng = _ranges(age)
    out: list[Finding] = []
    hr = _num(vitals.get("hr"))
    if hr is not None:
        lo, hi = rng["hr"]
        out.append(Finding(label="Pulse", value=f"{hr:g}/min, regular", abnormal=not (lo <= hr <= hi)))
    sbp, dbp = _bp(vitals.get("bp"))
    if sbp is not None and dbp is not None:
        out.append(Finding(label="Blood pressure", value=f"{sbp:g}/{dbp:g} mmHg",
                           abnormal=sbp >= 140 or dbp >= 90 or sbp < 90))
    rr = _num(vitals.get("rr"))
    if rr is not None:
        lo, hi = rng["rr"]
        out.append(Finding(label="Respiratory rate", value=f"{rr:g}/min", abnormal=not (lo <= rr <= hi)))
    spo2 = _num(vitals.get("spo2"))
    if spo2 is not None:
        out.append(Finding(label="SpO2", value=f"{spo2:g}% on room air", abnormal=spo2 < 94))
    temp = _num(vitals.get("temp"))
    if temp is not None:
        out.append(Finding(label="Temperature", value=f"{temp:g} °C", abnormal=temp >= 38.0 or temp < 35.5))
    return out


def vitals_summary(findings: list[Finding]) -> str:
    if not findings:
        return "No vital signs recorded."
    abn = [f.label.lower() for f in findings if f.abnormal]
    vals = ", ".join(f"{f.label} {f.value.split(',')[0]}" for f in findings)
    if not abn:
        return f"Vitals within normal limits: {vals}."
    return f"Vitals: {vals}. Abnormal: {', '.join(abn)}."


# Labels the model may echo; their values must match the documented vitals.
_VITAL_LABELS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(pulse|heart rate|hr)\b", re.I), "Pulse"),
    (re.compile(r"\b(blood pressure|bp)\b", re.I), "Blood pressure"),
    (re.compile(r"\b(respiratory rate|resp(iratory)? rate|rr)\b", re.I), "Respiratory rate"),
    (re.compile(r"\b(spo2|oxygen saturation|saturation|sats)\b", re.I), "SpO2"),
    (re.compile(r"\b(temperature|temp)\b", re.I), "Temperature"),
]


def reconcile_vitals(findings: list[Finding], vitals: dict[str, Any], age: Optional[float]) -> list[Finding]:
    """Never contradict the chart: a finding that restates a vital sign gets the
    documented value and abnormal flag (pulse rhythm/volume wording is kept)."""
    truth = {f.label: f for f in vitals_findings(vitals, age)}
    out: list[Finding] = []
    for f in findings:
        hit = next((name for rx, name in _VITAL_LABELS if rx.search(f.label)), None)
        t = truth.get(hit) if hit else None
        if t is None:
            out.append(f)
            continue
        value = t.value
        if hit == "Pulse":
            extra = re.sub(r"^[\d.\s]*(/\s*min|bpm|per minute)?[,;\s]*", "", f.value, flags=re.I).strip()
            extra = re.sub(r"\b\d{2,3}\s*(/\s*min|bpm|per minute)\b", "", extra, flags=re.I).strip(" ,;")
            if extra:
                value = f"{t.value.split(',')[0]}, {extra}"
        out.append(Finding(label=f.label, value=value, abnormal=t.abnormal or (hit == "Pulse" and f.abnormal)))
    return out


# ── sound selection ────────────────────────────────────────────────────────

_SOUND_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("pericardial-rub", re.compile(r"\b(pericardial )?(friction )?rub\b", re.I)),
    ("s3", re.compile(r"\b(s3|third heart sound|gallop)\b", re.I)),
    ("murmur-diastolic", re.compile(r"\bdiastolic\b.*\bmurmur\b|\bmurmur\b.*\bdiastolic\b|\bmid[- ]diastolic\b|\bearly diastolic\b", re.I)),
    ("murmur-systolic", re.compile(r"\b(pan|holo|ejection )?systolic\b.*\bmurmur\b|\bmurmur\b.*\bsystolic\b|\bmurmur\b", re.I)),
    ("absent", re.compile(r"\b(absent breath sounds|breath sounds absent|silent chest|no air entry)\b", re.I)),
    ("bronchial", re.compile(r"\bbronchial (breath(ing| sounds)?)\b", re.I)),
    ("crackles", re.compile(r"\b(crackles|crepitations|creps|rales|crepts)\b", re.I)),
    ("wheeze", re.compile(r"\b(wheeze|wheezes|wheezing|rhonchi)\b", re.I)),
]

_NEG_BEFORE = re.compile(r"\b(no|nil|without|absence of)\b[^,.;]*$", re.I)


def _negated(text: str, start: int) -> bool:
    """'No murmurs', 'no added sounds such as crackles': the term is negated."""
    return bool(_NEG_BEFORE.search(text[max(0, start - 40):start]))


def infer_sound(system: str, findings: list[Finding]) -> Optional[str]:
    """Pick the auscultation clip from the abnormal findings text, so the sound
    always agrees with what is written."""
    allowed = HEART_SOUNDS if system == "cardio" else LUNG_SOUNDS if system == "resp" else ()
    if not allowed:
        return None
    for f in findings:
        if not f.abnormal:
            continue
        text = f"{f.label}: {f.value}"
        for sound, rx in _SOUND_RULES:
            if sound not in allowed:
                continue
            m = rx.search(text)
            if m and (sound == "absent" or not _negated(text, m.start())):
                return sound
    return None


def choose_sound(system: str, model_sound: Any, findings: list[Finding]) -> Optional[str]:
    inferred = infer_sound(system, findings)
    if inferred:
        return inferred
    if system == "cardio":
        s = str(model_sound or "").strip().lower()
        abnormal_heart = any(f.abnormal and re.search(r"murmur|s3|gallop|rub", f.value, re.I) for f in findings)
        return s if s in HEART_SOUNDS and (s == "normal-heart" or abnormal_heart) else "normal-heart"
    if system == "resp":
        s = str(model_sound or "").strip().lower()
        abnormal_lung = any(f.abnormal and re.search(r"breath|crack|crep|wheez|rhonch|air entry", f.value, re.I) for f in findings)
        return s if s in LUNG_SOUNDS and (s == "normal-breath" or abnormal_lung) else "normal-breath"
    return None


# ── LLM prompt + coercion ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are the examination engine of a SYNTHETIC clinical training simulator for Indian MBBS students. "
    "The trainee doctor is physically examining the patient. Return the findings the trainee would elicit "
    "on examining ONE body system.\n"
    "Rules:\n"
    "1. Findings MUST be consistent with the case's correct diagnosis and must never contradict the documented "
    "vitals, history or test results. Use textbook-accurate, plausible signs (for example lobar pneumonia: "
    "dull percussion note, bronchial breathing and coarse crackles over the affected base, increased vocal "
    "resonance; acute appendicitis: tenderness and guarding in the right iliac fossa, rebound tenderness).\n"
    "2. If this system is not involved by the diagnosis, report it as normal with standard negative findings.\n"
    "3. Use 3 to 7 findings in the order of inspection, palpation, percussion, auscultation where relevant. "
    "Each finding: short label (max 5 words) and value (max 20 words), abnormal true or false.\n"
    "4. summary: 1 to 2 short sentences, the way the examining doctor would note it in the case sheet "
    "(for example: \"Chest: dull note and bronchial breathing at the right base with coarse crackles.\").\n"
    "5. sound: for the cardiovascular system one of normal-heart, murmur-systolic, murmur-diastolic, s3, "
    "pericardial-rub; for the respiratory system one of normal-breath, wheeze, crackles, bronchial, absent; "
    "otherwise null.\n"
    "6. Never name the diagnosis in the findings or the summary. Plain English medical terms, no markdown, "
    "no em dashes.\n"
    "Reply with JSON only: {\"findings\":[{\"label\":str,\"value\":str,\"abnormal\":bool}],"
    "\"summary\":str,\"sound\":str|null}"
)


def _user_prompt(req: ExamRequest) -> str:
    p = req.patient
    return json.dumps(
        {
            "system": req.system,
            "systemName": SYSTEM_LABELS[req.system],
            "maneuver": _cap(req.maneuver, 200) or None,
            "correctDiagnosis": _cap(req.correctDiagnosis, 200) or "unknown",
            "patient": {
                "name": p.name, "age": p.age, "gender": p.gender,
                "chiefComplaint": p.chiefComplaint,
                "vitals": p.vitals,
                "history": [{"q": _cap(qa.question, 200), "a": _cap(qa.answer, 400)} for qa in p.anamnesis][:30],
                "testResults": [
                    {"test": t.name or t.testId, "result": _cap(t.result, 400), "abnormal": t.abnormal}
                    for t in p.testResults
                ][:20],
            },
        },
        default=str,
        ensure_ascii=False,
    )[:14000]


_DISEASE_WORD = re.compile(
    r"^[a-z]{3,}(itis|osis|oma|emia|aemia|pathy|ectasis)$|^(pneumonia|asthma|infarction|copd|tuberculosis)$"
)


def _diagnosis_words(dx: str) -> list[str]:
    """The full diagnosis label plus its disease-name words (pneumonia,
    appendicitis, ...). Ordinary anatomy words (heart, chest) are kept."""
    full = _clean(dx).lower()
    words = [w for w in re.findall(r"[a-z]+", full) if _DISEASE_WORD.match(w)]
    return ([full] if len(full) >= 5 else []) + words


def _redact_dx(text: str, dx_words: list[str]) -> str:
    """The exam must not give away the diagnosis by name."""
    out = text
    for w in dx_words:
        out = re.sub(rf"\b{re.escape(w)}\b", "this condition", out, flags=re.I)
    return out


def coerce_exam(raw: Optional[dict[str, Any]], req: ExamRequest) -> Optional[ExamResponse]:
    if not isinstance(raw, dict):
        return None
    items = raw.get("findings")
    if not isinstance(items, list):
        return None
    dx_words = _diagnosis_words(req.correctDiagnosis)
    findings: list[Finding] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        label = _cap(it.get("label"), 48)
        value = _cap(_redact_dx(_clean(it.get("value")), dx_words), 180)
        if not label or not value:
            continue
        findings.append(Finding(label=label, value=value, abnormal=_as_bool(it.get("abnormal"))))
        if len(findings) >= 8:
            break
    if not findings:
        return None
    findings = reconcile_vitals(findings, req.patient.vitals, req.patient.age)
    summary = _sentences(_redact_dx(_clean(raw.get("summary")), dx_words), 2, 50)
    if not summary:
        abn = [f"{f.label.lower()} {f.value}" for f in findings if f.abnormal]
        name = SYSTEM_LABELS[req.system].capitalize()
        summary = f"{name}: {'; '.join(abn)}." if abn else f"{name}: no abnormality detected."
        summary = _sentences(summary, 2, 50)
    return ExamResponse(
        system=req.system,
        findings=findings,
        summary=summary,
        sound=choose_sound(req.system, raw.get("sound"), findings),
    )


# ── cache ──────────────────────────────────────────────────────────────────

_cache: "OrderedDict[tuple[str, str], ExamResponse]" = OrderedDict()
_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _cache_get(key: tuple[str, str]) -> Optional[ExamResponse]:
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
    return hit


def _cache_put(key: tuple[str, str], value: ExamResponse) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


def clear_cache() -> None:
    _cache.clear()
    _locks.clear()


async def _llm(req: ExamRequest) -> Optional[str]:
    try:
        msg, _ = await asyncio.wait_for(
            chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _user_prompt(req)}],
                json_mode=True,
                max_tokens=700,
                temperature=0.1,
                tier="fast",
                effort="low",
            ),
            timeout=EXAM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.warning("exam LLM timed out after %.0fs", EXAM_TIMEOUT_S)
        return None
    except HTTPException as e:
        log.warning("exam LLM unavailable (HTTP %s)", e.status_code)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("exam LLM failed: %s", type(e).__name__)
        return None
    return (msg or {}).get("content") or ""


@router.post("/agent/exam/perform", response_model=ExamResponse)
async def exam_perform(req: ExamRequest) -> ExamResponse:
    if req.system == "vitals":
        f = vitals_findings(req.patient.vitals, req.patient.age)
        return ExamResponse(system="vitals", findings=f, summary=vitals_summary(f), sound=None)

    key = (req.caseId.strip(), req.system)
    use_cache = bool(key[0])
    if use_cache and (hit := _cache_get(key)):
        return hit
    lock = _locks.setdefault(key, asyncio.Lock()) if use_cache else asyncio.Lock()
    async with lock:
        if use_cache and (hit := _cache_get(key)):
            return hit  # a concurrent identical request (voice + click) already filled it
        res: Optional[ExamResponse] = None
        for _attempt in range(2):  # one retry on malformed JSON
            content = await _llm(req)
            if content is None:
                break
            res = coerce_exam(parse_model_json(content), req)
            if res:
                break
        if not res:
            raise HTTPException(status_code=502, detail="Examination findings unavailable, try again.")
        if use_cache:
            _cache_put(key, res)
        return res
