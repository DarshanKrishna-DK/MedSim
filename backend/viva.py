"""Viva voce exam: an oral exam with an AI examiner, for every MBBS year.

Endpoints
  POST /agent/viva/plan   {subject, topic, year, lang, count?, fresh?}
                          -> {questions:[{id, q, keyPoints[], difficulty 1-3, followUps[]}], ...}
  POST /agent/viva/grade  {subject, topic, year?, plan, transcript, lang}
                          -> {score, asked, total, perQuestion[], strengths[], gaps[],
                              recommendedReading[], examinerComment}
  POST /agent/viva/ask    {system, messages, lang} -> {reply}   (typed fallback when voice is off)

The live examiner is an Agora Conversational AI agent started through the normal
/voice/start route with caseId "viva-examiner"; the browser pushes the question
plan into it with /voice/context. This module also owns that agent's Murf voice
and its filler phrases (voice.py asks for them by case id).

The LLM (Sarvam first, Groq fallback, via llm_provider.chat) writes strict JSON.
We validate and normalise here:
- plans are 8-10 questions with key points, a 1-3 difficulty and follow-ups,
  cached in memory by (lang, subject, year, topic, count);
- grading verdicts must quote the student's own words. Quotes that are not in
  the transcript are swapped for the student line they paraphrase, or dropped
  (and a "correct" without evidence becomes "partial");
- the 0-100 score is computed here from the verdicts, weighted by difficulty,
  over the questions actually asked, never taken from the model;
- reading suggestions are standard Indian textbooks by name and chapter only;
  page numbers are stripped.

Educational content for a training simulator, not clinical advice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Literal, Optional, Union

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_provider import chat, strip_think
import lang as langmod

log = logging.getLogger("medsim.viva")
router = APIRouter()

VIVA_CASE_ID = "viva-examiner"
# Senior male examiner voice (Google Gemini voice 'Charon').
EXAMINER_VOICE = "Charon"
GEMINI_EXAMINER_VOICE = "Charon"

# The examiner's "thinking" fillers. The patient pool ("Haan, doctor...")
# would sound wrong from a professor.
EXAMINER_FILLERS: dict[str, list[str]] = {
    "en": ["Hmm...", "Achha...", "Hmm, okay...", "I see...", "Right..."],
    "hi": ["हम्म...", "अच्छा...", "ठीक है...", "हम्म, देखिए..."],
}

MIN_Q, MAX_Q = 8, 10
REPAIR_MIN_Q = 6  # a second, repaired reply may come back a little short
CACHE_MAX = 64
ATTEMPT_TIMEOUT_S = 40.0
TOTAL_TIMEOUT_S = 80.0
VERDICTS = ("correct", "partial", "incorrect", "not asked")
CREDIT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

# Standard Indian MBBS textbooks per subject (names only). Used when the model
# suggests nothing usable, and given to the model as the preferred list.
TEXTBOOKS: dict[str, list[str]] = {
    "anatomy": ["BD Chaurasia's Human Anatomy", "Inderbir Singh's Textbook of Human Histology", "Gray's Anatomy for Students"],
    "physiology": ["Guyton and Hall Textbook of Medical Physiology", "AK Jain, Textbook of Physiology", "GK Pal, Textbook of Medical Physiology"],
    "biochemistry": ["DM Vasudevan, Textbook of Biochemistry for Medical Students", "U Satyanarayana, Biochemistry", "Harper's Illustrated Biochemistry"],
    "pharmacology": ["KD Tripathi, Essentials of Medical Pharmacology", "Goodman and Gilman's The Pharmacological Basis of Therapeutics"],
    "pathology": ["Robbins and Cotran Pathologic Basis of Disease", "Harsh Mohan, Textbook of Pathology"],
    "microbiology": ["Ananthanarayan and Paniker's Textbook of Microbiology", "Apurba Sastry, Essentials of Medical Microbiology"],
    "forensic-medicine": ["KS Narayan Reddy, The Essentials of Forensic Medicine and Toxicology", "Parikh's Textbook of Medical Jurisprudence"],
    "community-medicine": ["Park's Textbook of Preventive and Social Medicine", "Suryakantha, Community Medicine with Recent Advances"],
    "medicine": ["Davidson's Principles and Practice of Medicine", "API Textbook of Medicine", "Harrison's Principles of Internal Medicine"],
    "surgery": ["Bailey and Love's Short Practice of Surgery", "SRB's Manual of Surgery", "Somen Das, A Concise Textbook of Surgery"],
    "obg": ["DC Dutta's Textbook of Obstetrics", "DC Dutta's Textbook of Gynecology", "Shaw's Textbook of Gynaecology"],
    "paediatrics": ["OP Ghai, Essential Pediatrics", "Nelson Textbook of Pediatrics"],
}

_plan_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


# ───────────────────────── voice hooks (used by voice.py) ─────────────────────────


def examiner_voice(lang: Optional[str] = "en") -> str:
    lg = langmod.norm_lang(lang)
    if lg == "en":
        override = os.environ.get("MURF_VIVA_VOICE", "").strip()
        if override:
            return override
    return EXAMINER_VOICE


def examiner_fillers(lang: Optional[str] = "en") -> list[str]:
    lg = langmod.norm_lang(lang)
    return list(EXAMINER_FILLERS.get(lg, ["Hmm...", "Achha..."]))


# ───────────────────────── small text helpers ─────────────────────────


def _s(v: Any, n: int = 200) -> str:
    """Single-line clean string, dashes normalised, capped at n chars."""
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v) if isinstance(v, (int, float)) else ""
    v = v.replace(EM_DASH, ", ").replace(EN_DASH, "-")
    v = re.sub(r"[*_`#]+", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    v = re.sub(r"\s+,", ",", v)
    return v[:n].strip()


def _list(v: Any, max_n: int, n_chars: int) -> list[str]:
    if isinstance(v, str):
        v = [p for p in re.split(r"\n|;", v)]
    if not isinstance(v, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in v:
        if isinstance(item, dict):
            item = item.get("text") or item.get("point") or item.get("q") or ""
        s = _s(item, n_chars).lstrip("-• ").strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
        if len(out) >= max_n:
            break
    return out


def subject_key(subject: str) -> str:
    s = subject.lower()
    if "forensic" in s or "fmt" in s:
        return "forensic-medicine"
    if "community" in s or "psm" in s or "preventive" in s:
        return "community-medicine"
    if "obst" in s or "gyn" in s or s.strip() == "obg":
        return "obg"
    if "paed" in s or "ped" in s:
        return "paediatrics"
    for k in TEXTBOOKS:
        if k in s:
            return k
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def parse_json_object(text: str) -> Any:
    text = strip_think(text or "")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in reply")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e.msg})") from e


# ───────────────────────── plan ─────────────────────────


def _difficulty(v: Any) -> int:
    if isinstance(v, bool):
        return 2
    if isinstance(v, (int, float)):
        return max(1, min(3, int(round(v))))
    if isinstance(v, str):
        t = v.strip().lower()
        if t.isdigit():
            return max(1, min(3, int(t)))
        if t in ("easy", "basic", "low"):
            return 1
        if t in ("hard", "difficult", "advanced", "high"):
            return 3
    return 2


def validate_plan(raw: Any, want: int = MIN_Q, minimum: int = MIN_Q) -> list[dict[str, Any]]:
    """Normalise a model plan to [{id, q, keyPoints, difficulty, followUps}].
    Raises ValueError when fewer than `minimum` usable questions remain."""
    qs = raw.get("questions") if isinstance(raw, dict) else raw
    if not isinstance(qs, list):
        raise ValueError("questions must be a list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in qs:
        if not isinstance(item, dict):
            continue
        q = _s(item.get("q") or item.get("question"), 240)
        q = re.sub(r"^(q(uestion)?\s*\d+[:.)]\s*|\d+[:.)]\s*)", "", q, flags=re.I).strip()
        key = re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()
        if not q or key in seen:
            continue
        points = _list(item.get("keyPoints") or item.get("key_points") or item.get("answer"), 6, 140)
        if not points:
            continue
        seen.add(key)
        out.append({
            "id": "",
            "q": q,
            "keyPoints": points,
            "difficulty": _difficulty(item.get("difficulty")),
            "followUps": _list(item.get("followUps") or item.get("follow_ups"), 3, 160),
        })
    if len(out) < minimum:
        raise ValueError(f"only {len(out)} usable questions (need {minimum})")
    out = out[:want]
    for i, q in enumerate(out):
        q["id"] = f"q{i + 1}"
    return out


class PlanRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=60)
    topic: str = Field(min_length=1, max_length=120)
    year: str = Field(default="2nd year MBBS", max_length=40)
    lang: Optional[str] = None
    count: int = Field(default=MIN_Q, ge=MIN_Q, le=MAX_Q)
    fresh: bool = False


PLAN_SYSTEM = (
    "You are a senior professor of {subject} at an Indian medical college, setting the questions for a "
    "university viva voce (oral exam) of a {year} student on the topic \"{topic}\". Plan exactly {n} questions. "
    "Reply with ONE JSON object only, no markdown, no prose outside JSON:\n"
    '{{"questions": [{{"id": "q1", "q": str, "keyPoints": [2 to 5 short strings], "difficulty": 1 | 2 | 3, '
    '"followUps": [1 or 2 short probing questions]}}]}}\n'
    "Rules:\n"
    "- Classic Indian viva style: short, direct questions a student answers aloud in under 30 seconds, for example "
    "\"What is ...?\", \"Classify ...\", \"Name two ...\", \"What is the mechanism of ...?\", \"Why ...?\", "
    "\"What happens if ...?\". One idea per question. Do not number the questions inside q.\n"
    "- Order: the first 2 questions are difficulty 1 (definitions, classification, basic facts), the middle ones "
    "difficulty 2 (mechanisms, applied physiology or pharmacology, clinical correlation), the last 2 difficulty 3 "
    "(applied reasoning, a short clinical scenario, or a 'why' question).\n"
    "- keyPoints are what the examiner expects to hear, at standard textbook level for the year "
    "(CBME syllabus of the National Medical Commission), each under 15 words.\n"
    "- followUps are the examiner's probes after an answer, for example \"Why?\", \"And the mechanism?\", "
    "\"Name two side effects.\", \"Any exceptions?\".\n"
    "- Stay strictly inside the topic. Facts must be standard and uncontroversial.\n"
    "- Write q, keyPoints and followUps in English (the examiner translates when speaking)."
)


def plan_messages(req: PlanRequest) -> list[dict[str, Any]]:
    sys_msg = PLAN_SYSTEM.format(subject=_s(req.subject, 60), year=_s(req.year, 40) or "MBBS",
                                 topic=_s(req.topic, 120), n=req.count)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Subject: {_s(req.subject, 60)}\nTopic: {_s(req.topic, 120)}\n"
                                    f"Year: {_s(req.year, 40)}\nWrite the {req.count}-question viva plan JSON now."},
    ]


def plan_cache_key(req: PlanRequest) -> str:
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 +-]", "", s.lower())).strip()  # noqa: E731
    return f"{langmod.norm_lang(req.lang)}|{norm(req.subject)}|{norm(req.year)}|{norm(req.topic)}|{req.count}"


async def _generate_plan(req: PlanRequest) -> list[dict[str, Any]]:
    messages = plan_messages(req)
    last_err = ""
    async with httpx.AsyncClient(timeout=ATTEMPT_TIMEOUT_S) as client:
        for attempt in range(2):
            msg, provider = await asyncio.wait_for(
                chat(
                    messages,
                    json_mode=True,
                    max_tokens=2400,
                    temperature=0.5,
                    client=client,
                    tier="reasoning",
                    effort="medium",
                ),
                timeout=ATTEMPT_TIMEOUT_S * 1.5,
            )
            content = msg.get("content") or ""
            try:
                return validate_plan(parse_json_object(content), req.count, req.count if attempt == 0 else REPAIR_MIN_Q)
            except ValueError as e:
                last_err = str(e)
                log.warning("viva plan attempt %d via %s invalid: %s", attempt + 1, provider, last_err)
                messages = messages + [
                    {"role": "assistant", "content": content[:6000]},
                    {"role": "user", "content": f"That JSON was not usable: {last_err}. Reply again with the complete "
                     f"corrected JSON object only, exactly {req.count} questions, each with keyPoints."},
                ]
    raise HTTPException(status_code=502, detail=f"Viva plan model returned malformed JSON: {last_err}"[:300])


@router.post("/agent/viva/plan")
async def viva_plan(req: PlanRequest):
    if not _s(req.topic) or not _s(req.subject):
        raise HTTPException(status_code=422, detail="subject and topic are required")
    key = plan_cache_key(req)
    hit = None if req.fresh else _plan_cache.get(key)
    meta = {"subject": _s(req.subject, 60), "topic": _s(req.topic, 120), "year": _s(req.year, 40),
            "lang": langmod.norm_lang(req.lang)}
    if hit is not None:
        _plan_cache.move_to_end(key)
        return {**meta, "questions": hit["questions"], "cached": True}
    try:
        questions = await asyncio.wait_for(_generate_plan(req), timeout=TOTAL_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Viva plan generation timed out")
    _plan_cache[key] = {"questions": questions}
    while len(_plan_cache) > CACHE_MAX:
        _plan_cache.popitem(last=False)
    return {**meta, "questions": questions, "cached": False}


# ───────────────────────── grading ─────────────────────────


class TranscriptLine(BaseModel):
    role: str = Field(max_length=20)
    text: str = Field(default="", max_length=2000)


class GradeRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=60)
    topic: str = Field(min_length=1, max_length=120)
    year: Optional[str] = Field(default=None, max_length=40)
    plan: Union[list[dict[str, Any]], dict[str, Any]]
    transcript: list[TranscriptLine] = Field(default_factory=list, max_length=200)
    lang: Optional[str] = None


_INDIC = "ऀ-ॿঀ-৿஀-௿ఀ-౿ಀ-೿"
_NON_WORD = re.compile(rf"[^0-9a-z{_INDIC}]+")


def norm_text(s: str) -> str:
    return _NON_WORD.sub(" ", (s or "").lower()).strip()


def _role(r: str) -> str:
    r = (r or "").strip().lower()
    if r in ("student", "user", "you", "candidate"):
        return "student"
    return "examiner"


def student_lines(transcript: list[TranscriptLine]) -> list[str]:
    return [t.text.strip() for t in transcript if _role(t.role) == "student" and t.text.strip()]


def verify_quote(quote: str, lines: list[str]) -> str:
    """The student's words backing a verdict, or "" when the quote cannot be
    found. An exact (normalised) substring keeps the quote; a close paraphrase
    of one line is replaced by that line, verbatim."""
    nq = norm_text(quote)
    if not nq:
        return ""
    for line in lines:
        if nq in norm_text(line):
            return _s(quote, 240)
    toks = [t for t in nq.split() if len(t) > 1]
    if len(toks) < 2:
        return ""
    best, best_line = 0.0, ""
    for line in lines:
        lt = set(norm_text(line).split())
        frac = sum(1 for t in toks if t in lt) / len(toks)
        if frac > best:
            best, best_line = frac, line
    return _s(best_line, 240) if best >= 0.75 else ""


def _verdict(v: Any) -> str:
    t = re.sub(r"[^a-z]+", " ", str(v or "").lower()).strip()
    if t in ("correct", "right", "met", "full"):
        return "correct"
    if t in ("partial", "partially correct", "partly correct", "partly"):
        return "partial"
    if t in ("incorrect", "wrong", "not met", "no answer", "unanswered", "dont know"):
        return "incorrect"
    if t in ("not asked", "notasked", "unasked", "skipped"):
        return "not asked"
    return "not asked"


_PAGE_RE = re.compile(r"[,;(]?\s*\b(?:pp?|pg|pages?)\.?\s*\d+(?:\s*[-,]\s*\d+)*\)?", re.I)


def clean_reading(items: Any, subject: str) -> list[str]:
    out: list[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                book = _s(it.get("book") or it.get("title"), 100)
                ch = _s(it.get("chapter") or it.get("section"), 100)
                it = f"{book}: {ch}" if book and ch else book
            s = _PAGE_RE.sub("", _s(it, 180)).strip(" ,;:")
            if s and s.lower() not in {o.lower() for o in out}:
                out.append(s)
            if len(out) >= 4:
                break
    if not out:
        out = TEXTBOOKS.get(subject_key(subject), [])[:2]
    return out


def score_items(items: list[dict[str, Any]], plan: list[dict[str, Any]]) -> tuple[int, int]:
    """(score 0-100, asked count). Difficulty-weighted credit over asked questions."""
    weight = {q["id"]: q.get("difficulty", 2) for q in plan}
    num = den = 0.0
    asked = 0
    for it in items:
        if it["verdict"] == "not asked":
            continue
        asked += 1
        w = float(weight.get(it["id"], 2))
        den += w
        num += w * CREDIT[it["verdict"]]
    return (round(100 * num / den) if den else 0), asked


def assemble_grade(req: GradeRequest, plan: list[dict[str, Any]], raw: dict[str, Any]) -> dict[str, Any]:
    lines = student_lines(req.transcript)
    by_id: dict[str, dict[str, Any]] = {}
    for it in raw.get("perQuestion") or raw.get("per_question") or []:
        if isinstance(it, dict) and str(it.get("id", "")).strip():
            by_id.setdefault(str(it["id"]).strip(), it)
    items: list[dict[str, Any]] = []
    for q in plan:
        m = by_id.get(q["id"], {})
        verdict = _verdict(m.get("verdict"))
        said = verify_quote(_s(m.get("said") or m.get("quote"), 400), lines) if verdict != "not asked" else ""
        note = _s(m.get("note"), 240)
        if verdict == "correct" and not said:
            verdict = "partial"
            note = (note + " " if note else "") + "(No verbatim answer found in the transcript.)"
        items.append({
            "id": q["id"],
            "q": q["q"],
            "difficulty": q.get("difficulty", 2),
            "verdict": verdict,
            "expected": _s(m.get("expected"), 300) or "; ".join(q["keyPoints"]),
            "said": said,
            "note": note,
        })
    score, asked = score_items(items, plan)
    return {
        "score": score,
        "asked": asked,
        "total": len(plan),
        "perQuestion": items,
        "strengths": _list(raw.get("strengths"), 5, 200),
        "gaps": _list(raw.get("gaps"), 5, 200),
        "recommendedReading": clean_reading(raw.get("recommendedReading") or raw.get("reading"), req.subject),
        "examinerComment": _s(raw.get("examinerComment") or raw.get("comment"), 400),
    }


def no_answer_result(req: GradeRequest, plan: list[dict[str, Any]]) -> dict[str, Any]:
    items = [{"id": q["id"], "q": q["q"], "difficulty": q.get("difficulty", 2), "verdict": "not asked",
              "expected": "; ".join(q["keyPoints"]), "said": "", "note": ""} for q in plan]
    return {
        "score": 0, "asked": 0, "total": len(plan), "perQuestion": items, "strengths": [],
        "gaps": ["No answers were recorded, so nothing could be marked."],
        "recommendedReading": clean_reading([], req.subject),
        "examinerComment": "The viva ended before you answered any question. Take it again when you are ready.",
    }


GRADE_SYSTEM = (
    "You are the external examiner who just conducted an MBBS university viva voce in {subject} on \"{topic}\". "
    "Mark it from the transcript. Reply with ONE JSON object only, no markdown:\n"
    '{{"perQuestion": [{{"id": "q1", "verdict": "correct" | "partial" | "incorrect" | "not asked", '
    '"expected": str (the model answer in one short sentence), "said": str (the student\'s own words, copied '
    'VERBATIM from a STUDENT line, or "" if none), "note": str (one short sentence why)}}], '
    '"strengths": [up to 3 short strings], "gaps": [up to 4 short strings, specific concepts to revise], '
    '"recommendedReading": [up to 3 strings, "Book name: chapter or section name"], "examinerComment": str (2 sentences)}}\n'
    "Rules:\n"
    "- One perQuestion entry for every plan question, same ids, in plan order.\n"
    "- \"not asked\" when the examiner never put that question (or a translation of it) to the student.\n"
    "- \"correct\" needs most key points; \"partial\" some; \"incorrect\" wrong, \"I don't know\" or silence after it was asked.\n"
    "- Be fair to spoken answers: speech-to-text errors and Indian English or code-mixed phrasing are fine. Accept "
    "answers in any language. Credit a hinted answer as at most partial.\n"
    "- said MUST be copied exactly from the transcript; never invent or polish it.\n"
    "- recommendedReading: standard Indian MBBS textbooks by name and chapter or section name only, for example "
    "{books}. NEVER give page numbers or editions.\n"
    "- No em dashes."
)


def grade_messages(req: GradeRequest, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lg = langmod.norm_lang(req.lang)
    books = ", ".join(TEXTBOOKS.get(subject_key(req.subject), ["KD Tripathi", "Guyton", "BD Chaurasia", "Robbins"])[:3])
    system = GRADE_SYSTEM.format(subject=_s(req.subject, 60), topic=_s(req.topic, 120), books=books)
    if lg != "en":
        name = langmod.LANG_NAMES.get(lg, "English")
        system += (f"\nThe viva was spoken in {name}. Write note, strengths, gaps and examinerComment in English "
                   "so the result screen stays readable; said stays exactly as transcribed.")
    convo = []
    for t in req.transcript[-160:]:
        text = _s(t.text, 600)
        if text:
            convo.append(f"{'STUDENT' if _role(t.role) == 'student' else 'EXAMINER'}: {text}")
    payload = {
        "plan": [{"id": q["id"], "q": q["q"], "keyPoints": q["keyPoints"], "difficulty": q.get("difficulty", 2)}
                 for q in plan],
        "year": _s(req.year, 40),
    }
    user = "PLAN:\n" + json.dumps(payload, ensure_ascii=False) + "\n\nTRANSCRIPT:\n" + "\n".join(convo)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@router.post("/agent/viva/grade")
async def viva_grade(req: GradeRequest):
    try:
        plan = validate_plan(req.plan, MAX_Q, 1)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"plan is not usable: {e}")
    if not student_lines(req.transcript):
        return no_answer_result(req, plan)
    msg, _ = await asyncio.wait_for(
        chat(
            grade_messages(req, plan),
            json_mode=True,
            max_tokens=2600,
            temperature=0.1,
            tier="reasoning",
            effort="medium",
        ),
        timeout=ATTEMPT_TIMEOUT_S * 1.5,
    )
    try:
        raw = parse_json_object(msg.get("content") or "")
        if not isinstance(raw, dict) or not isinstance(raw.get("perQuestion") or raw.get("per_question"), list):
            raise ValueError("no perQuestion list")
    except ValueError:
        log.warning("viva grade: malformed model JSON")
        raise HTTPException(status_code=502, detail="Viva marker returned malformed JSON")
    return assemble_grade(req, plan, raw)


# ───────────────────────── typed fallback ─────────────────────────


class AskMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    system: str = Field(min_length=1, max_length=24000)
    lang: Optional[str] = None
    messages: list[AskMessage] = Field(min_length=1, max_length=60)


def clean_reply(text: str) -> str:
    text = strip_think(text or "")
    text = re.sub(r"[*_`#]+", "", text)
    text = text.replace(EM_DASH, ", ").replace(EN_DASH, "-")
    return re.sub(r"\s+", " ", text).strip()[:600]


@router.post("/agent/viva/ask")
async def viva_ask(req: AskRequest):
    """One examiner turn when the live voice agent is not connected. The browser
    sends the same persona + plan prompt the voice agent gets (language rules
    are already inside it)."""
    msgs = [{"role": "system", "content": req.system}] + [m.model_dump() for m in req.messages[-24:]]
    msg, _ = await asyncio.wait_for(
        chat(msgs, max_tokens=220, temperature=0.5, tier="reasoning", effort="medium"),
        timeout=ATTEMPT_TIMEOUT_S * 1.5,
    )
    text = clean_reply(msg.get("content") or "")
    if not text:
        raise HTTPException(status_code=502, detail="Examiner model returned an empty reply")
    return {"reply": text}
