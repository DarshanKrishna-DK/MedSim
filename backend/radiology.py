"""Radiology reading room: grade a student's image interpretation.

    POST /agent/radiology/grade  {caseId, description?, transcript?[], diagnosis, lang?}
        -> {score, foundFindings[], missedFindings[], missedCritical[], diagnosisCorrect,
            correctDiagnosis, feedback, gradedBy}
    POST /agent/radiology/probe  {caseId, description?, transcript?[], lang?}
        -> {question, gradedBy}

The answer key (findings with keywords, diagnosis, distractors) lives in
src/data/radiologyCases.json, the same file the frontend renders, so there is
one source of truth. Grading asks the LLM powered by Gemini (via
llm_provider.chat) for strict JSON and validates it against the key. If the
model is unavailable, slow or malformed, a deterministic keyword grader takes
over, so the endpoint always answers. Synthetic training content only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_provider import chat, strip_think
import lang as langmod

log = logging.getLogger("medsim.radiology")

router = APIRouter()

CASES_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "radiologyCases.json"
LLM_TIMEOUT_S = 25.0
FINDINGS_WEIGHT = 60
DIAGNOSIS_WEIGHT = 40

CONSULTANT = "Dr. Vikram Rao"


@lru_cache(maxsize=1)
def load_cases() -> dict[str, dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["cases"]}


def get_case(case_id: str) -> dict[str, Any]:
    case = load_cases().get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown radiology case: {case_id[:64]}")
    return case


# ───────────────────────── text helpers ─────────────────────────

_NEG = {"no", "not", "without", "absent", "nil", "negative", "none", "nor", "rule", "exclude"}
_CLAUSE = "|"
# Filler words dropped from both the text and the keywords ("angles are clear" == "angles clear").
_STOP = {"is", "are", "the", "a", "an", "seen", "there"}


def norm(text: Any) -> str:
    """Lower-case words; clause punctuation becomes a '|' token so negation never
    leaks across a comma or full stop ("no effusion, consolidation right base")."""
    t = str(text or "").lower()
    t = t.replace("\u2019", "'")
    t = re.sub(r"[,.;:!?\n]+|\bbut\b|\bhowever\b", f" {_CLAUSE} ", t)
    t = re.sub(r"[^a-z0-9'|\u0900-\u0dff]+", " ", t)
    return " ".join(w for w in t.split() if w not in _STOP)


def clean(text: Any, limit: int = 600) -> str:
    """Plain spoken-style text: no markdown, no em or en dashes."""
    t = strip_think(str(text or ""))
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"\s*[\u2014\u2013]\s*", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit].rstrip()


def keyword_hit(text_n: str, keyword: str) -> bool:
    """True if `keyword` occurs in normalised text and is not negated just before it.
    Keywords that are themselves negative ("no midline shift") skip the negation check."""
    kw = norm(keyword)
    if not kw:
        return False
    kw_is_negative = bool(set(kw.split()) & _NEG)
    for m in re.finditer(r"(?:^|\s)" + re.escape(kw) + r"(?=\s|$)", text_n):
        if kw_is_negative:
            return True
        before = text_n[: m.start()].split(_CLAUSE)[-1].split()[-3:]
        if not (set(before) & _NEG):
            return True
    return False


def student_text(description: str, transcript: list[str]) -> str:
    parts = [description or ""] + [t for t in transcript or [] if t]
    return " \n".join(p.strip() for p in parts if p and p.strip())[:8000]


# ───────────────────────── rule grader ─────────────────────────


def rule_found(case: dict[str, Any], text: str) -> list[str]:
    t = norm(text)
    return [f["id"] for f in case["findings"] if any(keyword_hit(t, k) for k in f.get("keywords", []))]


def diagnosis_exact(case: dict[str, Any], diagnosis: str) -> Optional[bool]:
    """True/False when the diagnosis is literally the key or a distractor, else None."""
    d = norm(diagnosis)
    if not d:
        return False
    if d == norm(case["diagnosis"]):
        return True
    if any(d == norm(x) for x in case.get("distractors", [])):
        return False
    return None


def rule_diagnosis(case: dict[str, Any], diagnosis: str) -> bool:
    exact = diagnosis_exact(case, diagnosis)
    if exact is not None:
        return exact
    d = norm(diagnosis)
    if any(keyword_hit(d, k) for k in case.get("diagnosisExclude", [])):
        return False
    return any(keyword_hit(d, k) for k in case.get("diagnosisKeywords", []))


def score_for(case: dict[str, Any], found: list[str], dx_correct: bool) -> int:
    weights = {f["id"]: (2 if f.get("critical") else 1) for f in case["findings"]}
    total = sum(weights.values()) or 1
    got = sum(weights[i] for i in found if i in weights)
    # floor(x + 0.5) so the score matches the browser's Math.round exactly
    return int(FINDINGS_WEIGHT * got / total + (DIAGNOSIS_WEIGHT if dx_correct else 0) + 0.5)


def rule_feedback(case: dict[str, Any], found: list[str], dx_correct: bool) -> str:
    missed = [f for f in case["findings"] if f["id"] not in found]
    crit = [f for f in missed if f.get("critical")]
    parts = []
    if dx_correct:
        parts.append(f"Good, the diagnosis is right: {case['diagnosis']}.")
    else:
        parts.append(f"The diagnosis here is {case['diagnosis']}.")
    if not missed:
        parts.append("You described every key finding. Keep reading films in the same systematic order.")
    else:
        key = (crit or missed)[0]
        parts.append(f"The most important thing you did not mention: {key['label'].rstrip('.')}.")
        if len(missed) > 1:
            parts.append(f"You missed {len(missed)} of {len(case['findings'])} key findings, so go through the film in a fixed order every time.")
    return " ".join(parts)


# ───────────────────────── LLM ─────────────────────────

GRADE_SYSTEM = (
    "You are {name}, a consultant radiologist at an Indian teaching hospital, grading an MBBS student's verbal "
    "report of an image in a SYNTHETIC training simulator. You get the answer key (findings with ids, the "
    "diagnosis, the distractors) and the student's own words.\n"
    "Decide which key findings the student clearly identified. Judge meaning, not exact wording; side and "
    "location must be right. A finding the student says is absent does not count, unless the key finding is "
    "itself a negative finding (for example no midline shift). Do not give credit for vague statements.\n"
    "Decide whether the student's committed diagnosis matches the key diagnosis (synonyms are fine; a "
    "distractor or a different condition is wrong).\n"
    "Reply with ONE JSON object only, no markdown:\n"
    '{{"found": [finding ids], "diagnosisCorrect": true|false, "feedback": "2 to 4 short sentences, warm and '
    'specific, name the most important missed finding (if any) and one practical tip for next time"}}\n'
    "Plain words. Never use em dashes."
)

PROBE_SYSTEM = (
    "You are {name}, a consultant radiologist at an Indian teaching hospital, teaching an MBBS student at the "
    "reporting desk in a SYNTHETIC training simulator. The student has just described a film. Ask exactly ONE "
    "short probing question (max 25 words) that steers them towards the most important key finding they have "
    "not yet mentioned, the way a good teacher would (for example: What do you make of the right heart "
    "border?). Never name the missing finding outright and never reveal the diagnosis. If they found "
    "everything, ask them to commit to a diagnosis and say what they would do next.\n"
    'Reply with ONE JSON object only: {{"question": "..."}}. Plain spoken words, no em dashes.'
)


def _lang_tail(lang: Optional[str]) -> str:
    lg = langmod.norm_lang(lang)
    if lg == "en":
        return ""
    return (
        "\n\n" + langmod.language_block(lg, role="teacher")
        + "\nWrite only the free-text field in this language. JSON keys and finding ids stay in English."
    )


def _key_for_llm(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "modality": case["modality"],
        "study": case["study"],
        "vignette": case["vignette"],
        "findings": [{"id": f["id"], "finding": f["label"], "critical": bool(f.get("critical"))} for f in case["findings"]],
        "diagnosis": case["diagnosis"],
        "distractors": case.get("distractors", []),
    }


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    t = strip_think(text or "")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def _llm_json(system: str, user: str, max_tokens: int) -> Optional[dict[str, Any]]:
    try:
        msg, _ = await asyncio.wait_for(
            chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                json_mode=True,
                max_tokens=max_tokens,
                temperature=0.1,
                tier="reasoning",
                effort="medium",
            ),
            timeout=LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.warning("radiology LLM timed out after %.0fs", LLM_TIMEOUT_S)
        return None
    except HTTPException as e:
        log.warning("radiology LLM unavailable (HTTP %s)", e.status_code)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("radiology LLM failed: %s", type(e).__name__)
        return None
    return parse_json_object((msg or {}).get("content") or "")


# ───────────────────────── routes ─────────────────────────


class GradeRequest(BaseModel):
    caseId: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=6000)
    transcript: list[str] = Field(default_factory=list, max_length=60)
    diagnosis: str = Field(min_length=1, max_length=200)
    lang: Optional[str] = None


class GradeResponse(BaseModel):
    score: int
    foundFindings: list[str]
    missedFindings: list[str]
    missedCritical: list[str]
    diagnosisCorrect: bool
    correctDiagnosis: str
    feedback: str
    gradedBy: str


class ProbeRequest(BaseModel):
    caseId: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=6000)
    transcript: list[str] = Field(default_factory=list, max_length=60)
    lang: Optional[str] = None


class ProbeResponse(BaseModel):
    question: str
    gradedBy: str


def _response(case: dict[str, Any], found: list[str], dx_correct: bool, feedback: str, by: str) -> GradeResponse:
    found_set = set(found)
    missed = [f for f in case["findings"] if f["id"] not in found_set]
    return GradeResponse(
        score=score_for(case, list(found_set), dx_correct),
        foundFindings=[f["label"] for f in case["findings"] if f["id"] in found_set],
        missedFindings=[f["label"] for f in missed],
        missedCritical=[f["label"] for f in missed if f.get("critical")],
        diagnosisCorrect=dx_correct,
        correctDiagnosis=case["diagnosis"],
        feedback=feedback,
        gradedBy=by,
    )


@router.post("/agent/radiology/grade", response_model=GradeResponse)
async def radiology_grade(req: GradeRequest) -> GradeResponse:
    case = get_case(req.caseId)
    text = student_text(req.description, [t[:1200] for t in req.transcript])
    rules_found = rule_found(case, text)
    rules_dx = rule_diagnosis(case, req.diagnosis)

    user = json.dumps(
        {"answerKey": _key_for_llm(case), "studentReport": text[:6000] or "(the student said nothing)",
         "studentDiagnosis": req.diagnosis[:200]},
        ensure_ascii=False,
    )
    obj = await _llm_json(GRADE_SYSTEM.format(name=CONSULTANT) + _lang_tail(req.lang), user, 700)
    valid_ids = {f["id"] for f in case["findings"]}
    if obj and isinstance(obj.get("found"), list) and isinstance(obj.get("diagnosisCorrect"), bool):
        found = [i for i in obj["found"] if isinstance(i, str) and i in valid_ids]
        exact = diagnosis_exact(case, req.diagnosis)
        dx = exact if exact is not None else obj["diagnosisCorrect"]
        feedback = clean(obj.get("feedback"), 700) or rule_feedback(case, found, dx)
        return _response(case, found, dx, feedback, "llm")
    return _response(case, rules_found, rules_dx, rule_feedback(case, rules_found, rules_dx), "rules")


def rule_probe(case: dict[str, Any], text: str) -> str:
    found = set(rule_found(case, text))
    missed = [f for f in case["findings"] if f["id"] not in found]
    missed.sort(key=lambda f: 0 if f.get("critical") else 1)
    for f in missed:
        if f.get("probe"):
            return f["probe"]
    return "Good, that covers it. So, what is your diagnosis, and what would you do next?"


@router.post("/agent/radiology/probe", response_model=ProbeResponse)
async def radiology_probe(req: ProbeRequest) -> ProbeResponse:
    case = get_case(req.caseId)
    text = student_text(req.description, [t[:1200] for t in req.transcript])
    user = json.dumps(
        {"answerKey": _key_for_llm(case), "studentReportSoFar": text[:6000] or "(nothing yet)"},
        ensure_ascii=False,
    )
    obj = await _llm_json(PROBE_SYSTEM.format(name=CONSULTANT) + _lang_tail(req.lang), user, 200)
    q = clean((obj or {}).get("question"), 300)
    if q:
        return ProbeResponse(question=q, gradedBy="llm")
    return ProbeResponse(question=rule_probe(case, text), gradedBy="rules")
