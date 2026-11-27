"""POST /agent/debrief/simple, shift/case debrief powered by Gemini.

Uses the OpenAI-compatible provider in JSON mode.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from llm_provider import chat, strip_think

router = APIRouter()

DEFAULT_RUBRIC = [
    "History taking", "Diagnostic reasoning", "Test selection", "Treatment and safety",
    "Communication", "Disposition",
]


class DebriefRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    caseId: Optional[str] = None
    patient: Optional[dict[str, Any]] = None  # summary: name, age, chiefComplaint, vitals...
    correctDiagnosis: Any = None  # {id,name} or string
    submittedDiagnosis: Any = None
    actions: Optional[list[Any]] = None  # [{t, kind, detail}] or strings
    transcript: Optional[list[dict[str, Any]]] = None  # [{role, content}] (text also accepted)
    elapsedSeconds: Optional[float] = None
    # Either a list of domain names, or a case's authored CaseRubric object
    # ({data_gathering, clinical_management, interpersonal, safety_netting, ...}).
    rubric: Any = None


def split_rubric(rubric: Any) -> tuple[list[str], list[str]]:
    """Return (domain names to score, case-specific criteria to grade against).

    A list of strings is taken as the domains. An authored rubric object keeps
    the standard domains and contributes its criteria labels and evidence as
    grading context, so cases with a rubric no longer fail validation."""
    if isinstance(rubric, list) and rubric and all(isinstance(x, str) for x in rubric):
        return [x for x in rubric if x.strip()], []
    criteria: list[str] = []
    if isinstance(rubric, dict):
        for section in ("data_gathering", "clinical_management", "interpersonal"):
            for c in rubric.get(section) or []:
                if isinstance(c, dict) and c.get("label"):
                    ev = str(c.get("evidence", "")).strip()
                    criteria.append(f"[{section}] {c['label']}" + (f": {ev}" if ev else ""))
        sn = rubric.get("safety_netting")
        if isinstance(sn, dict) and sn.get("required_elements"):
            criteria.append("[safety_netting] " + "; ".join(str(x) for x in sn["required_elements"]))
    return list(DEFAULT_RUBRIC), criteria[:40]


class Domain(BaseModel):
    name: str
    score: int
    notes: str


class DebriefResponse(BaseModel):
    score: int
    domains: list[Domain]
    strengths: list[str]
    improvements: list[str]
    summary: str


SYSTEM = (
    "You are a senior attending physician debriefing a trainee doctor after a simulated (synthetic) case. "
    "Grade ONLY from the evidence provided: the action log, the transcript and the known correct diagnosis. "
    "Do not invent events. Reply with JSON only, exactly: "
    '{"score":0-100,"domains":[{"name":str,"score":0-100,"notes":str}],"strengths":[str],'
    '"improvements":[str],"summary":str}. One domain per rubric item, in order. If caseCriteria is '
    "present, use those case-specific criteria as the marking scheme inside each domain. Notes and bullets are "
    "specific and short (max 25 words each); summary is 2-3 sentences. Not a claim of clinical accuracy."
)


def _clamp(v: Any) -> int:
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def _as_list(v: Any) -> list[str]:
    """Models sometimes return a single string (or a bulleted block) instead
    of a JSON list. Iterating a string would split it into characters, so
    normalise to a list of non-empty lines first."""
    if v is None:
        return []
    if isinstance(v, str):
        parts = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p).strip() for p in v.splitlines()]
        return [p for p in parts if p]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def coerce_debrief(raw: dict[str, Any], rubric: list[str]) -> DebriefResponse:
    domains = []
    for d in raw.get("domains") or []:
        if isinstance(d, dict) and d.get("name"):
            domains.append(Domain(name=str(d["name"]), score=_clamp(d.get("score")), notes=str(d.get("notes", ""))))
    if not domains:
        raise ValueError("no domains")
    score = _clamp(raw["score"]) if "score" in raw else round(sum(d.score for d in domains) / len(domains))
    return DebriefResponse(
        score=score,
        domains=domains,
        strengths=_as_list(raw.get("strengths"))[:6],
        improvements=_as_list(raw.get("improvements"))[:6],
        summary=str(raw.get("summary", "")),
    )


def parse_json_object(text: str) -> dict[str, Any]:
    text = strip_think(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json")
    v = json.loads(m.group(0))
    if not isinstance(v, dict):
        raise ValueError("not an object")
    return v


@router.post("/agent/debrief/simple", response_model=DebriefResponse)
async def debrief_simple(req: DebriefRequest):
    rubric, case_criteria = split_rubric(req.rubric)
    transcript = [
        {"role": t.get("role", ""), "text": t.get("content", t.get("text", ""))}
        for t in (req.transcript or [])
    ][-80:]
    user = json.dumps(
        {
            "caseId": req.caseId,
            "patient": req.patient or {},
            "correctDiagnosis": req.correctDiagnosis,
            "submittedDiagnosis": req.submittedDiagnosis,
            "elapsedSeconds": req.elapsedSeconds,
            "actions": (req.actions or [])[-80:],
            "transcript": transcript,
            "rubric": rubric,
            "caseCriteria": case_criteria,
        },
        default=str,
    )[:24000]
    msg, _ = await chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        json_mode=True,
        max_tokens=1200,
        temperature=0.2,
        tier="reasoning",
        effort="medium",
    )
    try:
        return coerce_debrief(parse_json_object(msg.get("content") or ""), rubric)
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Debrief model returned malformed JSON")
