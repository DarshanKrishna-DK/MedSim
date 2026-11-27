"""Operation theatre drill: short spoken teaching for a missed or wrong event.

    POST /agent/ot/explain  {scenario, problem, callout, outcome, attempts[], model, teaching, lang?}
        -> {explanation, source}

The drill itself (timeline, vitals, judging, scoring) runs client-side and is
deterministic (src/ot/engine.ts). This endpoint only turns the textbook
teaching point into two or three spoken sentences from the consultant
anaesthetist, via llm_provider.chat powered by Gemini. If the
model is unavailable, slow or empty, it answers with the textbook teaching
point itself, so it always returns something. Synthetic training content only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_provider import chat, strip_think
import lang as langmod

log = logging.getLogger("medsim.ot")

router = APIRouter()

LLM_TIMEOUT_S = 20.0

SYSTEM = (
    "You are Dr. Banerjee, a calm consultant anaesthetist teaching an Indian MBBS intern "
    "in the operation theatre after a drill. The intern is a junior who assists: they give "
    "drugs, run fluids and set equipment, they do not operate.\n"
    "Explain the event in 2 or 3 short spoken sentences: what was happening physiologically, "
    "the correct action with the exact dose or setting given in the teaching point, and one "
    "practical tip. Use ONLY the doses and settings in the teaching point; never invent a dose. "
    "If the intern's attempt was wrong, say kindly why. Plain spoken words, no lists, no markdown, "
    "no em dashes.\n"
    'Reply with ONE JSON object only: {"explanation": "..."}'
)


class ExplainRequest(BaseModel):
    scenario: str = Field(default="", max_length=120)
    problem: str = Field(min_length=1, max_length=120)
    callout: str = Field(default="", max_length=300)
    outcome: str = Field(default="missed", max_length=20)
    attempts: list[str] = Field(default_factory=list, max_length=12)
    model: str = Field(min_length=1, max_length=400)
    teaching: str = Field(default="", max_length=1200)
    lang: Optional[str] = None


class ExplainResponse(BaseModel):
    explanation: str
    source: str


def clean(text: object, limit: int = 600) -> str:
    """Plain spoken-style text: no markdown, no em or en dashes."""
    t = strip_think(str(text or ""))
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"\s*[\u2014\u2013]\s*", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit].rstrip()


def fallback(req: ExplainRequest) -> str:
    lead = "You missed this one." if req.outcome == "missed" else "Nearly." if req.outcome == "partial" else "Let us go over it."
    return clean(f"{lead} {req.model}. {req.teaching}", 600)


def _lang_tail(lang: Optional[str]) -> str:
    lg = langmod.norm_lang(lang)
    if lg == "en":
        return ""
    return "\n\n" + langmod.language_block(lg, role="teacher") + "\nWrite only the explanation in this language. Keep drug names, doses and units in English."


def _parse(text: str) -> Optional[str]:
    t = strip_think(text or "")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = clean(obj.get("explanation"), 600)
    return out or None


@router.post("/agent/ot/explain", response_model=ExplainResponse)
async def ot_explain(req: ExplainRequest) -> ExplainResponse:
    user = json.dumps(
        {
            "operation": req.scenario,
            "problem": req.problem,
            "calloutFromTheTeam": req.callout,
            "outcome": req.outcome,
            "internAttempts": [a[:200] for a in req.attempts],
            "correctAction": req.model,
            "teachingPoint": req.teaching,
        },
        ensure_ascii=False,
    )
    try:
        msg, _ = await asyncio.wait_for(
            chat(
                [{"role": "system", "content": SYSTEM + _lang_tail(req.lang)}, {"role": "user", "content": user}],
                json_mode=True,
                max_tokens=260,
                temperature=0.2,
                tier="fast",
                effort="low",
            ),
            timeout=LLM_TIMEOUT_S,
        )
        text = _parse(str(msg.get("content") or ""))
        if text:
            return ExplainResponse(explanation=text, source="llm")
    except asyncio.TimeoutError:
        log.warning("ot explain LLM timed out after %.0fs", LLM_TIMEOUT_S)
    except HTTPException as e:
        log.warning("ot explain LLM unavailable (HTTP %s)", e.status_code)
    except Exception as e:  # noqa: BLE001 - never fail the drill over a teaching line
        log.warning("ot explain LLM error: %s", type(e).__name__)
    return ExplainResponse(explanation=fallback(req), source="rules")
