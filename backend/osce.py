"""POST /agent/debrief/osce: examiner-style OSCE station marking.

The client sends the station encounter (transcript, action log, diagnosis),
the case rubric (authored or auto-derived, `CaseRubric` shape) and the coach
alerts that were collected silently during the station. We build a checklist
from the rubric plus four standard station tasks, ask the OpenAI-compatible
provider powered by Gemini for strict JSON verdicts, then do the
arithmetic ourselves:

- every checklist item is met / partial / not-met; items the model skipped
  are not-met;
- evidence quotes that do not appear in the transcript or action log are
  dropped (no invented evidence);
- domain and overall scores are weighted from the verdicts (met 1, partial
  0.5), not taken from the model;
- the global rating is the examiner's holistic call, kept within one band of
  the checklist score, and capped at Borderline when a danger-level coach
  alert (critical error) was raised;
- coach alerts are returned verbatim as critical errors.

No interaction at all short-circuits to a deterministic Clear fail without
calling the model. Synthetic training cases, not a claim of clinical accuracy.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from llm_provider import chat
from debrief import parse_json_object

log = logging.getLogger("medsim.osce")
router = APIRouter()

RATINGS = ("clear-fail", "borderline", "clear-pass", "excellent")
VERDICTS = ("met", "partial", "not-met")
SEVERITIES = ("danger", "warning", "info")
SKILLS = ("history", "reasoning", "tests", "treatment", "communication", "disposition")
DOMAINS = ("data_gathering", "clinical_management", "interpersonal")
DOMAIN_LABEL = {
    "data_gathering": "History and examination",
    "clinical_management": "Clinical management",
    "interpersonal": "Communication and professionalism",
}
CREDIT = {"met": 1.0, "partial": 0.5, "not-met": 0.0}
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

# Standard station tasks from the candidate brief ("take a focused history,
# examine, order investigations, and explain the plan"), marked on top of
# the case rubric.
STATION_ITEMS: list[dict[str, Any]] = [
    {"id": "st-exam", "domain": "data_gathering", "weight": 2,
     "label": "Examines (or asks to examine) the relevant system",
     "expect": "States or performs a focused examination relevant to the complaint."},
    {"id": "st-inv", "domain": "clinical_management", "weight": 2,
     "label": "Orders focused, relevant investigations",
     "expect": "Investigations ordered answer a clinical question; no harmful or scattergun tests."},
    {"id": "st-dx", "domain": "clinical_management", "weight": 3,
     "label": "Reaches the correct working diagnosis",
     "expect": "Commits to the case diagnosis (submitted or stated to the patient)."},
    {"id": "st-plan", "domain": "interpersonal", "weight": 2,
     "label": "Explains the diagnosis and plan in plain language",
     "expect": "Tells the patient what is wrong and what happens next, without jargon, and checks understanding."},
]

DEFAULT_RUBRIC_ITEMS: list[dict[str, Any]] = [
    {"id": "dg-01", "domain": "data_gathering", "weight": 2, "label": "Explores the presenting complaint",
     "expect": "Onset, duration, character, severity and associated symptoms."},
    {"id": "dg-02", "domain": "data_gathering", "weight": 1, "label": "Asks about red flags",
     "expect": "Screens for the dangerous causes of this complaint."},
    {"id": "dg-03", "domain": "data_gathering", "weight": 1, "label": "Past history, drugs and allergies",
     "expect": "Asks about past illness, current medicines and drug allergies."},
    {"id": "ip-01", "domain": "interpersonal", "weight": 1, "label": "Introduces self and confirms identity",
     "expect": "Greets the patient, gives name and role, confirms who they are."},
    {"id": "ip-02", "domain": "interpersonal", "weight": 1, "label": "Explores ideas, concerns and expectations",
     "expect": "Asks what the patient thinks is wrong, what worries them and what they hope for."},
]


class OsceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    caseId: Optional[str] = None
    patient: Optional[dict[str, Any]] = None
    correctDiagnosis: Any = None
    submittedDiagnosis: Any = None
    actions: Optional[list[Any]] = None
    transcript: Optional[list[dict[str, Any]]] = None
    elapsedSeconds: Optional[float] = None
    stationSeconds: Optional[float] = None
    rubric: Any = None  # CaseRubric: {data_gathering:[...], clinical_management:[...], interpersonal:[...], safety_netting?}
    coachAlerts: Optional[list[dict[str, Any]]] = None


class OsceItem(BaseModel):
    id: str
    domain: str
    label: str
    weight: float
    verdict: str
    evidence: str
    note: str


class OsceDomain(BaseModel):
    id: str
    label: str
    score: int
    items: list[OsceItem]


class OsceCritical(BaseModel):
    severity: str
    title: str
    detail: str


class OsceResponse(BaseModel):
    globalRating: str
    score: int
    domains: list[OsceDomain]
    criticalErrors: list[OsceCritical]
    skills: dict[str, int]
    strengths: list[str]
    improvements: list[str]
    examinerComment: str


SYSTEM = (
    "You are an OSCE examiner marking a final-year MBBS student or intern at a timed outpatient station "
    "(synthetic training case). Mark ONLY from the evidence given: the transcript (Doctor = candidate, "
    "Patient = simulated patient), the action log and the submitted diagnosis. Never invent events. "
    "For EVERY checklist item return a verdict: \"met\" (clearly done), \"partial\" (attempted or incomplete) "
    "or \"not-met\" (no evidence). For met or partial, \"evidence\" must be a short VERBATIM quote copied "
    "from the transcript or action log (max 20 words); otherwise \"evidence\" is \"\". \"note\" is one short "
    "examiner remark (max 18 words). Also score six skills 0-100 and give a holistic global rating. "
    "Reply with JSON only, exactly: "
    '{"items":[{"id":str,"verdict":"met|partial|not-met","evidence":str,"note":str}],'
    '"skills":{"history":int,"reasoning":int,"tests":int,"treatment":int,"communication":int,"disposition":int},'
    '"globalRating":"clear-fail|borderline|clear-pass|excellent","strengths":[str],"improvements":[str],'
    '"examinerComment":str}. Strengths and improvements: max 4 each, specific, max 22 words each. '
    "examinerComment: 2-3 sentences in English, addressed to the candidate. Do not use em dashes."
)


# ── helpers ──────────────────────────────────────────────────────────────


def _clean(text: Any, limit: int = 400) -> str:
    s = str(text or "").replace(EM_DASH, ", ").replace(EN_DASH, "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _tokens(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE).split())


def _as_list(v: Any, n: int = 4) -> list[str]:
    if isinstance(v, str):
        v = [p for p in v.splitlines()]
    if not isinstance(v, (list, tuple)):
        return []
    out = [_clean(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(x)), 240) for x in v]
    return [x for x in out if x][:n]


def _clamp(v: Any) -> Optional[int]:
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def _dx_name(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("name") or v.get("id") or "")
    return str(v or "")


def _dx_id(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("id") or "")
    return str(v or "")


def build_checklist(rubric: Any) -> list[dict[str, Any]]:
    """Checklist items from a CaseRubric dict, plus the standard station tasks."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(rubric, dict):
        for dom in DOMAINS:
            for i, c in enumerate(rubric.get(dom) or []):
                if not isinstance(c, dict) or not c.get("label"):
                    continue
                cid = str(c.get("criterion_id") or f"{dom[:2]}-{i + 1:02d}")
                if cid in seen:
                    continue
                seen.add(cid)
                try:
                    w = float(c.get("weight") or 1)
                except (TypeError, ValueError):
                    w = 1.0
                items.append({
                    "id": cid, "domain": dom, "weight": max(0.5, min(3.0, w)),
                    "label": _clean(c["label"], 140), "expect": _clean(c.get("evidence", ""), 240),
                })
        sn = rubric.get("safety_netting")
        if isinstance(sn, dict) and sn.get("required_elements"):
            items.append({
                "id": "sn-01", "domain": "clinical_management", "weight": 1.0, "label": "Safety-netting",
                "expect": "; ".join(_clean(x, 120) for x in sn["required_elements"] if x),
            })
    if not items:
        items = [dict(x) for x in DEFAULT_RUBRIC_ITEMS]
    have = {i["id"] for i in items}
    items.extend(dict(x) for x in STATION_ITEMS if x["id"] not in have)
    return items


def rating_for_score(score: int) -> str:
    if score < 40:
        return "clear-fail"
    if score < 55:
        return "borderline"
    if score < 80:
        return "clear-pass"
    return "excellent"


def _weighted(items: list[OsceItem]) -> int:
    total = sum(i.weight for i in items)
    if total <= 0:
        return 0
    return int(round(100 * sum(i.weight * CREDIT[i.verdict] for i in items) / total))


def critical_errors(alerts: Optional[list[dict[str, Any]]]) -> list[OsceCritical]:
    out: list[OsceCritical] = []
    seen: set[str] = set()
    for a in alerts or []:
        if not isinstance(a, dict):
            continue
        title = _clean(a.get("title"), 80)
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        sev = a.get("severity") if a.get("severity") in SEVERITIES else "warning"
        detail = _clean(a.get("explanation") or a.get("short") or "", 320)
        out.append(OsceCritical(severity=sev, title=title, detail=detail))
    out.sort(key=lambda e: SEVERITIES.index(e.severity))
    return out


def corpus(req: OsceRequest) -> str:
    parts = [str(t.get("content", t.get("text", ""))) for t in (req.transcript or []) if isinstance(t, dict)]
    for a in req.actions or []:
        parts.append(str(a.get("detail", "")) if isinstance(a, dict) else str(a))
    sub = _dx_name(req.submittedDiagnosis)
    if sub:
        parts.append(sub)
    return _tokens(" \n ".join(parts))


def has_interaction(req: OsceRequest) -> bool:
    doctor_turns = [t for t in (req.transcript or []) if isinstance(t, dict) and t.get("role") == "user"
                    and str(t.get("content", t.get("text", ""))).strip()]
    return bool(doctor_turns or (req.actions or []) or req.submittedDiagnosis)


def assemble(req: OsceRequest, checklist: list[dict[str, Any]], raw: dict[str, Any]) -> OsceResponse:
    """Join model verdicts onto the checklist and do the scoring."""
    by_id: dict[str, dict[str, Any]] = {}
    for it in raw.get("items") or []:
        if isinstance(it, dict) and it.get("id") is not None:
            by_id.setdefault(str(it["id"]), it)
    text = corpus(req)

    sub_id, cor_id = _dx_id(req.submittedDiagnosis), _dx_id(req.correctDiagnosis)
    items: list[OsceItem] = []
    for c in checklist:
        m = by_id.get(c["id"], {})
        verdict = m.get("verdict") if m.get("verdict") in VERDICTS else "not-met"
        evidence = _clean(m.get("evidence"), 220).strip("\"'")
        if evidence and (not _tokens(evidence) or _tokens(evidence) not in text):
            evidence = ""  # not a real quote: drop it
        note = _clean(m.get("note"), 200)
        if c["id"] == "st-dx" and sub_id and cor_id:
            # Deterministic when a diagnosis was submitted on the chart.
            verdict = "met" if sub_id == cor_id else "not-met"
            evidence = _clean(_dx_name(req.submittedDiagnosis), 120)
            if verdict == "not-met" and not note:
                note = f"Case diagnosis was {_dx_name(req.correctDiagnosis)}."
        if verdict == "not-met" and not note:
            note = "No evidence in the transcript or action log."
        items.append(OsceItem(id=c["id"], domain=c["domain"], label=c["label"], weight=float(c["weight"]),
                              verdict=verdict, evidence=evidence, note=note))

    domains = []
    for d in DOMAINS:
        its = [i for i in items if i.domain == d]
        if its:
            domains.append(OsceDomain(id=d, label=DOMAIN_LABEL[d], score=_weighted(its), items=its))
    score = _weighted(items)
    errors = critical_errors(req.coachAlerts)

    band = RATINGS.index(rating_for_score(score))
    proposed = raw.get("globalRating")
    idx = RATINGS.index(proposed) if proposed in RATINGS else band
    idx = max(band - 1, min(band + 1, idx))
    if any(e.severity == "danger" for e in errors):
        idx = min(idx, RATINGS.index("borderline"))

    dom_score = {d.id: d.score for d in domains}
    fallback = {
        "history": dom_score.get("data_gathering", score),
        "reasoning": next((i for i in items if i.id == "st-dx"), None),
        "tests": next((i for i in items if i.id == "st-inv"), None),
        "treatment": dom_score.get("clinical_management", score),
        "communication": dom_score.get("interpersonal", score),
        "disposition": dom_score.get("clinical_management", score),
    }
    skills_raw = raw.get("skills") if isinstance(raw.get("skills"), dict) else {}
    skills: dict[str, int] = {}
    for k in SKILLS:
        v = _clamp(skills_raw.get(k))
        if v is None:
            f = fallback[k]
            v = int(round(CREDIT[f.verdict] * 100)) if isinstance(f, OsceItem) else int(f if f is not None else score)
        skills[k] = v

    return OsceResponse(
        globalRating=RATINGS[idx],
        score=score,
        domains=domains,
        criticalErrors=errors,
        skills=skills,
        strengths=_as_list(raw.get("strengths")),
        improvements=_as_list(raw.get("improvements")),
        examinerComment=_clean(raw.get("examinerComment"), 700),
    )


def no_interaction_result(req: OsceRequest, checklist: list[dict[str, Any]]) -> OsceResponse:
    out = assemble(req, checklist, {"items": [], "globalRating": "clear-fail",
                                    "skills": {k: 0 for k in SKILLS}})
    out.examinerComment = ("No history, examination, orders or diagnosis were recorded during the station, "
                           "so nothing could be credited. Start by greeting the patient and exploring the complaint.")
    out.improvements = ["Engage the patient: introduce yourself and take a focused history before the bell."]
    return out


def _user_payload(req: OsceRequest, checklist: list[dict[str, Any]]) -> str:
    transcript = []
    for t in (req.transcript or [])[-90:]:
        if not isinstance(t, dict):
            continue
        who = "Doctor" if t.get("role") == "user" else "Patient"
        txt = _clean(t.get("content", t.get("text", "")), 500)
        if txt:
            transcript.append(f"{who}: {txt}")
    return json.dumps({
        "station": "Medicine OPD style outpatient station; tasks: focused history, examination, investigations, explain the plan.",
        "stationSeconds": req.stationSeconds,
        "elapsedSeconds": req.elapsedSeconds,
        "patient": req.patient or {},
        "correctDiagnosis": req.correctDiagnosis,
        "submittedDiagnosis": req.submittedDiagnosis,
        "actions": (req.actions or [])[-60:],
        "transcript": transcript,
        "checklist": [{"id": c["id"], "domain": c["domain"], "label": c["label"], "expect": c["expect"]}
                      for c in checklist],
        "criticalErrorsRaised": [e.title for e in critical_errors(req.coachAlerts)],
    }, default=str, ensure_ascii=False)[:26000]


@router.post("/agent/debrief/osce", response_model=OsceResponse)
async def debrief_osce(req: OsceRequest):
    checklist = build_checklist(req.rubric)
    if not has_interaction(req):
        return no_interaction_result(req, checklist)
    msg, _ = await chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _user_payload(req, checklist)}],
        json_mode=True,
        max_tokens=2600,
        temperature=0.1,
        tier="reasoning",
        effort="medium",
    )
    try:
        raw = parse_json_object(msg.get("content") or "")
        if not isinstance(raw.get("items"), list):
            raise ValueError("no items")
    except Exception:  # noqa: BLE001
        log.warning("osce marking: malformed model JSON")
        raise HTTPException(status_code=502, detail="OSCE marker returned malformed JSON")
    return assemble(req, checklist, raw)
