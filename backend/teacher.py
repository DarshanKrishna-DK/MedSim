"""POST /teach/lesson: a short whiteboard lesson for the Teacher Portal.

The LLM powered by Gemini (via llm_provider.chat) writes a lesson
as strict JSON. We validate and normalise it here so the frontend whiteboard
renderer can trust the shape:

    {title, summary, topic, level,
     steps: [{heading, points[<=3], narration, visual:{kind, ...}}] (4-6),
     quiz: {question, options[4], answerIndex, why}}

Visual kinds and their data (all strings short, all lists bounded):
    flow      {nodes:[{id,label}], edges:[{from,to,label?}]}
    cycle     {nodes:[{id,label}], center?}
    compare   {left:{title,items[]}, right:{title,items[]}}
    labelled  {center, labels:[{label, note?}]}
    timeline  {events:[{t, label}]}
    graph     {xLabel, yLabel, series:[{name, points:[[x,y],...]}]}
    waveform  {preset, highlight?, caption?}  (curve drawn in code from a preset)

One repair retry if the first reply is malformed. Lessons are cached in
memory by (level, topic). Educational content only, not clinical advice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_provider import chat, strip_think
import lang as langmod

log = logging.getLogger("medsim.teacher")

router = APIRouter()

VISUAL_KINDS = ("flow", "cycle", "compare", "labelled", "timeline", "graph", "waveform")
# Physiology curves drawn in code by the frontend (src/components/teach/waveforms.ts).
# The model only picks a preset; it never invents the points.
WAVEFORM_PRESETS = (
    "ecg-normal", "ecg-stemi", "ecg-af", "ecg-vt", "ecg-heart-block-1",
    "cardiac-cycle-wiggers", "pv-loop", "oxy-hb-dissociation", "frank-starling",
    "action-potential-ventricular", "action-potential-nerve",
    "spirometry-normal-vs-obstructive", "glucose-insulin-ogtt", "dose-response",
)
WAVEFORM_HIGHLIGHTS: dict[str, tuple[str, ...]] = {
    "ecg-normal": ("p", "pr", "qrs", "st", "t", "qt"),
    "oxy-hb-dissociation": ("right", "left"),
}
MIN_STEPS, MAX_STEPS = 4, 6
CACHE_MAX = 64
ATTEMPT_TIMEOUT_S = 40.0
TOTAL_TIMEOUT_S = 75.0

_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


class LessonRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=160)
    level: Literal["student", "intern"] = "student"
    lang: Optional[str] = None


SYSTEM = (
    "You are a warm Indian medical college professor preparing a short whiteboard explainer "
    "for a {level}. Reply with ONE JSON object only, no markdown, no prose outside JSON. Shape:\n"
    '{{"title": str, "summary": str (1 sentence), "steps": [4 to 6 steps], "quiz": {{"question": str, '
    '"options": [4 short strings], "answerIndex": 0-3, "why": str (1 sentence)}}}}\n'
    'Each step: {{"heading": str (max 6 words), "points": [1 to 3 strings, each max 10 words], '
    '"narration": str (1 or 2 plain spoken sentences, simple words, an analogy where it helps, no lists, '
    'no symbols), "visual": {{"kind": one of flow|cycle|compare|labelled|timeline|graph|waveform, ...data}}}}\n'
    "Visual data by kind (labels max 4 words):\n"
    '- flow: {{"kind":"flow","nodes":[{{"id":"a","label":str}}],"edges":[{{"from":"a","to":"b","label":str optional}}]}} 2-6 nodes\n'
    '- cycle: {{"kind":"cycle","nodes":[{{"id":"a","label":str}}],"center":str optional}} 3-6 nodes in order\n'
    '- compare: {{"kind":"compare","left":{{"title":str,"items":[str]}},"right":{{"title":str,"items":[str]}}}} 2-4 items each\n'
    '- labelled: {{"kind":"labelled","center":str,"labels":[{{"label":str,"note":str optional}}]}} 3-6 labels\n'
    '- timeline: {{"kind":"timeline","events":[{{"t":str,"label":str}}]}} 3-6 events\n'
    '- graph: {{"kind":"graph","xLabel":str,"yLabel":str,"series":[{{"name":str,"points":[[x,y],...]}}]}} '
    "numbers only, at most 6 points, 1-2 series, ONLY for a simple illustrative trend\n"
    '- waveform: {{"kind":"waveform","preset":one of {presets},"highlight":str optional,"caption":str optional max 12 words}} '
    "the app draws the exact curve with labelled landmarks, you never supply points. "
    "highlight: for ecg-normal one of p|pr|qrs|st|t|qt, for oxy-hb-dissociation one of right|left.\n"
    "RULE: whenever a step is about an ECG or rhythm strip, the cardiac cycle or pressure and volume curves, a pressure-volume loop, "
    "the oxygen-haemoglobin dissociation curve, Frank-Starling, an action potential, spirometry, glucose tolerance or insulin, "
    "or a dose-response curve, you MUST use kind waveform with the matching preset. "
    "NEVER use graph for ECG, pressure, dissociation, action potential or any such physiology curve.\n"
    "Use at least three different visual kinds across the steps. Keep facts standard textbook level. "
    "This is for synthetic training, not clinical advice."
)


def lesson_system(level: str, lang: Optional[str] = None) -> str:
    base = SYSTEM.format(
        level="medical student" if level == "student" else "junior doctor (intern)",
        presets=" | ".join(WAVEFORM_PRESETS),
    )
    lg = langmod.norm_lang(lang)
    if lg == "en":
        return base
    name = langmod.LANG_NAMES[lg]
    return (
        base
        + f"\nLANGUAGE: write every step's narration and the quiz why in natural spoken {name} in its native "
        "script; medical terms and common loanwords may stay in English. Headings, points, the quiz question and "
        "options stay short and may keep English medical terms. JSON keys, visual kinds and presets stay exactly "
        "in English."
        + ("\nUse feminine Hindi verb forms (a woman professor)." if lg == "hi" else "")
    )


# ───────────────────────── normalisation ─────────────────────────


def _s(v: Any, n: int = 80) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return ""
    t = re.sub(r"[*_`#>]+", "", str(v))
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > n:
        t = t[: n - 1].rstrip() + "…"
    return t


def _label_of(v: Any, n: int = 40) -> str:
    if isinstance(v, dict):
        for k in ("label", "name", "text", "title", "t", "value"):
            if v.get(k) not in (None, ""):
                return _s(v.get(k), n)
        return ""
    return _s(v, n)


def _nodes(raw: Any, max_n: int = 6) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw):
        label = _label_of(item)
        if not label:
            continue
        nid = _s(item.get("id"), 24) if isinstance(item, dict) and item.get("id") not in (None, "") else ""
        out.append({"id": nid or f"n{i}", "label": label})
        if len(out) >= max_n:
            break
    seen: set[str] = set()
    for i, n in enumerate(out):  # ids must be unique
        if n["id"] in seen:
            n["id"] = f"{n['id']}_{i}"
        seen.add(n["id"])
    return out


def _resolve(ref: Any, nodes: list[dict[str, str]]) -> Optional[str]:
    if isinstance(ref, int) and 0 <= ref < len(nodes):
        return nodes[ref]["id"]
    r = _s(ref, 40).lower()
    for n in nodes:
        if n["id"].lower() == r or n["label"].lower() == r:
            return n["id"]
    return None


def _items(raw: Any, max_n: int = 4) -> list[str]:
    if isinstance(raw, str):
        raw = [p for p in re.split(r"[\n;]+", raw)]
    if not isinstance(raw, list):
        return []
    return [x for x in (_label_of(i, 48) for i in raw) if x][:max_n]


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) < 1e9 else None


def _points(raw: Any) -> list[list[float]]:
    pts: list[list[float]] = []
    if not isinstance(raw, list):
        return pts
    for p in raw:
        x = y = None
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _num(p[0]), _num(p[1])
        elif isinstance(p, dict):
            x, y = _num(p.get("x")), _num(p.get("y"))
        if x is not None and y is not None:
            pts.append([x, y])
    pts.sort(key=lambda q: q[0])
    return pts[:24]


def normalize_visual(raw: Any, fallback_labels: list[str]) -> dict[str, Any]:
    """Coerce a model visual into one of the canonical shapes; falls back
    to a flow of the step's own points so a step always has a diagram."""
    v = raw if isinstance(raw, dict) else {}
    kind = _s(v.get("kind") or v.get("type"), 20).lower()
    kind = {"labeled": "labelled", "label": "labelled", "comparison": "compare", "line": "graph",
            "chart": "graph", "wave": "waveform", "curve": "waveform", "process": "flow", "flowchart": "flow"}.get(kind, kind)

    try:
        if kind == "flow":
            nodes = _nodes(v.get("nodes") or v.get("steps") or v.get("labels"))
            if len(nodes) >= 2:
                edges = []
                for e in v.get("edges") or []:
                    if not isinstance(e, dict):
                        continue
                    a, b = _resolve(e.get("from"), nodes), _resolve(e.get("to"), nodes)
                    if a and b and a != b:
                        edge = {"from": a, "to": b}
                        lab = _s(e.get("label"), 24)
                        if lab:
                            edge["label"] = lab
                        edges.append(edge)
                if not edges:
                    edges = [{"from": nodes[i]["id"], "to": nodes[i + 1]["id"]} for i in range(len(nodes) - 1)]
                return {"kind": "flow", "nodes": nodes, "edges": edges[:8]}
        elif kind == "cycle":
            nodes = _nodes(v.get("nodes") or v.get("stages") or v.get("labels") or v.get("steps"))
            if len(nodes) >= 3:
                out: dict[str, Any] = {"kind": "cycle", "nodes": nodes}
                c = _s(v.get("center"), 28)
                if c:
                    out["center"] = c
                return out
        elif kind == "compare":
            left, right = v.get("left"), v.get("right")
            cols = v.get("columns")
            if (not isinstance(left, dict) or not isinstance(right, dict)) and isinstance(cols, list) and len(cols) >= 2:
                left, right = cols[0], cols[1]
            if isinstance(left, dict) and isinstance(right, dict):
                lt, rt = _label_of(left.get("title") or left.get("name")), _label_of(right.get("title") or right.get("name"))
                li, ri = _items(left.get("items") or left.get("points")), _items(right.get("items") or right.get("points"))
                if lt and rt and li and ri:
                    return {"kind": "compare", "left": {"title": lt, "items": li}, "right": {"title": rt, "items": ri}}
        elif kind == "labelled":
            center = _label_of(v.get("center") or v.get("subject") or v.get("title"))
            labels = []
            for item in (v.get("labels") or v.get("parts") or [])[:6]:
                lab = _label_of(item)
                if not lab:
                    continue
                entry = {"label": lab}
                note = _s(item.get("note"), 40) if isinstance(item, dict) else ""
                if note:
                    entry["note"] = note
                labels.append(entry)
            if center and len(labels) >= 2:
                return {"kind": "labelled", "center": center, "labels": labels}
        elif kind == "timeline":
            events = []
            for e in (v.get("events") or v.get("items") or [])[:6]:
                if isinstance(e, dict):
                    t = _s(e.get("t") or e.get("time") or e.get("at") or e.get("when"), 18)
                    lab = _s(e.get("label") or e.get("event") or e.get("text"), 40)
                else:
                    t, lab = "", _s(e, 40)
                if lab:
                    events.append({"t": t or str(len(events) + 1), "label": lab})
            if len(events) >= 2:
                return {"kind": "timeline", "events": events}
        elif kind == "waveform":
            preset = re.sub(r"[\s_]+", "-", str(v.get("preset") or v.get("name") or "").strip().lower())[:48]
            if preset in WAVEFORM_PRESETS:
                out_w: dict[str, Any] = {"kind": "waveform", "preset": preset}
                hl = _s(v.get("highlight"), 12).lower()
                if hl in WAVEFORM_HIGHLIGHTS.get(preset, ()):
                    out_w["highlight"] = hl
                cap = _s(v.get("caption"), 90)
                if cap:
                    out_w["caption"] = cap
                return out_w
            # unknown preset: fall through to the flow of the step's points
        elif kind == "graph":
            series = []
            raw_series = v.get("series")
            if not isinstance(raw_series, list):
                raw_series = [{"name": v.get("name") or "", "points": v.get("points")}]
            for s in raw_series[:2]:
                if not isinstance(s, dict):
                    continue
                pts = _points(s.get("points"))
                if len(pts) < 2 and isinstance(s.get("x"), list) and isinstance(s.get("y"), list):
                    pts = _points([[a, b] for a, b in zip(s["x"], s["y"])])
                if len(pts) >= 2:
                    series.append({"name": _s(s.get("name"), 24), "points": pts})
            if series:
                return {"kind": "graph", "xLabel": _s(v.get("xLabel") or v.get("x_label"), 28),
                        "yLabel": _s(v.get("yLabel") or v.get("y_label"), 28), "series": series}
    except Exception:  # noqa: BLE001
        log.debug("visual normalisation failed", exc_info=True)

    labels = [x for x in fallback_labels if x][:4] or ["Key idea"]
    nodes = [{"id": f"n{i}", "label": _s(t, 40)} for i, t in enumerate(labels)]
    return {"kind": "flow", "nodes": nodes,
            "edges": [{"from": nodes[i]["id"], "to": nodes[i + 1]["id"]} for i in range(len(nodes) - 1)]}


def _sentences(text: str, max_n: int = 2, max_chars: int = 320) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(parts[:max_n]).strip()
    return out[:max_chars].rstrip()


def _answer_index(raw: Any, options: list[str]) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw < len(options) else None
    s = _s(raw, 80)
    if re.fullmatch(r"[0-3]", s):
        return int(s)
    if re.fullmatch(r"[A-Da-d]", s):
        return "abcd".index(s.lower())
    for i, o in enumerate(options):
        if s and s.lower() == o.lower():
            return i
    return None


def validate_lesson(raw: Any, topic: str, level: str) -> dict[str, Any]:
    """Raise ValueError with a short reason if the model output is unusable."""
    if not isinstance(raw, dict):
        raise ValueError("top level is not an object")
    if isinstance(raw.get("lesson"), dict):
        raw = raw["lesson"]
    title = _s(raw.get("title"), 70)
    if not title:
        raise ValueError("missing title")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError("steps must be a list")
    steps = []
    for st in steps_raw:
        if not isinstance(st, dict):
            continue
        heading = _s(st.get("heading") or st.get("title"), 48)
        points = _items(st.get("points") or st.get("bullets"), 3)
        points = [_s(p, 80) for p in points]
        narration = _sentences(_s(st.get("narration") or st.get("say") or st.get("script"), 400))
        if not heading or not narration:
            continue
        steps.append({
            "heading": heading,
            "points": points,
            "narration": narration,
            "visual": normalize_visual(st.get("visual"), points or [heading]),
        })
    if len(steps) < MIN_STEPS:
        raise ValueError(f"need {MIN_STEPS}-{MAX_STEPS} complete steps, got {len(steps)}")
    steps = steps[:MAX_STEPS]

    q = raw.get("quiz")
    if not isinstance(q, dict):
        raise ValueError("missing quiz object")
    question = _s(q.get("question"), 160)
    options = [_label_of(o, 70) for o in (q.get("options") or [])]
    options = [re.sub(r"^[A-Da-d][).:]\s+", "", o) for o in options if o][:4]
    if not question or len(options) != 4:
        raise ValueError("quiz needs a question and exactly 4 options")
    idx = _answer_index(q.get("answerIndex", q.get("answer")), options)
    if idx is None:
        raise ValueError("quiz answerIndex must be 0-3")
    return {
        "topic": topic,
        "level": level,
        "title": title,
        "summary": _s(raw.get("summary"), 200) or steps[0]["narration"],
        "steps": steps,
        "quiz": {"question": question, "options": options, "answerIndex": idx, "why": _s(q.get("why"), 220)},
    }


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


def cache_key(topic: str, level: str, lang: Optional[str] = None) -> str:
    t = re.sub(r"[^a-z0-9 +-]", "", topic.lower())
    t = re.sub(r"\s+", " ", t).strip()
    lg = langmod.norm_lang(lang)
    return f"{level}:{t}" if lg == "en" else f"{level}:{lg}:{t}"


# ───────────────────────── route ─────────────────────────


async def _generate(topic: str, level: str, lang: Optional[str] = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": lesson_system(level, lang)},
        {"role": "user", "content": f"Topic: {topic}\nWrite the lesson JSON now."},
    ]
    last_err = ""
    async with httpx.AsyncClient(timeout=ATTEMPT_TIMEOUT_S) as client:
        for attempt in range(2):
            msg, provider = await asyncio.wait_for(
                chat(
                    messages,
                    json_mode=True,
                    max_tokens=2400,
                    temperature=0.4,
                    client=client,
                    tier="reasoning",
                    effort="medium",
                ),
                timeout=ATTEMPT_TIMEOUT_S * 1.5,
            )
            content = msg.get("content") or ""
            try:
                return validate_lesson(parse_json_object(content), topic, level)
            except ValueError as e:
                last_err = str(e)
                log.warning("teach lesson attempt %d via %s invalid: %s", attempt + 1, provider, last_err)
                messages = messages + [
                    {"role": "assistant", "content": content[:6000]},
                    {"role": "user", "content": f"That JSON was not usable: {last_err}. Reply again with the "
                     "complete corrected JSON object only, following the exact shape."},
                ]
    raise HTTPException(status_code=502, detail=f"Lesson model returned malformed JSON: {last_err}"[:300])


@router.post("/teach/lesson")
async def teach_lesson(req: LessonRequest):
    topic = _s(req.topic, 120)
    if not topic:
        raise HTTPException(status_code=422, detail="topic is required")
    key = cache_key(topic, req.level, req.lang)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return {**hit, "cached": True}
    try:
        lesson = await asyncio.wait_for(_generate(topic, req.level, req.lang), timeout=TOTAL_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Lesson generation timed out")
    _cache[key] = lesson
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)
    return {**lesson, "cached": False}


class AskMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    system: str = Field(min_length=1, max_length=16000)
    lang: Optional[str] = None
    messages: list[AskMessage] = Field(min_length=1, max_length=40)


# Typed / voice-off Q&A: same short, human style as the live voice agent.
ASK_STYLE = (
    "REPLY STYLE: Answer like a person talking, in one to three short sentences (about 50 words at most), with a "
    "light acknowledgement when natural ('Achha, good question.'). Work out what the student means even from "
    "fragments, typos or Hinglish and answer that; never say you did not understand. If it is truly ambiguous, ask "
    "ONE short question with your best guess. End with a short check-in or an offer to go deeper, not a lecture."
)

# Spoken/shown when the model is down or empty: natural, moves the chat on.
ASK_FALLBACK = {
    "en": "Hmm, give me a moment on that one. Ask me again, or try it a different way?",
    "hi": "हम्म, इस पर मुझे एक पल दीजिए। एक बार फिर पूछिए, या थोड़ा अलग तरीके से?",
}


def ask_fallback(lang: Optional[str]) -> str:
    return ASK_FALLBACK.get((lang or "en").lower(), ASK_FALLBACK["en"])


@router.post("/teach/ask")
async def teach_ask(req: AskRequest):
    """Typed Q&A with the teacher when the live voice agent is not connected.
    Same persona prompt the voice agent gets (sent by the browser), plus ASK_STYLE.
    A model failure returns a natural fallback line (fallback: true), never an error."""
    system = langmod.with_language(req.system + " " + ASK_STYLE, req.lang, female=True, role="teacher")
    msgs = [{"role": "system", "content": system}] + [m.model_dump() for m in req.messages[-12:]]
    try:
        msg, _ = await asyncio.wait_for(
            chat(
                msgs,
                max_tokens=220,
                temperature=0.6,
                tier="reasoning",
                effort="medium",
            ),
            timeout=ATTEMPT_TIMEOUT_S * 1.5,
        )
        text = re.sub(r"[*_`#]+", "", strip_think(msg.get("content") or "")).strip()
    except Exception as e:  # noqa: BLE001 - provider down, timeout, bad body
        log.warning("teach/ask failed: %s", type(e).__name__)
        text = ""
    if not text:
        return {"reply": ask_fallback(req.lang), "fallback": True}
    return {"reply": text[:1200]}
