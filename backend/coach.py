"""Mistake coach: POST /agent/coach/check and POST /agent/coach/explain.

"Nurse Asha" watches the trainee's encounter and flags real mistakes only
(allergy conflict, contraindication, dangerous dose, missed red flag,
prescribing before any history, a harmful or irrelevant test, anchoring on a
wrong diagnosis). Two cheap deterministic rules run first; everything else
goes to the OpenAI-compatible provider powered by Gemini in JSON
mode with a short timeout. Any failure degrades to "no alert": the coach is
advisory and must never block the encounter. Keys are never logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from llm_provider import chat, strip_think
from nurse import ALLERGY_CLASSES, _norm
import lang as langmod

log = logging.getLogger("medsim.coach")
router = APIRouter()

COACH_TIMEOUT_S = 8.0
EM_DASH = chr(0x2014)
SEVERITIES = ("info", "warning", "danger")


# ── request / response models ──────────────────────────────────────────────


class QA(BaseModel):
    question: str = ""
    answer: str = ""


class CoachPatient(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    age: Optional[float] = None
    gender: str = ""
    chiefComplaint: str = ""
    allergies: Optional[str] = None
    anamnesis: list[QA] = Field(default_factory=list)
    vitals: dict[str, Any] = Field(default_factory=dict)


class CoachEvent(BaseModel):
    kind: Literal["order", "prescription", "diagnosis", "utterance"]
    detail: str = ""


class EncounterSoFar(BaseModel):
    model_config = ConfigDict(extra="ignore")

    orderedTests: list[str] = Field(default_factory=list)
    prescriptions: list[str] = Field(default_factory=list)
    askedHistory: bool = False
    transcriptTail: list[str] = Field(default_factory=list)
    examsPerformed: list[str] = Field(default_factory=list)


class CoachCheckRequest(BaseModel):
    caseId: str = ""
    event: CoachEvent
    patient: CoachPatient = Field(default_factory=CoachPatient)
    correctDiagnosis: Optional[str] = None
    encounterSoFar: EncounterSoFar = Field(default_factory=EncounterSoFar)
    lang: Optional[str] = None


class CoachAlert(BaseModel):
    id: str
    severity: Literal["info", "warning", "danger"]
    title: str
    short: str
    explanation: str
    suggestion: str = ""


class CoachCheckResponse(BaseModel):
    alert: Optional[CoachAlert] = None


class CoachExplainRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alert: dict[str, Any] = Field(default_factory=dict)
    question: Optional[str] = None
    caseId: str = ""
    patient: CoachPatient = Field(default_factory=CoachPatient)
    lang: Optional[str] = None


class CoachExplainResponse(BaseModel):
    answer: str


# ── text helpers ───────────────────────────────────────────────────────────


def _clean(text: Any) -> str:
    s = re.sub(r"\s*" + EM_DASH + r"\s*", ", ", str(text or "")).replace(chr(0x2013), "-")
    return re.sub(r"\s+", " ", s).strip()


def _words(text: Any, limit: int) -> str:
    parts = _clean(text).split(" ")
    if len(parts) <= limit:
        return " ".join(p for p in parts if p)
    return " ".join(parts[:limit]).rstrip(",;:") + "..."


def _new_id() -> str:
    return f"ca-{secrets.token_hex(5)}"


def make_alert(severity: str, title: str, short: str, explanation: str, suggestion: str = "") -> CoachAlert:
    return CoachAlert(
        id=_new_id(),
        severity=severity if severity in SEVERITIES else "warning",  # type: ignore[arg-type]
        title=_words(title, 6),
        short=_words(short, 18),
        explanation=_clean(explanation)[:700],
        suggestion=_words(suggestion, 30),
    )


def coerce_alert(raw: Any) -> Optional[CoachAlert]:
    """Turn model JSON into an alert, or None. Anything malformed is None."""
    if not isinstance(raw, dict):
        return None
    a = raw.get("alert", raw if "title" in raw else None)
    if not isinstance(a, dict):
        return None
    title, short, explanation = _clean(a.get("title")), _clean(a.get("short")), _clean(a.get("explanation"))
    if not title or not (short or explanation):
        return None
    sev = _clean(a.get("severity")).lower()
    return make_alert(
        sev if sev in SEVERITIES else "warning",
        title,
        short or explanation,
        explanation or short,
        a.get("suggestion") or "",
    )


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


# ── deterministic rules (no LLM) ───────────────────────────────────────────

_NEGATION = re.compile(r"\b(no|not|none|nkda|denies|deny|never)\b")


def _allergy_sentences(p: CoachPatient) -> list[str]:
    raw = [p.allergies or ""]
    raw += [qa.answer for qa in p.anamnesis if "allerg" in f"{qa.question} {qa.answer}".lower()]
    out: list[str] = []
    for chunk in raw:
        for sent in re.split(r"[.;\n]|\bbut\b", chunk or "", flags=re.IGNORECASE):
            s = _norm(sent)
            if s and not _NEGATION.search(s):
                out.append(s)
    return out


def allergy_conflict(p: CoachPatient, drug: str) -> Optional[str]:
    """Allergy class the drug falls into, if the patient's documented allergy covers it."""
    d = _norm(drug)
    if not d:
        return None
    for sent in _allergy_sentences(p):
        for cls, members in ALLERGY_CLASSES.items():
            if cls in sent or any(m in sent for m in members):
                if any(m in d for m in members):
                    return cls
    return None


# Hindi / Kannada text for the deterministic rule alerts: (title, short, explanation, suggestion).
RULE_TEXT: dict[str, dict[str, tuple[str, ...]]] = {
    "hi": {
        "allergy": (
            "एलर्जी का टकराव",
            "{drug} मरीज़ की दर्ज {cls} एलर्जी से टकराता है।",
            "मरीज़ को {cls} से एलर्जी दर्ज है और {drug} उसी दवा वर्ग की है। दवा लिखने से पहले हमेशा एलर्जी जाँचिए, "
            "क्योंकि एक ही वर्ग की दवाओं में क्रॉस रिएक्शन से एनाफिलेक्सिस हो सकता है। इसे रद्द कीजिए और दूसरे वर्ग की दवा चुनिए।",
            "यह दवा रोकिए और दूसरे वर्ग की सुरक्षित दवा चुनिए।",
        ),
        "history": (
            "हिस्ट्री से पहले दवा",
            "आपने मरीज़ की कोई हिस्ट्री लिए बिना दवा लिख दी।",
            "इलाज से पहले जाँच ज़रूरी है। हिस्ट्री के बिना आपको एलर्जी, चल रही दवाएँ, गर्भावस्था या रेड फ्लैग का पता नहीं होता, "
            "जो तय करते हैं कि क्या सुरक्षित है। पहले सही सवाल पूछिए, फिर दवा लिखिए।",
            "दवा लिखने से पहले एलर्जी, चल रही दवाएँ और मुख्य तकलीफ़ के बारे में पूछिए।",
        ),
    },
    "kn": {
        "allergy": (
            "ಅಲರ್ಜಿ ಸಂಘರ್ಷ",
            "{drug} ರೋಗಿಯ ದಾಖಲಾದ {cls} ಅಲರ್ಜಿಗೆ ವಿರುದ್ಧವಾಗಿದೆ.",
            "ರೋಗಿಗೆ {cls} ಅಲರ್ಜಿ ದಾಖಲಾಗಿದೆ ಮತ್ತು {drug} ಅದೇ ಔಷಧ ವರ್ಗಕ್ಕೆ ಸೇರುತ್ತದೆ. ಔಷಧ ಬರೆಯುವ ಮುನ್ನ ಯಾವಾಗಲೂ ಅಲರ್ಜಿ "
            "ಪರೀಕ್ಷಿಸಿ, ಏಕೆಂದರೆ ಒಂದೇ ವರ್ಗದ ಔಷಧಗಳಲ್ಲಿ ಕ್ರಾಸ್ ರಿಯಾಕ್ಷನ್‌ನಿಂದ ಅನಾಫಿಲ್ಯಾಕ್ಸಿಸ್ ಆಗಬಹುದು. ಇದನ್ನು ರದ್ದುಮಾಡಿ, ಬೇರೆ ವರ್ಗದ ಔಷಧ ಆರಿಸಿ.",
            "ಈ ಔಷಧ ನಿಲ್ಲಿಸಿ, ಬೇರೆ ವರ್ಗದ ಸುರಕ್ಷಿತ ಔಷಧ ಆರಿಸಿ.",
        ),
        "history": (
            "ಹಿಸ್ಟರಿ ಮೊದಲು ಔಷಧ",
            "ನೀವು ರೋಗಿಯ ಹಿಸ್ಟರಿ ತೆಗೆದುಕೊಳ್ಳದೆ ಔಷಧ ಬರೆದಿದ್ದೀರಿ.",
            "ಚಿಕಿತ್ಸೆಗೂ ಮೊದಲು ಪರೀಕ್ಷೆ ಅಗತ್ಯ. ಹಿಸ್ಟರಿ ಇಲ್ಲದೆ ಅಲರ್ಜಿ, ನಡೆಯುತ್ತಿರುವ ಔಷಧಗಳು, ಗರ್ಭಧಾರಣೆ ಅಥವಾ ರೆಡ್ ಫ್ಲ್ಯಾಗ್ ಗೊತ್ತಾಗುವುದಿಲ್ಲ, "
            "ಇವು ಯಾವುದು ಸುರಕ್ಷಿತ ಎಂದು ನಿರ್ಧರಿಸುತ್ತವೆ. ಮೊದಲು ಸರಿಯಾದ ಪ್ರಶ್ನೆ ಕೇಳಿ, ನಂತರ ಔಷಧ ಬರೆಯಿರಿ.",
            "ಔಷಧ ಬರೆಯುವ ಮುನ್ನ ಅಲರ್ಜಿ, ಈಗಿನ ಔಷಧಗಳು ಮತ್ತು ಮುಖ್ಯ ತೊಂದರೆ ಕುರಿತು ಕೇಳಿ.",
        ),
    },
    "ta": {
        "allergy": (
            "ஒவ்வாமை முரண்பாடு",
            "{drug} நோயாளியின் பதிவான {cls} ஒவ்வாமைக்கு முரணானது.",
            "நோயாளிக்கு {cls} ஒவ்வாமை பதிவாகியுள்ளது; {drug} அதே மருந்து வகையைச் சேர்ந்தது. மருந்து எழுதும் முன் எப்போதும் ஒவ்வாமையைச் சரிபாருங்கள், "
            "ஏனெனில் ஒரே வகை மருந்துகளில் குறுக்கு எதிர்வினையால் அனாபிலாக்ஸிஸ் ஏற்படலாம். இதை ரத்து செய்து வேறு வகை மருந்தைத் தேர்ந்தெடுங்கள்.",
            "இந்த மருந்தை நிறுத்தி, வேறு வகையின் பாதுகாப்பான மருந்தைத் தேர்ந்தெடுங்கள்.",
        ),
        "history": (
            "வரலாறு எடுக்கும் முன் மருந்து",
            "நோயாளியின் வரலாற்றை எடுக்காமல் மருந்து எழுதிவிட்டீர்கள்.",
            "சிகிச்சைக்கு முன் பரிசோதனை அவசியம். வரலாறு இல்லாமல் ஒவ்வாமை, நடப்பில் உள்ள மருந்துகள், கர்ப்பம் அல்லது ரெட் ஃபிளாக் தெரியாது; "
            "எது பாதுகாப்பானது என்பதை இவையே தீர்மானிக்கும். முதலில் சரியான கேள்விகளைக் கேளுங்கள், பிறகு மருந்து எழுதுங்கள்.",
            "மருந்து எழுதும் முன் ஒவ்வாமை, நடப்பு மருந்துகள், முக்கிய தொந்தரவு பற்றிக் கேளுங்கள்.",
        ),
    },
    "te": {
        "allergy": (
            "అలెర్జీ వైరుధ్యం",
            "{drug} రోగి నమోదు చేసిన {cls} అలెర్జీకి విరుద్ధం.",
            "రోగికి {cls} అలెర్జీ నమోదై ఉంది, {drug} అదే మందు వర్గానికి చెందినది. మందు రాసే ముందు ఎల్లప్పుడూ అలెర్జీ చూడండి, "
            "ఎందుకంటే ఒకే వర్గం మందులలో క్రాస్ రియాక్షన్ వల్ల అనాఫిలాక్సిస్ రావచ్చు. దీన్ని రద్దు చేసి వేరే వర్గం మందు ఎంచుకోండి.",
            "ఈ మందు ఆపి, వేరే వర్గంలోని సురక్షితమైన మందు ఎంచుకోండి.",
        ),
        "history": (
            "హిస్టరీ ముందు మందు",
            "మీరు రోగి హిస్టరీ తీసుకోకుండా మందు రాశారు.",
            "చికిత్సకు ముందు పరీక్ష అవసరం. హిస్టరీ లేకుండా అలెర్జీ, వాడుతున్న మందులు, గర్భం లేదా రెడ్ ఫ్లాగ్ తెలియదు; "
            "ఏది సురక్షితమో అవే నిర్ణయిస్తాయి. ముందు సరైన ప్రశ్నలు అడగండి, తర్వాత మందు రాయండి.",
            "మందు రాసే ముందు అలెర్జీ, ప్రస్తుత మందులు, ముఖ్య సమస్య గురించి అడగండి.",
        ),
    },
    "mr": {
        "allergy": (
            "ॲलर्जीचा संघर्ष",
            "{drug} रुग्णाच्या नोंदवलेल्या {cls} ॲलर्जीशी विसंगत आहे.",
            "रुग्णाला {cls} ॲलर्जी नोंदवलेली आहे आणि {drug} त्याच औषधवर्गातील आहे. औषध लिहिण्यापूर्वी नेहमी ॲलर्जी तपासा, "
            "कारण एकाच वर्गातील औषधांमध्ये क्रॉस रिॲक्शनमुळे ॲनाफायलॅक्सिस होऊ शकतो. हे रद्द करा आणि दुसऱ्या वर्गातील औषध निवडा.",
            "हे औषध थांबवा आणि दुसऱ्या वर्गातील सुरक्षित औषध निवडा.",
        ),
        "history": (
            "हिस्ट्रीपूर्वी औषध",
            "तुम्ही रुग्णाची हिस्ट्री न घेता औषध लिहिले आहे.",
            "उपचारापूर्वी तपासणी आवश्यक आहे. हिस्ट्रीशिवाय ॲलर्जी, चालू औषधे, गर्भधारणा किंवा रेड फ्लॅग कळत नाहीत; "
            "काय सुरक्षित आहे ते यावरच ठरते. आधी योग्य प्रश्न विचारा, मग औषध लिहा.",
            "औषध लिहिण्यापूर्वी ॲलर्जी, चालू औषधे आणि मुख्य त्रासाबद्दल विचारा.",
        ),
    },
    "bn": {
        "allergy": (
            "অ্যালার্জির সংঘাত",
            "{drug} রোগীর নথিভুক্ত {cls} অ্যালার্জির সঙ্গে সাংঘর্ষিক।",
            "রোগীর {cls} অ্যালার্জি নথিভুক্ত আছে এবং {drug} একই ওষুধ শ্রেণির। ওষুধ লেখার আগে সবসময় অ্যালার্জি দেখে নিন, "
            "কারণ একই শ্রেণির ওষুধে ক্রস রিঅ্যাকশনের ফলে অ্যানাফিল্যাক্সিস হতে পারে। এটি বাতিল করে অন্য শ্রেণির ওষুধ বেছে নিন।",
            "এই ওষুধ বন্ধ করে অন্য শ্রেণির নিরাপদ ওষুধ বেছে নিন।",
        ),
        "history": (
            "হিস্ট্রির আগে ওষুধ",
            "আপনি রোগীর হিস্ট্রি না নিয়েই ওষুধ লিখেছেন।",
            "চিকিৎসার আগে পরীক্ষা জরুরি। হিস্ট্রি ছাড়া অ্যালার্জি, চলমান ওষুধ, গর্ভাবস্থা বা রেড ফ্ল্যাগ জানা যায় না; "
            "কোনটা নিরাপদ তা এগুলোই ঠিক করে। আগে সঠিক প্রশ্ন করুন, তারপর ওষুধ লিখুন।",
            "ওষুধ লেখার আগে অ্যালার্জি, চলমান ওষুধ ও প্রধান সমস্যা সম্পর্কে জিজ্ঞাসা করুন।",
        ),
    },
}


def rule_alert(req: CoachCheckRequest) -> Optional[CoachAlert]:
    ev = req.event
    if ev.kind == "prescription":
        cls = allergy_conflict(req.patient, ev.detail)
        lg = langmod.norm_lang(req.lang)
        if cls and lg in RULE_TEXT:
            t = RULE_TEXT[lg]["allergy"]
            drug = _words(ev.detail, 6)
            return make_alert("danger", t[0], t[1].format(drug=drug, cls=cls), t[2].format(drug=drug, cls=cls), t[3])
        if cls:
            return make_alert(
                "danger",
                "Allergy conflict",
                f"{_words(ev.detail, 6)} conflicts with a documented {cls} allergy.",
                f"The patient has a documented {cls} allergy and {_words(ev.detail, 6)} belongs to that drug class. "
                "Always check allergies before prescribing, because cross-reactivity within a class can cause "
                "anaphylaxis. Cancel it and choose an agent from a different class.",
                "Stop this drug and pick a safe alternative from another class.",
            )
        if not req.encounterSoFar.askedHistory and lg in RULE_TEXT:
            return make_alert("warning", *RULE_TEXT[lg]["history"])
        if not req.encounterSoFar.askedHistory:
            return make_alert(
                "warning",
                "Prescribing before history",
                "You prescribed before taking any history from the patient.",
                "Treatment should follow an assessment. Without a history you do not know the allergies, current "
                "medicines, pregnancy status or red flags that change what is safe to give. Ask a focused history "
                "first, then prescribe.",
                "Ask about allergies, current medicines and the main symptom before prescribing.",
            )
    return None


def needs_llm(req: CoachCheckRequest) -> bool:
    ev = req.event
    if not _clean(ev.detail):
        return False
    if ev.kind == "diagnosis" and req.correctDiagnosis and _norm(ev.detail) == _norm(req.correctDiagnosis):
        return False  # correct diagnosis, nothing to flag
    return True


# ── LLM check ──────────────────────────────────────────────────────────────

CHECK_SYSTEM = (
    "You are Nurse Asha, a senior nurse educator quietly watching a trainee doctor in a SYNTHETIC clinical "
    "training simulation. Decide whether the trainee's LATEST EVENT is a real clinical mistake.\n"
    "Be conservative: most events are fine and the right answer is usually {\"alert\": null}. Only flag clear "
    "problems of these kinds: allergy conflict, contraindication or dangerous interaction, dangerous dose, a "
    "missed or ignored red flag, prescribing before any history, ordering a test that is irrelevant to the "
    "complaint or harmful, or anchoring on a wrong diagnosis that the findings do not support. Never flag style, "
    "reasonable extra tests, normal questions or small talk.\n"
    "For a diagnosis event that does not match the correct diagnosis, say which findings do not fit; you may "
    "name the better diagnosis.\n"
    "Reply with JSON only, one of:\n"
    "{\"alert\": null}\n"
    "{\"alert\": {\"severity\": \"info\"|\"warning\"|\"danger\", \"title\": \"max 6 words\", "
    "\"short\": \"max 18 words\", \"explanation\": \"2-4 sentences, warm teaching tone, cite the principle\", "
    "\"suggestion\": \"one concrete next step\"}}\n"
    "Use danger only for things that could seriously harm the patient. Plain words, no markdown, no em dashes."
)


def _check_user(req: CoachCheckRequest) -> str:
    p = req.patient
    return json.dumps(
        {
            "latestEvent": req.event.model_dump(),
            "patient": {
                "name": p.name, "age": p.age, "gender": p.gender, "chiefComplaint": p.chiefComplaint,
                "allergies": p.allergies or "not recorded", "vitals": p.vitals,
                "anamnesis": [qa.model_dump() for qa in p.anamnesis][:24],
            },
            "correctDiagnosis": req.correctDiagnosis or "unknown",
            "encounterSoFar": {
                "orderedTests": req.encounterSoFar.orderedTests[-20:],
                "prescriptions": req.encounterSoFar.prescriptions[-12:],
                "askedHistory": req.encounterSoFar.askedHistory,
                "examsPerformed": [_words(e, 40) for e in req.encounterSoFar.examsPerformed[-12:]],
                "transcriptTail": [_words(t, 60) for t in req.encounterSoFar.transcriptTail[-10:]],
            },
        },
        default=str,
    )[:12000]


async def _llm(messages: list[dict[str, Any]], *, json_mode: bool, max_tokens: int) -> Optional[str]:
    """One bounded LLM call. Returns content, or None on any failure/timeout."""
    try:
        msg, _ = await asyncio.wait_for(
            chat(
                messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                temperature=0.1,
                tier="reasoning",
                effort="medium",
            ),
            timeout=COACH_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.warning("coach LLM timed out after %.0fs", COACH_TIMEOUT_S)
        return None
    except HTTPException as e:
        log.warning("coach LLM unavailable (HTTP %s)", e.status_code)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("coach LLM failed: %s", type(e).__name__)
        return None
    return (msg or {}).get("content") or ""


@router.post("/agent/coach/check", response_model=CoachCheckResponse)
async def coach_check(req: CoachCheckRequest) -> CoachCheckResponse:
    hit = rule_alert(req)
    if hit:
        return CoachCheckResponse(alert=hit)
    if not needs_llm(req):
        return CoachCheckResponse(alert=None)
    content = await _llm(
        [{"role": "system", "content": langmod.with_language(CHECK_SYSTEM, req.lang, female=True, role="coach")}, {"role": "user", "content": _check_user(req)}],
        json_mode=True,
        max_tokens=500,
    )
    if content is None:
        return CoachCheckResponse(alert=None)
    return CoachCheckResponse(alert=coerce_alert(parse_model_json(content)))


# ── explain / follow-up ────────────────────────────────────────────────────

EXPLAIN_SYSTEM = (
    "You are Nurse Asha, a warm, experienced Indian senior nurse educator in a SYNTHETIC clinical training "
    "simulation. The trainee doctor is asking about an alert you raised. Answer in at most 80 words, spoken "
    "style: short plain sentences you would say out loud, no lists, no markdown, no em dashes. Explain the "
    "principle and what to do next. Stay on the topic of the alert and this patient."
)


def _fallback_answer(alert: dict[str, Any]) -> str:
    base = " ".join(x for x in (_clean(alert.get("explanation")), _clean(alert.get("suggestion"))) if x)
    return _words(base or "I flagged this because it could put the patient at risk. Let's review it together.", 80)


@router.post("/agent/coach/explain", response_model=CoachExplainResponse)
async def coach_explain(req: CoachExplainRequest) -> CoachExplainResponse:
    a = req.alert or {}
    question = _clean(req.question) or "Why is that wrong?"
    p = req.patient
    user = json.dumps(
        {
            "alert": {k: _clean(a.get(k)) for k in ("severity", "title", "short", "explanation", "suggestion")},
            "patient": {"name": p.name, "age": p.age, "gender": p.gender, "chiefComplaint": p.chiefComplaint,
                        "allergies": p.allergies or "not recorded"},
            "question": _words(question, 60),
        },
        default=str,
    )[:6000]
    content = await _llm(
        [{"role": "system", "content": langmod.with_language(EXPLAIN_SYSTEM, req.lang, female=True, role="coach")}, {"role": "user", "content": user}],
        json_mode=False,
        max_tokens=300,
    )
    answer = _words(strip_think(content or ""), 80) if content else ""
    return CoachExplainResponse(answer=answer or _fallback_answer(a))


# ── "How am I doing, Asha?" check-in ───────────────────────────────────────


class CoachHintRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    caseId: str = ""
    patient: CoachPatient = Field(default_factory=CoachPatient)
    correctDiagnosis: Optional[str] = None
    keyTests: list[str] = Field(default_factory=list)
    goodTreatments: list[str] = Field(default_factory=list)
    encounterSoFar: EncounterSoFar = Field(default_factory=EncounterSoFar)
    lang: Optional[str] = None


class CoachHintResponse(BaseModel):
    verdict: Literal["on_track", "partly", "off_track"]
    hint: str


HINT_SYSTEM = (
    "You are Nurse Asha, a warm senior nurse educator in a SYNTHETIC clinical training simulation. The trainee "
    "doctor just asked you 'How am I doing?'. You can see the hidden answer key (diagnosis, key tests, good "
    "treatments) and what the doctor has done so far. Give honest, Socratic feedback WITHOUT revealing the "
    "diagnosis name, the exact test names they are missing, or the drug names they should give. Say one specific "
    "thing they did well (if any), then one guiding question or nudge toward the most important next step "
    "(for example: more history about a symptom area, an examination, a category of test, reconsidering an "
    "order that does not fit). If an order or drug they gave does not fit the case, say so gently and why. "
    "Spoken style, at most 45 words, no lists, no markdown, no em dashes. "
    'Return JSON only: {"verdict": "on_track" | "partly" | "off_track", "hint": "<what you say>"}'
)


def fallback_hint(req: CoachHintRequest) -> CoachHintResponse:
    e = req.encounterSoFar
    if not e.askedHistory:
        return CoachHintResponse(verdict="partly", hint="We've only just started, doctor. Take a fuller history first: when it began, what makes it better or worse, and anything else going on.")
    if not e.examsPerformed:
        return CoachHintResponse(verdict="partly", hint="Good history so far. Have you examined the patient yet? Pick the body system that fits the complaint.")
    if not e.orderedTests:
        return CoachHintResponse(verdict="partly", hint="You've taken a history and examined. Which investigations would confirm or rule out your leading idea?")
    return CoachHintResponse(verdict="partly", hint="You have history, an exam and some results. Look at which results are abnormal and ask yourself what single diagnosis explains them.")


@router.post("/agent/coach/hint", response_model=CoachHintResponse)
async def coach_hint(req: CoachHintRequest) -> CoachHintResponse:
    p = req.patient
    e = req.encounterSoFar
    user = json.dumps(
        {
            "answerKey": {
                "diagnosis": _clean(req.correctDiagnosis) or "not given",
                "keyTests": [_words(t, 12) for t in req.keyTests[:10]],
                "goodTreatments": [_words(t, 12) for t in req.goodTreatments[:10]],
            },
            "patient": {"age": p.age, "gender": p.gender, "chiefComplaint": p.chiefComplaint,
                        "allergies": p.allergies or "not recorded", "vitals": p.vitals},
            "doneSoFar": {
                "orderedTests": e.orderedTests[:20],
                "prescriptions": e.prescriptions[:10],
                "examsPerformed": e.examsPerformed[:10],
                "transcriptTail": [_words(t, 60) for t in e.transcriptTail[-12:]],
            },
        },
        default=str,
    )[:12000]
    content = await _llm(
        [{"role": "system", "content": langmod.with_language(HINT_SYSTEM, req.lang, female=True, role="coach")},
         {"role": "user", "content": user}],
        json_mode=True,
        max_tokens=400,
    )
    data = parse_model_json(content or "") if content else None
    if not data or not _clean(data.get("hint")):
        return fallback_hint(req)
    verdict = data.get("verdict") if data.get("verdict") in ("on_track", "partly", "off_track") else "partly"
    hint = _words(_clean(data.get("hint")).replace(EM_DASH, ", "), 60)
    # Never leak the answer: drop the hint if it names the diagnosis outright.
    dx = _clean(req.correctDiagnosis).lower()
    if dx and len(dx) > 4 and dx in hint.lower():
        return fallback_hint(req)
    return CoachHintResponse(verdict=verdict, hint=hint)
