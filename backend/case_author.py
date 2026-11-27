"""Case-authoring agent: turns a Telegram interview (answers) into a full
MedSim PatientCase (src/game/types.ts), then checks it deterministically.

Flow: one LLM call (llm_provider.chat, JSON mode) -> normalise (the answers the
author gave always win: name, age, sex, severity, specialty) -> validate ->
on errors, ONE repair round that sends the error list back to the LLM ->
deterministic clean-up of what is safe to fix (unknown ids dropped, abnormal
flags matched to simple reference ranges) -> validate again. Anything still
wrong raises CaseAuthorError.

The validation mirrors scripts/verify (data-integrity + clinical-consistency)
for the parts that can be checked without the TS data files.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import catalog
import llm_provider

log = logging.getLogger("medsim.case_author")

EM_DASH = "\u2014"
SEVERITIES = ("stable", "urgent", "critical")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_TESTS = 12
MAX_ANAMNESIS = 14

ChatFn = Callable[..., Awaitable[tuple[dict[str, Any], str]]]


class CaseAuthorError(Exception):
    def __init__(self, message: str, errors: Optional[list[str]] = None):
        super().__init__(message)
        self.errors = errors or []


# ------------------------------------------------------------------ helpers

def new_case_id() -> str:
    return "tg-" + secrets.token_hex(3)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return s[:60].strip("-") or "diagnosis"


def no_em_dash(value: Any) -> Any:
    """Replace em dashes everywhere in a JSON-like value (house style)."""
    if isinstance(value, str):
        return value.replace(" " + EM_DASH + " ", ", ").replace(EM_DASH, "-")
    if isinstance(value, list):
        return [no_em_dash(v) for v in value]
    if isinstance(value, dict):
        return {k: no_em_dash(v) for k, v in value.items()}
    return value


def _contains_em_dash(value: Any) -> bool:
    return EM_DASH in json.dumps(value, ensure_ascii=False)


# ------------------------------------------------------------------ reference ranges
# Simplified from scripts/verify/clinical-consistency.ts.

NUM = r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)"


def _hb(c: dict) -> Optional[tuple[Optional[float], Optional[float]]]:
    age, sex = c["age"], c["gender"]
    if age < 0.5:
        return None
    if age < 5:
        return (11, 14.5)
    if age < 12:
        return (11.5, 15.5)
    if age < 15:
        return (12, 16)
    return (13, 17.5) if sex == "M" else (12, 16)


def _wbc(c: dict) -> tuple[Optional[float], Optional[float]]:
    age = c["age"]
    if age < 1:
        return (6, 17.5)
    if age < 6:
        return (5, 15.5)
    if age < 12:
        return (4.5, 13.5)
    return (4, 11)


def _creat(c: dict) -> tuple[Optional[float], Optional[float]]:
    if c["age"] < 12:
        return (None, 0.8)
    return (None, 1.3) if c["gender"] == "M" else (None, 1.1)


ANALYTES: list[dict[str, Any]] = [
    {"key": "Hb", "tests": {"cbc"}, "re": re.compile(rf"\b(?:Hgb|Hb|Haemoglobin|Hemoglobin)\s*(?:of|:|was)?\s*{NUM}", re.I),
     "range": _hb, "norm": lambda v, t: v / 10 if v > 30 else v},
    {"key": "WBC", "tests": {"cbc"}, "re": re.compile(rf"\b(?:WBC|TLC|white cell count|white cells)\s*(?:of|:)?\s*{NUM}", re.I),
     "range": _wbc, "norm": lambda v, t: v / 1000 if v >= 500 else v},
    {"key": "platelets", "tests": {"cbc"}, "re": re.compile(rf"\b(?:platelets?|plt)\s*(?:count)?\s*(?:of|:)?\s*{NUM}", re.I),
     "range": lambda c: (150, 450), "norm": lambda v, t: v * 100 if re.match(r"^\s*lakh", t, re.I) else (v / 1000 if v >= 2000 else v)},
    {"key": "Na", "tests": {"bmp"}, "re": re.compile(rf"\b(?:Na|sodium)\s*(?:of|:)?\s*{NUM}", re.I), "range": lambda c: (135, 145)},
    {"key": "K", "tests": {"bmp"}, "re": re.compile(rf"\b(?:K|potassium)\s*(?:of|:)?\s*{NUM}"), "range": lambda c: (3.5, 5.1)},
    {"key": "creatinine", "tests": {"bmp"}, "re": re.compile(rf"\b(?:creatinine|Cr)\s*(?:of|:)?\s*{NUM}", re.I),
     "range": _creat, "norm": lambda v, t: v / 88.4 if v > 20 else v},
    {"key": "HbA1c", "tests": {"hba1c"}, "re": re.compile(rf"(?:HbA1c\s*(?:of|:)?\s*)?{NUM}\s*%", re.I), "range": lambda c: (None, 5.6)},
    {"key": "TSH", "tests": {"tsh"}, "re": re.compile(rf"(?:TSH\s*(?:of|:)?\s*)?{NUM}\s*(?:mIU|uIU|µIU|mU)", re.I), "range": lambda c: (0.4, 4.5)},
    {"key": "ALT", "tests": {"lft"}, "re": re.compile(rf"\b(?:ALT|SGPT)\s*(?:of|:)?\s*{NUM}", re.I), "range": lambda c: (None, 40)},
    {"key": "AST", "tests": {"lft"}, "re": re.compile(rf"\b(?:AST|SGOT)\s*(?:of|:)?\s*{NUM}", re.I), "range": lambda c: (None, 40)},
    {"key": "CRP", "tests": {"crp"}, "re": re.compile(rf"(?:CRP\s*(?:of|:)?\s*)?{NUM}\s*mg/L", re.I), "range": lambda c: (None, 5)},
    {"key": "troponin", "tests": {"troponin"}, "re": re.compile(rf"{NUM}\s*ng/(L|mL)", re.I), "range": lambda c: (None, 30)},
]

_GLUCOSE = re.compile(r"\b(fasting|random|post-?prandial|glucose)\b[^0-9]{0,12}([0-9]+(?:\.[0-9]+)?)\s*(mg/dL|mmol/L)?", re.I)


def lab_out_of_range(c: dict, tr: dict) -> Optional[bool]:
    """True/False when a numeric value we understand is out of/in range, None when nothing parsed."""
    text = str(tr.get("result", ""))
    parsed, out = 0, False
    for a in ANALYTES:
        if tr.get("testId") not in a["tests"]:
            continue
        rng = a["range"](c)
        if not rng:
            continue
        lo, hi = rng
        for m in a["re"].finditer(text):
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            tail = text[m.end(): m.end() + 40]
            if a["key"] == "troponin" and m.group(2) == "mL":
                v *= 1000
            if a.get("norm"):
                v = a["norm"](v, tail)
            parsed += 1
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                out = True
    if tr.get("testId") in ("glucose", "bmp"):
        for m in _GLUCOSE.finditer(text):
            v = float(m.group(2))
            if m.group(3) and "mmol" in m.group(3).lower():
                v *= 18
            parsed += 1
            fasting = m.group(1).lower() == "fasting"
            if (v >= 126 if fasting else v >= 200) or v < 60:
                out = True
    return out if parsed else None


def vital_limits(age: float) -> dict[str, tuple[int, int]]:
    if age < 1:
        return {"hr": (80, 220), "rr": (20, 80), "sbp": (50, 120)}
    if age < 6:
        return {"hr": (70, 200), "rr": (15, 60), "sbp": (60, 130)}
    if age < 12:
        return {"hr": (55, 180), "rr": (12, 45), "sbp": (70, 145)}
    return {"hr": (35, 180), "rr": (8, 45), "sbp": (70, 230)}


def _vitals_errors(c: dict) -> list[str]:
    v = c.get("vitals")
    if not isinstance(v, dict):
        return ["vitals must be an object {hr, bp, spo2, temp, rr}"]
    errs: list[str] = []
    lim = vital_limits(c["age"])
    for k in ("hr", "spo2", "temp", "rr"):
        if not isinstance(v.get(k), (int, float)) or isinstance(v.get(k), bool):
            errs.append(f"vitals.{k} must be a number")
    m = re.match(r"^(\d{2,3})/(\d{2,3})$", str(v.get("bp", "")).strip())
    if not m:
        errs.append(f"vitals.bp must look like 120/80, got {v.get('bp')!r}")
    else:
        sbp, dbp = int(m.group(1)), int(m.group(2))
        if not lim["sbp"][0] <= sbp <= lim["sbp"][1]:
            errs.append(f"vitals: implausible systolic BP {sbp} for age {c['age']}")
        if dbp >= sbp or sbp - dbp < 15 or dbp < 25 or dbp > 140:
            errs.append(f"vitals: implausible diastolic BP / pulse pressure {v.get('bp')}")
    if errs:
        return errs
    if not lim["hr"][0] <= v["hr"] <= lim["hr"][1]:
        errs.append(f"vitals: implausible HR {v['hr']} for age {c['age']}")
    if not lim["rr"][0] <= v["rr"] <= lim["rr"][1]:
        errs.append(f"vitals: implausible RR {v['rr']} for age {c['age']}")
    if not 70 <= v["spo2"] <= 100:
        errs.append(f"vitals: implausible SpO2 {v['spo2']}")
    if not 34 <= v["temp"] <= 42:
        errs.append(f"vitals: temp {v['temp']} must be Celsius (34-42)")
    sev = c.get("severity")
    if sev == "stable" and (v["spo2"] < 92 or v["hr"] > 130 or (m and int(m.group(1)) < 90)):
        errs.append("vitals inconsistent with severity 'stable' (SpO2 >= 92, HR <= 130, SBP >= 90 expected)")
    if sev == "critical" and v["spo2"] >= 95 and lim["hr"][0] + 20 <= v["hr"] <= 110 and m and int(m.group(1)) >= 100 and v["rr"] <= 22:
        errs.append("vitals inconsistent with severity 'critical': at least one vital sign should be clearly deranged")
    return errs


# ------------------------------------------------------------------ validation

def validate_case(c: Any) -> list[str]:
    """Deterministic checks. Empty list = playable."""
    if not isinstance(c, dict):
        return ["case must be a JSON object"]
    errs: list[str] = []
    for k in ("id", "name", "arrivalBlurb", "chiefComplaint", "correctDiagnosisId", "specialty"):
        if not isinstance(c.get(k), str) or not c.get(k, "").strip():
            errs.append(f"{k} must be a non-empty string")
    if not isinstance(c.get("age"), (int, float)) or isinstance(c.get("age"), bool) or not 0 <= c["age"] <= 110:
        errs.append("age must be a number between 0 and 110")
        return errs
    if c.get("gender") not in ("M", "F"):
        errs.append("gender must be 'M' or 'F'")
        return errs
    if c.get("severity") not in SEVERITIES:
        errs.append("severity must be one of stable, urgent, critical")
    if c.get("specialty") not in {s["id"] for s in catalog.specialties()}:
        errs.append(f"unknown specialty {c.get('specialty')!r}")
    errs += _vitals_errors(c)

    an = c.get("anamnesis")
    if not isinstance(an, list) or len(an) < 4:
        errs.append("anamnesis must be a list of at least 4 {id, question, answer, relevant} items")
    else:
        ids = set()
        for i, q in enumerate(an):
            if not isinstance(q, dict) or not all(isinstance(q.get(k), str) and q.get(k) for k in ("id", "question", "answer")) \
                    or not isinstance(q.get("relevant"), bool):
                errs.append(f"anamnesis[{i}] needs string id/question/answer and boolean relevant")
                continue
            if q["id"] in ids:
                errs.append(f"duplicate anamnesis id {q['id']}")
            ids.add(q["id"])
        if not any(isinstance(q, dict) and q.get("relevant") is True for q in an):
            errs.append("at least one anamnesis item must be relevant")

    known_tests = catalog.test_ids()
    trs = c.get("testResults")
    if not isinstance(trs, list) or not trs:
        errs.append("testResults must be a non-empty list of {testId, result, abnormal}")
    else:
        seen = set()
        for tr in trs:
            if not isinstance(tr, dict) or not isinstance(tr.get("result"), str) or not isinstance(tr.get("abnormal"), bool):
                errs.append("each testResults item needs testId, result (string) and abnormal (boolean)")
                continue
            tid = tr.get("testId")
            if tid not in known_tests:
                errs.append(f"unknown testId {tid!r} (use only ids from the test list)")
            if tid in seen:
                errs.append(f"duplicate testResults testId {tid}")
            seen.add(tid)
            oor = lab_out_of_range(c, tr)
            if oor is True and not tr["abnormal"]:
                errs.append(f"{tid}: value out of reference range but abnormal=false ({tr['result']})")
            if oor is False and tr["abnormal"] and re.search(r"\bnormal\b", tr["result"], re.I) \
                    and not re.search(r"(ab|not |sub)normal|low|high|elevat|raised|positive|border|mild|reduced", tr["result"], re.I):
                errs.append(f"{tid}: values in range and described normal but abnormal=true")
        if c.get("gender") == "M" and seen & {"urine-hcg", "beta-hcg-q", "pap-smear", "us-pelvis", "mammogram"}:
            errs.append("female-only test result in a male patient")
        if c.get("gender") == "F" and "psa" in seen:
            errs.append("male-only test (psa) in a female patient")

    known_tx = catalog.treatment_ids()
    acc, crit = c.get("acceptableTreatmentIds"), c.get("criticalTreatmentIds")
    if not isinstance(acc, list) or not acc or not all(isinstance(x, str) for x in acc):
        errs.append("acceptableTreatmentIds must be a non-empty list of treatment ids")
        acc = []
    if not isinstance(crit, list) or not all(isinstance(x, str) for x in crit):
        errs.append("criticalTreatmentIds must be a list of treatment ids")
        crit = []
    for t in acc + crit:
        if t not in known_tx:
            errs.append(f"unknown treatment id {t!r} (use only ids from the treatment list)")
    for t in crit:
        if t not in acc:
            errs.append(f"criticalTreatmentIds {t!r} must also be in acceptableTreatmentIds")

    opts, labels = c.get("diagnosisOptions"), c.get("diagnosisLabels")
    if not isinstance(opts, list) or len(opts) != 4 or not all(isinstance(o, str) for o in opts):
        errs.append("diagnosisOptions must be exactly 4 diagnosis ids")
    else:
        if len(set(opts)) != 4:
            errs.append("diagnosisOptions must be distinct")
        if c.get("correctDiagnosisId") not in opts:
            errs.append("correctDiagnosisId must be one of diagnosisOptions")
        for o in opts:
            if not SLUG.match(o):
                errs.append(f"diagnosis id {o!r} must be a lowercase slug like 'acute-appendicitis'")
            if not isinstance(labels, dict) or not isinstance(labels.get(o), str) or not labels.get(o):
                errs.append(f"diagnosisLabels needs a label for {o!r}")
    if _contains_em_dash(c):
        errs.append("text must not contain em dashes")
    return errs


# ------------------------------------------------------------------ normalise + repair

def normalise(raw: dict, answers: dict, case_id: str, author: str) -> dict:
    """Force the author's answers over whatever the LLM wrote, fill provenance."""
    c = dict(raw) if isinstance(raw, dict) else {}
    c["id"] = case_id
    c["name"] = answers["name"]
    c["age"] = answers["age"]
    c["gender"] = answers["gender"]
    c["severity"] = answers["severity"]
    c["specialty"] = answers["specialty"]
    if not c.get("chiefComplaint"):
        c["chiefComplaint"] = answers["complaint"]
    if answers.get("vitals"):
        c["vitals"] = {**(c.get("vitals") if isinstance(c.get("vitals"), dict) else {}), **answers["vitals"]}
    v = c.get("vitals")
    if isinstance(v, dict):
        for k in ("hr", "spo2", "rr"):
            if isinstance(v.get(k), str) and re.fullmatch(r"\d+(\.\d+)?", v[k].strip()):
                v[k] = round(float(v[k]))
        if isinstance(v.get("temp"), str) and re.fullmatch(r"\d+(\.\d+)?", v["temp"].strip()):
            v["temp"] = float(v["temp"])
    # Diagnosis ids: slug them, keep labels in sync.
    labels = c.get("diagnosisLabels") if isinstance(c.get("diagnosisLabels"), dict) else {}
    opts = c.get("diagnosisOptions") if isinstance(c.get("diagnosisOptions"), list) else []
    new_opts, new_labels = [], {}
    for o in opts:
        if isinstance(o, dict):  # {id, label} variant
            oid, lab = o.get("id") or o.get("label") or "", o.get("label") or o.get("id") or ""
        else:
            oid, lab = str(o), str(labels.get(o) or o)
        sid = slugify(oid)
        if sid not in new_opts:
            new_opts.append(sid)
            new_labels[sid] = str(lab).strip() or sid.replace("-", " ").title()
    correct = slugify(c.get("correctDiagnosisId") or answers["diagnosis"])
    if correct not in new_labels:
        new_labels[correct] = answers["diagnosis"]
    c["correctDiagnosisId"] = correct
    c["diagnosisOptions"] = new_opts
    c["diagnosisLabels"] = new_labels
    c["source"] = {"title": f"Created via Telegram by {author or 'a MedSim user'}", "url": ""}
    c["synthetic"] = True
    c["createdVia"] = "telegram"
    c["createdAt"] = datetime.now(timezone.utc).isoformat()
    return no_em_dash(c)


def deterministic_fix(c: dict) -> dict:
    """Safe, mechanical fixes after the repair round."""
    known_tests, known_tx = catalog.test_ids(), catalog.treatment_ids()
    trs, seen = [], set()
    for tr in c.get("testResults") or []:
        if isinstance(tr, dict) and tr.get("testId") in known_tests and tr["testId"] not in seen \
                and isinstance(tr.get("result"), str):
            tr = {"testId": tr["testId"], "result": tr["result"], "abnormal": bool(tr.get("abnormal"))}
            if lab_out_of_range(c, tr) is True:
                tr["abnormal"] = True
            seen.add(tr["testId"])
            trs.append(tr)
    c["testResults"] = trs[:MAX_TESTS]
    acc = [t for t in (c.get("acceptableTreatmentIds") or []) if isinstance(t, str) and t in known_tx]
    crit = [t for t in (c.get("criticalTreatmentIds") or []) if isinstance(t, str) and t in known_tx]
    c["acceptableTreatmentIds"] = list(dict.fromkeys(acc + crit))
    c["criticalTreatmentIds"] = list(dict.fromkeys(crit))
    opts = list(c.get("diagnosisOptions") or [])
    if c.get("correctDiagnosisId") and c["correctDiagnosisId"] not in opts:
        opts = [c["correctDiagnosisId"]] + opts[:3]
    c["diagnosisOptions"] = opts[:4]
    c["anamnesis"] = (c.get("anamnesis") or [])[:MAX_ANAMNESIS]
    return no_em_dash(c)


# ------------------------------------------------------------------ prompt

def _catalog_text() -> str:
    cat = catalog.load()
    tests = "; ".join(f"{t['id']} ({t['name']})" for t in cat["tests"])
    tx = "; ".join(f"{t['id']} ({t['name']})" for t in cat["treatments"])
    return f"TEST IDS: {tests}\n\nTREATMENT IDS: {tx}"


SYSTEM = """You write synthetic outpatient teaching cases for MedSim, a clinical training simulator.
Return ONE JSON object (no prose) with exactly these keys:
{
 "arrivalBlurb": "one sentence of how the patient arrives and looks",
 "chiefComplaint": "the complaint in the patient's own words, first person",
 "vitals": {"hr": int, "bp": "SYS/DIA", "spo2": int, "temp": float Celsius, "rr": int},
 "anamnesis": [{"id": "short-slug", "question": "doctor's question", "answer": "patient's first-person answer", "relevant": true|false}],
 "testResults": [{"testId": "<id from TEST IDS>", "result": "result text with units", "abnormal": true|false}],
 "correctDiagnosisId": "slug",
 "diagnosisOptions": ["slug", "slug", "slug", "slug"],
 "diagnosisLabels": {"slug": "Readable Label"},
 "acceptableTreatmentIds": ["<id from TREATMENT IDS>"],
 "criticalTreatmentIds": ["<subset of acceptableTreatmentIds>"]
}
Rules:
- 8 to 12 anamnesis items: onset, duration, character, associated symptoms, the key history points,
  allergies, current medicines, past history, family/social history, and 2 to 3 irrelevant ones (relevant=false).
- 4 to 8 testResults, using ONLY test ids from TEST IDS. Pick tests that confirm the intended diagnosis
  plus a couple that are normal. abnormal=true only when the value is outside the usual adult/child reference range.
- Vitals must fit the severity: stable = near normal, urgent = mildly deranged, critical = clearly deranged.
  Vitals must be plausible for the patient's age.
- diagnosisOptions: exactly 4 distinct lowercase hyphenated slugs including correctDiagnosisId, the other 3 plausible
  differentials. Give every option a label in diagnosisLabels.
- Treatments ONLY from TREATMENT IDS; criticalTreatmentIds is a subset (may be empty for stable cases).
- Use the author's facts exactly (name, age, sex, complaint, allergies, medicines, diagnosis).
- Plain hyphens or commas only, never the em dash character. Synthetic data, no real people."""


def build_messages(answers: dict) -> list[dict[str, str]]:
    facts = {k: answers.get(k) for k in ("specialty", "name", "age", "gender", "complaint", "severity", "onset",
                                          "history", "allergies", "medications", "diagnosis", "vitals")}
    facts["specialty"] = catalog.specialty_label(answers["specialty"])
    return [
        {"role": "system", "content": SYSTEM + "\n\n" + _catalog_text()},
        {"role": "user", "content": "Author's case facts (JSON):\n" + json.dumps(facts, ensure_ascii=False)},
    ]


def parse_json(text: str) -> dict:
    t = llm_provider.strip_think(text or "")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        v = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise CaseAuthorError("The case writer did not return JSON")
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise CaseAuthorError("The case writer returned malformed JSON") from e
    if not isinstance(v, dict):
        raise CaseAuthorError("The case writer did not return a JSON object")
    return v


async def generate_case(answers: dict, author: str = "", *, chat: Optional[ChatFn] = None,
                        case_id: Optional[str] = None) -> dict:
    """Answers -> validated PatientCase dict (+ specialty, diagnosisLabels, source, synthetic)."""
    chat = chat or llm_provider.chat
    cid = case_id or new_case_id()
    messages = build_messages(answers)
    msg, _ = await chat(
        messages,
        json_mode=True,
        max_tokens=3000,
        temperature=0.4,
        tier="reasoning",
        effort="medium",
    )
    raw = msg.get("content") or ""
    case = normalise(parse_json(raw), answers, cid, author)
    errors = validate_case(case)
    if errors:
        log.info("case %s: %d validation errors, repair round", cid, len(errors))
        repair = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "That case failed validation. Fix every error and return the full corrected JSON object:\n- "
                                        + "\n- ".join(errors[:25])},
        ]
        try:
            msg2, _ = await chat(
                repair,
                json_mode=True,
                max_tokens=3000,
                temperature=0.2,
                tier="reasoning",
                effort="medium",
            )
            case = normalise(parse_json(msg2.get("content") or ""), answers, cid, author)
        except CaseAuthorError:
            pass  # keep the first draft, try the mechanical fixes below
        errors = validate_case(case)
    if errors:
        case = deterministic_fix(case)
        errors = validate_case(case)
    if errors:
        raise CaseAuthorError("The generated case did not pass the checks", errors)
    return case
