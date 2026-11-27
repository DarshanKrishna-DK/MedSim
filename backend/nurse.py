"""Agentic charge nurse: POST /agent/nurse/act.

An OpenAI-compatible tool-use loop powered by Gemini that turns the
doctor's spoken orders into executed actions. Tools run server-side against the
request payload only; nothing is fabricated beyond tool output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm_provider import chat, strip_think
import lang as langmod

log = logging.getLogger("medsim.nurse")
router = APIRouter()
MAX_ITERATIONS = 6


# ── request / response models ──────────────────────────────────────────────


class QA(BaseModel):
    question: str = ""
    answer: str = ""


class TestResultIn(BaseModel):
    testId: str
    result: str = ""
    abnormal: bool = False


class PatientIn(BaseModel):
    name: str = ""
    age: Optional[float] = None
    gender: str = ""
    chiefComplaint: str = ""
    allergies: Optional[str] = None
    anamnesis: list[QA] = Field(default_factory=list)
    vitals: dict[str, Any] = Field(default_factory=dict)
    testResults: list[TestResultIn] = Field(default_factory=list)


class CatalogTest(BaseModel):
    id: str
    name: str
    category: str = ""


class CatalogMed(BaseModel):
    id: str
    name: str
    dosing: Optional[str] = None


class Catalog(BaseModel):
    tests: list[CatalogTest] = Field(default_factory=list)
    medications: list[CatalogMed] = Field(default_factory=list)


class NurseActRequest(BaseModel):
    caseId: str
    utterance: str
    patient: PatientIn
    catalog: Catalog
    alreadyOrdered: list[str] = Field(default_factory=list)
    alreadyPrescribed: list[str] = Field(default_factory=list)
    lang: Optional[str] = None
    # Real-world delivery (send_to_patient): the patient's phone or email for this
    # demo encounter, and the doctor's display name for the message.
    contact: Optional[str] = None
    doctorName: Optional[str] = None
    # Everything already on the chart, so "send it to the patient" can include
    # prescriptions and the follow-up from earlier orders.
    prescriptionsSoFar: list[dict[str, Any]] = Field(default_factory=list)
    followUpSoFar: Optional[dict[str, Any]] = None
    # Logbook (log_case) and referral (refer_patient) context.
    student: Optional[str] = None
    university: Optional[str] = None
    specialty: Optional[str] = None
    caseName: Optional[str] = None
    diagnosis: Optional[str] = None
    skills: list[str] = Field(default_factory=list)


# ── tool executors (pure, no network) ──────────────────────────────────────

# Drug-class keywords so "penicillin allergy" blocks amoxicillin etc.
ALLERGY_CLASSES: dict[str, list[str]] = {
    "penicillin": ["penicillin", "amoxicillin", "ampicillin", "piperacillin", "flucloxacillin", "augmentin", "co-amoxiclav", "cloxacillin"],
    "sulfa": ["sulfa", "sulfamethoxazole", "trimethoprim", "cotrimoxazole", "co-trimoxazole", "bactrim"],
    "nsaid": ["nsaid", "ibuprofen", "diclofenac", "naproxen", "ketorolac", "aspirin"],
    "aspirin": ["aspirin", "nsaid", "ibuprofen", "diclofenac", "naproxen"],
    "cephalosporin": ["cephalosporin", "cefuroxime", "ceftriaxone", "cefalexin", "cephalexin", "cefixime", "cefazolin"],
    "macrolide": ["macrolide", "azithromycin", "clarithromycin", "erythromycin"],
    "fluoroquinolone": ["quinolone", "ciprofloxacin", "levofloxacin", "moxifloxacin"],
    "codeine": ["codeine", "opioid", "morphine", "tramadol"],
    "morphine": ["morphine", "opioid", "codeine"],
}

# (substance in current meds, substance being prescribed, warning)
INTERACTIONS: list[tuple[list[str], list[str], str]] = [
    (["warfarin", "apixaban", "rivaroxaban", "heparin", "enoxaparin"], ["ibuprofen", "diclofenac", "naproxen", "aspirin", "ketorolac"], "NSAID/aspirin with an anticoagulant raises bleeding risk"),
    (["ibuprofen", "diclofenac", "naproxen", "aspirin", "ketorolac"], ["warfarin", "apixaban", "rivaroxaban", "heparin", "enoxaparin"], "anticoagulant with NSAID/aspirin raises bleeding risk"),
    (["clopidogrel", "aspirin"], ["warfarin", "apixaban", "rivaroxaban", "heparin", "enoxaparin"], "dual antithrombotic therapy raises bleeding risk"),
    (["morphine", "tramadol", "codeine", "fentanyl"], ["diazepam", "lorazepam", "midazolam", "alprazolam"], "opioid + benzodiazepine: respiratory depression risk"),
    (["lisinopril", "enalapril", "ramipril", "losartan", "spironolactone"], ["spironolactone", "potassium", "lisinopril", "enalapril", "ramipril", "losartan"], "ACEi/ARB with potassium-sparing agent: hyperkalaemia risk"),
]


def _clean_recurrence(rec: Any) -> Optional[dict[str, Any]]:
    """Validate and clamp {freq, count, interval}; None when absent or unusable."""
    if not isinstance(rec, dict):
        return None
    freq = str(rec.get("freq", "")).upper()
    if freq not in ("DAILY", "WEEKLY", "MONTHLY"):
        return None
    try:
        count = max(1, min(12, int(rec.get("count", 1))))
        interval = max(1, min(4, int(rec.get("interval", 1) or 1)))
    except (TypeError, ValueError):
        return None
    return {"freq": freq, "count": count, "interval": interval}


def _safe_follow_up(fu: Any) -> Optional[dict[str, Any]]:
    """The client's followUpSoFar, reduced to what the server can rebuild links from.
    callUrl, googleCalendarUrl, startText, icsPath and booked are never trusted."""
    if not isinstance(fu, dict) or fu.get("inDays") is None:
        return None
    try:
        days = max(0, min(365, int(fu["inDays"])))
    except (TypeError, ValueError):
        return None
    out: dict[str, Any] = {"inDays": days, "reason": re.sub(r"\s+", " ", str(fu.get("reason") or "follow-up")).strip()[:160]}
    rec = _clean_recurrence(fu.get("recurrence"))
    if rec:
        out["recurrence"] = rec
    mode = str(fu.get("mode") or "").lower()
    if mode in ("visit", "call"):
        out["mode"] = mode
    return out


def _clip(value: Any, n: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:n]


def _clip_rx(rx: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"name": _clip(r.get("name"), 80), "dose": _clip(r.get("dose"), 80), "duration": _clip(r.get("duration"), 80)}
            for r in rx if isinstance(r, dict)][:12]


def _call_snapshot(req: "NurseActRequest", rx: list[dict[str, Any]], reason: Any, lang: str) -> dict[str, Any]:
    """Snapshot for the Asha call registry, with every client-supplied field clipped."""
    p = req.patient
    return {"caseId": _clip(req.caseId, 80), "patientName": _clip(p.name, 80), "age": p.age, "gender": _clip(p.gender, 16),
            "chiefComplaint": _clip(p.chiefComplaint, 200), "prescriptions": _clip_rx(rx),
            "reason": _clip(reason, 160), "lang": lang}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


class NurseState:
    def __init__(self, req: NurseActRequest):
        self.req = req
        self.tests = {t.id: t for t in req.catalog.tests}
        self.meds = {m.id: m for m in req.catalog.medications}
        self.results = {r.testId: r for r in req.patient.testResults}
        self.ordered: list[str] = list(req.alreadyOrdered)
        self.prescribed: list[str] = list(req.alreadyPrescribed)
        self.new_tests: list[str] = []
        self.new_rx: list[dict[str, str]] = []
        self.follow_up: Optional[dict[str, Any]] = None
        # Hard conflicts found by the real drug databases (NIH RxClass, openFDA).
        self.ext_hard: dict[str, list[str]] = {}
        # Asha follow-up call tokens created in this run (snapshots refreshed at the end).
        self.call_tokens: list[str] = []
        # A Telegram contact ('telegram' / '@user') resolved to a stable 'tg:<chat id>'.
        self.resolved_contact: Optional[str] = None
        # Call links whose reminders went out to an unresolved Telegram contact.
        self.unresolved_reminder_urls: list[str] = []

    # --- helpers
    def _allergy_text(self) -> str:
        parts = [self.req.patient.allergies or ""]
        for qa in self.req.patient.anamnesis:
            blob = f"{qa.question} {qa.answer}"
            if "allerg" in blob.lower():
                parts.append(qa.answer)
        return _norm(" ".join(parts))

    def _med_words(self, med_id: str) -> set[str]:
        m = self.meds.get(med_id)
        words = {_norm(med_id), *(_norm(m.name).split() if m else [])}
        words.add(_norm(m.name) if m else "")
        return {w for w in words if w}

    def _warnings(self, med_id: str) -> tuple[list[str], list[str]]:
        """Returns (hard_conflicts, soft_warnings)."""
        hard: list[str] = []
        soft: list[str] = []
        allergy = self._allergy_text()
        words = self._med_words(med_id)
        if allergy and allergy not in ("none", "no", "nkda", "no known allergies"):
            for cls, members in ALLERGY_CLASSES.items():
                if cls in allergy or any(m in allergy for m in members):
                    if any(m in w or w in m for m in members for w in words):
                        hard.append(f"documented allergy ({cls} class) conflicts with {self.meds[med_id].name if med_id in self.meds else med_id}")
                        break
            else:
                if any(w and w in allergy for w in words):
                    hard.append(f"documented allergy mentions {med_id}")
        current = [_norm(p) for p in self.prescribed]
        anam = _norm(" ".join(qa.answer for qa in self.req.patient.anamnesis if "medic" in qa.question.lower()))
        for have, adding, msg in INTERACTIONS:
            if any(w in a for w in words for a in adding):
                if any(h in c for h in have for c in current) or any(h in anam for h in have):
                    soft.append(msg)
        if med_id in self.prescribed:
            soft.append("already prescribed this encounter")
        for h in self.ext_hard.get(med_id, []):
            if h not in hard:
                hard.append(h)
        return hard, soft

    # --- tools
    def find_in_catalog(self, query: str, kind: str = "test") -> dict[str, Any]:
        q = _norm(query)
        qtok = set(q.split())
        pool: list[tuple[str, str, str]]
        if kind == "medication":
            pool = [(m.id, m.name, m.dosing or "") for m in self.meds.values()]
        else:
            pool = [(t.id, t.name, t.category) for t in self.tests.values()]
        scored = []
        for id_, name, extra in pool:
            hay = _norm(f"{id_} {name} {extra}")
            score = 0
            if q and (q in hay):
                score += 3
            score += len(qtok & set(hay.split()))
            if score:
                scored.append((score, id_, name, extra))
        scored.sort(key=lambda x: -x[0])
        matches = [{"id": i, "name": n, "info": e} for _, i, n, e in scored[:5]]
        if not matches:
            where = "medication" if kind == "medication" else "test"
            return {"ok": False, "kind": kind, "matches": [],
                    "error": f"no {where} matching '{query.strip()}' is stocked in the catalogue"}
        return {"ok": True, "kind": kind, "matches": matches}

    def order_test(self, testId: str) -> dict[str, Any]:
        if testId not in self.tests:
            return {"ok": False, "error": f"unknown testId '{testId}'; use find_in_catalog first"}
        t = self.tests[testId]
        dup = testId in self.ordered
        if not dup:
            self.ordered.append(testId)
            self.new_tests.append(testId)
        r = self.results.get(testId)
        return {
            "ok": True,
            "testId": testId,
            "name": t.name,
            "alreadyOrdered": dup,
            "result": r.result if r and r.result else "Within normal limits",
            "abnormal": bool(r.abnormal) if r else False,
        }

    def check_allergies_and_interactions(self, medicationId: str) -> dict[str, Any]:
        if medicationId not in self.meds:
            return {"ok": False, "error": f"unknown medicationId '{medicationId}'; use find_in_catalog first"}
        hard, soft = self._warnings(medicationId)
        return {"ok": True, "medicationId": medicationId, "safe": not hard, "hardConflicts": hard, "warnings": soft}

    def prescribe(self, medicationId: str, dose: str = "", duration: str = "") -> dict[str, Any]:
        if medicationId not in self.meds:
            return {"ok": False, "error": f"unknown medicationId '{medicationId}'; use find_in_catalog first"}
        hard, soft = self._warnings(medicationId)
        if hard:
            return {"ok": False, "refused": True, "reason": "; ".join(hard), "suggestion": "choose a non-conflicting alternative"}
        if medicationId in self.prescribed:
            return {"ok": True, "alreadyPrescribed": True, "warnings": soft}
        m = self.meds[medicationId]
        substitute = not self._named_by_doctor(medicationId, m.name)
        if substitute:
            # Safety: a drug the doctor did NOT name (e.g. doxycycline offered
            # instead of a held amoxicillin) must never inherit the dose the
            # doctor spoke for the original drug. Use the catalogue's standard
            # dosing for the substitute instead.
            dose = m.dosing or "standard dose"
        dose = dose or (m.dosing or "standard dose")
        duration = duration or "as directed"
        self.prescribed.append(medicationId)
        self.new_rx.append({"medicationId": medicationId, "dose": dose, "duration": duration})
        out = {"ok": True, "medicationId": medicationId, "name": m.name, "dose": dose, "duration": duration, "warnings": soft}
        if substitute:
            out["substitute"] = True
            out["note"] = "Not the drug the doctor named; standard catalogue dose used. Tell the doctor and ask them to confirm."
        return out

    def _named_by_doctor(self, medication_id: str, name: str) -> bool:
        """True if the doctor's words mention this drug (by its base name)."""
        said = re.sub(r"[^a-z0-9 ]+", " ", (self.req.utterance or "").lower())
        stems = {re.split(r"[\s\-_0-9]", medication_id.lower())[0], re.split(r"[\s\-_0-9(]", name.lower())[0]}
        return any(len(s) >= 4 and s in said for s in stems)

    def schedule_follow_up(self, inDays: Any, reason: str = "", recurrence: Any = None, mode: Any = None,
                           startsInMinutes: Any = None) -> dict[str, Any]:
        try:
            days = max(0, min(365, int(inDays)))
        except (TypeError, ValueError):
            return {"ok": False, "error": "inDays must be an integer"}
        self.follow_up = {"inDays": days, "reason": reason or "follow-up"}
        rec = _clean_recurrence(recurrence)
        if rec:
            self.follow_up["recurrence"] = rec
        m = str(mode or "").lower()
        if m in ("visit", "call"):
            self.follow_up["mode"] = m
        try:
            mins = int(startsInMinutes) if startsInMinutes not in (None, "") else 0
        except (TypeError, ValueError):
            mins = 0
        if 0 < mins <= 1440:
            # "check-in call in 5 minutes": a call today at an exact time.
            self.follow_up["startsInMinutes"] = mins
            self.follow_up["mode"] = "call"
        return {"ok": True, **self.follow_up}

    # --- real-world tools (network; never raise)
    async def aexecute(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool in ("send_to_patient", "log_case", "refer_patient"):
            try:
                return await getattr(self, tool)(**args)
            except TypeError as e:
                return {"ok": False, "error": f"bad arguments: {e}"}
            except Exception as e:  # noqa: BLE001 - one bad tool never fails the whole run
                log.warning("send_to_patient failed: %s", type(e).__name__)
                return {"ok": False, "error": f"sending failed ({type(e).__name__})"}
        res = self.execute(tool, args)
        try:
            if tool == "check_allergies_and_interactions" and res.get("ok"):
                await self._real_drug_check(str(args.get("medicationId", "")), res)
            elif tool == "schedule_follow_up" and res.get("ok"):
                await self._calendar(res)
        except Exception as e:  # noqa: BLE001 - keep the local result
            log.warning("%s real-world step failed: %s", tool, type(e).__name__)
        return res

    async def _real_drug_check(self, med_id: str, res: dict[str, Any]) -> None:
        from integrations import drugsafety

        m = self.meds.get(med_id)
        if not m:
            return
        allergies = " ".join([self.req.patient.allergies or ""] + [
            qa.answer for qa in self.req.patient.anamnesis if "allerg" in f"{qa.question} {qa.answer}".lower()])
        current = [self.meds[p].name for p in self.prescribed if p in self.meds] + [
            qa.answer for qa in self.req.patient.anamnesis if "medic" in qa.question.lower()]
        try:
            ext = await asyncio.wait_for(drugsafety.check_drug(m.name, allergies, current), timeout=7.0)
        except Exception as e:  # noqa: BLE001
            log.warning("real drug check failed: %s", type(e).__name__)
            ext = None
        if not ext or not ext.get("ok"):
            res["realCheck"] = {"ok": False, "note": "public drug databases unreachable; checked the clinic catalogue only"}
            return
        conflict = None if drugsafety.is_negated_allergy(allergies) else ext.get("allergyConflict")
        if conflict:
            msg = f"NIH RxClass: {conflict}"
            self.ext_hard.setdefault(med_id, []).append(msg)
            if msg not in res["hardConflicts"]:
                res["hardConflicts"].append(msg)
            res["safe"] = False
        for it in (ext.get("interactions") or [])[:2]:
            res["warnings"].append(f"FDA label, with {it.get('with')}: {it.get('excerpt')}")
        res["realCheck"] = {
            "ok": True,
            "drug": ext.get("drug"),
            "rxcui": ext.get("rxcui"),
            "classes": (ext.get("classes") or [])[:4],
            "boxedWarning": ext.get("boxedWarning"),
            "sources": ext.get("sources") or [],
        }

    async def _calendar(self, res: dict[str, Any]) -> None:
        from datetime import datetime
        from integrations import calendar_invite, followup_calls, gcal_script, notify

        rec = res.get("recurrence")
        # Every follow-up carries a link to a live Agora voice check-in with Asha
        # (adherence, side effects, red flags) that reports back to the doctor.
        is_call = True
        call_url: Optional[str] = None
        if is_call:
            try:
                rx = [{"medicationId": r["medicationId"],
                       "name": self.meds[r["medicationId"]].name if r["medicationId"] in self.meds else r["medicationId"],
                       "dose": r.get("dose", ""), "duration": r.get("duration", "")} for r in self.new_rx]
                for r in self.req.prescriptionsSoFar:
                    if isinstance(r, dict):
                        rx.append({k: str(r.get(k, "")) for k in ("medicationId", "name", "dose", "duration")})
                token = followup_calls.register_call({
                    **_call_snapshot(self.req, rx, res.get("reason", ""), langmod.norm_lang(self.req.lang)),
                    "contact": _clip((self.req.contact or "").strip(), 80)})
                self.call_tokens.append(token)
                path = f"/call/{token}"
                call_url = notify.absolute_url(path) or path
            except Exception as e:  # noqa: BLE001
                log.warning("follow-up call link failed: %s", type(e).__name__)
        try:
            from datetime import timedelta, timezone

            mins = res.get("startsInMinutes")
            cal = calendar_invite.make_follow_up(
                self.req.caseId, self.req.patient.name, res["inDays"], res["reason"],
                recurrence=rec, call_url=call_url,
                start_at=datetime.now(timezone.utc) + timedelta(minutes=int(mins)) if mins else None)
        except Exception as e:  # noqa: BLE001
            log.warning("calendar invite failed: %s", type(e).__name__)
            return
        res["calendar"] = {
            "startText": cal.get("startText"),
            "googleCalendarUrl": cal.get("googleCalendarUrl"),
            "icsUrl": notify.absolute_url(cal["icsPath"]) or cal["icsPath"],
        }
        if call_url:
            res["calendar"]["callUrl"] = call_url
        if self.follow_up is not None:
            self.follow_up.update(startText=cal.get("startText"), googleCalendarUrl=cal.get("googleCalendarUrl"),
                                  icsPath=cal["icsPath"], start=cal.get("start"))
            if call_url:
                self.follow_up["callUrl"] = call_url
        contact = (self.req.contact or "").strip()
        reminder_to = contact
        if contact and notify.is_telegram(contact):
            # 'telegram' / '@user' resolve to the chat id now, so a later Start from
            # someone else never redirects this patient's reminders.
            reminder_to = self.resolved_contact or await notify.resolve_telegram(contact) or contact
            if reminder_to.startswith("tg:"):
                self.resolved_contact = reminder_to
        try:
            booked = await asyncio.wait_for(gcal_script.book_event(
                cal["title"], cal.get("description", ""), datetime.fromisoformat(cal["start"]),
                minutes=cal.get("minutes", 20), recurrence=rec,
                guest_email=contact if "@" in contact else None), timeout=15.0)
        except Exception as e:  # noqa: BLE001
            log.warning("google calendar booking failed: %s", type(e).__name__)
            booked = None
        if booked and booked.get("ok"):
            res["calendar"]["booked"] = {
                "ok": True, "htmlLink": booked.get("htmlLink"),
                "recurring": bool(rec), "seriesId": booked.get("seriesId")}
            if self.follow_up is not None:
                self.follow_up["booked"] = True
        # At each call time the patient is messaged "Asha is ready, tap to join".
        if call_url and contact:
            from integrations import call_reminders

            times = call_reminders.schedule(reminder_to, self.req.patient.name, call_url,
                                            datetime.fromisoformat(cal["start"]), rec, res.get("reason", ""))
            if times:
                res["calendar"]["reminders"] = len(times)
                if not reminder_to.lower().startswith("tg:") and notify.is_telegram(reminder_to):
                    self.unresolved_reminder_urls.append(call_url)

    def _med_names(self, ids: list[str]) -> list[str]:
        return [self.meds[i].name if i in self.meds else i for i in ids]

    def _all_prescription_names(self) -> list[str]:
        names: list[str] = []
        for mid in list(self.prescribed):
            n = self.meds[mid].name if mid in self.meds else mid
            if n not in names:
                names.append(n)
        for rx in self.req.prescriptionsSoFar:
            n = str(rx.get("name") or "") or (self.meds[str(rx.get("medicationId"))].name
                                              if str(rx.get("medicationId")) in self.meds else str(rx.get("medicationId", "")))
            if n and n not in names:
                names.append(n)
        return names

    async def log_case(self, diagnosis: str = "", notes: str = "") -> dict[str, Any]:
        """Append this encounter to the student's CBME clinical logbook (Google Sheet)."""
        import logbook

        r = self.req
        lreq = logbook.LogbookRequest(
            caseId=r.caseId, caseName=r.caseName or "", specialty=r.specialty or "",
            patientName=r.patient.name, diagnosis=(diagnosis or r.diagnosis or "").strip(),
            skills=list(r.skills), tests=[self.tests[t].name if t in self.tests else t for t in self.ordered],
            prescriptions=self._all_prescription_names(),
            student=r.student or r.doctorName or "", university=r.university or "", notes=notes or "")
        out = await logbook.log_row(logbook.build_row(lreq))
        if out.get("ok"):
            out["case"] = lreq.caseName or lreq.caseId
        return out

    async def refer_patient(self, specialty: str = "", urgency: str = "routine", reason: str = "",
                            workingDiagnosis: str = "") -> dict[str, Any]:
        """FHIR referral ServiceRequest, SBAR letter (Doc/PDF) and a Murf voice handover."""
        import referral

        if not str(specialty or "").strip():
            return {"ok": False, "error": "which specialty should the patient be referred to?"}
        p = self.req.patient
        abnormal = [f"{self.tests[tid].name if tid in self.tests else tid}: {r.result}"
                    for tid, r in self.results.items() if r.abnormal and tid in self.ordered]
        ctx = {
            "caseId": self.req.caseId, "name": p.name, "age": p.age, "gender": p.gender,
            "complaint": p.chiefComplaint, "vitals": p.vitals, "allergies": p.allergies,
            "specialty": str(specialty).strip()[:80], "urgency": urgency, "reason": reason,
            "diagnosis": (workingDiagnosis or self.req.diagnosis or "").strip(),
            "abnormal": abnormal, "tests": [self.tests[t].name if t in self.tests else t for t in self.ordered],
            "prescriptions": self._all_prescription_names(),
        }
        return await referral.make_referral(ctx)

    async def send_to_patient(self, channel: str = "auto") -> dict[str, Any]:
        from integrations import calendar_invite, notify
        import rx_audio

        to = (self.req.contact or "").strip()
        if not to:
            return {"ok": False, "error": "no patient phone or email on file; add it in the Plan tab first"}
        meds: list[dict[str, str]] = []
        for rx in list(self.req.prescriptionsSoFar) + self.new_rx:
            mid = str(rx.get("medicationId", ""))
            name = self.meds[mid].name if mid in self.meds else str(rx.get("name") or mid)
            if name and all(x["name"] != name for x in meds):
                meds.append({"name": name, "dose": str(rx.get("dose", "")), "duration": str(rx.get("duration", ""))})
        if not meds:
            return {"ok": False, "error": "nothing prescribed yet, so there is nothing to send"}
        # This run's follow-up was built server-side; one from the chart is rebuilt
        # from inDays/reason so no client-supplied link reaches the patient.
        fu = dict(self.follow_up) if self.follow_up else _safe_follow_up(self.req.followUpSoFar)
        if fu and not fu.get("icsPath") and fu.get("inDays") is not None:
            try:
                cal = calendar_invite.make_follow_up(self.req.caseId, self.req.patient.name, int(fu["inDays"]),
                                                     str(fu.get("reason") or "follow-up"), recurrence=fu.get("recurrence"))
                fu.update(startText=cal.get("startText"), googleCalendarUrl=cal.get("googleCalendarUrl"), icsPath=cal["icsPath"])
            except Exception:  # noqa: BLE001
                pass
        lang = langmod.norm_lang(self.req.lang)
        # Murf voice prescription in the patient's language, hosted for the phone.
        mp3: Optional[bytes] = None
        audio_url: Optional[str] = None
        try:
            rq = rx_audio.RxAudioRequest(
                caseId=self.req.caseId[:80],
                patient=rx_audio.RxPatient(name=self.req.patient.name[:80]),
                prescriptions=[rx_audio.RxItem(name=m["name"][:80], dose=m["dose"][:80], duration=m["duration"][:60]) for m in meds[:12]],
                followUp=rx_audio.RxFollowUp(inDays=fu.get("inDays"), reason=str(fu.get("reason") or "")[:160]) if fu else None,
                lang=lang,
            )
            script = await rx_audio.generate_script(rq, lang)
            mp3, _voice, _locale = await rx_audio.synthesize(script, lang)
            audio_url = notify.absolute_url(notify.store_audio(mp3))
        except Exception as e:  # noqa: BLE001
            log.warning("voice prescription for send failed: %s", type(e).__name__)
        ics: Optional[bytes] = None
        if fu and fu.get("icsPath") and not fu.get("booked"):  # booked straight into the calendar: no invite file
            token = fu["icsPath"].rsplit("/", 1)[-1].removesuffix(".ics")
            text = calendar_invite.get_ics(token)
            ics = text.encode() if text else None
        message = notify.compose_message(
            self.req.patient.name, self.req.doctorName or "your doctor", meds,
            {"startText": fu.get("startText"), "reason": fu.get("reason")} if fu else None,
            audio_url, None if (fu or {}).get("booked") else (fu or {}).get("googleCalendarUrl"), lang)
        if fu and fu.get("booked"):
            # Already booked straight into the calendar: no "add to calendar" step.
            message = message.replace(notify.DISCLAIMER, f"It is already booked in your Google Calendar. {notify.DISCLAIMER}", 1)
        call_url = (fu or {}).get("callUrl")
        if fu and not call_url:
            from integrations import followup_calls

            token = followup_calls.register_call({**_call_snapshot(self.req, meds, fu.get("reason"), lang),
                                                  "contact": _clip(to, 80)})
            call_url = notify.absolute_url(f"/call/{token}") or f"/call/{token}"
        if call_url and call_url.startswith("http"):
            message = message.replace(notify.DISCLAIMER, f"Join your Asha follow-up call: {call_url}. {notify.DISCLAIMER}", 1)
        # The doctor may say "WhatsApp" when only Telegram or email is set up:
        # deliver through whatever is configured and fits the contact.
        if channel not in notify.configured_channels():
            channel = "auto"
        if channel == "auto" and notify.is_telegram(to):
            channel = "telegram"
        out = await notify.send_to_patient(channel, to, message, audio_url, mp3, ics)
        resolved = out.get("resolvedTo")
        if out.get("ok") and isinstance(resolved, str) and resolved.startswith("tg:"):
            # Later reminders go to this exact chat, not "the latest chat that pressed Start".
            self.resolved_contact = resolved
            if self.unresolved_reminder_urls:
                from integrations import call_reminders

                for url in self.unresolved_reminder_urls:
                    call_reminders.retarget(url, resolved)
                self.unresolved_reminder_urls.clear()
        out["audioUrl"] = audio_url
        out["medications"] = [m["name"] for m in meds]
        out["message"] = message
        if call_url:
            out["callUrl"] = call_url
        if fu:
            out["followUp"] = fu.get("startText")
        return out

    def execute(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = {
            "find_in_catalog": self.find_in_catalog,
            "order_test": self.order_test,
            "check_allergies_and_interactions": self.check_allergies_and_interactions,
            "prescribe": self.prescribe,
            "schedule_follow_up": self.schedule_follow_up,
        }.get(tool)
        if not fn:
            return {"ok": False, "error": f"unknown tool '{tool}'"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments: {e}"}


def summarize(tool: str, args: dict[str, Any], res: dict[str, Any]) -> str:
    if not res.get("ok"):
        if res.get("refused"):
            return f"Refused {args.get('medicationId')}: {res.get('reason')}"
        return f"{tool} failed: {res.get('error', 'unknown error')}"
    if tool == "find_in_catalog":
        return "Found: " + (", ".join(m["id"] for m in res["matches"]) or "nothing")
    if tool == "order_test":
        return f"Ordered {res['name']}: {res['result']}" + (" (abnormal)" if res["abnormal"] else "")
    if tool == "check_allergies_and_interactions":
        return "Safe" if res["safe"] and not res["warnings"] else "; ".join(res["hardConflicts"] + res["warnings"]) or "Safe"
    if tool == "prescribe":
        return f"Prescribed {res.get('name', args.get('medicationId'))} {res.get('dose', '')} x {res.get('duration', '')}".strip()
    if tool == "schedule_follow_up":
        cal = res.get("calendar") or {}
        when = cal.get("startText")
        rec = res.get("recurrence")
        mins = res.get("startsInMinutes")
        text = (f"Check-in call in {mins} minutes: {res['reason']}" if mins
                else f"Follow-up in {res['inDays']} days: {res['reason']}")
        if rec:
            unit = {"DAILY": "day", "WEEKLY": "week", "MONTHLY": "month"}[rec["freq"]]
            every = "" if rec["interval"] == 1 else f"{rec['interval']} "
            text += f", repeating every {every}{unit}{'s' if rec['interval'] > 1 else ''} x {rec['count']}"
        if res.get("mode") == "call" or rec:
            text += " (remote call with Nurse Asha)"
        if when:
            text += f" ({when}, " + ("added to Google Calendar" if (cal.get("booked") or {}).get("ok") else "calendar invite ready") + ")"
        if cal.get("reminders"):
            n = cal["reminders"]
            text += f"; the patient gets a join reminder at {'each call' if n > 1 else 'call time'}"
        return text
    if tool == "send_to_patient":
        return f"Sent to the patient by {res.get('channel')} ({res.get('to')}): " + ", ".join(res.get("medications") or [])
    if tool == "log_case":
        return "Logged to your CBME clinical logbook" + (f": {res['case']}" if res.get("case") else "")
    if tool == "refer_patient":
        bits = [b for b, ok in (("FHIR ServiceRequest", res.get("fhir")), ("referral PDF", res.get("pdfUrl")),
                                ("voice handover", res.get("audioUrl"))) if ok]
        text = f"Referred to {res.get('specialty')} ({res.get('urgency')})"
        if bits:
            text += ": " + ", ".join(bits)
        if (res.get("delivered") or {}).get("ok"):
            text += "; sent to the consultant"
        return text
    if tool == "record_to_ehr":
        return "Saved to the FHIR health record: " + ", ".join(f"{r['resourceType']}/{r['id']}" for r in res.get("records", []))
    return "ok"


TOOL_SPECS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "find_in_catalog",
        "description": "Search the clinic catalog for a test or medication by free text. Always use this to resolve ids.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "kind": {"type": "string", "enum": ["test", "medication"]}},
            "required": ["query", "kind"]}}},
    {"type": "function", "function": {
        "name": "order_test",
        "description": "Order a lab/imaging test by catalog id; returns the result and abnormal flag.",
        "parameters": {"type": "object", "properties": {"testId": {"type": "string"}}, "required": ["testId"]}}},
    {"type": "function", "function": {
        "name": "check_allergies_and_interactions",
        "description": "Check a medication against the patient's allergies, history and current prescriptions.",
        "parameters": {"type": "object", "properties": {"medicationId": {"type": "string"}}, "required": ["medicationId"]}}},
    {"type": "function", "function": {
        "name": "prescribe",
        "description": "Prescribe a medication. Refused if there is a hard allergy conflict.",
        "parameters": {"type": "object", "properties": {
            "medicationId": {"type": "string"}, "dose": {"type": "string"}, "duration": {"type": "string"}},
            "required": ["medicationId", "dose", "duration"]}}},
    {"type": "function", "function": {
        "name": "schedule_follow_up",
        "description": "Schedule a follow-up and put it in the doctor's Google Calendar. Use recurrence for repeating check-ins "
                       "(e.g. weekly for a month) and mode call for a remote voice call with Nurse Asha instead of a clinic visit. "
                       "For a call later today ('check-in call in 5 minutes') set inDays 0 and startsInMinutes.",
        "parameters": {"type": "object", "properties": {
            "inDays": {"type": "integer"}, "reason": {"type": "string"},
            "recurrence": {"type": "object", "properties": {
                "freq": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 12},
                "interval": {"type": "integer", "minimum": 1, "maximum": 4}}, "required": ["freq", "count"]},
            "mode": {"type": "string", "enum": ["visit", "call"]},
            "startsInMinutes": {"type": "integer", "minimum": 1, "maximum": 1440,
                                "description": "Only for a call later today, e.g. 'check-in call in 5 minutes' -> 5 (set inDays 0)."}},
            "required": ["inDays", "reason"]}}},
    {"type": "function", "function": {
        "name": "send_to_patient",
        "description": "Send the patient their prescription (with a Murf voice-prescription audio link in their language) "
                       "and the follow-up date with a calendar invite, by Telegram, WhatsApp, SMS or email to the contact on file (use auto unless the doctor names a channel). "
                       "Only when the doctor asks to send, share, message, Telegram, WhatsApp or email it to the patient.",
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string", "enum": ["auto", "telegram", "whatsapp", "sms", "email"]}}, "required": ["channel"]}}},
    {"type": "function", "function": {
        "name": "log_case",
        "description": "Log this case in the doctor's NMC CBME clinical logbook (a Google Sheet): specialty, diagnosis, skills, tests, "
                       "prescriptions. Only when the doctor asks to log, record or add this case to their logbook.",
        "parameters": {"type": "object", "properties": {
            "diagnosis": {"type": "string", "description": "Working or final diagnosis if the doctor stated one"},
            "notes": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "refer_patient",
        "description": "Refer the patient to a specialty: writes a FHIR referral ServiceRequest, an SBAR referral letter (Google Doc and PDF) "
                       "and a short Murf voice handover. Only when the doctor asks to refer, or hand over, the patient to a specialist or consultant.",
        "parameters": {"type": "object", "properties": {
            "specialty": {"type": "string", "description": "e.g. cardiology"},
            "urgency": {"type": "string", "enum": ["routine", "urgent", "emergency"]},
            "reason": {"type": "string"},
            "workingDiagnosis": {"type": "string", "description": "Only if the doctor stated one"}},
            "required": ["specialty", "urgency", "reason"]}}},
]

SYSTEM_PROMPT = (
    "You are the charge nurse of a clinic, executing the doctor's spoken orders for one patient. "
    "Be fast, and understand intent however the doctor says it.\n"
    "- Infer intent from loose language: partial sentences, abbreviations (CBC, FBC, 'bloods', 'sugar' = blood glucose, "
    "ECG/EKG, CXR or 'x ray' = chest X-ray, LFT, KFT/RFT, TSH, 'urine' = urinalysis), Hinglish ('ek ECG kara do', "
    "'paracetamol de do', 'sugar check karo', 'drip laga do'), indirect requests ('she needs an antibiotic', "
    "'get her sugar checked', 'let's put him on fluids', 'sort out the pain') and misrecognised speech "
    "('see bc' = CBC, 'amoxy' = amoxicillin, 'para' = paracetamol). Pick the most plausible clinical reading.\n"
    "- The doctor may mix English, Hindi, Kannada, Tamil, Telugu or other Indian languages in one sentence, in Roman "
    "letters or native script ('ಒಂದು ECG ಮಾಡಿಸಿ', 'ಪ್ಯಾರಸಿಟಮಾಲ್ ಕೊಡಿ', 'एक CBC करवा दो', 'sugar test maadi'). "
    "Translate the intent yourself and act on it; tool arguments stay in English catalog terms.\n"
    "- Vague drug requests ('an antibiotic', 'something for the pain', 'something for acidity'): choose the best "
    "catalog match for this patient's complaint and allergies, and say which one you chose and its dose.\n"
    "- Only when a request is truly ambiguous (no sensible catalog match) do the parts you can, and ask ONE short "
    "clarifying question in your spoken confirmation. Never say you did not understand, and never silently do nothing.\n"
    "- If the doctor's words contain NO clinical order (small talk, questions to the patient, thinking aloud), "
    "call NO tools and reply with exactly: NO_ORDER\n"
    "- Otherwise: first write one short plan sentence, then call tools. Resolve every test/medication id with "
    "find_in_catalog, order tests with order_test, run check_allergies_and_interactions before prescribe, then prescribe "
    "or schedule_follow_up. check_allergies_and_interactions also queries real public drug databases (NIH RxNorm/RxClass "
    "and openFDA labels): if it reports a source-backed conflict, hold the drug and name the source briefly (e.g. "
    "'the FDA label warns...'). For repeating follow-ups (\'weekly check-in calls for a month\', \'every 2 weeks, 3 times\') pass recurrence "
    "{freq, count, interval}; for phone, video or remote check-ins pass mode call (a remote voice call with Nurse Asha), "
    "otherwise mode visit. Follow-ups are added to the doctor's Google Calendar automatically. "
    "If the doctor asks to send, share, WhatsApp, SMS or email the prescription or follow-up "
    "to the patient, call send_to_patient (after any prescribe or schedule_follow_up in the same request). "
    "If the doctor asks to log or record the case in their logbook, call log_case (after any orders in the same request). "
    "If the doctor asks to refer the patient to a specialist, call refer_patient with the specialty, urgency (routine unless they say urgent or emergency) "
    "and a short reason; the letter and voice handover are created for you. If a tool fails or a drug is refused for allergy, adapt: say so and (only if the doctor "
    "named a class or indication) pick a safe alternative from the catalog; otherwise just hold it.\n"
    "- Never invent results, ids, or doses beyond tool outputs and the catalog.\n"
    "- Dose: pass the doctor's spoken dose and frequency to prescribe exactly as said (e.g. '75 mg once daily'). "
    "If the only catalog product has a different strength than the doctor asked for, still prescribe it but keep the "
    "doctor's dose in the dose field, and say the difference out loud in your final sentence "
    "(e.g. 'aspirin is stocked as 100 mg, prescribed 75 mg daily as you asked, please confirm').\n"
    "- Substitutes: NEVER give an alternative drug the dose the doctor said for a different drug. If you offer or "
    "prescribe a substitute (e.g. a macrolide instead of a held penicillin), the server uses that drug's own standard "
    "dose; say the substitute's name and its dose, and ask the doctor to confirm.\n"
    "- Finish with a short spoken confirmation, max 25 words, in the warm natural voice of an experienced Indian charge "
    "nurse talking to the doctor. Sound human: fillers like 'okay', 'right', 'sure doctor' are fine. Use contractions, say doses and numbers as words, and use commas and '...' for "
    "natural pauses. Never start with 'Done -', never use lists or markdown. Say what you did, flag anything held or "
    "different, and end with a short check-in sometimes. Example: 'Okay doctor, done. I've sent the troponin and the "
    "ECG... and aspirin is on, seventy-five milligrams once a day. Anything else?' Another: 'Okay doctor, CBC is sent, "
    "but I've held the amoxicillin... she's allergic to penicillin.'"
)


def _patient_brief(req: NurseActRequest, st: NurseState) -> str:
    p = req.patient
    return json.dumps({
        "patient": {"name": p.name, "age": p.age, "gender": p.gender, "chiefComplaint": p.chiefComplaint,
                    "allergies": p.allergies or "not recorded", "vitals": p.vitals,
                    "anamnesis": [qa.model_dump() for qa in p.anamnesis][:20]},
        "alreadyOrdered": st.ordered, "alreadyPrescribed": st.prescribed,
        "catalogSize": {"tests": len(st.tests), "medications": len(st.meds)},
    })


def _parse_json(text: str) -> Optional[dict[str, Any]]:
    text = strip_think(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


async def _run_native(st: NurseState, messages: list[dict[str, Any]], steps: list[dict[str, Any]]) -> tuple[str, str]:
    """Native function calling. Returns (plan, final_text)."""
    plan = ""
    provider: Optional[str] = None
    for _ in range(MAX_ITERATIONS):
        msg, provider = await chat(
            messages,
            tools=TOOL_SPECS,
            max_tokens=500,
            temperature=0.1,
            stick_to=provider,
            tier="fast",
            effort="low",
        )
        content = strip_think(msg.get("content") or "")
        calls = msg.get("tool_calls") or []
        if not calls:
            return plan, content
        if not plan and content:
            plan = content
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            res = await st.aexecute(name, args)
            steps.append({"tool": name, "args": args, "ok": bool(res.get("ok")), "summary": summarize(name, args, res), "result": res})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(res)})
    # iteration cap: ask for the closing sentence without tools
    msg, _ = await chat(
        messages + [{"role": "user", "content": "Give the one-sentence spoken confirmation now."}],
        max_tokens=120,
        stick_to=provider,
        tier="fast",
        effort="low",
    )
    return plan, strip_think(msg.get("content") or "")


async def _run_json_plan(st: NurseState, req: NurseActRequest, steps: list[dict[str, Any]]) -> tuple[str, str]:
    """Fallback when native tool calling is unavailable: ask for a JSON plan."""
    names = {s["function"]["name"]: list(s["function"]["parameters"]["properties"]) for s in TOOL_SPECS}
    sys_msg = (
        "You are a clinic charge nurse. Reply with JSON only: "
        '{"intent":"order"|"none","plan":"one sentence","calls":[{"tool":"...","args":{...}}]}. '
        f"Tools and arg names: {json.dumps(names)}. Resolve ids from the catalog list given; "
        "check_allergies_and_interactions must precede prescribe. intent none => calls [].\n"
        f"Catalog tests: {json.dumps([t.model_dump() for t in st.tests.values()])}\n"
        f"Catalog medications: {json.dumps([m.model_dump() for m in st.meds.values()])}\n"
        f"Patient: {_patient_brief(req, st)}"
    )
    msg, prov = await chat(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": req.utterance}],
        json_mode=True,
        max_tokens=500,
        temperature=0.1,
        tier="fast",
        effort="low",
    )
    plan_obj = _parse_json(msg.get("content") or "") or {}
    calls = plan_obj.get("calls")
    calls = [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
    if plan_obj.get("intent") == "none" or not calls:
        return "", "NO_ORDER"
    for c in calls[:8]:
        name, args = str(c.get("tool") or ""), c.get("args")
        args = args if isinstance(args, dict) else {}
        res = await st.aexecute(name, args)
        steps.append({"tool": name, "args": args, "ok": bool(res.get("ok")), "summary": summarize(name, args, res), "result": res})
    msg2, _ = await chat(
        [{"role": "system", "content": langmod.with_language("Write a short spoken confirmation (max 30 words, no markdown, no 'Done -' prefix) as a warm Indian charge nurse talking to the doctor: contractions, numbers as words, commas and '...' for pauses, e.g. 'Okay doctor, I've sent the troponin and ECG... anything else?'. Mention anything refused or abnormal. Use only the facts given.", req.lang, female=True, role="nurse")},
         {"role": "user", "content": json.dumps([{"tool": s["tool"], "summary": s["summary"]} for s in steps])}],
        max_tokens=120,
        stick_to=prov,
        tier="fast",
        effort="low",
    )
    return str(plan_obj.get("plan", "")), strip_think(msg2.get("content") or "")


NURSE_FALLBACK = {
    "ta": ("சரி டாக்டர், உங்கள் ஆர்டர் ஆகிவிட்டது. வேறு ஏதாவது?", "டாக்டர், அந்த ஆர்டரை முடிக்க முடியவில்லை."),
    "te": ("సరే డాక్టర్, మీ ఆర్డర్ అయింది. ఇంకేమైనా?", "డాక్టర్, ఆ ఆర్డర్ పూర్తి చేయలేకపోయాను."),
    "mr": ("ठीक आहे डॉक्टर, तुमची ऑर्डर झाली. आणखी काही?", "डॉक्टर, ती ऑर्डर पूर्ण करता आली नाही."),
    "bn": ("ঠিক আছে ডাক্তারবাবু, আপনার অর্ডার হয়ে গেছে। আর কিছু?", "ডাক্তারবাবু, ওই অর্ডারটা শেষ করতে পারলাম না।"),
}


def _fallback_confirmation(st: NurseState, steps: list[dict[str, Any]], lang: Optional[str] = None) -> str:
    bits = []
    if st.new_tests:
        bits.append(", ".join(st.tests[t].name for t in st.new_tests) + " ordered")
    if st.new_rx:
        bits.append(", ".join(st.meds[r["medicationId"]].name for r in st.new_rx) + " prescribed")
    refused = [s["summary"] for s in steps if s["tool"] == "prescribe" and not s["ok"]]
    if refused:
        bits.append("held: " + "; ".join(refused))
    if st.follow_up:
        bits.append(f"follow-up in {st.follow_up['inDays']} days")
    lg = langmod.norm_lang(lang)
    if lg == "hi":
        return "ठीक है डॉक्टर, आपका ऑर्डर हो गया। और कुछ?" if bits else "डॉक्टर, मैं वह ऑर्डर पूरा नहीं कर पाई।"
    if lg == "kn":
        return "ಸರಿ ಡಾಕ್ಟರ್, ನಿಮ್ಮ ಆರ್ಡರ್ ಆಯಿತು. ಇನ್ನೇನಾದರೂ?" if bits else "ಡಾಕ್ಟರ್, ಆ ಆರ್ಡರ್ ಮುಗಿಸಲು ಆಗಲಿಲ್ಲ."
    if lg in NURSE_FALLBACK:
        ok, fail = NURSE_FALLBACK[lg]
        return ok if bits else fail
    return ("Okay doctor, " + ", ".join(bits) + ". Anything else?") if bits else "I couldn't complete that order."


def _refresh_call_snapshots(st: NurseState) -> None:
    """A follow-up booked before the prescription in the same order still shows
    the final medicines on Asha's call page."""
    if not st.call_tokens:
        return
    from integrations import followup_calls

    rx = [{"name": st.meds[r["medicationId"]].name if r["medicationId"] in st.meds else r["medicationId"],
           "dose": r.get("dose", ""), "duration": r.get("duration", "")} for r in st.new_rx]
    for r in st.req.prescriptionsSoFar:
        if not isinstance(r, dict):
            continue
        mid = str(r.get("medicationId", ""))
        name = st.meds[mid].name if mid in st.meds else str(r.get("name") or mid)
        if name and all(x["name"] != name for x in rx):
            rx.append({"name": name, "dose": str(r.get("dose", "")), "duration": str(r.get("duration", ""))})
    for tok in st.call_tokens:
        followup_calls.update_call(tok, prescriptions=_clip_rx(rx))


async def _record_to_ehr(req: NurseActRequest, st: NurseState, steps: list[dict[str, Any]]) -> None:
    """Write this run's orders to a real FHIR R4 server (public HAPI test server by default)."""
    if not (st.new_tests or st.new_rx or st.follow_up):
        return
    from integrations import fhir

    p = req.patient
    case = {"id": req.caseId, "name": p.name, "age": p.age, "gender": p.gender}
    tests = [(t, st.tests[t].name) for t in st.new_tests if t in st.tests]
    meds = [{"name": st.meds[r["medicationId"]].name if r["medicationId"] in st.meds else r["medicationId"],
             "dose": r["dose"], "duration": r["duration"]} for r in st.new_rx]
    try:
        records = await asyncio.wait_for(fhir.record_batch(case, tests, meds, st.follow_up), timeout=8.0)
    except Exception as e:  # noqa: BLE001
        log.warning("FHIR write failed: %s", type(e).__name__)
        records = []
    res = {"ok": bool(records), "records": records or []}
    if not records:
        res["error"] = "FHIR server unreachable; the order is on the local chart only"
    steps.append({"tool": "record_to_ehr", "args": {"server": "HL7 FHIR R4"}, "ok": bool(records),
                  "summary": summarize("record_to_ehr", {}, res), "result": res})


@router.post("/agent/nurse/act")
async def nurse_act(req: NurseActRequest):
    run_id = "run_" + secrets.token_hex(6)
    st = NurseState(req)
    steps: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": langmod.with_language(
            SYSTEM_PROMPT + "\nContext: " + _patient_brief(req, st), req.lang, female=True, role="nurse")},
        {"role": "user", "content": req.utterance},
    ]
    from fastapi import HTTPException

    try:
        plan, final = await _run_native(st, messages, steps)
    except HTTPException as e:
        if e.status_code == 503:
            raise
        if steps:
            # Tools already ran (orders, sends): never replay them through the JSON plan.
            log.warning("native tool calling failed mid-run (%s); keeping %d steps", str(e.detail)[:120], len(steps))
            plan, final = "", ""
        else:
            log.warning("native tool calling failed (%s); using JSON-plan fallback", str(e.detail)[:120])
            st = NurseState(req)
            try:
                plan, final = await _run_json_plan(st, req, steps)
            except HTTPException as e2:
                if e2.status_code == 503:
                    raise
                log.warning("JSON-plan fallback failed too (%s)", str(e2.detail)[:120])
                plan, final = "", ""
            except Exception as e2:  # noqa: BLE001
                log.warning("JSON-plan fallback failed too (%s)", type(e2).__name__)
                plan, final = "", ""
            if not steps and not final:
                return {"runId": run_id, "intent": "none", "plan": "", "steps": [],
                        "applied": {"orderedTestIds": [], "prescriptions": []},
                        "spokenConfirmation": _fallback_confirmation(st, [], req.lang), "error": "nurse unavailable"}

    if not steps:
        return {"runId": run_id, "intent": "none", "plan": "", "steps": [],
                "applied": {"orderedTestIds": [], "prescriptions": []}, "spokenConfirmation": ""}
    _refresh_call_snapshots(st)
    await _record_to_ehr(req, st, steps)
    spoken = final if final and "NO_ORDER" not in final else _fallback_confirmation(st, steps, req.lang)
    applied: dict[str, Any] = {"orderedTestIds": st.new_tests, "prescriptions": st.new_rx}
    if st.follow_up:
        applied["followUp"] = st.follow_up
    return {
        "runId": run_id, "intent": "order",
        "plan": plan or "; ".join(s["summary"] for s in steps[:3]),
        "steps": steps, "applied": applied, "spokenConfirmation": spoken[:400],
    }


@router.get("/agent/notify/status")
async def notify_status():
    """Which delivery channels are configured, and the Telegram bot's Start link."""
    from integrations import notify

    return {"channels": notify.configured_channels(), "telegram": await notify.telegram_bot_info()}


@router.get("/agent/followup-call/{token}")
async def followup_call(token: str):
    """Case snapshot for the remote Asha follow-up call page."""
    from fastapi import HTTPException
    from integrations import followup_calls

    snap = followup_calls.get_call(token)
    if snap is None:
        raise HTTPException(status_code=404, detail="Not found")
    return snap
