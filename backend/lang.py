"""Session language (en / hi / kn): locale mapping, Murf voices, filler phrases and
prompt language blocks shared by the voice, nurse, coach and teacher routes.

Voice data comes from the live Murf voice list (GET /v1/speech/voices?model=FALCON,
2026-11-21): every voice below lists the target locale in `supportedLocales`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Optional

log = logging.getLogger("medsim.lang")

LANGS = ("en", "hi", "kn", "ta", "te", "mr", "bn")
LOCALES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
}
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
}

# Prebuilt Google Gemini Live / Gemini TTS voices:
# Female: Aoede, Kore, Leda, Zephyr
# Male: Puck, Charon, Fenrir, Orus, Iapetus
GEMINI_VOICES = {
    "en": {"F": "Aoede", "M": "Fenrir", "nurse": "Aoede", "examiner": "Charon"},
    "hi": {"F": "Kore", "M": "Orus", "nurse": "Kore", "examiner": "Charon"},
    "bn": {"F": "Leda", "M": "Fenrir", "nurse": "Leda", "examiner": "Charon"},
    "ta": {"F": "Zephyr", "M": "Orus", "nurse": "Zephyr", "examiner": "Charon"},
    "te": {"F": "Aoede", "M": "Fenrir", "nurse": "Aoede", "examiner": "Charon"},
    "mr": {"F": "Kore", "M": "Orus", "nurse": "Kore", "examiner": "Charon"},
    "gu": {"F": "Leda", "M": "Fenrir", "nurse": "Leda", "examiner": "Charon"},
    "kn": {"F": "Zephyr", "M": "Orus", "nurse": "Zephyr", "examiner": "Charon"},
    "ml": {"F": "Aoede", "M": "Fenrir", "nurse": "Aoede", "examiner": "Charon"},
    "pa": {"F": "Kore", "M": "Orus", "nurse": "Kore", "examiner": "Charon"},
}


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    name: str
    locale: str
    gemini_voice_f: str
    gemini_voice_m: str
    nurse_voice: str
    examiner_voice: str = "Charon"

    @property
    def gemini_voice(self) -> str:
        return self.gemini_voice_f


LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    code: LanguageConfig(
        code=code,
        name=LANG_NAMES.get(code, code),
        locale=LOCALES.get(code, f"{code}-IN"),
        gemini_voice_f=v["F"],
        gemini_voice_m=v["M"],
        nurse_voice=v["nurse"],
        examiner_voice=v["examiner"],
    )
    for code, v in GEMINI_VOICES.items()
}


def get_lang_config(lang: Optional[str] = None) -> LanguageConfig:
    """Retrieve LanguageConfig for a language code, falling back to English / en-IN."""
    code = (lang or "").strip().lower()[:2]
    return LANGUAGE_CONFIGS.get(code, LANGUAGE_CONFIGS["en"])


def supported_languages() -> list[str]:
    """List of all supported language codes."""
    return list(LANGUAGE_CONFIGS.keys())


def gemini_voice_for(lang: Optional[str], gender: str = "F") -> str:
    cfg = get_lang_config(lang)
    return cfg.gemini_voice_f if gender.upper().startswith("F") else cfg.gemini_voice_m


def gemini_nurse_voice(lang: Optional[str] = None) -> str:
    return get_lang_config(lang).nurse_voice


def gemini_examiner_voice() -> str:
    return "Charon"

# Native-script Unicode block per language (Marathi shares Devanagari with Hindi).
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "kn": (0x0C80, 0x0CFF),
    "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F), "bn": (0x0980, 0x09FF),
}

# Patient voice pools per language. English and Hindi share the en-IN pool
# (all of those voices list hi-IN), so a patient keeps the same voice when the
# language is switched. Kannada needs voices that list kn-IN.
PATIENT_VOICE_POOLS: dict[str, dict[str, list[str]]] = {
    "en": {
        "M": ["en-IN-aarav", "en-IN-nikhil", "en-IN-samar", "en-IN-abhinav", "en-IN-eashwar"],
        "F": ["en-IN-priya", "en-IN-pooja", "en-IN-anusha", "en-IN-isha", "en-IN-arohi", "en-IN-alia"],
    },
    "hi": {
        "M": ["en-IN-aarav", "en-IN-nikhil", "en-IN-samar", "en-IN-abhinav", "en-IN-eashwar"],
        "F": ["en-IN-priya", "en-IN-pooja", "en-IN-anusha", "en-IN-isha", "en-IN-arohi", "en-IN-alia"],
    },
    "kn": {
        "M": ["en-IN-nikhil", "en-IN-samar", "hi-IN-chirag", "hi-IN-pranav"],
        "F": ["en-IN-pooja", "hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "kn-IN-harshitha"],
    },
    # Verified live (Murf voices API, FALCON, 2026-11-21): each lists the locale in supportedLocales.
    "ta": {
        "M": ["en-IN-samar", "hi-IN-chirag", "hi-IN-pranav", "ta-IN-karthikeyan"],
        "F": ["en-IN-pooja", "hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna"],
    },
    "te": {
        "M": ["en-IN-samar", "hi-IN-chirag", "hi-IN-pranav"],
        "F": ["en-IN-pooja", "en-IN-alia", "hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "hi-IN-ayushi"],
    },
    "mr": {
        "M": ["en-IN-samar", "en-IN-abhinav", "hi-IN-chirag", "hi-IN-pranav", "mr-IN-prathamesh"],
        "F": ["hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "mr-IN-prajakta"],
    },
    "bn": {
        "M": ["hi-IN-chirag", "hi-IN-pranav", "bn-IN-subhankar"],
        "F": ["hi-IN-poonam", "hi-IN-neeta", "hi-IN-jaya", "hi-IN-rachna", "bn-IN-debarati"],
    },
}

# Nurse (female) voice per language; each lists the locale.
NURSE_VOICES = {
    "en": "en-IN-anisha", "hi": "en-IN-anisha", "kn": "kn-IN-harshitha",
    "ta": "en-IN-anisha", "te": "en-IN-anisha", "mr": "mr-IN-prajakta", "bn": "bn-IN-debarati",
}

FILLERS: dict[str, list[str]] = {
    "en": ["Hmm...", "Haan, doctor...", "Umm, let me think...", "Achha...", "One second, doctor...",
           "Hmm, let me see...", "Okay, doctor...", "Umm... yes..."],
    "hi": ["हम्म...", "हाँ डॉक्टर...", "एक मिनट...", "अच्छा...", "हम्म, देखिए...", "जी डॉक्टर...",
           "एक सेकंड डॉक्टर...", "हाँ... जी..."],
    "kn": ["ಹ್ಮ್...", "ಹೌದು ಡಾಕ್ಟರ್...", "ಒಂದು ನಿಮಿಷ...", "ಅಚ್ಛಾ...", "ಹ್ಮ್, ನೋಡಿ...", "ಸರಿ ಡಾಕ್ಟರ್...",
           "ಒಂದು ಸೆಕೆಂಡ್ ಡಾಕ್ಟರ್...", "ಹೌದು... ಹೌದು..."],
    "ta": ["ம்ம்...", "ஆமா டாக்டர்...", "ஒரு நிமிஷம்...", "சரி...", "ம்ம், பாருங்க...", "சரி டாக்டர்...",
           "ஒரு செகண்ட் டாக்டர்...", "ஆமா... ஆமா..."],
    "te": ["హ్మ్...", "అవును డాక్టర్...", "ఒక్క నిమిషం...", "సరే...", "హ్మ్, చూడండి...", "సరే డాక్టర్...",
           "ఒక్క సెకను డాక్టర్...", "అవును... అవును..."],
    "mr": ["हम्म...", "हो डॉक्टर...", "एक मिनिट...", "अच्छा...", "हम्म, बघा...", "ठीक आहे डॉक्टर...",
           "एक सेकंद डॉक्टर...", "हो... हो..."],
    "bn": ["হুম...", "হ্যাঁ ডাক্তারবাবু...", "এক মিনিট...", "আচ্ছা...", "হুম, দেখুন...", "ঠিক আছে ডাক্তারবাবু...",
           "এক সেকেন্ড ডাক্তারবাবু...", "হ্যাঁ... হ্যাঁ..."],
}

# Agora speaks llm.failure_message when the LLM call fails. One string only, so it
# must sound like a person who half-heard, never like a system error, and must not
# name the listener ("doctor"): every voice agent shares it.
FAILURE_MESSAGES = {
    "en": "Hmm, sorry, say that once more?",
    "hi": "हम्म, माफ़ कीजिए, एक बार फिर से कहेंगे?",
    "kn": "ಹ್ಮ್, ಕ್ಷಮಿಸಿ, ಇನ್ನೊಮ್ಮೆ ಹೇಳ್ತೀರಾ?",
    "ta": "ஹ்ம், மன்னிக்கணும், இன்னொரு தடவை சொல்றீங்களா?",
    "te": "హ్మ్, సారీ, ఇంకోసారి చెప్తారా?",
    "mr": "हम्म, माफ करा, पुन्हा एकदा सांगाल का?",
    "bn": "হুম, মাফ করবেন, আরেকবার বলবেন?",
}

# What the /llm proxy streams back, in character, when every provider fails
# (instead of an HTTP error, which would make Agora speak FAILURE_MESSAGES).
NATURAL_RETRY_LINES = {
    "en": ["Hmm... sorry, could you ask that once more?",
           "Sorry, my mind drifted for a second. What was that?",
           "Hmm, one more time, please?",
           "Oh, sorry, can you say that again?"],
    "hi": ["हम्म... माफ़ कीजिए, एक बार फिर पूछेंगे?",
           "माफ़ कीजिए, ध्यान थोड़ा भटक गया। क्या पूछा आपने?",
           "हम्म, एक बार और बोलिए ना?"],
    "kn": ["ಹ್ಮ್... ಕ್ಷಮಿಸಿ, ಇನ್ನೊಮ್ಮೆ ಕೇಳ್ತೀರಾ?",
           "ಕ್ಷಮಿಸಿ, ಸ್ವಲ್ಪ ಗಮನ ತಪ್ಪಿತು. ಏನು ಕೇಳಿದ್ರಿ?",
           "ಹ್ಮ್, ಇನ್ನೊಂದ್ಸಲ ಹೇಳಿ?"],
    "ta": ["ஹ்ம்... மன்னிக்கணும், இன்னொரு தடவை கேக்கறீங்களா?",
           "மன்னிக்கணும், கொஞ்சம் கவனம் சிதறிடுச்சு. என்ன கேட்டீங்க?",
           "ஹ்ம், இன்னொரு முறை சொல்லுங்க?"],
    "te": ["హ్మ్... సారీ, ఇంకోసారి అడుగుతారా?",
           "సారీ, కొంచెం ధ్యాస తప్పింది. ఏం అడిగారు?",
           "హ్మ్, ఇంకోసారి చెప్పండి?"],
    "mr": ["हम्म... माफ करा, पुन्हा एकदा विचाराल का?",
           "माफ करा, जरा लक्ष उडालं. काय विचारलंत?",
           "हम्म, परत एकदा सांगा ना?"],
    "bn": ["হুম... মাফ করবেন, আরেকবার জিজ্ঞেস করবেন?",
           "মাফ করবেন, একটু অন্যমনস্ক হয়ে গেছিলাম। কী বললেন?",
           "হুম, আরেকবার বলুন না?"],
}


def norm_lang(value: Optional[str]) -> str:
    """'hi', 'hi-IN', 'HI' -> 'hi'. Anything unknown -> 'en'."""
    v = (value or "").strip().lower()[:2]
    return v if v in LANGS else "en"


def locale_for(lang: Optional[str]) -> str:
    return LOCALES[norm_lang(lang)]


def pick_patient_voice(lang: Optional[str], gender: str, case_id: Optional[str]) -> str:
    pool = PATIENT_VOICE_POOLS[norm_lang(lang)]["F" if gender == "F" else "M"]
    if not case_id:
        return pool[0]
    # Stable across runs (unlike hash(), which is salted per process).
    return pool[int(hashlib.sha1(case_id.encode()).hexdigest(), 16) % len(pool)]


def nurse_voice(lang: Optional[str]) -> str:
    return NURSE_VOICES[norm_lang(lang)]


def fillers(lang: Optional[str]) -> list[str]:
    return list(FILLERS[norm_lang(lang)])


def failure_message(lang: Optional[str]) -> str:
    return FAILURE_MESSAGES[norm_lang(lang)]


def natural_retry_line(lang: Optional[str], rng: Any = None) -> str:
    """A short in-character 'say that again?' line, varied per call."""
    import random

    return (rng or random).choice(NATURAL_RETRY_LINES[norm_lang(lang)])


# ── prompt language blocks ─────────────────────────────────────────────────

_HI_RULES = (
    "LANGUAGE: Speak natural, everyday spoken Hindi, written in Devanagari script only (never Roman letters "
    "for Hindi words). Common English medical loanwords are natural in Hindi speech and allowed, for example "
    "BP, sugar, test, report, tablet, injection, X-ray, ECG, doctor, hospital. "
    "Keep sentences short and conversational, the way a person in India really talks to a doctor. "
    "Address the doctor respectfully as \"आप\". Always reply in Hindi, even when the doctor mixes in "
    "English or Kannada words."
)
_KN_RULES = (
    "LANGUAGE: Speak natural, everyday spoken Kannada, written in Kannada script only (never Roman letters "
    "for Kannada words). Common English medical loanwords are natural in spoken Kannada and allowed, for "
    "example BP, sugar, test, report, tablet, injection, X-ray, ECG, doctor, hospital. "
    "Keep sentences short and conversational, the way a person in Karnataka really talks to a doctor. "
    "Address the doctor respectfully as \"ನೀವು\". Always reply in Kannada, even when the doctor mixes in "
    "English or Hindi words."
)


def _rules(name: str, script: str, region: str, you: str) -> str:
    return (
        f"LANGUAGE: Speak natural, everyday spoken {name}, written in {script} only (never Roman letters "
        f"for {name} words). Common English medical loanwords are natural in spoken {name} and allowed, for "
        "example BP, sugar, test, report, tablet, injection, X-ray, ECG, doctor, hospital. "
        f"Keep sentences short and conversational, the way a person in {region} really talks to a doctor. "
        f"Address the doctor respectfully as \"{you}\". Always reply in {name}, even when the doctor mixes in "
        "English or Hindi words."
    )


_LANG_RULES = {
    "hi": _HI_RULES,
    "kn": _KN_RULES,
    "ta": _rules("Tamil", "Tamil script", "Tamil Nadu", "நீங்கள்"),
    "te": _rules("Telugu", "Telugu script", "Andhra Pradesh or Telangana", "మీరు"),
    "mr": _rules("Marathi", "Devanagari script", "Maharashtra", "तुम्ही"),
    "bn": _rules("Bengali", "Bengali script", "West Bengal", "আপনি"),
}

_ROLE_TAIL = {
    "patient": (
        "Your very first line and every reply must follow this language rule, even though the instructions "
        "above are written in English. No translations and no English duplicate of the same sentence."
    ),
    "nurse": (
        "Write only the final spoken confirmation in this language (max 30 words). Tool arguments, ids and "
        "plan text stay exactly as specified, in English."
    ),
    "coach": (
        "Write the free-text fields title, short, explanation and suggestion in this language. Keep the JSON "
        "keys and the severity values (info, warning, danger) in English."
    ),
    "followup": (
        "You are a nurse speaking to a patient on the phone, so wherever the rule above says \"doctor\", read "
        "\"patient\". Your first line and every reply must follow this language rule, even though the instructions "
        "above are written in English. No translations and no English duplicate of the same sentence."
    ),
    "teacher": (
        "Speak the whole lesson in this language. Medical terms may stay in English where doctors in India "
        "normally say them in English."
    ),
}


MIXED_SPEECH = (
    "MIXED LANGUAGES: The doctor may mix languages inside one sentence (Hinglish, Kannada with English and Hindi, Tamil or Telugu with English), in any script or in Roman letters, and speech-to-text may garble or half-transcribe it. Understand the meaning whatever the mix and answer what they meant. Never say you did not understand and never comment on their language. "
)
MIXED_SPEECH_EN = MIXED_SPEECH + "Reply in simple Indian English (a common word like haan or accha is fine)."


def language_block(lang: Optional[str], *, female: bool = False, role: str = "patient") -> str:
    """Prompt text that makes a persona understand mixed-language speech and speak the
    session language. Prompts stay English; only the output language changes."""
    lg = norm_lang(lang)
    if lg == "en":
        return MIXED_SPEECH_EN
    out = [_LANG_RULES[lg], MIXED_SPEECH, _ROLE_TAIL.get(role, _ROLE_TAIL["patient"])]
    if lg == "mr":
        out.append(
            "You are a woman: use feminine Marathi verb forms for yourself (\"मी आले होते\", \"मी करत आहे\")."
            if female
            else "You are a man: use masculine Marathi verb forms for yourself (\"मी आलो होतो\", \"मी करत आहे\")."
        )
    if lg == "hi":
        out.append(
            "You are a woman: use feminine Hindi verb forms for yourself (\"मैं आई थी\", \"मैं कर रही हूँ\")."
            if female
            else "You are a man: use masculine Hindi verb forms for yourself (\"मैं आया था\", \"मैं कर रहा हूँ\")."
        )
    return "\n".join(out)


def with_language(system: str, lang: Optional[str], *, female: bool = False, role: str = "patient") -> str:
    block = language_block(lang, female=female, role=role)
    return f"{system}\n\n{block}" if block else system


# ── greeting translation ───────────────────────────────────────────────────

_greeting_cache: dict[tuple[str, str], str] = {}


def has_script(text: str, lang: str) -> bool:
    if lang not in SCRIPT_RANGES:
        return False
    lo, hi = SCRIPT_RANGES[lang]
    return any(lo <= ord(ch) <= hi for ch in text)


async def translate_line(text: str, lang: Optional[str]) -> str:
    """Translate a short opening line to hi/kn with one cheap LLM call (cached).
    On any failure returns the original text so the call never blocks."""
    lg = norm_lang(lang)
    if lg == "en" or not text.strip():
        return text
    key = (lg, text)
    if key in _greeting_cache:
        return _greeting_cache[key]

    from llm_provider import chat, strip_think

    sys_msg = (
        f"Translate the user's line into natural spoken {LANG_NAMES[lg]}, in {LANG_NAMES[lg]} script, as an "
        "Indian patient would say it to a doctor. Keep common English medical loanwords (BP, sugar, ECG, test) "
        "as they are. Reply with the translated line only: no quotes, no notes, no transliteration."
    )
    try:
        msg, _ = await asyncio.wait_for(
            chat(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": text}],
                max_tokens=300,
                temperature=0.2,
                tier="fast",
                effort="low",
            ),
            timeout=8.0,
        )
        out = strip_think(msg.get("content") or "").strip().strip('"').strip()
    except Exception as e:  # noqa: BLE001
        log.warning("greeting translation failed (%s); using original", type(e).__name__)
        return text
    if not out or not has_script(out, lg):
        return text
    _greeting_cache[key] = out
    if len(_greeting_cache) > 256:
        _greeting_cache.pop(next(iter(_greeting_cache)))
    return out
