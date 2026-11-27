"""Google Gemini API LLM provider selection (Gemini primary, low/medium reasoning cap).

Enforces Google Gemini stack via OpenAI-compatible endpoint:
Base URL: https://generativelanguage.googleapis.com/v1beta/openai
Endpoint: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions

Models:
- Fast tier: gemini-3.5-flash-lite (default effort: low)
- Reasoning tier: gemini-3.8-flash (default effort: medium)

Strictly enforces reasoning effort caps: ALLOWED_EFFORTS = ("low", "medium").
High/xhigh/max values are clamped to "medium", preventing excessive latency and cost.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import httpx
from fastapi import HTTPException

log = logging.getLogger("medsim.llm")

DEFAULT_PROVIDER = "gemini"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_FAST_MODEL_DEFAULT = "gemini-3.5-flash-lite"
GEMINI_REASONING_MODEL_DEFAULT = "gemini-3.8-flash"

ALLOWED_EFFORTS = ("low", "medium")


def chat_url() -> str:
    base = os.environ.get("GEMINI_BASE_URL", GEMINI_BASE_URL).strip().rstrip("/")
    return f"{base}/chat/completions"


def clamp_effort(effort: Optional[str], default: str = "low") -> str:
    """Normalize and strictly clamp reasoning effort to ('low', 'medium').

    If 'high', 'xhigh', or 'max' -> 'medium'.
    If invalid/minimal/none/unset value -> default (clamped to ALLOWED_EFFORTS).
    Never returns or permits 'high'.
    """
    safe_default = (default or "low").strip().lower()
    if safe_default not in ALLOWED_EFFORTS:
        safe_default = "low"
    if not effort:
        return safe_default
    norm = effort.strip().lower()
    if norm in ("high", "xhigh", "max"):
        return "medium"
    if norm in ALLOWED_EFFORTS:
        return norm
    return safe_default


PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "url": GEMINI_BASE_URL + "/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_FAST_MODEL",
        "default_model": GEMINI_FAST_MODEL_DEFAULT,
    },
    "sarvam": {
        "url": "https://api.sarvam.ai/v1/chat/completions",
        "key_env": "SARVAM_API_KEY",
        "model_env": "SARVAM_MODEL",
        "default_model": "sarvam-105b-conversations",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-20b",
    },
}


class ProviderError(Exception):
    """Upstream LLM call failed (status/body kept for diagnostics, no keys)."""


def get_llm_config(
    tier: str = "fast",
    model: Optional[str] = None,
    effort: Optional[str] = None,
) -> dict[str, Any]:
    """Return configured Gemini LLM settings for the requested tier."""
    if tier == "reasoning":
        default_model = os.environ.get("GEMINI_REASONING_MODEL", "").strip() or GEMINI_REASONING_MODEL_DEFAULT
        default_effort = "medium"
    else:
        default_model = os.environ.get("GEMINI_FAST_MODEL", "").strip() or GEMINI_FAST_MODEL_DEFAULT
        default_effort = "low"

    eff = clamp_effort(effort, default=default_effort)
    return {
        "url": chat_url(),
        "api_key": os.environ.get("GEMINI_API_KEY", "").strip(),
        "model": model or default_model,
        "effort": eff,
        "tier": tier,
    }


def build_patient_llm_config(
    system_prompt: str = "",
    greeting: str = "",
    lang: Optional[str] = None,
    tier: str = "fast",
    model: Optional[str] = None,
    effort: Optional[str] = None,
) -> dict[str, Any]:
    """Helper for patient LLM configuration using Gemini."""
    cfg = get_llm_config(tier=tier, model=model, effort=effort)
    return {
        "url": cfg["url"],
        "api_key": cfg["api_key"],
        "model": cfg["model"],
        "reasoning_effort": cfg["effort"],
        "system_prompt": system_prompt,
        "greeting": greeting,
        "lang": lang,
    }


def provider_config(name: str, tier: str = "fast") -> Optional[dict[str, str]]:
    spec = PROVIDERS.get(name)
    if not spec:
        return None
    key = os.environ.get(spec["key_env"], "").strip()
    if not key:
        return None
    if name == "gemini":
        cfg = get_llm_config(tier=tier)
        model = cfg["model"]
        url = cfg["url"]
    else:
        model = os.environ.get(spec["model_env"], "").strip() or spec["default_model"]
        url = spec["url"]
    return {
        "name": name,
        "url": url,
        "api_key": key,
        "model": model,
    }


def provider_order(preferred: Optional[str] = None) -> list[str]:
    """Preferred provider first (PATIENT_LLM_PROVIDER, default gemini), then fallbacks."""
    first = (preferred or os.environ.get("PATIENT_LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if first not in PROVIDERS:
        first = DEFAULT_PROVIDER
    return [first] + [n for n in PROVIDERS if n != first]


def configured_providers(preferred: Optional[str] = None, tier: str = "fast") -> list[dict[str, str]]:
    return [c for n in provider_order(preferred) if (c := provider_config(n, tier=tier))]


def require_provider(preferred: Optional[str] = None, tier: str = "fast") -> dict[str, str]:
    cfgs = configured_providers(preferred, tier=tier)
    if not cfgs:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured: set GEMINI_API_KEY.",
        )
    return cfgs[0]


def llm_health() -> dict[str, Any]:
    cfgs = configured_providers()
    gemini_cfg = provider_config("gemini")
    return {
        "preferred": provider_order()[0],
        "gemini_configured": gemini_cfg is not None,
        "sarvam_configured": provider_config("sarvam") is not None,
        "groq_configured": provider_config("groq") is not None,
        "active": cfgs[0]["name"] if cfgs else None,
        "model": cfgs[0]["model"] if cfgs else None,
        "stack": "gemini",
        "effort_cap": clamp_effort(os.environ.get("GEMINI_EFFORT_CAP", "medium")),
    }


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop reasoning-model <think> blocks from visible output."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


async def _chat_one(
    cfg: dict[str, str],
    payload: dict[str, Any],
    client: Optional[httpx.AsyncClient],
) -> dict[str, Any]:
    body = {**payload, "model": cfg["model"]}
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    owns = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    try:
        r = await c.post(cfg["url"], json=body, headers=headers)
    except httpx.HTTPError as e:
        raise ProviderError(f"{cfg['name']} request failed: {type(e).__name__}") from e
    finally:
        if owns:
            await c.aclose()
    if r.status_code >= 400:
        clean_text = r.text.replace(cfg["api_key"], "***")[:300]
        raise ProviderError(f"{cfg['name']} HTTP {r.status_code}: {clean_text}")
    try:
        return r.json()["choices"][0]["message"]
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"{cfg['name']} returned an unexpected body") from e


async def chat(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
    max_tokens: int = 600,
    temperature: float = 0.2,
    client: Optional[httpx.AsyncClient] = None,
    stick_to: Optional[str] = None,
    tier: str = "fast",
    effort: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Chat completion with Gemini provider fallback. Returns (assistant_message, provider_name).

    ``tier``: 'fast' (gemini-3.5-flash-lite, effort='low') or 'reasoning' (gemini-3.8-flash, effort='medium').
    ``effort``: clamped to ('low', 'medium').
    ``stick_to`` pins one provider (needed mid tool-loop so tool_call ids stay valid).
    Raises HTTPException(503) if nothing is configured, 502 if every provider fails.
    """
    cfgs = configured_providers(tier=tier)
    if stick_to:
        cfgs = [c for c in cfgs if c["name"] == stick_to]
    if not cfgs:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured: set GEMINI_API_KEY.",
        )

    eff = clamp_effort(effort, default="medium" if tier == "reasoning" else "low")
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": eff,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    errors: list[str] = []
    for cfg in cfgs:
        current_cfg = dict(cfg)
        if model:
            current_cfg["model"] = model
        try:
            msg = await _chat_one(current_cfg, payload, client)
            return msg, current_cfg["name"]
        except ProviderError as e:
            log.warning("%s", e)
            errors.append(str(e))
    raise HTTPException(status_code=502, detail="; ".join(errors)[:600])


async def chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
    max_tokens: int = 600,
    temperature: float = 0.2,
    client: Optional[httpx.AsyncClient] = None,
    stick_to: Optional[str] = None,
    tier: str = "fast",
    effort: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Direct alias for chat completion."""
    return await chat(
        messages,
        tools=tools,
        json_mode=json_mode,
        max_tokens=max_tokens,
        temperature=temperature,
        client=client,
        stick_to=stick_to,
        tier=tier,
        effort=effort,
        model=model,
    )
