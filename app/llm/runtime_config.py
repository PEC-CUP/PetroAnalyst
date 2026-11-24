from __future__ import annotations

import os
from pathlib import Path
import socket
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from app.config import BASE_DIR, get_settings


ENV_PATH = Path(os.environ.get("PETROANALYST_ENV_FILE", str(BASE_DIR / ".env")))

LLM_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_TRUST_ENV",
    "DEEPSEEK_CHAT_MODEL",
    "DEEPSEEK_REASONER_MODEL",
    "DEEPSEEK_CHAT_MODEL_FALLBACKS",
    "DEEPSEEK_REASONER_MODEL_FALLBACKS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_STRONG_MODEL",
    "OPENAI_BALANCED_MODEL",
    "OPENAI_FAST_MODEL",
    "OPENAI_CHEAP_MODEL",
    "LLM_TIMEOUT_SECONDS",
)

SETTING_BY_ENV = {
    "DEEPSEEK_API_KEY": "deepseek_api_key",
    "DEEPSEEK_BASE_URL": "deepseek_base_url",
    "DEEPSEEK_TRUST_ENV": "deepseek_trust_env",
    "DEEPSEEK_CHAT_MODEL": "deepseek_chat_model",
    "DEEPSEEK_REASONER_MODEL": "deepseek_reasoner_model",
    "DEEPSEEK_CHAT_MODEL_FALLBACKS": "deepseek_chat_model_fallbacks_raw",
    "DEEPSEEK_REASONER_MODEL_FALLBACKS": "deepseek_reasoner_model_fallbacks_raw",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENAI_MODEL": "openai_model",
    "OPENAI_STRONG_MODEL": "openai_strong_model",
    "OPENAI_BALANCED_MODEL": "openai_balanced_model",
    "OPENAI_FAST_MODEL": "openai_fast_model",
    "OPENAI_CHEAP_MODEL": "openai_cheap_model",
    "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
}


def masked_runtime_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "deepseek": {
            "api_key_configured": bool((settings.deepseek_api_key or "").strip()),
            "base_url": settings.deepseek_base_url,
            "trust_env": bool(settings.deepseek_trust_env),
            "chat_model": settings.deepseek_chat_model,
            "reasoner_model": settings.deepseek_reasoner_model,
            "chat_model_fallbacks": settings.deepseek_chat_model_fallbacks(),
            "reasoner_model_fallbacks": settings.deepseek_reasoner_model_fallbacks(),
        },
        "openai_expert": {
            "api_key_configured": bool((settings.openai_api_key or "").strip()),
            "base_url": settings.openai_base_url,
            "model": settings.openai_model,
            "strong_model": settings.openai_strong_model,
            "balanced_model": settings.openai_balanced_model,
            "fast_model": settings.openai_fast_model,
            "cheap_model": settings.openai_cheap_model,
        },
        "timeout_seconds": settings.llm_timeout_seconds,
        "env_file": str(ENV_PATH),
    }


def apply_runtime_config(payload: dict[str, Any]) -> dict[str, Any]:
    updates = _flatten_update_payload(payload)
    if not updates:
        return masked_runtime_config()

    _write_env_updates(updates)
    settings = get_settings()
    for env_key, value in updates.items():
        attr = SETTING_BY_ENV.get(env_key)
        if not attr:
            continue
        if env_key == "LLM_TIMEOUT_SECONDS":
            setattr(settings, attr, float(value))
        elif env_key == "DEEPSEEK_TRUST_ENV":
            setattr(settings, attr, _to_bool(value))
        else:
            setattr(settings, attr, str(value))
    return masked_runtime_config()


def test_provider(provider: str, *, model: str | None = None, prompt: str = "Reply with OK only.") -> dict[str, Any]:
    settings = get_settings()
    selected_provider = (provider or "deepseek").strip().lower()
    if selected_provider == "openai":
        api_key = (settings.openai_api_key or "").strip()
        base_url = settings.openai_base_url
        selected_model = (model or settings.openai_model or settings.openai_strong_model).strip()
        trust_env = False
    else:
        api_key = (settings.deepseek_api_key or "").strip()
        base_url = settings.deepseek_base_url
        selected_model = (model or settings.deepseek_chat_model).strip()
        trust_env = bool(settings.deepseek_trust_env)
    if not api_key:
        return {"status": "error", "provider": selected_provider, "model": selected_model, "message": "api_key_not_configured"}

    diagnostics = _network_diagnostics(base_url=base_url, api_key=api_key, trust_env=trust_env)
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout_seconds,
            http_client=httpx.Client(trust_env=trust_env),
        )
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
        )
        content = response.choices[0].message.content if response.choices else ""
    except Exception as exc:
        return {
            "status": "error",
            "provider": selected_provider,
            "base_url": base_url,
            "model": selected_model,
            "message": _exception_chain(exc),
            "diagnostics": diagnostics,
        }
    return {
        "status": "success",
        "provider": selected_provider,
        "base_url": base_url,
        "model": selected_model,
        "response": (content or "").strip(),
        "diagnostics": diagnostics,
    }


def _network_diagnostics(*, base_url: str, api_key: str, trust_env: bool) -> dict[str, Any]:
    parsed = urlparse(base_url)
    host = parsed.hostname or base_url.replace("https://", "").replace("http://", "").split("/", 1)[0]
    result: dict[str, Any] = {"host": host, "trust_env": trust_env, "dns": [], "http_probe": None}
    try:
        infos = socket.getaddrinfo(host, 443)
        seen: set[str] = set()
        for info in infos:
            ip = str(info[4][0])
            if ip not in seen:
                seen.add(ip)
                result["dns"].append(ip)
    except Exception as exc:
        result["dns_error"] = _exception_chain(exc)

    try:
        probe = httpx.get(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=10,
            trust_env=trust_env,
        )
        result["http_probe"] = {
            "status_code": probe.status_code,
            "body_preview": probe.text[:300],
        }
    except Exception as exc:
        result["http_probe"] = {
            "status": "error",
            "message": _exception_chain(exc),
        }
    return result


def _exception_chain(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    current = exc.__cause__ or exc.__context__
    while current is not None and len(parts) < 4:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _flatten_update_payload(payload: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}

    deepseek = payload.get("deepseek") if isinstance(payload.get("deepseek"), dict) else {}
    openai_cfg = payload.get("openai_expert") if isinstance(payload.get("openai_expert"), dict) else {}

    mapping = {
        "DEEPSEEK_API_KEY": deepseek.get("api_key"),
        "DEEPSEEK_BASE_URL": deepseek.get("base_url"),
        "DEEPSEEK_TRUST_ENV": deepseek.get("trust_env"),
        "DEEPSEEK_CHAT_MODEL": deepseek.get("chat_model"),
        "DEEPSEEK_REASONER_MODEL": deepseek.get("reasoner_model"),
        "DEEPSEEK_CHAT_MODEL_FALLBACKS": deepseek.get("chat_model_fallbacks"),
        "DEEPSEEK_REASONER_MODEL_FALLBACKS": deepseek.get("reasoner_model_fallbacks"),
        "OPENAI_API_KEY": openai_cfg.get("api_key"),
        "OPENAI_BASE_URL": openai_cfg.get("base_url"),
        "OPENAI_MODEL": openai_cfg.get("model"),
        "OPENAI_STRONG_MODEL": openai_cfg.get("strong_model"),
        "OPENAI_BALANCED_MODEL": openai_cfg.get("balanced_model"),
        "OPENAI_FAST_MODEL": openai_cfg.get("fast_model"),
        "OPENAI_CHEAP_MODEL": openai_cfg.get("cheap_model"),
        "LLM_TIMEOUT_SECONDS": payload.get("timeout_seconds"),
    }
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, list):
            value = ";".join(str(item).strip() for item in value if str(item).strip())
        updates[key] = str(value).strip()
    return updates


def _write_env_updates(updates: dict[str, str]) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={_env_value(updates[key])}")
            seen.add(key)
        else:
            output.append(line)
    for key in LLM_ENV_KEYS:
        if key in updates and key not in seen:
            output.append(f"{key}={_env_value(updates[key])}")
    ENV_PATH.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def _env_value(value: str) -> str:
    if any(ch.isspace() for ch in value) or "#" in value:
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
