from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import httpx
from openai import APITimeoutError, OpenAI
from openai import APIConnectionError, APIError, AuthenticationError, BadRequestError

from app.config import get_settings
from app.llm.prompt_registry import get_prompt
from app.utils.logger import setup_logger


logger = setup_logger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")


class DeepSeekClientError(RuntimeError):
    """Friendly error raised for upstream LLM failures."""


def _exception_chain(exc: BaseException) -> str:
    parts = [str(exc)]
    current = exc.__cause__ or exc.__context__
    while current is not None and len(parts) < 4:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(part for part in parts if part)


def infer_output_language(text: str) -> str:
    """Prefer English for mainly ASCII input; otherwise prefer Chinese."""

    raw = (text or "").strip()
    if not raw:
        return "zh"

    cjk_chars = len(_CJK_RE.findall(raw))
    ascii_words = _ASCII_WORD_RE.findall(raw)
    ascii_letters = sum(len(word) for word in ascii_words)

    if cjk_chars == 0 and ascii_letters > 0:
        return "en"
    if ascii_letters >= max(12, cjk_chars * 2):
        return "en"
    return "zh"


class DeepSeekClient:
    """OpenAI SDK compatible DeepSeek client wrapper."""

    def __init__(self) -> None:
        self.settings = get_settings()
        http_client = httpx.Client(trust_env=bool(self.settings.deepseek_trust_env))
        self.client = OpenAI(
            api_key=self.settings.deepseek_api_key or "MISSING_API_KEY",
            base_url=self.settings.deepseek_base_url,
            timeout=self.settings.llm_timeout_seconds,
            http_client=http_client,
        )

    def _ensure_api_key(self) -> None:
        if not self.settings.deepseek_api_key:
            raise DeepSeekClientError("DEEPSEEK_API_KEY is not configured. Set it in the .env file first.")

    def _build_chat_messages(self, query: str, context: str | None = None) -> list[dict[str, str]]:
        output_language = infer_output_language(query)
        if output_language == "en":
            system_prompt = get_prompt("deepseek.chat.system.en")
        else:
            system_prompt = get_prompt("deepseek.chat.system.zh")
        user_prompt = (
            query
            if not context
            else f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_reason_messages(self, query: str) -> list[dict[str, str]]:
        system_prompt = get_prompt("deepseek.reason.system")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    def _chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        self._ensure_api_key()

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if thinking:
            request_kwargs["extra_body"] = {"thinking": {"enabled": True}}

        try:
            logger.info("Calling DeepSeek model=%s thinking=%s", model, thinking)
            response = self.client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content if response.choices else None
            if not content:
                raise DeepSeekClientError("The model returned an empty response. Please try again later.")
            return content.strip()
        except BadRequestError as exc:
            if thinking:
                logger.warning("Thinking parameter rejected, retrying without thinking: %s", exc)
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("extra_body", None)
                response = self.client.chat.completions.create(**retry_kwargs)
                content = response.choices[0].message.content if response.choices else None
                if not content:
                    raise DeepSeekClientError("The model returned an empty response. Please try again later.") from exc
                return content.strip()
            logger.exception("Bad request sent to DeepSeek.")
            raise DeepSeekClientError(f"Invalid model request parameters: {exc}") from exc
        except AuthenticationError as exc:
            logger.exception("DeepSeek authentication failed.")
            raise DeepSeekClientError("DeepSeek authentication failed. Check whether DEEPSEEK_API_KEY is correct.") from exc
        except APITimeoutError as exc:
            logger.exception("DeepSeek request timed out.")
            raise DeepSeekClientError("The model request timed out. Retry later or increase LLM_TIMEOUT_SECONDS.") from exc
        except APIConnectionError as exc:
            logger.exception("Failed to connect to DeepSeek.")
            raise DeepSeekClientError(
                "Could not connect to the DeepSeek endpoint. "
                f"Check network, DNS, proxy, TLS, or base_url settings. Detail: {_exception_chain(exc)}"
            ) from exc
        except APIError as exc:
            logger.exception("DeepSeek API error.")
            raise DeepSeekClientError(f"DeepSeek service error: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected DeepSeek client error.")
            raise DeepSeekClientError(f"Model call failed: {exc}") from exc

    def _extract_stream_delta(self, chunk: Any) -> str:
        if not getattr(chunk, "choices", None):
            return ""
        delta = getattr(chunk.choices[0], "delta", None)
        if not delta:
            return ""
        content = getattr(delta, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content)

    def _chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self._ensure_api_key()

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if thinking:
            request_kwargs["extra_body"] = {"thinking": {"enabled": True}}

        try:
            logger.info("Streaming DeepSeek model=%s thinking=%s", model, thinking)
            stream = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            if thinking:
                logger.warning("Thinking parameter rejected for stream, retrying without thinking: %s", exc)
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("extra_body", None)
                try:
                    stream = self.client.chat.completions.create(**retry_kwargs)
                except Exception as retry_exc:
                    logger.exception("Retry without thinking failed for streaming request.")
                    raise DeepSeekClientError(f"Model call failed: {retry_exc}") from retry_exc
            else:
                logger.exception("Bad request sent to DeepSeek.")
                raise DeepSeekClientError(f"Invalid model request parameters: {exc}") from exc
        except AuthenticationError as exc:
            logger.exception("DeepSeek authentication failed.")
            raise DeepSeekClientError("DeepSeek authentication failed. Check whether DEEPSEEK_API_KEY is correct.") from exc
        except APITimeoutError as exc:
            logger.exception("DeepSeek request timed out.")
            raise DeepSeekClientError("The model request timed out. Retry later or increase LLM_TIMEOUT_SECONDS.") from exc
        except APIConnectionError as exc:
            logger.exception("Failed to connect to DeepSeek.")
            raise DeepSeekClientError(
                "Could not connect to the DeepSeek endpoint. "
                f"Check network, DNS, proxy, TLS, or base_url settings. Detail: {_exception_chain(exc)}"
            ) from exc
        except APIError as exc:
            logger.exception("DeepSeek API error.")
            raise DeepSeekClientError(f"DeepSeek service error: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected DeepSeek client error.")
            raise DeepSeekClientError(f"Model call failed: {exc}") from exc

        yielded = False
        try:
            for chunk in stream:
                delta_text = self._extract_stream_delta(chunk)
                if delta_text:
                    yielded = True
                    yield delta_text
            if not yielded:
                raise DeepSeekClientError("The model returned an empty response. Please try again later.")
        except DeepSeekClientError:
            raise
        except Exception as exc:
            logger.exception("Unexpected DeepSeek streaming error.")
            raise DeepSeekClientError(f"Model call failed: {exc}") from exc

    def chat(self, query: str, context: str | None = None, thinking: bool = False) -> str:
        return self._chat_completion(
            model=self.settings.deepseek_chat_model,
            messages=self._build_chat_messages(query, context),
            thinking=thinking,
        )

    def chat_with_model(
        self,
        *,
        model: str,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        selected_model = (model or self.settings.deepseek_chat_model).strip() or self.settings.deepseek_chat_model
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        return self._chat_completion(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )

    def chat_stream(self, query: str, context: str | None = None, thinking: bool = False) -> Iterator[str]:
        return self._chat_completion_stream(
            model=self.settings.deepseek_chat_model,
            messages=self._build_chat_messages(query, context),
            thinking=thinking,
        )

    def reason(self, query: str) -> str:
        return self._chat_completion(
            model=self.settings.deepseek_reasoner_model,
            messages=self._build_reason_messages(query),
            temperature=0.1,
        )

    def reason_stream(self, query: str) -> Iterator[str]:
        return self._chat_completion_stream(
            model=self.settings.deepseek_reasoner_model,
            messages=self._build_reason_messages(query),
            temperature=0.1,
        )
