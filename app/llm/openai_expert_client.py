from __future__ import annotations

from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, BadRequestError, OpenAI

from app.config import get_settings
from app.llm.prompt_registry import get_prompt
from app.utils.logger import setup_logger


logger = setup_logger(__name__)


class OpenAIExpertClientError(RuntimeError):
    """Raised when the OpenAI expert-reasoning client cannot return an answer."""


class OpenAIExpertClient:
    """OpenAI client dedicated to petroleum-chemistry expert reasoning."""

    provider = "openai"

    def __init__(self) -> None:
        self.settings = get_settings()
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(
            api_key=(self.settings.openai_api_key or "").strip() or "MISSING_API_KEY",
            base_url=self.settings.openai_base_url,
            timeout=self.settings.llm_timeout_seconds,
            http_client=http_client,
        )
        self.model = (self.settings.openai_strong_model or self.settings.openai_model).strip()

    def _ensure_api_key(self) -> None:
        if not (self.settings.openai_api_key or "").strip():
            raise OpenAIExpertClientError("OPENAI_API_KEY is not configured.")

    def chat(self, query: str, context: str | None = None, thinking: bool = False) -> str:
        system_prompt = get_prompt("openai.expert.system")
        user_prompt = query if not context else f"{query}\n\nContext:\n{context}"
        return self.chat_with_model(model=self.model, query=user_prompt, system_prompt=system_prompt, temperature=None)

    def chat_with_model(
        self,
        *,
        model: str,
        query: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> str:
        self._ensure_api_key()
        selected_model = (model or self.model).strip()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        request_kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            if selected_model.lower().startswith("gpt-5"):
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens
        try:
            logger.info("Calling OpenAI expert model=%s", selected_model)
            response = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            logger.warning("OpenAI expert request rejected, retrying with compatible parameters: %s", exc)
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("temperature", None)
            message = str(exc)
            if "max_tokens" in message and "max_completion_tokens" in message:
                value = retry_kwargs.pop("max_tokens", None)
                if value is not None:
                    retry_kwargs["max_completion_tokens"] = value
            elif "max_completion_tokens" in message and "max_tokens" in message:
                value = retry_kwargs.pop("max_completion_tokens", None)
                if value is not None:
                    retry_kwargs["max_tokens"] = value
            try:
                response = self.client.chat.completions.create(**retry_kwargs)
            except Exception as retry_exc:
                raise OpenAIExpertClientError(f"OpenAI expert model request failed: {retry_exc}") from retry_exc
        except AuthenticationError as exc:
            raise OpenAIExpertClientError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
        except APITimeoutError as exc:
            raise OpenAIExpertClientError("OpenAI expert model request timed out.") from exc
        except APIConnectionError as exc:
            raise OpenAIExpertClientError("Could not connect to OpenAI API.") from exc
        except APIError as exc:
            raise OpenAIExpertClientError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            raise OpenAIExpertClientError(f"OpenAI expert model call failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise OpenAIExpertClientError("OpenAI expert model returned an empty response.")
        return content.strip()
