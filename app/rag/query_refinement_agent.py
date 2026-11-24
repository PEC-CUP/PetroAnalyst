from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx
from openai import OpenAI

from app.config import get_settings
from app.llm.prompt_registry import get_prompt, render_prompt


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def _looks_like_junk(q: str) -> bool:
    q2 = (q or "").strip()
    if not q2:
        return True
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", q2)
    alnum_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/.]{2,}", q2)
    return (not cjk_tokens) and (len(alnum_tokens) < 2)


@dataclass(frozen=True, slots=True)
class RefinementResult:
    queries: list[str]
    raw: dict


class QueryRefinementAgent:
    """LLM-driven multi-query generator.

    This agent is intentionally separated from the retrieval backend:
    - It can be driven by evaluation feedback (missing_points).
    - It logs/returns its own JSON output for reproducibility.
    - Retrieval remains responsible only for ranking/aggregation, not rewriting.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.deepseek_api_key or "MISSING_API_KEY",
            base_url=settings.deepseek_base_url,
            timeout=settings.llm_timeout_seconds,
            http_client=httpx.Client(trust_env=bool(settings.deepseek_trust_env)),
        )
        self._model = settings.deepseek_chat_model

    def refine(
        self,
        *,
        question: str,
        missing_points: list[str] | None = None,
        max_queries: int = 4,
    ) -> RefinementResult:
        q = (question or "").strip()
        max_q = max(1, min(int(max_queries), 8))
        missing = [str(x).strip() for x in (missing_points or []) if str(x).strip()]

        sys = get_prompt("rag.query_refinement.system")
        missing_points_block = ""
        if missing:
            missing_points_block = "\nEvaluation missing_points (use these to focus some queries):\n" + "\n".join(
                f"- {m}" for m in missing[:8]
            )
        user = render_prompt(
            "rag.query_refinement.user",
            max_queries=max_q,
            question=q,
            missing_points_block=missing_points_block,
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        payload = _extract_json((content or "").strip())

        queries = payload.get("queries")
        if not isinstance(queries, list):
            # Fail-soft: return the original question as the only query.
            return RefinementResult(queries=[q] if q else [], raw={"error": "invalid_json", "content": content})

        out: list[str] = []
        seen = set()
        for item in queries:
            if not isinstance(item, str):
                continue
            cand = item.strip()
            if not cand:
                continue
            if len(cand) > 160:
                cand = cand[:160].rstrip()
            if _looks_like_junk(cand):
                continue
            key = cand.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
            if len(out) >= max_q:
                break

        if not out and q:
            out = [q]
        return RefinementResult(queries=out, raw=payload)
