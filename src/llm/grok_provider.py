"""Grok (xAI) провайдер — OpenAI-совместимый API от xAI.

Модели: grok-4.3 (latest), grok-4.20-0309-reasoning, grok-4.20-0309-non-reasoning.
Base URL: https://api.x.ai/v1
API docs: https://docs.x.ai/docs

⚠️ xAI Grok не поддерживает embeddings. Провайдер использует OpenAICompatBaseMixin
(только chat + validate + list_models). Embeddings берутся из других провайдеров через fallback-цепочку.
"""

import httpx
from collections.abc import AsyncGenerator
from openai import AsyncOpenAI

from src.config import LLMDefaults
from src.llm._openai_compat_mixin import OpenAICompatBaseMixin
from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage


GROK_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(OpenAICompatBaseMixin):
    """Провайдер для Grok (xAI) — OpenAI-совместимый API. Без embeddings."""

    name = "grok"

    def __init__(
        self, api_key: str, *, base_url: str | None = None, model: str | None = None
    ) -> None:
        base_url = _validate_base_url(base_url)
        kwargs: dict = dict(
            api_key=api_key,
            base_url=base_url or GROK_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (
            LLMDefaults.GROK_CHAT_HEAVY if heavy else LLMDefaults.GROK_CHAT_LIGHT
        )

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = self._resolve_model(heavy)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(
        self, messages: list[ChatMessage], *, heavy: bool = False
    ) -> AsyncGenerator[str, None]:
        model = self._resolve_model(heavy)
        fmt = [{"role": m.role, "content": m.content} for m in messages]
        stream = await self._client.chat.completions.create(
            model=model, messages=fmt, stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
