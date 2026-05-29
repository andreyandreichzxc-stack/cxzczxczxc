"""Groq провайдер — быстрый инференс, OpenAI-совместимый API.

Модели: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it.
Base URL: https://api.groq.com/openai/v1
API docs: https://console.groq.com/docs

⚠️ Groq не поддерживает embeddings. Провайдер использует OpenAICompatBaseMixin
(только chat + validate + list_models). Embeddings берутся из других провайдеров через fallback-цепочку.
"""

import httpx
from collections.abc import AsyncGenerator
from openai import AsyncOpenAI

from src.llm._openai_compat_mixin import OpenAICompatBaseMixin
from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_LIGHT = "llama-3.3-70b-versatile"
GROQ_CHAT_HEAVY = "mixtral-8x7b-32768"


class GroqProvider(OpenAICompatBaseMixin):
    """Провайдер для Groq — OpenAI-совместимый API. Без embeddings."""

    name = "groq"

    def __init__(
        self, api_key: str, *, base_url: str | None = None, model: str | None = None
    ) -> None:
        base_url = _validate_base_url(base_url)
        kwargs: dict = dict(
            api_key=api_key,
            base_url=base_url or GROQ_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (GROQ_CHAT_HEAVY if heavy else GROQ_CHAT_LIGHT)

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
