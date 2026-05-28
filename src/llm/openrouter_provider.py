"""OpenRouter провайдер — бесплатный доступ к DeepSeek V4 Flash и другим моделям.

OpenRouter предоставляет единый OpenAI-совместимый endpoint для 300+ моделей.
Free tier: 20 RPM, 200 RPD (1000 с $10 lifetime депозитом), 33 бесплатные модели.
Подробнее: https://openrouter.ai/docs/api/reference/limits

DeepSeek V4 Flash (free): 1M контекст, reasoning, coding — топ бесплатная модель.
"""

import httpx
from openai import AsyncOpenAI

from src.llm._openai_compat_mixin import OpenAICompatBaseMixin
from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


DEFAULT_MODEL = "deepseek/deepseek-v4-flash:free"
HEAVY_MODEL = "deepseek/deepseek-v4-flash:free"


class OpenRouterProvider(OpenAICompatBaseMixin):
    """Провайдер для OpenRouter free models (DeepSeek V4 Flash и другие).

    OpenAI-совместимый API. Не поддерживает embeddings (free tier без эмбеддингов).
    """

    name = "openrouter"

    def __init__(
        self, api_key: str, *, base_url: str | None = None, model: str | None = None
    ) -> None:
        base_url = _validate_base_url(base_url)
        kwargs: dict = dict(
            api_key=api_key,
            base_url=base_url or OPENROUTER_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
            default_headers={
                "HTTP-Referer": "https://github.com/tashfeenahmed/freellmapi",
                "X-Title": "TelegramHelper",
            },
        )
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (HEAVY_MODEL if heavy else DEFAULT_MODEL)

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = self._resolve_model(heavy)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            extra_headers={
                "X-Title": "TelegramHelper",
            },
        )
        return resp.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "OpenRouter free tier не поддерживает embeddings. "
            "Используй OpenAI или другой провайдер для эмбеддингов."
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "OpenRouter free tier не поддерживает embeddings. "
            "Используй OpenAI или другой провайдер для эмбеддингов."
        )
