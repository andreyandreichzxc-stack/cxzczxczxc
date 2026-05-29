"""OpenAIVisionProvider — обёртка над OpenAI Vision API для анализа изображений.

Использует OpenAI-совместимый API (chat completions с image_url).
"""

import asyncio
import base64
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.llm._ssrf_guard import validate_base_url as _validate_base_url

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Результат анализа изображения с метриками использования токенов."""

    description: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIVisionProvider:
    """Анализирует изображения через OpenAI Vision API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ):
        base_url = _validate_base_url(base_url or "https://api.openai.com/v1")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model

    async def chat_with_image(
        self,
        image_data: bytes,
        image_mime: str,
        prompt: str = "Опиши что на изображении.",
    ) -> VisionResult:
        """Отправляет изображение в API и возвращает описание с метриками токенов."""
        b64_bytes = await asyncio.get_running_loop().run_in_executor(
            None, base64.b64encode, image_data
        )
        b64 = b64_bytes.decode()
        data_url = f"data:{image_mime};base64,{b64}"
        resp = await self._client.chat.completions.create(
            model=self._model or "gpt-5.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        usage = resp.usage
        return VisionResult(
            description=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    async def validate_key(self) -> bool:
        """Лёгкий запрос: валиден ли ключ."""
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self):
        """Закрыть HTTP-клиент."""
        await self._client.close()
