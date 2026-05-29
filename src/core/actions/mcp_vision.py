"""MCP Tool: анализ изображений через Vision API — любой vision-capable провайдер."""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

ALLOWED_BASE = Path("data").resolve()

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Базовые URL для vision-провайдеров
_VISION_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "mimo": "https://api.xiaomimimo.com/v1",
    "cloudflare": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    "grok": "https://api.x.ai/v1",
}


async def _try_openai_compat_vision(
    api_key: str,
    base_url: str,
    image_data: bytes,
    mime: str,
    prompt: str,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Пробует OpenAI-совместимый Vision API."""
    try:
        from src.llm.vision_provider import OpenAIVisionProvider

        provider = OpenAIVisionProvider(api_key, base_url=base_url, model=model)
        result = await asyncio.wait_for(
            provider.chat_with_image(image_data, mime, prompt),
            timeout=120.0,
        )
        return {
            "ok": True,
            "description": result.description[:2000],
            "tokens": result.total_tokens,
        }
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        logger.debug("OpenAI-compat vision failed (%s): %s", base_url, e)
        return None
    finally:
        if "provider" in locals():
            await provider.close()


async def _try_gemini_vision(
    api_key: str,
    image_data: bytes,
    mime: str,
    prompt: str,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Пробует Gemini multimodal Vision API."""
    try:
        from google import genai
    except ImportError:
        return None

    client = genai.Client(api_key=api_key)
    _model = model or "gemini-2.5-flash"

    def _run() -> str:
        from google.genai import types as genai_types

        part = genai_types.Part.from_bytes(data=image_data, mime_type=mime)
        resp = client.models.generate_content(
            model=_model,
            contents=[prompt, part],
        )
        return resp.text or ""

    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_run),
            timeout=120.0,
        )
        if text:
            return {"ok": True, "description": text[:2000], "tokens": 0}
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        logger.debug("Gemini vision failed: %s", e)

    return None


@tool(
    name="analyze_image",
    description=(
        "Анализирует изображение и возвращает текстовое описание. "
        "Работает с любым vision-провайдером (OpenAI, Gemini, MiMo, Cloudflare). "
        "Используй когда нужно понять что на фото."
    ),
    category="vision",
    risk="medium",
    params={
        "file_path": "str — путь к файлу изображения",
        "prompt": "str — что именно нужно узнать об изображении (по умолчанию: опиши что на фото)",
    },
)
async def analyze_image(
    file_path: str = "",
    prompt: str = "Опиши что на изображении.",
    **kwargs: Any,
) -> dict[str, Any]:
    """Анализирует изображение через первый доступный vision-провайдер."""
    if not file_path:
        return {"error": "file_path обязателен"}

    path = Path(file_path)
    if not path.exists():
        return {"error": f"Файл не найден: {file_path}"}

    # Path traversal protection
    resolved = path.resolve()
    if not (str(resolved) + os.sep).startswith(str(ALLOWED_BASE) + os.sep):
        return {"error": "Доступ к файлу запрещён"}

    try:
        from src.llm.provider_catalog import get_provider
        from src.db.repo import get_or_create_user, list_key_slots
        from src.db.session import get_session
        from src.crypto import decrypt

        # Читаем изображение (non-blocking)
        image_data = await asyncio.to_thread(path.read_bytes)
        mime = _MIME_MAP.get(path.suffix.lower(), "image/jpeg")

        # Получаем пользователя
        user = kwargs.get("user")
        telegram_id = user.telegram_id if user else 0
        if telegram_id == 0:
            return {"error": "Не удалось определить пользователя"}

        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            slots = await list_key_slots(session, owner)

        # Фильтруем: enabled + LLM + vision-capable провайдер
        vision_slots = []
        for slot in slots:
            if not slot.enabled:
                continue
            if slot.category not in (None, "llm", "vision"):
                continue
            pi = get_provider(slot.provider)
            if pi and pi.supports_vision:
                vision_slots.append((slot, pi))

        if not vision_slots:
            return {
                "error": (
                    "Нет доступного vision-провайдера. "
                    "Добавь ключ OpenAI, Gemini, MiMo или Cloudflare в /settings → API-ключи. "
                    "У этих провайдеров есть поддержка анализа изображений."
                )
            }

        # Пробуем каждый слот по порядку
        errors = []
        for slot, pi in vision_slots:
            try:
                api_key = decrypt(slot.key_enc)
            except Exception:
                continue

            logger.debug("Trying vision via %s (slot %s)", slot.provider, slot.id)

            # ── Gemini: отдельный код-путь ──
            if slot.provider == "gemini":
                result = await _try_gemini_vision(
                    api_key, image_data, mime, prompt, slot.model
                )
                if result:
                    result["provider"] = "gemini"
                    return result
                errors.append(f"gemini: vision failed")
                continue

            # ── OpenAI-совместимые: openai, mimo, cloudflare, custom ──
            base_url = _VISION_ENDPOINTS.get(slot.provider)
            if base_url is None:
                # custom/openai-compatible: пробуем endpoint из слота
                base_url = slot.endpoint
            if base_url is None:
                base_url = "https://api.openai.com/v1"

            result = await _try_openai_compat_vision(
                api_key, base_url, image_data, mime, prompt, slot.model
            )
            if result:
                result["provider"] = slot.provider
                return result
            errors.append(f"{slot.provider}: vision API failed")

        return {"error": f"Все vision-провайдеры отказали: {'; '.join(errors)}"}

    except Exception as e:
        return {"error": str(e)}
