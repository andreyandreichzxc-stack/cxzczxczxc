"""Summarizer Agent — саммаризация переписок, catchup, где остановились."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = """Ты — агент-саммаризатор. Сделай краткую сводку переписки.

## Формат
- Лаконично: 5-7 строк
- Только ключевое: договорённости, темы, эмоциональный фон
- Без HTML-тегов, простой текст с эмодзи для наглядности
- На русском

Верни JSON: {"summary": "текст саммари"}
"""

CATCHUP_SYSTEM = """Ты — агент «где мы остановились». Проанализируй конец переписки и скажи:
1. Где остановились (последняя тема/вопрос)
2. Что требует ответа или действия
3. Черновик ответа если нужно

Верни JSON: {
  "last_topic": "последняя тема (1 фраза)",
  "needs_response": true/false,
  "suggested_reply": "черновик ответа или null",
  "summary": "краткая сводка (2-3 предложения)"
}
"""


async def summarize(provider, messages_text: str) -> dict[str, Any]:
    """Саммаризирует переписку."""
    if not messages_text.strip():
        return {"summary": "Нет сообщений."}

    user_msg = f"Переписка:\n{messages_text[:4000]}"

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=SUMMARY_SYSTEM),
            ChatMessage(role="user", content=user_msg),
        ],
        heavy=False,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        raw = re.sub(r"\n?\s*```\s*$", "", raw)

    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        return {"summary": raw}
    except Exception:
        return {"summary": "Не удалось сделать саммари."}


async def catchup(provider, messages_text: str) -> dict[str, Any]:
    """Анализирует где остановились в переписке."""
    if not messages_text.strip():
        return {
            "last_topic": "Нет сообщений.",
            "needs_response": False,
            "suggested_reply": None,
            "summary": "",
        }

    user_msg = f"Последние сообщения переписки:\n{messages_text[:3000]}"

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=CATCHUP_SYSTEM),
            ChatMessage(role="user", content=user_msg),
        ],
        heavy=False,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        raw = re.sub(r"\n?\s*```\s*$", "", raw)

    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        return {
            "last_topic": raw,
            "needs_response": False,
            "suggested_reply": None,
            "summary": raw,
        }
    except Exception:
        return {
            "last_topic": "Ошибка анализа.",
            "needs_response": False,
            "suggested_reply": None,
            "summary": "",
        }
