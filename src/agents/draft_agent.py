"""Draft Agent — генерирует черновики ответов на входящие сообщения."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

DRAFT_SYSTEM = """Ты — агент-писатель AI-ассистента в Telegram. Пиши черновики ответов на входящие сообщения.

## Стиль
- Лаконично (1-3 предложения)
- На русском
- В стиле владельца (учитывай style_hint если передан)
- Учитывай контекст памяти (memory_hint если передан)
- Учитывай статус владельца (absence_hint если передан)
- Если владелец absent (ушёл) — не обещай быстрого ответа

## Формат ответа
Верни ТОЛЬКО JSON:
{
  "draft": "текст черновика ответа",
  "tone": "warm/friendly/professional/cold",
  "reasoning": "почему такой тон (1 фраза)"
}
"""


async def draft(
    provider,
    sender_name: str,
    incoming_text: str,
    *,
    history_text: str | None = None,
    style_hint: str | None = None,
    memory_hint: str | None = None,
    absence_hint: str | None = None,
) -> dict[str, Any]:
    """Генерирует черновик ответа на входящее сообщение.

    Args:
        provider: Объект LLMProvider с методом chat().
        sender_name: Имя отправителя.
        incoming_text: Текст входящего сообщения.
        history_text: Контекст предыдущей переписки.
        style_hint: Подсказка о стиле владельца.
        memory_hint: Подсказка сохранённых фактов о собеседнике.
        absence_hint: Статус отсутствия владельца.

    Returns:
        Словарь с ключами draft (str), tone (str), reasoning (str).
    """
    parts = [f"Собеседник: {sender_name}"]
    if history_text:
        parts.append(f"Контекст переписки:\n{history_text[:1500]}")
    parts.append(f"Входящее сообщение: {incoming_text}")

    hints = []
    if style_hint:
        hints.append(f"СТИЛЬ ВЛАДЕЛЬЦА:\n{style_hint}")
    if memory_hint:
        hints.append(f"ПАМЯТЬ О СОБЕСЕДНИКЕ:\n{memory_hint}")
    if absence_hint:
        hints.append(f"СТАТУС ВЛАДЕЛЬЦА:\n{absence_hint}")

    if hints:
        parts.append("\n\n".join(hints))

    parts.append("Напиши черновик ответа.")
    user_msg = "\n\n".join(parts)

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=DRAFT_SYSTEM),
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
        return {"draft": raw, "tone": "neutral", "reasoning": "raw output"}
    except Exception:
        logger.debug("Draft parse failed: %s", raw[:100])
        return {
            "draft": "Извини, не могу сейчас ответить.",
            "tone": "neutral",
            "reasoning": "fallback",
        }
