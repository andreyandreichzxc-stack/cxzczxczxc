"""Router Agent — классифицирует интент пользователя."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = """Ты — роутер AI-ассистента в Telegram. Классифицируй сообщение пользователя.

## Возможные интенты
- send_message — отправить сообщение контакту
- summarize_chat — саммаризировать/спросить про переписку
- draft_reply — написать черновик ответа
- catchup — «где мы остановились» с контактом
- search — найти сообщения/контакты
- memory — работа с памятью: запомнить факт, спросить что знаешь
- extract_memories — извлечь факты из переписки
- list_memories — показать что знаешь о контакте
- forget_memory — забыть факт
- check_memories — проверить актуальность памяти
- digest — дайджест новостей/входящих
- set_setting — изменить настройку
- list_todos — показать активные напоминания
- add_reminder — поставить напоминание
- remove_reminder — убрать напоминание
- add_news_topic — добавить тему для авто-новостей
- remove_news_topic — убрать тему
- chat — общий вопрос/болтовня
- unknown — не понял

## Параметры
- contact / peer_query — имя контакта (Оля, Настя, мама)
- text / query / fact — текстовая часть
- when — дата/время
- topic — тема
- sentiment — positive/negative/neutral

## Формат ответа
Верни ТОЛЬКО JSON:
{
  "intent": "...",
  "contact": "имя контакта или null",
  "text": "текст сообщения или null",
  "when": "дата или null",
  "query": "поисковый запрос или null",
  "fact": "факт для памяти или null",
  "sentiment": "positive/negative/neutral или null",
  "topic": "тема или null",
  "setting_key": "ключ настройки или null",
  "setting_value": "значение или null",
  "confidence": 0.95
}
"""


async def route(
    provider, user_text: str, *, history: str | None = None
) -> dict[str, Any]:
    """Классифицирует интент пользователя.

    Args:
        provider: Объект LLMProvider с методом chat().
        user_text: Текст сообщения пользователя.
        history: Необязательная история диалога для контекста.

    Returns:
        Словарь с классификацией: intent, contact, text, when, query,
        fact, sentiment, topic, setting_key, setting_value, confidence.
    """
    user_msg = f"Сообщение: {user_text}"
    if history:
        user_msg = f"История диалога:\n{history}\n\n{user_msg}"

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=ROUTER_SYSTEM),
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
        return {"intent": "unknown", "confidence": 0.0}
    except Exception:
        logger.debug("Router parse failed: %s", raw[:100])
        return {"intent": "unknown", "confidence": 0.0}
