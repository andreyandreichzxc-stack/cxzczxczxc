"""Memory Agent — извлекает и хранит факты о контактах."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

MEMORY_SYSTEM = """Ты — агент памяти AI-ассистента. Извлекай факты о людях из переписки.

## Что извлекать
- Личные факты: возраст, профессия, место жительства, хобби
- Предпочтения: что любит/не любит, вкусы
- Отношения: как относится к владельцу, конфликты, тёплые моменты
- События: важные даты, договорённости, планы
- Эмоциональное состояние: радость, грусть, злость (sentiment)

## Формат ответа
Верни ТОЛЬКО JSON:
{
  "facts": [
    {"fact": "текст факта (1 предложение)", "sentiment": "positive/negative/neutral"}
  ],
  "summary": "краткая сводка о контакте (2-3 предложения)"
}

Не выдумывай факты. Если в переписке нет значимой информации — верни пустой список facts.
"""


async def extract_facts(provider, messages_text: str) -> dict[str, Any]:
    """Извлекает факты из текста переписки.

    Args:
        provider: Объект LLMProvider с методом chat().
        messages_text: Текст переписки для анализа.

    Returns:
        Словарь с ключами facts (list[dict]) и summary (str).
    """
    if not messages_text or len(messages_text.strip()) < 20:
        return {"facts": [], "summary": ""}

    user_msg = f"Переписка:\n{messages_text[:3000]}"

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=MEMORY_SYSTEM),
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
        return {"facts": [], "summary": ""}
    except Exception:
        logger.debug("Memory parse failed: %s", raw[:100])
        return {"facts": [], "summary": ""}


RECALL_SYSTEM = """Ты — агент памяти. У тебя есть сохранённые факты о контакте.
Ответь на вопрос пользователя, используя ТОЛЬКО эти факты.

## Формат ответа
Верни JSON: {"answer": "ответ на основе фактов", "relevant_facts": ["факт1", "факт2"]}
Если фактов недостаточно — "answer": "недостаточно данных".
"""


async def recall(provider, query: str, facts: list[str]) -> dict[str, Any]:
    """Отвечает на вопрос о контакте на основе сохранённых фактов.

    Args:
        provider: Объект LLMProvider с методом chat().
        query: Вопрос пользователя о контакте.
        facts: Список сохранённых фактов (строки).

    Returns:
        Словарь с ключами answer (str) и relevant_facts (list[str]).
    """
    if not facts:
        return {"answer": "Нет сохранённых фактов.", "relevant_facts": []}

    facts_str = "\n".join(f"- {f}" for f in facts[:20])
    user_msg = f"Факты о контакте:\n{facts_str}\n\nВопрос: {query}"

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=RECALL_SYSTEM),
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
        return {"answer": raw, "relevant_facts": []}
    except Exception:
        return {"answer": "Не удалось проанализировать.", "relevant_facts": []}
