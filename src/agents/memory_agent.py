"""Memory Agent — извлекает и хранит факты о контактах."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

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
