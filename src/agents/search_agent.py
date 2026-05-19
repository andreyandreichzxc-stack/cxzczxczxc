"""Search Agent — находит контакты/чаты по нечёткому запросу."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

SEARCH_SYSTEM = """Ты — поисковый агент в Telegram. Найди контакт или чат по запросу пользователя.

Тебе дан список контактов (имя, username, номер телефона если есть). Выбери наиболее подходящий.

## Формат ответа
Верни ТОЛЬКО JSON:
{
  "found": true/false,
  "display_name": "точное имя контакта",
  "peer_id": 123456789,
  "confidence": 0.95,
  "reason": "почему выбрал именно этот контакт (1 фраза)"
}

Если не нашёл — "found": false, "display_name": null, "peer_id": null.
Если несколько похожих — выбери самый вероятный, укажи confidence < 0.7.
Учитывай ласкательные формы (Настя=Анастасия, Ксю=Ксения, Оля=Ольга).
Учитывай что пользователь может использовать «мама», «брат», «босс», «жена».
"""


async def resolve(provider, query: str, contacts: list[dict]) -> dict[str, Any]:
    """Резолвит контакт по нечёткому запросу.

    Args:
        provider: Объект LLMProvider с методом chat().
        query: Поисковый запрос (имя, ник, роль).
        contacts: Список контактов вида
                  [{"display_name": "...", "peer_id": 123, "username": "..."}, ...].

    Returns:
        Словарь с полями found, display_name, peer_id, confidence, reason.
    """
    contacts_preview = contacts[:50]
    contacts_json = json.dumps(contacts_preview, ensure_ascii=False)

    user_msg = f"Запрос: {query}\n\nСписок контактов:\n{contacts_json}"

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=SEARCH_SYSTEM),
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
        return {"found": False}
    except Exception:
        logger.debug("Search parse failed: %s", raw[:100])
        return {"found": False}
