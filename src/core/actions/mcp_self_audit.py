"""MCP Tool: самоанализ — что модель знает о пользователе."""

import logging
from collections import Counter
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="self_audit",
    description=(
        "Анализирует что модель знает о пользователе: факты из памяти, "
        "частые темы, настроение. Возвращает структурированный отчёт."
    ),
    category="memory",
    risk="low",
    params={
        "user_id": "int — Telegram ID пользователя (опционально, по умолчанию — владелец)",
        "limit": "int — макс. фактов для анализа (10-100, по умолчанию 30)",
    },
)
async def self_audit(
    user_id: int = 0,
    limit: int = 30,
    **kwargs: Any,
) -> dict[str, Any]:
    """Анализирует память модели о пользователе.

    Возвращает структурированный отчёт: секции по memory_type,
    частые слова, общее количество фактов и текстовую подсказку.
    """
    limit = max(10, min(100, limit))

    try:
        from src.config import settings
        from src.db.repos.memory_repo import list_memories
        from src.db.repos.session_repo import get_or_create_user
        from src.db.session import get_session

        user = kwargs.get("user")
        telegram_id = user_id or (
            user.telegram_id if user else settings.owner_telegram_id
        )

        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            memories = await list_memories(session, owner, is_active=True, limit=limit)

        if not memories:
            return {
                "ok": True,
                "total_facts": 0,
                "sections": [],
                "suggestion": (
                    "У меня пока нет сохранённых фактов о тебе. "
                    "Продолжай общаться со мной — я запоминаю важное."
                ),
            }

        # Группируем по memory_type
        by_type: dict[str, list[str]] = {}
        for m in memories:
            t = getattr(m, "memory_type", "general") or "general"
            fact = getattr(m, "fact", "") or ""
            if fact:
                by_type.setdefault(t, []).append(fact[:200])

        sections = []
        for mem_type, facts in by_type.items():
            sections.append(
                {
                    "type": mem_type,
                    "count": len(facts),
                    "sample": facts[:5],
                }
            )

        # Частые слова (простая аналитика)
        all_text = " ".join(f for facts in by_type.values() for f in facts)
        words = [w.lower() for w in all_text.split() if len(w) > 3]
        top_words = [
            w
            for w, _ in Counter(words).most_common(15)
            if w
            not in {
                "который",
                "когда",
                "чтобы",
                "потому",
                "очень",
                "может",
            }
        ]

        return {
            "ok": True,
            "total_facts": len(memories),
            "sections": sections,
            "top_words": top_words[:10],
            "suggestion": (
                f"У меня {len(memories)} фактов о тебе. "
                f"Частые темы: {', '.join(top_words[:5])}. "
                f"Категории: {', '.join(by_type.keys())}."
            ),
        }

    except Exception as e:
        return {"error": str(e)[:300]}
