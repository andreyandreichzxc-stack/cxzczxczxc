"""Генерация проактивных пингов — бот сам предлагает действия."""

import asyncio
import logging
from datetime import datetime, timezone

from src.core.infra.text_sanitizer import sanitize_html

logger = logging.getLogger(__name__)

PING_PROMPT = """Ты — ассистент, который анализирует контекст пользователя и предлагает полезные действия.

КОНТЕКСТ:
- Память (факты): {memory_facts}
- Активные напоминания: {reminders}
- Неотвеченные вопросы: {pending_questions}
- Предстоящие события: {upcoming_events}

Предложи 1-2 полезных действия. Формат: одно действие на строку.
Примеры:
- "Маша завтра прилетает в 14:00 — напомнить за час?"
- "Ты просил найти книгу 'Sapiens' — я нашёл, показать?"
- "Давно не общался с Петей — написать?"
- "Завтра дедлайн по проекту — создать напоминание?"

Если ничего полезного — ответь "ничего".

Действия:"""


async def generate_pings(owner, provider, session) -> list[str]:
    """Генерирует 1-2 проактивных пинга.

    Анализирует память, неотвеченные вопросы, предстоящие события
    и предлагает пользователю полезные действия.

    Параметры:
        owner: User ORM объект.
        provider: LLMProvider для генерации пингов.
        session: активная AsyncSession.

    Возвращает:
        Список строк-пингов (0-2 элемента).
    """
    try:
        from src.core.memory.memory_recall import recall
        from src.core.memory.pending_questions import get_pending
        from src.llm.base import ChatMessage

        # Собираем контекст
        recall_result = await recall(
            owner.telegram_id, query="", limit=10, mode="light"
        )
        facts = (
            "\n".join(f"- {f.fact}" for f in recall_result.facts[:10])
            if recall_result.facts
            else "нет"
        )

        pending = await get_pending(owner.telegram_id)
        pending_str = (
            "\n".join(f"- {q['question'][:100]}" for q in pending[:5])
            if pending
            else "нет"
        )

        # Упрощённо — upcoming events из памяти с датами
        upcoming = "нет данных"
        for f in recall_result.facts or []:
            if any(
                w in (f.fact or "").lower()
                for w in ["завтра", "сегодня", "через", "встреча", "дедлайн"]
            ):
                upcoming = f"- {f.fact[:100]}"
                break

        prompt = PING_PROMPT.format(
            memory_facts=facts[:500],
            reminders="нет активных",
            pending_questions=pending_str[:300],
            upcoming_events=upcoming[:300],
        )

        resp = await asyncio.wait_for(
            provider.chat([ChatMessage(role="user", content=prompt)]),
            timeout=30.0,
        )

        # Hallucination guard для проактивных пингов
        try:
            from src.core.intelligence.hallucination_guard import (
                verify_claims,
                apply_guard,
            )

            memory_facts = (
                [f" - {f.fact}" for f in recall_result.facts[:20]]
                if recall_result and recall_result.facts
                else []
            )
            contact_names = []  # Можно расширить если есть доступ к контактам

            verify_result = await verify_claims(resp, memory_facts, contact_names)
            if not verify_result.get("ok"):
                logger.warning("Proactive ping hallucination detected, applying guard")
                resp, modified = apply_guard(resp, verify_result, 0.5)
        except Exception:
            pass  # best-effort, не ломаем пинги

        lines = [
            sanitize_html(l.strip("- ").strip())
            for l in resp.split("\n")
            if l.strip() and l.strip().lower() != "ничего"
        ]
        return lines[:2]

    except Exception as e:
        logger.debug("Ping generation failed: %s", e)
        return []
