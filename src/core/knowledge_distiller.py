"""Knowledge Distillation — сжатие 10+ фактов в один summary fact."""

import json
import logging
import re
from datetime import datetime, timezone

from src.db.repo import add_memory, get_or_create_user, list_memories
from src.db.session import get_session
from src.config import settings
from src.core.notification_queue import notification_queue
from src.llm.base import ChatMessage
from src.llm.router import build_provider

logger = logging.getLogger(__name__)

DISTILLATION_PROMPT = """Ты сжимаешь факты памяти в одно связное знание.

Дано: список фактов о контакте или общих фактов.
Сожми их в ОДНО предложение-знание (до 80 символов), которое отражает САМУЮ СУТЬ.

Правила:
- Только ключевая суть, без деталей
- Если факты противоречивы — отрази противоречие: «с одной стороны ... с другой ...»
- Не повторяй очевидные вещи
- Используй настоящее время

Пример:
Факты: «Настя злилась 5 мая», «помирились 6 мая», «общаются нормально с 7 мая», «Настя прислала мем», «обсуждали планы на выходные»
→ «С Настей всё хорошо: после ссоры 5 мая помирились и активно общаются, планируют выходные»

Верни ТОЛЬКО JSON: {"fact": "сжатое знание", "sentiment": "positive|negative|neutral|contradictory"}

Факты для сжатия:
{input_text}
"""


async def distill_cluster(
    provider, owner_id: int, contact_id: int | None = None, min_facts: int = 10
) -> str | None:
    """
    Сжимает факты кластера (contact или общие) в одно summary-знание.
    Возвращает текст факта или None.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner, contact_id=contact_id)
        active = [
            m for m in memories if m.is_active and m.fact and len(m.fact.strip()) >= 5
        ]
        if len(active) < min_facts:
            return None

        # Собираем факты для промпта (первые 30 самых свежих)
        facts_text = "\n".join(f"- {m.fact}" for m in active[:30])

        system = DISTILLATION_PROMPT
        user_text = f"Сожми эти факты ({len(active)} шт.):\n{facts_text}"

        try:
            raw = await provider.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user_text),
                ],
                heavy=False,
            )
        except Exception:
            logger.exception("Distillation LLM call failed")
            return None

        # Парсинг JSON
        raw = raw.strip()
        raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
        raw = re.sub(r"\n?\s*```\s*$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        fact = data.get("fact", "").strip()
        if not fact or len(fact) < 5:
            return None

        sentiment = data.get("sentiment", "neutral")
        if sentiment not in ("positive", "negative", "neutral", "contradictory"):
            sentiment = "neutral"

        return fact


async def run_distillation(owner_id: int, contact_id: int | None = None) -> dict:
    """
    Запускает дистилляцию для контакта (или общих фактов).
    Возвращает {success: bool, fact: str | None, deactivated: int}
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        provider = await build_provider(session, owner)
        if not provider:
            return {"success": False, "fact": None, "deactivated": 0}

    fact = await distill_cluster(provider, owner_id, contact_id)
    if not fact:
        return {"success": False, "fact": None, "deactivated": 0}

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        # Сохраняем distillation-знание как tier-3 (месячное) с высокой важностью
        await add_memory(
            session,
            owner,
            fact=f"💡 {fact}",  # маркер distillation
            sentiment="neutral",
            source="distillation",
            contact_id=contact_id,
            importance=0.9,
            decay_rate=0.02,  # очень медленный decay — важное знание
            memory_tier=3,
        )
        # Недеструктивно: факты остаются активными, summary добавляется как новый факт
        # Для подсчёта сколько фактов "покрыто" дистилляцией
        memories = await list_memories(session, owner, contact_id=contact_id)
        deactivated = 0
        for m in memories:
            if m.is_active and m.source != "distillation":
                # m.is_active = False  # недеструктивно — факты остаются активными
                deactivated += 1
        await session.commit()
        return {"success": True, "fact": fact, "deactivated": deactivated}


async def distillation_loop(owner_id: int) -> None:
    """Фоновый цикл: раз в день (14:00) запускает дистилляцию общих фактов."""
    import asyncio

    from src.core.timeutil import get_user_tz, now_in_tz
    from src.db.models import Notification

    last_run_date = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = get_user_tz(owner)
            now = now_in_tz(tz_name)
            today = now.date()
            if now.hour == 14 and last_run_date != today:
                last_run_date = today
                # Дистилляция общих фактов
                result = await run_distillation(owner_id, contact_id=None)
                if result["success"]:
                    await notification_queue.enqueue(
                        topic="knowledge_distillation",
                        text=(
                            f"🧠 <b>Дистилляция знаний:</b>\n"
                            f"Сжато {result['deactivated']} фактов в одно знание:\n"
                            f"<i>«{result['fact'][:200]}»</i>"
                        ),
                        priority=Notification.PRIORITY_MEDIUM,
                    )
            await asyncio.sleep(settings.knowledge_distiller_interval_sec)
        except Exception:
            logger.exception("Distillation loop error")
            await asyncio.sleep(settings.knowledge_distiller_interval_sec)
