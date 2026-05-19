"""Weekly summarizer — раз в неделю саммари переписки с каждым активным контактом."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from src.core.chat_service import message_to_text
from src.core.notifier import notifier
from src.core.timeutil import now_in_tz
from src.db.repo import (
    add_memory,
    fetch_chat_messages,
    get_or_create_user,
    list_contacts,
    upsert_memory_cluster,
)
from src.db.session import get_session
from src.llm.base import ChatMessage
from src.llm.router import build_provider

logger = logging.getLogger(__name__)

WEEKLY_SUMMARY_PROMPT = (
    "Проанализируй переписку за неделю и извлеки КЛЮЧЕВЫЕ факты, темы, настроения.\n"
    "Верни JSON-список фактов (не больше 5):\n"
    '[{"fact": "краткий факт (1 предложение)", "sentiment": "positive|negative|neutral"}, ...]\n'
    "Пиши в третьем лице: «обсуждали проект», «Настя рассказала про отпуск».\n"
    "Не включай технические детали (ссылки, даты), только содержательные факты."
)


async def summarize_contact_week(
    provider, owner_id: int, contact, since_days: int = 7
) -> list[dict]:
    """Саммари переписки с одним контактом за N дней."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        messages = await fetch_chat_messages(session, owner, contact.peer_id, limit=200)
        if not messages:
            return []

        # Фильтр по дате (since)
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        recent = [m for m in messages if m.date and m.date >= cutoff]
        if len(recent) < 5:
            return []

        # Построить транскрипт (message_to_text уже добавляет [время] Кто: текст)
        transcript = "\n".join(message_to_text(m) for m in recent[-100:])

        # LLM
        try:
            raw = await provider.chat(
                [
                    ChatMessage(role="system", content=WEEKLY_SUMMARY_PROMPT),
                    ChatMessage(
                        role="user",
                        content=f"Контакт: {contact.display_name}\n\nПереписка:\n{transcript[:5000]}",
                    ),
                ],
                heavy=False,
            )
            # Парсинг
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            raw = re.sub(r"\n?\s*```\s*$", "", raw)
            items = json.loads(raw)
            if not isinstance(items, list):
                items = []
        except Exception:
            logger.exception("LLM weekly summary failed for %s", contact.display_name)
            return []
        return items


async def weekly_summary_loop(owner_id: int) -> None:
    """Фоновый цикл: раз в неделю (воскресенье 12:00) саммари всех контактов."""
    last_run_date = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = (
                    owner.settings.timezone
                    if owner.settings and owner.settings.timezone
                    else "UTC"
                )

            now = now_in_tz(tz_name)
            # Воскресенье, 12:00
            if (
                now.weekday() == 6
                and 12 <= now.hour < 13
                and last_run_date != now.date()
            ):
                last_run_date = now.date()
                async with get_session() as session:
                    owner_safe = await get_or_create_user(session, owner_id)
                    provider = await build_provider(session, owner_safe)
                    if not provider:
                        continue

                    # Получить папки для фильтра
                    monitored = (
                        json.loads(owner_safe.settings.monitored_folders)
                        if owner_safe.settings.monitored_folders
                        else []
                    )
                    contacts = await list_contacts(
                        session, owner_safe, kinds=("user",), include_bots=False
                    )
                    if monitored:
                        contacts = [
                            c
                            for c in contacts
                            if any(
                                f.strip() in (c.folder_names or "").split(",")
                                for f in monitored
                            )
                        ]
                    total_facts = 0
                    for contact in contacts[:20]:  # макс 20 контактов
                        facts = await summarize_contact_week(
                            provider, owner_id, contact
                        )
                        for f in facts:
                            await add_memory(
                                session,
                                owner_safe,
                                fact=f.get("fact", ""),
                                contact_id=contact.peer_id,
                                sentiment=f.get("sentiment"),
                                source="weekly",
                            )
                            total_facts += 1
                        await asyncio.sleep(0.3)  # rate-limit LLM
                    if total_facts > 0:
                        await notifier.notify(
                            f"📊📝 <b>Недельное саммари:</b> {total_facts} фактов "
                            f"из {len(contacts[:20])} контактов сохранено в память."
                        )
                        # После сохранения weekly фактов — консолидация
                        consolidated = await consolidate_tier(
                            provider, owner_id, from_tier=1, to_tier=2
                        )
                        if consolidated > 0:
                            logger.info(
                                "Consolidated %d episodic facts into weekly tier",
                                consolidated,
                            )
            await asyncio.sleep(3600)  # проверка раз в час
        except Exception as e:
            logger.error("Weekly summary error: %s", e)
            await asyncio.sleep(3600)


CONSOLIDATION_PROMPT = (
    "Ты сжимаешь список фактов-воспоминаний в краткое саммари (3-5 предложений). "
    "Сохрани ВСЕ важные детали, но убери повторения и слей похожие факты в один. "
    "Пиши на русском, в третьем лице.\n"
    'Верни JSON: {"summary": "текст саммари", "key_facts": ["факт1", "факт2"]}'
)


async def consolidate_tier(
    provider, owner_id, contact_id=None, from_tier=1, to_tier=2
) -> int:
    """Сжимает факты из from_tier в to_tier. Возвращает количество сжатых фактов."""
    from src.db.repo import list_memories as _list_memories

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await _list_memories(session, owner, contact_id=contact_id)

        # Фильтруем активные факты нужного тира
        targets = [m for m in memories if m.memory_tier == from_tier and m.is_active]
        if len(targets) < 5:
            return 0  # недостаточно для консолидации

        # Группируем по cluster_topic если есть, иначе все вместе
        import json as _json, re as _re

        groups = {}
        for m in targets:
            topic = m.cluster_topic or "general"
            groups.setdefault(topic, []).append(m)

        consolidated = 0
        for topic, group in groups.items():
            if len(group) < 5:
                continue

            facts_text = "\n".join(f"- {m.fact}" for m in group)

            try:
                raw = await provider.chat(
                    [
                        ChatMessage(role="system", content=CONSOLIDATION_PROMPT),
                        ChatMessage(
                            role="user",
                            content=f"Сожми эти факты:\n{facts_text[:4000]}",
                        ),
                    ],
                    heavy=False,
                )

                raw = raw.strip()
                raw = _re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = _re.sub(r"\n?\s*```\s*$", "", raw)
                result = _json.loads(raw)
                summary = result.get("summary", "")
                key_facts = result.get("key_facts", [])

                if summary:
                    # Сохраняем консолидированный факт как tier=to_tier
                    await add_memory(
                        session,
                        owner,
                        fact=summary,
                        contact_id=contact_id,
                        sentiment="neutral",
                        source="consolidation",
                        importance=0.7,
                        decay_rate=0.03,
                        memory_tier=to_tier,
                    )

                    # Сохраняем кластер
                    await upsert_memory_cluster(
                        session,
                        owner,
                        topic=topic,
                        summary=summary,
                        fact_count=len(group),
                    )

                    # Деактивируем исходные факты
                    for m in group:
                        m.is_active = False
                        m.validity_end = datetime.now(timezone.utc)

                    consolidated += len(group)
            except Exception:
                logger.exception("Consolidation LLM failed for topic %s", topic)

        if consolidated > 0:
            await session.commit()
            logger.info(
                "Consolidated %d facts from tier %d to tier %d",
                consolidated,
                from_tier,
                to_tier,
            )

        return consolidated
