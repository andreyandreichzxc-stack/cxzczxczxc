"""Weekly summarizer — раз в неделю саммари переписки с каждым активным контактом."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from src.core.contacts.chat_service import message_to_text
from src.core.scheduling.notification_queue import notification_queue
from src.db.models import Notification
from src.core.infra.timeutil import now_in_tz
from src.db.repo import (
    add_memory,
    fetch_chat_messages,
    get_or_create_user,
    list_active_conversations,
    list_contacts,
    list_open_commitments,
    upsert_memory_cluster,
)
from src.db.session import get_session
from src.llm.base import ChatMessage
from src.config import settings
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
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            days=since_days
        )
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
                    if owner_safe.settings.monitored_folders:
                        try:
                            monitored = json.loads(
                                owner_safe.settings.monitored_folders
                            )
                        except json.JSONDecodeError:
                            monitored = []
                    else:
                        monitored = []
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
                    sentiment_counts: dict[str, int] = {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                    }
                    semaphore = asyncio.Semaphore(3)

                    async def _summarize_one(contact):
                        async with semaphore:
                            return contact, await summarize_contact_week(
                                provider, owner_id, contact
                            )

                    contact_tasks = [_summarize_one(c) for c in contacts[:20]]
                    for coro in asyncio.as_completed(contact_tasks):
                        contact, facts = await coro
                        for f in facts:
                            await add_memory(
                                session,
                                owner_safe,
                                fact=f.get("fact", ""),
                                contact_id=contact.peer_id,
                                sentiment=f.get("sentiment"),
                                source="weekly",
                            )
                            sentiment = f.get("sentiment")
                            if sentiment in sentiment_counts:
                                sentiment_counts[sentiment] += 1
                            total_facts += 1
                    if total_facts > 0:
                        lines = [
                            f"📊📝 <b>Недельное саммари:</b> {total_facts} фактов "
                            f"из {len(contacts[:20])} контактов сохранено в память."
                        ]

                        # Section 1: Open commitments approaching deadline
                        commits = await list_open_commitments(session, owner_safe)
                        if commits:
                            lines.append("")
                            lines.append("📋 Обязательства на неделе:")
                            for c in commits[:5]:
                                deadline_str = (
                                    f" ({c.deadline_at.strftime('%d.%m')})"
                                    if c.deadline_at
                                    else ""
                                )
                                lines.append(f"  • {c.text}{deadline_str}")

                        # Section 2: People to reply to
                        conv_states = await list_active_conversations(
                            session, owner_safe, limit=50
                        )
                        cutoff_24h = datetime.now(timezone.utc).replace(
                            tzinfo=None
                        ) - timedelta(hours=24)
                        contact_map = {c.peer_id: c.display_name for c in contacts}
                        unreplied = []
                        for cs in conv_states:
                            if cs.last_incoming_at is None:
                                continue
                            if (
                                cs.last_outgoing_at is not None
                                and cs.last_outgoing_at > cs.last_incoming_at
                            ):
                                continue
                            last_incoming = (
                                cs.last_incoming_at.replace(tzinfo=None)
                                if cs.last_incoming_at.tzinfo
                                else cs.last_incoming_at
                            )
                            if last_incoming > cutoff_24h:
                                continue
                            name = contact_map.get(cs.peer_id, f"peer#{cs.peer_id}")
                            unreplied.append(name)
                        if unreplied:
                            lines.append("")
                            lines.append("👤 Стоит ответить:")
                            for name in unreplied[:3]:
                                lines.append(f"  • {name}")

                        # Section 3: Emotional summary
                        lines.append("")
                        lines.append("🎭 Настроение недели:")
                        total_s = sum(sentiment_counts.values())
                        if total_s > 0:
                            pos_ratio = sentiment_counts["positive"] / total_s * 100
                            neg_ratio = sentiment_counts["negative"] / total_s * 100
                            if pos_ratio > 60:
                                tone = "Неделя прошла позитивно 😊"
                            elif neg_ratio > 40:
                                tone = "Было много сложных разговоров 😐"
                            elif pos_ratio > neg_ratio:
                                tone = "В целом хорошая неделя 🙂"
                            else:
                                tone = "Смешанные эмоции, были и хорошие, и трудные моменты 🤔"
                        else:
                            tone = "Недостаточно данных для анализа настроения"
                        lines.append(f"  {tone}")

                        text = "\n".join(lines)
                        await notification_queue.enqueue(
                            topic="weekly_summary",
                            text=text,
                            priority=Notification.PRIORITY_MEDIUM,
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
            await asyncio.sleep(settings.weekly_summary_check_sec)  # проверка раз в час
        except Exception:
            logger.exception("Weekly summary error")
            await asyncio.sleep(settings.weekly_summary_check_sec)


from functools import partial
from src.core.infra.task_manager import task_manager

task_manager.register(
    "weekly-summary", partial(weekly_summary_loop, settings.owner_telegram_id)
)


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
        import json as _json
        import re as _re

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
                _key_facts = result.get("key_facts", [])

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

                    # НЕ деактивируем факты — повышаем tier и помечаем как сконсолидированные
                    for m in group:
                        m.memory_tier = to_tier
                        m.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        # Добавляем 'consolidated' в comma-separated tags
                        current_tags = [
                            t.strip()
                            for t in (m.tags or "").replace(",", "|").split("|")
                            if t.strip()
                        ]
                        if "consolidated" not in current_tags:
                            current_tags.append("consolidated")
                        m.tags = "|".join(current_tags)

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
