"""Smart memory extraction after sync — verifies dates, avoids confusion, deduplicates.

After sync completes, this module:
1. Extracts facts about the OWNER from their own messages (contact_id=NULL)
2. Extracts facts about each CONTACT from conversations (contact_id=peer_id)
3. Verifies dates — skips facts with future dates or dates >1 year old
4. Deduplicates — uses text similarity against existing memories
5. Shows progress per contact
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from src.core.contacts.chat_service import messages_to_transcript
from src.core.memory.memory_extractor import MEMORIES_SYSTEM, _parse_json_array
from src.core.memory.memory_queue import MemoryJob, enqueue
from src.db.repo import fetch_chat_messages, get_or_create_user, list_memories
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

# Regex для поиска дат в тексте факта
_DATE_PATTERNS = [
    re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})"),  # DD.MM.YYYY  DD/MM/YY
    re.compile(r"(\d{4})[-](\d{1,2})[-](\d{1,2})"),  # YYYY-MM-DD
    re.compile(
        r"(\d{1,2})\s+(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+(\d{4})",
        re.IGNORECASE,
    ),  # "15 мая 2024"
    re.compile(r"(\d{4})\s+года?"),  # "2024 года" — год в конце
]

_MONTH_MAP = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

# Порог схожести для дедупликации
_DEDUP_SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


async def smart_extract_after_sync(
    owner_id: int,
    provider: LLMProvider,
    contact_ids: list[int],
    progress_callback=None,
    progress_message=None,
) -> dict:
    """Запускает smart-извлечение памяти после синхронизации.

    Args:
        owner_id: Telegram ID владельца.
        provider: LLM-провайдер.
        contact_ids: список peer_id контактов для анализа.
        progress_callback: async (idx, total, name, status, extra) -> None.
            status: 'pending' | 'processing' | 'done' | 'skip'
            extra: str — например "+3 факта" для done.
        progress_message: aiogram Message для progress_tracker (per‑contact).

    Returns:
        {"owner_facts": N, "contact_facts": M, "skipped_stale": K}
    """
    total_owner_facts = 0
    total_contact_facts = 0
    total_skipped = 0

    total = len(contact_ids)

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)

    # Pre-build display-name map for progress_tracker (avoids DB calls in item_name_fn)
    _name_map: dict[int, str] = {}
    if progress_message and total > 0:
        from src.db.repo import get_contact

        for pid in contact_ids:
            async with get_session() as session:
                c = await get_contact(session, owner, pid)
            _name_map[pid] = c.display_name if c else str(pid)

    # Выбираем источник контактов: с прогрессом или без
    if progress_message and total > 0:
        from src.core.infra.progress import progress_tracker

        contact_iter = progress_tracker(
            progress_message,
            total,
            contact_ids,
            item_name_fn=lambda pid: _name_map.get(pid, str(pid)),
            prefix="🧠 Smart‑память",
        )
    else:
        # Вспомогательный async‑генератор
        async def _pid_iter():
            for pid in contact_ids:
                yield pid

        contact_iter = _pid_iter()

    _contact_idx = 0
    async for peer_id in contact_iter:
        _contact_idx += 1

        # Получаем объект контакта
        from src.db.repo import get_contact

        async with get_session() as session:
            contact = await get_contact(session, owner, peer_id)

        contact_name = contact.display_name if contact else str(peer_id)

        # --- progress: processing ---
        if progress_callback:
            await progress_callback(
                _contact_idx - 1, total, contact_name, "processing", ""
            )

        # Загружаем сообщения из БД (уже синхронизированы)
        async with get_session() as session:
            messages = await fetch_chat_messages(session, owner, peer_id, limit=80)

        if not messages:
            if progress_callback:
                await progress_callback(
                    _contact_idx - 1, total, contact_name, "skip", "нет сообщений"
                )
            continue

        transcript = messages_to_transcript(messages)

        # --- 1. Извлекаем факты о ВЛАДЕЛЬЦЕ (contact=None) ---
        owner_facts, skipped = await _extract_llm_filtered(
            provider,
            owner_id,
            contact=None,
            transcript=transcript,
        )
        total_skipped += skipped
        if owner_facts:
            await _save_facts_to_queue(owner_id, contact_id=None, facts=owner_facts)
            total_owner_facts += len(owner_facts)

        # --- 2. Извлекаем факты о КОНТАКТЕ ---
        contact_facts, skipped = await _extract_llm_filtered(
            provider,
            owner_id,
            contact=contact,
            transcript=transcript,
        )
        total_skipped += skipped
        if contact_facts:
            contact_peer_id = contact.peer_id if contact else None
            await _save_facts_to_queue(
                owner_id, contact_id=contact_peer_id, facts=contact_facts
            )
            total_contact_facts += len(contact_facts)

        # --- progress: done ---
        extra_parts = []
        if owner_facts:
            extra_parts.append(f"+{len(owner_facts)} о себе")
        if contact_facts:
            extra_parts.append(f"+{len(contact_facts)} о контакте")
        extra = ", ".join(extra_parts) if extra_parts else "0 фактов"
        if progress_callback:
            await progress_callback(
                _contact_idx - 1, total, contact_name, "done", extra
            )

    return {
        "owner_facts": total_owner_facts,
        "contact_facts": total_contact_facts,
        "skipped_stale": total_skipped,
    }


# ---------------------------------------------------------------------------
# Внутренние helpers
# ---------------------------------------------------------------------------


async def _extract_llm_filtered(
    provider: LLMProvider,
    telegram_id: int,
    contact,
    transcript: str,
) -> tuple[list[dict], int]:
    """Вызвать LLM, распарсить факты, отфильтровать по датам и дедуплицировать.

    Returns:
        (valid_facts, skipped_count)
    """
    if not transcript:
        return [], 0

    # Формируем промпт (как в memory_extractor.py)
    if contact is not None:
        user_prompt = (
            f"Собеседник: {contact.display_name}.\n"
            "Извлеки важные факты о собеседнике из этой переписки:\n\n"
            f"{transcript}"
        )
    else:
        user_prompt = (
            "Извлеки важные факты о пользователе (его предпочтения, личные данные, задачи) "
            "из этой переписки:\n\n"
            f"{transcript}"
        )

    try:
        raw = await provider.chat(
            [
                ChatMessage(role="system", content=MEMORIES_SYSTEM),
                ChatMessage(role="user", content=user_prompt),
            ],
            heavy=False,
        )
    except (ConnectionError, OSError, ValueError):
        logger.exception("Smart memory LLM call failed")
        return [], 0

    items = _parse_json_array(raw)
    if not items:
        return [], 0

    # Фильтруем и валидируем
    contact_id = contact.peer_id if contact else None
    valid: list[dict] = []
    skipped = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        fact = (item.get("fact") or "").strip()
        if not fact:
            continue

        # --- Проверка дат ---
        if _has_invalid_date(fact):
            skipped += 1
            logger.debug("Skipped fact with invalid date: %r", fact[:80])
            continue

        # --- Дедупликация ---
        if await _is_duplicate(telegram_id, contact_id, fact):
            skipped += 1
            logger.debug("Skipped duplicate fact: %r", fact[:80])
            continue

        # Валидация sentiment
        sentiment = item.get("sentiment")
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = None

        # importance 1-10 → 0.0-1.0
        raw_importance = item.get("importance")
        if isinstance(raw_importance, (int, float)):
            importance = max(0.0, min(1.0, raw_importance / 10.0))
        else:
            importance = None

        decay_rate = item.get("decay_rate")
        if not isinstance(decay_rate, (int, float)):
            decay_rate = None

        memory_type = item.get("memory_type")
        VALID_MEMORY_TYPES = {
            "personal",
            "contact_fact",
            "relationship",
            "task",
            "preference",
            "temporary",
        }
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = None

        valid.append(
            {
                "fact": fact,
                "sentiment": sentiment,
                "source": "chat",
                "importance": importance,
                "decay_rate": decay_rate,
                "memory_type": memory_type,
                "relation_type": item.get("relation_type"),
                "relation_to_index": item.get("relation_to_index"),
            }
        )

    return valid, skipped


def _has_invalid_date(fact_text: str) -> bool:
    """Проверяет, есть ли в тексте факта невалидная дата.

    Считается невалидной:
    - дата в будущем
    - дата старше 1 года (устаревший факт)

    Факты без дат считаются валидными.
    """
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)
    future_cutoff = now + timedelta(days=1)  # +1 день допуска (часовые пояса)

    found_dates = _extract_dates(fact_text)
    if not found_dates:
        return False  # нет дат — всё ок

    for d in found_dates:
        if d > future_cutoff:
            logger.debug("Future date %s in fact: %r", d.date(), fact_text[:60])
            return True  # будущая дата
        if d < one_year_ago:
            logger.debug("Stale date %s in fact: %r", d.date(), fact_text[:60])
            return True  # старше года

    return False


def _extract_dates(text: str) -> list[datetime]:
    """Извлекает даты из текста факта. Возвращает список datetime (UTC)."""
    now = datetime.now(timezone.utc)
    found: list[datetime] = []

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                dt = _parse_date_match(match)
                if dt is not None:
                    found.append(dt)
            except (ValueError, IndexError):
                continue

    return found


def _parse_date_match(match: re.Match) -> datetime | None:
    """Парсит одну regex-группу в datetime."""
    groups = match.groups()
    group_count = len(groups)
    group0_len = len(groups[0]) if group_count >= 1 else 0

    # DD.MM.YYYY или DD/MM/YY — первая группа короткая (1-2 цифры)
    if group_count == 3 and group0_len <= 2 and groups[1].isdigit():
        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        if year < 100:
            year += 2000
        if 1 <= day <= 31 and 1 <= month <= 12 and year >= 2000:
            return datetime(year, month, day, tzinfo=timezone.utc)

    # YYYY-MM-DD — первая группа 4 цифры, вторая/третья тоже цифры
    if (
        group_count == 3
        and group0_len == 4
        and groups[1].isdigit()
        and groups[2].isdigit()
    ):
        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
        if 1 <= month <= 12 and 1 <= day <= 31:
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Русская дата: "15 мая 2024" — вторая группа содержит буквы
    if group_count == 3 and not groups[1].isdigit():
        day = int(groups[0])
        month_name = groups[1].lower()
        year = int(groups[2])
        month = _MONTH_MAP.get(month_name)
        if month and 1 <= day <= 31 and year >= 2000:
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Просто год: "2024 года" — одна группа
    if group_count == 1:
        year = int(groups[0])
        if year >= 2000:
            return datetime(year, 1, 1, tzinfo=timezone.utc)

    return None


async def _is_duplicate(
    telegram_id: int,
    contact_id: int | None,
    fact_text: str,
) -> bool:
    """Проверяет, есть ли похожий факт в БД."""
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        existing = await list_memories(session, owner, contact_id=contact_id)

    fact_lower = fact_text.lower().strip()
    for mem in existing:
        if not mem.is_active or not mem.fact:
            continue
        existing_lower = mem.fact.lower().strip()
        # Точное совпадение
        if fact_lower == existing_lower:
            return True
        # Частичное совпадение — короткие факты строже
        min_len = min(len(fact_lower), len(existing_lower))
        if min_len < 10:
            if fact_lower == existing_lower:
                return True
            continue
        ratio = SequenceMatcher(None, fact_lower, existing_lower).ratio()
        if ratio >= _DEDUP_SIMILARITY_THRESHOLD:
            logger.debug(
                "Duplicate detected (ratio=%.3f): %r vs %r",
                ratio,
                fact_text[:60],
                mem.fact[:60],
            )
            return True

    return False


async def _save_facts_to_queue(
    telegram_id: int,
    contact_id: int | None,
    facts: list[dict],
) -> None:
    """Сохраняет факты через очередь (memory_queue)."""
    if not facts:
        return

    # Embedding batch
    from src.llm.router import build_provider

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        provider = await build_provider(session, owner)

    if provider:
        texts = [f["fact"] for f in facts]
        try:
            embeddings = await provider.embed_batch(texts)
        except Exception:
            logger.warning("Failed to embed batch of %d facts", len(texts))
            embeddings = [None] * len(texts)
        for idx, vf in enumerate(facts):
            if idx < len(embeddings) and embeddings[idx] is not None:
                vf["embedding"] = embeddings[idx]

    await enqueue(
        MemoryJob(
            telegram_id=telegram_id,
            contact_id=contact_id,
            facts=facts,
            job_type="save",
        )
    )
    logger.info(
        "Smart memory: enqueued %d facts for user %d, contact %s",
        len(facts),
        telegram_id,
        contact_id,
    )
