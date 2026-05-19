"""
Full Analyzer — пакетный анализатор переписок.
Извлекает память, обязательства, напоминания из последних N сообщений
всех контактов из выбранных папок.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.models import User, Contact
    from src.llm.base import LLMProvider
    from src.db.models import Message

logger = logging.getLogger(__name__)


@dataclass
class AnalysisProgress:
    """Прогресс анализа — передаётся в callback для UI."""

    phase: str = "init"
    current: int = 0
    total: int = 0
    contact_name: str = ""
    message: str = ""


@dataclass
class AnalysisResult:
    """Результат полного анализа."""

    contacts_processed: int = 0
    messages_scanned: int = 0
    memories_found: int = 0
    commitments_found: int = 0
    contradictions_found: int = 0
    errors: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


async def run_full_analysis(
    owner_id: int,
    provider,
    *,
    message_limit: int = 500,
    folder_names: list[str] | None = None,
    progress_callback=None,
) -> AnalysisResult:
    """
    Полный анализ всех контактов из выбранных папок.

    Args:
        owner_id: Telegram ID владельца
        provider: LLMProvider для извлечения фактов
        message_limit: сколько последних сообщений анализировать на контакт
        folder_names: список папок для анализа (None = все)
        progress_callback: async callable(AnalysisProgress) для UI-обновлений
    """
    result = AnalysisResult()

    from src.db.session import get_session
    from src.db.repo import (
        get_or_create_user,
        list_contacts,
        add_memory,
        find_similar_memories,
    )
    from src.core.memory_extractor import extract_and_save_memories
    from src.core.commitment_extractor import extract_and_save_commitments

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        # Получить контакты
        contacts = await list_contacts(
            session,
            owner,
            kinds=("user",),
            include_bots=False,
        )

    # Фильтр по папкам (fuzzy matching, ~25% tolerance)
    if folder_names:
        from rapidfuzz import fuzz

        FUZZY_THRESHOLD = 70  # ~25% допустимых ошибок
        filtered = []
        for c in contacts:
            cf = (c.folder_names or "").split(",")
            cf = [f.strip().lower() for f in cf if f.strip()]
            if not cf:
                continue
            for user_folder in folder_names:
                user_lower = user_folder.strip().lower()
                best = max(fuzz.ratio(user_lower, f) for f in cf)
                if best >= FUZZY_THRESHOLD:
                    filtered.append(c)
                    break
        contacts = filtered

    total = len(contacts)
    if total == 0:
        result.details.append("Нет контактов для анализа.")
        return result

    if progress_callback:
        await progress_callback(
            AnalysisProgress(
                phase="scan",
                total=total,
                message=f"Найдено {total} контактов",
            ),
        )

    # Обработка каждого контакта
    for idx, contact in enumerate(contacts):
        contact_name = contact.display_name or str(contact.peer_id)

        if progress_callback:
            await progress_callback(
                AnalysisProgress(
                    phase="processing",
                    current=idx + 1,
                    total=total,
                    contact_name=contact_name,
                    message=f"Анализ {contact_name}...",
                ),
            )

        try:
            # Загрузить сообщения
            async with get_session() as session:
                from src.db.repo import fetch_chat_messages

                messages = await fetch_chat_messages(
                    session,
                    owner,
                    contact.peer_id,
                    limit=message_limit,
                )

            if not messages:
                result.details.append(f"{contact_name}: нет сообщений")
                continue

            result.contacts_processed += 1
            result.messages_scanned += len(messages)

            # Извлечь память (extract_and_save_memories открывает свою сессию)
            try:
                mem_count = await extract_and_save_memories(
                    provider,
                    owner_id,
                    contact,
                    messages,
                )
                result.memories_found += mem_count
                if mem_count > 0:
                    result.details.append(
                        f"{contact_name}: +{mem_count} фактов в память",
                    )
            except Exception as e:
                logger.exception("Memory extraction failed for %s", contact_name)
                result.errors.append(f"Память {contact_name}: {e}")

            # Извлечь обязательства (keyword-only аргументы)
            try:
                async with get_session() as session:
                    owner_obj = await get_or_create_user(session, owner_id)
                    saved = await extract_and_save_commitments(
                        provider,
                        user_id=owner_obj.id,
                        contact=contact,
                        messages=messages,
                    )
                    commit_count = len(saved)
                    result.commitments_found += commit_count
                    if commit_count > 0:
                        result.details.append(
                            f"{contact_name}: +{commit_count} обязательств",
                        )
            except Exception as e:
                logger.exception(
                    "Commitment extraction failed for %s",
                    contact_name,
                )
                result.errors.append(f"Обязательства {contact_name}: {e}")

            # Искать противоречия с существующей памятью
            try:
                async with get_session() as session:
                    owner_obj = await get_or_create_user(session, owner_id)
                    contradictions = 0
                    recent_memories = await list_memories(
                        session,
                        owner_obj,
                        contact_id=contact.peer_id,
                    )
                    for mem in recent_memories[-10:]:  # последние 10
                        similar = await find_similar_memories(
                            session,
                            owner_obj,
                            mem.fact,
                        )
                        for sm in similar:
                            if (
                                sm.id != mem.id
                                and sm.sentiment
                                and mem.sentiment
                                and sm.sentiment != mem.sentiment
                            ):
                                # Если факты об одном и том же, но с разной
                                # тональностью — отметим противоречие
                                sm.sentiment = "contradictory"
                                await session.flush()
                                contradictions += 1
                    if contradictions > 0:
                        result.contradictions_found += contradictions
                        result.details.append(
                            f"{contact_name}: {contradictions} противоречий",
                        )
            except Exception as e:
                logger.exception(
                    "Contradiction check failed for %s",
                    contact_name,
                )
                result.errors.append(f"Противоречия {contact_name}: {e}")

            # Небольшая задержка чтобы не заспамить LLM API
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.exception("Analysis failed for %s", contact_name)
            result.errors.append(f"{contact_name}: {e}")

    if progress_callback:
        await progress_callback(
            AnalysisProgress(
                phase="done",
                total=total,
                message="Анализ завершён",
            ),
        )

    return result


async def list_memories(session, user, *, contact_id=None):
    """Локальная обёртка — импортирует repo.list_memories."""
    from src.db.repo import list_memories as _list_memories

    return await _list_memories(session, user, contact_id=contact_id)


def format_analysis_report(result: AnalysisResult) -> str:
    """Формирует красивый HTML-отчёт."""
    lines = [
        "🧠 <b>Полный анализ завершён</b>",
        "",
        f"👥 Контактов: <b>{result.contacts_processed}</b>",
        f"💬 Сообщений: <b>{result.messages_scanned}</b>",
        f"🧩 Фактов в память: <b>{result.memories_found}</b>",
        f"📝 Обязательств: <b>{result.commitments_found}</b>",
        f"⚠️ Противоречий: <b>{result.contradictions_found}</b>",
    ]

    if result.details:
        lines.append("")
        lines.append("<b>Детали:</b>")
        for d in result.details[:20]:
            lines.append(f"  • {d}")
        if len(result.details) > 20:
            lines.append(f"  ... и ещё {len(result.details) - 20}")

    if result.errors:
        lines.append("")
        lines.append(f"<b>Ошибки ({len(result.errors)}):</b>")
        for e in result.errors[:5]:
            lines.append(f"  ❌ {e}")

    return "\n".join(lines)
