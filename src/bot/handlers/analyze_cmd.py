"""Команда /analyze — полный анализ переписок."""

import logging
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.infra.text_sanitizer import sanitize_html
from src.db.session import get_session
from src.db.repo import get_or_create_user, list_contacts
from src.llm.base import TaskType
from src.llm.router import build_provider
from src.core.infra.full_analyzer import (
    run_full_analysis,
    format_analysis_report,
    AnalysisProgress,
)

logger = logging.getLogger(__name__)
router = Router(name="analyze_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


async def _resolve_contact_names(
    contacts, names: list[str], userbot_manager, telegram_id: int
) -> tuple[list[int], list[str]]:
    """Пытается найти контакты по имени — точное совпадение → fuzzy через contact_resolver.

    Returns:
        (resolved_peer_ids, unresolved_names)
    """
    resolved = []
    unresolved = []
    for name in names:
        nl = name.strip().lower()
        found = False
        # 1. Точное совпадение по display_name
        for c in contacts:
            cn = (c.display_name or "").lower()
            if nl == cn or (len(nl) > 2 and nl in cn):
                resolved.append(c.peer_id)
                found = True
                break
        # 2. Fuzzy через contact_resolver
        if not found:
            try:
                from src.core.contacts.contact_resolver import resolve

                client = (
                    userbot_manager.get_client(telegram_id) if userbot_manager else None
                )
                if client:
                    candidates = await resolve(client, name)
                    if candidates:
                        resolved.append(candidates[0].peer_id)
                        found = True
            except Exception:
                logger.debug("contact resolve failed", exc_info=True)
        if not found:
            unresolved.append(name)
    return resolved, unresolved


@router.message(Command("analyze"))
async def cmd_analyze(message: Message, userbot_manager=None):
    """Запуск полного анализа."""
    args = (message.text or "").strip().split()
    folder_filter = args[1:] if len(args) > 1 else []

    await message.answer("🧠 Запускаю полный анализ переписок...")
    status_msg = await message.answer("⏳ Подготовка...")

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner, task_type=TaskType.SUMMARIZE)

        import json

        if owner.settings is None:
            monitored = []
        else:
            monitored = (
                json.loads(owner.settings.monitored_folders)
                if owner.settings.monitored_folders
                else []
            )

        contact_ids_arg = None
        folders_to_analyze = folder_filter if folder_filter else monitored
        if not folders_to_analyze:
            folders_to_analyze = None  # все контакты

        if not provider:
            await status_msg.edit_text(
                "❌ Не удалось создать LLM провайдер. Проверь API ключи."
            )
            return

    # Если есть аргументы — пробуем разрешить как имена контактов
    if folder_filter:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            contacts = await list_contacts(
                session, owner, kinds=("user",), include_bots=False
            )
        resolved_ids, unresolved = await _resolve_contact_names(
            contacts, folder_filter, userbot_manager, message.from_user.id
        )
        if resolved_ids:
            contact_ids_arg = resolved_ids
            folders_to_analyze = None  # не фильтруем по папкам
        if unresolved and not resolved_ids:
            # Ни один аргумент не совпал с именем контакта —
            # продолжаем как обычно, folders_to_analyze уже установлен
            pass
        elif unresolved and resolved_ids:
            await status_msg.edit_text(
                "❌ Не удалось найти контакты: "
                + sanitize_html(", ".join(unresolved))
                + ".\n"
                + "Попробуй /analyze (без аргументов) или укажи папку."
            )
            return

    # Callback для обновления прогресса
    async def update_progress(progress: AnalysisProgress):
        try:
            if progress.phase == "scan":
                await status_msg.edit_text(f"🔍 {progress.message}")
            elif progress.phase == "processing":
                bar = "▓" * progress.current + "░" * (progress.total - progress.current)
                await status_msg.edit_text(
                    f"🔄 [{bar}] {progress.current}/{progress.total}\n"
                    f"📂 {progress.contact_name}"
                )
            elif progress.phase == "done":
                await status_msg.edit_text("✅ Анализ завершён, формирую отчёт...")
        except Exception:
            logger.exception("failed to update analysis progress")

    # Запустить анализ
    try:
        client = (
            userbot_manager.get_client(message.from_user.id)
            if userbot_manager
            else None
        )
        result = await run_full_analysis(
            owner_id=message.from_user.id,
            provider=provider,
            client=client,
            message_limit=500,
            folder_names=folders_to_analyze,
            contact_ids=contact_ids_arg,
            progress_callback=update_progress,
        )

        report = format_analysis_report(result)
        await status_msg.edit_text(report)

    except Exception as e:
        logger.exception("Full analysis failed")
        await status_msg.edit_text(f"❌ Ошибка анализа: {e}")
