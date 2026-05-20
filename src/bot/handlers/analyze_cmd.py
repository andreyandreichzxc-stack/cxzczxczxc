"""Команда /analyze — полный анализ переписок."""

import logging
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery

from src.bot.filters import OwnerOnly
from src.db.session import get_session
from src.db.repo import get_or_create_user
from src.llm.router import build_provider
from src.core.full_analyzer import (
    run_full_analysis,
    format_analysis_report,
    AnalysisProgress,
)

logger = logging.getLogger(__name__)
router = Router(name="analyze_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


@router.message(Command("analyze"))
async def cmd_analyze(message: Message, userbot_manager=None):
    """Запуск полного анализа."""
    args = (message.text or "").strip().split()
    folder_filter = args[1:] if len(args) > 1 else []

    await message.answer("🧠 Запускаю полный анализ переписок...")
    status_msg = await message.answer("⏳ Подготовка...")

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)

        import json

        monitored = (
            json.loads(owner.settings.monitored_folders)
            if owner.settings.monitored_folders
            else []
        )
        folders_to_analyze = folder_filter if folder_filter else monitored
        if not folders_to_analyze:
            folders_to_analyze = None  # все контакты

        if not provider:
            await status_msg.edit_text(
                "❌ Не удалось создать LLM провайдер. Проверь API ключи."
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
        result = await run_full_analysis(
            owner_id=message.from_user.id,
            provider=provider,
            message_limit=500,
            folder_names=folders_to_analyze,
            progress_callback=update_progress,
        )

        report = format_analysis_report(result)
        await status_msg.edit_text(report)

    except Exception as e:
        logger.exception("Full analysis failed")
        await status_msg.edit_text(f"❌ Ошибка анализа: {e}")
