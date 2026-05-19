import asyncio
import logging

from src.config import settings
from src.bot.app import run_bot
from src.core.auto_sync import auto_sync_loop
from src.core.digest import digest_scheduler_loop
from src.core.follow_up import follow_up_loop
from src.core.memory_checker import memory_decay_loop
from src.core.news import news_scheduler_loop
from src.core.proactive_briefing import proactive_briefing_loop
from src.core.reminders import reminders_loop
from src.core.sleep_tracker import sleep_tracker_loop
from src.core.smart_digest import smart_digest_loop
from src.core.weekly_summarizer import weekly_summary_loop
from src.db.session import init_db
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)


async def global_style_scheduler_loop(owner_telegram_id: int) -> None:
    """Обновляет глобальный стиль-профиль каждые 12 часов."""
    from src.core.style_profile import update_global_style_profile

    while True:
        try:
            await update_global_style_profile(owner_telegram_id)
        except Exception as e:
            logger.error("Global style update failed: %s", e)
        await asyncio.sleep(12 * 3600)  # 12 hours


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logger.info("Starting TelegramAssistant")

    await init_db()

    userbot_manager = UserbotManager()
    await userbot_manager.restore_all()

    bg_tasks = [
        asyncio.create_task(digest_scheduler_loop(), name="digest-scheduler"),
        asyncio.create_task(reminders_loop(), name="reminders-loop"),
        asyncio.create_task(news_scheduler_loop(), name="news-scheduler"),
        asyncio.create_task(auto_sync_loop(), name="auto-sync"),
        asyncio.create_task(
            memory_decay_loop(settings.owner_telegram_id), name="memory-decay"
        ),
        asyncio.create_task(
            global_style_scheduler_loop(settings.owner_telegram_id), name="global-style"
        ),
        asyncio.create_task(
            smart_digest_loop(settings.owner_telegram_id), name="smart-digest"
        ),
        asyncio.create_task(
            proactive_briefing_loop(settings.owner_telegram_id),
            name="proactive-briefing",
        ),
        asyncio.create_task(
            follow_up_loop(settings.owner_telegram_id), name="follow-up"
        ),
        asyncio.create_task(
            sleep_tracker_loop(settings.owner_telegram_id), name="sleep-tracker"
        ),
        asyncio.create_task(
            weekly_summary_loop(settings.owner_telegram_id), name="weekly-summary"
        ),
    ]

    try:
        await run_bot(userbot_manager)
    finally:
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


def run() -> None:
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested")


if __name__ == "__main__":
    run()
