import asyncio
import logging
import time

from src.config import settings
from src.bot.app import run_bot
from src.core.infra.auto_sync import auto_sync_loop
from src.core.scheduling.digest import digest_scheduler_loop
from src.core.scheduling.follow_up import follow_up_loop
from src.core.memory.memory_checker import memory_decay_loop
from src.core.memory.memory_queue import start_worker, stop_worker
from src.core.scheduling.notification_queue import notification_queue
from src.core.memory.temporal_layers import temporal_migration_loop
from src.core.memory.memory_patterns import patterns_loop
from src.core.scheduling.news import news_scheduler_loop
from src.core.scheduling.proactive_briefing import proactive_briefing_loop
from src.core.scheduling.reminders import reminders_loop
from src.core.scheduling.sleep_tracker import sleep_tracker_loop
from src.core.scheduling.smart_digest import smart_digest_loop
from src.core.scheduling.weekly_summarizer import weekly_summary_loop
from src.core.scheduling.weekly_digest import weekly_digest_loop
from src.core.memory.knowledge_distiller import distillation_loop
from src.core.actions.conflict_predictor import conflict_predictor_loop
from src.core.actions.conflict_resolver import conflict_check_loop
from src.core.scheduling.habit_tracker import habit_tracker_loop
from src.core.memory.memory_clusterer import cluster_loop
from src.core.intelligence.skills import skill_optimizer_loop
from src.core.intelligence.instruction_optimizer import instruction_optimizer
from src.core.infra.task_manager import BackgroundTaskManager
from src.db.session import init_db
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)

task_manager = BackgroundTaskManager()


async def global_style_scheduler_loop(owner_telegram_id: int) -> None:
    """Обновляет глобальный стиль-профиль каждые 12 часов."""
    from src.core.contacts.style_profile import update_global_style_profile

    while True:
        try:
            await update_global_style_profile(owner_telegram_id)
        except Exception:
            logger.exception("Global style update failed")
        await asyncio.sleep(settings.global_style_interval_sec)  # 12 hours


async def instruction_optimizer_scheduler_loop(owner_telegram_id: int) -> None:
    """Runs instruction optimization daily even if the optimizer implementation is one-shot."""
    while True:
        try:
            await instruction_optimizer.instruction_optimizer_loop(owner_telegram_id)
        except Exception:
            logger.exception("Instruction optimizer failed")
        await asyncio.sleep(settings.instruction_optimizer_interval_sec)


async def _disk_monitor_loop(owner_id: int) -> None:
    """Мониторинг свободного места на диске."""
    import shutil
    from src.core.scheduling.notification_queue import notification_queue

    last_warning_at = 0.0

    while True:
        try:
            usage = shutil.disk_usage(settings.data_dir)
            free_mb = usage.free / (1024 * 1024)

            if free_mb < settings.disk_critical_mb:
                await notification_queue.enqueue(
                    topic="disk_monitor",
                    text=f"⛔ КРИТИЧНО: свободно {free_mb:.0f} MB на диске!",
                    priority=0,
                )
            elif free_mb < settings.disk_warning_mb:
                now = time.monotonic()
                if now - last_warning_at > 3600:
                    await notification_queue.enqueue(
                        topic="disk_monitor",
                        text=f"⚠️ Мало места на диске: {free_mb:.0f} MB свободно.",
                        priority=1,
                    )
                    last_warning_at = now
        except Exception:
            logger.exception("disk monitor error")

        await asyncio.sleep(settings.disk_monitor_interval_sec)


def _register_background_tasks() -> None:
    """Register all background tasks into the global task_manager."""
    oid = settings.owner_telegram_id

    task_manager.register("digest-scheduler", digest_scheduler_loop)
    task_manager.register("reminders-loop", reminders_loop)
    task_manager.register("news-scheduler", news_scheduler_loop)
    task_manager.register("auto-sync", auto_sync_loop)
    task_manager.register("memory-decay", lambda: memory_decay_loop(oid))
    task_manager.register("global-style", lambda: global_style_scheduler_loop(oid))
    task_manager.register("smart-digest", lambda: smart_digest_loop(oid))
    task_manager.register("proactive-briefing", lambda: proactive_briefing_loop(oid))
    task_manager.register("follow-up", lambda: follow_up_loop(oid))
    task_manager.register("sleep-tracker", lambda: sleep_tracker_loop(oid))
    task_manager.register("weekly-summary", lambda: weekly_summary_loop(oid))
    task_manager.register("weekly-digest", lambda: weekly_digest_loop(oid))
    task_manager.register("memory-patterns", lambda: patterns_loop(oid))
    task_manager.register("distillation", lambda: distillation_loop(oid))
    task_manager.register("temporal-migration", lambda: temporal_migration_loop(oid))
    task_manager.register("conflict-check", lambda: conflict_check_loop(oid))
    task_manager.register("conflict-predictor", lambda: conflict_predictor_loop(oid))
    task_manager.register("habit-tracker", lambda: habit_tracker_loop(oid))
    task_manager.register("memory-cluster", lambda: cluster_loop(oid))
    task_manager.register(
        "instruction-optimizer", lambda: instruction_optimizer_scheduler_loop(oid)
    )
    task_manager.register("skill-optimizer", lambda: skill_optimizer_loop(oid))
    task_manager.register("disk-monitor", lambda: _disk_monitor_loop(oid))
    task_manager.register(
        "media_sweep",
        lambda oid=oid: _media_sweep_loop(oid),
    )


async def _media_sweep_loop(owner_id: int):
    """Периодическая очистка осиротевших медиа-файлов."""
    from src.core.contacts.chat_service import sweep_orphaned_media
    from src.db.models import Notification

    await asyncio.sleep(60)  # первый запуск через минуту после старта
    while True:
        try:
            deleted = await sweep_orphaned_media()
            if deleted:
                from src.core.scheduling.notification_queue import notification_queue

                await notification_queue.enqueue(
                    topic="media_sweep",
                    text=f"🧹 Очищено {deleted} временных медиа-файлов.",
                    priority=Notification.PRIORITY_LOW,
                )
        except Exception:
            logger.exception("media sweep loop error")
        await asyncio.sleep(6 * 3600)  # раз в 6 часов


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logger.info("Starting TelegramAssistant")

    await init_db()

    start_worker()

    from src.core.actions.vector_store import vector_store

    await vector_store.check_health_and_recover()

    userbot_manager = UserbotManager()
    await userbot_manager.restore_all()

    _register_background_tasks()
    task_manager.start_all()

    notification_queue.start()

    try:
        await run_bot(userbot_manager)
    finally:
        await task_manager.stop_all()
        await stop_worker()
        await notification_queue.stop()

        from src.core.actions.vector_store import vector_store

        await vector_store.shutdown()


def run() -> None:
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested")


if __name__ == "__main__":
    run()
