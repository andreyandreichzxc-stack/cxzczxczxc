import asyncio
import logging

from src.bot.app import run_bot
from src.bot.handlers.free_text import start_voice_worker, stop_voice_worker
from src.core.memory.memory_queue import start_worker, stop_worker
from src.core.scheduling.notification_queue import notification_queue
from src.core.infra.task_manager import task_manager, stop_ff_tasks
from src.core.infra.update_notifier import check_and_notify_update
from src.db.session import init_db
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)


def _register_background_tasks() -> None:
    """Импортирует модули с background-задачами — декораторы авторегистрируют их."""
    # noqa — импорты триггерят @task_manager.task() декораторы
    import src.core.infra.system_tasks  # noqa: F401
    import src.core.scheduling.digest  # noqa: F401
    import src.core.scheduling.reminders  # noqa: F401
    import src.core.scheduling.news  # noqa: F401
    import src.core.infra.auto_sync  # noqa: F401
    import src.core.memory.memory_checker  # noqa: F401
    import src.core.scheduling.smart_digest  # noqa: F401
    import src.core.scheduling.proactive_briefing  # noqa: F401
    import src.core.scheduling.follow_up  # noqa: F401
    import src.core.scheduling.sleep_tracker  # noqa: F401
    import src.core.scheduling.weekly_summarizer  # noqa: F401
    import src.core.scheduling.weekly_digest  # noqa: F401
    import src.core.memory.memory_patterns  # noqa: F401
    import src.core.memory.knowledge_distiller  # noqa: F401
    import src.core.memory.temporal_layers  # noqa: F401
    import src.core.actions.conflict_resolver  # noqa: F401
    import src.core.actions.conflict_predictor  # noqa: F401
    import src.core.scheduling.habit_tracker  # noqa: F401
    import src.core.memory.memory_clusterer  # noqa: F401
    import src.core.intelligence.skills  # noqa: F401
    import src.core.intelligence.skills_curator  # noqa: F401


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logger.info("Starting TelegramAssistant")

    await init_db()

    from src.core.memory.context_files import index_contexts_to_fts, init_owner_context

    init_owner_context()
    index_contexts_to_fts()

    try:
        from src.core.infra.hooks import hooks

        await hooks.emit("on_startup")
    except Exception:
        pass  # hooks are optional, never break core flow

    await start_worker()
    start_voice_worker()
    notification_queue.start()

    # Уведомить владельца об обновлении (фоном, ждёт 10с чтобы бот стартовал)
    asyncio.create_task(check_and_notify_update())

    from src.core.actions.vector_store import get_vector_store

    await get_vector_store().check_health_and_recover()

    userbot_manager = UserbotManager()
    await userbot_manager.restore_all()

    _register_background_tasks()
    task_manager.start_all()

    # Phase 2: регистрация MCP-инструментов в tool_registry
    import src.core.actions.mcp_tools  # noqa: F401
    import src.core.actions.cross_search_tool  # noqa: F401

    try:
        await run_bot(userbot_manager)
    finally:
        logger.info("Shutting down…")
        for step, coro in [
            ("userbot", userbot_manager.shutdown()),
            ("background tasks", task_manager.stop_all()),
            ("memory worker", stop_worker()),
            ("voice worker", stop_voice_worker()),
            ("notification queue", notification_queue.stop()),
        ]:
            try:
                logger.debug("Stopping %s…", step)
                await asyncio.wait_for(coro, timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("%s shutdown timed out — forcing", step)
            except Exception:
                logger.exception("%s shutdown failed", step)

        # Give fire-and-forget tasks (fact saves, trajectory, inbox) a chance
        # to finish so in-flight DB writes are not lost.
        try:
            await asyncio.wait_for(stop_ff_tasks(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("fire-and-forget tasks shutdown timed out")
        except Exception:
            logger.exception("fire-and-forget tasks shutdown failed")

        try:
            from src.core.actions.vector_store import get_vector_store

            await asyncio.wait_for(get_vector_store().shutdown(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("vector_store shutdown timed out")
        except Exception:
            logger.exception("vector_store shutdown failed")

        try:
            from src.core.infra.hooks import hooks

            await hooks.emit("on_shutdown")
        except Exception:
            pass  # hooks are optional, never break core flow

        logger.info("Shutdown complete")


def run() -> None:
    # Run Alembic migrations synchronously before the event loop starts.
    # This avoids asyncio.run() nesting inside the existing event loop.
    import alembic.command
    import alembic.config

    _cfg = alembic.config.Config("alembic.ini")
    alembic.command.upgrade(_cfg, "head")

    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested")
    except Exception:
        logger.exception("Unhandled error in main")
        try:
            from src.core.infra.hooks import hooks

            asyncio.run(
                hooks.emit(
                    "on_error", error="Unhandled error in main", context="main.run"
                )
            )
        except Exception:
            pass  # hooks are optional, never break core flow
        raise


if __name__ == "__main__":
    run()
