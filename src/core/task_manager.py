"""Background task manager with health monitoring and auto-restart."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class RegisteredTask:
    """Metadata for a single registered background task."""

    def __init__(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
        *,
        restart_on_failure: bool = True,
        restart_delay: float = 5.0,
    ) -> None:
        self.name = name
        self.factory = factory
        self.restart_on_failure = restart_on_failure
        self.restart_delay = restart_delay
        self.task: asyncio.Task[None] | None = None
        self.status = TaskStatus.STOPPED
        self.restart_count = 0


class BackgroundTaskManager:
    """Manages background asyncio tasks with health monitoring and auto-restart.

    Usage::

        manager = BackgroundTaskManager()
        manager.register("my-loop", my_loop, restart_on_failure=True, restart_delay=5.0)
        manager.register("my-other", lambda: other_loop(arg))
        manager.start_all()
        # ... later ...
        await manager.stop_all()

    Call ``get_status(name)`` or ``get_all_statuses()`` for health checks.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, RegisteredTask] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
        *,
        restart_on_failure: bool = True,
        restart_delay: float = 5.0,
    ) -> None:
        """Register a background task.

        Args:
            name: Unique task name (used for the asyncio task name).
            factory: A zero-argument callable that returns a coroutine.
                Use ``lambda`` or ``functools.partial`` to pass arguments.
            restart_on_failure: If True, restart the task on unhandled exception.
            restart_delay: Seconds to wait before restarting.
        """
        if name in self._tasks:
            raise ValueError(f"Task '{name}' is already registered")
        self._tasks[name] = RegisteredTask(
            name,
            factory,
            restart_on_failure=restart_on_failure,
            restart_delay=restart_delay,
        )

    def start_all(self) -> None:
        """Launch all registered tasks."""
        for task in self._tasks.values():
            self._start_single(task)

    def _start_single(self, task: RegisteredTask) -> None:
        """Create and start an asyncio Task that wraps *task.factory*."""

        async def wrapper() -> None:
            while True:
                try:
                    task.status = TaskStatus.RUNNING
                    await task.factory()
                except asyncio.CancelledError:
                    task.status = TaskStatus.STOPPED
                    logger.info("Background task '%s' cancelled", task.name)
                    break
                except Exception:  # noqa: BLE001
                    task.status = TaskStatus.FAILED
                    task.restart_count += 1
                    logger.exception(
                        "Background task '%s' failed (restart #%d, delay=%.1fs)",
                        task.name,
                        task.restart_count,
                        task.restart_delay,
                    )
                    if not task.restart_on_failure:
                        logger.info(
                            "Background task '%s' will NOT be restarted",
                            task.name,
                        )
                        break
                    await asyncio.sleep(task.restart_delay)
                else:
                    # Factory coroutine completed without exception.
                    task.status = TaskStatus.STOPPED
                    logger.info("Background task '%s' finished cleanly", task.name)
                    break

        task.task = asyncio.create_task(wrapper(), name=task.name)

    async def stop_all(self) -> None:
        """Cancel all running tasks and wait for completion."""
        for t in self._tasks.values():
            if t.task is not None and not t.task.done():
                t.task.cancel()

        for t in self._tasks.values():
            if t.task is not None and not t.task.done():
                try:
                    await t.task
                except (asyncio.CancelledError, Exception):
                    pass
            t.status = TaskStatus.STOPPED

    def get_status(self, name: str) -> TaskStatus | None:
        """Return the current status of a task, or None if not found."""
        t = self._tasks.get(name)
        if t is None:
            return None
        # Edge-case: task finished outside our wrapper (should not happen).
        if t.task is not None and t.task.done() and t.status is TaskStatus.RUNNING:
            t.status = TaskStatus.FAILED
        return t.status

    def get_all_statuses(self) -> dict[str, TaskStatus]:
        """Return a map of all task names → current status."""
        return {name: self._tasks[name].status for name in self._tasks}
