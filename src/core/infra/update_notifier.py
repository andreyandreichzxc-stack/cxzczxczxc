"""Уведомление владельца бота об обновлениях при старте."""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)

_VERSION_FILE = "data/.version"
_VERSION_CHECK_COOLDOWN = 10  # секунд: даём боту запуститься перед отправкой


async def check_and_notify_update() -> None:
    """Проверяет, была ли новая версия, и уведомляет владельца."""
    # Ждём, чтобы бот успел стартовать и notification_queue был готов
    await _sleep_sync(_VERSION_CHECK_COOLDOWN)

    try:
        current = _read_current_version()
        previous = _read_stored_version()
        if current and current != previous:
            logger.info("Update detected: %s → %s", previous or "none", current)
            changelog = _extract_changelog(previous)
            _save_version(current)
            await _notify_owner(current, changelog)
    except Exception:
        logger.exception("Update notification failed")


def _read_current_version() -> str | None:
    """Читает версию из CHANGELOG.md (первая строка после '## v').
    Возвращает None если файл не найден или версия не определена."""
    try:
        cl = Path("CHANGELOG.md").read_text(encoding="utf-8")
        for line in cl.split("\n"):
            if line.startswith("## v"):
                return line.strip()
    except OSError:
        pass
    return None


def _read_stored_version() -> str | None:
    """Читает сохранённую версию."""
    try:
        return Path(_VERSION_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _save_version(version: str) -> None:
    Path(_VERSION_FILE).write_text(version, encoding="utf-8")


def _extract_changelog(previous_version: str | None) -> str:
    """Извлекает записи ченджлога от previous_version до текущей."""
    try:
        cl = Path("CHANGELOG.md").read_text(encoding="utf-8")
    except OSError:
        return ""

    if not previous_version:
        # Первый запуск — показываем последнюю версию
        for line in cl.split("\n"):
            if line.startswith("## v"):
                return _collect_section(cl, line)
        return ""

    # Ищем записи между previous_version и current
    result: list[str] = []
    collecting = False
    for line in cl.split("\n"):
        if line.startswith("## v"):
            if collecting:
                break
            if line.strip() == previous_version.strip():
                collecting = True
        elif collecting and line.strip():
            result.append(line)
    return "\n".join(result) if result else ""


def _collect_section(cl_text: str, start_line: str) -> str:
    """Собирает все строки секции после start_line."""
    lines: list[str] = []
    found = False
    for line in cl_text.split("\n"):
        if line.strip() == start_line.strip():
            found = True
            lines.append(line)
        elif found and line.startswith("## "):
            break
        elif found and line.startswith("- "):
            lines.append(line)
    return "\n".join(lines) if lines else ""


async def _notify_owner(version: str, changelog: str) -> None:
    """Отправляет уведомление владельцу через notification_queue."""
    try:
        from src.core.scheduling.notification_queue import notification_queue

        await notification_queue.enqueue(
            topic="system",
            text=f"🔄 <b>Бот обновлён!</b>\n{version}\n\n{changelog}",
            priority=2,
        )
    except Exception:
        logger.exception("Failed to enqueue update notification")


async def _sleep_sync(seconds: float) -> None:
    """Асинхронный sleep (совместимость с разными реализациями)."""
    import asyncio

    await asyncio.sleep(seconds)
