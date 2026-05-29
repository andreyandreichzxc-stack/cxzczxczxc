"""Часовой пояс пользователя. В БД храним naive UTC, при показе и сравнении
с расписаниями (digest_time, news_digest_time) переводим в TZ владельца."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# валидация HH:MM
HM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

# популярные пресеты для быстрых кнопок
TZ_PRESETS: list[str] = [
    "UTC",
    "Europe/Moscow",
    "Europe/Kiev",
    "Europe/Warsaw",
    "Europe/London",
    "Europe/Berlin",
    "Asia/Almaty",
    "Asia/Tbilisi",
    "Asia/Dubai",
    "Asia/Tokyo",
    "America/New_York",
    "America/Los_Angeles",
]


def get_user_tz(user) -> str:
    """Get user's timezone string with UTC fallback."""
    tz = user.settings.timezone if user.settings else None
    return tz or "UTC"


def parse_tz(name: str | None) -> ZoneInfo:
    if not name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def is_valid_tz(name: str) -> bool:
    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False


def now_in_tz(tz_name: str | None) -> datetime:
    return datetime.now(parse_tz(tz_name))


def utc_to_local(dt: datetime, tz_name: str | None) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(parse_tz(tz_name))


def fmt_local(
    dt: datetime | None, tz_name: str | None, *, fmt: str = "%Y-%m-%d %H:%M"
) -> str:
    if dt is None:
        return "—"
    local = utc_to_local(dt, tz_name)
    return local.strftime(fmt)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Приводит naive datetime к UTC-aware.

    SQLite с DateTime(timezone=True) возвращает aware datetime для новых записей,
    но старые записи без TZ в ISO-строке приходят как naive.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def tz_short(tz_name: str | None) -> str:
    tz = parse_tz(tz_name)
    offset = datetime.now(tz).utcoffset()
    if offset is None:
        return tz.key
    total_min = int(offset.total_seconds() // 60)
    sign = "+" if total_min >= 0 else "-"
    h, m = divmod(abs(total_min), 60)
    suffix = f"UTC{sign}{h}" + (f":{m:02d}" if m else "")
    return f"{tz.key} ({suffix})"
