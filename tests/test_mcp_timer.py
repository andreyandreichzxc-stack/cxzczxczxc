import importlib
import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.core.actions.mcp_timer import _timer_alarm, _timer_cancel


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 31, 23, 59, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_alarm_after_passed_time_rolls_over_month_end() -> None:
    with patch("src.core.actions.mcp_timer.datetime", FixedDateTime):
        result = await _timer_alarm("23:58", "month-end")

    try:
        assert result["ok"] is True
        assert result["will_fire_at"].startswith("2026-02-01T23:58:00")
    finally:
        await _timer_cancel(result["timer_id"])


def test_import_does_not_open_timer_db(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_connect(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    import src.core.actions.mcp_timer as mcp_timer

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    importlib.reload(mcp_timer)
