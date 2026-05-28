"""mcp_timer tool — registered via @tool decorator.

Provides in-memory timer and alarm management:

- **set** — create a timer that fires after a given number of seconds.
- **alarm** — create an alarm that fires at a specific HH:MM time.
- **list** — return all active timers with remaining time.
- **cancel** — cancel a timer by its index from the list output.

On fire: sends a notification to the owner via ``notification_queue``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import ToolActionSpec, tool

logger = logging.getLogger(__name__)

# ── Module-level state ─────────────────────────────────────────────────────

_active_timers: dict[int, dict[str, Any]] = {}
"""Mapping: timer_id -> {task, label, created_at, duration_sec, fire_at}"""
_timer_lock = asyncio.Lock()
_timer_counter = 0
"""Monotonically increasing counter for timer IDs."""

# ── SQLite persistence ─────────────────────────────────────────────────────

_timer_store: dict[int, dict[str, Any]] = {}
"""Persistent timer metadata: timer_id -> {fire_at, label}"""
_timer_db_lock = threading.Lock()
_timer_db: sqlite3.Connection | None = None
_timer_db_loaded = False


def _get_timer_db_path() -> Path:
    """Extract the database file path from ``database_url``."""
    raw = str(settings.database_url)
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if raw.startswith(prefix):
            path_part = raw.removeprefix(prefix)
            if path_part in (":memory:", ""):
                break
            return settings.data_dir / Path(path_part).name
    return settings.data_dir / "app.db"


def _init_timer_db() -> sqlite3.Connection:
    """Create the ``timers`` table if it does not exist and return a connection."""
    db_path = _get_timer_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS timers (
            timer_id INTEGER PRIMARY KEY,
            fire_at TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.commit()
    return conn


def _get_timer_db() -> sqlite3.Connection:
    """Return a lazily initialized SQLite connection for timer persistence."""
    global _timer_db
    if _timer_db is None:
        _timer_db = _init_timer_db()
    return _timer_db


def _load_timers_from_db() -> None:
    """Load pending timers from DB, removing expired ones."""
    global _timer_counter
    now_str = datetime.now(timezone.utc).isoformat()
    with _timer_db_lock:
        db = _get_timer_db()
        db.execute("DELETE FROM timers WHERE fire_at <= ?", (now_str,))
        db.commit()
        rows = db.execute(
            "SELECT timer_id, fire_at, label FROM timers"
        ).fetchall()
        for tid, fire_at, label in rows:
            _timer_store[tid] = {"fire_at": fire_at, "label": label}
        if rows:
            max_tid = max(r[0] for r in rows)
            if max_tid >= _timer_counter:
                _timer_counter = max_tid + 1


def _ensure_timer_db_loaded() -> None:
    """Initialize timer persistence on first runtime use, not module import."""
    global _timer_db_loaded
    if _timer_db_loaded:
        return
    _load_timers_from_db()
    _timer_db_loaded = True


def _persist_timer(tid: int, fire_at: str, label: str) -> None:
    """Insert timer into SQLite and in-memory store."""
    _ensure_timer_db_loaded()
    with _timer_db_lock:
        db = _get_timer_db()
        db.execute(
            "INSERT OR REPLACE INTO timers (timer_id, fire_at, label) VALUES (?, ?, ?)",
            (tid, fire_at, label),
        )
        db.commit()
    _timer_store[tid] = {"fire_at": fire_at, "label": label}


def _remove_timer(tid: int) -> None:
    """Delete timer from SQLite and in-memory store."""
    _timer_store.pop(tid, None)
    _ensure_timer_db_loaded()
    with _timer_db_lock:
        db = _get_timer_db()
        db.execute("DELETE FROM timers WHERE timer_id = ?", (tid,))
        db.commit()


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_timer
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_timer",
    description=(
        "Manage in-memory timers and alarms.  Supports four actions:\n"
        "- 'set' — set a timer for a given duration in seconds.\n"
        "- 'alarm' — set an alarm for a specific HH:MM time (24h).\n"
        "- 'list' — list all active timers with remaining time.\n"
        "- 'cancel' — cancel a timer by its index from the list.\n"
        "Timers fire once and send an OS notification."
    ),
    category="utility",
    risk="medium",
    requires_confirmation=True,
    actions={
        "set": ToolActionSpec(
            name="set",
            risk="medium",
            read_only=False,
            destructive=False,
            idempotent=False,
            requires_confirmation=True,
            user_content=True,
        ),
        "alarm": ToolActionSpec(
            name="alarm",
            risk="medium",
            read_only=False,
            destructive=False,
            idempotent=False,
            requires_confirmation=True,
            user_content=True,
        ),
        "list": ToolActionSpec(
            name="list",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
        "cancel": ToolActionSpec(
            name="cancel",
            risk="high",
            read_only=False,
            destructive=True,
            idempotent=False,
            requires_confirmation=True,
            user_content=False,
        ),
    },
    params={
        "action": "str — 'set', 'alarm', 'list', or 'cancel'",
        "duration_sec": "int — timer duration in seconds (required for 'set')",
        "label": "str — optional label for the timer/alarm",
        "time": "str — alarm time in HH:MM 24h format (required for 'alarm')",
        "index": "int — timer index from list output (required for 'cancel')",
    },
)
async def mcp_timer(
    action: str,
    duration_sec: int = 0,
    label: str = "",
    time: str = "",
    index: int = -1,
    **kwargs: Any,
) -> dict[str, Any]:
    """In-memory timer and alarm management tool.

    Args:
        action: ``"set"``, ``"alarm"``, ``"list"``, or ``"cancel"``.
        duration_sec: Duration in seconds (required for ``action="set"``).
        label: Optional human-readable label for the timer/alarm.
        time: Time in HH:MM 24h format (required for ``action="alarm"``).
        index: Timer index from ``"list"`` output (required for ``action="cancel"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "set":
            if not bool(kwargs.get("_confirmed", False)):
                return {"error": "requires confirmation"}
            return await _timer_set(duration_sec, label)
        elif action == "alarm":
            if not bool(kwargs.get("_confirmed", False)):
                return {"error": "requires confirmation"}
            return await _timer_alarm(time, label)
        elif action == "list":
            return _timer_list()
        elif action == "cancel":
            if not bool(kwargs.get("_confirmed", False)):
                return {"error": "requires confirmation"}
            return await _timer_cancel(index)
        else:
            return {
                "error": (
                    f"Unknown action {action!r}. "
                    f"Valid actions: set, alarm, list, cancel"
                )
            }
    except Exception as exc:
        logger.exception("mcp_timer(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _timer_set(duration_sec: int, label: str) -> dict[str, Any]:
    """Create a countdown timer."""
    if duration_sec <= 0:
        return {"error": "duration_sec must be a positive integer"}

    _ensure_timer_db_loaded()

    fire_at = datetime.now(timezone.utc).timestamp() + duration_sec
    fire_dt = datetime.fromtimestamp(fire_at, tz=timezone.utc)
    lbl = label.strip() if label else f"Timer ({duration_sec}s)"

    async with _timer_lock:
        global _timer_counter  # noqa: PLW0603
        _timer_counter += 1
        tid = _timer_counter

        task = asyncio.create_task(
            _timer_task(tid, duration_sec, lbl),
            name=f"timer-{tid}",
        )

        _active_timers[tid] = {
            "task": task,
            "label": lbl,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_sec": duration_sec,
            "fire_at": fire_dt.isoformat(),
        }
        _persist_timer(tid, fire_dt.isoformat(), lbl)

    return {
        "ok": True,
        "timer_id": tid,
        "label": lbl,
        "will_fire_at": fire_dt.isoformat(),
        "duration_sec": duration_sec,
    }


async def _timer_alarm(time_str: str, label: str) -> dict[str, Any]:
    """Create an alarm for a specific HH:MM time."""
    if not time_str or ":" not in time_str:
        return {"error": "time must be in HH:MM format (e.g. '14:30')"}

    try:
        parts = time_str.strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (ValueError, IndexError):
        return {"error": "time must be in HH:MM 24h format (e.g. '14:30')"}

    _ensure_timer_db_loaded()

    now = datetime.now(timezone.utc)
    fire_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # If the time has already passed today, schedule for tomorrow
    if fire_dt <= now:
        fire_dt += timedelta(days=1)

    duration_sec = int((fire_dt - now).total_seconds())
    lbl = label.strip() if label else f"Alarm at {time_str}"

    async with _timer_lock:
        global _timer_counter  # noqa: PLW0603
        _timer_counter += 1
        tid = _timer_counter

        task = asyncio.create_task(
            _timer_task(tid, duration_sec, lbl),
            name=f"alarm-{tid}",
        )

        _active_timers[tid] = {
            "task": task,
            "label": lbl,
            "created_at": now.isoformat(),
            "duration_sec": duration_sec,
            "fire_at": fire_dt.isoformat(),
        }
        _persist_timer(tid, fire_dt.isoformat(), lbl)

    return {
        "ok": True,
        "timer_id": tid,
        "label": lbl,
        "will_fire_at": fire_dt.isoformat(),
        "duration_sec": duration_sec,
    }


def _timer_list() -> dict[str, Any]:
    """Return all active timers with remaining time."""
    now_ts = datetime.now(timezone.utc).timestamp()
    items: list[dict[str, Any]] = []

    # Work on a snapshot outside the lock to avoid blocking
    for tid, info in list(_active_timers.items()):
        task = info["task"]
        if task.done():
            continue

        fire_ts = datetime.fromisoformat(info["fire_at"]).timestamp()
        remaining = max(0, int(fire_ts - now_ts))

        items.append(
            {
                "index": tid,
                "label": info["label"],
                "remaining_sec": remaining,
                "fire_at": info["fire_at"],
            }
        )

    return {
        "ok": True,
        "timers": items,
        "count": len(items),
    }


async def _timer_cancel(index: int) -> dict[str, Any]:
    """Cancel a timer by its ID/index."""
    if index < 0:
        return {
            "error": "index is required for action='cancel' (use numeric index from list)"
        }

    async with _timer_lock:
        info = _active_timers.pop(index, None)
        _remove_timer(index)

    if info is None:
        return {"error": f"Timer with index {index} not found"}

    task = info["task"]
    if not task.done():
        task.cancel()

    return {
        "ok": True,
        "cancelled": True,
        "label": info["label"],
        "timer_id": index,
    }


# ══════════════════════════════════════════════════════════════════════════
# Timer task
# ══════════════════════════════════════════════════════════════════════════


async def _timer_task(tid: int, duration_sec: int, label: str) -> None:
    """Sleep for *duration_sec*, then fire the timer."""
    try:
        await asyncio.sleep(duration_sec)
    except asyncio.CancelledError:
        # Timer was cancelled — clean up
        async with _timer_lock:
            _active_timers.pop(tid, None)
        return

    # Timer fired — remove from active list and send notification
    async with _timer_lock:
        _active_timers.pop(tid, None)
    _remove_timer(tid)

    try:
        from src.core.scheduling.notification_queue import notification_queue

        await notification_queue.enqueue(
            topic="timer",
            text=f"⏰ {label}",
            priority=50,  # HIGH
            category="timer",
        )
    except Exception:
        logger.exception("Failed to enqueue timer notification for %r", label)
