"""DSM — Design-Structured Memory. Cross-session project memory."""

from __future__ import annotations
import asyncio
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from src.core.memory.context_files import _get_db_path

logger = logging.getLogger(__name__)

_DSM_CACHE: list[dict] = []  # in-memory cache of recent entries
_DSM_CACHE_TTL: float = 300  # 5 min
_DSM_CACHE_TS: float = 0


def _get_dsm_db() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dsm_entries (
            key TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            importance REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_dsm_tags ON dsm_entries(tags)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_dsm_created ON dsm_entries(created_at)")
    conn.commit()
    return conn


async def dsm_write(
    key: str, content: str, *, tags: str = "", source: str = "", importance: float = 0.5
) -> bool:
    """Write a fact/decision to DSM. Deduplicates by key (overwrites if exists)."""
    try:
        conn = await asyncio.to_thread(_get_dsm_db)
        now = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(
            lambda: conn.execute(
                "INSERT OR REPLACE INTO dsm_entries(key, content, tags, source, importance, created_at, accessed_at) "
                "VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM dsm_entries WHERE key=?), ?), ?)",
                (key, content[:2000], tags, source, importance, key, now, now),
            )
        )
        await asyncio.to_thread(conn.commit)
        logger.debug("DSM write: %s", key)
        return True
    except Exception:
        logger.debug("DSM write failed: %s", key, exc_info=True)
        return False


async def dsm_search(query: str, limit: int = 5) -> list[dict]:
    """Search DSM via substring match + ranking. Returns [{key, content, tags, importance}]."""
    try:
        conn = await asyncio.to_thread(_get_dsm_db)
        rows = await asyncio.to_thread(
            lambda: conn.execute(
                "SELECT key, content, tags, importance, created_at "
                "FROM dsm_entries "
                "WHERE content LIKE ? OR tags LIKE ? OR key LIKE ? "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        )
        return [
            {
                "key": r[0],
                "content": r[1],
                "tags": r[2],
                "importance": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
    except Exception:
        logger.debug("DSM search failed", exc_info=True)
        return []


async def dsm_get_recent(days: int = 7, limit: int = 10) -> list[dict]:
    """Load recent DSM entries for session start injection."""
    global _DSM_CACHE, _DSM_CACHE_TS
    now_ts = asyncio.get_event_loop().time()
    if _DSM_CACHE and (now_ts - _DSM_CACHE_TS) < _DSM_CACHE_TTL:
        return _DSM_CACHE[:limit]
    try:
        conn = await asyncio.to_thread(_get_dsm_db)
        rows = await asyncio.to_thread(
            lambda: conn.execute(
                "SELECT key, content, tags, importance FROM dsm_entries "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (limit * 2,),
            ).fetchall()
        )
        _DSM_CACHE = [
            {"key": r[0], "content": r[1], "tags": r[2], "importance": r[3]}
            for r in rows
        ]
        _DSM_CACHE_TS = now_ts
        return _DSM_CACHE[:limit]
    except Exception:
        logger.debug("DSM get_recent failed", exc_info=True)
        return []


async def dsm_list_tags() -> list[str]:
    """All unique tags."""
    try:
        conn = await asyncio.to_thread(_get_dsm_db)
        rows = await asyncio.to_thread(
            lambda: conn.execute(
                "SELECT DISTINCT tags FROM dsm_entries WHERE tags != ''"
            ).fetchall()
        )
        tags = set()
        for r in rows:
            for t in r[0].split(","):
                t = t.strip()
                if t:
                    tags.add(t)
        return sorted(tags)
    except Exception:
        return []


async def dsm_cleanup(days: int = 30) -> int:
    """Delete entries older than N days. Returns count."""
    try:
        conn = await asyncio.to_thread(_get_dsm_db)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        res = await asyncio.to_thread(
            lambda: conn.execute(
                "DELETE FROM dsm_entries WHERE created_at < ?", (cutoff,)
            )
        )
        await asyncio.to_thread(conn.commit)
        return res.rowcount if res else 0
    except Exception:
        return 0
