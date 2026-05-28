"""Cookie persistence via SQLite."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


class CookieStore:
    """Persist cookies across sessions (mimics real browser)."""

    def __init__(self) -> None:
        self._path: Path = settings.data_dir / "avito_cookies.db"

    def save(self, domain: str, cookies: dict[str, str]) -> None:
        """Save cookies for a domain."""
        conn = sqlite3.connect(str(self._path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cookies "
                "(domain TEXT PRIMARY KEY, data TEXT, updated_at TEXT)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO cookies VALUES (?, ?, ?)",
                (domain, json.dumps(cookies), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            logger.debug("Saved %d cookies for domain %s", len(cookies), domain)
        finally:
            conn.close()

    def load(self, domain: str) -> dict[str, str]:
        """Load cookies for a domain."""
        conn = sqlite3.connect(str(self._path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cookies "
                "(domain TEXT PRIMARY KEY, data TEXT, updated_at TEXT)"
            )
            row = conn.execute(
                "SELECT data FROM cookies WHERE domain = ?", (domain,)
            ).fetchone()
            return json.loads(row[0]) if row else {}
        finally:
            conn.close()

    def clear(self, domain: str) -> None:
        """Clear cookies for a domain."""
        conn = sqlite3.connect(str(self._path))
        try:
            conn.execute("DELETE FROM cookies WHERE domain = ?", (domain,))
            conn.commit()
            logger.debug("Cleared cookies for domain %s", domain)
        finally:
            conn.close()
