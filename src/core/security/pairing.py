"""Pairing manager — approve contacts before auto-reply."""

from __future__ import annotations

import logging
import secrets
import threading

from src.db.session import get_session

logger = logging.getLogger(__name__)


class PairingManager:
    """Security layer: unknown contacts must be approved before interaction."""

    def __init__(self) -> None:
        self._pending: dict[int, str] = {}  # sender_id → code
        self._allowlist: set[int] = set()
        self._lock = threading.Lock()

    async def is_allowed(self, sender_id: int) -> bool:
        """Check in-memory first, then DB fallback."""
        with self._lock:
            if sender_id in self._allowlist:
                return True
        # DB check
        try:
            async with get_session() as session:
                from src.db.repo import is_contact_allowed

                allowed = await is_contact_allowed(session, sender_id)
                if allowed:
                    # Cache in memory for speed
                    with self._lock:
                        self._allowlist.add(sender_id)
                return allowed
        except Exception:
            return False

    def is_pending(self, sender_id: int) -> bool:
        with self._lock:
            return sender_id in self._pending

    def start_pairing(self, sender_id: int) -> str:
        """Generate a pairing code for a new contact."""
        code = secrets.token_hex(3)  # 6-char hex, e.g. "a1b2c3"
        with self._lock:
            self._pending[sender_id] = code
        logger.info("Pairing started for sender %d (code: %s)", sender_id, code)
        return code

    async def approve(self, sender_id: int, code: str) -> bool:
        """Approve a pending contact and persist to DB."""
        with self._lock:
            if sender_id in self._pending and self._pending[sender_id] == code:
                self._allowlist.add(sender_id)
                del self._pending[sender_id]
                logger.info("Pairing approved: sender %d", sender_id)
                approved = True
            else:
                approved = False
        if approved:
            # Persist to DB
            try:
                async with get_session() as session:
                    from src.db.repo import add_allowed_contact

                    await add_allowed_contact(session, sender_id)
            except Exception:
                logger.exception("Failed to persist pairing")
            return True
        return False

    async def revoke(self, sender_id: int) -> None:
        """Remove from allowlist (in-memory + DB)."""
        with self._lock:
            self._allowlist.discard(sender_id)
            self._pending.pop(sender_id, None)
        try:
            async with get_session() as session:
                from src.db.repo import remove_allowed_contact

                await remove_allowed_contact(session, sender_id)
        except Exception:
            logger.exception("Failed to remove allowed contact from DB")

    @property
    def allowlist_size(self) -> int:
        with self._lock:
            return len(self._allowlist)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


# Module-level singleton
pairing = PairingManager()
