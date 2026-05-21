"""Checkpoint persistence for compressed context state.

Saves offloaded state to the database so subsequent requests
can reuse compressed representations without re-compressing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select

from src.db.models import Base
from sqlalchemy import Column, Integer, BigInteger, Text, DateTime

logger = logging.getLogger(__name__)


# Simple in-memory cache as primary store (DB as optional persistence)
# Format: {user_id: {"messages": [...], "mermaid": str, "refs": dict, "tokens": int, "ts": datetime}}
_checkpoint_cache: dict[int, dict[str, Any]] = {}


async def save_offload_state(
    user_id: int,
    messages: list[dict],
    *,
    mermaid_graph: str | None = None,
    drilldown_refs: dict[str, int] | None = None,
    tokens_saved: int = 0,
) -> None:
    """Save compressed context state for a user.

    Stored in-memory with JSON-serializable format.
    Messages are stored as-is (list of role/content dicts).
    """
    _checkpoint_cache[user_id] = {
        "messages": messages,
        "mermaid_graph": mermaid_graph,
        "drilldown_refs": drilldown_refs or {},
        "tokens_saved": tokens_saved,
        "updated_at": datetime.utcnow(),
    }
    logger.debug(
        "Offload checkpoint saved for user %d (%d msgs, %s mermaid)",
        user_id,
        len(messages),
        "with" if mermaid_graph else "no",
    )


async def load_offload_state(user_id: int) -> dict[str, Any] | None:
    """Load compressed context state for a user.

    Returns None if no checkpoint exists or it's stale (>30 min).
    """
    state = _checkpoint_cache.get(user_id)
    if state is None:
        return None

    # Stale check: 30 minute TTL
    age = (datetime.utcnow() - state["updated_at"]).total_seconds()
    if age > 1800:
        logger.debug("Offload checkpoint expired for user %d (%.0fs old)", user_id, age)
        del _checkpoint_cache[user_id]
        return None

    logger.debug(
        "Offload checkpoint loaded for user %d (%d msgs)",
        user_id,
        len(state["messages"]),
    )
    return state


async def clear_offload_state(user_id: int) -> None:
    """Clear compressed state (e.g., when new messages arrive)."""
    _checkpoint_cache.pop(user_id, None)
    logger.debug("Offload checkpoint cleared for user %d", user_id)


__all__ = [
    "save_offload_state",
    "load_offload_state",
    "clear_offload_state",
]
