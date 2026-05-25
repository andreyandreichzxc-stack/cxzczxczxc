"""Contact memory digest — precomputed per-contact summary.

Replaces expensive full recall() calls in auto_reply, /chat, and catchup
with a lightweight cached JSON blob.

Lifetime:
    - Built on first access, cached in-memory (10 min TTL) + DB.
    - Invalidated when new memories are saved for the contact.
    - Falls back to normal recall if the digest is stale or missing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.db.session import get_session

logger = logging.getLogger(__name__)

# In-memory cache: {peer_id: (monotonic_ts, digest_dict)}
_DIGEST_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}
_DIGEST_CACHE_TTL: float = 3600.0  # 1 hour
_DIGEST_CACHE_MAX: int = 500  # LRU eviction threshold
_DIGEST_LOCK = asyncio.Lock()


async def get_contact_digest(owner_telegram_id: int, peer_id: int) -> dict[str, Any]:
    """Return the precomputed digest for a contact, building it if needed.

    The digest is a lightweight JSON dict with:
        - display_name, style, topics, promises, risks, facts, health

    On cache miss: queries DB → if DB cache is fresh → returns it.
    Otherwise builds a new digest from live data and persists it.
    """
    now_mono = asyncio.get_event_loop().time()

    # 1. In-memory cache
    async with _DIGEST_LOCK:
        if peer_id in _DIGEST_CACHE:
            ts, digest = _DIGEST_CACHE[peer_id]
            if now_mono - ts < _DIGEST_CACHE_TTL:
                return digest

    # 2. DB cache → build fresh if needed
    async with get_session() as session:
        from src.db.repo import get_or_create_user, get_contact_profile
        from src.db.models._contacts import Contact
        from sqlalchemy import select

        owner = await get_or_create_user(session, owner_telegram_id)

        # Look up the Contact row
        contact_r = await session.execute(
            select(Contact).where(
                Contact.peer_id == peer_id, Contact.user_id == owner.id
            )
        )
        contact = contact_r.scalar_one_or_none()
        if not contact:
            return _empty_digest()

        # Load ContactProfile
        profile = await get_contact_profile(session, owner, peer_id)

        # 2a. Check DB cache freshness (10 min)
        if (
            profile is not None
            and profile.memory_digest
            and profile.memory_digest_updated_at
        ):
            age = (
                datetime.now(timezone.utc) - profile.memory_digest_updated_at
            ).total_seconds()
            if age < 600:
                try:
                    digest = json.loads(profile.memory_digest)
                    async with _DIGEST_LOCK:
                        if len(_DIGEST_CACHE) >= _DIGEST_CACHE_MAX:
                            oldest = min(_DIGEST_CACHE.items(), key=lambda x: x[1][0])
                            del _DIGEST_CACHE[oldest[0]]
                        _DIGEST_CACHE[peer_id] = (now_mono, digest)
                    return digest
                except json.JSONDecodeError:
                    logger.debug(
                        "Corrupt memory_digest for peer %d, rebuilding", peer_id
                    )

        # 2b. Build fresh digest
        digest = await _build_digest(session, owner, contact, profile)

        # 2c. Persist
        if profile:
            profile.memory_digest = json.dumps(digest, ensure_ascii=False)
            profile.memory_digest_updated_at = datetime.now(timezone.utc)
        else:
            from src.db.repo import upsert_contact_profile

            profile = await upsert_contact_profile(
                session,
                owner,
                contact_id=peer_id,
                memory_digest=json.dumps(digest, ensure_ascii=False),
                memory_digest_updated_at=datetime.now(timezone.utc),
            )
        await session.flush()

        # 2d. Update in-memory cache
        async with _DIGEST_LOCK:
            if len(_DIGEST_CACHE) >= _DIGEST_CACHE_MAX:
                oldest = min(_DIGEST_CACHE.items(), key=lambda x: x[1][0])
                del _DIGEST_CACHE[oldest[0]]
            _DIGEST_CACHE[peer_id] = (now_mono, digest)

        return digest


async def _build_digest(session, owner, contact, profile) -> dict[str, Any]:
    """Build a fresh digest from live data.

    Runs lightweight queries only — no full recall() with BFS.
    """
    digest: dict[str, Any] = {
        "display_name": contact.display_name or str(contact.peer_id),
        "style": {},
        "topics": [],
        "promises": [],
        "risks": [],
        "facts": [],
        "health": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Style ──────────────────────────────────────────────────────
    if profile:
        digest["style"]["closeness"] = profile.closeness
        digest["style"]["archetype"] = contact.archetype
    if contact.style_profile:
        try:
            sp = json.loads(contact.style_profile)
            digest["style"]["directness"] = sp.get("directness", "")
            digest["style"]["tone"] = sp.get("tone", "")
        except json.JSONDecodeError:
            pass

    # ── Top 5 memory facts (light recall) ──────────────────────────
    try:
        from src.core.memory.memory_recall import recall

        result = await recall(
            owner.telegram_id,
            contact_id=contact.peer_id,
            query="",
            limit=5,
            include_self=False,
            include_pinned=True,
            include_tasks=False,
            include_deep=False,
            mode="light",
        )
        digest["facts"] = [
            {
                "fact": f.fact[:200],
                "confidence": round(f.confidence or 0.5, 2),
                "reason": f.reason,
            }
            for f in result.facts[:5]
        ]
    except Exception:
        logger.debug(
            "recall failed during digest build for peer %d",
            contact.peer_id,
            exc_info=True,
        )

    # ── Open promises/commitments ──────────────────────────────────
    try:
        from src.db.repo import list_open_commitments

        commitments = await list_open_commitments(
            session, owner, peer_id=contact.peer_id
        )
        now = datetime.now(timezone.utc)
        for c in commitments[:3]:
            deadline_str = ""
            if c.deadline_at:
                # Normalize both datetimes to naive for delta calculation
                dl = c.deadline_at
                if dl.tzinfo is not None:
                    dl = dl.replace(tzinfo=None)
                cn = now.replace(tzinfo=None)
                delta = dl - cn
                if delta.total_seconds() < 0:
                    deadline_str = f"ПРОСРОЧЕНО ({abs(int(delta.days))}дн)"
                else:
                    deadline_str = f"через {int(delta.days)}дн"
            digest["promises"].append(
                {
                    "text": (c.text or "")[:150],
                    "deadline": deadline_str,
                    "status": c.status,
                }
            )
    except Exception:
        logger.debug(
            "commitments fetch failed during digest build for peer %d",
            contact.peer_id,
            exc_info=True,
        )

    # ── Health score ───────────────────────────────────────────────
    try:
        from src.core.contacts.health_score import get_contact_health

        health = await get_contact_health(owner.telegram_id, contact.peer_id)
        digest["health"] = health
        if health.get("score", 100) < 60:
            digest["risks"].append(
                {
                    "type": "low_health",
                    "score": health["score"],
                    "detail": health.get("status", ""),
                }
            )
    except Exception:
        logger.debug(
            "health_score failed during digest build for peer %d",
            contact.peer_id,
            exc_info=True,
        )

    # ── Last 3 message topics ──────────────────────────────────────
    try:
        from src.db.models._messaging import Message
        from sqlalchemy import select, desc

        msgs_r = await session.execute(
            select(Message.text)
            .where(
                Message.user_id == owner.id,
                Message.peer_id == contact.peer_id,
            )
            .order_by(desc(Message.date))
            .limit(3)
        )
        digest["topics"] = [m[:100] for m in msgs_r.scalars().all() if m]
    except Exception:
        logger.debug(
            "message fetch failed during digest build for peer %d",
            contact.peer_id,
            exc_info=True,
        )

    return digest


def _empty_digest() -> dict[str, Any]:
    """Minimal digest stub for unknown/missing contacts."""
    return {
        "display_name": "?",
        "style": {},
        "topics": [],
        "promises": [],
        "risks": [],
        "facts": [],
        "health": None,
    }


async def invalidate_contact_digest(peer_id: int) -> None:
    """Drop the in-memory cache entry for a contact.

    Called after new memories are saved so the next access rebuilds
    the digest with fresh data.
    """
    async with _DIGEST_LOCK:
        _DIGEST_CACHE.pop(peer_id, None)
    logger.debug("Invalidated memory digest cache for peer %d", peer_id)
