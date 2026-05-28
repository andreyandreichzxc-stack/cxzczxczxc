"""mcp_avito_watch tool — registered via @tool decorator.

Manage Avito listing watches and price alerts.

Actions:
- ``action="list"`` — list all active watches with listing info.
- ``action="add" listing_id=N price_threshold=30000`` — watch an existing listing.
- ``action="remove" watch_id=N`` — delete a watch.
- ``action="pause" watch_id=N`` — pause a watch.
- ``action="resume" watch_id=N`` — resume a watch.
- ``action="alerts" limit=5`` — recent price alerts for watched listings.

Uses SQLAlchemy directly on ``AvitoListing``, ``AvitoWatch``, ``AvitoPriceHistory``.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.core.actions.tool_registry import tool
from src.db.models._avito import AvitoListing, AvitoPriceHistory, AvitoWatch
from src.db.repo import get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)


@tool(
    name="mcp_avito_watch",
    description=(
        "Manage Avito listing watches and price alerts. Supports six actions:\n"
        "- 'list' — return all watches with listing details.\n"
        "- 'add' — create a watch on an existing listing (requires listing_id).\n"
        "- 'remove' — delete a watch by watch_id.\n"
        "- 'pause' — pause a watch (is_active=False).\n"
        "- 'resume' — resume a paused watch (is_active=True).\n"
        "- 'alerts' — show recent price alerts for watched listings."
    ),
    category="productivity",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — 'list', 'add', 'remove', 'pause', 'resume' or 'alerts'",
        "listing_id": "int — AvitoListing id (required for 'add')",
        "watch_id": "int — watch id (required for 'remove', 'pause', 'resume')",
        "price_threshold": "int|None — alert when price drops below (optional, for 'add')",
        "limit": "int — max alerts to return (default 5, used with 'alerts')",
    },
)
async def mcp_avito_watch(
    action: str,
    listing_id: int = 0,
    watch_id: int = 0,
    price_threshold: int | None = None,
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Avito watch management tool.

    Args:
        action: ``"list"``, ``"add"``, ``"remove"``, ``"pause"``, ``"resume"``
            or ``"alerts"``.
        listing_id: Listing DB id (required for ``action="add"``).
        watch_id: Watch id (required for ``action="remove"``, ``"pause"``,
            ``"resume"``).
        price_threshold: Alert threshold price (optional, for ``action="add"``).
        limit: Max alerts to return (default 5, for ``action="alerts"``).

    Keyword Args:
        user: Owner's Telegram ID (int, defaults to 0).

    Returns:
        A dict with result data or an ``"error"`` key on failure.
    """
    # Normalise user
    _user_val = kwargs.get("user", 0)
    if hasattr(_user_val, "telegram_id"):
        telegram_id: int = _user_val.telegram_id
    else:
        telegram_id = int(_user_val)

    try:
        if action == "list":
            return await _list_watches(telegram_id)
        elif action == "add":
            if not listing_id:
                return {"error": "listing_id is required for action='add'"}
            return await _add_watch(telegram_id, listing_id, price_threshold)
        elif action == "remove":
            if not watch_id:
                return {"error": "watch_id is required for action='remove'"}
            return await _remove_watch(telegram_id, watch_id)
        elif action == "pause":
            if not watch_id:
                return {"error": "watch_id is required for action='pause'"}
            return await _set_watch_active(telegram_id, watch_id, False)
        elif action == "resume":
            if not watch_id:
                return {"error": "watch_id is required for action='resume'"}
            return await _set_watch_active(telegram_id, watch_id, True)
        elif action == "alerts":
            limit = max(1, min(limit, 50))
            return await _get_alerts(telegram_id, limit)
        else:
            return {
                "error": f"Unknown action {action!r}. "
                "Valid actions: list, add, remove, pause, resume, alerts",
            }
    except Exception as exc:
        logger.exception("mcp_avito_watch(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _list_watches(telegram_id: int) -> dict[str, Any]:
    """List all watches for the user with listing details."""
    async with get_session() as session:
        user = await get_or_create_user(session, telegram_id)

        stmt = (
            select(AvitoWatch)
            .options(selectinload(AvitoWatch.listing))
            .where(AvitoWatch.user_id == user.id)
            .order_by(AvitoWatch.created_at.desc())
        )
        result = await session.execute(stmt)
        watches = result.scalars().all()

        items = []
        for w in watches:
            listing = w.listing
            items.append(
                {
                    "watch_id": w.id,
                    "listing_id": w.listing_id,
                    "is_active": w.is_active,
                    "price_threshold": w.price_threshold,
                    "created_at": w.created_at.isoformat(),
                    "listing": {
                        "avito_id": listing.avito_id,
                        "title": listing.title,
                        "price": listing.price,
                        "url": listing.url,
                        "city": listing.city,
                        "condition": listing.condition,
                        "is_active": listing.is_active,
                    },
                }
            )

        return {
            "ok": True,
            "watches": items,
            "count": len(items),
        }


async def _add_watch(
    telegram_id: int,
    listing_id: int,
    price_threshold: int | None,
) -> dict[str, Any]:
    """Create a new watch on an existing listing."""
    async with get_session() as session:
        user = await get_or_create_user(session, telegram_id)

        # Verify the listing exists
        stmt = select(AvitoListing).where(
            AvitoListing.id == listing_id,
            AvitoListing.user_id == user.id,
        )
        result = await session.execute(stmt)
        listing = result.scalar_one_or_none()

        if listing is None:
            return {
                "error": f"Listing with id={listing_id} not found",
            }

        # Check for duplicate watch
        stmt = select(AvitoWatch).where(
            AvitoWatch.user_id == user.id,
            AvitoWatch.listing_id == listing_id,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            return {
                "error": f"Already watching listing #{listing_id} "
                f"(watch_id={existing.id})",
            }

        watch = AvitoWatch(
            user_id=user.id,
            listing_id=listing_id,
            price_threshold=price_threshold,
            is_active=True,
        )
        session.add(watch)
        await session.flush()

        return {
            "ok": True,
            "watch_id": watch.id,
            "listing_id": listing_id,
            "listing_title": listing.title,
            "price_threshold": price_threshold,
            "is_active": True,
        }


async def _remove_watch(
    telegram_id: int,
    watch_id: int,
) -> dict[str, Any]:
    """Delete a watch."""
    async with get_session() as session:
        user = await get_or_create_user(session, telegram_id)

        stmt = select(AvitoWatch).where(
            AvitoWatch.id == watch_id,
            AvitoWatch.user_id == user.id,
        )
        result = await session.execute(stmt)
        watch = result.scalar_one_or_none()

        if watch is None:
            return {"error": f"Watch with id={watch_id} not found"}

        await session.delete(watch)

        return {
            "ok": True,
            "watch_id": watch_id,
            "deleted": True,
        }


async def _set_watch_active(
    telegram_id: int,
    watch_id: int,
    is_active: bool,
) -> dict[str, Any]:
    """Pause (is_active=False) or resume (is_active=True) a watch."""
    label = "resume" if is_active else "pause"

    async with get_session() as session:
        user = await get_or_create_user(session, telegram_id)

        stmt = select(AvitoWatch).where(
            AvitoWatch.id == watch_id,
            AvitoWatch.user_id == user.id,
        )
        result = await session.execute(stmt)
        watch = result.scalar_one_or_none()

        if watch is None:
            return {"error": f"Watch with id={watch_id} not found"}

        watch.is_active = is_active

        return {
            "ok": True,
            "watch_id": watch_id,
            "is_active": is_active,
            "action": label,
        }


async def _get_alerts(
    telegram_id: int,
    limit: int,
) -> dict[str, Any]:
    """Show recent price changes for watched listings."""
    async with get_session() as session:
        user = await get_or_create_user(session, telegram_id)

        # Get listing_ids the user watches
        stmt_watches = select(AvitoWatch.listing_id).where(
            AvitoWatch.user_id == user.id,
        )
        result = await session.execute(stmt_watches)
        listing_ids = [row[0] for row in result.all()]

        if not listing_ids:
            return {
                "ok": True,
                "alerts": [],
                "count": 0,
            }

        # Latest price history entries for watched listings
        stmt = (
            select(AvitoPriceHistory)
            .options(selectinload(AvitoPriceHistory.listing))
            .where(AvitoPriceHistory.listing_id.in_(listing_ids))
            .order_by(AvitoPriceHistory.recorded_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        entries = result.scalars().all()

        alerts = []
        for entry in entries:
            listing = entry.listing
            alerts.append(
                {
                    "price": entry.price,
                    "recorded_at": entry.recorded_at.isoformat(),
                    "listing_id": entry.listing_id,
                    "listing_title": listing.title if listing else None,
                    "listing_url": listing.url if listing else None,
                }
            )

        return {
            "ok": True,
            "alerts": alerts,
            "count": len(alerts),
        }
