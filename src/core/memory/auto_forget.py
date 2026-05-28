"""Auto-forget: deactivate memories with retention below Ebbinghaus threshold."""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.memory.temporal_layers import compute_retention
from src.db.models._memory import Memory

logger = logging.getLogger(__name__)


async def auto_forget_sweep(session: AsyncSession, user_id: int) -> int:
    """Find and deactivate memories with retention < auto_forget_threshold.

    Returns: number of deactivated facts.
    """
    if not settings.auto_forget_enabled:
        return 0

    threshold = settings.auto_forget_threshold
    now = datetime.now(timezone.utc)

    # Load active, non-pinned, non-task memories for user
    result = await session.execute(
        select(Memory).where(
            Memory.user_id == user_id,
            Memory.is_active == True,
            Memory.pinned == False,
            Memory.memory_type != "task",
        )
    )
    memories = list(result.scalars().all())

    to_deactivate: list[int] = []
    for m in memories:
        retention = compute_retention(
            m,
            now,
            decay_base=settings.ebbinghaus_decay_base,
            access_weight=settings.ebbinghaus_access_weight,
        )
        if retention < threshold:
            to_deactivate.append(m.id)

    if not to_deactivate:
        return 0

    # Bulk deactivate
    await session.execute(
        sa_update(Memory)
        .where(Memory.id.in_(to_deactivate))
        .values(
            is_active=False,
            validity_end=now,
        )
    )

    logger.info(
        "Auto-forget: deactivated %d facts for user %d (threshold=%.2f)",
        len(to_deactivate),
        user_id,
        threshold,
    )
    return len(to_deactivate)
