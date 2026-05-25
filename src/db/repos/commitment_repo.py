"""Commitment repository — Commitment, PendingAction, PendingQuestion."""

from __future__ import annotations

import logging

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Commitment,
    PendingAction,
    PendingQuestion,
)

logger = logging.getLogger(__name__)


# ─── Pending Questions ─────────────────────────────────────────────────


async def add_pending_question(
    session: AsyncSession, owner_id: int, question: str
) -> None:
    session.add(PendingQuestion(owner_id=owner_id, question=question))
    await session.flush()


async def get_pending_questions(session: AsyncSession, owner_id: int) -> list[str]:
    r = await session.execute(
        select(PendingQuestion.question)
        .where(PendingQuestion.owner_id == owner_id)
        .order_by(PendingQuestion.created_at)
    )
    result = list(r.scalars().all())
    # Delete after reading
    await session.execute(
        delete(PendingQuestion).where(PendingQuestion.owner_id == owner_id)
    )
    await session.flush()
    return result


# ─── Commitments ────────────────────────────────────────────────────────


async def add_commitment(
    session: AsyncSession,
    *,
    user_id: int,
    peer_id: int,
    peer_name: str | None,
    message_id: int | None,
    direction: str,
    text: str,
    deadline_at=None,
    source_memory_id: int | None = None,
) -> Commitment:
    c = Commitment(
        user_id=user_id,
        peer_id=peer_id,
        peer_name=peer_name,
        message_id=message_id,
        direction=direction,
        text=text,
        deadline_at=deadline_at,
        source_memory_id=source_memory_id,
    )
    session.add(c)
    await session.flush()
    return c


async def list_open_commitments(
    session: AsyncSession,
    user,
    *,
    direction: str | None = None,
    peer_id: int | None = None,
) -> list[Commitment]:
    query = select(Commitment).where(
        Commitment.user_id == user.id,
        Commitment.status == "open",
    )
    if direction:
        query = query.where(Commitment.direction == direction)
    if peer_id is not None:
        query = query.where(Commitment.peer_id == peer_id)
    query = query.order_by(
        Commitment.deadline_at.is_(None), Commitment.deadline_at.asc()
    )
    result = await session.execute(query)
    return list(result.scalars().all())


async def update_commitment_status(
    session: AsyncSession, commitment_id: int, status: str
) -> None:
    c = await session.get(Commitment, commitment_id)
    if c is not None:
        c.status = status
        await session.flush()


async def get_commitment(
    session: AsyncSession, commitment_id: int
) -> Commitment | None:
    return await session.get(Commitment, commitment_id)


async def get_commitment_by_source_memory(
    session: AsyncSession, user_id: int, source_memory_id: int
) -> Commitment | None:
    result = await session.execute(
        select(Commitment).where(
            Commitment.user_id == user_id,
            Commitment.source_memory_id == source_memory_id,
        )
    )
    return result.scalar_one_or_none()


# ─── Pending Actions ────────────────────────────────────────────────────


async def create_pending_action(
    session: AsyncSession,
    *,
    user_id: int,
    kind: str,
    payload: str,
) -> PendingAction:
    pa = PendingAction(user_id=user_id, kind=kind, payload=payload)
    session.add(pa)
    await session.flush()
    return pa


async def get_pending_action(
    session: AsyncSession, action_id: int, user
) -> PendingAction | None:
    result = await session.execute(
        select(PendingAction).where(
            PendingAction.id == action_id, PendingAction.user_id == user.id
        )
    )
    return result.scalar_one_or_none()


async def delete_pending_action(session: AsyncSession, action_id: int, user) -> None:
    pa = await session.get(PendingAction, action_id)
    if pa is not None and pa.user_id == user.id:
        await session.delete(pa)
        await session.flush()
