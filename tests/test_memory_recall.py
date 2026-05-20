"""Тесты для unified MemoryRecallService."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from src.db.session import get_session
from src.db.repo import get_or_create_user, add_memory, add_commitment
from src.core.memory.memory_recall import recall, format_recall_for_prompt


def utc_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_pinned_above_normal():
    """Pinned факты всегда первые, даже с низким confidence."""
    async with get_session() as session:
        owner = await get_or_create_user(session, 123456)
        # pinned с низким confidence
        await add_memory(
            session, owner, fact="закреплённый факт", pinned=True, confidence=0.2
        )
        # обычный с высоким confidence
        await add_memory(
            session, owner, fact="обычный важный факт", pinned=False, confidence=0.95
        )
        await session.commit()

    result = await recall(123456, limit=5)
    assert len(result.facts) >= 2
    assert result.facts[0].fact == "закреплённый факт"
    assert "📌" in result.facts[0].reason


@pytest.mark.asyncio
async def test_task_priority():
    """Факты с memory_type=task и активным commitment попадают в результат."""
    async with get_session() as session:
        owner = await get_or_create_user(session, 123457)
        mem = await add_memory(session, owner, fact="сделать отчёт", memory_type="task")
        await session.flush()
        await add_commitment(
            session,
            user_id=owner.id,
            peer_id=0,
            peer_name=None,
            message_id=None,
            direction="mine",
            text="сделать отчёт",
            deadline_at=None,
            source_memory_id=mem.id,
        )
        await session.commit()

    result = await recall(123457, include_tasks=True, limit=5)
    assert any("📋" in f.reason for f in result.facts), (
        "task-факт должен быть с reason «активная задача»"
    )


@pytest.mark.asyncio
async def test_expires_at_excludes():
    """Истёкшие факты не попадают в recall."""
    async with get_session() as session:
        owner = await get_or_create_user(session, 123458)
        past = utc_naive() - timedelta(hours=1)
        await add_memory(session, owner, fact="просроченный факт", expires_at=past)
        await add_memory(session, owner, fact="живой факт")
        await session.commit()

    result = await recall(123458, limit=5)
    facts_text = [f.fact for f in result.facts]
    assert "просроченный факт" not in facts_text
    assert "живой факт" in facts_text


@pytest.mark.asyncio
async def test_use_count_increments():
    """use_count растёт после каждого recall."""
    async with get_session() as session:
        owner = await get_or_create_user(session, 123459)
        await add_memory(session, owner, fact="тестовый факт", confidence=0.8)
        await session.commit()

    # первый вызов
    r1 = await recall(123459, limit=5)
    assert len(r1.facts) >= 1
    mid = r1.facts[0].memory_id

    # проверяем use_count после первого вызова
    async with get_session() as session:
        from src.db.models import Memory
        from sqlalchemy import select

        m = (await session.execute(select(Memory).where(Memory.id == mid))).scalar_one()
        assert m.use_count >= 1

    # второй вызов
    r2 = await recall(123459, limit=5)
    async with get_session() as session:
        from src.db.models import Memory
        from sqlalchemy import select

        m = (await session.execute(select(Memory).where(Memory.id == mid))).scalar_one()
        assert m.use_count >= 2


@pytest.mark.asyncio
async def test_self_vs_contact_facts():
    """Self и contact факты корректно разделяются."""
    async with get_session() as session:
        owner = await get_or_create_user(session, 123460)
        await add_memory(session, owner, fact="я люблю кофе", contact_id=None)
        await add_memory(session, owner, fact="Настя любит чай", contact_id=999)
        await session.commit()

    result = await recall(123460, contact_id=999, limit=10)
    facts_text = " ".join(f.fact for f in result.facts)
    reasons = " ".join(f.reason for f in result.facts)
    assert "люблю кофе" in facts_text
    assert "любит чай" in facts_text
    assert "тебе" in reasons or "контакте" in reasons or "свежий" in reasons


@pytest.mark.asyncio
async def test_format_recall_for_prompt():
    """Форматтер выдаёт XML-тег <recall_context>."""
    async with get_session() as session:
        owner = await get_or_create_user(session, 123461)
        await add_memory(session, owner, fact="памятный факт", pinned=True)
        await session.commit()

    result = await recall(123461, limit=5)
    text = format_recall_for_prompt(result)
    assert "<recall_context>" in text
    assert "</recall_context>" in text
    assert "памятный факт" in text


@pytest.mark.asyncio
async def test_no_facts_graceful():
    """Пустая память — не падает, возвращает пустой результат."""
    async with get_session() as session:
        await get_or_create_user(session, 123462)
        await session.commit()

    result = await recall(123462, limit=5)
    assert result.facts == []
    assert result.meta["total_active"] == 0
