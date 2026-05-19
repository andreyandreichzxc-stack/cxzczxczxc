"""Smoke-тесты memory-системы TelegramHelper."""

import asyncio
import os
import sys
import pytest
from datetime import datetime, timezone, timedelta

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Переопределяем DATABASE_URL на in-memory ДО импорта src-модулей
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["ENCRYPTION_KEY"] = "HmsOzSAxuyfb7zet2nmwhFkgWfH5z6Lsr3tW7MO8GDI="
os.environ["BOT_TOKEN"] = "test:token"
os.environ["OWNER_TELEGRAM_ID"] = "123456789"

from src.db.session import init_db, get_session
from src.db.models import Memory, MemoryCandidate, Commitment
from src.db.repo import (
    add_memory,
    add_memory_candidate,
    delete_memory_candidate,
    list_memory_candidates,
    get_or_create_user,
    add_commitment,
    update_commitment_status,
    list_memories,
)
from src.core.temporal_layers import get_prompt_facts

OWNER_TG_ID = 123456789


@pytest.fixture(autouse=True)
def setup_db():
    """Пересоздаёт таблицы перед каждым тестом (чтобы не копились данные)."""
    from src.db.session import engine, Base

    async def _recreate():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await init_db()

    asyncio.run(_recreate())


async def _get_owner():
    async with get_session() as session:
        return await get_or_create_user(session, OWNER_TG_ID)


@pytest.mark.asyncio
async def test_init_db_creates_memory_tables():
    """init_db() создаёт все memory-поля и таблицы."""
    async with get_session() as session:
        from sqlalchemy import text

        result = await session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%memor%'"
            )
        )
        tables = [r[0] for r in result.all()]
        assert "memories" in tables
        assert "memories_fts" in tables
        assert "memory_candidates" in tables
        assert "memory_links" in tables
        assert "memory_clusters" in tables


@pytest.mark.asyncio
async def test_memory_candidate_confirm():
    """MemoryCandidate → confirm → Memory — факт переносится."""
    owner = await _get_owner()
    async with get_session() as session:
        c = await add_memory_candidate(
            session, owner, fact="Тестовый факт", sentiment="positive", source="chat"
        )
        cand_id = c.id

    # Confirm
    async with get_session() as session:
        owner2 = await get_or_create_user(session, OWNER_TG_ID)
        cand = await session.get(MemoryCandidate, cand_id)
        assert cand is not None
        await add_memory(
            session,
            owner2,
            fact=cand.fact,
            contact_id=cand.contact_id,
            sentiment=cand.sentiment,
            source=cand.source,
            importance=cand.importance,
            decay_rate=cand.decay_rate,
        )
        await session.delete(cand)

    # Проверка
    async with get_session() as session:
        owner3 = await get_or_create_user(session, OWNER_TG_ID)
        mems = await list_memories(session, owner3)
        assert len(mems) == 1
        assert mems[0].fact == "Тестовый факт"


@pytest.mark.asyncio
async def test_memory_candidate_temporary():
    """temporary → memory_type='temporary', decay_rate=0.3."""
    owner = await _get_owner()
    async with get_session() as session:
        owner2 = await get_or_create_user(session, OWNER_TG_ID)
        await add_memory(
            session,
            owner2,
            fact="Временный факт",
            source="chat",
            memory_type="temporary",
            decay_rate=0.3,
            importance=0.5,
        )
    async with get_session() as session:
        owner3 = await get_or_create_user(session, OWNER_TG_ID)
        mems = await list_memories(session, owner3)
        temp_mems = [m for m in mems if m.memory_type == "temporary"]
        assert len(temp_mems) == 1
        assert temp_mems[0].fact == "Временный факт"
        assert temp_mems[0].decay_rate == 0.3


@pytest.mark.asyncio
async def test_permanent_decay_rate():
    """permanent → decay_rate=0.01."""
    owner = await _get_owner()
    async with get_session() as session:
        owner2 = await get_or_create_user(session, OWNER_TG_ID)
        await add_memory(
            session,
            owner2,
            fact="Вечный факт",
            source="user",
            decay_rate=0.01,
            importance=1.0,
        )
    async with get_session() as session:
        owner3 = await get_or_create_user(session, OWNER_TG_ID)
        mems = await list_memories(session, owner3)
        perm_mems = [m for m in mems if m.decay_rate == 0.01]
        assert len(perm_mems) == 1
        assert perm_mems[0].fact == "Вечный факт"


@pytest.mark.asyncio
async def test_commitment_to_task_memory():
    """Commitment done → task memory пишет в того же пользователя."""
    owner = await _get_owner()
    async with get_session() as session:
        owner2 = await get_or_create_user(session, OWNER_TG_ID)
        c = await add_commitment(
            session,
            user_id=owner2.id,
            peer_id=0,
            peer_name="",
            message_id=None,
            direction="mine",
            text="Позвонить маме",
            deadline_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        cid = c.id

    async with get_session() as session:
        await update_commitment_status(session, cid, "done")
        owner3 = await get_or_create_user(session, OWNER_TG_ID)
        await add_memory(
            session,
            owner3,
            fact=f"Выполнено: Позвонить маме",
            source="commitment",
            memory_type="task",
        )

    async with get_session() as session:
        owner4 = await get_or_create_user(session, OWNER_TG_ID)
        mems = await list_memories(session, owner4)
        task_mems = [m for m in mems if m.memory_type == "task"]
        assert len(task_mems) == 1
        assert "Позвонить маме" in task_mems[0].fact
        assert task_mems[0].user_id == owner.id  # DB user id


@pytest.mark.asyncio
async def test_get_prompt_facts_is_active():
    """get_prompt_facts() возвращает только active факты."""
    owner = await _get_owner()
    async with get_session() as session:
        owner2 = await get_or_create_user(session, OWNER_TG_ID)
        await add_memory(
            session, owner2, fact="Активный", source="chat", importance=0.8
        )
        await add_memory(
            session, owner2, fact="Неактивный", source="chat", importance=0.5
        )
        mems = await list_memories(session, owner2)
        mems[1].is_active = False  # деактивируем второй

    async with get_session() as session:
        owner3 = await get_or_create_user(session, OWNER_TG_ID)
        facts = await get_prompt_facts(session, owner3, total_limit=10)
        assert len(facts) == 1
        assert facts[0].fact == "Активный"
