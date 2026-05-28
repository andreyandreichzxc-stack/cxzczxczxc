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
os.environ["ENCRYPTION_KEY"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
os.environ["BOT_TOKEN"] = "test:token"
os.environ["OWNER_TELEGRAM_ID"] = "123456789"

from src.db.session import init_db, get_session
from src.db.models import MemoryCandidate
from src.db.repo import (
    add_memory,
    add_memory_candidate,
    get_or_create_user,
    add_commitment,
    update_commitment_status,
    list_memories,
    upsert_conversation_state,
    upsert_message,
)
from src.core.actions.conflict_predictor import detect_silence_triggers
from src.core.memory.temporal_layers import get_prompt_facts
from src.core.memory.memory_checker import _run_decay_and_validation

OWNER_TG_ID = 123456789


@pytest.fixture(autouse=True)
def setup_db():
    """Пересоздаёт таблицы перед каждым тестом (чтобы не копились данные)."""
    from src.db.session import engine, Base
    from sqlalchemy import text

    async def _recreate():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            # Drop artifacts that survive drop_all and would confuse init_db
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            await conn.execute(text("DROP TABLE IF EXISTS messages_fts"))
            await conn.execute(text("DROP TABLE IF EXISTS memories_fts"))
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
    _owner = await _get_owner()
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
    _owner = await _get_owner()
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
            fact="Выполнено: Позвонить маме",
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
    _owner = await _get_owner()
    async with get_session() as session:
        owner2 = await get_or_create_user(session, OWNER_TG_ID)
        await add_memory(
            session, owner2, fact="Активный", source="chat", importance=0.8
        )
        await add_memory(
            session, owner2, fact="Неактивный", source="chat", importance=0.5
        )
        mems = await list_memories(session, owner2)
        inactive = next(m for m in mems if m.fact == "Неактивный")
        inactive.is_active = False

    async with get_session() as session:
        owner3 = await get_or_create_user(session, OWNER_TG_ID)
        facts = await get_prompt_facts(session, owner3, total_limit=10)
        assert len(facts) == 1
        assert facts[0].fact == "Активный"


@pytest.mark.asyncio
async def test_get_prompt_facts_skips_expired_and_tracks_usage():
    """get_prompt_facts() пропускает expired факты. use_count больше НЕ бампится здесь —
    только в recall() (Phase 0.2: убран двойной бамп)."""
    async with get_session() as session:
        owner = await get_or_create_user(session, OWNER_TG_ID)
        await add_memory(session, owner, fact="Свежий факт", source="chat")
        await add_memory(session, owner, fact="Истекший факт", source="chat")
        mems = await list_memories(session, owner)
        expired = next(m for m in mems if m.fact == "Истекший факт")
        expired.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

    async with get_session() as session:
        owner = await get_or_create_user(session, OWNER_TG_ID)
        facts = await get_prompt_facts(session, owner, total_limit=10)
        assert [m.fact for m in facts] == ["Свежий факт"]
        # use_count НЕ бампится в get_prompt_facts — только в recall()
        assert facts[0].use_count == 0


@pytest.mark.asyncio
async def test_decay_processes_all_expired_without_offset_skip():
    """Decay keyset-проход не пропускает строки при деактивации."""
    async with get_session() as session:
        owner = await get_or_create_user(session, OWNER_TG_ID)
        for idx in range(3):
            await add_memory(
                session, owner, fact=f"Временный факт {idx}", source="chat"
            )
        mems = await list_memories(session, owner)
        for mem in mems:
            mem.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

    _decayed, closed = await _run_decay_and_validation(OWNER_TG_ID)
    assert closed == 3

    async with get_session() as session:
        owner = await get_or_create_user(session, OWNER_TG_ID)
        mems = await list_memories(session, owner)
        assert all(not m.is_active for m in mems)


@pytest.mark.asyncio
async def test_conflict_predictor_uses_historical_outgoing_before_negative():
    """Conflict predictor считает молчание перед каждым негативом, а не от текущего last_outgoing_at."""
    contact_id = 4242
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with get_session() as session:
        owner = await get_or_create_user(session, OWNER_TG_ID)
        await upsert_message(
            session,
            user_id=owner.id,
            peer_id=contact_id,
            message_id=1,
            sender_id=OWNER_TG_ID,
            sender_name="me",
            is_outgoing=True,
            date=now - timedelta(hours=40),
            kind="text",
            text="пишу первый раз",
        )
        await upsert_message(
            session,
            user_id=owner.id,
            peer_id=contact_id,
            message_id=2,
            sender_id=OWNER_TG_ID,
            sender_name="me",
            is_outgoing=True,
            date=now - timedelta(hours=20),
            kind="text",
            text="пишу второй раз",
        )
        first_neg = await add_memory(
            session,
            owner,
            fact="контакт раздражён из-за молчания",
            contact_id=contact_id,
            sentiment="negative",
        )
        second_neg = await add_memory(
            session,
            owner,
            fact="контакт снова недоволен долгим ответом",
            contact_id=contact_id,
            sentiment="negative",
        )
        first_neg.created_at = now - timedelta(hours=30)
        second_neg.created_at = now - timedelta(hours=10)
        await upsert_conversation_state(
            session,
            owner,
            contact_id,
            last_outgoing_at=now - timedelta(hours=8),
        )

    triggers = await detect_silence_triggers(OWNER_TG_ID)
    trigger = next((t for t in triggers if t["contact_id"] == contact_id), None)
    assert trigger is not None
    assert trigger["silence_hours"] == 10
    assert trigger["current_hours"] == 8


@pytest.mark.asyncio
async def test_init_db_duplicate_radar_snoozed_until():
    """Повторный init_db() не должен падать если radar_snoozed_until уже существует."""
    # init_db уже вызван в setup_db fixture → вызываем повторно
    # должно пройти без ошибок (ловит "duplicate column name" / "already exists")
    from src.db.session import init_db as _init_db

    await _init_db()
