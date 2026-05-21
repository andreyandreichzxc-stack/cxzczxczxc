import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.database_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Alembic is configured for schema migrations (alembic/).
# Future model changes should be captured via:
#   alembic revision --autogenerate -m "description"
# The ALTER TABLE blocks below handle legacy migrations for existing DBs.

# SQLite FTS5: virtual table + триггеры синхронизации с messages.
# Хранит rowid = messages.id.
_FTS_SETUP = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        text,
        transcript,
        extracted_text,
        sender_name,
        content='messages',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    );
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, text, transcript, extracted_text, sender_name)
        VALUES (new.id, new.text, new.transcript, new.extracted_text, new.sender_name);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text, transcript, extracted_text, sender_name)
        VALUES('delete', old.id, old.text, old.transcript, old.extracted_text, old.sender_name);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text, transcript, extracted_text, sender_name)
        VALUES('delete', old.id, old.text, old.transcript, old.extracted_text, old.sender_name);
        INSERT INTO messages_fts(rowid, text, transcript, extracted_text, sender_name)
        VALUES (new.id, new.text, new.transcript, new.extracted_text, new.sender_name);
    END;
    """,
]

# Memory FTS5: virtual table + триггеры синхронизации с memories.
_MEMORY_FTS_SETUP = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        fact, sentiment, cluster_topic,
        content='memories',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    );
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, fact, sentiment, cluster_topic)
        VALUES (new.id, new.fact, new.sentiment, new.cluster_topic);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, fact, sentiment, cluster_topic)
        VALUES('delete', old.id, old.fact, old.sentiment, old.cluster_topic);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, fact, sentiment, cluster_topic)
        VALUES('delete', old.id, old.fact, old.sentiment, old.cluster_topic);
        INSERT INTO memories_fts(rowid, fact, sentiment, cluster_topic)
        VALUES (new.id, new.fact, new.sentiment, new.cluster_topic);
    END;
    """,
]


async def init_db() -> None:
    settings.data_dir  # триггерит создание директории
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.execute(text("PRAGMA cache_size=-8000"))  # ~8MB cache
        await conn.execute(text("PRAGMA busy_timeout=30000"))
        await conn.run_sync(Base.metadata.create_all)
        for stmt in _FTS_SETUP:
            await conn.execute(text(stmt))
        for stmt in _MEMORY_FTS_SETUP:
            await conn.execute(text(stmt))

        # Миграция: user-колонки (last_seen_online и др.)
        for col, col_def in [
            ("last_seen_online", "TIMESTAMP"),
            ("absence_status", "VARCHAR(16)"),
            ("absence_message", "TEXT"),
            ("global_style_profile", "TEXT"),
            ("global_style_updated_at", "TIMESTAMP"),
        ]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
                )
            except Exception as e:
                if (
                    "duplicate column name" in str(e).lower()
                    or "already exists" in str(e).lower()
                ):
                    logger.debug("Migration for %s: column already exists", col)
                else:
                    raise

        # Миграция: добавляем колонки adaptive scoring если их нет
        for col, col_def in [
            ("memory_type", "VARCHAR(24)"),
            ("use_count", "INTEGER DEFAULT 0"),
            ("last_used_at", "TIMESTAMP"),
            ("expires_at", "TIMESTAMP"),
            ("pinned", "BOOLEAN DEFAULT 0"),
        ]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE memories ADD COLUMN {col} {col_def}")
                )
            except Exception as e:
                if (
                    "duplicate column name" in str(e).lower()
                    or "already exists" in str(e).lower()
                ):
                    logger.debug("Migration for %s: column already exists", col)
                else:
                    raise

        # Миграция: source_memory_id в commitments
        try:
            await conn.execute(
                text("ALTER TABLE commitments ADD COLUMN source_memory_id BIGINT")
            )
        except Exception as e:
            if (
                "duplicate column name" in str(e).lower()
                or "already exists" in str(e).lower()
            ):
                logger.debug("Migration for source_memory_id: column already exists")
            else:
                raise
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_commitments_source_memory_id "
                    "ON commitments(source_memory_id)"
                )
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(
                    "Migration for ix_commitments_source_memory_id: index already exists"
                )
            else:
                raise

        # Миграция старых связей памяти (related_memory_id → memory_links)
        try:
            result = await conn.execute(
                text(
                    "SELECT id, related_memory_id, relation_type FROM memories WHERE related_memory_id IS NOT NULL"
                )
            )
            for row in result.all():
                mid, related_id, rel_type = row
                # Проверить нет ли уже связи в memory_links
                check = await conn.execute(
                    text(
                        "SELECT id FROM memory_links WHERE source_id = :sid AND target_id = :tid"
                    ),
                    {"sid": mid, "tid": related_id},
                )
                if not check.first():
                    await conn.execute(
                        text(
                            "INSERT INTO memory_links (user_id, source_id, target_id, weight, relation_type, created_at) "
                            "SELECT user_id, :sid, :tid, 0.7, :rel, datetime('now') FROM memories WHERE id = :sid"
                        ),
                        {"sid": mid, "tid": related_id, "rel": rel_type},
                    )
        except Exception as e:
            if (
                "duplicate column name" in str(e).lower()
                or "already exists" in str(e).lower()
            ):
                logger.debug(
                    "Migration for related_memory_id → memory_links: already applied"
                )
            else:
                raise

        # Миграция: radar_snoozed_until для ConversationState
        try:
            await conn.execute(
                text(
                    "ALTER TABLE conversation_states ADD COLUMN radar_snoozed_until DATETIME"
                )
            )
        except Exception as e:
            if (
                "duplicate column name" in str(e).lower()
                or "already exists" in str(e).lower()
            ):
                logger.debug("Migration for radar_snoozed_until: column already exists")
            else:
                raise


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
