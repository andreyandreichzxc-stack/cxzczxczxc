import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import PROJECT_ROOT, settings
from src.db.models import Base

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.database_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Ensure performance PRAGMAs are set on every new connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.execute("PRAGMA mmap_size=134217728")  # 128 MB (safe for containers)
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA wal_autocheckpoint=1000")
    cursor.close()


# Alembic is the CANONICAL schema migration path.
# Future model changes MUST be captured via:
#   alembic revision --autogenerate -m "description"
#   alembic upgrade head
# Base.metadata.create_all is only used as a one-time bootstrap fallback
# when the alembic_version table is missing (fresh DB / direct run of main()).

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

# Agent Session FTS5: external-content virtual table linked to agent_session_messages.
# No triggers needed — FTS5 reads directly from the content table via content_rowid.
_SESSION_FTS_SETUP = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS agent_session_messages_fts USING fts5(
        content, role, session_id UNINDEXED,
        content='agent_session_messages', content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    );
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


async def _migrate_related_memory_to_links(conn) -> None:
    """One-time data migration: related_memory_id → memory_links.

    Copies old-style ``memories.related_memory_id`` + ``memories.relation_type``
    into the ``memory_links`` table if the link doesn't already exist.
    Safe to re-run: checks for existing links before inserting.

    This is a DATA migration, not a schema migration, so it lives here
    rather than in Alembic.
    """
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


async def init_db() -> None:
    """Initialise database: PRAGMAs, schema, FTS5 tables, data migrations.

    Schema management policy
    ------------------------
    Alembic is the **canonical** migration path.  ``run()`` in ``main.py``
    runs ``alembic upgrade head`` synchronously before the event loop starts,
    so by the time this function executes the ``alembic_version`` table exists
    and all migrations have been applied.

    ``Base.metadata.create_all`` is only used as a **one-time bootstrap
    fallback** when the ``alembic_version`` table is missing (fresh database,
    or ``main()`` called directly without the ``run()`` wrapper).  In that
    case the head revision is stamped immediately so subsequent runs can use
    Alembic.

    This avoids the "belt and suspenders" anti-pattern where both Alembic
    *and* ``create_all`` run on every startup, which can silently hide
    missing migrations (developer adds a column to an ORM model but forgets
    ``alembic revision --autogenerate`` → ``create_all`` silently creates it,
    masking the desync).
    """
    settings.data_dir  # триггерит создание директории
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.execute(text("PRAGMA cache_size=-64000"))  # 64 MB page cache
        await conn.execute(text("PRAGMA mmap_size=134217728"))  # 128 MB mmap
        await conn.execute(text("PRAGMA busy_timeout=30000"))  # 30s busy timeout
        await conn.execute(text("PRAGMA temp_store=MEMORY"))  # temp tables in memory
        await conn.execute(
            text("PRAGMA wal_autocheckpoint=1000")
        )  # checkpoint every 1000 pages

        # --- Schema: Alembic-canonical, create_all as bootstrap fallback ---
        result = await conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='alembic_version'"
            )
        )
        alembic_applied = result.first() is not None

        if alembic_applied:
            logger.debug(
                "alembic_version table found — Alembic is canonical; "
                "skipping Base.metadata.create_all"
            )
        else:
            logger.warning(
                "alembic_version table NOT found — bootstrapping schema via "
                "Base.metadata.create_all. This should only happen on a "
                "fresh database or if main() is called directly. Alembic "
                "remains the canonical migration path going forward."
            )
            await conn.run_sync(Base.metadata.create_all)

            # Stamp the head revision so Alembic knows all migrations are
            # already applied (create_all built the current ORM schema).
            _alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
            _script = ScriptDirectory.from_config(_alembic_cfg)
            head_rev = _script.get_current_head()
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version "
                    "(version_num VARCHAR(32) NOT NULL)"
                )
            )
            await conn.execute(
                text(
                    "INSERT OR IGNORE INTO alembic_version (version_num) VALUES (:rev)"
                ),
                {"rev": head_rev},
            )
            logger.info(
                "Stamped alembic head revision %s after create_all bootstrap",
                head_rev,
            )

        # FTS5 virtual tables are not tracked by Alembic — raw SQL.
        for stmt in _FTS_SETUP:
            await conn.execute(text(stmt))
        for stmt in _SESSION_FTS_SETUP:
            await conn.execute(text(stmt))
        for stmt in _MEMORY_FTS_SETUP:
            await conn.execute(text(stmt))

        # Data migration (not schema): related_memory_id → memory_links.
        await _migrate_related_memory_to_links(conn)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
