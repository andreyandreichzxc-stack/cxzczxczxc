"""Global test fixtures for TelegramHelper test suite.

Provides ``_db_init`` session-scoped fixture that creates all database tables
once for the entire test session.  Test files that need a database can either:

* ``@pytest.mark.usefixtures("_db_init")`` at module/class level, or
* declare ``_db_init`` as a dependency in their own fixture chain.

The fixture uses ``Base.metadata.create_all`` + raw FTS setup **without**
creating the ``alembic_version`` table, so that individual test-file fixtures
that later call ``init_db()`` (which checks for ``alembic_version``) always
detect a fresh database and run the full ``create_all`` boot sequence.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(scope="session")
def _db_init():
    """Create all DB tables — runs once at session start, tears down at end.

    This is a **sync** fixture (matching the pattern used throughout the
    test suite) that uses ``asyncio.run()`` internally.
    """
    from src.db.session import engine, Base
    from sqlalchemy import text

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            from src.db.session import _FTS_SETUP, _MEMORY_FTS_SETUP

            for stmt in _FTS_SETUP:
                await conn.execute(text(stmt))
            for stmt in _MEMORY_FTS_SETUP:
                await conn.execute(text(stmt))

    asyncio.run(_create())

    yield

    async def _drop():
        async with engine.begin() as conn:
            # Virtual tables not in Base.metadata — drop them first
            for tbl in ("messages_fts", "memories_fts"):
                await conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    asyncio.run(_drop())
