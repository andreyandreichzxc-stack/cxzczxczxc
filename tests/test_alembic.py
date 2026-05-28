"""Alembic migration integrity tests.

Verifies:
- Exactly one migration head (no forks).
- Migration history is a connected, unbroken chain.
- ``alembic upgrade head`` succeeds on a clean SQLite database.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_env() -> dict[str, str]:
    """Return the bare env vars needed for project code to import cleanly."""
    return {
        "BOT_TOKEN": "test:token",
        "OWNER_TELEGRAM_ID": "123456789",
        "ENCRYPTION_KEY": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "PYTHONUNBUFFERED": "1",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_alembic_heads_single() -> None:
    """Verify there is exactly one active alembic head."""
    env = {**os.environ, **_minimal_env()}

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(f"alembic heads failed:\n{result.stderr}\n{result.stdout}")

    # Alembic output format: ``<revision> (head)``
    heads = [line for line in result.stdout.split("\n") if "(head)" in line]

    assert len(heads) == 1, (
        f"Expected 1 alembic head, got {len(heads)}: {heads}\n"
        "Run 'alembic merge heads' to consolidate."
    )


def test_alembic_history_chain() -> None:
    """Verify migration history forms a connected chain (no missing revisions)."""
    env = {**os.environ, **_minimal_env()}

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "history"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        if "Multiple heads" in result.stderr:
            pytest.fail(
                "Multiple alembic heads detected. "
                "Run 'alembic merge heads' to consolidate.\n" + result.stderr
            )
        pytest.fail(f"alembic history failed:\n{result.stderr}\n{result.stdout}")

    assert result.stdout.strip(), "Alembic history is empty — no migrations registered."


def test_alembic_upgrade_clean_db() -> None:
    """Verify ``alembic upgrade head`` works on a fresh SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir, "test.db")
        # env.py uses an async engine, so the URL must carry the aiosqlite driver.
        db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

        env = {**os.environ, **_minimal_env(), "DATABASE_URL": db_url}

        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            pytest.fail(
                f"alembic upgrade head failed on clean DB:\n"
                f"STDERR:\n{result.stderr}\n"
                f"STDOUT:\n{result.stdout}"
            )

        # Verify the resulting DB contains the expected tables.
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        conn.close()

        table_names = [t[0] for t in tables]

        assert "alembic_version" in table_names, (
            f"alembic_version table missing after upgrade. Tables found: {table_names}"
        )
        assert len(table_names) > 1, (
            f"Only {len(table_names)} table(s) after upgrade — "
            f"migrations likely produced no real tables. "
            f"Tables: {table_names}"
        )
