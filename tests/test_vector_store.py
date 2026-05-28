"""Tests for VectorStore — point_id collisions, lazy singleton, search with existing collection."""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import PropertyMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.core.actions.vector_store import VectorStore, get_vector_store
from src.config import Settings


# ---------------------------------------------------------------------------
# _point_id — no collisions
# ---------------------------------------------------------------------------


def test_point_id_no_collisions():
    """_point_id generates unique IDs without collisions (including large Telegram IDs)."""
    ids: set[int] = set()
    test_cases = [
        (1234567890123, 1234567890123, 1234567890123),
        (1234567890123, 1234567890123, 1234567890124),
        (1234567890123, 1234567890124, 1234567890123),
        (1234567890124, 1234567890123, 1234567890123),
        (1, 1, 1),
        (65535, 16777215, 16777215),  # edge cases for old bit-packing
        (65536, 1, 1),
        (1, 16777216, 1),
        (1, 1, 16777216),
    ]
    for case in test_cases:
        point_id = VectorStore._point_id(*case)
        assert point_id not in ids, f"Collision for {case}: {point_id}"
        ids.add(point_id)


# ---------------------------------------------------------------------------
# get_vector_store — lazy singleton
# ---------------------------------------------------------------------------


def test_get_vector_store_lazy():
    """get_vector_store is a lazy singleton: None at import, same instance on repeat calls."""
    import src.core.actions.vector_store as vs

    vs._vector_store = None

    # Verify lazy: not initialized at import time
    assert vs._vector_store is None, (
        "VectorStore must be None before first get_vector_store() call"
    )

    vs_instance = None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with patch.object(
            Settings, "data_dir", new_callable=PropertyMock
        ) as mock_data_dir:
            mock_data_dir.return_value = Path(tmpdir)
            vs1 = get_vector_store()
            vs2 = get_vector_store()
            vs_instance = vs1
            assert vs1 is vs2, (
                "get_vector_store() must return the same instance on repeated calls"
            )
    # Ensure Qdrant client is closed BEFORE tempdir cleanup
    if vs_instance is not None:
        try:
            vs_instance._client.close()
        except Exception:
            pass
    # Reset singleton to prevent stale closed-client singleton from leaking
    vs._vector_store = None


# ---------------------------------------------------------------------------
# search with pre-existing Qdrant collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_store_search_existing_collection():
    """search works when a VectorStore is created against an existing Qdrant collection."""
    vs = None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        qdrant_dir = Path(tmpdir) / "qdrant"
        qdrant_dir.mkdir()

        # Pre-create a Qdrant collection at the test path
        pre_client = QdrantClient(path=str(qdrant_dir))
        pre_client.create_collection(
            "messages",
            vectors_config=qmodels.VectorParams(
                size=4, distance=qmodels.Distance.COSINE
            ),
        )
        pre_client.close()

        # Create VectorStore pointing to the same directory
        with patch.object(
            Settings, "data_dir", new_callable=PropertyMock
        ) as mock_data_dir:
            mock_data_dir.return_value = Path(tmpdir)
            vs = VectorStore()
            # Should find existing collection, set _dim, and return empty result
            results = await vs.search(user_id=1, embedding=[0.1, 0.2, 0.3, 0.4])
            assert results == []
            assert vs._dim == 4, (
                f"Expected _dim=4 from existing collection, got {vs._dim}"
            )
    # Ensure clients are closed before tempdir cleanup
    if vs is not None:
        try:
            vs._client.close()
        except Exception:
            pass
