"""Context Files — per-contact knowledge stored as markdown files.

Bot stores and reads data/contexts/{contact_name}.md files.
When a contact name is mentioned in a message, the relevant context
is injected into the system prompt so the LLM "knows" about that person.

LLM-WIKI: Generic key-based API for arbitrary knowledge files.
See: save_context / get_context / append_to_context / search_in_contexts.

Semantic search: embeds context files into Qdrant "contexts" collection.
Hybrid search combines FTS5 (keywords) + Qdrant (semantic) via RRF.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from src.config import PROJECT_ROOT, settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

CONTEXTS_DIR: Path = settings.data_dir / "contexts"

_MAX_CONTEXT_CHARS = 2000

# === LLM-WIKI constants ===
OWNER_KEY = "_owner"  # special key for owner profile

# === FTS5 helpers ===
_FTS5_KEYWORDS = frozenset({"or", "and", "not", "near"})

# === Qdrant semantic search ===
_QDRANT_COLLECTION = "contexts"
_qdrant_client: QdrantClient | None = None
_qdrant_dim: int | None = None


def _fts5_simple_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free-text query.

    Each word becomes a prefix-match joined with OR.
    FTS5 operator keywords are escaped with double-quotes.
    """
    parts: list[str] = []
    for raw in query.split():
        clean = "".join(ch for ch in raw if ch.isalnum() or ch in "_-")
        if len(clean) < 2:
            continue
        lower = clean.lower()
        if lower in _FTS5_KEYWORDS:
            parts.append(f'"{lower}"')
        else:
            parts.append(lower + "*")
    if not parts:
        return ""
    return " OR ".join(parts)


def _get_db_path() -> Path:
    """Resolve the SQLite database file path from settings.database_url."""
    db_url = str(settings.database_url)
    parsed = urlparse(db_url)
    db_path = Path(parsed.path.lstrip("/"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return db_path


# Per-file locks for thread-safe append (TOCTOU prevention)
_file_locks: dict[str, threading.Lock] = {}
_file_cleanup_counter = 0


def _get_file_lock(key: str) -> threading.Lock:
    global _file_cleanup_counter
    if key not in _file_locks:
        _file_locks[key] = threading.Lock()
    # Cleanup every 1000 accesses
    _file_cleanup_counter += 1
    if _file_cleanup_counter % 1000 == 0:
        for k in list(_file_locks.keys()):
            if not _file_locks[k].locked():
                del _file_locks[k]
    return _file_locks[key]


def get_contact_context(contact_name: str) -> str | None:
    """Read data/contexts/{name}.md and return content, or None."""
    path = CONTEXTS_DIR / f"{contact_name.lower()}.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return text[:_MAX_CONTEXT_CHARS]
    return None


def save_contact_context(contact_name: str, content: str) -> None:
    """Save/update context file for a contact."""
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTEXTS_DIR / f"{contact_name.lower()}.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Saved context for '%s' (%d chars)", contact_name, len(content))


def find_relevant_contexts(user_message: str) -> dict[str, str]:
    """Scan data/contexts/*.md, check if any contact name appears in user_message.

    Returns {name: content} for matched contacts.
    Empty dict if no matches, no files, or directory doesn't exist.
    """
    if not CONTEXTS_DIR.exists():
        return {}

    result: dict[str, str] = {}
    try:
        for md_file in CONTEXTS_DIR.iterdir():
            if md_file.suffix != ".md":
                continue
            contact_name = md_file.stem  # filename without .md
            if not contact_name:
                continue

            # Case-insensitive word-boundary match in user_message
            pattern = re.compile(rf"\b{re.escape(contact_name)}\b", re.IGNORECASE)
            if not pattern.search(user_message):
                continue

            # Read content (capped)
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                logger.warning("Failed to read context file: %s", md_file)
                continue

            if not text.strip():
                continue

            result[contact_name] = text[:_MAX_CONTEXT_CHARS]
    except PermissionError:
        logger.warning("Permission denied reading contexts directory")
    except OSError:
        logger.warning("OS error reading contexts directory")

    return result


# ============================================================================
# LLM-WIKI: Generic key-based API
# ============================================================================


def save_context(key: str, content: str) -> None:
    """Save/overwrite context file for any key (contact name, owner, arbitrary topic)."""
    # sanitize key: lowercase, replace spaces with -
    safe_k = key.lower().replace(" ", "-")[:64]
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTEXTS_DIR / f"{safe_k}.md"
    path.write_text(content, encoding="utf-8")
    _schedule_semantic_index(safe_k, content)
    logger.info("Saved context '%s' (%d chars)", key, len(content))


def get_context(key: str) -> str | None:
    """Read context file for any key."""
    safe_k = key.lower().replace(" ", "-")[:64]
    path = CONTEXTS_DIR / f"{safe_k}.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return text[:_MAX_CONTEXT_CHARS]
    return None


def append_to_context(key: str, text: str, max_lines: int = 500) -> None:
    """Append a fact/line to an existing context file.

    - If file doesn't exist, creates with basic header
    - Deduplicates exact text matches (case-insensitive)
    - Caps at max_lines lines
    - Adds timestamp prefix to new lines
    """
    safe_k = key.lower().replace(" ", "-")[:64]
    lock = _get_file_lock(safe_k)
    with lock:
        CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
        path = CONTEXTS_DIR / f"{safe_k}.md"

        # Normalize text
        line = text.strip()
        if not line:
            return

        # Read existing or create header
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            existing_lowered = existing.lower()
        else:
            header = f"# {key}\n\nАвто-сгенерированный контекст.\n\n"
            path.write_text(header, encoding="utf-8")
            existing = header
            existing_lowered = header.lower()

        # Dedup — skip if same text already exists
        if line.lower() in existing_lowered:
            return

        # Append with timestamp
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_line = f"- [{ts}] {line}\n"

        # Cap total lines
        current_lines = existing.split("\n")
        if len(current_lines) >= max_lines:
            # Remove oldest fact line (keep header)
            header_end = 0
            for i, line_text in enumerate(current_lines):
                if line_text.startswith("- ["):
                    header_end = i
                    break
            if header_end > 0:
                current_lines.pop(header_end)
            # Remove first fact line after header
            for i, line_text in enumerate(current_lines):
                if line_text.startswith("- ["):
                    current_lines.pop(i)
                    break

        current_lines.append(new_line.rstrip("\n"))
        path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
        _schedule_semantic_index(safe_k, "\n".join(current_lines))
        logger.debug("Appended to '%s': %s", key, line[:80])


def list_context_files() -> list[str]:
    """List all context file keys (without .md extension)."""
    if not CONTEXTS_DIR.exists():
        return []
    return sorted(
        f.stem for f in CONTEXTS_DIR.iterdir() if f.suffix == ".md" and f.stem
    )


def search_in_contexts(query: str, limit: int = 5) -> list[dict]:
    """Search across all context files using FTS5 with ranked results.

    Falls back to substring search if the FTS5 table doesn't exist.
    Returns [{"key": "оля", "snippet": "...<b>контекст</b>...", "rank": 0.5}, ...]
    """
    if not CONTEXTS_DIR.exists():
        return []

    db_path = _get_db_path()

    # ── Try FTS5 first ──────────────────────────────────────────────
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            fts_q = _fts5_simple_query(query)
            if fts_q:
                rows = conn.execute(
                    "SELECT key, snippet(contexts_fts, 1, '<b>', '</b>', '…', 64) AS snippet, "
                    "       rank "
                    "FROM contexts_fts WHERE contexts_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_q, limit),
                ).fetchall()
                conn.close()
                if rows:
                    return [
                        {
                            "key": r[0],
                            "snippet": r[1] or "",
                            "rank": float(r[2]) if r[2] is not None else 0.0,
                        }
                        for r in rows
                    ]
        except sqlite3.OperationalError:
            pass  # FTS5 table doesn't exist → fall back to substring
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ── Fallback: substring search ──────────────────────────────────
    results: list[dict] = []
    ql = query.lower()
    for md_file in CONTEXTS_DIR.iterdir():
        if md_file.suffix != ".md":
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        pos = text.lower().find(ql)
        if pos >= 0:
            start = max(0, pos - 40)
            end = min(len(text), pos + len(query) + 80)
            snippet = text[start:end].strip()
            results.append({"key": md_file.stem, "snippet": snippet, "rank": 0.0})
            if len(results) >= limit:
                break
    return results


def init_owner_context() -> None:
    """Create _owner.md template if it doesn't exist."""
    if get_context(OWNER_KEY) is not None:
        return
    template = (
        "# Владелец\n\n"
        "Авто-генерируемый профиль. Бот дополняет файл при обнаружении новых фактов.\n\n"
        "## Личное\n\n"
        "## Работа\n\n"
        "## Предпочтения\n\n"
        "## Принципы\n\n"
    )
    save_context(OWNER_KEY, template)
    logger.info("Initialized _owner.md context file")


# ============================================================================
# FTS5 indexing: index all .md context files for fast search
# ============================================================================


def index_contexts_to_fts() -> int:
    """Index all context .md files into the FTS5 virtual table.

    Creates ``contexts_fts(key, content)`` if it doesn't exist,
    then INSERT OR REPLACE every .md file's content.

    Called once at startup from ``main.py``.
    Returns count of indexed files.
    """
    if not CONTEXTS_DIR.exists():
        logger.debug("Contexts directory does not exist, skipping FTS5 indexing")
        return 0

    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("Database file not found at %s, skipping FTS5 indexing", db_path)
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS contexts_fts "
            "USING fts5(key, content, tokenize='unicode61 remove_diacritics 2')"
        )
        conn.execute("DELETE FROM contexts_fts")  # clear stale entries

        count = 0
        for md_file in sorted(CONTEXTS_DIR.iterdir()):
            if md_file.suffix != ".md":
                continue
            key = md_file.stem
            if not key:
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                logger.warning("Failed to read context file: %s", md_file)
                continue
            conn.execute(
                "INSERT INTO contexts_fts(key, content) VALUES (?, ?)",
                (key, content),
            )
            count += 1

        conn.commit()
        logger.info("Indexed %d context files into FTS5", count)
        return count
    finally:
        conn.close()


# ============================================================================
# Auto-extract facts: lightweight regex-based extraction from dialog turns
# ============================================================================

# Self-referential fact patterns
_SELF_FACT_PATTERNS = [
    re.compile(rf"(?:я\s+{pat}[^.]*\.)", re.IGNORECASE | re.UNICODE)
    for pat in [
        r"работаю",
        r"живу",
        r"учусь",
        r"занимаюсь",
        r"люблю",
        r"не\s+люблю",
        r"предпочитаю",
        r"хочу",
    ]
]

# Broader self-fact: "мне нравится X", "у меня Y"
_BROAD_SELF_FACT_RE = re.compile(
    r"(?:мне\s+нравится|у\s+меня\s+есть|мой\s+\w+|моя\s+\w+)[^.]*\.(?!\s*\d)",
    re.IGNORECASE | re.UNICODE,
)

# Contact-fact detection: "<Name> — fact" or "<Name> - fact" or "<Name> это fact"
# Built dynamically from existing context keys


async def try_extract_context_updates(
    session,
    user_text: str,
    assistant_text: str,
    owner_id: int,
) -> int:
    """Try to extract new facts from the current turn and update context files.

    Uses lightweight regex-based heuristics (no LLM call — fast):
    1. Detect self-referential facts: "я работаю в X", "мне нравится Y", etc.
    2. Detect contact references: "Оля — веган", "с Артёмом лучше не спорить"

    Args:
        session: DB session (unused — kept for caller compatibility).
        user_text: What the user said.
        assistant_text: What the bot replied (unused — kept for future use).
        owner_id: Telegram user ID (unused — kept for compatibility).

    Returns:
        Count of updated context files.
    """
    if not user_text:
        return 0

    updated = 0

    # ── 1. Self-facts → _owner.md ────────────────────────────────────
    for pat in _SELF_FACT_PATTERNS:
        for match in pat.finditer(user_text):
            fact = match.group(0).strip()
            if len(fact) > 10:
                await asyncio.to_thread(append_to_context, OWNER_KEY, fact)
                updated += 1

    for match in _BROAD_SELF_FACT_RE.finditer(user_text):
        fact = match.group(0).strip()
        # Avoid duplicates from exact patterns above
        if len(fact) > 10:
            await asyncio.to_thread(append_to_context, OWNER_KEY, fact)
            updated += 1

    # ── 2. Contact-facts → <contact>.md ──────────────────────────────
    existing_keys = list_context_files()
    for key_name in existing_keys:
        if key_name == OWNER_KEY or key_name.startswith("_"):
            continue
        # Pattern: "Name — fact" or "Name - fact" or "Name это fact"
        contact_pattern = re.compile(
            rf"\b{re.escape(key_name)}\s*(?:—|–|—|это|\s+-\s+)\s*(.+?)(?:\.\s|$)",
            re.IGNORECASE | re.UNICODE,
        )
        for match in contact_pattern.finditer(user_text):
            fact = match.group(1).strip()
            if 5 < len(fact) < 500:
                await asyncio.to_thread(append_to_context, key_name, fact)
                updated += 1

    if updated:
        logger.debug("try_extract_context_updates: updated %d context files", updated)
    return updated


# ============================================================================
# Auto-save hook: on_memory_saved → update context files
# ============================================================================


def _setup_auto_save_hook() -> None:
    """Register on_memory_saved → update context files."""
    try:
        from src.core.infra.hooks import hooks

        async def _on_memory_saved(
            user_id: int,
            contact_id: int | None,
            contact_name: str | None,
            fact: str,
            confidence: float,
            **kwargs,
        ):
            text = f"{fact} (уверенность: {confidence:.0%})"
            if contact_id and contact_name:
                # Per-contact fact
                await asyncio.to_thread(append_to_context, contact_name, text)
            else:
                # Owner fact
                await asyncio.to_thread(append_to_context, OWNER_KEY, text)

        hooks.on("on_memory_saved", _on_memory_saved)
        logger.info("Auto-save hook registered for context files")
    except Exception:
        logger.debug("Failed to register auto-save hook (hooks not ready yet)")


# ============================================================================
# Semantic search: Qdrant-based vector search for context files
# ============================================================================


def _get_qdrant() -> "QdrantClient":
    """Lazy-init Qdrant client for context vector search."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        path = settings.data_dir / "qdrant"
        path.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(path))
    return _qdrant_client


async def index_context_for_semantic(key: str, content: str, provider=None) -> bool:
    """Embed context file content and store in Qdrant 'contexts' collection."""
    if provider is None:
        return False
    try:
        embedding = await provider.embed(content[:2000])
        if not embedding:
            return False
        client = await asyncio.to_thread(_get_qdrant)
        dim = len(embedding)
        global _qdrant_dim
        if _qdrant_dim != dim:
            try:
                from qdrant_client.http import models as qmodels

                await asyncio.to_thread(
                    client.create_collection,
                    collection_name=_QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=dim, distance=qmodels.Distance.COSINE
                    ),
                )
            except Exception:
                pass
            _qdrant_dim = dim
        from qdrant_client.http import models as qmodels

        point_id = abs(hash(key)) % (10**9)
        await asyncio.to_thread(
            client.upsert,
            collection_name=_QDRANT_COLLECTION,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"key": key, "preview": content[:300]},
                )
            ],
        )
        logger.debug("Semantic index: '%s' (%d chars, dim=%d)", key, len(content), dim)
        return True
    except Exception:
        logger.debug("Semantic index failed for '%s'", key, exc_info=True)
        return False


async def search_contexts_semantic(query: str, provider, limit: int = 5) -> list[dict]:
    """Search context files semantically via Qdrant embeddings."""
    try:
        embedding = await provider.embed(query)
        if not embedding:
            return []
    except Exception:
        logger.debug("Failed to embed query for semantic search", exc_info=True)
        return []
    try:
        client = await asyncio.to_thread(_get_qdrant)
        try:
            await asyncio.to_thread(
                client.get_collection, collection_name=_QDRANT_COLLECTION
            )
        except Exception:
            return []

        results = await asyncio.to_thread(
            client.search,  # type: ignore[attr-defined]
            collection_name=_QDRANT_COLLECTION,
            query_vector=embedding,
            limit=limit,
        )
        return [
            {
                "key": r.payload.get("key", "?"),
                "snippet": r.payload.get("preview", "")[:200],
                "score": float(r.score),
            }
            for r in results
        ]
    except Exception:
        logger.debug("Semantic search failed", exc_info=True)
        return []


async def search_contexts_hybrid(
    query: str, provider=None, limit: int = 5
) -> list[dict]:
    """Hybrid search: FTS5 + semantic via RRF. Falls back to FTS5-only if no provider."""
    fts_results = search_in_contexts(query, limit=limit * 2)
    sem_results: list[dict] = []
    if provider:
        sem_results = await search_contexts_semantic(query, provider, limit=limit * 2)
    if not sem_results:
        return fts_results[:limit]
    # RRF merge
    K = 60
    scores: dict[str, float] = {}
    for i, r in enumerate(fts_results):
        scores[r["key"]] = scores.get(r["key"], 0) + 1.0 / (K + i + 1)
    for i, r in enumerate(sem_results):
        scores[r["key"]] = scores.get(r["key"], 0) + 1.0 / (K + i + 1)
    merged: dict[str, dict] = {}
    for r in fts_results + sem_results:
        k = r["key"]
        if k not in merged:
            merged[k] = {
                "key": k,
                "snippet": r.get("snippet", ""),
                "score": scores.get(k, 0),
            }
        else:
            merged[k]["score"] = scores.get(k, merged[k]["score"])
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]


async def rebuild_semantic_index(provider) -> int:
    """Re-index ALL context files into Qdrant. Returns count of indexed files."""
    if not CONTEXTS_DIR.exists():
        return 0
    count = 0
    for md_file in sorted(CONTEXTS_DIR.iterdir()):
        if md_file.suffix != ".md" or not md_file.stem:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if await index_context_for_semantic(md_file.stem, content, provider):
            count += 1
    logger.info("Rebuilt semantic index: %d context files", count)
    return count


def _schedule_semantic_index(key: str, content: str) -> None:
    """Fire-and-forget: schedule semantic indexing for a context file.

    Tries to get a running event loop and create a task.
    Gracefully skips if no event loop is running.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_index_with_provider(key, content))
    except RuntimeError:
        pass  # no running event loop — skip


async def _index_with_provider(key: str, content: str) -> None:
    """Lazy-load a provider and index the context file."""
    try:
        from src.llm.router import build_provider
        from src.db.repo import get_or_create_user
        from src.db.session import get_session

        async with get_session() as session:
            owner = await get_or_create_user(session, settings.owner_telegram_id)
            provider = await build_provider(session, owner)
            if provider:
                await index_context_for_semantic(key, content, provider)
    except Exception:
        logger.debug("Semantic index schedule failed for '%s'", key, exc_info=True)


_setup_auto_save_hook()
