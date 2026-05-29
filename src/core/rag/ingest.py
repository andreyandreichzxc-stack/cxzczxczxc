"""Document ingestion — scan, extract, chunk, embed, store.

Handles the full pipeline from a file or directory to Qdrant.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from src.config import settings
from src.core.rag.chunker import chunk_text, ChunkConfig
from src.core.rag.document_store import get_document_store

logger = logging.getLogger(__name__)

# Directory where user documents are placed for auto-indexing
DOCUMENTS_DIR = settings.data_dir / "documents"

# Supported file extensions
_SUPPORTED = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
}


def _compute_hash(file_path: Path) -> str:
    """SHA-256 hash of file content for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


async def _extract_text(path: Path) -> str:
    """Extract text from supported document formats."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return await _read_pdf(path)
    if suffix == ".docx":
        return await _read_docx(path)
    if suffix in (".html", ".htm"):
        return await _read_html(path)
    # Text formats
    return await asyncio.to_thread(_read_text_sync, path)


def _read_text_sync(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


async def _read_pdf(path: Path) -> str:
    def _inner():
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        chunks: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    chunks.append(t)
            except Exception:
                continue
        return "\n".join(chunks).strip()

    return await asyncio.to_thread(_inner)


async def _read_docx(path: Path) -> str:
    def _inner():
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    return await asyncio.to_thread(_inner)


async def _read_html(path: Path) -> str:
    """Extract text from HTML, stripping tags."""

    def _inner():
        import re

        text = _read_text_sync(path)
        # Remove scripts, styles, and HTML tags
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    return await asyncio.to_thread(_inner)


async def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for text chunks using the primary provider."""
    from src.core.actions.embedding_cache import get as cache_get
    from src.core.actions.embedding_cache import set as cache_set
    from src.llm.router import build_provider
    from src.llm.base import TaskType
    from src.db.session import get_session
    from src.db.repo import get_or_create_user

    # Get session + user for build_provider
    try:
        async with get_session() as session:
            user = await get_or_create_user(session, settings.owner_telegram_id)
            provider = await build_provider(
                session, user, task_type=TaskType.BACKGROUND
            )
    except Exception:
        provider = None

    if provider is None:
        logger.warning("No embedding provider available — skipping embeddings")
        return [[0.0] * settings.embedding_dim for _ in texts]

    model = getattr(provider, "_embed_model", None) or "text-embedding-3-small"
    embeddings: list[list[float]] = []

    for text in texts:
        # Check cache
        cached = cache_get(text, model)
        if cached is not None:
            embeddings.append(cached)
            continue

        try:
            emb = await provider.embed(text)
            cache_set(text, emb, model)
            embeddings.append(emb)
        except Exception as e:
            logger.debug("Embedding failed for chunk: %s", e)
            embeddings.append([0.0] * settings.embedding_dim)

    return embeddings


async def _scan_directory(root: Path) -> list[Path]:
    """Recursively find all supported files in a directory."""
    files: list[Path] = []
    try:
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix.lower() in _SUPPORTED:
                files.append(entry)
            elif entry.is_dir() and not entry.name.startswith("."):
                files.extend(await _scan_directory(entry))
    except PermissionError:
        logger.warning("Permission denied scanning %s", root)
    return sorted(files)


async def ingest_file(
    file_path: Path,
    user_id: int,
    chunk_config: ChunkConfig | None = None,
) -> dict:
    """Ingest a single file into the document store.

    Returns:
        dict with keys: ok, filename, chunks, content_hash, error (if any)
    """
    if not file_path.exists():
        return {"ok": False, "filename": str(file_path.name), "error": "Файл не найден"}

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED:
        return {
            "ok": False,
            "filename": file_path.name,
            "error": f"Неподдерживаемый формат: {suffix}",
        }

    try:
        # 1. Hash for deduplication
        content_hash = await asyncio.to_thread(_compute_hash, file_path)
        store = get_document_store()

        # 2. Check if already indexed (same hash)
        existing = await store.list_documents(user_id)
        for doc in existing:
            if doc.get("content_hash") == content_hash:
                return {
                    "ok": True,
                    "filename": file_path.name,
                    "chunks": doc.get("total_chunks", 0),
                    "content_hash": content_hash[:12],
                    "skipped": True,
                    "message": "Уже проиндексирован (хэш совпадает)",
                }

        # 3. Extract text
        text = await _extract_text(file_path)
        if not text or not text.strip():
            return {
                "ok": False,
                "filename": file_path.name,
                "error": "Не удалось извлечь текст",
            }

        # 4. Chunk
        if chunk_config is None:
            chunk_config = ChunkConfig()
        chunks = chunk_text(text, chunk_config)
        if not chunks:
            return {
                "ok": False,
                "filename": file_path.name,
                "error": "Текст пуст после разбивки",
            }

        # 5. Embed
        embeddings = await _get_embeddings(chunks)

        # 6. Store in Qdrant
        await store.upsert_chunks(
            user_id=user_id,
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=suffix,
            chunks=chunks,
            embeddings=embeddings,
            content_hash=content_hash,
        )

        logger.info(
            "Ingested '%s': %d chunks, hash=%s",
            file_path.name,
            len(chunks),
            content_hash[:12],
        )

        return {
            "ok": True,
            "filename": file_path.name,
            "chunks": len(chunks),
            "content_hash": content_hash[:12],
            "file_type": suffix,
        }

    except Exception as e:
        logger.exception("Failed to ingest %s", file_path)
        return {
            "ok": False,
            "filename": file_path.name,
            "error": str(e)[:500],
        }


async def ingest_directory(
    directory: Path,
    user_id: int,
    chunk_config: ChunkConfig | None = None,
) -> dict:
    """Ingest all supported files in a directory.

    Returns:
        dict with keys: ok, total_files, ingested, skipped, errors
    """
    if not directory.exists() or not directory.is_dir():
        return {"ok": False, "error": "Директория не найдена"}

    files = await _scan_directory(directory)
    if not files:
        return {
            "ok": True,
            "total_files": 0,
            "ingested": 0,
            "skipped": 0,
            "errors": [],
            "message": "Нет поддерживаемых файлов в директории",
        }

    ingested = 0
    skipped = 0
    errors: list[str] = []

    for file_path in files:
        result = await ingest_file(file_path, user_id, chunk_config)
        if result.get("ok"):
            if result.get("skipped"):
                skipped += 1
            else:
                ingested += 1
        else:
            errors.append(f"{file_path.name}: {result.get('error', 'unknown')}")

    return {
        "ok": True,
        "total_files": len(files),
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
    }


async def get_ingested_documents(user_id: int) -> list[dict]:
    """Get list of all indexed documents."""
    store = get_document_store()
    return await store.list_documents(user_id)


async def delete_document(content_hash: str) -> bool:
    """Remove a document from the index by content hash."""
    store = get_document_store()
    result = await store.delete_document(content_hash)
    return result > 0


async def rebuild_index(
    directory: Path | None = None,
    user_id: int | None = None,
) -> dict:
    """Full rebuild: delete all documents and re-ingest from directory.

    If user_id is None, uses settings.owner_telegram_id.
    If directory is None, uses DOCUMENTS_DIR.
    """
    if directory is None:
        directory = DOCUMENTS_DIR
    if user_id is None:
        user_id = settings.owner_telegram_id

    # We can't delete all from Qdrant without user_id filter,
    # so just re-ingest everything (upsert will replace)
    return await ingest_directory(directory, user_id)


# ── Background task: watch and auto-index ──


async def _watch_and_reindex():
    """Background task that periodically re-indexes DOCUMENTS_DIR.

    Registered as a task_manager task.
    """
    user_id = settings.owner_telegram_id
    directory = DOCUMENTS_DIR

    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info("Created documents directory: %s", directory)
        except Exception:
            logger.exception("Failed to create documents dir")
            return

    # Track known file hashes to detect changes
    known_hashes: dict[str, str] = {}

    while True:
        try:
            files = await _scan_directory(directory)
            changed = False

            for fpath in files:
                try:
                    fhash = await asyncio.to_thread(_compute_hash, fpath)
                    fname = fpath.name

                    if fname not in known_hashes or known_hashes[fname] != fhash:
                        result = await ingest_file(fpath, user_id)
                        if result.get("ok") and not result.get("skipped"):
                            logger.info(
                                "Auto-indexed '%s': %d chunks",
                                fname,
                                result.get("chunks", 0),
                            )
                            changed = True
                        known_hashes[fname] = fhash
                except Exception:
                    continue

            # Clean up hashes for deleted files
            current_names = {f.name for f in files}
            for name in list(known_hashes):
                if name not in current_names:
                    del known_hashes[name]

        except Exception:
            logger.exception("watch_and_reindex iteration failed")

        await asyncio.sleep(settings.digest_check_sec)  # reuse interval


# Register with task manager if available
try:
    from src.core.infra.task_manager import task_manager

    @task_manager.task("rag-watchdog")
    async def rag_watchdog():
        await _watch_and_reindex()

except ImportError:
    logger.debug("task_manager not available — rag_watchdog not registered")
