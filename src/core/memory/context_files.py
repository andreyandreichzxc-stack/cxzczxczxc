"""Context Files — per-contact knowledge stored as markdown files.

Bot stores and reads data/contexts/{contact_name}.md files.
When a contact name is mentioned in a message, the relevant context
is injected into the system prompt so the LLM "knows" about that person.

LLM-WIKI: Generic key-based API for arbitrary knowledge files.
See: save_context / get_context / append_to_context / search_in_contexts.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

CONTEXTS_DIR: Path = settings.data_dir / "contexts"

_MAX_CONTEXT_CHARS = 2000

# === LLM-WIKI constants ===
OWNER_KEY = "_owner"  # special key for owner profile


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
        for i, l in enumerate(current_lines):
            if l.startswith("- ["):
                header_end = i
                break
        if header_end > 0:
            current_lines.pop(header_end)
        # Remove first fact line after header
        for i, l in enumerate(current_lines):
            if l.startswith("- ["):
                current_lines.pop(i)
                break

    current_lines.append(new_line.rstrip("\n"))
    path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
    logger.debug("Appended to '%s': %s", key, line[:80])


def list_context_files() -> list[str]:
    """List all context file keys (without .md extension)."""
    if not CONTEXTS_DIR.exists():
        return []
    return sorted(
        f.stem for f in CONTEXTS_DIR.iterdir() if f.suffix == ".md" and f.stem
    )


def search_in_contexts(query: str, limit: int = 5) -> list[dict]:
    """Search across all context files for a query (substring, case-insensitive).

    Returns [{"key": "оля", "snippet": "...контекст..."}, ...]
    """
    if not CONTEXTS_DIR.exists():
        return []
    results = []
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
            results.append({"key": md_file.stem, "snippet": snippet})
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


_setup_auto_save_hook()
