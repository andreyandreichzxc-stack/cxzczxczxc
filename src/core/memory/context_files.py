"""Context Files — per-contact knowledge stored as markdown files.

Bot stores and reads data/contexts/{contact_name}.md files.
When a contact name is mentioned in a message, the relevant context
is injected into the system prompt so the LLM "knows" about that person.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

CONTEXTS_DIR: Path = settings.data_dir / "contexts"

_MAX_CONTEXT_CHARS = 2000


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
