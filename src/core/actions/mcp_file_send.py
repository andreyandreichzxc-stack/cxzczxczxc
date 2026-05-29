"""mcp_file_send tool — registered via @tool decorator.

Send text-based files to the user via Telegram control bot.

Actions:
- ``action="create_and_send"`` — create a temp file and send it as
  a Telegram document to the owner.

Supported extensions: .txt, .json, .csv, .md, .html, .css, .py, .js,
.log, .xml, .yaml, .yml, .ini, .cfg, .toml.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES: int = 10 * 1024 * 1024  # 10 MB DoS protection

from aiogram.types import FSInputFile

from src.config import settings
from src.core.actions.tool_registry import ToolActionSpec, tool
from src.core.infra.notifier import notifier

logger = logging.getLogger(__name__)

# Разрешённые расширения (текстовые форматы)
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".json",
        ".csv",
        ".md",
        ".html",
        ".css",
        ".py",
        ".js",
        ".log",
        ".xml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".toml",
    }
)

_CAPTION_MAX_LEN = 1024  # Telegram caption limit


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_file_send
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_file_send",
    description=(
        "Create a text file and send it to the user via Telegram. "
        "Supports TXT, JSON, CSV, MD, HTML, CSS, Python, JS, "
        "LOG, XML, YAML, INI, CFG, TOML.\n\n"
        "Action:\n"
        "- 'create_and_send' — create a temporary file from content "
        "and send as a Telegram document with optional caption.\n\n"
        "Examples:\n"
        '  action="create_and_send" filename="report.json" content=\'{"key":"value"}\'\n'
        '  action="create_and_send" filename="notes.md" content="# Hello" caption="my notes"'
    ),
    category="utility",
    risk="low",
    actions={
        "create_and_send": ToolActionSpec(
            name="create_and_send",
            risk="low",
            read_only=False,
            destructive=False,
            idempotent=False,
            user_content=True,
        ),
    },
    params={
        "action": "str — 'create_and_send'",
        "filename": "str — file name with extension (e.g. output.txt)",
        "content": "str — file content",
        "caption": "str|None — optional file caption (max 1024 chars)",
    },
)
async def mcp_file_send(
    action: str = "create_and_send",
    filename: str = "output.txt",
    content: str = "",
    caption: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a temporary file and send it via Telegram.

    Args:
        action: Must be ``"create_and_send"``.
        filename: Output file name (with extension). Determines file type.
        content: Text content to write into the file.
        caption: Optional caption for the Telegram document (max 1024 chars).

    Keyword Args (injected at runtime):
        _bot: aiogram ``Bot`` instance (optional — falls back to notifier).
        _chat_id: Target chat ID (optional — falls back to owner).

    Returns:
        A dict with ``"ok"``, ``"file"``, ``"size_bytes"``, ``"message_id"``
        on success, or ``"error"`` on failure.
    """
    if action != "create_and_send":
        return {"error": f"Unknown action {action!r}. Valid actions: create_and_send"}

    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return {
            "error": (
                f"Unsupported extension {ext!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            )
        }

    # Resolve bot instance — prefer explicit kwargs, fall back to notifier
    bot = kwargs.get("_bot") or notifier._bot
    if bot is None:
        return {
            "error": (
                "No bot instance available. The control bot may not be started yet."
            )
        }

    # Resolve chat_id — prefer explicit kwargs, fall back to owner
    chat_id = kwargs.get("_chat_id")
    if chat_id is None:
        chat_id = settings.owner_telegram_id

    if not chat_id:
        return {"error": "No chat_id available (owner not configured)"}

    # Trim caption to Telegram limit
    safe_caption: str | None = None
    if caption:
        safe_caption = caption[:_CAPTION_MAX_LEN]

    # ── DoS protection: reject oversized content ────────────────────────
    content_bytes: bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_CONTENT_BYTES:
        return {
            "error": (
                f"Content exceeds {MAX_CONTENT_BYTES // 1024 // 1024}MB limit "
                f"({len(content_bytes)} bytes)"
            )
        }

    tmp_path: str | None = None
    try:
        # Create temporary file with the requested extension
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=ext,
            delete=False,
        ) as f:
            tmp_path = f.name  # capture BEFORE write so finally can clean up
            f.write(content_bytes)

        # Send via Telegram
        fs_file = FSInputFile(tmp_path, filename=filename)
        msg = await bot.send_document(
            chat_id=chat_id,
            document=fs_file,
            caption=safe_caption,
        )

        size_bytes = len(content_bytes)

        return {
            "ok": True,
            "file": filename,
            "size_bytes": size_bytes,
            "message_id": msg.message_id,
        }

    except Exception as exc:
        logger.exception(
            "mcp_file_send failed: filename=%r, size=%d",
            filename,
            len(content),
        )
        return {"error": str(exc)}

    finally:
        # Clean up the temporary file
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
