"""mcp_pdf tool — registered via @tool decorator.

Read and analyse PDF files.

Actions:
- ``action="read" path="data/file.pdf" page=1`` — extract text from a PDF page
- ``action="info" path="data/file.pdf"`` — get PDF metadata (pages, size, author, title)
- ``action="merge" paths=["data/a.pdf","data/b.pdf"] output="data/merged.pdf"`` — merge multiple PDFs

Path validation uses ``_safe_resolve`` from ``mcp_tools`` — only paths under ``data/`` are
allowed. Encrypted PDFs are rejected.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.core.actions.mcp_tools import _safe_resolve
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_MAX_PAGES_READ = 10  # max pages to extract text from per call


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_pdf
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_pdf",
    description=(
        "Read and analyse PDF files. Supports three actions:\n"
        "- 'read' — extract text from a PDF page (max 10 pages per call).\n"
        "- 'info' — get PDF metadata (pages, file size, author, title).\n"
        "- 'merge' — merge multiple PDFs into one file.\n"
        "Paths are restricted to data/ directory."
    ),
    category="utility",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — 'read', 'info', or 'merge'",
        "path": "str — path to a PDF file (required for 'read' and 'info')",
        "page": "int — starting page number (1-based, default 1, used with 'read')",
        "pages": "int — number of pages to read (default 1, max 10, used with 'read')",
        "paths": "list[str] — list of PDF paths to merge (required for 'merge')",
        "output": "str — output path for merged PDF (required for 'merge')",
    },
)
async def mcp_pdf(
    action: str,
    path: str = "",
    page: int = 1,
    pages: int = 1,
    paths: list[str] | None = None,
    output: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """PDF file reading and analysis tool.

    Args:
        action: ``"read"``, ``"info"``, or ``"merge"``.
        path: Path to a PDF file (required for ``"read"`` and ``"info"``).
        page: Starting page number, 1-based (default 1, used with ``"read"``).
        pages: Number of pages to read (default 1, max 10, used with ``"read"``).
        paths: List of PDF paths to merge (required for ``"merge"``).
        output: Output path for merged PDF (required for ``"merge"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "read":
            if not path or not path.strip():
                return {"error": "path parameter is required for action='read'"}
            return await _pdf_read(
                path.strip(), max(1, page), max(1, min(pages, _MAX_PAGES_READ))
            )
        elif action == "info":
            if not path or not path.strip():
                return {"error": "path parameter is required for action='info'"}
            return await _pdf_info(path.strip())
        elif action == "merge":
            if not paths or len(paths) < 2:
                return {
                    "error": "paths parameter must contain at least 2 PDF paths for action='merge'"
                }
            if not output or not output.strip():
                return {"error": "output parameter is required for action='merge'"}
            return await _pdf_merge(paths, output.strip())
        else:
            return {
                "error": f"Unknown action {action!r}. Valid actions: read, info, merge"
            }
    except Exception as exc:
        logger.exception("mcp_pdf(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _import_pypdf() -> Any:
    """Lazy import of pypdf. Returns the module or raises ImportError."""
    try:
        import pypdf

        return pypdf
    except ImportError:
        raise ImportError("pypdf not installed: pip install pypdf")


def _check_is_encrypted(reader: Any) -> bool:
    """Check if a pypdf PdfReader is encrypted."""
    try:
        return reader.is_encrypted  # type: ignore[no-any-return]
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _pdf_read(file_path: str, start_page: int, num_pages: int) -> dict[str, Any]:
    """Extract text from *num_pages* pages starting at *start_page*."""
    pypdf = _import_pypdf()

    resolved = _safe_resolve(file_path)
    if resolved is None:
        return {
            "error": f"Path {file_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_file():
        return {"error": f"File not found: {resolved}"}

    loop = asyncio.get_running_loop()

    def _extract() -> dict[str, Any]:
        try:
            reader = pypdf.PdfReader(str(resolved))
        except Exception as exc:
            raise ValueError(f"Cannot open PDF: {exc}")

        if _check_is_encrypted(reader):
            raise ValueError("PDF is password protected — cannot read")

        total_pages = len(reader.pages)
        if start_page > total_pages:
            raise ValueError(
                f"Start page {start_page} exceeds total pages ({total_pages})"
            )

        end_page = min(start_page + num_pages - 1, total_pages)
        extracted: dict[int, str] = {}
        for i in range(start_page - 1, end_page):
            try:
                text = reader.pages[i].extract_text() or ""
                extracted[i + 1] = text.strip()
            except Exception as exc:
                extracted[i + 1] = f"[Error extracting page {i + 1}: {exc}]"

        return {
            "pages": extracted,
            "page_range": f"{start_page}-{end_page}",
            "total_pages": total_pages,
        }

    try:
        result = await loop.run_in_executor(None, _extract)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("PDF read error: %s", exc)
        return {"error": f"Failed to read PDF: {exc}"}

    return {"ok": True, **result}


async def _pdf_info(file_path: str) -> dict[str, Any]:
    """Get metadata for a PDF file."""
    pypdf = _import_pypdf()

    resolved = _safe_resolve(file_path)
    if resolved is None:
        return {
            "error": f"Path {file_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_file():
        return {"error": f"File not found: {resolved}"}

    loop = asyncio.get_running_loop()

    def _metadata() -> dict[str, Any]:
        try:
            reader = pypdf.PdfReader(str(resolved))
        except Exception as exc:
            raise ValueError(f"Cannot open PDF: {exc}")

        if _check_is_encrypted(reader):
            raise ValueError("PDF is password protected — cannot read metadata")

        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title", None),
            "author": meta.get("/Author", None),
            "subject": meta.get("/Subject", None),
            "creator": meta.get("/Creator", None),
            "producer": meta.get("/Producer", None),
            "pages": len(reader.pages),
            "file_size_bytes": resolved.stat().st_size,
            "file_size_mb": round(resolved.stat().st_size / (1024**2), 2),
        }

    try:
        info = await loop.run_in_executor(None, _metadata)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("PDF info error: %s", exc)
        return {"error": f"Failed to read PDF metadata: {exc}"}

    return {"ok": True, **info}


async def _pdf_merge(paths: list[str], output: str) -> dict[str, Any]:
    """Merge multiple PDFs into one."""
    pypdf = _import_pypdf()

    out_resolved = _safe_resolve(output)
    if out_resolved is None:
        return {
            "error": f"Output path {output!r} is outside allowed directories or contains '..'"
        }

    resolved_paths: list[Path] = []
    for raw_path in paths:
        r = _safe_resolve(raw_path)
        if r is None:
            return {
                "error": f"Path {raw_path!r} is outside allowed directories or contains '..'"
            }
        if not r.is_file():
            return {"error": f"File not found: {r}"}
        resolved_paths.append(r)

    loop = asyncio.get_running_loop()

    def _merge() -> None:
        writer = pypdf.PdfWriter()
        for rp in resolved_paths:
            reader = pypdf.PdfReader(str(rp))
            if _check_is_encrypted(reader):
                raise ValueError(f"PDF is password protected — cannot merge: {rp}")
            for page_obj in reader.pages:
                writer.add_page(page_obj)
        # Ensure output directory exists
        out_resolved.parent.mkdir(parents=True, exist_ok=True)
        with out_resolved.open("wb") as fh:
            writer.write(fh)

    try:
        await loop.run_in_executor(None, _merge)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("PDF merge error: %s", exc)
        return {"error": f"Failed to merge PDFs: {exc}"}

    return {
        "ok": True,
        "output": str(out_resolved),
        "source_count": len(resolved_paths),
        "size_bytes": out_resolved.stat().st_size,
        "size_mb": round(out_resolved.stat().st_size / (1024**2), 2),
    }
