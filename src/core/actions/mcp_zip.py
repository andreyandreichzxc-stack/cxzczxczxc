"""mcp_zip tool — registered via @tool decorator.

Zip/unzip file operations.

Actions:
- ``action="list" path="data/archive.zip"`` — list files inside a zip archive
- ``action="extract" path="data/archive.zip" dest="data/extracted/"`` — extract to a directory
- ``action="create" paths=["data/file1.txt","data/file2.txt"] output="data/packed.zip"`` — create a zip archive

Path validation uses ``_safe_resolve`` from ``mcp_tools`` — only paths under ``data/`` are
allowed.
"""

from __future__ import annotations

import asyncio
import logging
import zipfile
from pathlib import Path
from typing import Any

from src.core.actions.mcp_tools import _safe_resolve
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_zip
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_zip",
    description=(
        "Zip and unzip file operations. Supports three actions:\n"
        "- 'list' — list files inside a zip archive with sizes.\n"
        "- 'extract' — extract a zip archive to a destination directory.\n"
        "- 'create' — create a zip archive from a list of file paths.\n"
        "Paths are restricted to data/ directory."
    ),
    category="utility",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — 'list', 'extract', or 'create'",
        "path": "str — path to zip file (required for 'list' and 'extract')",
        "dest": "str — extraction destination directory (required for 'extract')",
        "paths": "list[str] — list of file paths to zip (required for 'create')",
        "output": "str — output path for new zip file (required for 'create')",
    },
)
async def mcp_zip(
    action: str,
    path: str = "",
    dest: str = "",
    paths: list[str] | None = None,
    output: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Zip/unzip file operations tool.

    Args:
        action: ``"list"``, ``"extract"``, or ``"create"``.
        path: Path to a zip file (required for ``"list"`` and ``"extract"``).
        dest: Destination directory for extraction (required for ``"extract"``).
        paths: List of file paths to add to the archive (required for ``"create"``).
        output: Output path for the new zip file (required for ``"create"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "list":
            if not path or not path.strip():
                return {"error": "path parameter is required for action='list'"}
            return await _zip_list(path.strip())
        elif action == "extract":
            if not path or not path.strip():
                return {"error": "path parameter is required for action='extract'"}
            if not dest or not dest.strip():
                return {"error": "dest parameter is required for action='extract'"}
            return await _zip_extract(path.strip(), dest.strip())
        elif action == "create":
            if not paths or len(paths) == 0:
                return {"error": "paths parameter is required for action='create'"}
            if not output or not output.strip():
                return {"error": "output parameter is required for action='create'"}
            return await _zip_create(paths, output.strip())
        else:
            return {
                "error": (
                    f"Unknown action {action!r}. Valid actions: list, extract, create"
                )
            }
    except Exception as exc:
        logger.exception("mcp_zip(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _zip_list(file_path: str) -> dict[str, Any]:
    """List all files inside a zip archive with their sizes."""
    resolved = _safe_resolve(file_path)
    if resolved is None:
        return {
            "error": f"Path {file_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_file():
        return {"error": f"File not found: {resolved}"}

    loop = asyncio.get_running_loop()

    def _list_files() -> dict[str, Any]:
        total_uncompressed = 0
        total_compressed = 0
        entries: list[dict[str, Any]] = []
        with zipfile.ZipFile(str(resolved), "r") as zf:
            for info in zf.infolist():
                if info.filename.endswith("/"):
                    continue  # skip directories
                entries.append(
                    {
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "ratio_pct": (
                            round((1 - info.compress_size / info.file_size) * 100, 1)
                            if info.file_size > 0
                            else 0
                        ),
                    }
                )
                total_uncompressed += info.file_size
                total_compressed += info.compress_size
            return {
                "entries": entries,
                "count": len(entries),
                "total_uncompressed_bytes": total_uncompressed,
                "total_uncompressed_mb": round(total_uncompressed / (1024**2), 2),
                "total_compressed_bytes": total_compressed,
                "total_compressed_mb": round(total_compressed / (1024**2), 2),
            }

    try:
        result = await loop.run_in_executor(None, _list_files)
    except zipfile.BadZipFile as exc:
        return {"error": f"Invalid zip file: {exc}"}
    except Exception as exc:
        logger.warning("Zip list error: %s", exc)
        return {"error": f"Failed to list zip contents: {exc}"}

    return {"ok": True, "path": str(resolved), **result}


async def _zip_extract(file_path: str, dest: str) -> dict[str, Any]:
    """Extract a zip archive to *dest*."""
    resolved = _safe_resolve(file_path)
    if resolved is None:
        return {
            "error": f"Path {file_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_file():
        return {"error": f"File not found: {resolved}"}

    dest_resolved = _safe_resolve(dest)
    if dest_resolved is None:
        return {
            "error": f"Dest path {dest!r} is outside allowed directories or contains '..'"
        }

    loop = asyncio.get_running_loop()

    def _extract() -> dict[str, Any]:
        # Ensure destination exists
        dest_resolved.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(resolved), "r") as zf:
            # Security: prevent zip slip (path traversal via ../ in entry names)
            extracted_count = 0
            for info in zf.infolist():
                # Resolve and check destination is within target dir
                target_path = (dest_resolved / info.filename).resolve()
                try:
                    target_path.relative_to(dest_resolved)
                except ValueError:
                    logger.warning(
                        "Zip slip detected: %s resolves outside %s",
                        info.filename,
                        dest_resolved,
                    )
                    continue
                zf.extract(info, str(dest_resolved))
                extracted_count += 1

            return {
                "extracted_count": extracted_count,
                "destination": str(dest_resolved),
            }

    try:
        result = await loop.run_in_executor(None, _extract)
    except zipfile.BadZipFile as exc:
        return {"error": f"Invalid zip file: {exc}"}
    except Exception as exc:
        logger.warning("Zip extract error: %s", exc)
        return {"error": f"Failed to extract zip: {exc}"}

    return {"ok": True, **result}


async def _zip_create(source_paths: list[str], output_path: str) -> dict[str, Any]:
    """Create a zip archive from *source_paths*."""
    out_resolved = _safe_resolve(output_path)
    if out_resolved is None:
        return {
            "error": f"Output path {output_path!r} is outside allowed directories or contains '..'"
        }

    resolved_sources: list[Path] = []
    for raw in source_paths:
        r = _safe_resolve(raw)
        if r is None:
            return {
                "error": f"Source path {raw!r} is outside allowed directories or contains '..'"
            }
        if not r.exists():
            return {"error": f"Source path not found: {r}"}
        resolved_sources.append(r)

    loop = asyncio.get_running_loop()

    def _create() -> dict[str, Any]:
        # Ensure output directory exists
        out_resolved.parent.mkdir(parents=True, exist_ok=True)

        total_added = 0
        with zipfile.ZipFile(str(out_resolved), "w", zipfile.ZIP_DEFLATED) as zf:
            for src in resolved_sources:
                if src.is_file():
                    zf.write(str(src), arcname=src.name)
                    total_added += 1
                elif src.is_dir():
                    # Add all files in directory recursively
                    for fpath in src.rglob("*"):
                        if fpath.is_file():
                            arcname = str(fpath.relative_to(src.parent))
                            zf.write(str(fpath), arcname=arcname)
                            total_added += 1

        return {
            "output": str(out_resolved),
            "source_count": len(resolved_sources),
            "total_files_added": total_added,
            "size_bytes": out_resolved.stat().st_size,
            "size_mb": round(out_resolved.stat().st_size / (1024**2), 2),
        }

    try:
        result = await loop.run_in_executor(None, _create)
    except Exception as exc:
        logger.warning("Zip create error: %s", exc)
        return {"error": f"Failed to create zip: {exc}"}

    return {"ok": True, **result}
