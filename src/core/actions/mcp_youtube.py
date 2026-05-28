"""mcp_youtube tool — registered via @tool decorator.

Search and download YouTube content via yt-dlp.

Actions:
- ``action="search" query="..." limit=5`` — search YouTube, return results
- ``action="info" url="..."`` — get video metadata without downloading
- ``action="audio" url="..."`` — download audio-only, save to data/youtube/
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.core.actions.tool_registry import tool
from src.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_YOUTUBE_DIR: Path = settings.data_dir / "youtube"
_SEARCH_LIMIT_MAX = 20
_DOWNLOAD_TIMEOUT = 120  # seconds


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_youtube
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_youtube",
    description=(
        "Search and download YouTube content via yt-dlp. Supports three actions:\n"
        "- 'search' — search YouTube and return video metadata (title, url, duration, views, channel).\n"
        "- 'info' — get detailed metadata for a specific video URL.\n"
        "- 'audio' — download audio-only track from a video URL and save to data/youtube/."
    ),
    category="utility",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — 'search', 'info', or 'audio'",
        "query": "str — search query (required for action='search')",
        "url": "str — YouTube video URL (required for action='info' and 'audio')",
        "limit": "int — max search results (default 5, max 20, used with 'search')",
    },
)
async def mcp_youtube(
    action: str,
    query: str = "",
    url: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """YouTube search and download tool.

    Args:
        action: ``"search"``, ``"info"``, or ``"audio"``.
        query: Search query (required for ``action="search"``).
        url: YouTube video URL (required for ``action="info"`` and ``"audio"``).
        limit: Max search results (default 5, max 20, used with ``"search"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "search":
            if not query or not query.strip():
                return {"error": "query parameter is required for action='search'"}
            limit = max(1, min(limit, _SEARCH_LIMIT_MAX))
            return await _youtube_search(query.strip(), limit)
        elif action == "info":
            if not url or not url.strip():
                return {"error": "url parameter is required for action='info'"}
            return await _youtube_info(url.strip())
        elif action == "audio":
            if not url or not url.strip():
                return {"error": "url parameter is required for action='audio'"}
            return await _youtube_audio(url.strip())
        else:
            return {
                "error": f"Unknown action {action!r}. Valid actions: search, info, audio"
            }
    except Exception as exc:
        logger.exception("mcp_youtube(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


def _import_ytdlp() -> Any:
    """Lazy import of yt-dlp. Returns the module or raises ImportError."""
    try:
        import yt_dlp  # type: ignore[import-untyped]

        return yt_dlp
    except ImportError:
        raise ImportError("yt-dlp not installed: pip install yt-dlp")


def _sanitize_filename(name: str) -> str:
    """Remove or replace characters unsafe for filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:128] or "audio"


async def _youtube_search(query: str, limit: int) -> dict[str, Any]:
    """Search YouTube via ytsearch and return metadata for each result."""

    yt_dlp = _import_ytdlp()
    loop = asyncio.get_running_loop()

    def _search() -> list[dict[str, Any]]:
        search_query = f"ytsearch{limit}:{query}"
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "default_search": "ytsearch",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(search_query, download=False)
            except Exception as exc:
                logger.warning("yt-dlp search failed for %r: %s", query, exc)
                raise

            entries = info.get("entries", [])
            results = []
            for entry in entries[:limit]:
                if entry is None:
                    continue
                results.append(
                    {
                        "title": entry.get("title", ""),
                        "url": f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                        "duration": entry.get("duration"),
                        "views": entry.get("view_count"),
                        "channel": entry.get("channel", entry.get("uploader", "")),
                    }
                )
            return results

    try:
        results = await loop.run_in_executor(None, _search)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("yt-dlp search error: %s", exc)
        return {"error": f"YouTube search failed: {exc}"}

    return {
        "ok": True,
        "query": query,
        "results": results,
        "count": len(results),
    }


async def _youtube_info(url: str) -> dict[str, Any]:
    """Get detailed metadata for a video without downloading."""

    yt_dlp = _import_ytdlp()
    loop = asyncio.get_running_loop()

    def _info() -> dict[str, Any]:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as exc:
                logger.warning("yt-dlp info failed for %r: %s", url, exc)
                raise

            return {
                "title": info.get("title", ""),
                "url": info.get("webpage_url", url),
                "duration": info.get("duration"),
                "views": info.get("view_count"),
                "channel": info.get("channel", info.get("uploader", "")),
                "upload_date": info.get("upload_date", ""),
                "description": (info.get("description") or "")[:500],
                "tags": info.get("tags") or [],
                "categories": info.get("categories") or [],
            }

    try:
        metadata = await loop.run_in_executor(None, _info)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("yt-dlp info error: %s", exc)
        return {"error": f"Failed to get video info: {exc}"}

    return {"ok": True, **metadata}


async def _youtube_audio(url: str) -> dict[str, Any]:
    """Download audio-only track from *url*, save to data/youtube/."""

    yt_dlp = _import_ytdlp()
    loop = asyncio.get_running_loop()

    # Ensure output directory exists
    _YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)

    def _download() -> dict[str, Any]:
        outtmpl = str(_YOUTUBE_DIR / "%(title)s.%(ext)s")
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as exc:
                logger.warning("yt-dlp download failed for %r: %s", url, exc)
                raise

            # Determine the actual output file path
            title = info.get("title", "audio")
            safe_title = _sanitize_filename(title)
            # yt-dlp replaces spaces, but we look for the mp3
            for fname in os.listdir(str(_YOUTUBE_DIR)):
                if (
                    safe_title.lower() in fname.lower()
                    or title.lower() in fname.lower()
                ):
                    fp = _YOUTUBE_DIR / fname
                    if fp.suffix.lower() in (".mp3", ".m4a", ".webm", ".opus"):
                        return {
                            "title": title,
                            "path": str(fp),
                            "size_bytes": fp.stat().st_size,
                            "size_mb": round(fp.stat().st_size / (1024**2), 2),
                        }

            # Fallback: glob for any new file in the directory
            files = list(_YOUTUBE_DIR.iterdir())
            audio_files = [
                f
                for f in files
                if f.suffix.lower() in (".mp3", ".m4a", ".webm", ".opus")
            ]
            if audio_files:
                latest = max(audio_files, key=lambda f: f.stat().st_mtime)
                return {
                    "title": title,
                    "path": str(latest),
                    "size_bytes": latest.stat().st_size,
                    "size_mb": round(latest.stat().st_size / (1024**2), 2),
                }

            return {
                "title": title,
                "path": None,
                "size_bytes": 0,
                "size_mb": 0,
                "note": "File not found after download",
            }

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _download), timeout=_DOWNLOAD_TIMEOUT
        )
    except ImportError as exc:
        return {"error": str(exc)}
    except asyncio.TimeoutError:
        return {"error": "Audio download timed out"}
    except Exception as exc:
        logger.warning("yt-dlp audio download error: %s", exc)
        return {"error": f"Audio download failed: {exc}"}

    return {"ok": True, **result}
