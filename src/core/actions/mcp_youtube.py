"""mcp_youtube tool — registered via @tool decorator.

Search and download YouTube content via yt-dlp.

Actions:
- ``action="search" query="..." limit=5`` — search YouTube, return results
- ``action="info" url="..."`` — get video metadata without downloading
- ``action="audio" url="..."`` — download audio-only, save to data/youtube/
- ``action="subtitles" url="..." lang="ru"`` — fetch video subtitles/transcript as text
- ``action="comments" url="..." limit=10`` — fetch video comments
- ``action="summarize" url="..." lang="ru"`` — auto-summarize video via subtitles + LLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.core.actions.tool_registry import tool
from src.config import settings
from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_YOUTUBE_DIR: Path = settings.data_dir / "youtube"
_SEARCH_LIMIT_MAX = 20
_DOWNLOAD_TIMEOUT = 120  # seconds
_SUBTITLES_TIMEOUT = 60  # seconds
_COMMENTS_TIMEOUT = 30  # seconds
_COMMENTS_LIMIT_MAX = 50


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_youtube
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_youtube",
    description=(
        "Search and download YouTube content via yt-dlp. Supports six actions:\n"
        "- 'search' — search YouTube and return video metadata (title, url, duration, views, channel).\n"
        "- 'info' — get detailed metadata for a specific video URL.\n"
        "- 'audio' — download audio-only track from a video URL and save to data/youtube/.\n"
        "- 'subtitles' — fetch video subtitles/transcript (useful for content analysis).\n"
        "- 'comments' — fetch recent video comments.\n"
        "- 'summarize' — auto-summarize video via subtitles + LLM provider."
    ),
    category="utility",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — 'search', 'info', 'audio', 'subtitles', 'comments', or 'summarize'",
        "query": "str — search query (required for action='search')",
        "url": "str — YouTube video URL (required for all actions except 'search')",
        "limit": "int — max search results (default 5, max 20) or max comments (default 10, max 50)",
        "lang": "str — subtitle language code (default 'ru', used with 'subtitles' and 'summarize')",
    },
)
async def mcp_youtube(
    action: str,
    query: str = "",
    url: str = "",
    limit: int = 5,
    lang: str = "ru",
    **kwargs: Any,
) -> dict[str, Any]:
    """YouTube search, download, subtitles, comments, and summarization tool.

    Args:
        action: ``"search"``, ``"info"``, ``"audio"``, ``"subtitles"``, ``"comments"``, or ``"summarize"``.
        query: Search query (required for ``action="search"``).
        url: YouTube video URL (required for ``action="info"``, ``"audio"``, ``"subtitles"``, ``"comments"``, and ``"summarize"``).
        limit: Max search results (default 5, max 20) or max comments (default 10, max 50).
        lang: Subtitle language code (default ``"ru"``, used with ``"subtitles"`` and ``"summarize"``).

    Keyword Args:
        provider: LLM provider with a ``chat()`` method (injected at runtime, used by ``"summarize"``).

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
        elif action == "subtitles":
            if not url or not url.strip():
                return {"error": "url parameter is required for action='subtitles'"}
            return await _youtube_subtitles(url.strip(), lang.strip() or "ru")
        elif action == "comments":
            if not url or not url.strip():
                return {"error": "url parameter is required for action='comments'"}
            comment_limit = max(1, min(int(limit), _COMMENTS_LIMIT_MAX))
            return await _youtube_comments(url.strip(), comment_limit)
        elif action == "summarize":
            if not url or not url.strip():
                return {"error": "url parameter is required for action='summarize'"}
            return await _youtube_summarize(url.strip(), lang.strip() or "ru", kwargs)
        else:
            return {
                "error": f"Unknown action {action!r}. Valid actions: search, info, audio, subtitles, comments, summarize"
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
    # Also collapse path traversal sequences like ../ or ..\\
    sanitized = re.sub(r"\.\.+", "_", sanitized)
    return sanitized[:128] or "audio"


def _validate_youtube_url(url: str) -> str | None:
    """Return error message if *url* is not a YouTube URL, else None."""
    if not re.match(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", url):
        return "Only YouTube URLs are supported"
    return None


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

    url_err = _validate_youtube_url(url)
    if url_err:
        return {"error": url_err}

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

    url_err = _validate_youtube_url(url)
    if url_err:
        return {"error": url_err}

    yt_dlp = _import_ytdlp()
    loop = asyncio.get_running_loop()

    # Ensure output directory exists
    _YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)

    def _download() -> dict[str, Any]:
        # Step 1: Fetch video info without downloading to get a safe title
        # (fixes BUG 1 — path traversal via malicious video title)
        info_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        title: str = info.get("title", "audio")
        video_id: str = info.get("id", "")
        safe_title = _sanitize_filename(title)

        # Step 2: Clean up old audio files to prevent unbounded disk usage
        # (fixes BUG 4 — infinite accumulation)
        for old_file in _YOUTUBE_DIR.iterdir():
            if old_file.suffix.lower() in (".mp3", ".m4a", ".webm", ".opus"):
                try:
                    old_file.unlink()
                except OSError:
                    pass

        # Step 3: Download with a sanitized filename that includes video_id
        # (fixes BUG 3 — race condition in parallel downloads)
        stem = f"{video_id}_{safe_title}" if video_id else safe_title
        outtmpl = str(_YOUTUBE_DIR / f"{stem}.%(ext)s")

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

            # Determine the actual output file path.
            # Filter by video_id first (BUG 3 fix), then match by title.
            for fname in os.listdir(str(_YOUTUBE_DIR)):
                fp = _YOUTUBE_DIR / fname
                if fp.suffix.lower() not in (".mp3", ".m4a", ".webm", ".opus"):
                    continue
                if video_id and video_id not in fp.stem:
                    continue
                if safe_title.lower() in fp.stem.lower():
                    return {
                        "title": title,
                        "path": str(fp),
                        "size_bytes": fp.stat().st_size,
                        "size_mb": round(fp.stat().st_size / (1024**2), 2),
                    }

            # Fallback: any audio file whose stem contains video_id
            audio_files = [
                f
                for f in _YOUTUBE_DIR.iterdir()
                if f.suffix.lower() in (".mp3", ".m4a", ".webm", ".opus")
            ]
            if video_id:
                audio_files = [f for f in audio_files if video_id in f.stem]

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


# ══════════════════════════════════════════════════════════════════════════
# Action: subtitles
# ══════════════════════════════════════════════════════════════════════════


async def _youtube_subtitles(url: str, lang: str) -> dict[str, Any]:
    """Fetch subtitles/transcript from a YouTube video via yt-dlp CLI.

    Downloads auto-generated or manual subtitles, converts to SRT,
    parses the text, and returns plain transcript.
    """
    url_err = _validate_youtube_url(url)
    if url_err:
        return {"error": url_err}

    loop = asyncio.get_running_loop()

    def _fetch() -> dict[str, Any]:
        workdir = tempfile.mkdtemp()
        try:
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-lang",
                lang,
                "--convert-subs",
                "srt",
                "--print",
                "after_move:filepath",
                "-o",
                "%(id)s.%(ext)s",
                url,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SUBTITLES_TIMEOUT,
                cwd=workdir,
            )
            if result.returncode != 0:
                stderr_tail = result.stderr[-500:] if result.stderr else "(empty)"
                logger.warning(
                    "yt-dlp subtitles failed for %r (lang=%s): %s",
                    url,
                    lang,
                    stderr_tail,
                )
                return {"error": f"yt-dlp failed: {stderr_tail}"}

            # Find the SRT file in the workdir
            srt_files = [f for f in os.listdir(workdir) if f.endswith(".srt")]
            if not srt_files:
                return {"error": f"No subtitles found for lang={lang}"}

            srt_path = os.path.join(workdir, srt_files[0])
            with open(srt_path, "r", encoding="utf-8") as f:
                raw = f.read()

            # Parse SRT: skip index lines (short digit-only ≤4 chars),
            # timestamps, blank lines; keep only the spoken text lines
            lines = []
            for line in raw.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                # BUG 2 fix: only skip short digit-only strings (real
                # subtitle indices are ≤4 chars); keep numeric dialogue
                if stripped.isdigit() and len(stripped) <= 4:
                    continue
                if "-->" in stripped:
                    continue
                # Skip HTML-like tags often present in auto-subs
                if stripped.startswith("<") and stripped.endswith(">"):
                    continue
                lines.append(stripped)

            text = " ".join(lines)
            return {
                "ok": True,
                "lang": lang,
                "text": text[:4000],
                "full_length": len(raw),
            }
        except subprocess.TimeoutExpired:
            return {"error": "yt-dlp subtitles timed out after 60s"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    try:
        result = await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        logger.warning("YouTube subtitles error: %s", exc)
        return {"error": f"Subtitles fetch failed: {exc}"}

    if "error" in result:
        return result
    return result


# ══════════════════════════════════════════════════════════════════════════
# Action: comments
# ══════════════════════════════════════════════════════════════════════════


async def _youtube_comments(url: str, limit: int) -> dict[str, Any]:
    """Fetch recent comments from a YouTube video via yt-dlp CLI."""
    url_err = _validate_youtube_url(url)
    if url_err:
        return {"error": url_err}

    loop = asyncio.get_running_loop()

    def _fetch() -> dict[str, Any]:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-comments",
            "--print",
            "%(comments)j",
            "--max-comments",
            str(limit),
            url,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_COMMENTS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"error": "yt-dlp comments timed out after 30s"}

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "(empty)"
            logger.warning(
                "yt-dlp comments failed for %r: %s",
                url,
                stderr_tail,
            )
            return {"error": f"yt-dlp failed: {stderr_tail}"}

        # yt-dlp --print %(comments)j outputs JSON array of comment objects
        try:
            parsed = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            # Fallback: treat each non-empty line as a comment text
            parsed = [
                {"text": line.strip()}
                for line in result.stdout.split("\n")
                if line.strip()
            ]

        comments = []
        for c in parsed[:limit]:
            if isinstance(c, dict):
                text = c.get("text", str(c))
            else:
                text = str(c)
            # Truncate each comment to 300 chars
            comments.append(text[:300] if text else "")

        return {
            "ok": True,
            "count": len(comments),
            "comments": comments,
        }

    try:
        result = await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        logger.warning("YouTube comments error: %s", exc)
        return {"error": f"Comments fetch failed: {exc}"}

    if "error" in result:
        return result
    return result


# ══════════════════════════════════════════════════════════════════════════
# Action: summarize
# ══════════════════════════════════════════════════════════════════════════


async def _youtube_summarize(
    url: str, lang: str, extra_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Auto-summarize a YouTube video via its subtitles and an LLM provider.

    Fetches subtitles first, then asks the injected LLM provider to
    summarise the transcript.  If no provider is available, returns the
    raw transcript for manual summarisation.
    """
    sub_result = await _youtube_subtitles(url, lang)
    if "error" in sub_result:
        return sub_result

    transcript = sub_result.get("text", "")
    if not transcript:
        return {"error": "Transcript is empty, cannot summarize"}

    provider = extra_kwargs.get("provider")
    if provider is None:
        # No LLM available — return the transcript with a note
        return {
            "ok": True,
            "transcript": transcript[:3000],
            "note": "No LLM provider available for summarization; transcript returned",
            "source": url,
        }

    prompt = (
        f"Суммаризируй транскрипт видео. Выдели ключевые темы, факты и выводы. "
        f"Ответь на русском, кратко (3-5 предложений).\n\nТранскрипт:\n{transcript[:3000]}"
    )

    try:
        summary = await provider.chat([ChatMessage(role="user", content=prompt)])
    except Exception as exc:
        logger.warning("YouTube summarize LLM call failed: %s", exc)
        return {"error": f"LLM summarization failed: {exc}"}

    if not summary or not summary.strip():
        return {"error": "LLM returned empty summary"}

    return {
        "ok": True,
        "summary": summary.strip(),
        "source": url,
        "lang": lang,
    }
