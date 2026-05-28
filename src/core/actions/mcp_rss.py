"""mcp_rss tool — registered via @tool decorator.

Read RSS/Atom feeds.

Actions:
- ``action="read" url="https://feeds.example.com/rss" limit=10``
    — fetch and parse the feed, return entries with title, link, published,
      summary (HTML stripped).
- ``action="discover" url="https://site.com"``
    — try to find an RSS/Atom feed link in the HTML ``<head>`` of a page.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_FETCH_TIMEOUT = 10  # seconds
_DISCOVER_TIMEOUT = 10  # seconds
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_rss
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_rss",
    description=(
        "Read RSS/Atom feeds. Two actions:\n"
        "- 'read' — fetch and parse a feed URL, return entries with title, "
        "link, published, summary.\n"
        "- 'discover' — scan a website's HTML <head> for RSS/Atom feed links."
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'read' or 'discover'",
        "url": "str — feed or site URL",
        "limit": "int — max entries to return (default 10, max 50, used with 'read')",
    },
)
async def mcp_rss(
    action: str = "",
    url: str = "",
    limit: int = _DEFAULT_LIMIT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read RSS/Atom feeds.

    Args:
        action: ``"read"`` or ``"discover"``.
        url: Feed URL (for ``"read"``) or website URL (for ``"discover"``).
        limit: Max entries (1–50, default 10, used with ``"read"``).

    Returns:
        A dict with ``"entries"`` (list of dicts) or ``"error"``.
    """
    try:
        if action not in ("read", "discover"):
            return {"error": f"Unknown action {action!r}. Valid: read, discover"}

        if not url or not url.strip():
            return {"error": "url parameter is required"}

        url = url.strip()
        limit = max(1, min(limit, _MAX_LIMIT))

        if action == "read":
            return await _read_feed(url, limit)
        else:
            return await _discover_feed(url)

    except Exception as exc:
        logger.exception("mcp_rss(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Implementation
# ══════════════════════════════════════════════════════════════════════════


def _strip_html(text: str | None) -> str:
    """Remove HTML tags from *text* and collapse whitespace."""
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


async def _read_feed(url: str, limit: int) -> dict[str, Any]:
    """Fetch and parse an RSS/Atom feed."""
    loop = asyncio.get_running_loop()

    def _fetch() -> list[dict[str, Any]]:
        # Lazy import — feedparser is an optional dependency
        try:
            import feedparser  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("feedparser not installed: pip install feedparser")

        import requests  # noqa: PLC0415 — nested import

        headers = {"User-Agent": _USER_AGENT}
        try:
            resp = requests.get(url, headers=headers, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            raise ValueError(f"Failed to fetch feed: {exc}")

        try:
            parsed = feedparser.parse(resp.content)
        except Exception as exc:
            raise ValueError(f"Failed to parse feed: {exc}")

        if parsed.bozo and not parsed.entries:
            # bozo_exception may provide details about malformed XML
            detail = ""
            if parsed.bozo_exception:
                detail = f" ({parsed.bozo_exception})"
            raise ValueError(
                f"Feed appears to be invalid or non-RSS/Atom content{detail}"
            )

        entries = []
        for entry in parsed.entries[:limit]:
            entries.append(
                {
                    "title": getattr(entry, "title", None) or "",
                    "link": getattr(entry, "link", None) or "",
                    "published": (
                        getattr(entry, "published", None)
                        or getattr(entry, "updated", None)
                        or ""
                    ),
                    "summary": _strip_html(
                        getattr(entry, "summary", None)
                        or getattr(entry, "description", None)
                        or ""
                    ),
                }
            )

        return entries

    try:
        entries = await loop.run_in_executor(None, _fetch)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Feed read error: %s", exc)
        return {"error": f"Failed to read feed: {exc}"}

    if not entries:
        return {
            "ok": True,
            "entries": [],
            "count": 0,
            "note": "No entries found in feed",
        }

    return {
        "ok": True,
        "entries": entries,
        "count": len(entries),
    }


async def _discover_feed(url: str) -> dict[str, Any]:
    """Scan a website's HTML <head> for RSS/Atom feed links."""
    loop = asyncio.get_running_loop()

    def _discover() -> list[dict[str, str]]:
        try:
            import requests  # noqa: PLC0415 — nested import
            from bs4 import BeautifulSoup  # noqa: PLC0415 — nested import
        except ImportError:
            raise ImportError(
                "requests and beautifulsoup4 are required for feed discovery: "
                "pip install requests beautifulsoup4"
            )

        headers = {"User-Agent": _USER_AGENT}
        try:
            resp = requests.get(url, headers=headers, timeout=_DISCOVER_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            raise ValueError(f"Failed to fetch page: {exc}")

        # Detect encoding from headers/content
        resp.encoding = resp.apparent_encoding

        soup = BeautifulSoup(resp.text, "html.parser")
        feed_links: list[dict[str, str]] = []

        # Look for <link> tags with RSS/Atom types
        for link_tag in soup.head.find_all("link") if soup.head else []:
            link_type = (link_tag.get("type") or "").lower()
            link_title = link_tag.get("title") or ""
            link_href = link_tag.get("href") or ""

            if not link_href:
                continue

            if "application/rss+xml" in link_type:
                feed_links.append(
                    {
                        "title": link_title or "RSS Feed",
                        "href": link_href,
                        "type": "rss",
                    }
                )
            elif "application/atom+xml" in link_type:
                feed_links.append(
                    {
                        "title": link_title or "Atom Feed",
                        "href": link_href,
                        "type": "atom",
                    }
                )

        return feed_links

    try:
        feeds = await loop.run_in_executor(None, _discover)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Feed discovery error: %s", exc)
        return {"error": f"Failed to discover feeds: {exc}"}

    if not feeds:
        return {
            "ok": True,
            "feeds": [],
            "count": 0,
            "note": "No RSS/Atom feed links found on this page",
        }

    return {
        "ok": True,
        "feeds": feeds,
        "count": len(feeds),
        "source_url": url,
    }
