"""mcp_web tool — registered via @tool decorator.

Provides web search and page fetching capabilities:

- **search** — query DuckDuckGo for web results (title, url, snippet).
- **fetch** — retrieve and extract clean text from a URL (first 2000 chars).

Uses ``requests`` + ``BeautifulSoup`` for both actions.  No external MCP
server is required — the tool works purely over HTTP.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

_DDG_HTML_URL = "https://html.duckduckgo.com/html"
_FETCH_TIMEOUT = 10  # seconds
_MAX_FETCH_CHARS = 2000
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_REQUEST_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ── SSRF blocklist ───────────────────────────────────────────────────────

_SSRF_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "[::1]",
        "169.254.169.254",
    }
)


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_web
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_web",
    description=(
        "Search the web or fetch the contents of a URL.  Supports two actions:\n"
        "- 'search' — query DuckDuckGo for web results, returns list of {title, url, snippet}\n"
        "- 'fetch' — download a page and extract clean text (first 2000 chars)"
    ),
    category="search",
    risk="medium",
    params={
        "action": "str — 'search' or 'fetch'",
        "query": "str — search query (required for action='search')",
        "url": "str — page URL (required for action='fetch')",
        "max_results": "int — max results for search (default 5)",
    },
)
async def mcp_web(
    action: str,
    query: str = "",
    url: str = "",
    max_results: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Web search and page-fetching tool.

    Args:
        action: ``"search"`` or ``"fetch"``.
        query: Search query (required when ``action="search"``).
        url: Page URL (required when ``action="fetch"``).
        max_results: Maximum number of search results to return.

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "search":
            return await _web_search(query, max_results=max_results)
        elif action == "fetch":
            return await _web_fetch(url)
        else:
            return {"error": f"Unknown action {action!r}. Valid actions: search, fetch"}
    except Exception as exc:
        logger.exception("mcp_web(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search DuckDuckGo for *query* and return result items.

    Uses the DuckDuckGo ``/html`` endpoint and parses the HTML response
    with BeautifulSoup.  This works without any API key.
    """
    if not query or not query.strip():
        return {"error": "query parameter is required for action='search'"}

    # Clamp to a reasonable range
    max_results = max(1, min(max_results, 50))

    loop = asyncio.get_running_loop()

    def _do_search() -> list[dict[str, str]]:
        try:
            resp = requests.get(
                _DDG_HTML_URL,
                params={"q": query.strip()},
                headers=_REQUEST_HEADERS,
                timeout=_FETCH_TIMEOUT,
            )
            resp.raise_for_status()
            # Validate final URL after redirects — must stay on DuckDuckGo
            if not resp.url.startswith("https://html.duckduckgo.com/"):
                logger.warning(
                    "DDG search redirected to unexpected domain: %s", resp.url
                )
                raise requests.RequestException(
                    f"Redirected to untrusted domain: {resp.url}"
                )
        except requests.RequestException as exc:
            logger.warning("DuckDuckGo search request failed: %s", exc)
            raise  # re-raised in the async wrapper below

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict[str, str]] = []

        # DuckDuckGo HTML results are in <a class="result__a"> with
        # sibling <a class="result__snippet"> for the description.
        for result_div in soup.select(".result"):
            if len(results) >= max_results:
                break

            link_el = result_div.select_one(".result__a")
            snippet_el = result_div.select_one(".result__snippet")

            if link_el is None:
                continue

            title = link_el.get_text(strip=True)
            raw_href = link_el.get("href")
            href: str = raw_href if isinstance(raw_href, str) else ""

            # DuckDuckGo wraps real URLs — try to extract from ``uddg=``
            # or ``//duckduckgo.com/l/?uddg=`` redirect.
            actual_url = _extract_ddg_url(href)

            snippet = ""
            if snippet_el is not None:
                snippet = snippet_el.get_text(strip=True)

            if title and actual_url:
                results.append(
                    {
                        "title": title,
                        "url": actual_url,
                        "snippet": snippet,
                    }
                )

        return results

    try:
        results = await loop.run_in_executor(None, _do_search)
    except requests.RequestException as exc:
        return {"error": f"Search request failed: {exc}"}

    return {
        "ok": True,
        "query": query.strip(),
        "results": results,
        "count": len(results),
    }


async def _web_fetch(url: str) -> dict[str, Any]:
    """Fetch *url*, extract clean text, return first 2000 characters."""
    if not url or not url.strip():
        return {"error": "url parameter is required for action='fetch'"}

    # Basic URL sanity check
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    # ── SSRF protection ───────────────────────────────────────────────
    ssrf_error = _check_ssrf(url)
    if ssrf_error:
        return ssrf_error

    loop = asyncio.get_running_loop()

    def _do_fetch() -> tuple[str, int]:
        try:
            resp = requests.get(
                url,
                headers=_REQUEST_HEADERS,
                timeout=_FETCH_TIMEOUT,
                allow_redirects=False,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Fetch request failed for %r: %s", url, exc)
            raise

        # Determine encoding from headers or content, fallback to utf-8
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script / style elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        total_len = len(text)
        return text[:_MAX_FETCH_CHARS], total_len

    try:
        content, total_len = await loop.run_in_executor(None, _do_fetch)
    except requests.RequestException as exc:
        return {"error": f"Failed to fetch URL: {exc}"}
    except Exception as exc:
        logger.exception("Unexpected error fetching %r", url)
        return {"error": f"Unexpected error: {exc}"}

    return {
        "ok": True,
        "url": url,
        "content": content,
        "truncated": total_len > _MAX_FETCH_CHARS,
        "total_chars": total_len,
    }


# ── SSRF guard ─────────────────────────────────────────────────────────


def _check_ssrf(url: str) -> dict[str, Any] | None:
    """Return an error dict if *url* targets a blocked host, else ``None``.

    Resolves the hostname to an IP first (prevents DNS rebinding attacks),
    then checks against blocklists for:
    - ``localhost`` and all variants (``127.0.0.1``, ``0.0.0.0``, ``::1``).
    - Private / link-local / reserved IP ranges (``10.x.x.x``,
      ``172.16-31.x.x``, ``192.168.x.x``, ``169.254.x.x``,
      ``127.x.x.x``, ``255.255.255.255``).
    - IPv6 loopback (``::1``), link-local (``fe80::/10``), ULA (``fc00::/7``).
    - AWS metadata endpoint (``169.254.169.254``).
    - IPv4-mapped IPv6 addresses that resolve to private IPv4.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return {"error": f"Cannot parse URL: {exc}"}

    hostname = parsed.hostname or ""

    # Direct match against blocklist
    if hostname.lower() in _SSRF_BLOCKED_HOSTS:
        return {
            "error": (
                f"SSRF protection: requests to {hostname!r} are not allowed. "
                f"Use an external URL instead."
            )
        }

    # Resolve DNS first — prevents rebinding attacks
    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return {"error": f"SSRF protection: cannot resolve hostname {hostname!r}."}

    # Check resolved IP against blocklist
    if ip in _SSRF_BLOCKED_HOSTS:
        return {
            "error": (
                f"SSRF protection: requests to {hostname!r} (resolved to {ip!r}) "
                f"are not allowed."
            )
        }

    # 127.x.x.x range (common loopback)
    if ip.startswith("127.") or ip == "255.255.255.255":
        return {
            "error": (
                f"SSRF protection: requests to {hostname!r} (resolved to {ip!r}) "
                f"are not allowed."
            )
        }

    # Use ipaddress module for thorough checking
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return {"error": f"SSRF protection: invalid IP {ip!r} for {hostname!r}."}

    if addr.version == 4:
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return {
                "error": (
                    f"SSRF protection: requests to {hostname!r} "
                    f"(resolved to private IP {ip!r}) are not allowed."
                )
            }
    elif addr.version == 6:
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return {
                "error": (
                    f"SSRF protection: requests to {hostname!r} "
                    f"(resolved to private IPv6 {ip!r}) are not allowed."
                )
            }
        # Check IPv4-mapped IPv6 addresses
        try:
            v6 = ipaddress.IPv6Address(ip)
            if v6.ipv4_mapped:
                mapped = v6.ipv4_mapped
                if mapped.is_private or mapped.is_loopback:
                    return {
                        "error": (
                            f"SSRF protection: requests to {hostname!r} "
                            f"(resolved to mapped IPv4 {mapped}) are not allowed."
                        )
                    }
        except (ValueError, AttributeError):
            pass

    return None


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_ddg_url(href: str) -> str:
    """Extract the real URL from a DuckDuckGo redirect link.

    DuckDuckGo HTML search results wrap external URLs in a redirect like::

        //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...

    For direct links the ``href`` may already be usable.
    """
    if "uddg=" in href:
        match = re.search(r"uddg=([^&]+)", href)
        if match:
            from urllib.parse import unquote

            return unquote(match.group(1))
    # Some results have direct URLs
    if href.startswith("http://") or href.startswith("https://"):
        return href
    # Fallback: return as-is (may be a relative path)
    return href
