"""mcp_http tool — registered via @tool decorator.

Executes arbitrary HTTP requests (GET / POST / PUT / DELETE) and returns
the response status code, response headers, and body (first 3000 chars).

Features:
- SSRF protection: blocks requests to localhost / 127.0.0.1 / 0.0.0.0.
- Custom headers via JSON string.
- Optional request body for POST / PUT.
- 10-second timeout.
- Graceful error handling for connection errors, timeouts, DNS failures.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
from typing import Any
from urllib.parse import urlparse

import requests

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_HTTP_TIMEOUT = 10  # seconds
_MAX_BODY_CHARS = 3000
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

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

_VALID_METHODS = frozenset({"GET", "POST", "PUT", "DELETE"})


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_http
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_http",
    description=(
        "Execute an HTTP request to an external API.  Supports GET, POST, "
        "PUT, and DELETE methods.  Returns the response status code, "
        "response headers, and the first 3000 characters of the body.\n"
        "SSRF protection is enabled — localhost / 127.0.0.1 / 0.0.0.0 are "
        "blocked."
    ),
    category="system",
    risk="medium",
    params={
        "method": "str — HTTP method: GET, POST, PUT, DELETE",
        "url": "str — full URL to call (must be http:// or https://)",
        "headers": "str | None — optional JSON string of extra headers",
        "body": "str | None — optional request body (JSON string) for POST/PUT",
    },
)
async def mcp_http(
    method: str,
    url: str,
    headers: str | None = None,
    body: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute an HTTP request to an external API.

    Args:
        method: ``"GET"``, ``"POST"``, ``"PUT"``, or ``"DELETE"``.
        url: The full URL (must start with ``http://`` or ``https://``).
        headers: Optional JSON string of extra request headers.
        body: Optional request body (plain text / JSON string).

    Returns:
        A dict with ``status_code``, ``headers``, and ``body`` on success,
        or an ``"error"`` key on failure.
    """
    try:
        return await _do_request(method, url, headers=headers, body=body)
    except Exception as exc:
        logger.exception("mcp_http(%s %s) failed unexpectedly", method, url)
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# Implementation
# ══════════════════════════════════════════════════════════════════════════


async def _do_request(
    method: str,
    url: str,
    *,
    headers: str | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    """Core request logic — runs ``requests`` in an executor thread."""
    # ── Normalise method ──────────────────────────────────────────────
    method = method.upper().strip()
    if method not in _VALID_METHODS:
        return {
            "error": (
                f"Invalid method {method!r}. "
                f"Valid methods: {', '.join(sorted(_VALID_METHODS))}"
            ),
        }

    # ── Validate URL ──────────────────────────────────────────────────
    if not url or not url.strip():
        return {"error": "url parameter is required"}

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    # ── SSRF protection ───────────────────────────────────────────────
    ssrf_error = _check_ssrf(url)
    if ssrf_error:
        return ssrf_error

    # ── Parse custom headers ──────────────────────────────────────────
    request_headers: dict[str, str] = {
        "User-Agent": _USER_AGENT,
        "Accept": "*/*",
    }

    if headers:
        if not isinstance(headers, str):
            return {"error": "headers must be a JSON string"}
        try:
            custom = json.loads(headers)
        except json.JSONDecodeError as exc:
            return {"error": f"headers is not valid JSON: {exc}"}

        if not isinstance(custom, dict):
            return {"error": "headers JSON must be an object (dict)"}

        # Merge — custom keys override defaults
        for k, v in custom.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return {
                    "error": (
                        f"Invalid header entry {k!r}: both key and value "
                        f"must be strings"
                    )
                }
        request_headers.update(custom)

    # ── Execute request (threaded via executor) ────────────────────────
    loop = asyncio.get_running_loop()

    def _do_http() -> dict[str, Any]:
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=request_headers,
                data=body,
                timeout=_HTTP_TIMEOUT,
                allow_redirects=False,
            )
        except requests.ConnectionError as exc:
            logger.warning("Connection error for %s %s: %s", method, url, exc)
            return {"error": f"Connection error: {exc}"}
        except requests.Timeout as exc:
            logger.warning("Timeout for %s %s: %s", method, url, exc)
            return {"error": f"Request timed out after {_HTTP_TIMEOUT}s"}
        except requests.RequestException as exc:
            logger.warning("Request failed for %s %s: %s", method, url, exc)
            return {"error": f"Request failed: {exc}"}

        # ── Build response ──────────────────────────────────────────
        status_code = resp.status_code
        response_headers = dict(resp.headers)
        body_text = resp.text[:_MAX_BODY_CHARS]
        truncated = len(resp.text) > _MAX_BODY_CHARS

        return {
            "ok": True,
            "status_code": status_code,
            "headers": response_headers,
            "body": body_text,
            "truncated": truncated,
            "total_chars": len(resp.text),
        }

    return await loop.run_in_executor(None, _do_http)


# ══════════════════════════════════════════════════════════════════════════
# SSRF guard
# ══════════════════════════════════════════════════════════════════════════


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
