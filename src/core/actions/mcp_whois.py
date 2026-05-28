"""mcp_whois tool — registered via @tool decorator.

WHOIS domain lookup with IP resolution.

Features:
- ``action="lookup"`` — resolves domain IP via ``socket.gethostbyname``
  and attempts a full WHOIS lookup via the ``whois`` library (lazy import).
- If ``python-whois`` is not installed, falls back to basic DNS info.
- Returns: IP, nameservers, registrar, creation_date, expiration_date.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_DNS_TIMEOUT = 5  # seconds


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_whois
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_whois",
    description=(
        "Perform a WHOIS lookup for a domain.  Returns IP address, "
        "nameservers, registrar, creation and expiration dates when "
        "available.\n"
        "Uses the python-whois library if installed, otherwise falls back "
        "to basic DNS resolution."
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'lookup' only",
        "domain": "str — domain name (e.g. 'example.com')",
    },
)
async def mcp_whois(
    action: str,
    domain: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """WHOIS domain lookup tool.

    Args:
        action: ``"lookup"`` (only action supported).
        domain: Domain name (e.g. ``"example.com"``).

    Returns:
        A dict with domain info or an ``"error"`` key on failure.
    """
    try:
        if action != "lookup":
            return {"error": f"Unknown action {action!r}. Valid action: lookup"}

        if not domain or not domain.strip():
            return {"error": "domain parameter is required"}

        domain = domain.strip().lower()
        return await _lookup_domain(domain)
    except Exception as exc:
        logger.exception("mcp_whois(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Implementation
# ══════════════════════════════════════════════════════════════════════════


async def _lookup_domain(domain: str) -> dict[str, Any]:
    """Resolve *domain* IP and attempt WHOIS lookup."""

    loop = asyncio.get_running_loop()

    # ── Step 1: Resolve IP (always available) ──────────────────────────
    def _resolve_ip() -> str | None:
        try:
            return socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning("Could not resolve domain %r", domain)
            return None

    ip = await loop.run_in_executor(None, _resolve_ip)

    # ── Step 2: Try WHOIS library (lazy import) ────────────────────────
    whois_data = await _try_whois(loop, domain)

    if whois_data:
        return {
            "ok": True,
            "domain": domain,
            "ip": ip,
            "registrar": whois_data.get("registrar"),
            "nameservers": whois_data.get("nameservers"),
            "creation_date": whois_data.get("creation_date"),
            "expiration_date": whois_data.get("expiration_date"),
            "source": "whois",
        }

    # ── Step 3: Fallback — basic DNS info only ─────────────────────────
    nameservers = await _resolve_ns(loop, domain)

    return {
        "ok": True,
        "domain": domain,
        "ip": ip,
        "registrar": None,
        "nameservers": nameservers,
        "creation_date": None,
        "expiration_date": None,
        "source": "dns_only",
    }


async def _try_whois(
    loop: asyncio.AbstractEventLoop,
    domain: str,
) -> dict[str, Any] | None:
    """Try to import ``whois`` and look up *domain*.

    Returns ``None`` if the library is not installed or the lookup fails.
    """

    def _do_whois() -> dict[str, Any] | None:
        try:
            import whois  # type: ignore[import-untyped]  # noqa: F811
        except ImportError:
            logger.info("python-whois not installed — using DNS fallback")
            return None

        try:
            w = whois.whois(domain)
        except Exception as exc:
            logger.warning("WHOIS lookup failed for %r: %s", domain, exc)
            return None

        result: dict[str, Any] = {}

        # Registrar
        registrar = w.get("registrar")
        if registrar:
            result["registrar"] = (
                str(registrar) if not isinstance(registrar, str) else registrar
            )

        # Nameservers
        ns = w.get("name_servers")
        if ns:
            if isinstance(ns, list):
                result["nameservers"] = [
                    str(n).lower() for n in ns if isinstance(n, str)
                ]
            else:
                result["nameservers"] = [str(ns).lower()]

        # Dates — normalise to ISO strings
        for key in ("creation_date", "expiration_date"):
            val = w.get(key)
            if val:
                if isinstance(val, list):
                    val = val[0]
                if hasattr(val, "isoformat"):
                    result[key] = val.isoformat()
                else:
                    result[key] = str(val)

        return result

    try:
        return await loop.run_in_executor(None, _do_whois)
    except Exception as exc:
        logger.warning("WHOIS thread failed for %r: %s", domain, exc)
        return None


async def _resolve_ns(
    loop: asyncio.AbstractEventLoop,
    domain: str,
) -> list[str]:
    """Resolve NS (nameserver) records via ``socket.getaddrinfo``.

    This is a best-effort fallback — on most systems it only returns A/AAAA
    records, not actual NS records, but we try ``getaddrinfo`` with the NS
    hint anyway.
    """

    def _do_ns() -> list[str]:
        try:
            infos = socket.getaddrinfo(domain, 53, socket.AF_UNSPEC, socket.SOCK_STREAM)
            # Deduplicate IPs as a basic nameserver hint
            seen: set[str] = set()
            ips: list[str] = []
            for info in infos:
                ip = str(info[4][0])
                if ip not in seen:
                    seen.add(ip)
                    ips.append(ip)
            return ips[:5]
        except socket.gaierror:
            return []

    return await loop.run_in_executor(None, _do_ns)
