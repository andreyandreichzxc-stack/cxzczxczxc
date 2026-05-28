"""mcp_network tool — registered via @tool decorator.

Network diagnostics tool.

Actions:
- ``action="ping" host="google.com" count=3`` — ICMP ping via subprocess
- ``action="dns" host="google.com"`` — DNS resolution (A + AAAA records)
- ``action="ports" host="localhost" ports=[80,443,3000]`` — TCP port check
- ``action="external_ip"`` — public IP address via api.ipify.org

All blocking operations are run in a thread-pool executor with timeout.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import subprocess
import sys
from typing import Any

from src.core.actions.tool_registry import ToolActionSpec, tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_PORT_TIMEOUT = 2  # seconds per port probe
_PING_TIMEOUT = 10  # seconds for ping command
_PING_COUNT_MAX = 10


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_network
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_network",
    description=(
        "Network diagnostics tool.  Supports four actions:\n"
        "- 'ping' — ICMP echo to a host (via system ping command).\n"
        "- 'dns' — DNS resolution for a host (A + AAAA records).\n"
        "- 'ports' — check TCP port connectivity on a host.\n"
        "- 'external_ip' — get the public IP address.\n"
        "All operations have timeouts and run in a background thread."
    ),
    category="system",
    risk="medium",
    requires_confirmation=True,
    actions={
        "ping": ToolActionSpec(
            name="ping",
            risk="medium",
            read_only=True,
            idempotent=True,
            requires_confirmation=False,
            open_world=True,
            user_content=True,
        ),
        "dns": ToolActionSpec(
            name="dns",
            risk="medium",
            read_only=True,
            idempotent=True,
            requires_confirmation=False,
            open_world=True,
            user_content=True,
        ),
        "ports": ToolActionSpec(
            name="ports",
            risk="medium",
            read_only=True,
            idempotent=True,
            requires_confirmation=False,
            open_world=True,
            user_content=True,
        ),
        "external_ip": ToolActionSpec(
            name="external_ip",
            risk="medium",
            read_only=True,
            idempotent=True,
            requires_confirmation=False,
            open_world=True,
            user_content=True,
        ),
    },
    params={
        "action": "str — 'ping', 'dns', 'ports', or 'external_ip'",
        "host": "str — hostname or IP (required for 'ping', 'dns', 'ports')",
        "count": "int — ping count (default 3, max 10, used with action='ping')",
        "ports": "list[int] — list of ports to check (required for action='ports')",
    },
)
async def mcp_network(
    action: str,
    host: str = "",
    count: int = 3,
    ports: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Network diagnostics tool.

    Args:
        action: ``"ping"``, ``"dns"``, ``"ports"``, or ``"external_ip"``.
        host: Hostname or IP (required for ``"ping"``, ``"dns"``, ``"ports"``).
        count: Ping count (default 3, max 10).
        ports: List of TCP ports to check (required for ``"ports"``).

    Returns:
        A dict with diagnostic results or an ``"error"`` key.
    """
    try:
        if action == "ping":
            if not host or not host.strip():
                return {"error": "host parameter is required for action='ping'"}
            host = host.strip()
            unsafe = _validate_public_host(host)
            if unsafe:
                return unsafe
            n = max(1, min(count, _PING_COUNT_MAX))
            return await _ping(host, n)
        elif action == "dns":
            if not host or not host.strip():
                return {"error": "host parameter is required for action='dns'"}
            host = host.strip()
            unsafe = _validate_public_host(host)
            if unsafe:
                return unsafe
            return await _dns(host)
        elif action == "ports":
            if not host or not host.strip():
                return {"error": "host parameter is required for action='ports'"}
            if not ports:
                return {"error": "ports parameter is required for action='ports'"}
            host = host.strip()
            unsafe = _validate_public_host(host)
            if unsafe:
                return unsafe
            return await _check_ports(host, ports)
        elif action == "external_ip":
            return await _external_ip()
        else:
            return {
                "error": f"Unknown action {action!r}. "
                f"Valid actions: ping, dns, ports, external_ip"
            }
    except Exception as exc:
        logger.exception("mcp_network(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


def _validate_public_host(host: str) -> dict[str, Any] | None:
    """Block network probes against local or private infrastructure."""
    if host.lower().strip("[]") in {"localhost", "localhost.localdomain"}:
        return {"error": f"Refusing to probe local/private host {host!r}"}

    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None

    if literal is None:
        try:
            raw_addresses = [info[4][0] for info in socket.getaddrinfo(host, None)]
        except socket.gaierror as exc:
            return {"error": f"DNS resolution failed for {host!r}: {exc}"}
    else:
        raw_addresses = [str(literal)]

    for raw_addr in raw_addresses:
        try:
            addr = ipaddress.ip_address(raw_addr)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return {
                "error": (
                    f"Refusing to probe local/private host {host!r} "
                    f"(resolved to {raw_addr!r})"
                )
            }

    return None


async def _ping(host: str, count: int) -> dict[str, Any]:
    """Ping *host* using the system ping command."""

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        # Windows uses -n, Unix uses -c
        is_windows = sys.platform == "win32"
        cmd = ["ping"]
        if is_windows:
            cmd.extend(["-n", str(count)])
        else:
            cmd.extend(["-c", str(count)])
        cmd.append(host)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=_PING_TIMEOUT,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"ping to {host} timed out after {_PING_TIMEOUT}s"}
        except FileNotFoundError:
            return {"error": "ping command not found on this system"}
        except OSError as exc:
            return {"error": f"ping failed: {exc}"}

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        # Parse success/failure from return code
        success = result.returncode == 0

        return {
            "ok": True,
            "host": host,
            "count": count,
            "success": success,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    return await loop.run_in_executor(None, _do)


async def _dns(host: str) -> dict[str, Any]:
    """Resolve *host* via socket.gethostbyname and getaddrinfo."""

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        # Basic A-record resolution
        try:
            ipv4 = socket.gethostbyname(host)
        except socket.gaierror as exc:
            return {"error": f"DNS resolution failed for {host!r}: {exc}"}

        # Get all address info (A + AAAA)
        addresses: list[dict[str, Any]] = []
        try:
            for info in socket.getaddrinfo(
                host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM
            ):
                family = "IPv4" if info[0] == socket.AF_INET else "IPv6"
                addr = info[4][0]
                addresses.append({"family": family, "address": addr})
        except socket.gaierror:
            pass

        return {
            "ok": True,
            "host": host,
            "ipv4": ipv4,
            "addresses": addresses,
            "count": len(addresses),
        }

    return await loop.run_in_executor(None, _do)


async def _check_ports(host: str, ports: list[int]) -> dict[str, Any]:
    """Check TCP port connectivity on *host*."""

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(_PORT_TIMEOUT)
                result = sock.connect_ex((host, port))
                sock.close()
                results.append(
                    {
                        "port": port,
                        "open": result == 0,
                    }
                )
            except socket.gaierror:
                results.append(
                    {
                        "port": port,
                        "open": False,
                        "error": "Host not found",
                    }
                )
            except OSError as exc:
                results.append(
                    {
                        "port": port,
                        "open": False,
                        "error": str(exc),
                    }
                )

        open_ports = [r["port"] for r in results if r.get("open")]

        return {
            "ok": True,
            "host": host,
            "results": results,
            "open_ports": open_ports,
            "open_count": len(open_ports),
            "total": len(results),
        }

    return await loop.run_in_executor(None, _do)


async def _external_ip() -> dict[str, Any]:
    """Get the public IP address via api.ipify.org."""

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        try:
            import urllib.request

            req = urllib.request.Request(
                "https://api.ipify.org",
                headers={"User-Agent": "curl/8.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                ip = resp.read().decode("utf-8").strip()
            return {
                "ok": True,
                "external_ip": ip,
                "source": "api.ipify.org",
            }
        except Exception as exc:
            return {"error": f"Failed to get external IP: {exc}"}

    return await loop.run_in_executor(None, _do)
