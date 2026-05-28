"""mcp_oauth tool — registered via @tool decorator.

Provides the LLM with the ability to connect to external MCP servers
that require OAuth authentication (Linear, GitHub, Sentry, etc.).

Supports two actions:

- **connect** — run the full OAuth 2.1 + PKCE flow for a given server.
- **status** — list currently connected servers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.actions.tool_registry import tool
from src.core.actions.mcp_oauth import mcp_oauth, TOKENS_DIR

logger = logging.getLogger(__name__)


@tool(
    name="mcp_oauth",
    description=(
        "Connect to external MCP servers with OAuth.  Use when the owner wants "
        "to add a hosted service (Linear, Sentry, GitHub MCP, etc.).\n"
        "- 'connect' — starts the OAuth 2.1 + PKCE flow (opens a URL in the "
        "browser for authorisation).\n"
        "- 'status' — lists currently connected servers."
    ),
    category="system",
    risk="high",
    requires_confirmation=True,
    params={
        "action": "str — 'connect' to start OAuth flow or 'status' to list connected servers",
        "server_name": "str — unique name for this server (e.g. 'linear', 'github')",
        "server_url": "str — base URL of the MCP server (e.g. 'https://mcp.linear.app')",
    },
)
async def mcp_oauth_tool(
    action: str = "status",
    server_name: str = "",
    server_url: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Connect to or inspect external MCP servers that require OAuth.

    Args:
        action: ``"connect"`` or ``"status"``.
        server_name: Unique name for the server (required for ``connect``).
        server_url: Base URL of the MCP server (required for ``connect``).

    Returns:
        A dict with connection status or error information.
    """
    if action == "status":
        servers = sorted(Path(TOKENS_DIR).glob("*.json"))
        return {
            "ok": True,
            "connected": [s.stem for s in servers],
            "count": len(servers),
        }

    if action == "connect":
        if not server_name or not server_url:
            return {
                "error": "server_name and server_url are required for action='connect'",
            }
        return await mcp_oauth.connect(server_name, server_url)

    return {"error": f"Unknown action: {action!r}. Use 'connect' or 'status'."}
