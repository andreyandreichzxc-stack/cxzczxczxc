#!/usr/bin/env python3
"""Entry point for MCP server.

Usage:
    python scripts/mcp_server.py <telegram_id>

Spawns an MCP protocol server over stdio that exposes a curated subset of
TelegramHelper tools to external AI agents (Claude Desktop, Cursor, OpenCode).

Example:
    python scripts/mcp_server.py 123456789

Configuration for Claude Desktop (claude_desktop_config.json)::

    {
      "mcpServers": {
        "telegram-helper": {
          "command": "python",
          "args": ["path/to/scripts/mcp_server.py", "123456789"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/mcp_server.py <telegram_id>",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        telegram_id = int(sys.argv[1])
    except ValueError:
        print(
            f"Error: telegram_id must be an integer, got {sys.argv[1]!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.core.mcp_server import run_stdio_server

    asyncio.run(run_stdio_server(telegram_id))


if __name__ == "__main__":
    main()
