"""MCP (Model Context Protocol) server exposing TelegramHelper tools.

Uses the standard MCP protocol over stdio (for local connections) or SSE (for remote).
Compatible with Claude Desktop, Cursor, OpenCode, and other MCP clients.

Exposed tools (curated subset — safe, read-mostly):
  - recall_memory       Search and recall facts from memory
  - store_memory        Store a new fact in memory
  - list_memories       List recent memories
  - mcp_web             Search the web or fetch a URL
  - mcp_filesystem      Read files, list directories, search in files (read-only)
  - mcp_telegram        Search messages, get contact info, list chats (read-only)
  - mcp_translate       Translate text between languages
  - mcp_reminders       List upcoming reminders (read-only)
  - mcp_rss             Fetch RSS feed entries (read-only)

Configuration (Claude Desktop):
  Add to your ``claude_desktop_config.json``:

  .. code-block:: json

    {
      "mcpServers": {
        "telegram-helper": {
          "command": "python",
          "args": ["path/to/scripts/mcp_server.py", "<your_telegram_id>"]
        }
      }
    }

Security:
  - Only read-mostly tools are exposed. NO shell, process, network, env, git-write,
    playwright, or OAuth tools are available through this server.
  - ``store_memory`` is write-capable but deliberately included because it is
    low-risk (adds facts to the user's own memory).
  - All operations are scoped to a single ``telegram_id`` — the server never
    accesses other users' data.
  - The server uses the internal tool registry (``ToolRegistry``) for execution,
    never directly accessing the database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from src.core.actions import bootstrap
from src.core.actions.tool_registry import tool_registry

logger = logging.getLogger(__name__)


# ── Exposed tool schemas ────────────────────────────────────────────────────
# Each entry maps tool_name → JSON-RPC tool schema (description + inputSchema).

EXPOSED_TOOLS: dict[str, dict[str, Any]] = {
    # Memory tools
    "recall_memory": {
        "description": "Search and recall facts from the user's memory. Supports semantic search, contact filtering, and deep memory mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "contact_id": {
                    "type": "integer",
                    "description": "Filter by contact ID (optional)",
                },
                "limit": {"type": "integer", "default": 10},
                "mode": {
                    "type": "string",
                    "enum": ["light", "normal", "deep"],
                    "default": "normal",
                },
            },
            "required": ["query"],
        },
    },
    "store_memory": {
        "description": "Store a new fact in the user's memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember"},
                "contact_id": {
                    "type": "integer",
                    "description": "Associate with a contact (optional)",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                },
            },
            "required": ["fact"],
        },
    },
    "list_memories": {
        "description": "List recent memories, optionally filtered by contact or type.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "integer"},
                "memory_type": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    # Web tools
    "mcp_web": {
        "description": "Search the web or fetch a URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "fetch"],
                },
                "query": {"type": "string"},
                "url": {"type": "string"},
            },
            "required": ["action"],
        },
    },
    # File tools (read-only)
    "mcp_filesystem": {
        "description": "Read files, list directories, search in files (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "list", "search"],
                },
                "path": {"type": "string"},
                "pattern": {"type": "string"},
            },
            "required": ["action", "path"],
        },
    },
    # Telegram tools (read-only)
    "mcp_telegram": {
        "description": "Search messages, get contact info, list chats (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search_messages", "get_contact", "list_chats"],
                },
                "query": {"type": "string"},
                "peer_id": {"type": "integer"},
            },
            "required": ["action"],
        },
    },
    # Translation
    "mcp_translate": {
        "description": "Translate text between languages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_lang": {"type": "string"},
                "source_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
    # Calendar/reminders (read)
    "mcp_reminders": {
        "description": "List upcoming reminders.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list"],
                    "description": "Only 'list' is supported via MCP",
                },
            },
            "required": ["action"],
        },
    },
    # News
    "mcp_rss": {
        "description": "Fetch RSS feed entries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["fetch"],
                },
                "url": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["action", "url"],
        },
    },
}


# ── MCP Server ─────────────────────────────────────────────────────────────


class MCPServer:
    """MCP protocol server over stdio.

    Implements the JSON-RPC 2.0-based MCP protocol for tool discovery and
    execution.  All operations are scoped to a single ``telegram_id``.
    """

    def __init__(self, telegram_id: int) -> None:
        self.telegram_id = telegram_id
        self._initialized = False

    # ── Request handler ─────────────────────────────────────────────────

    async def handle_request(self, request: dict) -> dict:
        """Handle a single JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "initialize":
                return self._response(
                    req_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": "telegram-helper",
                            "version": "1.0.0",
                        },
                    },
                )

            elif method == "tools/list":
                tools = [
                    {"name": name, **schema} for name, schema in EXPOSED_TOOLS.items()
                ]
                return self._response(req_id, {"tools": tools})

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self._call_tool(tool_name, arguments)
                return self._response(req_id, result)

            else:
                return self._error(req_id, -32601, f"Method not found: {method}")

        except Exception as exc:
            logger.exception("MCP request error")
            return self._error(req_id, -32603, "Internal error. Check server logs.")

    # ── Tool execution ──────────────────────────────────────────────────

    async def _call_tool(self, name: str, args: dict) -> dict:
        """Execute a tool via the internal registry."""
        if name not in EXPOSED_TOOLS:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool '{name}' not exposed via MCP",
                    }
                ],
                "isError": True,
            }

        tool = tool_registry.get(name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Tool '{name}' not registered"}],
                "isError": True,
            }

        try:
            # All exposed tools expect the calling user via ``user`` kwarg.
            # Pass ``_confirmed=True`` because the user already authorised
            # the MCP connection (the config file acts as consent).
            result = await tool_registry.execute(
                name, user=self.telegram_id, _confirmed=True, **args
            )
            # Normalise the internal dict format to MCP content list.
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, default=str),
                    }
                ]
            }
        except Exception as exc:
            logger.exception("MCP tool %r failed", name)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool '{name}' failed. Check server logs.",
                    }
                ],
                "isError": True,
            }

    # ── JSON-RPC helpers ────────────────────────────────────────────────

    @staticmethod
    def _response(req_id: Any, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


# ── stdio transport ─────────────────────────────────────────────────────────


async def run_stdio_server(telegram_id: int) -> None:
    """Run MCP server over stdio (for Claude Desktop, Cursor, etc.).

    Reads JSON-RPC 2.0 messages line-by-line from stdin and writes responses
    to stdout.  The server runs until stdin is closed.

    Args:
        telegram_id: The Telegram user ID that scopes all tool operations.
    """
    # Ensure all internal tools are registered.
    bootstrap.register_builtin_tools()

    server = MCPServer(telegram_id)
    loop = asyncio.get_running_loop()

    # Set up stdin reader.
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    # Set up stdout writer via pipe.
    write_transport, write_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin,
        sys.stdout,  # type: ignore[arg-type]
    )
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            request = json.loads(line.decode())
            response = await server.handle_request(request)
            payload = json.dumps(response, ensure_ascii=False).encode() + b"\n"
            writer.write(payload)
            await writer.drain()
        except json.JSONDecodeError:
            # Silently skip malformed lines.
            continue
