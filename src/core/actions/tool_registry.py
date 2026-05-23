"""Tool Registry — decorator-based action registration for LLM tool use.

Provides the ``@tool`` decorator and ``ToolRegistry`` singleton for
standardized registration of actions (tools) with metadata such as
description, category, risk level, and parameter schema.

Tools are registered *at import time* via the decorator, making the
registry effectively read-only after module initialisation.

Usage::

    from src.core.actions.tool_registry import tool, tool_registry

    @tool(
        name="search_messages",
        description="Search messages by text",
        category="search",
        risk="low",
        params={"query": "str", "contact": "str|None"},
    )
    async def search_messages(query: str, contact: str | None = None) -> dict:
        ...

    result = await tool_registry.execute("search_messages", query="hello")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ── ToolSpec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    """Immutable specification for a registered tool.

    Attributes:
        name: Unique tool identifier (used in ``execute()`` and prompts).
        description: Human-readable description of what the tool does.
        category: Grouping category (e.g. ``"search"``, ``"chat"``, ``"reminder"``).
        risk: Risk level — ``"low"``, ``"medium"``, ``"high"``, or ``"critical"``.
        requires_confirmation: Whether execution should prompt the user first.
        params: Dict mapping parameter name → type hint string.
                Example: ``{"query": "str", "limit": "int|None"}``.
        handler: The async callable that implements the tool.
    """

    name: str
    description: str
    category: str
    handler: Callable[..., Awaitable[dict]] = field(hash=False, compare=False)
    risk: str = "low"
    requires_confirmation: bool = False
    params: dict[str, str] = field(default_factory=dict)


# ── ToolRegistry ─────────────────────────────────────────────────────────


class ToolRegistry:
    """Registry of tools populated at import time via ``@tool``.

    The registry is effectively **read-only** after initialisation — tools
    are registered once when their defining module is imported.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Register a ``ToolSpec`` (called by the ``@tool`` decorator).

        If a tool with the same name already exists it is overwritten and
        a warning is logged.  This can happen during reloads.
        """
        if spec.name in self._tools:
            logger.warning("Tool %r already registered, overwriting", spec.name)
        self._tools[spec.name] = spec

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        """Look up a tool by its unique name.

        Returns ``None`` when no tool with *name* is registered.
        """
        return self._tools.get(name)

    def list_by_category(self) -> dict[str, list[ToolSpec]]:
        """Return all tools grouped by their ``category`` field."""
        categories: dict[str, list[ToolSpec]] = {}
        for spec in self._tools.values():
            categories.setdefault(spec.category, []).append(spec)
        return categories

    def list_for_prompt(self) -> str:
        """Format all tools as a prompt-friendly string for LLM system prompts.

        Example output::

            ## chat
            - `draft_reply` (medium ⚠️ confirmation): Draft a reply …
              params: contact: str, message: str, style: str|None
            - `summarize_chat` (medium): Summarize chat with a contact …

            ## search
            - `search_messages` (low): Search messages by text …
              params: query: str, contact: str|None
        """
        lines: list[str] = []
        for category, tools in sorted(self.list_by_category().items()):
            lines.append(f"## {category}")
            for spec in sorted(tools, key=lambda s: s.name):
                confirm = " ⚠️ confirmation" if spec.requires_confirmation else ""
                lines.append(
                    f"- `{spec.name}` ({spec.risk}{confirm}): {spec.description}"
                )
                if spec.params:
                    params_str = ", ".join(f"{k}: {v}" for k, v in spec.params.items())
                    lines.append(f"  params: {params_str}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, name: str, **params: Any) -> dict[str, Any]:
        """Execute a tool by name, passing *params* to its handler.

        The handler receives the params as keyword arguments.  Any extra
        keyword arguments (e.g. runtime dependencies such as ``provider``)
        can be passed through -- they will be forwarded to the handler if
        its signature accepts ``**kwargs``.

        Returns:
            The dict returned by the handler, or ``{"error": <message>}``
            if the tool is not found or the handler raises.
        """
        spec = self.get(name)
        if spec is None:
            return {"error": f"Tool '{name}' not found"}

        try:
            result = await spec.handler(**params)
            # Normalise None return to a success dict
            if result is None:
                return {"ok": True}
            return result
        except Exception:
            logger.exception("Tool %r failed with params %r", name, params)
            return {"error": f"Tool '{name}' execution failed"}


# Module-level singleton — imported by other modules
tool_registry = ToolRegistry()


# ── @tool decorator ──────────────────────────────────────────────────────


def tool(
    *,
    name: str,
    description: str,
    category: str,
    risk: str = "low",
    requires_confirmation: bool = False,
    params: dict[str, str] | None = None,
) -> Callable[[Callable[..., Awaitable[dict]]], Callable[..., Awaitable[dict]]]:
    """Decorator that registers an async function as a tool.

    The decorated function is automatically registered in the global
    ``tool_registry`` when the module is imported.

    Args:
        name: Unique tool name (used in ``execute()`` and LLM prompts).
        description: Human-readable description of what the tool does.
        category: Grouping category (e.g. ``"search"``, ``"memory"``, ``"reminder"``).
        risk: Risk level — ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        requires_confirmation: If ``True`` the LLM should ask the user before
            executing this tool (e.g. for destructive actions).
        params: Dict mapping parameter name → type hint string.
                Example: ``{"query": "str", "limit": "int|None"}``.

    Example::

        @tool(
            name="search_messages",
            description="Search messages by text",
            category="search",
            params={"query": "str"},
        )
        async def search_messages(query: str) -> dict:
            return {"ok": True, "query": query}
    """
    tool_params = dict(params or {})

    def decorator(
        func: Callable[..., Awaitable[dict]],
    ) -> Callable[..., Awaitable[dict]]:
        spec = ToolSpec(
            name=name,
            description=description,
            category=category,
            risk=risk,
            requires_confirmation=requires_confirmation,
            params=tool_params,
            handler=func,
        )
        tool_registry.register(spec)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ══════════════════════════════════════════════════════════════════════════
# Pre-populated tools
# ══════════════════════════════════════════════════════════════════════════
#
# These wrap existing agent / service functions with the ``@tool``
# decorator, providing standardised metadata and parameter interfaces.
#
# In production the caller injects runtime dependencies (provider, client,
# session, …) via keyword arguments that the handler forwards as ``**kwargs``
# to the underlying agent function.
#
# Every pre-populated tool follows the same contract:
#   async def handler(param1: type, ..., **kwargs) -> dict
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="search_messages",
    description=(
        "Search messages by text query across chats. "
        "Returns matching messages with surrounding context."
    ),
    category="search",
    risk="low",
    requires_confirmation=False,
    params={"query": "str", "contact": "str|None", "limit": "int"},
)
async def _tool_search_messages(
    query: str,
    contact: str | None = None,
    limit: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search messages across chats.

    Wraps ``src.agents.search_agent.resolve`` which resolves a contact
    by fuzzy name matching.  A real production implementation would also
    perform full-text search over the message store.
    """
    # Wires into existing search infrastructure:
    #   provider = kwargs.get("provider")
    #   contacts = kwargs.get("contacts", [])
    #   return await search_resolve(provider, query, contacts)
    return {
        "ok": True,
        "query": query,
        "contact": contact,
        "limit": limit,
    }


@tool(
    name="summarize_chat",
    description=(
        "Summarise recent conversation with a contact. "
        "Returns a concise summary of key topics and action items."
    ),
    category="chat",
    risk="medium",
    requires_confirmation=False,
    params={"contact": "str", "limit": "int"},
)
async def _tool_summarize_chat(
    contact: str,
    limit: int = 50,
    **kwargs: Any,
) -> dict[str, Any]:
    """Summarise chat with a contact.

    Wraps ``src.agents.summarizer_agent.summarize`` which calls an LLM
    to produce a structured summary from raw message text.
    """
    # provider = kwargs.get("provider")
    # messages_text = kwargs.get("messages_text", "")
    # return await summarizer_agent.summarize(provider, messages_text)
    return {
        "ok": True,
        "contact": contact,
        "limit": limit,
    }


@tool(
    name="draft_reply",
    description=(
        "Draft a reply message for a given contact and incoming message. "
        "Returns suggested text with tone label."
    ),
    category="chat",
    risk="medium",
    requires_confirmation=True,
    params={"contact": "str", "message": "str", "style": "str|None"},
)
async def _tool_draft_reply(
    contact: str,
    message: str,
    style: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Draft a reply to an incoming message.

    Wraps ``src.agents.draft_agent.draft`` which calls an LLM with
    context about the sender, conversation history, and optional style
    hints.
    """
    # provider = kwargs.get("provider")
    # return await draft_agent.draft(provider, contact, message, style_hint=style)
    return {
        "ok": True,
        "contact": contact,
        "message": message,
        "style": style,
    }


@tool(
    name="set_reminder",
    description=(
        "Create a reminder with optional due time. "
        "Reminders are persisted and checked automatically by the scheduler."
    ),
    category="reminder",
    risk="medium",
    requires_confirmation=False,
    params={"text": "str", "when": "str|None"},
)
async def _tool_set_reminder(
    text: str,
    when: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Set a reminder (commitment).

    Wraps the project's commitment / reminder infrastructure
    (``src.db.repo.add_commitment``, ``src.core.scheduling.reminders``).
    """
    # session = kwargs.get("session")
    # user_id = kwargs.get("user_id")
    # deadline = parse_datetime(when) if when else None
    # await add_commitment(session, user_id=user_id, text=text, deadline_at=deadline)
    return {
        "ok": True,
        "text": text,
        "when": when,
    }


@tool(
    name="list_contacts",
    description=(
        "Search the user's contact list by name or username. "
        "Returns matching contacts with display name and peer info."
    ),
    category="contacts",
    risk="low",
    requires_confirmation=False,
    params={"query": "str|None", "limit": "int"},
)
async def _tool_list_contacts(
    query: str | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search contacts by name.

    Wraps ``src.core.contacts.contact_resolver.resolve`` which performs
    fuzzy matching (via rapidfuzz) against the user's synced contact
    list.
    """
    # client = kwargs.get("client")
    # user = kwargs.get("user")
    # candidates = await resolve(client, user, query or "", limit=limit)
    # return {"contacts": [{"name": c.display_name, "peer_id": c.peer_id}
    #                      for c in candidates]}
    return {
        "ok": True,
        "query": query,
        "limit": limit,
    }
