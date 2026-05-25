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

# Module-level: cached import of dateutil for _tool_set_reminder
try:
    from dateutil.parser import parse as _dateparse

    _HAS_DATEUTIL = True
except ImportError:
    _dateparse = None  # type: ignore[assignment]
    _HAS_DATEUTIL = False


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

        **Security enforcement:** if the tool's ``ToolSpec.requires_confirmation``
        is ``True``, the caller **must** pass ``_confirmed=True``.  Callers that
        have not yet obtained user consent should pass ``_confirmed=False``
        (or omit it) and this method will return ``{"error": "requires
        confirmation"}`` without executing.

        Returns:
            The dict returned by the handler, or ``{"error": <message>}``
            if the tool is not found, requires confirmation, or the handler
            raises.
        """
        spec = self.get(name)
        if spec is None:
            return {"error": f"Tool '{name}' not found"}

        # Enforce requires_confirmation — caller must pass _confirmed=True
        confirmed = params.pop("_confirmed", False)
        if spec.requires_confirmation and not confirmed:
            return {"error": "requires confirmation"}

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
    """Search messages by text query across chats.

    Resolves *contact* (fuzzy name match) to scope the search to a
    single chat, then performs FTS5 full-text search over stored messages
    via ``cross_chat_search``.

    Runtime dependencies expected in **kwargs**:
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
        client (:class:`telethon.TelegramClient`, optional — for contact resolution)
    """
    session = kwargs["session"]
    user = kwargs["user"]
    client = kwargs.get("client")

    # Resolve contact name → peer_id if specified
    peer_id: int | None = None
    if contact:
        if client is None:
            return {
                "ok": False,
                "error": "No Telegram client available for contact resolution",
            }
        try:
            from src.core.contacts.contact_resolver import resolve as resolve_contact

            candidates = await resolve_contact(client, user, contact, limit=1)
            if candidates:
                peer_id = candidates[0].peer_id
        except Exception:
            logger.exception(
                "search_messages: contact resolution failed for %r", contact
            )
            return {"ok": False, "error": f"Contact resolution failed for '{contact}'"}

    try:
        from src.db.repo import cross_chat_search

        results = await cross_chat_search(
            session,
            user,
            query,
            limit=limit,
            peer_id=peer_id,
        )
        return {"ok": True, "results": results, "query": query}
    except Exception:
        logger.exception("search_messages: FTS search failed for %r", query)
        return {"ok": False, "error": f"Search failed for '{query}'"}


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
    limit: int = 100,
    **kwargs: Any,
) -> dict[str, Any]:
    """Summarise recent conversation with a contact via LLM.

    Resolves *contact* to a peer, fetches recent messages, converts them
    to a transcript, and calls ``summarizer_agent.summarize``.

    Runtime dependencies expected in **kwargs**:
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
        client (:class:`telethon.TelegramClient`, optional — for contact resolution)
        provider (LLM provider with ``chat()`` method)
    """
    session = kwargs["session"]
    user = kwargs["user"]
    client = kwargs.get("client")
    provider = kwargs.get("provider")

    if provider is None:
        return {"ok": False, "error": "No LLM provider available"}

    # Resolve contact → peer_id
    if client is None:
        return {
            "ok": False,
            "error": "No Telegram client available for contact resolution",
        }
    try:
        from src.core.contacts.contact_resolver import resolve as resolve_contact

        candidates = await resolve_contact(client, user, contact, limit=1)
    except Exception:
        logger.exception("summarize_chat: contact resolution failed for %r", contact)
        return {"ok": False, "error": f"Contact resolution failed for '{contact}'"}

    if not candidates:
        return {"ok": False, "error": f"Contact '{contact}' not found"}
    peer_id = candidates[0].peer_id
    display_name = candidates[0].display_name

    # Fetch messages
    try:
        from src.db.repo import fetch_chat_messages

        messages = await fetch_chat_messages(session, user, peer_id, limit=limit)
    except Exception:
        logger.exception("summarize_chat: fetch failed for peer_id=%s", peer_id)
        return {"ok": False, "error": "Failed to fetch messages"}

    if not messages:
        return {
            "ok": True,
            "summary": None,
            "contact": display_name,
            "note": "No messages found",
        }

    # Convert to transcript text
    try:
        from src.core.contacts.chat_service import messages_to_transcript

        text = messages_to_transcript(messages)
    except Exception:
        logger.exception("summarize_chat: transcript conversion failed")
        return {"ok": False, "error": "Failed to convert messages to transcript"}

    # Summarise via LLM
    try:
        from src.agents.summarizer_agent import summarize

        result = await summarize(provider, text)
        return {
            "ok": True,
            "summary": result.get("summary", ""),
            "contact": display_name,
            "message_count": len(messages),
        }
    except Exception:
        logger.exception("summarize_chat: LLM summarization failed")
        return {"ok": False, "error": "LLM summarization failed"}


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
    """Draft a reply to an incoming message from a contact via LLM.

    Resolves *contact* to a peer, fetches recent conversation history
    as context, and calls ``draft_agent.draft`` with optional style hint.

    Runtime dependencies expected in **kwargs**:
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
        client (:class:`telethon.TelegramClient`, optional — for contact resolution)
        provider (LLM provider with ``chat()`` method)
    """
    session = kwargs["session"]
    user = kwargs["user"]
    client = kwargs.get("client")
    provider = kwargs.get("provider")

    if provider is None:
        return {"ok": False, "error": "No LLM provider available"}

    # Resolve contact
    if client is None:
        return {
            "ok": False,
            "error": "No Telegram client available for contact resolution",
        }
    try:
        from src.core.contacts.contact_resolver import resolve as resolve_contact

        candidates = await resolve_contact(client, user, contact, limit=1)
    except Exception:
        logger.exception("draft_reply: contact resolution failed for %r", contact)
        return {"ok": False, "error": f"Contact resolution failed for '{contact}'"}

    if not candidates:
        return {"ok": False, "error": f"Contact '{contact}' not found"}
    peer_id = candidates[0].peer_id
    sender_name = candidates[0].display_name

    # Fetch history for context
    history_text: str | None = None
    try:
        from src.db.repo import fetch_chat_messages
        from src.core.contacts.chat_service import messages_to_transcript

        history = await fetch_chat_messages(session, user, peer_id, limit=20)
        if history:
            history_text = messages_to_transcript(history)
    except Exception:
        logger.exception("draft_reply: history fetch failed for peer_id=%s", peer_id)
        # Non-fatal: proceed without history

    # Draft via LLM
    try:
        from src.agents.draft_agent import draft

        result = await draft(
            provider,
            sender_name,
            message,
            history_text=history_text,
            style_hint=style,
        )
        return {
            "ok": True,
            "draft": result.get("draft", ""),
            "tone": result.get("tone", ""),
            "contact": sender_name,
        }
    except Exception:
        logger.exception("draft_reply: LLM drafting failed")
        return {"ok": False, "error": "LLM draft generation failed"}


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
    """Create a reminder (commitment) with optional due date/time.

    Parses *when* via ``dateutil.parser`` (supports "tomorrow",
    "in 2 hours", ISO-8601, etc.) and persists as a ``Commitment``
    row via ``add_commitment``.

    Runtime dependencies expected in **kwargs**:
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
    """
    session = kwargs["session"]
    user = kwargs["user"]

    # Parse deadline
    deadline = None
    if when:
        try:
            if _HAS_DATEUTIL and _dateparse is not None:
                deadline = _dateparse(when)
            else:
                # Fallback: ISO-format only (e.g. "2026-05-25T14:00:00")
                from datetime import datetime as _dt

                s2 = when.strip().replace("Z", "+00:00")
                deadline = _dt.fromisoformat(s2)

            if deadline.tzinfo is None:
                from datetime import timezone

                deadline = deadline.replace(tzinfo=timezone.utc)
        except Exception:
            logger.exception("set_reminder: cannot parse date %r", when)
            return {"ok": False, "error": f"Cannot parse date/time: {when}"}

    # Store commitment
    try:
        from src.db.repo import add_commitment

        c = await add_commitment(
            session,
            user_id=user.id,
            peer_id=0,
            peer_name="self",
            message_id=0,
            direction="mine",
            text=text,
            deadline_at=deadline,
        )
        return {
            "ok": True,
            "id": c.id,
            "text": text,
            "deadline": deadline.isoformat() if deadline else None,
        }
    except Exception:
        logger.exception("set_reminder: failed to persist commitment")
        return {"ok": False, "error": "Failed to save reminder"}


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
    limit: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """List or search the user's synced contacts.

    When *query* is provided, resolves via fuzzy matching
    (``contact_resolver.resolve``).  When *query* is omitted, returns
    contacts from the local database (``list_contacts`` in repo).

    Runtime dependencies expected in **kwargs**:
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
        client (:class:`telethon.TelegramClient`, optional — for contact resolution)
    """
    session = kwargs["session"]
    user = kwargs["user"]
    client = kwargs.get("client")

    try:
        if query:
            if client is None:
                return {
                    "ok": False,
                    "error": "No Telegram client available for contact search",
                }
            from src.core.contacts.contact_resolver import resolve as resolve_contact

            candidates = await resolve_contact(client, user, query, limit=limit)
            results = [
                {
                    "peer_id": c.peer_id,
                    "name": c.display_name,
                    "username": c.username,
                    "score": c.score,
                }
                for c in candidates
            ]
        else:
            from src.db.repo import list_contacts as db_list_contacts

            contacts = await db_list_contacts(session, user)
            results = [
                {
                    "peer_id": c.peer_id,
                    "name": c.display_name,
                    "username": c.username,
                    "kind": c.peer_kind,
                }
                for c in contacts[:limit]
            ]
        return {"ok": True, "contacts": results, "count": len(results)}
    except Exception:
        logger.exception("list_contacts: failed for query=%r", query)
        return {"ok": False, "error": "Failed to list contacts"}
