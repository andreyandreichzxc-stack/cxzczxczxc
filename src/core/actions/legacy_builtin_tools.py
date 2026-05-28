"""Legacy built-in tools registered explicitly by action bootstrap."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

try:
    from dateutil.parser import parse as _dateparse

    _HAS_DATEUTIL = True
except ImportError:
    _dateparse = None  # type: ignore[assignment]
    _HAS_DATEUTIL = False

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
    name="ask_chat",
    description=(
        "Analyse a conversation with a contact and answer user's question about it. "
        "Returns insights, tone analysis, key topics, and a direct answer if a question was asked."
    ),
    category="chat",
    risk="medium",
    requires_confirmation=False,
    params={"contact": "str", "query": "str", "limit": "int"},
)
async def _tool_ask_chat(
    contact: str,
    query: str = "",
    limit: int = 100,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyse a conversation with a contact and answer a question.

    Resolves *contact* to a peer, fetches recent messages, and calls
    ``ask_chat()`` from the summarizer with the user's question.

    Runtime dependencies expected in **kwargs**:
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
        client (:class:`telethon.TelegramClient`, optional)
        provider (LLM provider with ``chat()`` method)
    """
    session = kwargs["session"]
    user = kwargs["user"]
    client = kwargs.get("client")
    provider = kwargs.get("provider")

    if provider is None:
        return {"ok": False, "error": "No LLM provider available"}

    if client is None:
        return {
            "ok": False,
            "error": "No Telegram client available for contact resolution",
        }

    # Resolve contact → peer_id
    try:
        from src.core.contacts.contact_resolver import resolve as resolve_contact

        candidates = await resolve_contact(client, user, contact, limit=1)
    except Exception:
        logger.exception("ask_chat: contact resolution failed for %r", contact)
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
        logger.exception("ask_chat: fetch failed for peer_id=%s", peer_id)
        return {"ok": False, "error": "Failed to fetch messages"}

    if not messages:
        return {
            "ok": True,
            "analysis": None,
            "contact": display_name,
            "note": "No messages found",
        }

    # Load full Contact from DB
    try:
        from src.db.repo import get_contact as _get_contact

        contact_obj = await _get_contact(session, user, peer_id)
    except Exception:
        logger.exception("ask_chat: failed to load contact for peer_id=%s", peer_id)
        return {"ok": False, "error": "Failed to load contact"}

    if contact_obj is None:
        return {"ok": False, "error": "Contact not found in DB"}

    # Load memory context (facts about this contact)
    memory_context = ""
    try:
        from src.core.contacts.contact_memory_digest import (
            get_contact_digest as _digest,
        )

        digest = await _digest(user.id, peer_id)
        facts = digest.get("facts") or []
        if facts:
            lines = ["<recall_context>\n📌 Факты о контакте:"]
            for f in facts[:5]:
                from src.core.infra.text_sanitizer import sanitize_html as _sh

                lines.append(f"• {_sh(f.get('fact', ''))}")
            promises = digest.get("promises") or []
            if promises:
                lines.append("\n📋 Обещания:")
                for p in promises[:3]:
                    from src.core.infra.text_sanitizer import sanitize_html as _sh2

                    lines.append(f"• {_sh2(p.get('text', ''))}")
            lines.append("</recall_context>")
            memory_context = "\n".join(lines)
    except Exception:
        logger.debug("ask_chat tool: failed to load memory context", exc_info=True)

    # Analyse via LLM
    try:
        from src.core.intelligence.summarizer import ask_chat as ask_chat_fn

        analysis = await ask_chat_fn(
            provider,
            contact_obj,
            messages,
            user_query=query,
            owner_id=user.id,
            heavy=False,
            global_style=user.global_style_profile,
            memory_context=memory_context,
        )

        # Check for error indicators in LLM output
        error_prefixes = ("⏱️", "❌", "⚠️ Ошибка")
        if analysis and any(analysis.startswith(p) for p in error_prefixes):
            return {"ok": False, "error": analysis}

        return {
            "ok": True,
            "analysis": analysis,
            "contact": display_name,
            "message_count": len(messages),
        }
    except Exception:
        logger.exception("ask_chat: LLM analysis failed")
        return {"ok": False, "error": "LLM analysis failed"}


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


@tool(
    name="delegate_task",
    description=(
        "Create a sub-agent to analyse a specific question or sub-task. "
        "The sub-agent runs its own independent LLM call and returns structured findings. "
        "Use this to decompose complex multi-step problems into smaller parallel analyses."
    ),
    category="agent",
    risk="medium",
    requires_confirmation=False,
    params={
        "task": "str — what to analyse (the core question)",
        "context": "str|None — additional data or transcript to analyse",
        "instructions": "str|None — custom system prompt instructions for the sub-agent",
        "contact": "str|None — contact name to scope analysis to a specific chat",
    },
)
async def _tool_delegate_task(
    task: str,
    context: str | None = None,
    instructions: str | None = None,
    contact: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a sub-agent for independent analysis of a sub-task.

    The sub-agent receives its own LLM context (system + user prompt) and
    returns an analysis.  Optionally fetches recent messages from *contact*
    if provided, for scoped chat analysis.

    Runtime dependencies expected in **kwargs**:
        provider (LLM provider with ``chat()`` method)
        session (:class:`sqlalchemy.ext.asyncio.AsyncSession`)
        user (:class:`src.db.models.User`)
        client (:class:`telethon.TelegramClient`, optional)
    """
    provider = kwargs.get("provider")
    session = kwargs.get("session")
    user = kwargs.get("user")
    client = kwargs.get("client")

    if provider is None:
        return {"ok": False, "error": "No LLM provider available"}

    # --- Build sub-agent system prompt ---
    system = (
        "Ты — аналитический суб-агент. Твоя задача: выполнить анализ "
        "или ответить на поставленный вопрос.\n"
        "Будь точен, аргументирован и структурирован. "
        "Не добавляй лишнего — только анализ по задаче."
    )
    if instructions:
        system += f"\n\nДополнительные инструкции:\n{instructions}"

    # --- Build user prompt ---
    user_prompt = f"Задача: {task}"
    if context:
        user_prompt += f"\n\nКонтекст:\n{context}"

    # --- Optionally fetch recent messages from a contact ---
    if contact and session and user and client:
        try:
            from src.core.contacts.contact_resolver import resolve as resolve_contact

            candidates = await resolve_contact(client, user, contact, limit=1)
            if candidates:
                peer_id = candidates[0].peer_id
                from src.db.repo import fetch_chat_messages
                from src.core.contacts.chat_service import messages_to_transcript

                messages = await fetch_chat_messages(session, user, peer_id, limit=30)
                if messages:
                    transcript = messages_to_transcript(messages)
                    user_prompt += (
                        f"\n\nПоследние сообщения из чата с {contact}:\n{transcript}"
                    )
        except Exception:
            logger.debug(
                "delegate_task: failed to fetch contact messages", exc_info=True
            )

    # --- Call LLM for the sub-agent ---
    try:
        from src.llm.base import ChatMessage

        raw = await asyncio.wait_for(
            provider.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user_prompt),
                ],
                heavy=False,
            ),
            timeout=90.0,
        )
        return {
            "ok": True,
            "analysis": raw.strip(),
            "task": task,
        }
    except Exception:
        logger.exception("delegate_task: sub-agent LLM call failed")
        return {"ok": False, "error": "Sub-agent analysis failed"}
