"""recall_memory tool — registered via @tool decorator.

Wraps ``src.core.memory.memory_recall.recall`` as a tool that the LLM can invoke
on demand instead of pre-loading memory into every prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="recall_memory",
    description=(
        "Ищет факты в памяти бота по ключевой фразе. "
        "Используй когда нужно вспомнить что-то о человеке или событии."
    ),
    category="memory",
    risk="low",
    params={
        "query": "str — текст для поиска в памяти",
        "limit": "int=8 — макс. число фактов",
        "mode": "str=normal — режим поиска: 'light' (быстрый, только pinned+fresh+frequent), 'normal' (с self-фактами + hybrid), 'deep' (полный граф памяти)",
        "include_self": "bool=true — включать ли self-факты владельца",
        "contact_id": "int|None — ограничить поиск конкретным контактом",
    },
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for memory"},
            "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
            "mode": {
                "type": "string",
                "enum": ["light", "normal", "deep"],
                "default": "normal",
            },
            "include_self": {
                "type": "boolean",
                "default": True,
                "description": "Include owner self-facts",
            },
            "contact_id": {
                "type": ["integer", "null"],
                "description": "Optional Telegram peer/contact id for scoped recall",
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string", "description": "The memory fact"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "0-1 confidence score",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this fact was returned",
                        },
                    },
                },
            },
            "found": {"type": "integer", "description": "Total facts found"},
            "error": {"type": "string", "description": "Error message when ok=false"},
        },
        "required": ["ok", "facts", "found"],
    },
)
async def recall_memory(
    query: str,
    limit: int = 8,
    mode: str = "normal",
    include_self: bool = True,
    contact_id: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search bot memory for facts matching *query*.

    Expects ``user`` (telegram_id) in *kwargs* (injected by the caller).

    Args:
        query: text to search for in memory.
        limit: max number of facts to return.
        mode: 'light' (fast, pinned+fresh+frequent only),
              'normal' (with self-facts + hybrid),
              'deep' (full memory graph).
        include_self: whether to include owner self-facts.

    Returns:
        A dict with ``"ok"``, ``"facts"`` and ``"found"``.
    """
    _user_val = kwargs.get("user")

    if _user_val is None:
        return {"ok": False, "facts": [], "found": 0, "error": "user not provided"}

    # user may be an int (telegram_id) or a User ORM object — normalise
    if hasattr(_user_val, "telegram_id"):
        telegram_id: int = _user_val.telegram_id
    else:
        telegram_id = int(_user_val)

    if contact_id is None:
        runtime_contact_id = kwargs.get("runtime_contact_id")
        if runtime_contact_id is not None:
            try:
                contact_id = int(runtime_contact_id)
            except (TypeError, ValueError):
                contact_id = None

    try:
        from src.core.memory.memory_recall import recall

        result = await recall(
            telegram_id=telegram_id,
            contact_id=contact_id,
            query=query,
            limit=limit,
            include_deep=(mode == "deep"),
            include_self=include_self,
            mode=mode,
        )
        facts = [
            {
                "fact": f.fact,
                "reason": f.reason,
                "confidence": f.confidence,
            }
            for f in result.facts
        ]
        return {"ok": True, "facts": facts, "found": len(facts)}
    except Exception:
        logger.exception("recall_memory tool failed")
        return {
            "ok": False,
            "facts": [],
            "found": 0,
            "error": "Memory recall failed, please try again later",
        }
