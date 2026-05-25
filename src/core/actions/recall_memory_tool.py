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
    },
)
async def recall_memory(
    query: str,
    limit: int = 8,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search bot memory for facts matching *query*.

    Expects ``user`` (telegram_id) in *kwargs* (injected by the caller).
    Uses ``mode="light"`` for quick recall without deep memory expansion.

    Returns:
        A dict with ``"facts": [...]`` and ``"found": N``, or
        ``{"error": "..."}`` on failure.
    """
    _user_val = kwargs.get("user")

    if _user_val is None:
        return {"error": "user not provided"}

    # user may be an int (telegram_id) or a User ORM object — normalise
    if hasattr(_user_val, "telegram_id"):
        telegram_id: int = _user_val.telegram_id
    else:
        telegram_id = int(_user_val)

    try:
        from src.core.memory.memory_recall import recall

        result = await recall(
            telegram_id=telegram_id,
            query=query,
            limit=limit,
            include_deep=False,
            mode="light",
        )
        facts = [
            {
                "fact": f.fact,
                "reason": f.reason,
                "confidence": f.confidence,
            }
            for f in result.facts
        ]
        return {"facts": facts, "found": len(facts)}
    except Exception:
        logger.exception("recall_memory tool failed")
        return {"error": "Memory recall failed, please try again later"}
