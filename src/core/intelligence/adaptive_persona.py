"""Adaptive Persona — бот подстраивает стиль общения под пользователя."""

import json
import logging
from datetime import datetime, timezone

from src.db.session import get_session
from src.db.repo import get_or_create_user, get_persona, update_persona

logger = logging.getLogger(__name__)

# Маппинг NL-инструкций в поля persona
INSTRUCTION_MAP = {
    "short": (
        ("короч", "кратк", "покороч", "лаконичн", "сократи"),
        {"brevity": "short"},
    ),
    "detailed": (
        ("подробн", "развёрнут", "детальн", "распиши"),
        {"brevity": "detailed"},
    ),
    "formal": (
        ("формальн", "официальн", "серьёзн", "строг"),
        {"formality": "formal"},
    ),
    "friendly": (
        ("дружелюбн", "прощ", "веселе", "полегч"),
        {"formality": "friendly"},
    ),
    "no_emoji": (
        (
            "без смайл",
            "без эмодз",
            "убери смайлы",
            "не используй смайл",
            "не ставь смайл",
        ),
        {"emoji_usage": "none"},
    ),
    "more_emoji": (
        ("больше смайл", "добавь смайл", "эмодзи"),
        {"emoji_usage": "rich"},
    ),
    "proactive": (
        ("инициатив", "предлаг", "сам решай", "будь актив"),
        {"initiative": "proactive"},
    ),
    "reactive": (
        ("не лез", "не предлаг", "только когда спрашива", "будь пассив"),
        {"initiative": "reactive"},
    ),
    "bullets": (
        ("списк", "пункт", "буллит", "через маркер"),
        {"preferred_format": "bullets"},
    ),
    "numbered": (
        ("цифр", "нумер", "по порядку"),
        {"preferred_format": "numbered"},
    ),
    "focus": (
        ("фокус", "не отвлекай", "работаю", "занят"),
        {"work_mode": "focus"},
    ),
    "relax": (
        ("отдых", "расслаб", "отдыхаю", "релакс"),
        {"work_mode": "relax"},
    ),
}


async def detect_persona_change(user_text: str) -> dict | None:
    """Распознаёт изменение persona в тексте.

    Returns:
        {"changes": dict, "auto_apply": bool, "reason": str} или None
    """
    t = user_text.lower()
    for name, (triggers, changes) in INSTRUCTION_MAP.items():
        for trigger in triggers:
            if trigger in t:
                return {"changes": changes, "auto_apply": True, "reason": name}
    return None


async def apply_persona_changes(telegram_id: int, changes: dict):
    """Применяет изменения к persona."""
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        persona = await get_persona(session, owner)
        await update_persona(
            session,
            persona,
            total_corrections=persona.total_corrections + 1,
            last_correction_at=datetime.now(timezone.utc),
            **changes,
        )


async def format_persona_for_prompt(telegram_id: int) -> str:
    """Форматирует persona для инжекции в промпт через prompt_assembler.

    Результат устанавливается как ctx.persona_block в AssemblyContext
    и инжектится через PromptAssembler._tier2_context().
    """
    from src.core.context_cache import get as cache_get

    cached = cache_get(f"persona:{telegram_id}")
    if cached is not None:
        return cached

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

    rules = []
    if p.brevity == "short":
        rules.append("отвечай коротко (1-2 предложения)")
    elif p.brevity == "detailed":
        rules.append("отвечай подробно")
    if p.formality == "formal":
        rules.append("формальный тон, на «вы»")
    elif p.formality == "casual":
        rules.append("очень неформально, с юмором")
    if p.emoji_usage == "none":
        rules.append("НЕ используй эмодзи")
    elif p.emoji_usage == "minimal":
        rules.append("минимум эмодзи")
    elif p.emoji_usage == "rich":
        rules.append("используй больше эмодзи")
    if p.initiative == "proactive":
        rules.append("проявляй инициативу — предлагай, напоминай, спрашивай")
    elif p.initiative == "reactive":
        rules.append("только отвечай на вопросы, не предлагай сам")
    if p.preferred_format == "bullets":
        rules.append("форматируй списком")
    elif p.preferred_format == "numbered":
        rules.append("нумеруй пункты")
    if p.max_response_len:
        rules.append(f"ответ не длиннее {p.max_response_len} символов")
    if p.work_mode == "focus":
        rules.append("режим фокуса — не отвлекай, только срочное")
    elif p.work_mode == "relax":
        rules.append("режим отдыха — только приятное общение")

    if not rules:
        result = ""
    else:
        result = "\n\n## ТВОЙ СТИЛЬ ОБЩЕНИЯ (установлен владельцем):\n" + "\n".join(
            f"- {r}" for r in rules
        )

    from src.core.context_cache import put as cache_put

    cache_put(f"persona:{telegram_id}", result, ttl=30)
    return result
