"""Adaptive Instructions — бот учится на реакциях пользователя."""

import json
import logging
from datetime import datetime, timezone

from src.db.session import get_session
from src.db.repo import get_or_create_user

logger = logging.getLogger(__name__)

# Правила которые применяются АВТОМАТИЧЕСКИ (безопасные)
SAFE_CATEGORIES = {"tone", "format"}

# Правила которые требуют ПОДТВЕРЖДЕНИЯ
CONFIRM_CATEGORIES = {"privacy", "memory", "agent"}

# Паттерны для детекции инструкций
INSTRUCTION_PATTERNS = [
    # Формат
    (
        r"(не|перестань)\s*(используй|ставь|пиши|добавляй)\s*(смайлики|эмодзи|emoji)",
        "tone",
        "не использовать эмодзи",
    ),
    (
        r"(используй|добавляй|ставь)\s*(смайлики|эмодзи|emoji)",
        "tone",
        "использовать эмодзи",
    ),
    (r"(отвечай|пиши)\s*короче", "format", "отвечать короче (1-2 предложения)"),
    (r"(отвечай|пиши)\s*подробнее", "format", "отвечать подробнее"),
    (
        r"(не используй|убери)\s*(HTML|разметку|теги)",
        "format",
        "не использовать HTML-разметку",
    ),
    (
        r"(пиши|отвечай)\s*(на|по)[-\s]?(русски|английски)",
        "format",
        "отвечать на заданном языке",
    ),
    # Тон
    (r"(будь|стань)\s*(формальнее|серьёзнее|официальнее)", "tone", "формальный тон"),
    (r"(будь|стань)\s*(дружелюбнее|веселее|проще)", "tone", "дружелюбный тон"),
    (r"не\s*(дерзи|хами|груби)", "tone", "вежливый тон"),
    # Приватность
    (
        r"(не сохраняй|не запоминай|не пиши)\s*(это|такое|в память)",
        "privacy",
        "не сохранять в память",
    ),
    (r"(запоминай|сохраняй)\s*(вс[её]|факты)", "memory", "сохранять все факты"),
    # Агенты
    (r"(не используй|отключи)\s*(агентов|agent)", "agent", "не использовать агентов"),
    (r"(используй|включи)\s*(агентов|agent)", "agent", "использовать агентов"),
]


async def detect_instruction(user_text: str, telegram_id: int) -> dict | None:
    """Распознаёт инструкцию в тексте пользователя. Возвращает {rule, category, is_safe} или None."""
    import re

    for pattern, category, rule in INSTRUCTION_PATTERNS:
        if re.search(pattern, user_text.lower()):
            is_safe = category in SAFE_CATEGORIES
            return {
                "rule": rule,
                "category": category,
                "is_safe": is_safe,
                "action": "applied" if is_safe else "asked",
            }
    return None


async def apply_instruction(telegram_id: int, rule: str):
    """Применяет правило (добавляет в активный профиль)."""
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        from sqlalchemy import select
        from src.db.models import InstructionProfile

        result = await session.execute(
            select(InstructionProfile).where(InstructionProfile.user_id == owner.id)
        )
        profile = result.scalar_one_or_none()
        if profile is None:
            profile = InstructionProfile(user_id=owner.id, rules_json="[]")
            session.add(profile)
        rules = json.loads(profile.rules_json)
        if rule not in rules:
            rules.append(rule)
        profile.rules_json = json.dumps(rules, ensure_ascii=False)
        profile.updated_at = datetime.now(timezone.utc)
        await session.flush()


async def get_active_rules(telegram_id: int) -> list[str]:
    """Возвращает активные правила."""
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        from sqlalchemy import select
        from src.db.models import InstructionProfile

        result = await session.execute(
            select(InstructionProfile).where(InstructionProfile.user_id == owner.id)
        )
        profile = result.scalar_one_or_none()
        return json.loads(profile.rules_json) if profile else []


async def format_rules_for_prompt(telegram_id: int) -> str:
    """Форматирует правила для инжекции в промпт."""
    rules = await get_active_rules(telegram_id)
    if not rules:
        return ""
    lines = ["\n\n## АКТИВНЫЕ ПРАВИЛА (владелец установил):"]
    for r in rules:
        lines.append(f"- {r}")
    return "\n".join(lines)
