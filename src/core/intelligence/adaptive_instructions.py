"""Adaptive Instructions — бот учится на реакциях пользователя."""

import json
import logging
import re
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
    """Распознаёт инструкцию в тексте пользователя. Возвращает {rule, category, is_safe} или None.

    InstructionCandidate создаётся в free_text.py (с llm_reviewed=False).
    """
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


async def review_pending_candidates(telegram_id: int) -> int:
    """
    Запускает LLM-ревью для всех InstructionCandidate с llm_reviewed=False.

    Вызывается периодически (например, раз в день или при входе в /settings).
    Возвращает количество обработанных кандидатов.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        from sqlalchemy import select as _sel, update as _upd
        from src.db.models import InstructionCandidate

        result = await session.execute(
            _sel(InstructionCandidate).where(
                InstructionCandidate.user_id == owner.id,
                InstructionCandidate.llm_reviewed == False,
            )
        )
        pending = result.scalars().all()

        if not pending:
            return 0

        try:
            from src.core.intelligence.instruction_optimizer import instruction_optimizer

            for cand in pending:
                # LLM проверяет кандидата через consolidate_rules
                # (которая анализирует правила на противоречия)
                await instruction_optimizer.consolidate_rules(
                    session=session,
                    user_id=owner.id,
                    user_obj=owner,
                )
                cand.llm_reviewed = True

            await session.commit()
            return len(pending)
        except Exception:
            logger.debug("LLM review of pending candidates failed", exc_info=True)
            return 0


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
    """Форматирует правила для инжекции в промпт (с проверкой safety gate).

    Каждое правило проверяется через soul_snapshot.safety_gate() и
    prompt_assembler.inject_rule() перед включением в промпт.
    """
    rules = await get_active_rules(telegram_id)
    if not rules:
        return ""

    safe_rules = []
    for rule in rules:
        # Определяем tier правила
        rule_lower = rule.lower()
        if any(kw in rule_lower for kw in ["не использ", "отключ", "забудь"]):
            tier = "context"  # потенциально опасные → context (требует confirm)
        elif any(
            kw in rule_lower
            for kw in ["формат", "тон", "стиль", "эмодзи", "отвечай", "пиши"]
        ):
            tier = "volatile"  # безопасные → auto-apply
        else:
            tier = "context"  # неизвестное → context по умолчанию

        # Проверяем safety gate (fail-closed: ошибка → REJECT)
        try:
            from src.core.intelligence.soul_snapshot import soul_snapshot

            allowed, reason = soul_snapshot.safety_gate(tier, rule)
            if not allowed:
                logger.warning(
                    "Правило отклонено safety_gate: %s — %s", rule[:80], reason
                )
                continue
        except Exception:
            logger.error("safety_gate crashed, REJECTING rule", exc_info=True)
            continue

        # Проверяем inject_rule (fail-closed: ошибка → REJECT)
        try:
            from src.core.intelligence.prompt_assembler import prompt_assembler

            if prompt_assembler.inject_rule(tier, rule):
                safe_rules.append(rule)
            else:
                logger.warning("Правило отклонено inject_rule: %s", rule[:80])
        except Exception:
            logger.error("inject_rule crashed, REJECTING rule", exc_info=True)

    if not safe_rules:
        return ""

    lines = ["\n\n## АКТИВНЫЕ ПРАВИЛА (владелец установил):"]
    for r in safe_rules:
        lines.append(f"- {r}")
    return "\n".join(lines)
