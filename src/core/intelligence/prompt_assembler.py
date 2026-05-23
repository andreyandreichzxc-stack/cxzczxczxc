"""Prompt Assembler — единственная точка сборки system-prompt из трёх tiers.

Tier 1 (STABLE):   неизменяемый якорь — core identity, safety rules.
Tier 2 (CONTEXT):  полу-стабильный контекст — persona, confirmed rules, agents.
Tier 3 (VOLATILE): динамический контекст — memory, history, RAG, candidates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.intelligence.soul_blocks import ANTI_AI_BLOCK, _load_blocks
from src.db.repo import get_or_create_user, get_self_profile
from src.db.session import get_session

logger = logging.getLogger(__name__)

# Максимальная длина промпта в символах (безопасный лимит для большинства LLM)
MAX_PROMPT_CHARS = 32_000


def _truncate_smart(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence-ending punctuation within limit."""
    if len(text) <= max_chars:
        return text
    # Find last sentence boundary (., !, ?, newline + letter)
    truncated = text[:max_chars]
    matches = list(re.finditer(r"[.!?]\s", truncated))
    if matches:
        cut = matches[-1].end()
        return truncated[:cut].rstrip()
    # Fallback: last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space].rstrip() + "…"
    return truncated.rstrip() + "…"


# Приоритет при усечении: что выкидывать в первую очередь
TRUNCATION_PRIORITY = [
    "preview_candidates",
    "rag_context",
    "conversation_history",
    "deep_memory",
    "memory_context",
]


@dataclass
class AssemblyContext:
    """Контекст для сборки промпта — передаётся в PromptAssembler.assemble()."""

    target: str  # "maestro" | "agent" | "summarizer"
    user_id: int
    contact_id: Optional[int] = None
    conversation_history: list = field(default_factory=list)
    memory_context: str = ""
    deep_memory: str = ""
    persona_block: str = ""
    style_match_block: str = ""
    confirmed_rules: list = field(default_factory=list)
    preview_candidates: list = field(default_factory=list)
    rag_context: str = ""
    skill_index: str = ""
    # Anti-AI humanizer
    anti_ai: bool = True
    # Дополнительные поля для agent target
    now_local: str = ""
    tz_name: str = ""
    history_block: str = ""
    self_profile: str = ""


class PromptAssembler:
    """Собирает system prompt из трёх tiers.

    Используется как синглтон через prompt_assembler.
    """

    def __init__(self):
        self._blocks = _load_blocks()

    # ------------------------------------------------------------------
    # Tier helpers
    # ------------------------------------------------------------------

    def _tier1_stable(self, target: str) -> str:
        """Tier 1 — неизменяемый якорь."""
        if target == "maestro":
            return (
                self._blocks["stable_maestro_core"]
                + "\n\n"
                + self._blocks["stable_maestro_convictions"]
                + "\n\n"
                + self._blocks["stable_maestro_safety"]
            )
        elif target == "agent":
            return self._blocks["stable_agent_core"]
        else:
            return ""

    def _tier2_context(self, target: str, ctx: AssemblyContext) -> str:
        """Tier 2 — полу-стабильный контекст."""
        parts = []

        # Agent list / intents / format для maestro
        if target == "maestro":
            parts.append(self._blocks["context_maestro_agents"])
            parts.append(self._blocks["context_maestro_intents"])
            parts.append(self._blocks["context_maestro_format"])
        elif target == "agent":
            parts.append(self._blocks["context_agent_intents"])
            parts.append(self._blocks["context_agent_format"])

        # Anti-AI block (controlled by per-user setting)
        if ctx.anti_ai:
            parts.append(ANTI_AI_BLOCK)

        # Persona block (из adaptive_persona)
        if ctx.persona_block:
            parts.append(ctx.persona_block)

        # Style‑match block (динамический анализ стиля пользователя)
        if ctx.style_match_block:
            parts.append(ctx.style_match_block)

        # Confirmed rules (из adaptive_instructions)
        if ctx.confirmed_rules:
            rules_lines = ["\n\n## АКТИВНЫЕ ПРАВИЛА (владелец установил):"]
            for r in ctx.confirmed_rules:
                rules_lines.append(f"- {r}")
            parts.append("\n".join(rules_lines))

        return "\n".join(parts)

    def _tier3_volatile(self, ctx: AssemblyContext) -> str:
        """Tier 3 — динамический контекст."""
        parts = []

        # Temporal context для agent
        if ctx.target == "agent" and ctx.now_local and ctx.tz_name:
            parts.append(
                f"Текущее локальное время владельца: {ctx.now_local} ({ctx.tz_name}).\n"
                f"Когда нужно превратить относительную дату («завтра», «через час», «в пятницу 18:00») "
                f"в ISO-8601, используй ЛОКАЛЬНОЕ время в TZ владельца (НЕ конвертируй в UTC). "
                f"Формат: YYYY-MM-DDTHH:MM (без Z, без смещения)."
            )

        # Memory context
        if ctx.memory_context:
            if ctx.target == "maestro":
                parts.append(ctx.memory_context)
            elif ctx.target == "agent":
                parts.append(f"Факты из памяти:\n{ctx.memory_context}")

        if ctx.skill_index and ctx.target in {"agent", "maestro", "summarizer"}:
            parts.append(ctx.skill_index)

        # Deep memory
        if ctx.deep_memory:
            parts.append(ctx.deep_memory)

        # Conversation history
        if ctx.history_block:
            parts.append(ctx.history_block)
        elif ctx.conversation_history:
            history_text = "\n".join(str(m) for m in ctx.conversation_history[-20:])
            if history_text:
                parts.append(f"История диалога:\n{history_text}")

        # Self profile (для agent)
        if ctx.target == "agent" and ctx.self_profile:
            parts.append(ctx.self_profile)

        # RAG context
        if ctx.rag_context:
            parts.append(
                f"Релевантный контекст из истории переписок:\n{ctx.rag_context}"
            )

        # Preview candidates
        if ctx.preview_candidates:
            cand_lines = ["\n\n## КАНДИДАТЫ В ПАМЯТЬ:"]
            for c in ctx.preview_candidates[:5]:
                cand_lines.append(f"- {c}")
            parts.append("\n".join(cand_lines))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(self, ctx: AssemblyContext) -> str:
        """Собирает полный system prompt из трёх tiers.

        Порядок: STABLE → CONTEXT → VOLATILE.
        """
        tier1 = self._tier1_stable(ctx.target)
        tier2 = self._tier2_context(ctx.target, ctx)
        tier3 = self._tier3_volatile(ctx)

        parts = [p for p in [tier1, tier2, tier3] if p]
        prompt = "\n\n".join(parts)

        # Проверка ёмкости
        prompt = self._capacity_check(prompt)

        return prompt

    def _capacity_check(self, prompt: str) -> str:
        """Проверяет длину промпта и усекает при необходимости."""
        if len(prompt) <= MAX_PROMPT_CHARS:
            return prompt

        logger.warning(
            "Prompt too long (%d chars), truncating to %d",
            len(prompt),
            MAX_PROMPT_CHARS,
        )

        # Smart truncation: режем по границе предложения, не посередине слова
        truncated = _truncate_smart(prompt, MAX_PROMPT_CHARS)
        # Добавляем предупреждение
        truncated = (
            _truncate_smart(truncated, MAX_PROMPT_CHARS - 100)
            + "\n\n[Промпт усечён из-за ограничения длины. Часть контекста опущена.]"
        )
        return truncated

    def inject_rule(self, rule_tier: str, rule_text: str) -> bool:
        """Проверяет, можно ли инжектить правило в tier.

        Args:
            rule_tier: "stable" | "context" | "volatile"
            rule_text: текст правила

        Returns:
            True если инжект разрешён, False если REJECT.
        """
        tier = rule_tier.lower().strip()
        if tier == "stable":
            logger.warning("REJECT: попытка инжекта в stable tier: %s", rule_text[:100])
            return False
        elif tier == "context":
            # OK, но требует подтверждения (снапшот)
            logger.info("CONTEXT инжект (требует confirm): %s", rule_text[:100])
            return True
        elif tier == "volatile":
            # OK, auto-apply
            logger.debug("VOLATILE инжект (auto-apply): %s", rule_text[:100])
            return True
        else:
            logger.warning("REJECT: неизвестный tier '%s'", rule_tier)
            return False

    def get_block(self, name: str) -> str:
        """Возвращает конкретный блок по имени (для тестов/инспекции)."""
        return self._blocks.get(name, "")

    def update_context_block(self, name: str, new_text: str) -> bool:
        """Обновляет tier-2 блок (только CONTEXT блоки).

        Returns:
            True если обновлён, False если блок не найден или это STABLE блок.
        """
        if name not in self._blocks:
            logger.warning("update_context_block: блок '%s' не найден", name)
            return False
        if name.startswith("stable_"):
            logger.warning(
                "update_context_block: REJECT — блок '%s' является STABLE", name
            )
            return False
        self._blocks[name] = new_text
        logger.info("update_context_block: блок '%s' обновлён", name)
        return True

    def get_context_blocks(self) -> dict[str, str]:
        """Возвращает все tier-2 блоки (для снапшотов)."""
        return {
            name: text
            for name, text in self._blocks.items()
            if name.startswith("context_")
        }


# Глобальный синглтон (ленивый — не создаёт БД-соединений при импорте)
prompt_assembler = PromptAssembler()


async def assemble_self_profile_prompt(owner_id: int, session=None) -> str:
    """Собирает блок self-profile из БД.

    Args:
        owner_id: ID владельца.
        session: опциональная асинхронная сессия (если None — создаёт новую).

    Returns:
        отформатированный блок профиля или "" если профиля нет / ошибка.
    """
    if session is not None:
        owner = await get_or_create_user(session, owner_id)
        profile = await get_self_profile(session, owner)
    else:
        async with get_session() as _session:
            owner = await get_or_create_user(_session, owner_id)
            profile = await get_self_profile(_session, owner)

    if not profile:
        return ""

    lines = ["ТВОЙ ПРОФИЛЬ (владелец):"]
    if profile.preferences:
        lines.append(f"Предпочтения: {profile.preferences}")
    if profile.goals:
        lines.append(f"Цели: {profile.goals}")
    if profile.current_projects:
        lines.append(f"Проекты: {profile.current_projects}")
    if profile.decision_style:
        lines.append(f"Стиль решений: {profile.decision_style}")
    if profile.communication_preferences:
        lines.append(f"Коммуникация: {profile.communication_preferences}")
    if profile.sleep_pattern:
        lines.append(f"Сон: {profile.sleep_pattern}")
    if profile.work_hours:
        lines.append(f"Рабочие часы: {profile.work_hours}")
    return "\n".join(lines)
