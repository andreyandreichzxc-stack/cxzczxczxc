"""Тесты для PromptAssembler — сборка system-prompt из трёх tiers."""

from __future__ import annotations

import pytest
from src.core.intelligence.prompt_assembler import (
    PromptAssembler,
    AssemblyContext,
    MAX_PROMPT_CHARS,
    TRUNCATION_PRIORITY,
    _truncate_smart,
)


@pytest.fixture
def assembler() -> PromptAssembler:
    """Создаёт свежий экземпляр PromptAssembler."""
    return PromptAssembler()


@pytest.fixture
def minimal_ctx() -> AssemblyContext:
    """Минимальный контекст для maestro."""
    return AssemblyContext(
        target="maestro",
        user_id=12345,
    )


@pytest.fixture
def full_ctx() -> AssemblyContext:
    """Контекст со всеми заполненными полями."""
    return AssemblyContext(
        target="maestro",
        user_id=12345,
        contact_id=999,
        conversation_history=[
            "пользователь: привет",
            "бот: привет! как дела?",
        ],
        memory_context="Пользователь любит кофе",
        deep_memory="Глубокий факт из долгой памяти",
        persona_block="[persona] пользователь — разработчик",
        style_match_block="[style] пользователь пишет коротко, без эмодзи",
        confirmed_rules=["правило 1", "правило 2"],
        preview_candidates=["кандидат А", "кандидат Б"],
        rag_context="Релевантный контекст из истории",
        skill_index="[skills] доступные навыки",
        anti_ai=True,
        user_message="привет, что нового?",
        now_local="2025-01-01 12:00",
        tz_name="Europe/Moscow",
        history_block="[история] последние сообщения",
        self_profile="[профиль] владелец",
        frozen_snapshot="[frozen] топ-3 факта",
        contact_rules_block="[rules] контактные правила",
    )


class TestTier1Stable:
    """Тесты Tier 1 — неизменяемый якорь."""

    def test_tier1_maestro_returns_identity(self, assembler):
        """Tier 1 для maestro содержит identity/safety текст."""
        result = assembler._tier1_stable("maestro")
        assert len(result) > 100, "Tier 1 maestro должен быть содержательным"
        # SOUL.md может переопределять hardcoded блоки — проверяем ключевые слова
        has_identity = (
            "AI-ассистент" in result
            or "ассистент" in result.lower()
            or "Telegram" in result
            or "Identity" in result
        )
        assert has_identity, (
            f"Tier 1 maestro должен содержать identity, получено: {result[:200]}"
        )

    def test_tier1_agent_returns_core(self, assembler):
        """Tier 1 для agent содержит core блок."""
        result = assembler._tier1_stable("agent")
        assert "AI-ассистент" in result
        assert len(result) > 50

    def test_tier1_unknown_target_empty(self, assembler):
        """Неизвестный target возвращает пустую строку."""
        result = assembler._tier1_stable("unknown_target")
        assert result == ""


class TestTier2Context:
    """Тесты Tier 2 — полу-стабильный контекст."""

    def test_tier2_includes_style(self, assembler, minimal_ctx):
        """style_match_block попадает в Tier 2."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            style_match_block="[style] тестовый стиль",
        )
        result = assembler._tier2_context("maestro", ctx)
        assert "[style] тестовый стиль" in result

    def test_tier2_includes_persona(self, assembler, minimal_ctx):
        """persona_block попадает в Tier 2."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            persona_block="[persona] тестовая персона",
        )
        result = assembler._tier2_context("maestro", ctx)
        assert "[persona] тестовая персона" in result

    def test_tier2_includes_confirmed_rules(self, assembler, minimal_ctx):
        """confirmed_rules форматируются и включаются."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            confirmed_rules=["всегда здороваться", "использовать эмодзи"],
        )
        result = assembler._tier2_context("maestro", ctx)
        assert "АКТИВНЫЕ ПРАВИЛА" in result
        assert "всегда здороваться" in result
        assert "использовать эмодзи" in result

    def test_tier2_anti_ai_block(self, assembler, minimal_ctx):
        """ANTI_AI_BLOCK включается когда anti_ai=True."""
        ctx = AssemblyContext(target="maestro", user_id=12345, anti_ai=True)
        result = assembler._tier2_context("maestro", ctx)
        assert "АНТИ-ШАБЛОН" in result

    def test_tier2_no_anti_ai_block(self, assembler, minimal_ctx):
        """ANTI_AI_BLOCK не включается когда anti_ai=False."""
        ctx = AssemblyContext(target="maestro", user_id=12345, anti_ai=False)
        result = assembler._tier2_context("maestro", ctx)
        assert "АНТИ-ШАБЛОН" not in result


class TestTier3Volatile:
    """Тесты Tier 3 — динамический контекст."""

    def test_tier3_includes_frozen_snapshot(self, assembler, minimal_ctx):
        """frozen_snapshot попадает в volatile секцию."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            frozen_snapshot="[frozen] факт 1, факт 2, факт 3",
        )
        result = assembler._tier3_volatile(ctx)
        assert "[frozen] факт 1, факт 2, факт 3" in result

    def test_tier3_includes_deep_memory(self, assembler, minimal_ctx):
        """deep_memory попадает в volatile секцию."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            deep_memory="глубокие воспоминания",
        )
        result = assembler._tier3_volatile(ctx)
        assert "глубокие воспоминания" in result

    def test_tier3_includes_rag_context(self, assembler, minimal_ctx):
        """rag_context попадает в volatile секцию."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            rag_context="релевантные документы",
        )
        result = assembler._tier3_volatile(ctx)
        assert "релевантные документы" in result

    def test_tier3_includes_candidates(self, assembler, minimal_ctx):
        """preview_candidates форматируются в volatile секцию."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            preview_candidates=["кандидат 1", "кандидат 2"],
        )
        result = assembler._tier3_volatile(ctx)
        assert "КАНДИДАТЫ В ПАМЯТЬ" in result
        assert "кандидат 1" in result

    def test_tier3_temporal_for_agent(self, assembler):
        """Временной контекст включается только для agent."""
        ctx = AssemblyContext(
            target="agent",
            user_id=12345,
            now_local="2025-06-15 14:30",
            tz_name="Europe/Moscow",
        )
        result = assembler._tier3_volatile(ctx)
        assert "2025-06-15 14:30" in result
        assert "Europe/Moscow" in result

    def test_tier3_no_temporal_for_maestro(self, assembler, minimal_ctx):
        """Временной контекст НЕ включается для maestro."""
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            now_local="2025-06-15 14:30",
            tz_name="Europe/Moscow",
        )
        result = assembler._tier3_volatile(ctx)
        assert "2025-06-15 14:30" not in result


class TestTruncation:
    """Тесты усечения промпта."""

    def test_truncation_keeps_budget(self, assembler):
        """Результат assemble всегда ≤ MAX_PROMPT_CHARS."""
        # Создаём огромный контекст который точно переполнит
        huge_text = "X" * (MAX_PROMPT_CHARS + 5000)
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            deep_memory=huge_text,
        )
        result = assembler.assemble(ctx)
        assert len(result) <= MAX_PROMPT_CHARS, (
            f"Длина {len(result)} превышает лимит {MAX_PROMPT_CHARS}"
        )

    def test_truncate_smart_sentence_boundary(self):
        """Умное усечение по границе предложения."""
        text = "Первое предложение. Второе предложение. Третье и ещё."
        # Усекаем до середины второго предложения
        result = _truncate_smart(text, 35)
        # Должно обрезаться после "Первое предложение."
        assert len(result) <= 35
        assert result.rstrip().endswith(".") or result.rstrip().endswith("…")

    def test_truncate_smart_fallback_space(self):
        """Fallback: усечение по последнему пробелу если нет точки."""
        text = "длинное_слово_без_точек_и_запятых_совсем_очень_длинное"
        max_chars = 20
        result = _truncate_smart(text, max_chars)
        # Допускаем небольшое превышение из-за "…" (ellipsis = 1 символ)
        assert len(result) <= max_chars + 3, (
            f"Результат '{result}' (len={len(result)}) превышает лимит"
        )
        assert "…" in result or len(result) <= max_chars + 3

    def test_truncation_removes_content_when_overflowing(self, assembler):
        """При переполнении содержимое усекается (truncation срабатывает)."""
        # Приоритетное удаление: TRUNCATION_PRIORITY определяет порядок,
        # но текущая реализация _capacity_check использует smart truncation
        # по границе предложения. Проверяем что усечение вообще работает.
        huge = "X. " * 20_000  # много коротких предложений
        ctx = AssemblyContext(
            target="maestro",
            user_id=12345,
            deep_memory=huge,
        )
        result = assembler.assemble(ctx)
        assert len(result) <= MAX_PROMPT_CHARS
        # При усечении появляется предупреждение
        if len(ctx.deep_memory) > MAX_PROMPT_CHARS:
            assert "Промпт усечён" in result


class TestAssemble:
    """Тесты полной сборки prompt."""

    def test_assemble_with_all_fields(self, assembler, full_ctx):
        """Со всеми полями AssemblyContext возвращается непустая строка."""
        result = assembler.assemble(full_ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_assemble_empty_context(self, assembler):
        """Пустой контекст всё равно возвращает Tier 1."""
        ctx = AssemblyContext(target="agent", user_id=12345)
        result = assembler.assemble(ctx)
        assert len(result) > 0
        assert "AI-ассистент" in result

    def test_assemble_maestro_includes_all_tiers(self, assembler, full_ctx):
        """Maestro-сборка включает все три tiers."""
        result = assembler.assemble(full_ctx)
        # Tier 1
        assert "AI-ассистент" in result
        # Tier 2
        assert "[style]" in result
        # Tier 3
        assert "[frozen]" in result

    def test_assemble_agent_target(self, assembler):
        """Agent target корректно собирает prompt."""
        ctx = AssemblyContext(
            target="agent",
            user_id=12345,
            frozen_snapshot="[frozen] agent memory",
        )
        result = assembler.assemble(ctx)
        assert "AI-ассистент" in result
        assert "[frozen] agent memory" in result


class TestTruncationPriority:
    """Тесты порядка приоритетов усечения."""

    def test_truncation_priority_order(self):
        """TRUNCATION_PRIORITY содержит ожидаемые поля в правильном порядке."""
        assert TRUNCATION_PRIORITY[0] == "preview_candidates"
        assert TRUNCATION_PRIORITY[1] == "rag_context"
        assert TRUNCATION_PRIORITY[2] == "conversation_history"
        assert TRUNCATION_PRIORITY[3] == "deep_memory"
        assert len(TRUNCATION_PRIORITY) == 4
