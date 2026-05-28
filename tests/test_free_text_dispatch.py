"""Smoke tests for intent dispatch registries in free_text_pipeline.py."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

from src.bot.handlers.free_text_pipeline import (
    CLASSIC_INTENT_HANDLERS,
    INTENT_HANDLERS,
    _dispatch,
    _execute_intent,
)


# ===========================================================================
# TestIntentHandlersRegistry
# ===========================================================================


class TestIntentHandlersRegistry:
    """Реестр INTENT_HANDLERS: валидность записей."""

    def test_all_intents_have_handler(self):
        """Каждая запись в INTENT_HANDLERS имеет callable handler и str description."""
        for kind, (handler, description) in INTENT_HANDLERS.items():
            assert callable(handler), f"{kind}: handler is not callable"
            assert isinstance(description, str), f"{kind}: description is not str"
            assert description, f"{kind}: description is empty"

    def test_no_duplicate_intents(self):
        """Нет дубликатов kind в INTENT_HANDLERS."""
        keys = list(INTENT_HANDLERS.keys())
        assert len(keys) == len(set(keys)), "Duplicate intent keys found"

    def test_handler_functions_are_imported(self):
        """Все handler-функции действительно импортированы (не ленивые строки)."""
        for kind, (handler, _) in INTENT_HANDLERS.items():
            assert not isinstance(handler, str), f"{kind}: handler is a string"
            assert inspect.iscoroutinefunction(handler) or callable(handler), (
                f"{kind}: handler is a non-imported proxy"
            )


# ===========================================================================
# TestClassicIntentHandlersRegistry
# ===========================================================================


class TestClassicIntentHandlersRegistry:
    """Реестр CLASSIC_INTENT_HANDLERS: валидность записей."""

    def test_all_classic_intents_have_handler(self):
        """Каждая запись в CLASSIC_INTENT_HANDLERS имеет callable handler и str description."""
        for kind, (handler, description) in CLASSIC_INTENT_HANDLERS.items():
            assert callable(handler), f"{kind}: handler is not callable"
            assert isinstance(description, str), f"{kind}: description is not str"
            assert description, f"{kind}: description is empty"

    def test_no_duplicate_classic_intents(self):
        """Нет дубликатов kind в CLASSIC_INTENT_HANDLERS."""
        keys = list(CLASSIC_INTENT_HANDLERS.keys())
        assert len(keys) == len(set(keys)), "Duplicate classic intent keys found"

    def test_handler_functions_are_imported(self):
        """Все handler-функции действительно импортированы (не ленивые строки)."""
        for kind, (handler, _) in CLASSIC_INTENT_HANDLERS.items():
            assert not isinstance(handler, str), f"{kind}: handler is a string"
            assert callable(handler), f"{kind}: handler is a non-imported proxy"


# ===========================================================================
# TestDispatchLogic
# ===========================================================================


@pytest.fixture
def msg():
    """Mock Message с минимальными атрибутами."""
    m = MagicMock()
    m.from_user.id = 123456789
    m.text = "test"
    return m


@pytest.fixture
def allowed_guard():
    """GuardResult: разрешённый intent."""
    return MagicMock(allowed=True, intent=None)


class TestDispatchLogic:
    """Dispatch-механизм: роутинг intent → handler."""

    @pytest.mark.asyncio
    async def test_known_intent_dispatches_to_handler(self, msg, allowed_guard):
        """_dispatch() вызывает правильный handler для известного intent."""
        mock_handler = AsyncMock()
        mock_state = MagicMock()
        mock_ubm = MagicMock()

        intent = {"intent": "set_setting", "key": "theme", "value": "dark"}
        allowed_guard.intent = intent

        from src.core.intelligence.guardrails import GuardrailResult

        with (
            patch.dict(
                "src.bot.handlers.free_text_pipeline.INTENT_HANDLERS",
                {"set_setting": (mock_handler, "test")},
                clear=True,
            ),
            patch(
                "src.bot.handlers.free_text_pipeline.guard_intent",
                return_value=allowed_guard,
            ),
            patch(
                "src.bot.handlers.free_text_pipeline.guardrail_evaluate",
                return_value=GuardrailResult(needs_confirm=False),
            ),
        ):
            await _dispatch(intent, msg, mock_state, mock_ubm, tz_name="UTC")

        mock_handler.assert_called_once_with(
            intent, msg, mock_state, mock_ubm, tz_name="UTC"
        )

    @pytest.mark.asyncio
    async def test_unknown_intent_falls_back_to_execute_intent(
        self, msg, allowed_guard
    ):
        """Неизвестный intent в INTENT_HANDLERS → вызов _execute_intent."""
        mock_state = MagicMock()
        mock_ubm = MagicMock()

        intent = {"intent": "nonexistent_intent"}
        allowed_guard.intent = intent

        from src.core.intelligence.guardrails import GuardrailResult

        with (
            patch(
                "src.bot.handlers.free_text_pipeline.guard_intent",
                return_value=allowed_guard,
            ),
            patch(
                "src.bot.handlers.free_text_pipeline.guardrail_evaluate",
                return_value=GuardrailResult(needs_confirm=False),
            ),
            patch(
                "src.bot.handlers.free_text_pipeline._execute_intent",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            await _dispatch(intent, msg, mock_state, mock_ubm, tz_name="UTC")

        mock_exec.assert_called_once_with(
            intent, msg, mock_state, mock_ubm, tz_name="UTC"
        )

    @pytest.mark.asyncio
    async def test_execute_intent_known_classic(self, msg):
        """Известный классический intent → правильный handler."""
        mock_handler = AsyncMock()
        mock_state = MagicMock()
        mock_ubm = MagicMock()

        intent = {"intent": "chat", "text": "hello"}

        with patch.dict(
            "src.bot.handlers.free_text_pipeline.CLASSIC_INTENT_HANDLERS",
            {"chat": (mock_handler, "Чат")},
            clear=True,
        ):
            await _execute_intent(intent, msg, mock_state, mock_ubm, tz_name="UTC")

        mock_handler.assert_called_once_with(
            intent, msg, mock_state, mock_ubm, tz_name="UTC"
        )

    @pytest.mark.asyncio
    async def test_execute_intent_unknown_classic(self, msg):
        """Неизвестный классический intent → сообщение об ошибке."""
        msg.answer = AsyncMock()
        mock_state = MagicMock()
        mock_ubm = MagicMock()

        intent = {"intent": "nonexistent_classic"}

        await _execute_intent(intent, msg, mock_state, mock_ubm, tz_name="UTC")

        msg.answer.assert_called_once()
        call_arg = msg.answer.call_args[0][0]
        assert "Неизвестный intent" in call_arg


# ===========================================================================
# TestDagDispatch
# ===========================================================================


@pytest.mark.asyncio
async def _run_dag(patch_dispatch, sub_intents, msg, state, ubm):
    """Helper: вызывает _dag_dispatch с замоканным _dispatch."""
    with (
        patch("src.bot.handlers.free_text_pipeline._dispatch", patch_dispatch),
        patch(
            "src.bot.handlers.free_text_pipeline.guard_intent",
            return_value=MagicMock(allowed=True, intent=None),
        ),
    ):
        from src.bot.handlers.free_text_pipeline import _dag_dispatch

        await _dag_dispatch(sub_intents, msg, state, ubm, tz_name="UTC")


class TestDagDispatch:
    """DAG-диспетчер для multi-intent: параллельное выполнение независимых действий."""

    @pytest.mark.asyncio
    async def test_empty_list_shows_error(self, msg):
        """Пустой список → сообщение об ошибке."""
        msg.answer = AsyncMock()
        state = MagicMock()
        ubm = MagicMock()

        with patch("src.bot.handlers.free_text_pipeline._dispatch") as mock_dispatch:
            from src.bot.handlers.free_text_pipeline import _dag_dispatch

            await _dag_dispatch([], msg, state, ubm, tz_name="UTC")

        msg.answer.assert_called_once()
        assert "Не понял" in msg.answer.call_args[0][0]
        mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_intent_calls_dispatch_once(self, msg):
        """Один sub-intent → _dispatch вызывается 1 раз."""
        state = MagicMock()
        ubm = MagicMock()
        sub = {"intent": "chat", "reply": "hello"}
        mock_dispatch = AsyncMock()

        await _run_dag(mock_dispatch, [sub], msg, state, ubm)

        mock_dispatch.assert_awaited_once_with(sub, msg, state, ubm, tz_name="UTC")

    @pytest.mark.asyncio
    async def test_all_independent_run_in_parallel(self):
        """Все sub-intents без depends_on → запускаются параллельно.

        Проверка: с задержкой в 0.2с на каждый, 3 элемента не дадут 0.6с при параллелизме.
        """
        msg = MagicMock()
        msg.answer = AsyncMock()
        msg.from_user.id = 1
        msg.text = "test"
        state = MagicMock()
        ubm = MagicMock()

        call_order = []

        async def _slow_dispatch(intent, *args, **kwargs):
            await asyncio.sleep(0.2)
            call_order.append(intent.get("intent"))

        subs = [
            {"intent": "chat", "reply": "a"},
            {"intent": "store_memory", "fact": "b"},
            {"intent": "add_reminder", "text": "c"},
        ]

        t0 = time.monotonic()
        await _run_dag(_slow_dispatch, subs, msg, state, ubm)
        elapsed = time.monotonic() - t0

        # Должно быть ~0.2с (параллельно), а не ~0.6с (последовательно)
        assert elapsed < 0.5, (
            f"Expected parallel execution (<0.5s), got {elapsed:.2f}s — "
            "actions were likely executed sequentially"
        )
        assert len(call_order) == 3
        assert set(call_order) == {"chat", "store_memory", "add_reminder"}

    @pytest.mark.asyncio
    async def test_depends_on_executes_sequential_levels(self, msg):
        """sub-intents с depends_on выполняются по уровням."""
        state = MagicMock()
        ubm = MagicMock()
        call_log = []

        async def _logging_dispatch(intent, *args, **kwargs):
            call_log.append(intent.get("intent"))

        # action_0 (level 0) → action_1 (level 1, depends_on: [0])
        subs = [
            {"intent": "search", "query": "контакт", "depends_on": []},
            {
                "intent": "send_message",
                "recipient": "...",
                "text": "Привет!",
                "depends_on": [0],
            },
        ]

        await _run_dag(_logging_dispatch, subs, msg, state, ubm)

        # Проверяем порядок: search → send_message
        assert call_log == ["search", "send_message"], (
            f"Expected ordered execution [search, send_message], got {call_log}"
        )

    @pytest.mark.asyncio
    async def test_chain_dependency(self, msg):
        """Цепочка A→B→C: строгий порядок по уровням."""
        state = MagicMock()
        ubm = MagicMock()
        call_log = []

        async def _chain_dispatch(intent, *args, **kwargs):
            call_log.append(intent.get("intent"))

        subs = [
            {"intent": "search", "query": "x", "depends_on": []},
            {"intent": "draft", "text": "y", "depends_on": [0]},
            {"intent": "send_message", "text": "z", "depends_on": [1]},
        ]

        await _run_dag(_chain_dispatch, subs, msg, state, ubm)

        assert call_log == ["search", "draft", "send_message"], (
            f"Expected [search, draft, send_message], got {call_log}"
        )

    @pytest.mark.asyncio
    async def test_cycle_falls_back_to_sequential(self, msg):
        """Циклическая зависимость → fallback на последовательное выполнение."""
        state = MagicMock()
        ubm = MagicMock()
        call_log = []

        async def _cycle_dispatch(intent, *args, **kwargs):
            call_log.append(intent.get("intent"))
            await asyncio.sleep(0.05)

        # A→B→C→A — цикл
        subs = [
            {"intent": "a", "depends_on": [2]},
            {"intent": "b", "depends_on": [0]},
            {"intent": "c", "depends_on": [1]},
        ]

        with patch("src.bot.handlers.free_text_pipeline.logger") as mock_logger:
            await _run_dag(_cycle_dispatch, subs, msg, state, ubm)

        assert len(call_log) == 3, f"Expected 3 calls, got {len(call_log)}"
        # Sequential: order preserved even with cycle
        assert call_log == ["a", "b", "c"], (
            f"Expected fallback order [a, b, c], got {call_log}"
        )
        mock_logger.warning.assert_called()
        warning_text = mock_logger.warning.call_args[0][0]
        assert "cycle" in warning_text.lower()

    @pytest.mark.asyncio
    async def test_invalid_depends_on_ignored(self, msg):
        """Невалидные depends_on (отрицательные, самоссылки, вне диапазона) игнорируются."""
        state = MagicMock()
        ubm = MagicMock()
        call_log = []

        async def _ignore_dispatch(intent, *args, **kwargs):
            call_log.append(intent.get("intent"))

        subs = [
            {"intent": "a", "depends_on": [-1]},
            {"intent": "b", "depends_on": [999]},
            {"intent": "c", "depends_on": [1]},  # 1 — сам себе, игнорируется
        ]

        await _run_dag(_ignore_dispatch, subs, msg, state, ubm)

        # Все независимы (невалидные deps проигнорированы) → все параллельно
        assert len(call_log) == 3
        assert set(call_log) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_mixed_deps_and_independent(self, msg):
        """Смешанный граф: level 0 (независимые + те, чьи deps выполнены) и level 1."""
        state = MagicMock()
        ubm = MagicMock()
        call_log = []

        async def _mixed_dispatch(intent, *args, **kwargs):
            call_log.append(intent.get("intent"))

        # 0: search (depends_on: []) — level 0
        # 1: store_memory (no depends_on) — level 0 (параллельно с search)
        # 2: send_message (depends_on: [0]) — level 1 (после search)
        subs = [
            {"intent": "search", "query": "контакт", "depends_on": []},
            {"intent": "store_memory", "fact": "запрос", "depends_on": []},
            {"intent": "send_message", "text": "Привет!", "depends_on": [0]},
        ]

        await _run_dag(_mixed_dispatch, subs, msg, state, ubm)

        # Level 0: search и store_memory (оба на level 0)
        # Level 1: send_message (зависит от search)
        assert call_log[2] == "send_message", (
            f"send_message должен быть последним, got {call_log}"
        )
        assert "search" in call_log[:2], f"search должен быть в level 0, got {call_log}"
        assert "store_memory" in call_log[:2], (
            f"store_memory должен быть в level 0, got {call_log}"
        )
