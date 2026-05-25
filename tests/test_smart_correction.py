"""Unit tests for smart_correction module."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("ENCRYPTION_KEY", "HmsOzSAxuyfb7zet2nmwhFkgWfH5z6Lsr3tW7MO8GDI=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

from src.bot.handlers.smart_correction import (
    apply_correction,
    detect_correction,
    record_action,
)


# ===========================================================================
# Helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_actions():
    """Очищать _last_actions между тестами."""
    from src.bot.handlers import smart_correction as sc

    sc._last_actions.clear()
    yield
    sc._last_actions.clear()


# ===========================================================================
# TestCorrectionDetection
# ===========================================================================


class TestCorrectionDetection:
    """detect_correction: распознавание правок/отмен."""

    @pytest.mark.asyncio
    async def test_no_previous_action_returns_none(self):
        """Без записанного действия — None."""
        result = await detect_correction(42, "нет, через два часа")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_correction_pattern_returns_none(self):
        """Обычный текст без паттернов коррекции — None."""
        await record_action(
            42, {"intent": "add_reminder", "params": {"text": "что-то"}}
        )
        result = await detect_correction(42, "привет как дела")
        assert result is None

    @pytest.mark.asyncio
    async def test_explicit_cancel_otmeni(self):
        """«отмени» — отмена последнего действия."""
        await record_action(
            42, {"intent": "add_reminder", "params": {"text": "позвонить"}}
        )
        result = await detect_correction(42, "отмени")
        assert result is not None
        assert result["action"] == "cancel"
        assert result["previous"]["intent"] == "add_reminder"

    @pytest.mark.asyncio
    async def test_explicit_cancel_ne_nado(self):
        """«не надо» — отмена."""
        await record_action(42, {"intent": "draft_reply", "params": {}})
        result = await detect_correction(42, "не надо")
        assert result is not None
        assert result["action"] == "cancel"

    @pytest.mark.asyncio
    async def test_explicit_cancel_otboy(self):
        """«отбой» — отмена."""
        await record_action(42, {"intent": "send_message", "params": {}})
        result = await detect_correction(42, "отбой")
        assert result is not None
        assert result["action"] == "cancel"

    @pytest.mark.asyncio
    async def test_bare_net_cancels(self):
        """Просто «нет» (без аргументов) — отмена."""
        await record_action(
            42, {"intent": "add_reminder", "params": {"text": "звонок"}}
        )
        result = await detect_correction(42, "нет")
        assert result is not None
        assert result["action"] == "cancel"

    @pytest.mark.asyncio
    async def test_bare_ne_tak_cancels(self):
        """«не так» — отмена."""
        await record_action(42, {"intent": "send_message", "params": {}})
        result = await detect_correction(42, "не так")
        assert result is not None
        assert result["action"] == "cancel"

    @pytest.mark.asyncio
    async def test_bare_oshibsya_cancels(self):
        """«ошибся» — отмена."""
        await record_action(42, {"intent": "draft_reply", "params": {}})
        result = await detect_correction(42, "ошибся")
        assert result is not None
        assert result["action"] == "cancel"

    @pytest.mark.asyncio
    async def test_net_with_args_replaces(self):
        """«нет, через два часа» — замена параметров."""
        await record_action(
            42, {"intent": "add_reminder", "params": {"text": "что-то"}}
        )
        result = await detect_correction(42, "нет, через два часа")
        assert result is not None
        assert result["action"] == "replace"
        assert "через два часа" in result["new_text"]

    @pytest.mark.asyncio
    async def test_ne_with_args_replaces(self):
        """«не, в среду» — замена."""
        await record_action(
            42, {"intent": "add_reminder", "params": {"text": "доклад"}}
        )
        result = await detect_correction(42, "не, в среду днём")
        assert result is not None
        assert result["action"] == "replace"

    @pytest.mark.asyncio
    async def test_net_k_contact_extraction(self):
        """«нет, для оли» — извлекает контакт."""
        await record_action(
            42, {"intent": "send_message", "params": {"text": "привет"}}
        )
        result = await detect_correction(42, "нет, контакту оля")
        assert result is not None
        assert result["action"] == "replace"
        assert result["new_params"].get("contact") == "оля"

    @pytest.mark.asyncio
    async def test_consumes_previous_action(self):
        """После коррекции прошлое действие удаляется из хранилища."""
        await record_action(42, {"intent": "add_reminder", "params": {"text": "a"}})
        result = await detect_correction(42, "нет, через час")
        assert result is not None
        # Second correction should return None (nothing to correct)
        result2 = await detect_correction(42, "отмени")
        assert result2 is None

    @pytest.mark.asyncio
    async def test_stale_action_ignored(self):
        """Просроченное действие (>60 сек) игнорируется."""
        from src.bot.handlers import smart_correction as sc

        sc._last_actions[42] = {
            "intent": "add_reminder",
            "params": {"text": "old"},
            "timestamp": 0,  # way in the past
        }
        result = await detect_correction(42, "отмени")
        assert result is None


# ===========================================================================
# TestApplyCorrection
# ===========================================================================


class TestApplyCorrection:
    """apply_correction: формирование ответа на коррекцию."""

    def test_cancel_message(self):
        corr = {
            "action": "cancel",
            "previous": {"intent": "add_reminder"},
        }
        msg = asyncio.run(apply_correction(42, corr))
        assert "Отменил" in msg
        assert "add_reminder" in msg

    def test_replace_message(self):
        corr = {
            "action": "replace",
            "previous": {"intent": "draft_reply"},
            "new_text": "через пять минут",
        }
        msg = asyncio.run(apply_correction(42, corr))
        assert "обновил" in msg
        assert "через пять минут" in msg

    def test_unknown_action_fallback(self):
        corr = {"action": "unknown"}
        msg = asyncio.run(apply_correction(42, corr))
        assert "Готово" in msg
