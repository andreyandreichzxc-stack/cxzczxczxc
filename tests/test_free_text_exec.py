"""Unit tests for exec handler functions in free_text_exec.py."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

from src.bot.handlers.free_text_exec import (
    exec_add_api_key,
    exec_classic_chat,
    exec_classic_find_in_chats,
    exec_classic_list_todos,
    exec_classic_search,
    exec_classic_summarize_chat,
    exec_classic_unknown,
    exec_clarify,
    exec_index_chats,
    exec_list_keys,
    exec_list_memories,
    exec_remove_api_key,
    exec_remove_reminder,
    exec_set_setting,
    exec_show_digest,
    exec_show_inbox,
    exec_show_self,
    exec_show_skills,
    exec_show_threads,
    exec_show_today,
    exec_store_memory,
    exec_toggle_api_key,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_msg(user_id: int = 123456789) -> MagicMock:
    m = MagicMock()
    m.from_user.id = user_id
    m.answer = AsyncMock()
    return m


def _make_userbot_manager(has_client: bool = True) -> MagicMock:
    ubm = MagicMock()
    ubm.get_client.return_value = MagicMock() if has_client else None
    return ubm


def _mock_session(*, get_return=None):
    """Return a mock async context manager for get_session()."""
    cm = MagicMock()
    sess = MagicMock()
    sess.get = AsyncMock(return_value=get_return)
    sess.commit = AsyncMock()
    sess.delete = MagicMock()
    sess.flush = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=sess)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


# ===========================================================================
# TestKeyHandlers
# ===========================================================================


class TestKeyHandlers:
    """Handler-функции управления ключами — list_keys, remove_api_key, toggle_api_key."""

    @pytest.mark.asyncio
    async def test_list_keys_empty(self):
        msg = _make_msg()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch("src.bot.handlers.free_text_exec.list_key_slots", return_value=[]),
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_list_keys(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Нет ключевых слотов" in call_text

    @pytest.mark.asyncio
    async def test_remove_key_not_found(self):
        msg = _make_msg()
        intent = {"slot_id": 999}
        with (
            patch(
                "src.bot.handlers.free_text_exec.get_session",
                return_value=_mock_session(get_return=None),
            ),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_remove_api_key(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Слот не найден или не твой" in call_text

    @pytest.mark.asyncio
    async def test_toggle_key_not_found(self):
        msg = _make_msg()
        intent = {"slot_id": 42}
        with (
            patch(
                "src.bot.handlers.free_text_exec.get_session",
                return_value=_mock_session(get_return=None),
            ),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_toggle_api_key(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Слот не найден или не твой" in call_text


# ===========================================================================
# TestSimpleHandlers
# ===========================================================================


class TestSimpleHandlers:
    """Простые handler-функции (intent, message) — _h адаптер."""

    @pytest.mark.asyncio
    async def test_clarify_with_question(self):
        msg = _make_msg()
        intent = {"question": "что ты имеешь в виду?"}
        await exec_clarify(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "что ты имеешь в виду?" in call_text

    @pytest.mark.asyncio
    async def test_clarify_without_question(self):
        msg = _make_msg()
        intent = {}
        await exec_clarify(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не совсем понял" in call_text

    @pytest.mark.asyncio
    async def test_show_self_no_profile(self):
        msg = _make_msg()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch(
                "src.bot.handlers.free_text_exec.get_self_profile", return_value=None
            ) as _mock_prof,
            patch("src.bot.handlers.free_text_exec.get_session"),
        ):
            mock_user.return_value = MagicMock(telegram_id=123456789)
            await exec_show_self(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Что я знаю о тебе" in call_text

    @pytest.mark.asyncio
    async def test_set_setting_unknown_key(self):
        msg = _make_msg()
        intent = {"key": "nonexistent", "value": "foo"}
        await exec_set_setting(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не умею менять" in call_text

    @pytest.mark.asyncio
    async def test_show_skills_empty(self):
        msg = _make_msg()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch("src.db.repo.list_skills", return_value=[]),
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_show_skills(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Навыков пока нет" in call_text

    @pytest.mark.asyncio
    async def test_show_digest_empty(self):
        msg = _make_msg()
        intent = {}
        with patch("src.core.scheduling.digest.build_digest", return_value=""):
            await exec_show_digest(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Дайджест пуст" in call_text

    @pytest.mark.asyncio
    async def test_index_chats_no_chats(self):
        msg = _make_msg()
        intent = {}
        await exec_index_chats(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "используй команду /index" in call_text


# ===========================================================================
# TestDBHandlers
# ===========================================================================


class TestDBHandlers:
    """Handler-функции с взаимодействием через БД."""

    @pytest.mark.asyncio
    async def test_store_memory_empty_fact(self):
        msg = _make_msg()
        intent = {"fact": ""}
        await exec_store_memory(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не понял, что запомнить" in call_text

    @pytest.mark.asyncio
    async def test_store_memory_with_fact_no_confidence(self):
        msg = _make_msg()
        intent = {"fact": "тестовый факт"}
        with (
            patch("src.bot.handlers.free_text_memory.get_session"),
            patch("src.bot.handlers.free_text_memory.get_or_create_user") as mock_user,
            patch(
                "src.bot.handlers.free_text_memory.add_memory_candidate",
            ) as mock_add,
            patch(
                "src.bot.handlers.free_text_memory.get_active_telethon_client",
                return_value=None,
            ),
        ):
            mock_user.return_value = MagicMock(id=1)
            mock_add.return_value = MagicMock()
            await exec_store_memory(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "черновик" in call_text
        assert "тестовый факт" in call_text

    @pytest.mark.asyncio
    async def test_list_memories_empty(self):
        msg = _make_msg()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_memory.get_session"),
            patch("src.bot.handlers.free_text_memory.get_or_create_user") as mock_user,
            patch(
                "src.bot.handlers.free_text_memory.list_memories", return_value=[]
            ) as _mock_list,
            patch(
                "src.bot.handlers.free_text_memory.get_active_telethon_client",
                return_value=None,
            ),
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_list_memories(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Память пуста" in call_text

    @pytest.mark.asyncio
    async def test_add_api_key_missing_data(self):
        msg = _make_msg()
        intent = {}
        await exec_add_api_key(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Укажи провайдера" in call_text

    @pytest.mark.asyncio
    async def test_add_api_key_wrong_provider(self):
        msg = _make_msg()
        intent = {"provider": "invalid_provider", "key": "sk-test"}
        await exec_add_api_key(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Провайдер:" in call_text

    @pytest.mark.asyncio
    async def test_remove_reminder_no_reminders(self):
        msg = _make_msg()
        intent = {"query": "несуществующее"}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch(
                "src.bot.handlers.free_text_exec.list_open_commitments", return_value=[]
            ),
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_remove_reminder(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не нашёл напоминаний" in call_text

    @pytest.mark.asyncio
    async def test_show_threads_empty(self):
        msg = _make_msg()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch("src.db.repo.list_active_conversations", return_value=[]),
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_show_threads(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Активных тредов нет" in call_text

    @pytest.mark.asyncio
    async def test_show_today_empty(self):
        msg = _make_msg()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch(
                "src.core.scheduling.smart_digest.collect_recent_messages",
                return_value=[],
            ),
            patch(
                "src.core.scheduling.smart_digest.build_smart_digest", return_value=""
            ),
        ):
            user_mock = MagicMock()
            user_mock.settings.smart_digest_interval_min = 60
            mock_user.return_value = user_mock
            await exec_show_today(intent, msg)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "На сегодня ничего" in call_text


# ===========================================================================
# TestUserbotHandlers
# ===========================================================================


class TestUserbotHandlers:
    """Handler-функции с userbot_manager — _hu адаптер."""

    @pytest.mark.asyncio
    async def test_show_inbox_empty(self):
        msg = _make_msg()
        ubm = _make_userbot_manager(has_client=False)
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch("src.db.repo.list_active_conversations", return_value=[]),
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_show_inbox(intent, msg, ubm)
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Нет активных переписок" in call_text


# ===========================================================================
# TestClassicHandlers
# ===========================================================================


class TestClassicHandlers:
    """Классические handler-функции (intent, message, state, ubm, *, tz_name)."""

    @pytest.mark.asyncio
    async def test_classic_unknown(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = MagicMock()
        intent = {}
        await exec_classic_unknown(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не понял, что нужно сделать" in call_text

    @pytest.mark.asyncio
    async def test_classic_list_todos_empty(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = MagicMock()
        intent = {}
        with (
            patch("src.bot.handlers.free_text_exec.get_session"),
            patch("src.bot.handlers.free_text_exec.get_or_create_user") as mock_user,
            patch(
                "src.bot.handlers.free_text_exec.list_open_commitments", return_value=[]
            ) as _mock_list,
        ):
            mock_user.return_value = MagicMock(id=1)
            await exec_classic_list_todos(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Открытых обязательств нет" in call_text

    @pytest.mark.asyncio
    async def test_classic_chat_with_reply(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = MagicMock()
        intent = {"reply": "Привет! Как дела?"}
        await exec_classic_chat(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Привет! Как дела?" in call_text

    @pytest.mark.asyncio
    async def test_classic_chat_no_reply(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = MagicMock()
        intent = {}
        await exec_classic_chat(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Готов помочь" in call_text

    @pytest.mark.asyncio
    async def test_search_no_query(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = _make_userbot_manager(has_client=True)
        intent = {}
        await exec_classic_search(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не понял, что искать" in call_text

    @pytest.mark.asyncio
    async def test_find_in_chats_no_query(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = _make_userbot_manager(has_client=True)
        intent = {}
        await exec_classic_find_in_chats(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не понял, по какой теме искать" in call_text

    @pytest.mark.asyncio
    async def test_summarize_chat_no_chat(self):
        msg = _make_msg()
        state = MagicMock()
        ubm = _make_userbot_manager(has_client=True)
        intent = {}
        await exec_classic_summarize_chat(intent, msg, state, ubm, tz_name="UTC")
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Не понял, с каким контактом работать" in call_text
