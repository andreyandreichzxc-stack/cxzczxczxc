"""Comprehensive tests for smart_autorouter: classify_mode, get_instant_reply, classify_risk, make_plan."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

from src.core.intelligence.smart_autorouter import (
    classify_mode,
    classify_risk,
    get_instant_reply,
    make_plan,
    ResponseMode,
    RiskLevel,
    RouterPlan,
)


# =============================================================================
# classify_mode
# =============================================================================


class TestClassifyMode:
    @pytest.mark.asyncio
    async def test_classify_instant_greeting(self):
        assert await classify_mode("привет") == ResponseMode.INSTANT

    @pytest.mark.asyncio
    async def test_classify_instant_bye(self):
        assert await classify_mode("пока") == ResponseMode.INSTANT

    @pytest.mark.asyncio
    async def test_classify_instant_ok(self):
        assert await classify_mode("ок") == ResponseMode.INSTANT

    @pytest.mark.asyncio
    async def test_classify_instant_spasibo(self):
        assert await classify_mode("спасибо") == ResponseMode.INSTANT

    @pytest.mark.asyncio
    async def test_classify_instant_case_insensitive(self):
        assert await classify_mode("ПриВеТ") == ResponseMode.INSTANT

    @pytest.mark.asyncio
    async def test_classify_fast_route_short(self):
        assert await classify_mode("расскажи новости") == ResponseMode.FAST_ROUTE

    @pytest.mark.asyncio
    async def test_classify_fast_route_boundary_99(self):
        assert await classify_mode("x" * 99) == ResponseMode.FAST_ROUTE

    @pytest.mark.asyncio
    async def test_classify_maestro_heavy_word(self):
        assert await classify_mode("сделай анализ переписки") == ResponseMode.MAESTRO

    @pytest.mark.asyncio
    async def test_classify_maestro_long(self):
        assert await classify_mode("x" * 150) == ResponseMode.MAESTRO

    @pytest.mark.asyncio
    async def test_classify_maestro_boundary_100(self):
        assert await classify_mode("x" * 100) == ResponseMode.MAESTRO


# =============================================================================
# get_instant_reply
# =============================================================================


class TestGetInstantReply:
    def test_instant_reply_exact_match(self):
        assert get_instant_reply("привет") == "Привет! 👋"

    def test_instant_reply_substring(self):
        assert get_instant_reply("спасибо большое") == "Всегда пожалуйста! 🤗"

    def test_instant_reply_no_match(self):
        assert get_instant_reply("сделай анализ") == ""

    def test_instant_reply_case_insensitive(self):
        assert get_instant_reply("ПРИВЕТ") == "Привет! 👋"


# =============================================================================
# classify_risk
# =============================================================================


class TestClassifyRisk:
    @pytest.mark.asyncio
    async def test_risk_critical_delete(self):
        assert await classify_risk("удали все сообщения") == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_risk_high_send(self):
        assert await classify_risk("отправь сообщение Оле") == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_risk_medium_search(self):
        assert await classify_risk("найди где договор") == RiskLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_risk_low_greeting(self):
        assert await classify_risk("привет") == RiskLevel.LOW


# =============================================================================
# make_plan
# =============================================================================


class TestMakePlan:
    @pytest.mark.asyncio
    async def test_make_plan_instant_mode(self):
        plan = await make_plan("привет", 123456789)
        assert plan.response_mode == "instant"
        assert plan.final_response != ""
        assert len(plan.tasks) == 0

    @pytest.mark.asyncio
    async def test_make_plan_returns_router_plan(self):
        mock_session = AsyncMock()
        mock_session.execute.return_value = MagicMock()
        mock_owner = AsyncMock()
        mock_owner.telegram_id = 123456789

        with (
            patch(
                "src.core.intelligence.routing.planner.get_session"
            ) as mock_get_session,
            patch(
                "src.core.intelligence.routing.planner.get_or_create_user",
                return_value=mock_owner,
            ),
            patch(
                "src.core.memory.memory_recall.recall",
            ) as mock_recall,
            patch(
                "src.core.memory.memory_recall.format_recall_for_prompt",
                return_value="",
            ),
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_session
            mock_cm.__aexit__.return_value = None
            mock_get_session.return_value = mock_cm
            mock_recall.return_value = MagicMock(facts=[])

            plan = await make_plan("найди договор", 123456789)
            assert isinstance(plan, RouterPlan)
            assert len(plan.tasks) > 0

    @pytest.mark.asyncio
    async def test_make_plan_without_provider(self):
        plan = await make_plan("привет", 123456789, provider_available=False)
        assert isinstance(plan, RouterPlan)
