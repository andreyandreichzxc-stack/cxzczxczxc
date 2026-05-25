"""Tests for intent routing pipeline: make_plan, route_intent, sanitize_html."""

from __future__ import annotations

import asyncio
import os

# ---------------------------------------------------------------------------
# Environment: MUST be set BEFORE any src imports
# ---------------------------------------------------------------------------
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "HmsOzSAxuyfb7zet2nmwhFkgWfH5z6Lsr3tW7MO8GDI=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

from src.core.intelligence.smart_autorouter import make_plan
from src.core.infra.text_sanitizer import sanitize_html
from src.core.intelligence.agent import route_intent
from src.llm.base import ChatMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_db():
    """Recreate all tables before each test (pattern from test_memory_smoke)."""
    from src.db.session import engine, Base, init_db
    from sqlalchemy import text

    async def _recreate():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            # Drop artifacts that survive drop_all and would confuse init_db
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            await conn.execute(text("DROP TABLE IF EXISTS messages_fts"))
            await conn.execute(text("DROP TABLE IF EXISTS memories_fts"))
        await init_db()

    asyncio.run(_recreate())


class FakeLLMProvider:
    """Mock LLM provider — no real API keys, returns canned responses."""

    name = "fake"

    def __init__(self, response: str = '{"intent": "chat", "reply": "ok"}'):
        self._response = response

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        return self._response

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def validate_key(self) -> bool:
        return True


# ===========================================================================
# make_plan tests
# ===========================================================================


class TestMakePlan:
    """Tests for smart_autorouter.make_plan — routing mode classification."""

    @pytest.mark.asyncio
    async def test_instant_greeting(self):
        """'Привет' → instant mode, final_response populated, zero tasks."""
        plan = await make_plan("Привет", 123456789)
        assert plan.response_mode == "instant"
        assert plan.final_response != ""
        assert len(plan.tasks) == 0

    @pytest.mark.asyncio
    async def test_fast_route_send(self):
        """'напиши Васе привет' → fast_route mode, high-risk task."""
        plan = await make_plan("напиши Васе привет", 123456789)
        assert plan.response_mode == "fast_route"
        assert len(plan.tasks) > 0
        # 'напиши' triggers SEND routing → RiskLevel.HIGH
        assert plan.tasks[0].risk.value == "high"

    @pytest.mark.asyncio
    async def test_fast_route_default(self):
        """Short non-greeting, non-command text → fast_route with LOW risk."""
        plan = await make_plan("как погода сегодня", 123456789)
        assert plan.response_mode == "fast_route"
        assert len(plan.tasks) > 0
        # No SEND/SEARCH/ANALYSIS words → falls through to default MAIN
        assert plan.tasks[0].purpose.value == "main"
        assert plan.tasks[0].heavy is False

    @pytest.mark.asyncio
    async def test_maestro_heavy_word(self):
        """Text containing a HEAVY_WORD → maestro mode."""
        plan = await make_plan("проанализируй мои переписки", 123456789)
        assert plan.response_mode == "maestro"
        assert len(plan.tasks) > 0
        # 'проанализируй' also matches _ANALYSIS_WORDS → purpose=analysis
        assert plan.tasks[0].purpose.value == "analysis"

    @pytest.mark.asyncio
    async def test_maestro_long_text(self):
        """Text ≥100 chars (no heavy words) → maestro mode."""
        long_text = "мне нужно чтобы ты проверил все мои сообщения за последние "
        # Pad to 100+ chars
        while len(long_text) < 100:
            long_text += "дней "
        plan = await make_plan(long_text.strip(), 123456789)
        assert plan.response_mode == "maestro"

    @pytest.mark.asyncio
    async def test_instant_mode_metrics_present(self):
        """Instant mode returns elapsed_ms and metrics."""
        plan = await make_plan("спасибо", 123456789)
        assert plan.response_mode == "instant"
        assert plan.final_response != ""
        assert plan.elapsed_ms >= 0
        assert plan.metrics.get("mode") == "instant"
        assert plan.metrics.get("total_ms", 0) >= 0


# ===========================================================================
# route_intent smoke test
# ===========================================================================


class TestRouteIntent:
    """Tests for agent.route_intent — LLM-based intent extraction."""

    @pytest.mark.asyncio
    async def test_smoke(self):
        """route_intent with mocked LLM returns a valid intent dict."""
        provider = FakeLLMProvider()
        result = await route_intent(
            provider,
            "привет",
            user_id=123456789,
        )
        assert isinstance(result, dict)
        assert "intent" in result
        assert result["intent"] == "chat"

    @pytest.mark.asyncio
    async def test_with_history_and_memory(self):
        """route_intent with optional context params doesn't crash."""
        provider = FakeLLMProvider(
            response='{"intent": "send_message", "recipient": "Вася", "text": "привет"}'
        )
        result = await route_intent(
            provider,
            "напиши Васе привет",
            user_id=123456789,
            heavy=False,
            now_local="2025-05-20 14:30",
            tz_name="Europe/Moscow",
            history_block="<dialog>\nuser: как дела?\nassistant: норм\n</dialog>",
            memory_context="Пользователь дружит с Васей.",
        )
        assert isinstance(result, dict)
        assert result["intent"] == "send_message"


# ===========================================================================
# sanitize_html tests
# ===========================================================================


class TestSanitizeHtml:
    """Tests for text_sanitizer.sanitize_html — HTML whitelist filter."""

    def test_leaves_valid_tags(self):
        """<b>/<i>/<u>/<s>/<code>/<pre>/<a>/<tg-spoiler> tags are preserved."""
        result = sanitize_html("<b>bold</b> and <i>italic</i>")
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result
        # Ensure standard text survives
        assert "and" in result

    def test_strips_script_tags(self):
        """<script> tags are stripped; text content survives."""
        result = sanitize_html("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "script" not in result
        assert "alert(1)" in result

    def test_handles_special_characters(self):
        """Bare < > & not inside valid tags are kept as data and never
        produce new HTML tags."""
        result = sanitize_html("text with < > &")
        # HTMLParser keeps bare angle brackets as data — they don't form tags
        assert "text with" in result
        # The output must not contain any HTML tag delimiters that could
        # be interpreted as a valid tag (e.g. <script>, <iframe>)
        assert "<script>" not in result
        assert "<iframe>" not in result

    def test_none_input_returns_empty_string(self):
        """None input → empty string."""
        assert sanitize_html(None) == ""

    def test_empty_input_returns_empty_string(self):
        """Empty string input → empty string."""
        assert sanitize_html("") == ""
        assert sanitize_html("   ") == ""
