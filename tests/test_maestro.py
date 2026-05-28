"""Tests for Maestro orchestrator — the most critical file in the project.

Tests: process(), run_pipeline(), _execute_agent(), _execute_agents_parallel(),
_agent_result_as_text() — 17+ tests covering JSON parsing, error handling,
agent dispatch, formatting, and full pipeline integration.
"""

from __future__ import annotations

import asyncio
import json
import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest
from unittest import mock

from src.llm.router import ExhaustedError
from src.core.intelligence.maestro import (
    process,
    run_pipeline,
    _agent_result_as_text,
    _execute_agent,
    _execute_agents_parallel,
    _extract_json_object,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_db():
    """Recreate in-memory SQLite tables before each test."""
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
    """Returns predefined text for process() to parse."""

    def __init__(self, response_text: str = "{}"):
        self._response = response_text
        self.name = "fake"
        self.messages = None

    async def chat(
        self, messages, *, heavy: bool = False, task_type: str = "default"
    ) -> str:
        self.messages = messages
        return self._response

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 768


# ===========================================================================
# process() tests (10 tests)
# ===========================================================================


class TestProcess:
    """Tests for maestro.process — the main entry point."""

    @pytest.mark.asyncio
    async def test_returns_simple_response(self):
        """Valid JSON with no agents → result has all expected keys."""
        response = json.dumps(
            {
                "understood": "greeting",
                "plan": [],
                "agents_to_call": [],
                "final_response": "Привет!",
                "needs_clarification": None,
            }
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "Привет", owner_id=None, rag_enabled=False)
        assert result["understood"] == "greeting"
        assert result["plan"] == []
        assert result["agents_to_call"] == []
        assert result["final_response"] == "Привет!"
        assert result.get("needs_clarification") is None

    @pytest.mark.asyncio
    async def test_parses_json_with_markdown_fence(self):
        """JSON wrapped in ```json ... ``` → stripped and parsed correctly."""
        response = (
            "```json\n"
            '{"understood": "query", "plan": [], "agents_to_call": [], '
            '"final_response": "Ок", "needs_clarification": null}\n'
            "```"
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "что-то", owner_id=None, rag_enabled=False)
        assert result["understood"] == "query"
        assert result["final_response"] == "Ок"

    @pytest.mark.asyncio
    async def test_parses_json_without_lang_tag(self):
        """JSON wrapped in ``` (no language tag) → parsed correctly."""
        response = (
            "```\n"
            '{"understood": "bare", "plan": [], "agents_to_call": [], '
            '"final_response": "yes", "needs_clarification": null}\n'
            "```"
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "ok", owner_id=None, rag_enabled=False)
        assert result["understood"] == "bare"

    def test_extract_json_object_uses_first_valid_object(self):
        raw = 'before {not json} {"final_response": "ok"} {"final_response": "late"}'

        assert _extract_json_object(raw) == {"final_response": "ok"}

    @pytest.mark.asyncio
    async def test_parses_first_json_when_multiple_objects_present(self):
        response = (
            "noise {not json} "
            '{"understood": "first", "plan": [], "agents_to_call": [], '
            '"final_response": "ok", "needs_clarification": null} '
            '{"understood": "second"}'
        )
        provider = FakeLLMProvider(response)

        result = await process(provider, "ok", owner_id=None, rag_enabled=False)

        assert result["understood"] == "first"
        assert result["final_response"] == "ok"

    @pytest.mark.asyncio
    async def test_with_agents_to_call(self):
        """Plan includes agents → agents_to_call list populated."""
        response = json.dumps(
            {
                "understood": "search contact",
                "plan": ["find Alice", "draft reply"],
                "agents_to_call": [
                    {"agent": "search", "query": "Alice", "cache": True}
                ],
                "final_response": "Сейчас найду...",
                "needs_clarification": None,
            }
        )
        provider = FakeLLMProvider(response)
        result = await process(
            provider, "найди Алису", owner_id=None, rag_enabled=False
        )
        assert len(result["agents_to_call"]) == 1
        assert result["agents_to_call"][0]["agent"] == "search"

    @pytest.mark.asyncio
    async def test_needs_clarification(self):
        """needs_clarification is not null → result contains question."""
        response = json.dumps(
            {
                "understood": "ambiguous",
                "plan": [],
                "agents_to_call": [],
                "final_response": "Уточни...",
                "needs_clarification": "Кому именно написать?",
            }
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "напиши ему", owner_id=None, rag_enabled=False)
        assert result["needs_clarification"] == "Кому именно написать?"

    @pytest.mark.asyncio
    async def test_handles_exhausted_error(self):
        """Provider raises ExhaustedError → result has error/exhausted keys."""

        class ExhaustingProvider:
            name = "exhausted"

            async def chat(self, messages, *, heavy=False, task_type="default"):
                raise ExhaustedError("no keys left")

            async def embed(self, text):
                return [0.0] * 768

        provider = ExhaustingProvider()
        result = await process(provider, "Привет", owner_id=None, rag_enabled=False)
        assert result["plan"] == []
        assert result["agents_to_call"] == []
        assert "ключ" in result["final_response"].lower()

    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        """Provider raises asyncio.TimeoutError → result has timeout understood."""

        class TimeoutProvider:
            name = "timeout"

            async def chat(self, messages, *, heavy=False, task_type="default"):
                raise asyncio.TimeoutError("timed out")

            async def embed(self, text):
                return [0.0] * 768

        provider = TimeoutProvider()
        result = await process(provider, "Привет", owner_id=None, rag_enabled=False)
        assert result["understood"] == "таймаут"

    @pytest.mark.asyncio
    async def test_handles_context_overflow(self):
        """Provider raises Exception with 'context_length' → error result."""

        class OverflowProvider:
            name = "overflow"

            async def chat(self, messages, *, heavy=False, task_type="default"):
                raise RuntimeError("context_length exceeded: too many tokens")

            async def embed(self, text):
                return [0.0] * 768

        provider = OverflowProvider()
        result = await process(provider, "long text", owner_id=None, rag_enabled=False)
        assert result["understood"] == "контекст переполнен"

    @pytest.mark.asyncio
    async def test_handles_rate_limit(self):
        """Provider raises Exception with 'rate' → error result."""

        class RateLimitProvider:
            name = "ratelimit"

            async def chat(self, messages, *, heavy=False, task_type="default"):
                raise RuntimeError("rate limit exceeded")

            async def embed(self, text):
                return [0.0] * 768

        provider = RateLimitProvider()
        result = await process(provider, "text", owner_id=None, rag_enabled=False)
        assert result["understood"] == "лимит"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        """Provider returns non-JSON text → gracefully handled, all keys present."""
        provider = FakeLLMProvider("not json at all, just random text")
        result = await process(provider, "Привет", owner_id=None, rag_enabled=False)
        assert "final_response" in result
        assert isinstance(result["plan"], list)
        assert isinstance(result["agents_to_call"], list)

    @pytest.mark.asyncio
    async def test_with_empty_text(self):
        """user_text="" → still calls LLM, gets valid response dict."""
        response = json.dumps(
            {
                "understood": "empty input",
                "plan": [],
                "agents_to_call": [],
                "final_response": "Ты что-то хотел?",
                "needs_clarification": None,
            }
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "", owner_id=None, rag_enabled=False)
        assert "final_response" in result
        assert result["understood"] == "empty input"

    @pytest.mark.asyncio
    async def test_preserves_understood_field(self):
        """Verify understood is always present in result dict."""
        response = json.dumps(
            {
                "understood": "user wants to search",
                "plan": ["step1"],
                "agents_to_call": [],
                "final_response": "ok",
                "needs_clarification": None,
            }
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "поищи", owner_id=None, rag_enabled=False)
        assert result["understood"] == "user wants to search"

    @pytest.mark.asyncio
    async def test_handles_non_dict_json(self):
        """LLM returns a number (no braces) → falls through gracefully."""
        provider = FakeLLMProvider("42")
        result = await process(provider, "?", owner_id=None, rag_enabled=False)
        assert "final_response" in result
        assert result["plan"] == []

    @pytest.mark.asyncio
    async def test_handles_plain_string_json(self):
        """LLM returns a JSON string literal → falls through gracefully."""
        provider = FakeLLMProvider('"just text"')
        result = await process(provider, "?", owner_id=None, rag_enabled=False)
        assert "final_response" in result
        assert isinstance(result["plan"], list)

    @pytest.mark.asyncio
    async def test_json_with_whitespace(self):
        """JSON with leading/trailing whitespace → still parsed."""
        response = (
            '\n  \n  {"understood": "ws", "plan": [], '
            '"agents_to_call": [], "final_response": "ok", '
            '"needs_clarification": null}  \n'
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "test", owner_id=None, rag_enabled=False)
        assert result["understood"] == "ws"

    @pytest.mark.asyncio
    async def test_json_interleaved_with_text(self):
        """JSON embedded in explanatory text → extracted and parsed."""
        response = (
            "Here is my plan:\n"
            '{"understood": "embedded", "plan": [], "agents_to_call": [], '
            '"final_response": "done", "needs_clarification": null}\n'
            "And that is all."
        )
        provider = FakeLLMProvider(response)
        result = await process(provider, "test", owner_id=None, rag_enabled=False)
        assert result["understood"] == "embedded"

    @pytest.mark.asyncio
    async def test_multiple_agents_to_call(self):
        """Multiple agents in plan → all present in result."""
        response = json.dumps(
            {
                "understood": "multi agent",
                "plan": ["step1", "step2", "step3"],
                "agents_to_call": [
                    {"agent": "search", "query": "Alice", "cache": True},
                    {"agent": "memory", "query": "Alice facts", "cache": True},
                    {"agent": "draft", "query": "reply to Alice", "cache": False},
                ],
                "final_response": "Сейчас всё проверю...",
                "needs_clarification": None,
            }
        )
        provider = FakeLLMProvider(response)
        result = await process(
            provider, "отправь Алисе письмо", owner_id=None, rag_enabled=False
        )
        assert len(result["agents_to_call"]) == 3
        agent_types = [a["agent"] for a in result["agents_to_call"]]
        assert "search" in agent_types
        assert "memory" in agent_types
        assert "draft" in agent_types

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        """Unknown/unexpected exception → generic 'не понял' error result."""

        class GenericErrorProvider:
            name = "generic_error"

            async def chat(self, messages, *, heavy=False, task_type="default"):
                raise RuntimeError("something unexpected happened")

            async def embed(self, text):
                return [0.0] * 768

        provider = GenericErrorProvider()
        result = await process(provider, "test", owner_id=None, rag_enabled=False)
        assert result["understood"] == "не понял"
        assert "Попробуй одну из команд" in result["final_response"]


# ===========================================================================
# _agent_result_as_text() tests (4 tests)
# ===========================================================================


class TestAgentResultAsText:
    """Tests for _agent_result_as_text — agent output formatting."""

    def test_formats_correctly(self):
        """Search agent with contacts → formatted string with agent type and data."""
        result = {
            "success": True,
            "data": {"contacts": ["Alice", "Bob"]},
        }
        text = _agent_result_as_text("search", result)
        assert "[search]" in text
        assert "Alice" in text
        assert "Bob" in text

    def test_empty_result(self):
        """data={} → returns 'данных нет' message."""
        result = {"success": True, "data": {}}
        text = _agent_result_as_text("memory", result)
        assert "данных нет" in text

    def test_agent_error(self):
        """Agent failed (success=False) → error message included with ❌."""
        result = {
            "success": False,
            "error": "connection refused",
            "data": {},
        }
        text = _agent_result_as_text("summarizer", result)
        assert "connection refused" in text
        assert "❌" in text

    def test_data_truncated(self):
        """Long data values are truncated at 400 chars (… suffix)."""
        long_value = "X" * 600
        result = {"success": True, "data": {"long_field": long_value}}
        text = _agent_result_as_text("digest", result)
        assert "…" in text
        assert len(text) < len(long_value) + 200


# ===========================================================================
# _execute_agent() tests (1 test)
# ===========================================================================


class TestExecuteAgent:
    """Tests for _execute_agent — single agent dispatch."""

    @pytest.mark.asyncio
    async def test_unknown_type(self):
        """agent_type="nonexistent" → error dict, does not crash."""
        spec = {"agent": "nonexistent", "query": "anything"}
        result = await _execute_agent(FakeLLMProvider(), spec, owner_id=123456789)
        assert result["success"] is False
        assert "Неизвестный агент" in result["error"]


# ===========================================================================
# _execute_agents_parallel() tests (2 tests)
# ===========================================================================


class TestExecuteAgentsParallel:
    """Tests for _execute_agents_parallel — parallel agent execution."""

    @pytest.mark.asyncio
    async def test_runs_multiple(self):
        """Two agent specs → both called via asyncio.gather, results collected."""

        async def fake_execute(_provider, spec, *, owner_id):
            return {"data": {spec["agent"]: "done"}, "success": True}

        specs = [
            {"agent": "search", "query": "test1"},
            {"agent": "memory", "query": "test2"},
        ]
        with mock.patch(
            "src.core.intelligence.maestro._execute_agent",
            side_effect=fake_execute,
        ) as mock_exec:
            results = await _execute_agents_parallel(
                FakeLLMProvider(), specs, owner_id=123456789
            )
            assert len(results) == 2
            assert mock_exec.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Empty agents list → returns empty list immediately."""
        results = await _execute_agents_parallel(
            FakeLLMProvider(), [], owner_id=123456789
        )
        assert results == []


# ===========================================================================
# run_pipeline() integration tests (3 tests)
# ===========================================================================


class TestRunPipeline:
    """Integration tests for run_pipeline — full orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_simple_response_no_agents(self):
        """process returns response without agents → returns final_response immediately."""
        plan_response = {
            "understood": "greeting",
            "plan": [],
            "agents_to_call": [],
            "final_response": "Привет, как дела?",
            "needs_clarification": None,
        }

        async def fake_process(_provider, _user_text, **kwargs):
            return plan_response

        with mock.patch(
            "src.core.intelligence.maestro.process", side_effect=fake_process
        ):
            result = await run_pipeline(
                FakeLLMProvider(),
                "привет",
                owner_id=123456789,
                self_profile="",
            )
            assert "final_response" in result
            assert result["used_agents"] == []
            assert result["agent_errors"] == []
            assert "Привет" in result["final_response"]

    @pytest.mark.asyncio
    async def test_run_pipeline_forwards_memory_context(self):
        """RoutingPlan memory must reach Maestro process."""
        seen_kwargs = {}

        async def fake_process(_provider, _user_text, **kwargs):
            seen_kwargs.update(kwargs)
            return {
                "understood": "memory aware",
                "plan": [],
                "agents_to_call": [],
                "final_response": "Помню контекст",
                "needs_clarification": None,
            }

        with mock.patch(
            "src.core.intelligence.maestro.process", side_effect=fake_process
        ):
            await run_pipeline(
                FakeLLMProvider(),
                "что у меня с проектом?",
                owner_id=123456789,
                memory_context="<recall_context>важный факт</recall_context>",
                self_profile="",
            )

        assert seen_kwargs["memory_context"] == (
            "<recall_context>важный факт</recall_context>"
        )

    @pytest.mark.asyncio
    async def test_process_puts_memory_and_self_profile_in_system_prompt(self):
        provider = FakeLLMProvider(
            json.dumps(
                {
                    "understood": "memory aware",
                    "plan": [],
                    "agents_to_call": [],
                    "final_response": "ok",
                    "needs_clarification": None,
                }
            )
        )

        await process(
            provider,
            "what do you remember?",
            owner_id=None,
            rag_enabled=False,
            memory_context="<recall_context>system-only fact</recall_context>",
            self_profile="[self-profile] system-only profile",
        )

        system_prompt = provider.messages[0].content
        user_prompt = provider.messages[1].content
        assert "<recall_context>system-only fact</recall_context>" in system_prompt
        assert "[self-profile] system-only profile" in system_prompt
        assert "<recall_context>system-only fact</recall_context>" not in user_prompt

    @pytest.mark.asyncio
    async def test_clarification_response(self):
        """process returns needs_clarification → is_clarification=True, question shown."""
        plan_response = {
            "understood": "ambiguous",
            "plan": [],
            "agents_to_call": [],
            "final_response": "",
            "needs_clarification": "Кому написать?",
        }

        async def fake_process(_provider, _user_text, **kwargs):
            return plan_response

        with mock.patch(
            "src.core.intelligence.maestro.process", side_effect=fake_process
        ):
            result = await run_pipeline(
                FakeLLMProvider(),
                "напиши ему",
                owner_id=123456789,
                self_profile="",
            )
            assert result.get("is_clarification") is True
            assert "Кому написать" in result["final_response"]

    @pytest.mark.asyncio
    async def test_with_agents_and_synthesis(self):
        """Agents run successfully → results synthesized via LLM after-agents prompt."""
        plan_response = {
            "understood": "search and draft",
            "plan": ["find", "draft"],
            "agents_to_call": [
                {"agent": "search", "query": "Alice", "cache": True},
            ],
            "final_response": "Сейчас гляну...",
            "needs_clarification": None,
        }
        agent_result = {
            "data": {"name": "Alice", "id": 123},
            "success": True,
            "agent": "search",
        }
        synthesis_json = '{"final_response": "Нашла Алису! ID: 123"}'

        async def fake_process(_provider, _user_text, **kwargs):
            return plan_response

        async def fake_execute_parallel(_agents, _provider, _owner_id):
            return [agent_result], []  # orchestrator returns (results, errors)

        provider = FakeLLMProvider(synthesis_json)
        with (
            mock.patch(
                "src.core.intelligence.maestro.process", side_effect=fake_process
            ),
            mock.patch(
                "src.core.intelligence.maestro.orchestrator.execute",
                side_effect=fake_execute_parallel,
            ),
        ):
            result = await run_pipeline(
                provider,
                "найди Алису",
                owner_id=123456789,
                self_profile="",
            )
            assert "final_response" in result
            assert "used_agents" in result
            assert "agent_errors" in result
            assert "search" in result["used_agents"]
