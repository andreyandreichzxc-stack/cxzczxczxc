"""Tests for project connector registry and tool bridge."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

from src.core.actions.tool_registry import tool_registry
from src.core.connectors import (
    ConnectorActionAnnotations,
    ConnectorActionSpec,
    ConnectorRegistry,
    ConnectorResult,
    ConnectorRuntime,
    ConnectorSpec,
    connector_registry,
    register_builtin_connectors,
)
from src.core.connectors.credentials import redact_secrets
from src.core.intelligence.guardrails import ActionRisk, evaluate

import src.core.actions.mcp_connectors  # noqa: F401


async def _sample_handler(
    action: str,
    params: dict,
    runtime: ConnectorRuntime,
) -> ConnectorResult:
    return ConnectorResult(
        ok=True,
        data={"action": action, "params": params, "has_session": runtime.session is not None},
        metadata={"access_token": "secret-token-value"},
    )


def _sample_spec(name: str = "sample") -> ConnectorSpec:
    return ConnectorSpec(
        name=name,
        description="Sample connector",
        category="test",
        auth_mode="api_key",
        capabilities=("search",),
        actions=(
            ConnectorActionSpec(
                name="search",
                description="Search sample data",
                risk="low",
                annotations=ConnectorActionAnnotations(read_only=True, user_content=True),
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_connector_registry_execute_and_redacts_metadata():
    registry = ConnectorRegistry()
    registry.register(_sample_spec(), _sample_handler)

    result = await registry.execute("sample", "search", {"query": "hello"}, ConnectorRuntime(session=object()))

    assert result["ok"] is True
    assert result["data"]["action"] == "search"
    assert result["data"]["params"] == {"query": "hello"}
    assert result["data"]["has_session"] is True
    assert result["metadata"]["connector"] == "sample"
    assert result["metadata"]["action"] == "search"
    assert result["metadata"]["access_token"] == "sec...lue"
    assert result["metadata"]["annotations"]["read_only"] is True


def test_connector_registry_rejects_duplicate_names():
    registry = ConnectorRegistry()
    registry.register(_sample_spec(), _sample_handler)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_sample_spec(), _sample_handler)


def test_redact_secrets_recurses_through_dicts_and_lists():
    payload = {
        "token": "1234567890",
        "nested": [{"Authorization": "Bearer abcdefghij"}, {"safe": "visible"}],
        "query": "not secret",
    }

    assert redact_secrets(payload) == {
        "token": "123...890",
        "nested": [{"Authorization": "Bea...hij"}, {"safe": "visible"}],
        "query": "not secret",
    }


def test_redact_secrets_redacts_secret_url_query_values():
    result = redact_secrets(
        {
            "source_url": (
                "https://cdn.example.com/file.zip?"
                "X-Amz-Signature=1234567890abcdef&plain=visible&access_token=abcdef123456"
            )
        }
    )

    assert result == {
        "source_url": "https://cdn.example.com/file.zip?X-Amz-Signature=123...def&plain=visible&access_token=abc...456"
    }


def test_redact_secrets_redacts_url_userinfo():
    result = redact_secrets({"url": "https://user:password@example.com/path?plain=visible"})

    assert result == {"url": "https://***@example.com/path?plain=visible"}


def test_mcp_connectors_tool_is_registered():
    spec = tool_registry.get("mcp_connectors")

    assert spec is not None
    assert spec.category == "connectors"
    assert {"action", "connector", "connector_action", "params", "exposure", "target"}.issubset(spec.params)


def test_local_mcp_connector_exposes_existing_mcp_tools():
    register_builtin_connectors()
    registered = connector_registry.get("local_mcp")

    assert registered is not None
    action_names = {action.name for action in registered.spec.actions}
    assert "mcp_connectors" not in action_names
    assert "mcp_filesystem" in action_names
    assert "mcp_system" in action_names


def test_local_mcp_uses_tool_action_metadata_for_read_only_describe():
    register_builtin_connectors()

    described = connector_registry.describe("local_mcp", exposure="read-only")
    action_names = {action["name"] for action in described["actions"]}

    assert "mcp_filesystem" in action_names
    assert "mcp_system" in action_names
    assert "mcp_env" in action_names
    assert "mcp_shell" in action_names


@pytest.mark.asyncio
async def test_local_mcp_connector_executes_existing_tool():
    register_builtin_connectors()

    result = await connector_registry.execute("local_mcp", "mcp_system", {"action": "version"})

    assert result["ok"] is True
    assert result["metadata"]["connector"] == "local_mcp"
    assert result["metadata"]["action"] == "mcp_system"


@pytest.mark.asyncio
async def test_local_mcp_read_only_exposure_uses_nested_tool_action_metadata():
    register_builtin_connectors()

    allowed = await tool_registry.execute(
        "mcp_connectors",
        action="execute",
        connector="local_mcp",
        connector_action="mcp_system",
        params={"action": "doctor"},
        exposure="read-only",
    )
    blocked = await tool_registry.execute(
        "mcp_connectors",
        action="execute",
        connector="local_mcp",
        connector_action="mcp_env",
        params={"action": "set", "key": "MCP_TEST", "value": "1"},
        exposure="read-only",
        _confirmed=True,
    )

    assert allowed["ok"] is True
    assert allowed["metadata"]["action"] == "mcp_system"
    assert blocked["ok"] is False
    assert blocked["error"] == "Connector action is not exposed in read-only mode"
    assert blocked["metadata"]["tool_action"] == "set"


@pytest.mark.asyncio
async def test_local_mcp_confirmation_uses_nested_tool_action_metadata():
    register_builtin_connectors()

    allowed = await tool_registry.execute(
        "mcp_connectors",
        action="execute",
        connector="local_mcp",
        connector_action="mcp_env",
        params={"action": "get", "key": "PATH"},
        _confirmed=False,
    )
    blocked = await tool_registry.execute(
        "mcp_connectors",
        action="execute",
        connector="local_mcp",
        connector_action="mcp_env",
        params={"action": "set", "key": "MCP_TEST", "value": "1"},
        _confirmed=False,
    )

    assert allowed["ok"] is True
    assert blocked["ok"] is False
    assert blocked["error"] == "requires confirmation"
    assert blocked["metadata"]["tool_action"] == "set"


@pytest.mark.asyncio
async def test_local_mcp_high_risk_action_requires_bridge_confirmation():
    register_builtin_connectors()

    blocked = await tool_registry.execute(
        "mcp_connectors",
        action="execute",
        connector="local_mcp",
        connector_action="mcp_telegram",
        params={"action": "send", "chat": "me", "text": "hello"},
        _confirmed=False,
    )

    assert blocked["ok"] is False
    assert blocked["error"] == "requires confirmation"
    assert blocked["metadata"]["connector"] == "local_mcp"
    assert blocked["metadata"]["action"] == "mcp_telegram"


@pytest.mark.asyncio
async def test_mcp_connectors_bridge_executes_registered_connector():
    name = "sample_bridge"
    connector_registry.unregister(name)
    connector_registry.register(_sample_spec(name), _sample_handler)
    try:
        from src.core.actions.mcp_connectors import mcp_connectors

        result = await mcp_connectors(
            action="execute",
            connector=name,
            connector_action="search",
            params='{"query": "phone firmware"}',
            session=object(),
        )
    finally:
        connector_registry.unregister(name)

    assert result["ok"] is True
    assert result["data"]["params"] == {"query": "phone firmware"}
    assert result["metadata"]["connector"] == name


def test_guardrails_preserve_registered_tool_params_and_low_read_risk():
    result = evaluate("mcp_connectors", {"action": "list", "connector": "ignored"})

    assert result.risk == ActionRisk.LOW
    assert result.needs_confirm is False
    assert result.sanitized_params["action"] == "list"
    assert result.sanitized_params["connector"] == "ignored"


@pytest.mark.asyncio
async def test_read_only_exposure_filters_and_blocks_write_actions():
    name = "sample_exposure"
    connector_registry.unregister(name)
    connector_registry.register(
        ConnectorSpec(
            name=name,
            description="Exposure sample connector",
            actions=(
                ConnectorActionSpec(
                    name="search",
                    description="Search sample data",
                    risk="low",
                    annotations=ConnectorActionAnnotations(read_only=True),
                ),
                ConnectorActionSpec(
                    name="publish",
                    description="Publish external content",
                    risk="high",
                    annotations=ConnectorActionAnnotations(read_only=False, destructive=True),
                ),
            ),
        ),
        _sample_handler,
    )
    try:
        from src.core.actions.mcp_connectors import mcp_connectors

        described = await mcp_connectors(
            action="describe",
            connector=name,
            exposure="read-only",
        )
        blocked = await mcp_connectors(
            action="execute",
            connector=name,
            connector_action="publish",
            params={"text": "hello"},
            exposure="read-only",
            _confirmed=True,
        )
    finally:
        connector_registry.unregister(name)

    action_names = [action["name"] for action in described["data"]["actions"]]
    assert action_names == ["search"]
    assert blocked["ok"] is False
    assert blocked["error"] == "Connector action is not exposed in read-only mode"


@pytest.mark.asyncio
async def test_connector_registry_blocks_high_risk_without_confirmation():
    calls = 0

    async def handler(action: str, params: dict, runtime: ConnectorRuntime) -> ConnectorResult:
        nonlocal calls
        calls += 1
        return ConnectorResult(ok=True, data={"action": action})

    registry = ConnectorRegistry()
    registry.register(
        ConnectorSpec(
            name="sample_registry_confirmation",
            description="Confirmation sample connector",
            actions=(
                ConnectorActionSpec(
                    name="publish",
                    description="Publish external content",
                    risk="high",
                ),
            ),
        ),
        handler,
    )

    blocked = await registry.execute(
        "sample_registry_confirmation",
        "publish",
        {"text": "hello"},
    )
    allowed = await registry.execute(
        "sample_registry_confirmation",
        "publish",
        {"text": "hello"},
        confirmed=True,
    )

    assert blocked["ok"] is False
    assert blocked["error"] == "requires confirmation"
    assert blocked["metadata"]["risk"] == "high"
    assert allowed["ok"] is True
    assert calls == 1


def test_guardrails_read_only_exposure_does_not_create_pending_for_write_action():
    name = "sample_read_only_mismatch"
    connector_registry.unregister(name)
    connector_registry.register(
        ConnectorSpec(
            name=name,
            description="Read-only mismatch sample connector",
            actions=(
                ConnectorActionSpec(
                    name="publish",
                    description="Publish external content",
                    risk="high",
                    annotations=ConnectorActionAnnotations(read_only=False, destructive=True),
                ),
            ),
        ),
        _sample_handler,
    )
    try:
        result = evaluate(
            "mcp_connectors",
            {
                "action": "execute",
                "connector": name,
                "connector_action": "publish",
                "params": {"text": "hello"},
                "exposure": "read-only",
            },
        )
    finally:
        connector_registry.unregister(name)

    assert result.risk == ActionRisk.LOW
    assert result.needs_confirm is False


@pytest.mark.asyncio
async def test_user_content_is_sanitized_in_connector_results():
    async def handler(action: str, params: dict, runtime: ConnectorRuntime) -> ConnectorResult:
        return ConnectorResult(ok=True, data={"text": "hello\u202eworld\n\n\nnext"})

    registry = ConnectorRegistry()
    registry.register(_sample_spec(), handler)

    result = await registry.execute("sample", "search")

    assert result["data"]["text"] == "helloworld\n\nnext"
    assert result["metadata"]["content_safety"]["user_content"] is True


def test_guardrails_use_connector_action_risk_for_execute():
    name = "sample_guardrails"
    connector_registry.unregister(name)
    connector_registry.register(
        ConnectorSpec(
            name=name,
            description="Risky sample connector",
            actions=(
                ConnectorActionSpec(
                    name="publish",
                    description="Publish external content",
                    risk="high",
                ),
            ),
        ),
        _sample_handler,
    )
    try:
        result = evaluate(
            "mcp_connectors",
            {
                "action": "execute",
                "connector": name,
                "connector_action": "publish",
                "params": {"text": "hello"},
            },
        )
    finally:
        connector_registry.unregister(name)

    assert result.risk == ActionRisk.HIGH
    assert result.needs_confirm is True
    assert result.sanitized_params["params"] == {"text": "hello"}


@pytest.mark.asyncio
async def test_tool_registry_direct_execute_enforces_connector_confirmation():
    name = "sample_direct_confirmation"
    connector_registry.unregister(name)
    connector_registry.register(
        ConnectorSpec(
            name=name,
            description="Direct confirmation sample connector",
            actions=(
                ConnectorActionSpec(
                    name="publish",
                    description="Publish external content",
                    risk="high",
                ),
            ),
        ),
        _sample_handler,
    )
    try:
        blocked = await tool_registry.execute(
            "mcp_connectors",
            action="execute",
            connector=name,
            connector_action="publish",
            params={"text": "hello"},
            _confirmed=False,
        )
        allowed = await tool_registry.execute(
            "mcp_connectors",
            action="execute",
            connector=name,
            connector_action="publish",
            params={"text": "hello"},
            _confirmed=True,
        )
    finally:
        connector_registry.unregister(name)

    assert blocked["ok"] is False
    assert blocked["error"] == "requires confirmation"
    assert blocked["metadata"]["risk"] == "high"
    assert allowed["ok"] is True
    assert allowed["metadata"]["action"] == "publish"
