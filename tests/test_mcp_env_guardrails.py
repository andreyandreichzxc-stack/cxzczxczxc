import pytest

from src.core.actions import mcp_env as mcp_env_module
from src.core.intelligence.guardrails import ActionRisk, get_action_risk


def test_env_get_masks_even_plain_variable(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BUT_STILL_PRIVATE", "plain-value-123456")

    result = mcp_env_module._get_env("PUBLIC_BUT_STILL_PRIVATE")

    assert result["ok"] is True
    assert result["found"] is True
    assert result["masked"] is True
    assert result["value"] != "plain-value-123456"


def test_guardrails_prefer_registered_tool_confirmation_metadata() -> None:
    assert get_action_risk("mcp_env") is ActionRisk.HIGH


@pytest.mark.asyncio
async def test_env_set_requires_direct_call_confirmation(monkeypatch) -> None:
    monkeypatch.delenv("MCP_ENV_DIRECT_SET_TEST", raising=False)

    blocked = await mcp_env_module.mcp_env(
        "set",
        key="MCP_ENV_DIRECT_SET_TEST",
        value="secret",
    )
    allowed = await mcp_env_module.mcp_env(
        "set",
        key="MCP_ENV_DIRECT_SET_TEST",
        value="secret",
        _confirmed=True,
    )

    assert blocked == {"error": "requires confirmation"}
    assert allowed["ok"] is True
