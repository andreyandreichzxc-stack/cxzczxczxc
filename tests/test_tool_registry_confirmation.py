import pytest

from src.core.actions import register_builtin_tools
from src.core.actions.mcp_processes import mcp_processes
from src.core.actions.tool_registry import ToolActionSpec, ToolRegistry, ToolSpec, tool_registry
from src.core.actions.recall_memory_tool import recall_memory


@pytest.mark.asyncio
async def test_execute_blocks_unconfirmed_tool() -> None:
    calls = 0

    async def handler() -> dict:
        nonlocal calls
        calls += 1
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="dangerous",
            description="Dangerous test tool",
            category="test",
            handler=handler,
            risk="high",
            requires_confirmation=True,
        )
    )

    result = await registry.execute("dangerous")

    assert result == {"error": "requires confirmation"}
    assert calls == 0


@pytest.mark.asyncio
async def test_execute_runs_confirmed_tool() -> None:
    calls = 0

    async def handler() -> dict:
        nonlocal calls
        calls += 1
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="dangerous",
            description="Dangerous test tool",
            category="test",
            handler=handler,
            risk="high",
            requires_confirmation=True,
        )
    )

    result = await registry.execute("dangerous", _confirmed=True)

    assert result == {"ok": True}
    assert calls == 1


@pytest.mark.asyncio
async def test_recall_memory_error_matches_schema() -> None:
    result = await recall_memory("любой запрос")

    assert result["ok"] is False
    assert result["facts"] == []
    assert result["found"] == 0
    assert "error" in result


@pytest.mark.asyncio
async def test_execute_blocks_high_risk_tool_without_confirmation_flag() -> None:
    calls = 0

    async def handler() -> dict:
        nonlocal calls
        calls += 1
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="dangerous_by_risk",
            description="Dangerous test tool",
            category="test",
            handler=handler,
            risk="high",
            requires_confirmation=False,
        )
    )

    result = await registry.execute("dangerous_by_risk")

    assert result == {"error": "requires confirmation"}
    assert calls == 0


def test_register_builtin_tools_is_idempotent() -> None:
    register_builtin_tools()
    before = sorted(spec.name for specs in tool_registry.list_by_category().values() for spec in specs)

    register_builtin_tools()
    after = sorted(spec.name for specs in tool_registry.list_by_category().values() for spec in specs)

    assert after == before


@pytest.mark.asyncio
async def test_execute_uses_action_level_confirmation_metadata() -> None:
    calls: list[str] = []

    async def handler(action: str) -> dict:
        calls.append(action)
        return {"ok": True, "action": action}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="multi",
            description="Multi action test tool",
            category="test",
            handler=handler,
            risk="medium",
            requires_confirmation=False,
            actions={
                "read": ToolActionSpec(name="read", risk="low", read_only=True),
                "write": ToolActionSpec(
                    name="write",
                    risk="high",
                    read_only=False,
                    requires_confirmation=True,
                ),
            },
        )
    )

    blocked = await registry.execute("multi", action="write")
    read = await registry.execute("multi", action="read")
    write = await registry.execute("multi", action="write", _confirmed=True)

    assert blocked == {"error": "requires confirmation"}
    assert read == {"ok": True, "action": "read"}
    assert write == {"ok": True, "action": "write"}
    assert calls == ["read", "write"]


@pytest.mark.asyncio
async def test_process_kill_requires_direct_call_confirmation() -> None:
    result = await mcp_processes("kill", pid=1)

    assert result["requires_confirmation"] is True
    assert result["action"] == "kill"
