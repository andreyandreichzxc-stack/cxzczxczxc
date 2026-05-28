"""Connector adapter over existing project MCP tools."""

from __future__ import annotations

from typing import Any

from src.core.actions.tool_registry import CONFIRMATION_RISKS, ToolSpec, tool_registry

from .base import (
    ConnectorActionAnnotations,
    ConnectorActionSpec,
    ConnectorResult,
    ConnectorRuntime,
    ConnectorSpec,
)
from .registry import ConnectorRegistry


EXCLUDED_TOOLS = {"mcp_connectors"}
CONNECTOR_NAME = "local_mcp"


def register_tool_registry_connector(registry: ConnectorRegistry) -> None:
    registry.unregister(CONNECTOR_NAME)
    registry.register(_build_spec(), _handler)


def _tool_specs() -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for category_tools in tool_registry.list_by_category().values():
        for spec in category_tools:
            if spec.name in EXCLUDED_TOOLS:
                continue
            if not spec.name.startswith("mcp_"):
                continue
            tools.append(spec)
    return sorted(tools, key=lambda item: item.name)


def _build_spec() -> ConnectorSpec:
    return ConnectorSpec(
        name=CONNECTOR_NAME,
        description="Adaptive connector that exposes the project's existing mcp_* tools through mcp_connectors.",
        category="local",
        auth_mode="custom",
        capabilities=("local_tools", "mcp_tools", "adaptive_actions"),
        actions=tuple(_action_from_tool(spec) for spec in _tool_specs()),
    )


def _action_from_tool(spec: ToolSpec) -> ConnectorActionSpec:
    action_names = list(spec.actions) or [None]
    risks = [_valid_risk(spec.effective_risk(action)) for action in action_names]
    risk = _max_risk(risks)
    requires_confirmation = any(
        spec.effective_requires_confirmation(action)
        or spec.effective_risk(action) in CONFIRMATION_RISKS
        for action in action_names
    )
    read_only = any(spec.effective_read_only(action) for action in action_names)
    destructive = any(spec.effective_destructive(action) for action in action_names)
    idempotent = all(spec.effective_idempotent(action) for action in action_names)
    open_world = any(spec.effective_open_world(action) for action in action_names)
    user_content = any(spec.effective_user_content(action) for action in action_names)
    return ConnectorActionSpec(
        name=spec.name,
        description=spec.description,
        risk=risk,  # type: ignore[arg-type]
        requires_confirmation=requires_confirmation,
        annotations=ConnectorActionAnnotations(
            title=spec.name,
            read_only=read_only,
            destructive=destructive,
            idempotent=idempotent,
            open_world=open_world,
            user_content=user_content,
        ),
        input_schema=spec.input_schema or _schema_from_params(spec.params),
        output_schema=spec.output_schema,
    )


def _valid_risk(value: str) -> str:
    return value if value in {"low", "medium", "high", "critical"} else "medium"


def _max_risk(values: list[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return max(values or ["medium"], key=lambda item: order.get(item, 1))


def _schema_from_params(params: dict[str, str]) -> dict[str, Any] | None:
    if not params:
        return None
    return {
        "type": "object",
        "properties": {
            name: {"type": _json_type(type_hint), "description": type_hint}
            for name, type_hint in params.items()
        },
        "additionalProperties": True,
    }


def _json_type(type_hint: str) -> str:
    hint = type_hint.lower()
    if "int" in hint:
        return "integer"
    if "float" in hint or "number" in hint:
        return "number"
    if "bool" in hint:
        return "boolean"
    if "list" in hint or "[]" in hint:
        return "array"
    if "dict" in hint or "json" in hint:
        return "object"
    return "string"


async def _handler(action: str, params: dict[str, Any], runtime: ConnectorRuntime) -> ConnectorResult:
    tool_params = dict(params)
    tool_params.update(_runtime_kwargs(runtime))
    confirmed = bool(tool_params.pop("_confirmed", False))
    spec = tool_registry.get(action)
    tool_action = tool_params.get("action")

    result = await tool_registry.execute(action, _confirmed=confirmed, **tool_params)
    ok = "error" not in result
    metadata: dict[str, Any] = {}
    if spec is not None:
        metadata["tool_action"] = tool_action
        metadata["annotations"] = {
            "read_only": spec.effective_read_only(tool_action),
            "destructive": spec.effective_destructive(tool_action),
            "idempotent": spec.effective_idempotent(tool_action),
            "open_world": spec.effective_open_world(tool_action),
            "user_content": spec.effective_user_content(tool_action),
        }
    return ConnectorResult(
        ok=ok,
        data=result if ok else None,
        error=result.get("error") if not ok else None,
        metadata=metadata,
    )


def _runtime_kwargs(runtime: ConnectorRuntime) -> dict[str, Any]:
    values = {
        "session": runtime.session,
        "user": runtime.user,
        "client": runtime.client,
        "provider": runtime.provider,
        "userbot_manager": runtime.userbot_manager,
    }
    return {key: value for key, value in values.items() if value is not None}
