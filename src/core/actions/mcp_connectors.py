"""Bridge tool for MCP-style connectors."""

from __future__ import annotations

import json
from typing import Any

from src.core.actions.tool_registry import ToolActionSpec, ToolSpec, tool, tool_registry
from src.core.connectors.base import ConnectorExposure
from src.core.connectors import ConnectorRuntime, connector_registry, register_builtin_connectors
from src.core.connectors.credentials import redact_secrets

CONFIRMATION_RISKS = {"high", "critical"}
EXPOSURE_MODES = {"all", "read-only"}


def _parse_params(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("params JSON must decode to an object")
        return parsed
    raise ValueError("params must be an object or JSON object string")


def _connector_confirmation_status(connector: str, connector_action: str) -> tuple[bool, dict[str, Any]]:
    registered = connector_registry.get(connector)
    if registered is None:
        return False, {"connector": connector}

    action_spec = registered.spec.get_action(connector_action)
    if action_spec is None:
        return False, {"connector": connector}

    risk = action_spec.risk.strip().lower()
    needs_confirmation = action_spec.requires_confirmation or risk in CONFIRMATION_RISKS
    return needs_confirmation, {
        "connector": connector,
        "action": action_spec.name,
        "risk": action_spec.risk,
    }


def _local_mcp_tool_spec(connector: str, connector_action: str) -> ToolSpec | None:
    if connector.strip().lower() != "local_mcp":
        return None
    return tool_registry.get(connector_action)


def _local_mcp_action_name(params: dict[str, Any]) -> str | None:
    action = params.get("action")
    if action is None:
        return None
    return str(action).strip().lower() or None


def _local_mcp_confirmation_status(
    connector: str,
    connector_action: str,
    params: dict[str, Any],
) -> tuple[bool, dict[str, Any]] | None:
    spec = _local_mcp_tool_spec(connector, connector_action)
    if spec is None:
        return None
    action_name = _local_mcp_action_name(params)
    risk = spec.effective_risk(action_name)
    needs_confirmation = spec.effective_requires_confirmation(action_name) or risk in CONFIRMATION_RISKS
    return needs_confirmation, {
        "connector": connector,
        "action": spec.name,
        "tool_action": action_name,
        "risk": risk,
    }


def _parse_exposure(value: str | None) -> ConnectorExposure:
    exposure = (value or "all").strip().lower()
    if exposure not in EXPOSURE_MODES:
        raise ValueError("exposure must be one of: all, read-only")
    return exposure  # type: ignore[return-value]


@tool(
    name="mcp_connectors",
    description=(
        "List, inspect, and execute project connectors for external services. "
        "Use action=list to see available connectors, action=describe for one connector, "
        "and action=execute with connector, connector_action, params for a connector call."
    ),
    category="connectors",
    risk="medium",
    actions={
        "list": ToolActionSpec(
            name="list",
            risk="low",
            read_only=True,
            idempotent=True,
        ),
        "describe": ToolActionSpec(
            name="describe",
            risk="low",
            read_only=True,
            idempotent=True,
        ),
        "execute": ToolActionSpec(
            name="execute",
            risk="medium",
            read_only=False,
            idempotent=False,
            user_content=True,
        ),
    },
    params={
        "action": "One of: list, describe, execute",
        "connector": "Connector name for describe/execute",
        "connector_action": "Connector action name for execute",
        "params": "Connector action parameters as JSON object or object",
        "exposure": "Tool surface mode: all or read-only",
        "target": "Optional target/account/workspace label for connectors that support routing",
    },
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "describe", "execute"]},
            "connector": {"type": "string"},
            "connector_action": {"type": "string"},
            "params": {"type": ["object", "string", "null"]},
            "exposure": {"type": "string", "enum": ["all", "read-only"]},
            "target": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "data": {},
            "error": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
        },
    },
)
async def mcp_connectors(
    action: str = "list",
    connector: str = "",
    connector_action: str = "",
    params: Any = None,
    exposure: str = "all",
    target: str = "",
    **runtime_kwargs: Any,
) -> dict[str, Any]:
    register_builtin_connectors()
    confirmed = bool(runtime_kwargs.pop("_confirmed", False))
    normalized_action = (action or "list").strip().lower()
    try:
        exposure_mode = _parse_exposure(exposure)
    except ValueError as exc:
        return {"ok": False, "data": None, "error": str(exc), "metadata": {}}

    if normalized_action == "list":
        return {
            "ok": True,
            "data": connector_registry.list(exposure=exposure_mode),
            "error": None,
            "metadata": {"exposure": exposure_mode},
        }

    if normalized_action == "describe":
        if not connector:
            return {"ok": False, "data": None, "error": "connector is required", "metadata": {}}
        try:
            return {
                "ok": True,
                "data": connector_registry.describe(connector, exposure=exposure_mode),
                "error": None,
                "metadata": {"connector": connector, "exposure": exposure_mode},
            }
        except KeyError as exc:
            return {"ok": False, "data": None, "error": str(exc), "metadata": {"connector": connector}}

    if normalized_action != "execute":
        return {"ok": False, "data": None, "error": f"Unknown action: {action}", "metadata": {}}
    if not connector:
        return {"ok": False, "data": None, "error": "connector is required", "metadata": {}}
    if not connector_action:
        return {"ok": False, "data": None, "error": "connector_action is required", "metadata": {"connector": connector}}

    try:
        parsed_params = _parse_params(params)
    except json.JSONDecodeError as exc:
        return {"ok": False, "data": None, "error": f"Invalid params JSON: {exc}", "metadata": {"connector": connector}}
    except ValueError as exc:
        return {"ok": False, "data": None, "error": str(exc), "metadata": {"connector": connector}}

    registered = connector_registry.get(connector)
    action_spec = registered.spec.get_action(connector_action) if registered is not None else None
    local_spec = _local_mcp_tool_spec(connector, connector_action)
    local_tool_action = _local_mcp_action_name(parsed_params)
    if exposure_mode == "read-only" and action_spec is not None and not action_spec.read_only:
        return {
            "ok": False,
            "data": None,
            "error": "Connector action is not exposed in read-only mode",
            "metadata": {"connector": connector, "action": action_spec.name, "exposure": exposure_mode},
        }
    if (
        exposure_mode == "read-only"
        and local_spec is not None
        and not local_spec.effective_read_only(local_tool_action)
    ):
        return {
            "ok": False,
            "data": None,
            "error": "Connector action is not exposed in read-only mode",
            "metadata": {
                "connector": connector,
                "action": local_spec.name,
                "tool_action": local_tool_action,
                "exposure": exposure_mode,
            },
        }

    local_confirmation = _local_mcp_confirmation_status(connector, connector_action, parsed_params)
    if local_confirmation is not None:
        needs_confirmation, confirmation_metadata = local_confirmation
    else:
        needs_confirmation, confirmation_metadata = _connector_confirmation_status(connector, connector_action)
    if needs_confirmation and not confirmed:
        return {
            "ok": False,
            "data": None,
            "error": "requires confirmation",
            "metadata": confirmation_metadata,
        }

    if target:
        parsed_params.setdefault("target", target)

    if connector.strip().lower() == "local_mcp":
        parsed_params["_confirmed"] = confirmed
    runtime = ConnectorRuntime(
        session=runtime_kwargs.get("session"),
        user=runtime_kwargs.get("user"),
        client=runtime_kwargs.get("client"),
        provider=runtime_kwargs.get("provider"),
        userbot_manager=runtime_kwargs.get("userbot_manager"),
    )
    result = await connector_registry.execute(
        connector,
        connector_action,
        parsed_params,
        runtime,
        exposure=exposure_mode,
        confirmed=confirmed or local_confirmation is not None,
    )
    return redact_secrets(result)
