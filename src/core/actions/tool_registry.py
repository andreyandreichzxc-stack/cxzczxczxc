"""Tool Registry — decorator-based action registration for LLM tool use.

Provides the ``@tool`` decorator and ``ToolRegistry`` singleton for
standardized registration of actions (tools) with metadata such as
description, category, risk level, and parameter schema.

Tools are registered *at import time* via the decorator, making the
registry effectively read-only after module initialisation.

Usage::

    from src.core.actions.tool_registry import tool, tool_registry

    @tool(
        name="search_messages",
        description="Search messages by text",
        category="search",
        risk="low",
        params={"query": "str", "contact": "str|None"},
    )
    async def search_messages(query: str, contact: str | None = None) -> dict:
        ...

    result = await tool_registry.execute("search_messages", query="hello")
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

CONFIRMATION_RISKS = {"high", "critical"}


def _handler_accepts_kwarg(handler: Callable[..., Awaitable[dict]], name: str) -> bool:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name
        for param in signature.parameters.values()
    )

# ── ToolSpec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolActionSpec:
    """Optional per-action metadata for multi-action tools."""

    name: str
    description: str = ""
    risk: str = "low"
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    requires_confirmation: bool = False
    open_world: bool = False
    user_content: bool = True


@dataclass(frozen=True)
class ToolActionMetadata:
    """Optional per-action metadata for multi-action tools."""

    risk: str | None = None
    requires_confirmation: bool | None = None
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None
    user_content: bool | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Immutable specification for a registered tool.

    Attributes:
        name: Unique tool identifier (used in ``execute()`` and prompts).
        description: Human-readable description of what the tool does.
        category: Grouping category (e.g. ``"search"``, ``"chat"``, ``"reminder"``).
        risk: Risk level — ``"low"``, ``"medium"``, ``"high"``, or ``"critical"``.
        requires_confirmation: Whether execution should prompt the user first.
        params: Dict mapping parameter name → type hint string.
                Example: ``{"query": "str", "limit": "int|None"}``.
        handler: The async callable that implements the tool.
    """

    name: str
    description: str
    category: str
    handler: Callable[..., Awaitable[dict]] = field(hash=False, compare=False)
    risk: str = "low"
    requires_confirmation: bool = False
    params: dict[str, str] = field(default_factory=dict)
    input_schema: dict[str, Any] | None = None  # JSON Schema for params
    output_schema: dict[str, Any] | None = None  # JSON Schema for return value
    action_metadata: dict[str, ToolActionMetadata] = field(default_factory=dict)

    def get_action_metadata(self, action: Any) -> ToolActionMetadata | None:
        action_spec = self.action_spec(action) if self.actions else None
        if action_spec is not None:
            return ToolActionMetadata(
                risk=action_spec.risk,
                requires_confirmation=action_spec.requires_confirmation,
                read_only=action_spec.read_only,
                destructive=action_spec.destructive,
                idempotent=action_spec.idempotent,
                open_world=action_spec.open_world,
                user_content=action_spec.user_content,
            )
        if action is None:
            return None
        return self.action_metadata.get(str(action).strip().lower())

    def effective_risk(self, action: Any = None) -> str:
        metadata = self.get_action_metadata(action)
        return (metadata.risk if metadata and metadata.risk is not None else self.risk).strip().lower()

    def effective_requires_confirmation(self, action: Any = None) -> bool:
        metadata = self.get_action_metadata(action)
        if metadata and metadata.requires_confirmation is not None:
            return metadata.requires_confirmation
        return self.requires_confirmation

    def effective_read_only(self, action: Any = None) -> bool:
        metadata = self.get_action_metadata(action)
        if metadata and metadata.read_only is not None:
            return metadata.read_only
        return self.effective_risk(action) == "low" and not self.effective_requires_confirmation(action)

    def effective_destructive(self, action: Any = None) -> bool:
        metadata = self.get_action_metadata(action)
        if metadata and metadata.destructive is not None:
            return metadata.destructive
        return self.effective_risk(action) in CONFIRMATION_RISKS

    def effective_idempotent(self, action: Any = None) -> bool:
        metadata = self.get_action_metadata(action)
        if metadata and metadata.idempotent is not None:
            return metadata.idempotent
        return self.effective_read_only(action)

    def effective_open_world(self, action: Any = None) -> bool:
        metadata = self.get_action_metadata(action)
        if metadata and metadata.open_world is not None:
            return metadata.open_world
        return False

    def effective_user_content(self, action: Any = None) -> bool:
        metadata = self.get_action_metadata(action)
        if metadata and metadata.user_content is not None:
            return metadata.user_content
        return True
    actions: dict[str, ToolActionSpec] = field(default_factory=dict)

    def action_spec(self, action: Any) -> ToolActionSpec | None:
        if not action:
            return None
        return self.actions.get(str(action).strip().lower())


# ── ToolRegistry ─────────────────────────────────────────────────────────


class ToolRegistry:
    """Registry of tools populated at import time via ``@tool``.

    The registry is effectively **read-only** after initialisation — tools
    are registered once when their defining module is imported.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Register a ``ToolSpec`` (called by the ``@tool`` decorator).

        If a tool with the same name already exists it is overwritten and
        a warning is logged.  This can happen during reloads.
        """
        if spec.name in self._tools:
            logger.warning("Tool %r already registered, overwriting", spec.name)
        self._tools[spec.name] = spec

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        """Look up a tool by its unique name.

        Returns ``None`` when no tool with *name* is registered.
        """
        return self._tools.get(name)

    def list_by_category(self) -> dict[str, list[ToolSpec]]:
        """Return all tools grouped by their ``category`` field."""
        categories: dict[str, list[ToolSpec]] = {}
        for spec in self._tools.values():
            categories.setdefault(spec.category, []).append(spec)
        return categories

    def list_for_prompt(self) -> str:
        """Format all tools as a prompt-friendly string for LLM system prompts.

        Example output::

            ## chat
            - `draft_reply` (medium ⚠️ confirmation): Draft a reply …
              params: contact: str, message: str, style: str|None
            - `summarize_chat` (medium): Summarize chat with a contact …

            ## search
            - `search_messages` (low): Search messages by text …
              params: query: str, contact: str|None
        """
        lines: list[str] = []
        for category, tools in sorted(self.list_by_category().items()):
            lines.append(f"## {category}")
            for spec in sorted(tools, key=lambda s: s.name):
                confirm = " ⚠️ confirmation" if spec.requires_confirmation else ""
                lines.append(
                    f"- `{spec.name}` ({spec.risk}{confirm}): {spec.description}"
                )
                if spec.params:
                    params_str = ", ".join(f"{k}: {v}" for k, v in spec.params.items())
                    lines.append(f"  params: {params_str}")
                if spec.input_schema:
                    lines.append(f"  input_schema: {spec.input_schema}")
                if spec.output_schema:
                    lines.append(f"  output_schema: {spec.output_schema}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def format_tools_with_schemas(self) -> str:
        """Generate a compact text description of each tool with its JSON schemas.

        The output is designed for LLM prompt injection — it describes what
        each tool returns (output schema) and expects as input (input schema).

        Example::

            recall_memory (memory):
              input:  {"query": "str", "limit": "int (default 8)", "mode": "normal|light|deep"}
              output: {"ok": bool, "facts": [{fact, confidence, reason}], "found": int}
        """
        lines: list[str] = []
        for category, tools in sorted(self.list_by_category().items()):
            lines.append(f"## {category}")
            for spec in sorted(tools, key=lambda s: s.name):
                confirm = " ⚠️ confirmation" if spec.requires_confirmation else ""
                lines.append(
                    f"### {spec.name} ({spec.risk}{confirm})\n{spec.description}"
                )
                if spec.params:
                    params_str = ", ".join(f"{k}: {v}" for k, v in spec.params.items())
                    lines.append(f"  params: {params_str}")
                if spec.input_schema:
                    # Compact input schema description
                    props = spec.input_schema.get("properties", {})
                    required = set(spec.input_schema.get("required", []))
                    param_descs: list[str] = []
                    for pname, pinfo in props.items():
                        typ = pinfo.get("type", "any")
                        desc = pinfo.get("description", "")
                        default = pinfo.get("default", None)
                        extras = []
                        if pname in required:
                            extras.append("required")
                        if default is not None:
                            extras.append(f"default={default}")
                        if pinfo.get("enum"):
                            extras.append(f"enum={pinfo['enum']}")
                        suffix = f" ({', '.join(extras)})" if extras else ""
                        param_descs.append(f"    {pname}: {typ}{suffix} — {desc}")
                    if param_descs:
                        lines.append("  input_schema:")
                        lines.extend(param_descs)
                if spec.output_schema:
                    # Compact output schema description
                    props = spec.output_schema.get("properties", {})
                    required = set(spec.output_schema.get("required", []))
                    out_descs: list[str] = []
                    for pname, pinfo in props.items():
                        typ = pinfo.get("type", "any")
                        desc = pinfo.get("description", "")
                        extras = []
                        if pname in required:
                            extras.append("required")
                        if pinfo.get("items"):
                            items = pinfo["items"]
                            if isinstance(items, dict):
                                item_props = items.get("properties", {})
                                if item_props:
                                    sub = ", ".join(item_props.keys())
                                    extras.append(f"items: {{{sub}}}")
                        suffix = f" ({', '.join(extras)})" if extras else ""
                        out_descs.append(f"    {pname}: {typ}{suffix} — {desc}")
                    if out_descs:
                        lines.append("  output_schema:")
                        lines.extend(out_descs)
                lines.append("")
        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, name: str, **params: Any) -> dict[str, Any]:
        """Execute a tool by name, passing *params* to its handler.

        The handler receives the params as keyword arguments.  Any extra
        keyword arguments (e.g. runtime dependencies such as ``provider``)
        can be passed through -- they will be forwarded to the handler if
        its signature accepts ``**kwargs``.

        **Security enforcement:** if the tool's ``ToolSpec.requires_confirmation``
        is ``True``, the caller **must** pass ``_confirmed=True``.  Callers that
        have not yet obtained user consent should pass ``_confirmed=False``
        (or omit it) and this method will return ``{"error": "requires
        confirmation"}`` without executing.

        Returns:
            The dict returned by the handler, or ``{"error": <message>}``
            if the tool is not found, requires confirmation, or the handler
            raises.
        """
        spec = self.get(name)
        if spec is None:
            return {"error": f"Tool '{name}' not found"}

        # Enforce confirmation for declared high-risk tools even if a spec
        # forgot to set requires_confirmation.
        confirmed = params.pop("_confirmed", False)
        action_name = params.get("action")
        risk = spec.effective_risk(action_name)
        requires_confirmation = spec.effective_requires_confirmation(action_name)
        if (requires_confirmation or risk in CONFIRMATION_RISKS) and not confirmed:
            return {"error": "requires confirmation"}
        if _handler_accepts_kwarg(spec.handler, "_confirmed"):
            params["_confirmed"] = confirmed

        try:
            result = await spec.handler(**params)
            # Normalise None return to a success dict
            if result is None:
                return {"ok": True}
            return result
        except Exception:
            logger.exception("Tool %r failed with params %r", name, params)
            return {"error": f"Tool '{name}' execution failed"}


# Module-level singleton — imported by other modules
tool_registry = ToolRegistry()


# ── @tool decorator ──────────────────────────────────────────────────────


def tool(
    *,
    name: str,
    description: str,
    category: str,
    risk: str = "low",
    requires_confirmation: bool = False,
    params: dict[str, str] | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    actions: dict[str, ToolActionSpec | dict[str, Any]] | None = None,
    action_metadata: dict[str, ToolActionMetadata | dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Awaitable[dict]]], Callable[..., Awaitable[dict]]]:
    """Decorator that registers an async function as a tool.

    The decorated function is automatically registered in the global
    ``tool_registry`` when the module is imported.

    Args:
        name: Unique tool name (used in ``execute()`` and LLM prompts).
        description: Human-readable description of what the tool does.
        category: Grouping category (e.g. ``"search"``, ``"memory"``, ``"reminder"``).
        risk: Risk level — ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        requires_confirmation: If ``True`` the LLM should ask the user before
            executing this tool (e.g. for destructive actions).
        params: Dict mapping parameter name → type hint string.
                Example: ``{"query": "str", "limit": "int|None"}``.
        input_schema: Optional JSON Schema describing input parameters.
        output_schema: Optional JSON Schema describing the return value.

    Example::

        @tool(
            name="search_messages",
            description="Search messages by text",
            category="search",
            params={"query": "str"},
        )
        async def search_messages(query: str) -> dict:
            return {"ok": True, "query": query}
    """
    tool_params = dict(params or {})
    tool_actions = _normalize_tool_actions(actions)
    normalized_action_metadata = _normalize_action_metadata(action_metadata)

    def decorator(
        func: Callable[..., Awaitable[dict]],
    ) -> Callable[..., Awaitable[dict]]:
        spec = ToolSpec(
            name=name,
            description=description,
            category=category,
            risk=risk,
            requires_confirmation=requires_confirmation,
            params=tool_params,
            handler=func,
            input_schema=input_schema,
            output_schema=output_schema,
            action_metadata=normalized_action_metadata,
            actions=tool_actions,
        )
        tool_registry.register(spec)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def _normalize_tool_actions(
    values: dict[str, ToolActionSpec | dict[str, Any]] | None,
) -> dict[str, ToolActionSpec]:
    if not values:
        return {}
    normalized: dict[str, ToolActionSpec] = {}
    for action, spec in values.items():
        key = str(action).strip().lower()
        if not key:
            continue
        if isinstance(spec, ToolActionSpec):
            normalized[key] = spec
            continue
        payload = dict(spec)
        payload.setdefault("name", key)
        normalized[key] = ToolActionSpec(**payload)
    return normalized


def _normalize_action_metadata(
    values: dict[str, ToolActionMetadata | dict[str, Any]] | None,
) -> dict[str, ToolActionMetadata]:
    if not values:
        return {}
    normalized: dict[str, ToolActionMetadata] = {}
    for action, metadata in values.items():
        key = str(action).strip().lower()
        if not key:
            continue
        if isinstance(metadata, ToolActionMetadata):
            normalized[key] = metadata
        else:
            normalized[key] = ToolActionMetadata(**metadata)
    return normalized


# ══════════════════════════════════════════════════════════════════════════
