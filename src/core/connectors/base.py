"""Base types for MCP-style external connectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


ConnectorRisk = Literal["low", "medium", "high", "critical"]
ConnectorAuthMode = Literal["none", "api_key", "oauth", "cookie", "session", "custom"]
ConnectorExposure = Literal["all", "read-only"]


@dataclass(frozen=True)
class ConnectorActionAnnotations:
    """MCP-style hints that let a connector expose an adaptive tool surface."""

    title: str | None = None
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    open_world: bool = True
    user_content: bool = True


@dataclass(frozen=True)
class ConnectorActionSpec:
    """Public description of one connector action."""

    name: str
    description: str
    risk: ConnectorRisk = "low"
    requires_confirmation: bool = False
    annotations: ConnectorActionAnnotations = field(default_factory=ConnectorActionAnnotations)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    @property
    def read_only(self) -> bool:
        return self.annotations.read_only


@dataclass(frozen=True)
class ConnectorSpec:
    """Public description of an external connector."""

    name: str
    description: str
    actions: tuple[ConnectorActionSpec, ...]
    category: str = "general"
    auth_mode: ConnectorAuthMode = "none"
    docs_url: str | None = None
    capabilities: tuple[str, ...] = ()
    supports_targets: bool = False

    def get_action(self, action_name: str) -> ConnectorActionSpec | None:
        normalized = action_name.strip().lower()
        for action in self.actions:
            if action.name.strip().lower() == normalized:
                return action
        return None


@dataclass
class ConnectorRuntime:
    """Runtime objects passed from the app tool layer to a connector."""

    session: Any | None = None
    user: Any | None = None
    client: Any | None = None
    provider: Any | None = None
    userbot_manager: Any | None = None
    credentials: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorResult:
    """Normalized connector response."""

    ok: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


ConnectorHandler = Callable[[str, dict[str, Any], ConnectorRuntime], Awaitable[ConnectorResult | dict[str, Any]]]
