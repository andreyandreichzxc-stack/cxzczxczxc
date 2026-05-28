"""Connector SDK for external services and content sources."""

from .base import (
    ConnectorActionAnnotations,
    ConnectorActionSpec,
    ConnectorResult,
    ConnectorRuntime,
    ConnectorSpec,
)
from .registry import ConnectorRegistry, connector_registry

_BUILTINS_REGISTERED = False


def register_builtin_connectors() -> None:
    """Register built-in connectors.

    Site-specific connectors are intentionally added separately. This keeps the
    base layer small until a concrete source is requested.
    """

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return None
    from src.core.actions import register_builtin_tools

    register_builtin_tools()
    from .site_connectors import register_site_connectors
    from .tool_registry_adapter import register_tool_registry_connector

    register_tool_registry_connector(connector_registry)
    register_site_connectors(connector_registry)
    _BUILTINS_REGISTERED = True
    return None


__all__ = [
    "ConnectorActionSpec",
    "ConnectorActionAnnotations",
    "ConnectorRegistry",
    "ConnectorResult",
    "ConnectorRuntime",
    "ConnectorSpec",
    "connector_registry",
    "register_builtin_connectors",
]
