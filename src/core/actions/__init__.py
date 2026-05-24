# Actions: action validation, conflict checks, indexing, trajectories, tool registry

from src.core.actions.tool_registry import tool, tool_registry, ToolRegistry, ToolSpec

# Side-effect imports: register tools via @tool decorator at module load
import src.core.actions.mcp_tools  # noqa: F401
import src.core.actions.mcp_web  # noqa: F401
import src.core.actions.cross_search_tool  # noqa: F401
import src.core.actions.recall_memory_tool  # noqa: F401
import src.core.actions.search_contexts_tool  # noqa: F401
import src.core.actions.sdd_executor  # noqa: F401
import src.core.actions.mcp_telegram  # noqa: F401
import src.core.actions.mcp_http  # noqa: F401
import src.core.actions.mcp_reminders  # noqa: F401

__all__ = [
    "tool",
    "tool_registry",
    "ToolRegistry",
    "ToolSpec",
]
