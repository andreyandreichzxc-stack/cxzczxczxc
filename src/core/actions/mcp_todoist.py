"""mcp_todoist tool — registered via @tool decorator.

Todoist task management via REST API.

Actions:
- ``action="tasks" filter="today" limit=10`` — fetch tasks matching a filter
- ``action="add" content="..." due_string="tomorrow" priority=2`` — add a new task
- ``action="complete" task_id="..."`` — close a task
- ``action="projects"`` — list all projects

Requires TODOIST_API_TOKEN set in settings or environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_TODOIST_BASE = "https://api.todoist.com/rest/v2"
_REQUEST_TIMEOUT = 10  # seconds
_TASKS_LIMIT_MAX = 50


def _get_api_token() -> str:
    """Retrieve the Todoist API token from settings or environment."""
    token = os.environ.get("TODOIST_API_TOKEN", "")
    if not token:
        raise ValueError("Set TODOIST_API_TOKEN in .env or environment variable")
    return token


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_todoist
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_todoist",
    description=(
        "Manage Todoist tasks via REST API. Supports four actions:\n"
        "- 'tasks' — fetch tasks matching a filter string (e.g. 'today', 'p1').\n"
        "- 'add' — create a new task with optional due date and priority.\n"
        "- 'complete' — close/complete a task by its ID.\n"
        "- 'projects' — list all Todoist projects.\n"
        "Requires TODOIST_API_TOKEN set in .env or environment."
    ),
    category="productivity",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — 'tasks', 'add', 'complete', or 'projects'",
        "filter": "str — Todoist filter string (default 'today', used with 'tasks')",
        "limit": "int — max tasks to return (default 10, max 50, used with 'tasks')",
        "content": "str — task description (required for 'add')",
        "due_string": "str — natural-language due date (e.g. 'tomorrow', 'next monday', used with 'add')",
        "priority": "int — task priority 1-4 (default 2, used with 'add')",
        "task_id": "str — Todoist task ID (required for 'complete')",
    },
)
async def mcp_todoist(
    action: str,
    filter: str = "today",
    limit: int = 10,
    content: str = "",
    due_string: str = "",
    priority: int = 2,
    task_id: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Todoist task management tool.

    Args:
        action: ``"tasks"``, ``"add"``, ``"complete"``, or ``"projects"``.
        filter: Todoist filter string (default ``"today"``, used with ``"tasks"``).
        limit: Max tasks to return (default 10, max 50, used with ``"tasks"``).
        content: Task description (required for ``action="add"``).
        due_string: Natural-language due date (optional, used with ``"add"``).
        priority: Task priority 1-4 (default 2, used with ``"add"``).
        task_id: Todoist task ID (required for ``action="complete"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "tasks":
            limit = max(1, min(limit, _TASKS_LIMIT_MAX))
            return await _todoist_tasks(filter, limit)
        elif action == "add":
            if not content or not content.strip():
                return {"error": "content parameter is required for action='add'"}
            priority = max(1, min(priority, 4))
            return await _todoist_add(content.strip(), due_string.strip(), priority)
        elif action == "complete":
            if not task_id or not task_id.strip():
                return {"error": "task_id parameter is required for action='complete'"}
            return await _todoist_complete(task_id.strip())
        elif action == "projects":
            return await _todoist_projects()
        else:
            return {
                "error": (
                    f"Unknown action {action!r}. "
                    "Valid actions: tasks, add, complete, projects"
                )
            }
    except Exception as exc:
        logger.exception("mcp_todoist(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


def _todoist_headers() -> dict[str, str]:
    token = _get_api_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def _todoist_tasks(filter_str: str, limit: int) -> dict[str, Any]:
    """Fetch tasks matching *filter_str*."""
    import requests

    loop = asyncio.get_running_loop()

    def _fetch() -> list[dict[str, Any]]:
        url = f"{_TODOIST_BASE}/tasks"
        params = {"filter": filter_str, "limit": limit}
        try:
            resp = requests.get(
                url, headers=_todoist_headers(), params=params, timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Todoist fetch tasks failed: %s", exc)
            raise

    try:
        tasks = await loop.run_in_executor(None, _fetch)
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"Todoist API request failed: {exc}"}

    results = []
    for t in tasks:
        results.append(
            {
                "id": t.get("id"),
                "content": t.get("content", ""),
                "due": t.get("due", {}).get("string") if t.get("due") else None,
                "priority": t.get("priority", 1),
                "url": t.get("url", ""),
                "project_id": t.get("project_id"),
            }
        )

    return {
        "ok": True,
        "tasks": results,
        "count": len(results),
    }


async def _todoist_add(content: str, due_string: str, priority: int) -> dict[str, Any]:
    """Create a new Todoist task."""
    import requests

    loop = asyncio.get_running_loop()

    def _create() -> dict[str, Any]:
        url = f"{_TODOIST_BASE}/tasks"
        payload: dict[str, Any] = {
            "content": content,
            "priority": priority,
        }
        if due_string:
            payload["due_string"] = due_string
        try:
            resp = requests.post(
                url, headers=_todoist_headers(), json=payload, timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Todoist create task failed: %s", exc)
            raise

    try:
        task = await loop.run_in_executor(None, _create)
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"Todoist API request failed: {exc}"}

    return {
        "ok": True,
        "id": task.get("id"),
        "content": task.get("content", ""),
        "due": task.get("due", {}).get("string") if task.get("due") else None,
        "priority": task.get("priority", 1),
        "url": task.get("url", ""),
    }


async def _todoist_complete(task_id: str) -> dict[str, Any]:
    """Close a task by its ID."""
    import requests

    loop = asyncio.get_running_loop()

    def _close() -> None:
        url = f"{_TODOIST_BASE}/tasks/{task_id}/close"
        try:
            resp = requests.post(
                url, headers=_todoist_headers(), timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Todoist close task %s failed: %s", task_id, exc)
            raise

    try:
        await loop.run_in_executor(None, _close)
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"Todoist API request failed: {exc}"}

    return {"ok": True, "task_id": task_id, "status": "completed"}


async def _todoist_projects() -> dict[str, Any]:
    """List all Todoist projects."""
    import requests

    loop = asyncio.get_running_loop()

    def _fetch() -> list[dict[str, Any]]:
        url = f"{_TODOIST_BASE}/projects"
        try:
            resp = requests.get(
                url, headers=_todoist_headers(), timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Todoist fetch projects failed: %s", exc)
            raise

    try:
        projects = await loop.run_in_executor(None, _fetch)
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"Todoist API request failed: {exc}"}

    results = [
        {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "color": p.get("color", ""),
            "comment_count": p.get("comment_count", 0),
            "is_shared": p.get("is_shared", False),
            "is_favorite": p.get("is_favorite", False),
            "url": p.get("url", ""),
        }
        for p in projects
    ]

    return {
        "ok": True,
        "projects": results,
        "count": len(results),
    }
