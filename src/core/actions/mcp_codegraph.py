"""mcp_codegraph tool — CodeGraph SQLite read-only queries.

Reads the pre-built CodeGraph SQLite database (``.codegraph/codegraph.db``)
to provide symbol search, call graphs, file structure, and indexing status.

Actions:
- ``action="search" name=...`` — search symbols by name (LIKE match)
- ``action="callers" symbol=...`` — find functions that call a given symbol
- ``action="callees" symbol=...`` — find functions called by a given symbol
- ``action="node" symbol=...`` — full info about a symbol
- ``action="files" pattern=...`` — list indexed files (optional path filter)
- ``action="status"`` — database statistics (symbols, edges, size, last indexed)
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_CODEGRAPH_DIR = Path(__file__).parent.parent.parent.parent / ".codegraph"
_DB_PATH = _CODEGRAPH_DIR / "codegraph.db"
_QUERY_LIMIT = 100  # default max results per query

# ══════════════════════════════════════════════════════════════════════════
# Tool: codegraph
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="codegraph",
    description=(
        "Read-only access to the CodeGraph SQLite database. Supports actions:\n"
        "- 'search' — search symbols by name (LIKE match, use query=...)\n"
        "- 'callers' — find functions that directly call a given symbol\n"
        "- 'callees' — find functions directly called by a given symbol\n"
        "- 'node' — full info about a specific symbol (name, kind, file, line, "
        "signature, docstring)\n"
        "- 'files' — list indexed project files with symbol counts "
        "(optional pattern filter like '*.py' or 'src/**')\n"
        "- 'status' — database statistics (symbol count, edge count, db size, "
        "last indexed time)"
    ),
    category="system",
    risk="low",
    params={
        "action": (
            "str — 'search', 'callers', 'callees', 'node', 'files', or 'status'"
        ),
        "query": "str — symbol name to search for (used with action='search')",
        "symbol": (
            "str — exact symbol name (used with action='callers', 'callees', 'node')"
        ),
        "pattern": (
            "str — file path glob pattern (used with action='files', "
            "e.g. '*.py', 'src/**')"
        ),
        "limit": "int — max results (default 100, max 500)",
    },
)
async def codegraph(
    action: str,
    query: str = "",
    symbol: str = "",
    pattern: str = "",
    limit: int = _QUERY_LIMIT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read-only CodeGraph database tool.

    Args:
        action: One of ``"search"``, ``"callers"``, ``"callees"``, ``"node"``,
            ``"files"``, ``"status"``.
        query: Symbol name to search (LIKE match, used with ``action="search"``).
        symbol: Exact symbol name (used with ``action="callers"/"callees"/"node"``).
        pattern: Optional file path filter (used with ``action="files"``).
        limit: Maximum number of results (default 100, max 500).

    Returns:
        A dict with query results or an ``"error"`` key on failure.
    """
    try:
        # Validate action
        valid_actions = ("search", "callers", "callees", "node", "files", "status")
        if action not in valid_actions:
            return {
                "error": (
                    f"Unknown action {action!r}. "
                    f"Valid actions: {', '.join(valid_actions)}"
                ),
            }

        # Check .codegraph directory and db file exist
        if not _CODEGRAPH_DIR.is_dir():
            return {
                "error": (
                    ".codegraph directory not found — run codegraph index first. "
                    f"Expected at: {_CODEGRAPH_DIR}"
                ),
            }
        if not _DB_PATH.is_file():
            return {
                "error": (
                    "codegraph.db not found — run codegraph index first. "
                    f"Expected at: {_DB_PATH}"
                ),
            }

        # Clamp limit to safe bounds
        limit = max(1, min(limit, 500))

        # Route to handler
        if action == "search":
            return await _search(query, limit)
        elif action == "callers":
            return await _callers(symbol, limit)
        elif action == "callees":
            return await _callees(symbol, limit)
        elif action == "node":
            return await _node(symbol)
        elif action == "files":
            return await _files(pattern, limit)
        else:  # status
            return await _status()
    except Exception as exc:
        logger.exception("codegraph(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════


def _get_connection() -> sqlite3.Connection:
    """Open a read-only connection to the CodeGraph database.

    Uses ``?mode=ro`` in the URI to enforce a true read-only connection,
    preventing any accidental writes.
    """
    uri = f"file:{_DB_PATH.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _check_tables(conn: sqlite3.Connection) -> str | None:
    """Verify that required tables exist.

    Returns an error message string if tables are missing, or ``None``
    if everything is fine.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name IN ('nodes', 'edges')"
    )
    existing = {row["name"] for row in cur.fetchall()}
    missing = {"nodes", "edges"} - existing
    if missing:
        return (
            f"CodeGraph database is missing tables: "
            f"{', '.join(sorted(missing))}. Run: codegraph index"
        )
    return None


# ══════════════════════════════════════════════════════════════════════════
# Action implementations (run in executor for non-blocking I/O)
# ══════════════════════════════════════════════════════════════════════════


async def _search(name_query: str, limit: int) -> dict[str, Any]:
    """Search symbols by name using a LIKE match."""
    name_query = name_query.strip()
    if not name_query:
        return {"error": "'query' parameter is required for action='search'"}

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        conn = _get_connection()
        try:
            err = _check_tables(conn)
            if err:
                return {"error": err}

            pattern = f"%{name_query}%"
            cur = conn.execute(
                """SELECT name, kind, file_path, start_line, signature
                   FROM nodes
                   WHERE name LIKE ?
                   ORDER BY kind, name
                   LIMIT ?""",
                (pattern, limit),
            )
            rows = [
                {
                    "name": r["name"],
                    "kind": r["kind"],
                    "file": r["file_path"],
                    "line": r["start_line"],
                    "signature": r["signature"],
                }
                for r in cur.fetchall()
            ]
            return {
                "ok": True,
                "results": rows,
                "count": len(rows),
                "query": name_query,
            }
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do)


async def _callers(symbol_name: str, limit: int) -> dict[str, Any]:
    """Find functions that directly call the given symbol."""
    symbol_name = symbol_name.strip()
    if not symbol_name:
        return {"error": "'symbol' parameter is required for action='callers'"}

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        conn = _get_connection()
        try:
            err = _check_tables(conn)
            if err:
                return {"error": err}

            cur = conn.execute(
                """SELECT DISTINCT caller.name AS caller_name,
                          caller.file_path AS caller_file,
                          caller.start_line AS caller_line
                   FROM edges e
                   JOIN nodes callee ON e.target = callee.id
                   JOIN nodes caller ON e.source = caller.id
                   WHERE callee.name = ? AND e.kind = 'calls'
                   ORDER BY caller.file_path, caller.start_line
                   LIMIT ?""",
                (symbol_name, limit),
            )
            rows = [
                {
                    "caller": r["caller_name"],
                    "file": r["caller_file"],
                    "line": r["caller_line"],
                }
                for r in cur.fetchall()
            ]
            return {
                "ok": True,
                "symbol": symbol_name,
                "results": rows,
                "count": len(rows),
            }
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do)


async def _callees(symbol_name: str, limit: int) -> dict[str, Any]:
    """Find functions directly called by the given symbol."""
    symbol_name = symbol_name.strip()
    if not symbol_name:
        return {"error": "'symbol' parameter is required for action='callees'"}

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        conn = _get_connection()
        try:
            err = _check_tables(conn)
            if err:
                return {"error": err}

            cur = conn.execute(
                """SELECT DISTINCT callee.name AS callee_name,
                          callee.file_path AS callee_file,
                          callee.start_line AS callee_line
                   FROM edges e
                   JOIN nodes caller ON e.source = caller.id
                   JOIN nodes callee ON e.target = callee.id
                   WHERE caller.name = ? AND e.kind = 'calls'
                   ORDER BY callee.file_path, callee.start_line
                   LIMIT ?""",
                (symbol_name, limit),
            )
            rows = [
                {
                    "callee": r["callee_name"],
                    "file": r["callee_file"],
                    "line": r["callee_line"],
                }
                for r in cur.fetchall()
            ]
            return {
                "ok": True,
                "symbol": symbol_name,
                "results": rows,
                "count": len(rows),
            }
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do)


async def _node(symbol_name: str) -> dict[str, Any]:
    """Get full info about a specific symbol."""
    symbol_name = symbol_name.strip()
    if not symbol_name:
        return {"error": "'symbol' parameter is required for action='node'"}

    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        conn = _get_connection()
        try:
            err = _check_tables(conn)
            if err:
                return {"error": err}

            cur = conn.execute(
                """SELECT name, kind, file_path, start_line, end_line,
                          start_column, end_column, signature, docstring,
                          visibility, is_exported, is_async, is_static, language
                   FROM nodes
                   WHERE name = ?
                   ORDER BY file_path, start_line
                   LIMIT 1""",
                (symbol_name,),
            )
            row = cur.fetchone()
            if row is None:
                return {
                    "ok": False,
                    "error": (f"Symbol {symbol_name!r} not found in CodeGraph index"),
                }
            return {
                "ok": True,
                "name": row["name"],
                "kind": row["kind"],
                "file": row["file_path"],
                "line": row["start_line"],
                "end_line": row["end_line"],
                "col": row["start_column"],
                "signature": row["signature"],
                "docstring": row["docstring"],
                "visibility": row["visibility"],
                "exported": bool(row["is_exported"]),
                "async": bool(row["is_async"]),
                "static": bool(row["is_static"]),
                "language": row["language"],
            }
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do)


async def _files(pattern: str, limit: int) -> dict[str, Any]:
    """List indexed project files with symbol counts.

    If *pattern* is provided, it is treated as a glob-like pattern where
    ``*`` matches any sequence of characters and ``?`` matches any single
    character.  The pattern is matched against the file path in the DB.
    """
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        conn = _get_connection()
        try:
            err = _check_tables(conn)
            if err:
                return {"error": err}

            pat = pattern.strip() if pattern else ""
            if pat:
                # Convert simple glob to SQL LIKE pattern
                sql_pat = pat.replace("*", "%").replace("?", "_")
                cur = conn.execute(
                    """SELECT f.path, f.node_count, f.language, f.size
                       FROM files f
                       WHERE f.path LIKE ?
                       ORDER BY f.path
                       LIMIT ?""",
                    (sql_pat, limit),
                )
            else:
                cur = conn.execute(
                    """SELECT f.path, f.node_count, f.language, f.size
                       FROM files f
                       ORDER BY f.path
                       LIMIT ?""",
                    (limit,),
                )

            rows = [
                {
                    "file": r["path"],
                    "symbols": r["node_count"],
                    "language": r["language"],
                    "size": r["size"],
                }
                for r in cur.fetchall()
            ]
            return {
                "ok": True,
                "results": rows,
                "count": len(rows),
                "pattern": pat or None,
            }
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do)


async def _status() -> dict[str, Any]:
    """Return database statistics: counts, DB size, last indexed time."""
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        if not _DB_PATH.is_file():
            return {
                "ok": False,
                "error": (
                    f"codegraph.db not found at {_DB_PATH} — run codegraph index first"
                ),
            }

        conn = _get_connection()
        try:
            err = _check_tables(conn)
            if err:
                return {"error": err}

            total_symbols = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            total_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

            node_kinds = conn.execute(
                "SELECT kind, COUNT(*) as cnt FROM nodes "
                "GROUP BY kind ORDER BY cnt DESC"
            ).fetchall()

            db_stat = _DB_PATH.stat()
            db_size_mb = round(db_stat.st_size / (1024**2), 2)
            last_indexed_ts = db_stat.st_mtime

            return {
                "ok": True,
                "total_symbols": total_symbols,
                "total_edges": total_edges,
                "total_files": total_files,
                "node_kinds": {r["kind"]: r["cnt"] for r in node_kinds},
                "db_size_mb": db_size_mb,
                "last_indexed": datetime.fromtimestamp(
                    last_indexed_ts, tz=timezone.utc
                ).isoformat(),
            }
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do)
