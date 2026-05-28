"""mcp_hash tool — registered via @tool decorator.

File / string hashing using Python's built-in hashlib.

Actions:
  - **file** — hash the contents of a file inside data/.
  - **string** — hash a text string.
  - **verify** — hash a file and compare against an expected digest.

Supported algorithms: md5, sha1, sha256, sha512.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import ToolActionSpec, tool

logger = logging.getLogger(__name__)

_SUPPORTED_ALGORITHMS = frozenset({"md5", "sha1", "sha256", "sha512"})


@tool(
    name="mcp_hash",
    description=(
        "Hash file contents or strings with hashlib.\n\n"
        "Actions:\n"
        "- **file** — hash a file inside data/.\n"
        "- **string** — hash a text string.\n"
        "- **verify** — hash a file and compare vs expected digest.\n\n"
        "Supported algorithms: md5, sha1, sha256, sha512.\n\n"
        "Examples:\n"
        '  action="file" path="data.txt" algorithm="sha256"\n'
        '  action="string" text="hello" algorithm="md5"\n'
        '  action="verify" path="data.txt" expected="abc123..." algorithm="sha256"'
    ),
    category="utility",
    risk="low",
    actions={
        "file": ToolActionSpec(
            name="file",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
        "string": ToolActionSpec(
            name="string",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
        "verify": ToolActionSpec(
            name="verify",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
    },
    params={
        "action": "str — 'file', 'string' or 'verify'",
        "path": "str — relative path inside data/ to the file (required for file/verify)",
        "text": "str — text to hash (required for string)",
        "algorithm": "str — one of: md5, sha1, sha256, sha512 (default sha256)",
        "expected": "str — expected hex digest (required for verify)",
    },
)
async def mcp_hash(
    action: str = "",
    path: str = "",
    text: str = "",
    algorithm: str = "sha256",
    expected: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Hash file contents or strings using Python's built-in hashlib."""
    try:
        algorithm = algorithm.strip().lower()
        if algorithm not in _SUPPORTED_ALGORITHMS:
            return {
                "error": (
                    f"Unsupported algorithm {algorithm!r}. "
                    f"Supported: {', '.join(sorted(_SUPPORTED_ALGORITHMS))}"
                )
            }

        if action == "file":
            return await _hash_file(path, algorithm)
        elif action == "string":
            return _hash_string(text, algorithm)
        elif action == "verify":
            return await _verify_file(path, expected, algorithm)
        else:
            return {
                "error": (
                    f"Unknown action {action!r}. Valid actions: file, string, verify"
                )
            }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("mcp_hash(%r) failed", action)
        return {"error": f"Unexpected error: {exc}"}


# ── Helpers ──────────────────────────────────────────────────────────────


def _new_hasher(algorithm: str) -> hashlib._Hash:
    return hashlib.new(algorithm)


async def _hash_file(path: str, algorithm: str) -> dict[str, Any]:
    if not path:
        return {"error": "path parameter is required for action='file'"}

    data_dir: Path = settings.data_dir
    file_path = _resolve_path(data_dir, path)

    if not file_path.exists():
        return {"error": f"File not found: {path} (resolved: {file_path})"}
    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}

    hasher = _new_hasher(algorithm)
    # Read in chunks for memory efficiency
    data = file_path.read_bytes()
    hasher.update(data)
    digest = hasher.hexdigest()

    return {
        "ok": True,
        "action": "file",
        "path": str(file_path),
        "algorithm": algorithm,
        "hash": digest,
    }


def _hash_string(text: str, algorithm: str) -> dict[str, Any]:
    if not text:
        return {"error": "text parameter is required for action='string'"}

    hasher = _new_hasher(algorithm)
    hasher.update(text.encode("utf-8"))
    digest = hasher.hexdigest()

    return {
        "ok": True,
        "action": "string",
        "algorithm": algorithm,
        "hash": digest,
    }


async def _verify_file(path: str, expected: str, algorithm: str) -> dict[str, Any]:
    if not path:
        return {"error": "path parameter is required for action='verify'"}
    if not expected:
        return {"error": "expected parameter is required for action='verify'"}

    data_dir: Path = settings.data_dir
    file_path = _resolve_path(data_dir, path)

    if not file_path.exists():
        return {"error": f"File not found: {path} (resolved: {file_path})"}
    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}

    hasher = _new_hasher(algorithm)
    data = file_path.read_bytes()
    hasher.update(data)
    actual = hasher.hexdigest()
    match = actual == expected.strip().lower()

    return {
        "ok": True,
        "action": "verify",
        "path": str(file_path),
        "algorithm": algorithm,
        "expected": expected.strip(),
        "actual": actual,
        "match": match,
    }


def _resolve_path(data_dir: Path, user_path: str) -> Path:
    """Resolve a user-supplied relative path inside data_dir.

    Raises ValueError if the path attempts directory traversal.
    """
    resolved = (data_dir / user_path).resolve()
    if not str(resolved).startswith(str(data_dir.resolve())):
        raise ValueError(f"Path {user_path!r} escapes the data directory")
    return resolved
