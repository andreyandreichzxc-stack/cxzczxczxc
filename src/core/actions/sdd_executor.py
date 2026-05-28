"""SDD (Structured Data Dispatcher) — safe Python sandbox for LLM-generated code.

Allows an LLM to write and execute short Python scripts for batch operations
(e.g. marking all reminders as done, bulk-tagging facts, mass-updating data).
Provides 10-50x speed improvement over sequential LLM calls by running
user-generated code in a restricted AST-level sandbox.
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── AST whitelist ─────────────────────────────────────────────────────────
# Only these node types are allowed in submitted code.

_ALLOWED_NODES: set[type[ast.AST]] = {
    # Top-level
    ast.Module,
    ast.Expr,
    ast.Pass,
    # Assignment
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    # Primitives / literals
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.List,
    ast.Dict,
    ast.Tuple,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.comprehension,  # generator inside comprehensions (Python 3.12+)
    ast.IfExp,  # ternary: a if cond else b
    ast.JoinedStr,  # f-strings: f"...{x}..."
    ast.FormattedValue,  # {expr} inside f-strings
    ast.Starred,  # *args unpacking
    ast.Slice,  # slicing: x[1:2]
    # Calls
    ast.Call,
    ast.keyword,
    # Operators
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    # Control flow
    ast.If,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Return,
    ast.Delete,
    # Expression context
    ast.Load,
    ast.Store,
    ast.Del,
    # Arithmetic operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # Comparison operators
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    # Boolean operators
    ast.And,
    ast.Or,
    ast.Not,
    # Augmented assignment operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
}

# Names that are strictly forbidden in submitted code (both as Name and as
# Attribute targets).
_BLACKLIST: set[str] = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "__builtins__",
    "__subclasses__",
    "__class__",
}


# ── Validation helpers ────────────────────────────────────────────────────


def _is_safe(node: ast.AST) -> bool:
    """Recursively check that all nodes in the AST are allowed.

    Returns ``True`` if the entire tree passes the whitelist and blacklist
    checks.
    """
    if type(node) not in _ALLOWED_NODES:
        return False
    # Check blacklisted names
    if isinstance(node, ast.Name) and node.id in _BLACKLIST:
        return False
    # Check blacklisted attributes
    if isinstance(node, ast.Attribute) and node.attr in _BLACKLIST:
        return False
    for child in ast.iter_child_nodes(node):
        if not _is_safe(child):
            return False
    return True


# ── Execution ─────────────────────────────────────────────────────────────


async def execute_code(code: str, **kwargs: Any) -> dict[str, Any]:
    """Safely execute LLM-generated Python code in a restricted sandbox.

    The submitted *code* is parsed, validated against an AST whitelist, and
    executed in a controlled namespace with a limited set of builtins.

    **Available builtins:** ``print``, ``len``, ``range``, ``int``, ``str``,
    ``float``, ``bool``, ``list``, ``dict``, ``set``, ``tuple``, ``zip``,
    ``enumerate``, ``sorted``, ``min``, ``max``, ``sum``, ``any``, ``all``,
    ``isinstance``.

    **Available via kwargs:** any keyword arguments passed by the caller
    (e.g. ``session``, ``user``, ``provider``, ``test_data``).

    **Convention:** the code may set a ``_result`` variable in the global
    namespace; its value will be returned in the ``"result"`` field of the
    response dict.

    Args:
        code: Valid Python source code (no imports, no ``eval``/``exec``).
        **kwargs: Names to inject into the execution namespace.

    Returns:
        A dict with keys:
        - ``"output"``: captured ``print()`` output (truncated to 5000 chars).
        - ``"result"``: string representation of ``_result`` (truncated to
          2000 chars), or ``None`` if not set.
        - ``"error"``: error message on failure, or ``None`` on success.
    """
    # 1. Parse and validate AST
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return {"output": "", "result": None, "error": f"Syntax error: {e}"}

    if not _is_safe(tree):
        # Pinpoint the first disallowed node for a helpful error message
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_NODES:
                return {
                    "output": "",
                    "result": None,
                    "error": f"Unsafe operation: {type(node).__name__} is not allowed",
                }
        return {
            "output": "",
            "result": None,
            "error": "Code contains unsafe operations (blacklisted names)",
        }

    # 2. Prepare safe builtins namespace
    safe_builtins: dict[str, Any] = {
        # Functions
        "print": print,
        "len": len,
        "range": range,
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "zip": zip,
        "enumerate": enumerate,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        # Constants
        "True": True,
        "False": False,
        "None": None,
    }

    # 3. Build execution namespace
    # kwargs dict is available so code can access params by key: data = kwargs.get('test_data', [])
    # Sanitize kwargs — never pass DB/callbacks to sandbox
    _safe_kwargs: dict[str, Any] = {
        k: v
        for k, v in kwargs.items()
        if k not in ("session", "user", "provider", "userbot_manager", "owner", "bot")
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "kwargs": _safe_kwargs,
        **_safe_kwargs,
    }

    # 4. Capture print() output
    output_buffer = io.StringIO()
    safe_builtins["print"] = (
        lambda *args, _sep=" ", _end="\n", _file=output_buffer, **kw: print(  # noqa: E731
            *args, sep=kw.get("sep", _sep), end=kw.get("end", _end), file=_file
        )
    )

    # 5. Execute (with timeout)
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: exec(compile(tree, "<sdd>", "exec"), namespace),  # nosec: B102 — sandboxed via AST whitelist
            ),
            timeout=5.0,
        )
        result_value = namespace.get("_result", None)
        output = output_buffer.getvalue()
        return {
            "output": output.strip()[:5000],
            "result": str(result_value)[:2000] if result_value is not None else None,
            "error": None,
        }
    except asyncio.TimeoutError:
        return {
            "error": "execution timed out (5s limit)",
            "output": output_buffer.getvalue().strip()[:2000],
            "result": None,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "output": output_buffer.getvalue().strip()[:2000],
            "result": None,
        }


# ── @tool registration ────────────────────────────────────────────────────


@tool(
    name="execute_code",
    description=(
        "Выполняет безопасный Python-код для пакетных операций. "
        "Используй для: отметить несколько напоминаний done, "
        "проставить теги фактам, массово обновить данные. "
        "НЕ используй для единичных операций. "
        "(нет доступа к БД/сессии — для массовых вычислений, не для запросов данных)"
    ),
    category="system",
    risk="critical",
    requires_confirmation=True,
    params={
        "code": "str — валидный Python-код (безопасный, без импортов, без eval/exec)"
    },
    output_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "result": {"description": "Value of _result variable from executed code"},
            "output": {"type": "string", "description": "Captured print() output"},
            "error": {
                "type": "string",
                "description": "Execution error or sandbox rejection",
            },
        },
        "required": ["ok"],
    },
)
async def _execute_code_tool(code: str, **kwargs: Any) -> dict[str, Any]:
    """Tool wrapper — delegates to the sandbox ``execute_code``."""
    return await execute_code(code, **kwargs)
