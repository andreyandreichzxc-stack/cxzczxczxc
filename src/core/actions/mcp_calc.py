"""mcp_calc tool — registered via @tool decorator.

Safe mathematical expression evaluator.

Features:
- Operators: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``.
- Functions: ``sqrt()``, ``sin()``, ``cos()``, ``abs()``, ``round()``.
- Constants: ``pi``, ``e``.
- AST-based safety checks block: ``__`` dunders, ``import``, ``exec``,
  ``eval``, ``open``, and any attribute access.
"""

from __future__ import annotations

import ast
import logging
import math
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Allowed names and functions ──────────────────────────────────────────

_MATH_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
}

_MATH_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "abs": abs,
    "round": round,
}

_ALLOWED_FUNC_NAMES = frozenset(_MATH_FUNCS)
_ALLOWED_NAMES = frozenset(_MATH_NAMES)

# AST node types permitted in a safe math expression
_SAFE_NODE_TYPES = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Call,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    }
)


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_calc
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_calc",
    description=(
        "Evaluate a mathematical expression safely.  Supports:\n"
        "- Operators: +, -, *, /, ** (power), % (modulo)\n"
        "- Functions: sqrt(), sin(), cos(), abs(), round()\n"
        "- Constants: pi, e\n"
        "Example: '2 + 2 * 5', 'sqrt(144)', 'sin(pi/2)'"
    ),
    category="utility",
    risk="low",
    params={
        "expression": "str — mathematical expression to evaluate",
    },
)
async def mcp_calc(
    expression: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Safe mathematical expression evaluator.

    Args:
        expression: A mathematical expression string.

    Returns:
        A dict with ``result`` and ``expression`` or an ``"error"`` key.
    """
    try:
        if not expression or not expression.strip():
            return {"error": "expression parameter is required"}

        result = _safe_eval(expression.strip())

        return {
            "ok": True,
            "expression": expression.strip(),
            "result": result,
        }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("mcp_calc(%r) failed", expression)
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# Safe expression evaluator (AST-based)
# ══════════════════════════════════════════════════════════════════════════


def _safe_eval(expr: str) -> Any:
    """Parse, validate, and evaluate a mathematical expression.

    The expression is parsed into an AST, validated against a whitelist of
    allowed node types, then executed in a restricted namespace with no
    builtins and only pre-approved math functions and constants.

    Raises:
        ValueError: If the expression contains disallowed constructs.
    """
    # ── 1. Parse ───────────────────────────────────────────────────────
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    # ── 2. Validate AST ────────────────────────────────────────────────
    for node in ast.walk(tree):
        node_type = type(node)

        if node_type not in _SAFE_NODE_TYPES:
            raise ValueError(
                f"Unsupported construct: {node_type.__name__}. "
                f"Only basic math operations are allowed."
            )

        # Block attribute access (prevents e.g. ``math.sqrt``, ``__builtins__``)
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")

        # Only allow specific function names
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            if node.func.id not in _ALLOWED_FUNC_NAMES:
                raise ValueError(
                    f"Function '{node.func.id}' is not allowed. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_FUNC_NAMES))}"
                )

        # Only allow pre-approved names
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES and node.id not in _ALLOWED_FUNC_NAMES:
                raise ValueError(
                    f"Name '{node.id}' is not allowed. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_NAMES))}"
                )

    # ── 3. Evaluate in restricted namespace ─────────────────────────────
    namespace: dict[str, Any] = {
        "__builtins__": {},
        **_MATH_NAMES,
        **_MATH_FUNCS,
    }

    try:
        compiled = compile(tree, "<string>", "eval")
        return eval(compiled, namespace)
    except Exception as exc:
        raise ValueError(f"Evaluation error: {exc}") from exc
