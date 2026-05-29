"""MCP tool: калькулятор и конвертер единиц.

Actions:
- ``action="calc" expression="2 + 2 * 5"`` — вычислить математическое выражение
- ``action="convert" expression="10 km to miles"`` — конвертация единиц

Поддерживает базовые операции, тригонометрию, логарифмы, конвертацию валют,
температуры, расстояний, веса.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── DoS protection limits ───────────────────────────────────────────────
MAX_FACTORIAL_ARG: int = 1000
MAX_EXPONENT: int = 1000

# ── Allowed math names, functions, and constants ────────────────────────

_MATH_CONSTANTS: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _safe_factorial(n: int) -> int:
    """Wrap math.factorial with a reasonable upper bound (DoS protection)."""
    if n > MAX_FACTORIAL_ARG:
        raise ValueError(f"factorial argument {n} exceeds maximum {MAX_FACTORIAL_ARG}")
    if n < 0:
        raise ValueError("factorial not defined for negative numbers")
    return math.factorial(n)


def _safe_pow(base: float, exp: float) -> float:
    """Wrap pow() with an exponent bound (DoS protection)."""
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"Exponent {exp} exceeds maximum {MAX_EXPONENT}")
    return pow(base, exp)


_MATH_FUNCS: dict[str, Any] = {
    # Trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Logarithms / roots
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ln": math.log,  # natural log alias
    # Helpers
    "abs": abs,
    "round": round,
    "pow": _safe_pow,
    "ceil": math.ceil,
    "floor": math.floor,
    # Conversions
    "degrees": math.degrees,
    "radians": math.radians,
    # Number theory
    "factorial": _safe_factorial,
    "gcd": math.gcd,
}

# lcm was added in Python 3.9
_lcm = getattr(math, "lcm", lambda a, b: abs(a * b) // math.gcd(a, b) if b else 0)
_MATH_FUNCS["lcm"] = _lcm

_ALLOWED_FUNC_NAMES = frozenset(_MATH_FUNCS)
_ALLOWED_NAMES = frozenset(_MATH_CONSTANTS)

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
        # Expression context nodes (appear in ast.walk)
        ast.Load,
        ast.Store,
        ast.Del,
    }
)


# ── Conversion table ────────────────────────────────────────────────────

_CONVERSIONS: dict[tuple[str, str], Any] = {
    # Distance
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("km", "m"): 1000,
    ("m", "km"): 0.001,
    ("miles", "m"): 1609.34,
    ("m", "miles"): 0.000621371,
    ("cm", "inches"): 0.393701,
    ("inches", "cm"): 2.54,
    ("m", "ft"): 3.28084,
    ("ft", "m"): 0.3048,
    # Weight
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("g", "oz"): 0.035274,
    ("oz", "g"): 28.3495,
    # Temperature
    ("c", "f"): lambda v: v * 9 / 5 + 32,
    ("f", "c"): lambda v: (v - 32) * 5 / 9,
    # Currency (approximate rates)
    ("usd", "eur"): 0.92,
    ("eur", "usd"): 1.09,
    ("usd", "rub"): 90.0,
    ("rub", "usd"): 0.011,
    ("eur", "rub"): 98.0,
    ("rub", "eur"): 0.0102,
}


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_calculator
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_calculator",
    description=(
        "Математический калькулятор и конвертер единиц.\n"
        "Два действия:\n"
        "- 'calc' — вычислить выражение (поддерживает +, -, *, /, **, %, "
        "скобки, тригонометрию, логарифмы). Пример: 'sqrt(144) + sin(pi/2)'\n"
        "- 'convert' — конвертировать единицы. "
        "Пример: '10 km to miles', '100 usd to eur', '37 c to f'"
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'calc' или 'convert'",
        "expression": (
            "str — математическое выражение для 'calc' "
            "или строка конвертации '10 km to miles' для 'convert'"
        ),
    },
)
async def mcp_calculator(
    action: str = "calc",
    expression: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Калькулятор и конвертер единиц.

    Args:
        action: ``"calc"`` или ``"convert"``.
        expression: Выражение для вычисления или строка конвертации.

    Returns:
        dict с ``ok`` / ``error``, и результатом.
    """
    # ── Calc ────────────────────────────────────────────────────────────
    if action == "calc":
        if not expression or not expression.strip():
            return {"error": "expression is required for calc action"}

        try:
            result = _safe_eval(expression.strip())
            return {
                "ok": True,
                "expression": expression.strip(),
                "result": result,
            }
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("mcp_calculator calc(%r) failed", expression)
            return {"error": f"Calculation error: {exc}"}

    # ── Convert ─────────────────────────────────────────────────────────
    if action == "convert":
        try:
            if not expression or not expression.strip():
                return {
                    "error": "expression is required for convert action (e.g. '10 km to miles')"
                }

            m = re.match(
                r"([\d.]+)\s*(\w+)\s+(?:to|in|в)\s+(\w+)",
                expression.strip(),
                re.IGNORECASE,
            )
            if not m:
                return {
                    "error": (
                        "Format: '10 km to miles' or '100 usd to eur'. "
                        "Supported: km/miles/m/cm/inches/ft, kg/lbs/g/oz, C/F, USD/EUR/RUB."
                    )
                }

            value = float(m.group(1))
            from_unit = m.group(2).lower()
            to_unit = m.group(3).lower()

            conv = _CONVERSIONS.get((from_unit, to_unit))
            if conv is None:
                supported = ", ".join(f"{f}→{t}" for f, t in sorted(_CONVERSIONS))
                return {
                    "error": (
                        f"Unknown conversion: {from_unit} → {to_unit}. "
                        f"Supported pairs: {supported}"
                    )
                }

            result = conv(value) if callable(conv) else value * conv

            return {
                "ok": True,
                "from": f"{value} {from_unit}",
                "to": f"{result:.2f} {to_unit}",
                "result": result,
            }
        except Exception as exc:
            return {"error": str(exc)}

    return {"error": f"Unknown action: {action}. Use 'calc' or 'convert'."}


# ══════════════════════════════════════════════════════════════════════════
# Safe expression evaluator (AST-based)
# ══════════════════════════════════════════════════════════════════════════


def _safe_eval(expr: str) -> Any:
    """Parse, validate, and evaluate a mathematical expression.

    Uses AST whitelisting (same approach as mcp_calc.py) to block dangerous
    constructs.  Supports ``^`` as power (rewritten to ``**`` before parse).

    Raises:
        ValueError: If the expression contains disallowed constructs.
    """
    # Replace ^ with ** (power operator) before parsing
    expr = expr.replace("^", "**")

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

        # Block attribute access (prevents e.g. math.sqrt, __builtins__)
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
            # Static guard: reject factorial(constant) > MAX_FACTORIAL_ARG
            if node.func.id == "factorial" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(
                    arg.value, (int, float)
                ):
                    if arg.value > MAX_FACTORIAL_ARG:
                        raise ValueError(
                            f"factorial argument {arg.value} exceeds "
                            f"maximum {MAX_FACTORIAL_ARG}"
                        )

        # Static guard: reject constant exponents > MAX_EXPONENT
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(
                node.right.value, (int, float)
            ):
                if abs(node.right.value) > MAX_EXPONENT:
                    raise ValueError(
                        f"Exponent {node.right.value} exceeds maximum {MAX_EXPONENT}"
                    )

        # Only allow pre-approved names
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES and node.id not in _ALLOWED_FUNC_NAMES:
                raise ValueError(
                    f"Name '{node.id}' is not allowed. "
                    f"Allowed constants: {', '.join(sorted(_ALLOWED_NAMES))}"
                )

    # ── 3. Evaluate in restricted namespace ─────────────────────────────
    namespace: dict[str, Any] = {
        "__builtins__": {},
        **_MATH_CONSTANTS,
        **_MATH_FUNCS,
    }

    try:
        compiled = compile(tree, "<string>", "eval")
        return eval(compiled, namespace)
    except Exception as exc:
        raise ValueError(f"Evaluation error: {exc}") from exc
