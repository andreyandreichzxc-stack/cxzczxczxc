"""Smart AutoRouter — оркестратор, сам решает как обработать запрос.

This is a FACADE that re-exports from sub-modules:
  - routing/classifier.py   → intent classification, enums, risk/mode detection
  - routing/planner.py      → RouterPlan, RouterTask, make_plan
"""

from __future__ import annotations

from .routing.classifier import (
    RoutePurpose,
    RiskLevel,
    ResponseMode,
    classify_mode,
    classify_risk,
    get_instant_reply,
)
from .routing.planner import (
    RouterPlan,
    RouterTask,
    make_plan,
)

__all__ = [
    "RoutePurpose",
    "RiskLevel",
    "ResponseMode",
    "RouterPlan",
    "RouterTask",
    "classify_mode",
    "classify_risk",
    "get_instant_reply",
    "make_plan",
]
