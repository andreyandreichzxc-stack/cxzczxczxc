"""Tests for the SDD sandbox executor."""

from __future__ import annotations

import pytest

from src.core.actions.sdd_executor import execute_code


@pytest.mark.asyncio
async def test_execute_code_allows_basic_batch_calculation():
    result = await execute_code("_result = sum(range(5))")

    assert result["error"] is None
    assert result["result"] == "10"


@pytest.mark.asyncio
async def test_execute_code_rejects_while_loops():
    result = await execute_code("while True:\n    pass")

    assert result["error"] == "Unsafe operation: While is not allowed"
