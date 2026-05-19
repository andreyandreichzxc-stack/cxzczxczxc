"""Оркестратор сабагентов: запуск, кэширование, параллелизм."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from typing import Any

from src.agents.base import AgentResult, AgentTask


logger = logging.getLogger(__name__)


async def run_agent(
    provider,  # LLMProvider
    agent_type: str,
    system_prompt: str,
    user_prompt: str,
    *,
    context: dict | None = None,
    cache_ttl: int = 0,
    heavy: bool = False,
) -> AgentResult:
    """Запустить одного сабагента. С кэшированием."""
    from src.core.agent_cache import cache_get_or_set
    from src.llm.base import ChatMessage

    task_id = uuid.uuid4().hex[:12]
    context = context or {}

    # Ключ кэша: agent_type + хэш от user_prompt
    params_hash = hashlib.md5((agent_type + user_prompt).encode()).hexdigest()[:12]

    async def _call_llm():
        start = time.time()
        try:
            raw = await provider.chat(
                [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ],
                heavy=heavy,
            )
            elapsed = (time.time() - start) * 1000
            # Очистка от markdown
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
                raw = re.sub(r"\n?\s*```\s*$", "", raw)
            if raw.startswith("{"):
                data = json.loads(raw)
            else:
                # Попробовать найти JSON внутри
                m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                else:
                    data = {"raw": raw}
            return AgentResult(
                task_id=task_id,
                agent_type=agent_type,
                success=True,
                data=data,
                tokens_used=0,
                cache_key=f"{agent_type}:{params_hash}",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            logger.exception("Agent %s failed", agent_type)
            return AgentResult(
                task_id=task_id,
                agent_type=agent_type,
                success=False,
                data={},
                error=str(e),
                elapsed_ms=(time.time() - start) * 1000,
            )

    if cache_ttl > 0:
        result = await cache_get_or_set(
            agent_type,
            params_hash,
            factory=_call_llm,
            ttl_seconds=cache_ttl,
        )
        return result if isinstance(result, AgentResult) else await _call_llm()
    return await _call_llm()


async def run_parallel(
    provider,  # LLMProvider
    tasks: list[AgentTask],
    heavy: bool = False,
) -> list[AgentResult]:
    """Запустить несколько агентов параллельно."""

    async def _run_one(task: AgentTask) -> AgentResult:
        return await run_agent(
            provider,
            task.agent_type,
            task.system_prompt,
            task.user_prompt,
            context=task.context,
            cache_ttl=task.cache_ttl,
            heavy=heavy,
        )

    return await asyncio.gather(*[_run_one(t) for t in tasks])
