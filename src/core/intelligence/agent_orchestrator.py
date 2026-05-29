"""
Agent Orchestrator — умный планировщик sub-агентов.

Решает три проблемы:
1. Один зависший агент не блокирует остальных (per-agent timeout)
2. Упавший агент не убивает всю пачку (partial results)
3. Агент с repeat-фейлами получает cooldown — не дёргаем зря
4. Одинаковые запросы кешируются (TTL-кеш)

Пример использования:
    orchestrator = AgentOrchestrator(AGENT_SPECS)

    results, errors = await orchestrator.execute(
        agents_to_call=[
            {"agent": "search", "query": "последние новости"},
            {"agent": "memory", "query": "что я просил запомнить"},
            {"agent": "urgency", "query": "срочно ли это"},
        ],
        provider=llm_provider,
        owner_id=123,
    )
    # results: [{"data": ..., "agent": "search", ...}, ...]
    # errors:  []  или ["urgency agent in cooldown: ..."]

    health = orchestrator.get_health()
    # {"search": {"ok": True}, "urgency": {"ok": False, "cooldown_until": ...}}
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

HEALTH_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "agent_health.json"
)


# ═══════════════════════════════════════════════════════════════════
# AgentSpec — метаданные агента
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AgentSpec:
    """Спецификация агента: лимиты, кеш, зависимости."""

    name: str
    timeout: float = 30.0  # секунд на один вызов агента
    max_retries: int = 2  # повторных попыток при transient-ошибке
    cooldown_seconds: float = 120.0  # кулдаун после max_retries неудач подряд
    cache_ttl: float = 300.0  # секунд жизни кеша (0 = без кеша)
    dependencies: list[str] = field(
        default_factory=list
    )  # имена агентов, которые должны выполниться первыми
    priority: int = (
        2  # 1=низкий, 2=нормальный, 3=высокий (резерв для будущих DAG-планировщиков)
    )
    purpose: str = (
        "main"  # Для _PURPOSE_SEMAPHORES в router.py (concurrency-лимит на тип задачи).
    )
    # Оркестратор это поле НЕ использует — оно нужно только LLM-роутеру.


# ═══════════════════════════════════════════════════════════════════
# AgentHealth — трекинг здоровья агентов
# ═══════════════════════════════════════════════════════════════════


class AgentHealth:
    """Отслеживает последовательные неудачи агентов и кулдаун.

    При рестарте счётчик неудач сбрасывается, но кулдаун сохраняется в JSON.
    """

    def __init__(self) -> None:
        self._failures: dict[str, int] = {}  # agent_name → consecutive failures
        self._cooldown_until: dict[str, float] = {}  # agent_name → monotonic timestamp
        self._load()

    def is_ok(self, agent_name: str) -> bool:
        """True если агент готов к вызову (не в кулдауне)."""
        cooldown = self._cooldown_until.get(agent_name)
        if cooldown is not None and time.monotonic() < cooldown:
            return False
        return True

    def cooldown_remaining(self, agent_name: str) -> float:
        """Секунд до конца кулдауна. 0 если не в кулдауне."""
        cooldown = self._cooldown_until.get(agent_name, 0.0)
        remaining = cooldown - time.monotonic()
        return max(0.0, remaining)

    def mark_success(self, agent_name: str) -> None:
        """Сброс счётчика неудач при успехе."""
        self._failures.pop(agent_name, None)
        self._cooldown_until.pop(agent_name, None)

    def mark_failure(
        self, agent_name: str, max_retries: int, cooldown_seconds: float
    ) -> bool:
        """Регистрирует неудачу. Возвращает True если агент ушёл в кулдаун."""
        count = self._failures.get(agent_name, 0) + 1
        self._failures[agent_name] = count
        if count >= max_retries:
            self._cooldown_until[agent_name] = time.monotonic() + cooldown_seconds
            self._save()
            logger.warning(
                "Agent %s failed %d times — cooldown for %.0fs",
                agent_name,
                count,
                cooldown_seconds,
            )
            return True
        return False

    def _load(self) -> None:
        """Восстанавливает неистекшие кулдауны из JSON-файла."""
        try:
            with open(HEALTH_FILE, "r", encoding="utf-8") as f:
                data = __import__("json").load(f)
        except (FileNotFoundError, OSError, ValueError):
            return
        now_wall = time.time()
        for agent_name, cooldown_wall in data.get("cooldowns", {}).items():
            if cooldown_wall > now_wall:
                remaining = cooldown_wall - now_wall
                self._cooldown_until[agent_name] = time.monotonic() + remaining

    def _save(self) -> None:
        """Сохраняет кулдауны в JSON-файл (wall-clock таймстемпы)."""
        now = time.time()
        now_mono = time.monotonic()
        cooldowns: dict[str, float] = {}
        for agent_name, cooldown_mono in self._cooldown_until.items():
            if cooldown_mono > now_mono:
                cooldowns[agent_name] = now + (cooldown_mono - now_mono)
        try:
            os.makedirs(os.path.dirname(HEALTH_FILE), exist_ok=True)
            with open(HEALTH_FILE, "w", encoding="utf-8") as f:
                __import__("json").dump({"cooldowns": cooldowns}, f)
        except OSError:
            pass

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Снимок состояния для отладки / health-check эндпоинта."""
        now = time.monotonic()
        out: dict[str, dict[str, Any]] = {}
        all_agents = set(self._failures) | set(self._cooldown_until)
        for name in all_agents:
            cooldown = self._cooldown_until.get(name, 0.0)
            out[name] = {
                "consecutive_failures": self._failures.get(name, 0),
                "in_cooldown": cooldown > now,
                "cooldown_remaining": max(0.0, cooldown - now),
            }
        return out


# ═══════════════════════════════════════════════════════════════════
# AgentResultCache — TTL-кеш результатов
# ═══════════════════════════════════════════════════════════════════


class AgentResultCache:
    """LRU TTL-кеш на OrderedDict: (agent_name, query) → (result, stored_at, ttl)."""

    def __init__(self, max_size: int = 200) -> None:
        self._store: collections.OrderedDict[
            tuple[str, str, int], tuple[dict[str, Any], float, float]
        ] = collections.OrderedDict()
        self._max_size = max_size
        self.hits: int = 0
        self.misses: int = 0

    def _make_key(
        self, agent_name: str, query: str, owner_id: int
    ) -> tuple[str, str, int]:
        return (agent_name, query.strip().lower(), owner_id)

    def get(self, agent_name: str, query: str, owner_id: int) -> dict[str, Any] | None:
        """Возвращает закешированный результат или None (LRU touch + TTL evict)."""
        key = self._make_key(agent_name, query, owner_id)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        result, stored_at, ttl = entry
        if time.monotonic() - stored_at > ttl:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        logger.debug("Agent %s cache hit for: %s", agent_name, query[:50])
        return result

    def set(
        self,
        agent_name: str,
        query: str,
        result: dict[str, Any],
        ttl: float,
        owner_id: int,
    ) -> None:
        """Сохраняет результат в кеш с LRU-эвикцией."""
        key = self._make_key(agent_name, query, owner_id)
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = (result, time.monotonic(), ttl)

    def _cleanup_expired(self) -> None:
        """Удаляет все просроченные записи."""
        now = time.monotonic()
        for key in list(self._store.keys()):
            _, stored_at, ttl = self._store[key]
            if now - stored_at > ttl:
                del self._store[key]

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ═══════════════════════════════════════════════════════════════════
# AgentOrchestrator — главный класс
# ═══════════════════════════════════════════════════════════════════


class AgentOrchestrator:
    """Умный планировщик: timeout + partial results + cache + health.

    Пример:
        AGENT_SPECS = {
            "search": AgentSpec(name="search", timeout=25, cache_ttl=300),
            "memory": AgentSpec(name="memory", timeout=15, cache_ttl=120),
            "urgency": AgentSpec(name="urgency", timeout=5, cache_ttl=0),
        }
        orch = AgentOrchestrator(AGENT_SPECS)

        results, errors = await orch.execute(
            agents_to_call=[{"agent": "search", "query": "новости"}, ...],
            provider=llm,
            owner_id=123,
        )
        # results всегда list[dict], errors всегда list[str] — даже если всё упало
    """

    def __init__(self, specs: dict[str, AgentSpec]) -> None:
        self._specs = specs
        self._health = AgentHealth()
        self._cache = AgentResultCache()
        self._stats: dict[str, int] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "successes": 0,
            "failures": 0,
            "cooldowns_hit": 0,
            "timeouts": 0,
            "total_calls": 0,
        }

    # ── публичное API ─────────────────────────────────────────────

    async def execute(
        self,
        agents_to_call: list[dict[str, Any]],
        provider: Any,
        owner_id: int,
        *,
        executor: Any = None,  # callable: (provider, spec_ref, owner_id) -> dict
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Запускает агентов: с таймаутом, кешем, health-чеком.

        Args:
            agents_to_call: [{"agent": "search", "query": "..."}, ...]
                — план от maestro LLM
            provider: LLMProvider (или ProviderFallback)
            owner_id: telegram user id владельца
            executor: callable для вызова одиночного агента.
                По умолчанию — _execute_agent из maestro.
                Передаётся явно чтобы избежать circular import.

        Returns:
            (results, errors):
              - results: всегда list[dict], даже если агент упал — содержит error
              - errors:  list[str] с человекочитаемыми ошибками
                (cooldown, timeout, unknown agent)
        """
        if executor is None:
            from src.core.intelligence.maestro import _execute_agent as executor  # noqa: E402

        if not agents_to_call:
            return [], []

        # Сортируем по зависимостям (уровни — независимые группы)
        levels = self._topo_sort(agents_to_call)

        all_results: list[dict[str, Any]] = []
        all_errors: list[str] = []

        for level in levels:
            # Уровень: все агенты независимы — можно параллельно
            tasks = [
                self._execute_single(spec_ref, provider, owner_id, executor=executor)
                for spec_ref in level
            ]
            level_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(level_results):
                agent_name = level[i].get("agent", "?")
                if isinstance(result, Exception):
                    all_errors.append(
                        f"[{agent_name}] unhandled: {type(result).__name__}: {result}"
                    )
                    all_results.append(
                        {
                            "agent": agent_name,
                            "success": False,
                            "error": str(result),
                            "data": {},
                        }
                    )
                elif isinstance(result, dict):
                    if not result.get("success", False):
                        err = result.get("error", "неизвестная ошибка")
                        all_errors.append(f"[{agent_name}] {err}")
                    all_results.append(result)

        return all_results, all_errors

    async def execute_parallel(
        self,
        owner_id: int,
        query: str,
        agents: list[str],
        provider=None,
        max_concurrent: int = 4,
        *,
        executor=None,
    ) -> list[dict[str, Any]]:
        """Legion-style parallel execution of multiple agents.

        Все агенты запускаются одновременно (с ограничением max_concurrent).
        Результаты собираются и синтезируются в один ответ.

        Args:
            owner_id: telegram user id владельца
            query: общий запрос для всех агентов
            agents: список имён агентов для параллельного запуска
            provider: LLMProvider (обязателен при первом вызове)
            max_concurrent: макс. параллельных LLM-вызовов
            executor: callable для вызова одиночного агента.
                По умолчанию — _execute_agent из maestro.

        Returns:
            list[dict] — результаты агентов (порядок не гарантирован).
        """
        if executor is None:
            from src.core.intelligence.maestro import _execute_agent as executor  # noqa: E402

        # Фильтруем только известных агентов
        known_agents = [name for name in agents if name in self._specs]
        unknown = set(agents) - set(known_agents)
        if unknown:
            logger.warning("execute_parallel: unknown agents skipped: %s", unknown)

        if not known_agents:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_one(agent_name: str) -> dict[str, Any]:
            """Запуск одного агента под семафором."""
            async with semaphore:
                spec_ref: dict[str, Any] = {
                    "agent": agent_name,
                    "query": query,
                }
                return await self._execute_single(
                    spec_ref, provider, owner_id, executor=executor
                )

        tasks = [_run_one(name) for name in known_agents]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Собираем результаты
        results: list[dict[str, Any]] = []
        for i, result in enumerate(raw_results):
            agent_name = known_agents[i]
            if isinstance(result, Exception):
                results.append(
                    {
                        "agent": agent_name,
                        "success": False,
                        "error": f"{type(result).__name__}: {result}",
                        "data": {},
                    }
                )
            elif isinstance(result, dict):
                results.append(result)

        # Синтез
        synthesized = await self._synthesize(provider, query, results)
        if synthesized:
            results.append(
                {
                    "agent": "_synthesis_",
                    "success": True,
                    "data": synthesized,
                    "query": query,
                }
            )

        return results

    def get_health(self) -> dict[str, dict[str, Any]]:
        """Снимок здоровья всех агентов (для /health, мониторинга)."""
        return self._health.snapshot()

    def clear_cache(self) -> None:
        """Сброс кеша (можно дёргать периодически или вручную)."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Возвращает статистику: hit rate, success rate, etc."""
        successes = self._stats["successes"]
        failures = self._stats["failures"]
        return {
            **self._stats,
            "cache_hit_rate": self._cache.hits
            / max(self._cache.hits + self._cache.misses, 1),
            "success_rate": successes / max(successes + failures, 1),
            "cache_size": len(self._cache),
            "health": self._health.snapshot(),
        }

    # ── внутренние методы ─────────────────────────────────────────

    async def _execute_single(
        self,
        agent_spec_ref: dict[str, Any],
        provider: Any,
        owner_id: int,
        *,
        executor: Any,
    ) -> dict[str, Any]:
        """Исполняет одного агента со всей защитой.

        Порядок проверок:
        1. Валидация имени агента
        2. Health-check (cooldown)
        3. Cache lookup
        4. Вызов с timeout + retries
        5. Обновление health / cache
        """
        agent_name = agent_spec_ref.get("agent", "")
        query = agent_spec_ref.get("query", "")
        spec = self._specs.get(agent_name)

        # 1. Неизвестный агент
        if spec is None:
            logger.warning("Unknown agent type: %s", agent_name)
            return {
                "agent": agent_name,
                "success": False,
                "error": f"Неизвестный агент: {agent_name}",
                "data": {},
            }

        self._stats["total_calls"] += 1

        # 2. Health-check
        if not self._health.is_ok(agent_name):
            remaining = self._health.cooldown_remaining(agent_name)
            msg = f"Агент {agent_name} в кулдауне ещё {remaining:.0f}с"
            logger.info(msg)
            self._stats["cooldowns_hit"] += 1
            return {
                "agent": agent_name,
                "success": False,
                "error": msg,
                "data": {},
            }

        # 3. Cache lookup
        if spec.cache_ttl > 0:
            cached = self._cache.get(agent_name, query, owner_id)
            if cached is not None:
                return cached

        # 4. Вызов с timeout + retries
        result = await self._call_with_retries(
            agent_spec_ref, provider, owner_id, spec, executor=executor
        )

        # 5. Health + cache
        if result.get("success", False):
            self._health.mark_success(agent_name)
            self._stats["successes"] += 1
            if spec.cache_ttl > 0:
                self._cache.set(agent_name, query, result, spec.cache_ttl, owner_id)
        else:
            self._stats["failures"] += 1
            in_cooldown = self._health.mark_failure(
                agent_name, spec.max_retries, spec.cooldown_seconds
            )
            if in_cooldown:
                result["error"] = (
                    f"Агент {agent_name} отключён на {spec.cooldown_seconds:.0f}с "
                    f"после {spec.max_retries} неудач. " + result.get("error", "")
                )

        return result

    async def _call_with_retries(
        self,
        agent_spec_ref: dict[str, Any],
        provider: Any,
        owner_id: int,
        spec: AgentSpec,
        *,
        executor: Any,
    ) -> dict[str, Any]:
        """Вызов агента с таймаутом и повторными попытками."""
        last_error: Exception | None = None
        agent_name = agent_spec_ref.get("agent", "")

        for attempt in range(1 + spec.max_retries):
            try:
                return await asyncio.wait_for(
                    executor(provider, agent_spec_ref, owner_id=owner_id),
                    timeout=spec.timeout,
                )
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"Таймаут {spec.timeout:.0f}с")
                self._stats["timeouts"] += 1
                logger.warning(
                    "Agent %s timeout (attempt %d/%d)",
                    agent_name,
                    attempt + 1,
                    spec.max_retries + 1,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Agent %s failed (attempt %d/%d): %s",
                    agent_name,
                    attempt + 1,
                    spec.max_retries + 1,
                    exc,
                )

            if attempt < spec.max_retries:
                await asyncio.sleep(1.0 * (2**attempt))  # exp backoff: 1s, 2s, 4s, ...

        return {
            "agent": agent_name,
            "success": False,
            "error": str(last_error) if last_error else "неизвестная ошибка",
            "data": {},
        }

    async def _synthesize(
        self,
        provider: Any,
        query: str,
        agent_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Синтезирует результаты параллельных агентов в один ответ.

        Простая реализация: конкатенирует с заголовками.
        В будущем — LLM-сводка.
        """
        if not agent_results:
            return {"summary": "No agent results", "query": query}
        if len(agent_results) == 1:
            r = agent_results[0]
            return {
                "summary": r.get("data", ""),
                "query": query,
                "agent": r.get("agent", "agent"),
            }

        parts: list[str] = []
        for r in agent_results:
            agent_name = r.get("agent", "unknown")
            data = r.get("data", "")
            status = "OK" if r.get("success", False) else "FAIL"
            parts.append(f"[{agent_name}] ({status}): {data}")

        return {
            "summary": "\n".join(parts),
            "query": query,
            "agents_merged": len(agent_results),
            "success": any(r.get("success", False) for r in agent_results),
        }

    def _topo_sort(
        self, agents_to_call: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Сортирует агентов по уровням зависимостей.

        Агенты без зависимостей → уровень 0 (можно параллельно).
        Агенты, зависящие от уровня 0 → уровень 1 (после уровня 0).
        """
        if not agents_to_call:
            return []

        agent_names = {a.get("agent", "") for a in agents_to_call}
        levels: list[set[str]] = []
        assigned: set[str] = set()

        # Итеративно находим агентов, чьи зависимости уже назначены
        while len(assigned) < len(agent_names):
            current: set[str] = set()
            for name in agent_names - assigned:
                spec = self._specs.get(name)
                deps = set(spec.dependencies) if spec else set()
                if deps <= assigned:  # все зависимости уже на предыдущих уровнях
                    current.add(name)

            if not current:
                # Циклическая зависимость: собираем минимальный цикл, не все что остались
                cycle_candidates = set()
                for name in agent_names - assigned:
                    spec = self._specs.get(name)
                    deps = set(spec.dependencies) & agent_names if spec else set()
                    if deps - assigned:
                        cycle_candidates.add(name)
                # Если ничего не нашли (крайний случай) — берём всех
                current = (
                    cycle_candidates if cycle_candidates else (agent_names - assigned)
                )

            levels.append(current)
            assigned |= current

        # Преобразуем имена обратно в spec_refs
        result: list[list[dict[str, Any]]] = []
        for level_names in levels:
            level_specs = [a for a in agents_to_call if a.get("agent") in level_names]
            if level_specs:
                result.append(level_specs)

        return result


# ═══════════════════════════════════════════════════════════════════
# Глобальный экземпляр — синглтон на всё приложение
# ═══════════════════════════════════════════════════════════════════

# Создаётся в maestro.py при инициализации:
#   from .agent_orchestrator import AgentOrchestrator, AgentSpec, AGENT_SPECS
#   orchestrator = AgentOrchestrator(AGENT_SPECS)

AGENT_SPECS: dict[str, AgentSpec] = {
    "search": AgentSpec(name="search", timeout=30, cache_ttl=300, purpose="search"),
    "memory": AgentSpec(name="memory", timeout=20, cache_ttl=120, purpose="memory"),
    "urgency": AgentSpec(name="urgency", timeout=10, cache_ttl=0, purpose="analysis"),
    "commitment": AgentSpec(
        name="commitment", timeout=20, cache_ttl=60, purpose="memory"
    ),
    "summarizer": AgentSpec(
        name="summarizer", timeout=30, cache_ttl=180, purpose="analysis"
    ),
    "draft": AgentSpec(name="draft", timeout=30, cache_ttl=60, purpose="draft"),
    "digest": AgentSpec(name="digest", timeout=30, cache_ttl=300, purpose="fallback"),
    "random": AgentSpec(name="random", timeout=60, cache_ttl=0, purpose="fallback"),
}
