"""Tests for LLM key/provider fallback behavior."""

from __future__ import annotations

import asyncio

import pytest

from src.llm.base import ChatMessage
from src.llm.router import ExhaustedError, MultiKeyProvider, ProviderFallback


class CapacityError(Exception):
    status_code = 429
    code = "service_tier_capacity_exceeded"


class FakeProvider:
    calls: list[str] = []

    @classmethod
    def clear_calls(cls) -> None:
        cls.calls.clear()

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(
        self, messages, *, heavy: bool = False, task_type: str = "default"
    ) -> str:
        self.calls.append(self.api_key)
        if self.api_key.startswith("bad"):
            raise CapacityError("Service tier capacity exceeded for this model")
        return f"ok:{self.api_key}"

    async def embed(self, text: str) -> list[float]:
        self.calls.append(f"embed:{self.api_key}")
        if self.api_key.startswith("bad"):
            raise CapacityError("429")
        return [1.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def validate_key(self) -> bool:
        return not self.api_key.startswith("bad")

    async def close(self) -> None:
        pass


class SlowProvider(FakeProvider):
    entered: list[str] = []
    release: asyncio.Event | None = None

    @classmethod
    def reset(cls) -> None:
        cls.entered = []
        cls.release = None

    async def chat(
        self, messages, *, heavy: bool = False, task_type: str = "default"
    ) -> str:
        self.entered.append(self.api_key)
        assert self.release is not None
        await self.release.wait()
        return f"ok:{self.api_key}"


@pytest.fixture(autouse=True)
def _cleanup_fake_providers():
    """Очистить shared mutable state после каждого теста."""
    FakeProvider.clear_calls()
    ExhaustingProvider.clear_calls()
    GoodProvider.clear_calls()
    SlowProvider.reset()
    yield


@pytest.mark.asyncio
async def test_multikey_rotates_on_capacity_error():
    FakeProvider.calls = []
    provider = MultiKeyProvider("fake", FakeProvider, ["bad-1", "good-2"])

    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result == "ok:good-2"
    # With exponential backoff (MAX_RETRIES_PER_KEY=3), the bad key is retried
    # 3 times before rotating to the next key.
    assert FakeProvider.calls == ["bad-1", "bad-1", "bad-1", "good-2"]


@pytest.mark.asyncio
async def test_provider_fallback_tries_next_provider_for_chat():
    FakeProvider.calls = []
    primary = MultiKeyProvider("primary", FakeProvider, ["bad-primary"])
    secondary = MultiKeyProvider("secondary", FakeProvider, ["good-secondary"])
    provider = ProviderFallback([primary, secondary])

    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result == "ok:good-secondary"
    assert FakeProvider.calls == [
        "bad-primary",
        "bad-primary",
        "bad-primary",
        "good-secondary",
    ]


@pytest.mark.asyncio
async def test_provider_fallback_keeps_embeddings_on_primary_provider():
    FakeProvider.calls = []
    primary = MultiKeyProvider("primary-embed", FakeProvider, ["good-primary"])
    secondary = MultiKeyProvider("secondary-embed", FakeProvider, ["good-secondary"])
    provider = ProviderFallback([primary, secondary])

    result = await provider.embed("hello")

    assert result == [1.0]
    assert FakeProvider.calls == ["embed:good-primary"]


@pytest.mark.asyncio
async def test_multikey_reserves_distinct_start_indices_for_concurrent_calls():
    provider = MultiKeyProvider("slow", SlowProvider, ["k1", "k2", "k3"])
    SlowProvider.release = asyncio.Event()

    tasks = [
        asyncio.create_task(provider.chat([ChatMessage(role="user", content="hi")]))
        for _ in range(2)
    ]
    for _ in range(100):
        if len(SlowProvider.entered) >= 2:
            break
        await asyncio.sleep(0.01)
    assert SlowProvider.entered == ["k1", "k2"]
    SlowProvider.release.set()
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# ExhaustedError fallback — primary exhausted → secondary used
# ---------------------------------------------------------------------------


class ExhaustingProvider:
    """Provider that always raises ExhaustedError."""

    calls: list[str] = []

    @classmethod
    def clear_calls(cls) -> None:
        cls.calls.clear()

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(
        self, messages, *, heavy: bool = False, task_type: str = "default"
    ) -> str:
        self.calls.append(self.api_key)
        raise ExhaustedError(f"All keys for {self.api_key} exhausted")

    async def close(self) -> None:
        pass


class GoodProvider:
    """Provider that always succeeds."""

    calls: list[str] = []

    @classmethod
    def clear_calls(cls) -> None:
        cls.calls.clear()

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(
        self, messages, *, heavy: bool = False, task_type: str = "default"
    ) -> str:
        self.calls.append(self.api_key)
        return f"ok:{self.api_key}"

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_provider_fallback_exhausted_error():
    """When primary provider raises ExhaustedError, ProviderFallback switches to secondary."""
    ExhaustingProvider.calls = []
    GoodProvider.calls = []

    primary = MultiKeyProvider("exhaust-prov", ExhaustingProvider, ["exhaust-key-1"])
    secondary = MultiKeyProvider("good-prov", GoodProvider, ["good-key-1"])
    fallback = ProviderFallback([primary, secondary])

    result = await fallback.chat([ChatMessage(role="user", content="hi")])

    assert result == "ok:good-key-1"
    assert len(ExhaustingProvider.calls) == 1
    assert ExhaustingProvider.calls == ["exhaust-key-1"]
    assert len(GoodProvider.calls) == 1
    assert GoodProvider.calls == ["good-key-1"]


@pytest.mark.asyncio
async def test_cooldown_until_handles_naive_datetime():
    """_ensure_utc нормализует naive datetime как UTC-aware — без TypeError."""
    from src.llm.router import _ensure_utc
    from datetime import datetime, timezone

    # naive datetime (как из старых записей SQLite)
    naive = datetime(2025, 1, 15, 12, 0, 0)
    result = _ensure_utc(naive)
    assert result is not None
    assert result.tzinfo is not None
    assert result == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    # Сравнение с aware не должно падать
    assert result < datetime.now(timezone.utc)

    # aware datetime (как из новых записей)
    aware = datetime.now(timezone.utc)
    result2 = _ensure_utc(aware)
    assert result2 is aware  # возвращается без изменений

    # None
    assert _ensure_utc(None) is None


# =============================================================================
# Circuit Breaker tests
# =============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_starts_closed():
    """CircuitBreaker начинает в CLOSED и ready."""
    from src.llm.router import _KeyCircuitBreaker, _CircuitState

    cb = _KeyCircuitBreaker(failure_threshold=3, base_timeout=90.0)
    now = 1000.0
    assert cb.state == _CircuitState.CLOSED
    assert cb.is_ready(now) is True


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    """После failure_threshold ошибок подряд → OPEN."""
    from src.llm.router import _KeyCircuitBreaker, _CircuitState

    cb = _KeyCircuitBreaker(failure_threshold=2, base_timeout=10.0)
    now = 1000.0
    assert cb.is_ready(now) is True
    cb.record_failure(now)  # 1-я ошибка
    assert cb.is_ready(now)  # ещё CLOSED
    cb.record_failure(now)  # 2-я → OPEN
    assert cb.state == _CircuitState.OPEN
    assert cb.is_ready(now) is False


@pytest.mark.asyncio
async def test_circuit_breaker_exponential_backoff():
    """Экспоненциальный backoff: base*2^tripped, кап 3600."""
    from src.llm.router import _KeyCircuitBreaker

    cb = _KeyCircuitBreaker(failure_threshold=1, base_timeout=10.0)
    now = 1000.0
    cb.record_failure(now)  # → OPEN, tripped=0, timeout=10
    assert cb.ready_at(now) == 1010.0
    # Ещё ошибка в HALF_OPEN: снова OPEN, tripped=1, timeout=20
    cb.try_half_open(1015.0)  # → HALF_OPEN
    cb.record_failure(1015.0)  # → OPEN, tripped=1
    assert cb.ready_at(1015.0) == 1035.0
    # Ещё: tripped=2, timeout=40
    cb.try_half_open(1040.0)
    cb.record_failure(1040.0)
    assert cb.ready_at(1040.0) == 1080.0


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_success():
    """HALF_OPEN → success → CLOSED."""
    from src.llm.router import _KeyCircuitBreaker, _CircuitState

    cb = _KeyCircuitBreaker(failure_threshold=1, base_timeout=10.0)
    now = 1000.0
    cb.record_failure(now)  # → OPEN
    assert cb.is_ready(now) is False
    cb.try_half_open(1010.0)  # → HALF_OPEN
    assert cb.is_ready(1010.0) is True
    assert cb.state == _CircuitState.HALF_OPEN
    cb.record_success()  # → CLOSED
    assert cb.state == _CircuitState.CLOSED
    assert cb.is_ready(1015.0) is True


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    """HALF_OPEN → fail → OPEN снова."""
    from src.llm.router import _KeyCircuitBreaker, _CircuitState

    cb = _KeyCircuitBreaker(failure_threshold=1, base_timeout=10.0)
    cb.record_failure(1000.0)  # → OPEN, tripped=0
    cb.try_half_open(1010.0)  # → HALF_OPEN
    assert cb.state == _CircuitState.HALF_OPEN
    cb.record_failure(1010.0)  # → OPEN, tripped=1
    assert cb.state == _CircuitState.OPEN
    # timeout = 10 * 2^1 = 20 (от last_failure_time=1010)
    assert cb.ready_at(1010.0) == 1030.0


@pytest.mark.asyncio
async def test_circuit_breaker_single_failure_does_not_open():
    """Одна ошибка при threshold=3 не открывает цепь."""
    from src.llm.router import _KeyCircuitBreaker, _CircuitState

    cb = _KeyCircuitBreaker(failure_threshold=3, base_timeout=90.0)
    now = 1000.0
    cb.record_failure(now)
    assert cb.state == _CircuitState.CLOSED
    assert cb.is_ready(1001.0) is True


@pytest.mark.asyncio
async def test_circuit_breaker_max_timeout_cap():
    """Таймаут не превышает 3600 секунд."""
    from src.llm.router import _KeyCircuitBreaker

    cb = _KeyCircuitBreaker(failure_threshold=1, base_timeout=60.0)
    now = 1000.0
    # Первая ошибка: CLOSED → OPEN, tripped=0
    cb.record_failure(now)
    # 6 ре-трипов через HALF_OPEN → OPEN: tripped растёт 1..6
    for i in range(6):
        ready = cb.ready_at(now)
        cb.try_half_open(ready)  # → HALF_OPEN
        now = ready
        cb.record_failure(now)  # → OPEN, tripped++
    # 60 * 2^6 = 3840 > 3600 → capped at 3600
    assert cb.ready_at(now) == now + 3600


# =============================================================================
# Adaptive Provider Selection tests
# =============================================================================


@pytest.mark.asyncio
async def test_provider_metrics_initial_state():
    """Новый _ProviderMetrics: success_rate=1.0, score=1.0 (exploration bias)."""
    from src.llm.router import _ProviderMetrics

    m = _ProviderMetrics()
    assert m.success_rate == 1.0
    assert m.avg_latency == 0.0
    # 0.6*1.0 (sr) + 0.4*0.5 (latency, no data=0.5) = 0.8
    assert m.score(1000.0) == pytest.approx(0.8, rel=1e-3)


@pytest.mark.asyncio
async def test_provider_metrics_score_formula():
    """Score = 0.6*success_rate + 0.4*latency_score, c recency penalty."""
    from src.llm.router import _ProviderMetrics

    m = _ProviderMetrics()
    # 100% success, avg 1s latency, no recent failure → score ≈ 0.96
    m.success_count = 10
    m.call_count = 10
    m.total_latency = 10.0  # avg 1.0
    score = m.score(2000.0)
    # 0.6*1.0 + 0.4*(1-1/10) = 0.6 + 0.36 = 0.96
    assert score == pytest.approx(0.96, rel=1e-3)

    # 0% success → score = 0.0 + 0.4*0.5*1.0 = 0.2
    m2 = _ProviderMetrics()
    m2.failure_count = 5
    m2.last_failure_time = 1900.0  # >60s ago → no penalty
    assert m2.score(2000.0) == pytest.approx(0.2, rel=1e-3)

    # Recent failure (<60s) → penalty 0.3
    m2.last_failure_time = 1980.0  # 20s ago
    recent_score = m2.score(2000.0)
    # 0.2 * 0.3 = 0.06
    assert recent_score == pytest.approx(0.06, rel=1e-3)


@pytest.mark.asyncio
async def test_provider_metrics_latency_normalization():
    """Латентность нормализуется: 0s → 1.0, >=10s → 0.0."""
    from src.llm.router import _ProviderMetrics

    # Very fast provider
    fast = _ProviderMetrics()
    fast.success_count = 5
    fast.call_count = 5
    fast.total_latency = 0.5  # avg 0.1s
    # sr=1.0*0.6 + (1-0.1/10=0.99)*0.4 = 0.6+0.396 = 0.996
    assert fast.score(1000.0) == pytest.approx(0.996, rel=1e-3)

    # Slow provider (10s+)
    slow = _ProviderMetrics()
    slow.success_count = 5
    slow.call_count = 5
    slow.total_latency = 100.0  # avg 20s
    # sr=1.0*0.6 + max(0,1-20/10)=0 *0.4 = 0.6
    assert slow.score(1000.0) == pytest.approx(0.6, rel=1e-3)


@pytest.mark.asyncio
async def test_record_provider_success_failure():
    """_record_provider_success и _record_provider_failure обновляют метрики."""
    from src.llm.router import (
        _record_provider_success,
        _record_provider_failure,
        _PROVIDER_METRICS,
    )

    _PROVIDER_METRICS.clear()

    await _record_provider_success("aps-test-svc", 1.5)
    metrics = _PROVIDER_METRICS["aps-test-svc"]
    assert metrics.success_count == 1
    assert metrics.call_count == 1
    assert metrics.total_latency == 1.5
    assert metrics.failure_count == 0
    assert metrics.last_success_time > 0

    await _record_provider_failure("aps-test-svc")
    assert metrics.failure_count == 1
    assert metrics.success_count == 1
    assert metrics.last_failure_time > 0

    # Новый провайдер без истории
    _PROVIDER_METRICS.clear()
    await _record_provider_failure("aps-unknown")
    assert _PROVIDER_METRICS["aps-unknown"].failure_count == 1


@pytest.mark.asyncio
async def test_score_provider_unknown_returns_one():
    """Неизвестный провайдер получает score=1.0 (exploration)."""
    from src.llm.router import _score_provider, _PROVIDER_METRICS

    _PROVIDER_METRICS.clear()
    assert _score_provider("nonexistent", 1000.0) == 1.0


@pytest.mark.asyncio
async def test_adaptive_sorting_failing_provider_last():
    """Провайдер с историей отказов пробуется последним."""
    from src.llm.router import (
        MultiKeyProvider,
        ProviderFallback,
        _PROVIDER_METRICS,
        _record_provider_failure,
        _record_provider_success,
    )
    from src.llm.base import ChatMessage

    FakeProvider.calls = []
    _PROVIDER_METRICS.clear()

    # Два провайдера: good-reliable всегда успешен, bad-failing всегда падает
    good = MultiKeyProvider("aps-good", FakeProvider, ["aps-good-key"])
    bad = MultiKeyProvider("aps-bad", FakeProvider, ["aps-bad-key"])

    # Seed: bad — 4 failures, good — 4 successes
    for _ in range(4):
        await _record_provider_failure("aps-bad")
        await _record_provider_success("aps-good", 0.5)

    # Sort: хотя good идёт вторым в списке, он должен быть выбран первым
    fallback = ProviderFallback([bad, good])
    FakeProvider.calls = []
    result = await fallback.chat([ChatMessage(role="user", content="hi")])

    # Результат от good
    assert result == "ok:aps-good-key"
    # Убедимся, что good был вызван, а bad нет
    assert "aps-bad-key" not in FakeProvider.calls
    assert "aps-good-key" in FakeProvider.calls


@pytest.mark.asyncio
async def test_adaptive_sorting_all_providers_succeed_first():
    """При одинаковых score порядок сохраняется (stable sort)."""
    from src.llm.router import (
        MultiKeyProvider,
        ProviderFallback,
        _PROVIDER_METRICS,
    )
    from src.llm.base import ChatMessage

    FakeProvider.calls = []
    _PROVIDER_METRICS.clear()

    # Все провайдеры с score=1.0 (нет истории) — stable sort сохраняет порядок
    first = MultiKeyProvider("aps-first", FakeProvider, ["aps-1st"])
    second = MultiKeyProvider("aps-second", FakeProvider, ["aps-2nd"])

    fallback = ProviderFallback([first, second])
    FakeProvider.calls = []
    result = await fallback.chat([ChatMessage(role="user", content="hi")])

    # Первый в списке — первый вызван
    assert result == "ok:aps-1st"
    assert FakeProvider.calls == ["aps-1st"]


@pytest.mark.asyncio
async def test_adaptive_sorting_embed_unchanged():
    """embed/embed_batch НЕ сортируются — остаются на primary."""
    from src.llm.router import (
        MultiKeyProvider,
        ProviderFallback,
        _PROVIDER_METRICS,
        _record_provider_failure,
    )

    FakeProvider.calls = []
    _PROVIDER_METRICS.clear()

    # Failing primary, good secondary
    primary = MultiKeyProvider("aps-embed-bad", FakeProvider, ["aps-embed-bad-key"])
    secondary = MultiKeyProvider("aps-embed-good", FakeProvider, ["aps-embed-good"])
    for _ in range(4):
        await _record_provider_failure("aps-embed-bad")

    fallback = ProviderFallback([primary, secondary])
    # embed: всегда на primary, даже если он плохой
    result = await fallback.embed("hello")
    assert result == [1.0]
    assert FakeProvider.calls == ["embed:aps-embed-bad-key"]
