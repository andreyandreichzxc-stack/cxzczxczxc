import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import enum

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.timeutil import ensure_utc as _ensure_utc


from src.crypto import decrypt
from src.db.models import User
from src.db.repo import get_active_keys, get_api_keys, mark_key_failure, mark_key_used
from src.db.session import get_session
from collections.abc import AsyncGenerator

from src.llm.base import ChatMessage, LLMProvider, TaskType
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.cloudflare_provider import CloudflareProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.mistral_provider import MistralProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.openrouter_provider import OpenRouterProvider
from src.llm.deepseek_provider import DeepSeekProvider

logger = logging.getLogger(__name__)

# Sentinel to distinguish "heavy not passed" from "heavy=False".
# Used so that `_default_heavy` (from user's use_heavy_model setting)
# is respected when callers don't explicitly specify heavy/light.
_UNSET = object()


class ExhaustedError(Exception):
    """Все API-ключи провайдера исчерпаны (колдаун/отключены)."""

    pass


class _CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _KeyCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, base_timeout: float = 90.0) -> None:
        self._failure_count = 0
        self._tripped_count = 0
        self._state = _CircuitState.CLOSED
        self._last_failure_time = 0.0
        self._base_timeout = base_timeout
        self._failure_threshold = failure_threshold

    @property
    def state(self) -> _CircuitState:
        return self._state

    def ready_at(self, now: float) -> float:
        """Возвращает монотонное время, когда ключ снова готов."""
        if self._state != _CircuitState.OPEN:
            return now
        timeout = self._base_timeout * (2**self._tripped_count)
        return self._last_failure_time + min(timeout, 3600.0)

    def is_ready(self, now: float) -> bool:
        if self._state == _CircuitState.CLOSED:
            return True
        if self._state == _CircuitState.HALF_OPEN:
            return True
        # OPEN always returns False — only try_half_open() gates recovery probes,
        # ensuring single-probe + exponential backoff on re-trip.
        return False

    def record_success(self) -> None:
        self._failure_count = 0
        self._tripped_count = 0
        self._state = _CircuitState.CLOSED

    def record_failure(self, now: float) -> None:
        self._failure_count += 1
        self._last_failure_time = now
        if self._state == _CircuitState.HALF_OPEN:
            self._state = _CircuitState.OPEN
            self._tripped_count += 1
        elif (
            self._state == _CircuitState.CLOSED
            and self._failure_count >= self._failure_threshold
        ):
            self._state = _CircuitState.OPEN
            # NOTE: _tripped_count НЕ инкрементится при первом открытии —
            # он растёт только при re-trip'ах (HALF_OPEN → OPEN),
            # чтобы экспоненциальный backoff считался с base*2^0 = base.

    def try_half_open(self, now: float) -> bool:
        """Переводит в HALF_OPEN если пришло время пробовать."""
        if self._state != _CircuitState.OPEN:
            return False
        if now >= self.ready_at(now):
            self._state = _CircuitState.HALF_OPEN
            return True
        return False


# ─── Adaptive Provider Selection ─────────────────────────────────────


@dataclass
class _ProviderMetrics:
    """Per-provider performance metrics for adaptive selection.

    Хранит историю успехов/неудач и среднюю латентность для каждого
    LLM-провайдера (openai, gemini, mistral, ...). Используется в
    ProviderFallback.chat() для сортировки провайдеров — наиболее
    надёжный и быстрый пробуется первым.
    """

    success_count: int = 0
    failure_count: int = 0
    total_latency: float = 0.0
    call_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0  # неизвестный = оптимистичный (exploration bias)
        return self.success_count / total

    @property
    def avg_latency(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency / self.call_count

    def score(self, now: float) -> float:
        """Composite score 0..1. Higher = provider to try first.

        Формула: 60% успешность + 40% латентность, с штрафом за
        недавние (последние 60s) ошибки.
        """
        sr = self.success_rate
        lat = self.avg_latency
        # Normalize latency: 0s → 1.0, >=10s → 0.0
        lat_score = max(0.0, 1.0 - lat / 10.0) if self.call_count > 0 else 0.5
        # Recent failure penalty
        if self.last_failure_time > 0 and now - self.last_failure_time < 60.0:
            recency_penalty = 0.3
        else:
            recency_penalty = 1.0
        return (sr * 0.6 + lat_score * 0.4) * recency_penalty


_PROVIDER_METRICS: dict[str, _ProviderMetrics] = {}
_PROVIDER_METRICS_LOCK: asyncio.Lock | None = (
    None  # lazy-init on first use (Python 3.10+ loop safety)
)


async def _record_provider_success(name: str, latency: float) -> None:
    """Записывает успешный вызов провайдера с замеренной латентностью."""
    global _PROVIDER_METRICS_LOCK
    if _PROVIDER_METRICS_LOCK is None:
        _PROVIDER_METRICS_LOCK = asyncio.Lock()
    now = asyncio.get_running_loop().time()
    async with _PROVIDER_METRICS_LOCK:
        metrics = _PROVIDER_METRICS.get(name)
        if metrics is None:
            metrics = _ProviderMetrics()
            _PROVIDER_METRICS[name] = metrics
        metrics.success_count += 1
        metrics.call_count += 1
        metrics.total_latency += latency
        metrics.last_success_time = now


async def _record_provider_failure(name: str) -> None:
    """Записывает неудачный вызов провайдера."""
    global _PROVIDER_METRICS_LOCK
    if _PROVIDER_METRICS_LOCK is None:
        _PROVIDER_METRICS_LOCK = asyncio.Lock()
    now = asyncio.get_running_loop().time()
    async with _PROVIDER_METRICS_LOCK:
        metrics = _PROVIDER_METRICS.get(name)
        if metrics is None:
            metrics = _ProviderMetrics()
            _PROVIDER_METRICS[name] = metrics
        metrics.failure_count += 1
        metrics.last_failure_time = now


def _score_provider(name: str, now: float) -> float:
    """Public score lookup. 1.0 для провайдеров без истории (exploration)."""
    metrics = _PROVIDER_METRICS.get(name)
    if metrics is None:
        return 1.0
    return metrics.score(now)


PROVIDER_ORDER = ("deepseek", "openrouter", "openai", "gemini", "mistral", "cloudflare")
RETRYABLE_MARKERS = (
    "429",
    "500",
    "503",
    "capacity",
    "capacity exceeded",
    "service_tier_capacity_exceeded",
    "rate limit",
    "ratelimit",
    "resource_exhausted",
    "quota",
    "overloaded",
    "temporarily unavailable",
    "raw_status_code': 429",
    'raw_status_code": 429',
    # Cloudflare Workers AI async-модели (cold start, async queue)
    "async queue",
    "queued",
    "model is busy",
    "cold start",
    "workers ai",
    "cf-ray",
)
KEY_COOLDOWN_SECONDS = 90.0
MAX_RETRIES_PER_KEY = 3
RETRY_BASE_DELAY = 1.0  # seconds
_CIRCUIT_BREAKERS: dict[tuple[str, str], _KeyCircuitBreaker] = {}
_CIRCUIT_BREAKERS_LOCK: asyncio.Lock | None = (
    None  # lazy-init on first use (Python 3.10+ loop safety)
)

# Per-purpose лимиты параллельных запросов
_PURPOSE_SEMAPHORES: dict[str, asyncio.Semaphore] | None = (
    None  # lazy-init on first use
)


async def acquire_purpose_slot(purpose: str) -> asyncio.Semaphore:
    """Захватывает слот для purpose. Возвращает семафор."""
    global _PURPOSE_SEMAPHORES
    if _PURPOSE_SEMAPHORES is None:
        _PURPOSE_SEMAPHORES = {
            "main": asyncio.Semaphore(2),
            "draft": asyncio.Semaphore(1),
            "memory": asyncio.Semaphore(1),
            "background": asyncio.Semaphore(3),
            "analysis": asyncio.Semaphore(1),
            "urgent": asyncio.Semaphore(2),
            "fallback": asyncio.Semaphore(2),
        }
    sem = _PURPOSE_SEMAPHORES.get(purpose)
    if sem is None:
        sem = _PURPOSE_SEMAPHORES.get("fallback", asyncio.Semaphore(1))
    await sem.acquire()
    return sem


def release_purpose_slot(sem: asyncio.Semaphore) -> None:
    """Освобождает слот."""
    sem.release()


def _is_retryable_llm_error(exc: Exception) -> bool:
    """True for transient capacity/rate-limit/server errors worth trying another key/provider."""
    # Timeouts are always retryable — rotate key / fallback to next provider.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    # httpx timeout exceptions (ReadTimeout, ConnectTimeout, etc.)
    if type(exc).__name__ in (
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
    ):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in {429, 500, 503}:
        return True
    code = str(getattr(exc, "code", "") or "").lower()
    if code in {"429", "500", "503", "3505", "service_tier_capacity_exceeded"}:
        return True
    # NOTE: body намеренно не включено — полный ответ LLM может содержать
    # чувствительные данные пользователя (PII, секреты, содержимое диалога).
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}…{key[-4:]}"


async def _restore_cooldowns(slot_ids: list[int]) -> None:
    """Восстанавливает circuit breaker'ы для ключей в кулдауне после рестарта.

    После перезапуска in-memory _KeyCircuitBreaker объекты теряются.
    DB-поле cooldown_until переживает рестарт — используем его для восстановления
    OPEN-состояния и экспоненциального backoff'а.

    Принимает slot_ids от одного или нескольких провайдеров — запрашивает
    все за один проход (единая DB-сессия).
    """
    if not slot_ids:
        return

    global _CIRCUIT_BREAKERS_LOCK
    if _CIRCUIT_BREAKERS_LOCK is None:
        _CIRCUIT_BREAKERS_LOCK = asyncio.Lock()

    try:
        from sqlalchemy import select
        from src.db.models import LlmKeySlot

        async with get_session() as session:
            now_utc = datetime.now(timezone.utc)

            # Запрашиваем конкретные слоты с активным кулдауном на уровне SQL
            # (DateTime(timezone=True) гарантирует корректное сравнение для новых записей)
            q = select(LlmKeySlot).where(
                LlmKeySlot.id.in_(slot_ids),
                LlmKeySlot.cooldown_until.is_not(None),
                LlmKeySlot.cooldown_until > now_utc,
            )
            r = await session.execute(q)
            all_candidates = list(r.scalars().all())

            # Safety net: Python-фильтрация для legacy наивных дат,
            # которые SQL-уровень может пропустить/недопустить при строковом сравнении
            cooldown_slots: list[LlmKeySlot] = []
            for slot in all_candidates:
                if (
                    slot.cooldown_until is not None
                    and slot.cooldown_until.tzinfo is None
                ):
                    logger.debug(
                        "Legacy naive datetime in cooldown_until for slot %d (provider=%s)",
                        slot.id,
                        slot.provider,
                    )
                cu = _ensure_utc(slot.cooldown_until)
                if cu is not None and cu > now_utc:
                    cooldown_slots.append(slot)

            if not cooldown_slots:
                return

            now_mono = asyncio.get_running_loop().time()
            restored_by_provider: dict[str, int] = {}

            async with _CIRCUIT_BREAKERS_LOCK:
                for slot in cooldown_slots:
                    cu = _ensure_utc(slot.cooldown_until)
                    if cu is None:
                        continue
                    cache_key = (slot.provider, str(slot.id))
                    if cache_key in _CIRCUIT_BREAKERS:
                        continue  # уже восстановлен (повторный вызов build_provider)

                    remaining = (cu - now_utc).total_seconds()
                    if remaining <= 0:
                        continue

                    cb = _KeyCircuitBreaker(
                        failure_threshold=3,
                        base_timeout=KEY_COOLDOWN_SECONDS,
                    )

                    # Подбираем _tripped_count: наименьшее значение,
                    # при котором backoff >= оставшегося времени кулдауна.
                    # Экспонента: base * 2^0 = 90s, 2^1 = 180s, 2^2 = 360s, …
                    tripped = 0
                    while (
                        KEY_COOLDOWN_SECONDS * (2**tripped) < remaining and tripped < 10
                    ):
                        tripped += 1

                    cb._state = _CircuitState.OPEN
                    cb._failure_count = cb._failure_threshold
                    cb._tripped_count = tripped
                    cb._last_failure_time = max(
                        0.0,
                        now_mono - (KEY_COOLDOWN_SECONDS * (2**tripped) - remaining),
                    )

                    _CIRCUIT_BREAKERS[cache_key] = cb
                    restored_by_provider[slot.provider] = (
                        restored_by_provider.get(slot.provider, 0) + 1
                    )

            for provider_name, count in restored_by_provider.items():
                logger.info(
                    "Restored %d circuit breaker(s) for %s from DB cooldown",
                    count,
                    provider_name,
                )
    except Exception:
        logger.exception("Failed to restore cooldowns from DB")


# ─── MultiKey: обёртка для ротации ключей ─────────────────────────────


class MultiKeyProvider:
    """Обёртка: ротирует ключи провайдера при ошибке 429/503/500.

    Позволяет указать несколько API-ключей для одного LLM-провайдера.
    Round-robin распределяет параллельные вызовы по ключам,
    Semaphore(N) ограничивает число одновременных запросов.
    При получении ошибки пропускной способности (rate limit, capacity exceeded)
    автоматически переключается на следующий ключ.
    """

    def __init__(
        self,
        provider_name: str,
        provider_class: type,
        keys: list[str],
        slot_ids: list[int] | None = None,
        endpoints: list[str | None] | None = None,
        models: list[str | None] | None = None,
        session_provider: Callable[[], tuple[AsyncSession, object]] | None = None,
        purpose: str = "main",
        **kwargs: object,
    ) -> None:
        if not keys:
            raise ValueError("MultiKeyProvider requires at least one key")
        self.provider_name = provider_name
        self._provider_class = provider_class
        self._keys = keys
        self._slot_ids = slot_ids or []
        self._endpoints = endpoints or []
        self._models = models or []
        self._session_provider = session_provider
        self._kwargs = kwargs
        self._idx = 0
        self._idx_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(len(self._keys))
        self._current_purpose = purpose
        self._model: str | None = None  # global override; None = use per-slot model
        self._default_heavy: bool = False  # overridden by use_heavy_model setting
        self.name = f"{provider_name}(×{len(self._keys)})"

    async def _reserve_start_idx(self) -> int:
        async with self._idx_lock:
            start_idx = self._idx
            self._idx = (self._idx + 1) % len(self._keys)
            return start_idx

    async def _advance_idx_after_success(self, idx: int) -> None:
        async with self._idx_lock:
            self._idx = (idx + 1) % len(self._keys)

    async def _try_with_retry(
        self,
        operation,
        *args: object,
        model_override: str | None = None,
        **kwargs: object,
    ):
        """Пробует операцию со всеми ключами по очереди.

        Пропускает ключи, которые фейлились менее 60 секунд назад.
        При успехе обновляет активный индекс и отмечает слот (DB).
        При временной ошибке помечает слот как упавший (DB cooldown).
        Записывает метрики для Adaptive Provider Selection.
        """
        global _CIRCUIT_BREAKERS_LOCK
        if _CIRCUIT_BREAKERS_LOCK is None:
            _CIRCUIT_BREAKERS_LOCK = asyncio.Lock()
        start_time = asyncio.get_running_loop().time()
        last_error: Exception | None = None
        now = start_time

        # Round-robin: reserve a unique start index for concurrent calls.
        start_idx = await self._reserve_start_idx()

        skipped = 0
        for attempt in range(len(self._keys)):
            idx = (start_idx + attempt) % len(self._keys)
            key = self._keys[idx]
            cache_key = (
                (self.provider_name, str(self._slot_ids[idx]))
                if self._slot_ids and idx < len(self._slot_ids)
                else (self.provider_name, key)
            )
            async with _CIRCUIT_BREAKERS_LOCK:
                cb = _CIRCUIT_BREAKERS.get(cache_key)
            if cb is not None and not cb.is_ready(now):
                _ = cb.try_half_open(now)  # проверяем, не пора ли попробовать
                if not cb.is_ready(now):
                    skipped += 1
                    continue
            # Create provider instance — handle creation failure separately
            try:
                endpoint = (
                    self._endpoints[idx]
                    if self._endpoints and idx < len(self._endpoints)
                    else None
                )
                provider_kwargs = dict(self._kwargs)
                if endpoint:
                    provider_kwargs["base_url"] = endpoint
                if model_override:
                    provider_kwargs["model"] = model_override
                elif self._model:
                    provider_kwargs["model"] = self._model
                elif self._models and idx < len(self._models):
                    provider_kwargs["model"] = self._models[idx]
                provider = self._provider_class(key, **provider_kwargs)
            except Exception as exc:
                last_error = exc
                continue

            try:
                for retry in range(MAX_RETRIES_PER_KEY):
                    try:
                        result = await asyncio.wait_for(
                            operation(provider, *args, **kwargs), timeout=90.0
                        )
                    except Exception as exc:
                        if not _is_retryable_llm_error(exc):
                            raise
                        if retry < MAX_RETRIES_PER_KEY - 1:
                            delay = RETRY_BASE_DELAY * (2**retry)
                            logger.warning(
                                "LLM %s key %s attempt %d/%d failed, retrying in %.1fs: %s",
                                self.provider_name,
                                _mask_key(key),
                                retry + 1,
                                MAX_RETRIES_PER_KEY,
                                delay,
                                str(exc)[:200],
                            )
                            await asyncio.sleep(delay)
                        else:
                            raise
                    else:
                        async with _CIRCUIT_BREAKERS_LOCK:
                            cb = _CIRCUIT_BREAKERS.get(cache_key)
                            if cb:
                                cb.record_success()
                                if cb.state == _CircuitState.CLOSED:
                                    _CIRCUIT_BREAKERS.pop(cache_key, None)
                        # DB: отметить успешное использование (fresh session)
                        if self._slot_ids:
                            try:
                                async with get_session() as fresh_s:
                                    await mark_key_used(fresh_s, self._slot_ids[idx])
                            except Exception:
                                logger.exception(
                                    "Failed to mark key slot %d as used",
                                    self._slot_ids[idx],
                                )
                        # Adaptive Provider Selection: запись метрик успеха
                        # (try/except — потеря result при ошибке метрики дороже, чем сама метрика)
                        latency = asyncio.get_running_loop().time() - start_time
                        try:
                            await _record_provider_success(self.provider_name, latency)
                        except Exception:
                            logger.exception(
                                "Failed to record provider success metric for %s",
                                self.provider_name,
                            )
                        # Round-robin: advance to the next key for load distribution
                        await self._advance_idx_after_success(idx)
                        return result
            except Exception as exc:
                if _is_retryable_llm_error(exc):
                    async with _CIRCUIT_BREAKERS_LOCK:
                        if cache_key not in _CIRCUIT_BREAKERS:
                            _CIRCUIT_BREAKERS[cache_key] = _KeyCircuitBreaker()
                        _CIRCUIT_BREAKERS[cache_key].record_failure(now)
                    last_error = exc
                    logger.warning(
                        "LLM %s key %s temporarily failed, rotating: %s",
                        self.provider_name,
                        _mask_key(key),
                        str(exc)[:200],
                    )
                    # DB: отметить падение слота (fresh session)
                    if self._slot_ids:
                        try:
                            async with get_session() as fresh_s:
                                error_msg = (
                                    f"{type(exc).__name__}: {str(exc).split(chr(10))[0]}"
                                )[:256]
                                await mark_key_failure(
                                    fresh_s, self._slot_ids[idx], error_msg
                                )
                        except Exception:
                            logger.exception(
                                "Failed to mark key slot %d as failed",
                                self._slot_ids[idx],
                            )
                    continue
                raise
            finally:
                try:
                    await provider.close()
                except Exception:
                    pass  # close failures should never mask the actual result
        if last_error:
            try:
                await _record_provider_failure(self.provider_name)
            except Exception:
                logger.exception(
                    "Failed to record provider failure metric for %s",
                    self.provider_name,
                )
            raise ExhaustedError(
                f"Все {len(self._keys)} ключей {self.provider_name} недоступны "
                f"(последняя ошибка: {last_error})"
            )
        try:
            await _record_provider_failure(self.provider_name)
        except Exception:
            logger.exception(
                "Failed to record provider failure metric for %s",
                self.provider_name,
            )
        raise ExhaustedError(
            f"Все {len(self._keys)} ключей {self.provider_name} в кулдауне"
        )

    async def chat(
        self, messages, *, heavy=_UNSET, task_type: str = TaskType.DEFAULT
    ) -> str:
        sem = await acquire_purpose_slot(self._current_purpose)
        try:
            return await self._chat_with_retry(
                messages, heavy=heavy, task_type=task_type
            )
        finally:
            release_purpose_slot(sem)

    async def _chat_with_retry(
        self, messages, *, heavy=_UNSET, task_type: str = TaskType.DEFAULT
    ) -> str:
        await self._semaphore.acquire()
        try:
            return await self._retry_inner(messages, heavy=heavy, task_type=task_type)
        finally:
            self._semaphore.release()

    async def _retry_inner(
        self, messages, *, heavy=_UNSET, task_type: str = TaskType.DEFAULT
    ) -> str:
        """Core retry logic WITHOUT semaphore acquisition.

        Both chat_stream (which already holds the semaphore) and
        _chat_with_retry (which acquires it) call this.
        """
        # Resolve heavy: explicit True/False wins; if not passed, use _default_heavy
        # (set from user's use_heavy_model setting by build_provider).
        effective_heavy = self._default_heavy if heavy is _UNSET else heavy
        model_override = self._resolve_model_for_task(task_type)
        return await self._try_with_retry(
            lambda p: p.chat(messages, heavy=effective_heavy),
            model_override=model_override,
        )

    def _resolve_model_for_task(self, task_type: str) -> str | None:
        """Resolve model for task type.

        Returns model name or None to use provider default.
        Priority: _model (set by build_provider from task overrides) > None
        """
        if self._model:
            return self._model
        return None

    async def chat_stream(
        self, messages, *, heavy=_UNSET, task_type: str = TaskType.DEFAULT
    ) -> AsyncGenerator[str, None]:
        """Stream chat output token by token with key rotation.
        Falls back to regular chat() if no provider supports streaming."""
        # Resolve heavy: explicit True/False wins; if not passed, use _default_heavy
        effective_heavy = self._default_heavy if heavy is _UNSET else heavy
        global _CIRCUIT_BREAKERS_LOCK
        if _CIRCUIT_BREAKERS_LOCK is None:
            _CIRCUIT_BREAKERS_LOCK = asyncio.Lock()
        model_override = self._resolve_model_for_task(task_type)
        sem = await acquire_purpose_slot(self._current_purpose)
        try:
            await self._semaphore.acquire()
            try:
                start_time = asyncio.get_running_loop().time()
                start_idx = await self._reserve_start_idx()
                last_error: Exception | None = None
                for attempt in range(len(self._keys)):
                    idx = (start_idx + attempt) % len(self._keys)
                    key = self._keys[idx]
                    cache_key = (
                        (self.provider_name, str(self._slot_ids[idx]))
                        if self._slot_ids and idx < len(self._slot_ids)
                        else (self.provider_name, key)
                    )
                    # Circuit breaker check — skip keys in cooldown
                    async with _CIRCUIT_BREAKERS_LOCK:
                        cb = _CIRCUIT_BREAKERS.get(cache_key)
                    if cb is not None:
                        now = asyncio.get_running_loop().time()
                        _ = cb.try_half_open(now)
                        if not cb.is_ready(now):
                            continue
                    endpoint = (
                        self._endpoints[idx]
                        if self._endpoints and idx < len(self._endpoints)
                        else None
                    )
                    provider_kwargs = dict(self._kwargs)
                    if endpoint:
                        provider_kwargs["base_url"] = endpoint
                    if model_override:
                        provider_kwargs["model"] = model_override
                    elif self._model:
                        provider_kwargs["model"] = self._model
                    elif self._models and idx < len(self._models):
                        provider_kwargs["model"] = self._models[idx]
                    provider = self._provider_class(key, **provider_kwargs)
                    try:
                        total_text = ""
                        # 180s overall timeout; httpx 60s socket-level timeout per read
                        async with asyncio.timeout(180):
                            async for token in provider.chat_stream(
                                messages, heavy=effective_heavy
                            ):
                                total_text += token
                                yield token
                        # Stream completed successfully — record metrics
                        # Circuit breaker: record success
                        async with _CIRCUIT_BREAKERS_LOCK:
                            cb = _CIRCUIT_BREAKERS.get(cache_key)
                            if cb:
                                cb.record_success()
                                if cb.state == _CircuitState.CLOSED:
                                    _CIRCUIT_BREAKERS.pop(cache_key, None)
                        # DB: mark key as used (fresh session)
                        if self._slot_ids:
                            try:
                                async with get_session() as fresh_s:
                                    await mark_key_used(fresh_s, self._slot_ids[idx])
                            except Exception:
                                logger.exception(
                                    "Failed to mark key slot %d as used",
                                    self._slot_ids[idx],
                                )
                        # Adaptive Provider Selection: record success metrics
                        latency = asyncio.get_running_loop().time() - start_time
                        try:
                            await _record_provider_success(self.provider_name, latency)
                        except Exception:
                            logger.exception(
                                "Failed to record provider success metric for %s",
                                self.provider_name,
                            )
                        await self._advance_idx_after_success(idx)
                        return
                    except (AttributeError, NotImplementedError):
                        continue
                    except Exception as e:
                        if _is_retryable_llm_error(e):
                            # Circuit breaker: record failure
                            async with _CIRCUIT_BREAKERS_LOCK:
                                if cache_key not in _CIRCUIT_BREAKERS:
                                    _CIRCUIT_BREAKERS[cache_key] = _KeyCircuitBreaker()
                                _CIRCUIT_BREAKERS[cache_key].record_failure(
                                    asyncio.get_running_loop().time()
                                )
                            last_error = e
                            logger.warning(
                                "Stream key %s failed: %s",
                                _mask_key(key),
                                str(e)[:200],
                            )
                            # DB: mark key slot as failed
                            if self._slot_ids:
                                try:
                                    async with get_session() as fresh_s:
                                        error_msg = (
                                            f"{type(e).__name__}: {str(e).split(chr(10))[0]}"
                                        )[:256]
                                        await mark_key_failure(
                                            fresh_s, self._slot_ids[idx], error_msg
                                        )
                                except Exception:
                                    logger.exception(
                                        "Failed to mark key slot %d as failed",
                                        self._slot_ids[idx],
                                    )
                            continue
                        raise
                    finally:
                        try:
                            await provider.close()
                        except Exception:
                            pass  # close failures should never mask the actual result
                # All streaming attempts failed — record failure and fallback
                if last_error:
                    try:
                        await _record_provider_failure(self.provider_name)
                    except Exception:
                        logger.exception(
                            "Failed to record provider failure metric for %s",
                            self.provider_name,
                        )
                yield await self._retry_inner(
                    messages, heavy=effective_heavy, task_type=task_type
                )
            finally:
                self._semaphore.release()
        finally:
            release_purpose_slot(sem)

    async def embed(self, text: str) -> list[float]:
        """Embed с защитой backpressure (background семафор)."""
        sem = await acquire_purpose_slot("background")
        try:
            await self._semaphore.acquire()
            try:
                return await self._try_with_retry(lambda p: p.embed(text))
            finally:
                self._semaphore.release()
        finally:
            release_purpose_slot(sem)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed_batch с защитой backpressure (background семафор)."""
        sem = await acquire_purpose_slot("background")
        try:
            await self._semaphore.acquire()
            try:
                return await self._try_with_retry(lambda p: p.embed_batch(texts))
            finally:
                self._semaphore.release()
        finally:
            release_purpose_slot(sem)

    async def validate_key(self) -> bool:
        try:
            return await self._try_with_retry(lambda p: p.validate_key())
        except Exception:
            return False

    async def close(self) -> None:
        """MultiKeyProvider is a factory — instances are closed in _try_with_retry."""
        pass

    async def list_models(self) -> list[str]:
        """Delegate to underlying provider. Not key-rotated — uses first key."""
        key = self._keys[0]
        provider = self._provider_class(key)
        try:
            return await provider.list_models()
        finally:
            await provider.close()


@dataclass
class ProviderFallback:
    """Primary provider with chat fallback to other configured providers.

    Embeddings intentionally stay on the primary provider to avoid mixing vector
    dimensions in Qdrant.
    """

    providers: list[MultiKeyProvider]
    _last_primary_dim: int | None = None

    @property
    def name(self) -> str:
        return " → ".join(p.name for p in self.providers)

    @property
    def primary(self) -> MultiKeyProvider:
        return self.providers[0]

    @property
    def _model(self) -> str | None:
        """Global model override propagated from settings (e.g. maestro_model)."""
        return self.providers[0]._model if self.providers else None

    @_model.setter
    def _model(self, value: str | None) -> None:
        for p in self.providers:
            p._model = value

    @property
    def _default_heavy(self) -> bool:
        """Default heavy flag propagated from user's use_heavy_model setting."""
        return self.providers[0]._default_heavy if self.providers else False

    @_default_heavy.setter
    def _default_heavy(self, value: bool) -> None:
        for p in self.providers:
            p._default_heavy = value

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        heavy=_UNSET,
        task_type: str = TaskType.DEFAULT,
    ) -> str:
        """Chat c адаптивным выбором провайдера.

        Сортирует провайдеров по композитному score (успешность + латентность)
        и пробует наиболее надёжного/быстрого первым. Embeddings не сортируются —
        остаются на primary для совместимости размерностей векторов.
        """
        last_error: Exception | None = None
        now = asyncio.get_running_loop().time()
        sorted_providers = sorted(
            self.providers,
            key=lambda p: _score_provider(p.provider_name, now),
            reverse=True,
        )
        for provider in sorted_providers:
            try:
                return await provider.chat(messages, heavy=heavy, task_type=task_type)
            except Exception as exc:
                if not isinstance(exc, ExhaustedError) and not _is_retryable_llm_error(
                    exc
                ):
                    raise
                last_error = exc
                logger.warning(
                    "LLM provider %s failed, trying next: %s",
                    provider.name,
                    str(exc)[:200],
                )
        raise last_error or RuntimeError("All LLM providers failed")

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        heavy=_UNSET,
        task_type: str = TaskType.DEFAULT,
    ) -> AsyncGenerator[str, None]:
        """Stream chat with adaptive provider fallback. Falls back to regular chat."""
        now = asyncio.get_running_loop().time()
        sorted_providers = sorted(
            self.providers,
            key=lambda p: _score_provider(p.provider_name, now),
            reverse=True,
        )
        for provider in sorted_providers:
            try:
                async for token in provider.chat_stream(
                    messages, heavy=heavy, task_type=task_type
                ):
                    yield token
                return
            except (AttributeError, NotImplementedError):
                continue
            except Exception as exc:
                if not isinstance(exc, ExhaustedError) and not _is_retryable_llm_error(
                    exc
                ):
                    raise
                logger.warning(
                    "LLM provider %s streaming failed, trying next: %s",
                    provider.name,
                    str(exc)[:200],
                )
        # All streaming failed — fallback to regular chat
        yield await self.chat(messages, heavy=heavy, task_type=task_type)

    async def embed(self, text: str) -> list[float]:
        """Embed с fallback по цепочке провайдеров.

        При фейле primary — пробует следующих. ВАЖНО: размерности векторов
        могут отличаться между провайдерами (BGE-M3: 1024, OpenAI: 1536).
        Fallback с несовпадающей размерностью вызывает ValueError.
        """
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                result = await provider.embed(text)
                if i == 0:
                    self._last_primary_dim = len(result)
                elif (
                    self._last_primary_dim is not None
                    and len(result) != self._last_primary_dim
                ):
                    raise ValueError(
                        f"Embedding dimension mismatch: primary={self._last_primary_dim}, "
                        f"fallback {provider.name}={len(result)}. "
                        "Vectors would corrupt Qdrant index."
                    )
                return result
            except Exception as exc:
                if not isinstance(
                    exc, (ExhaustedError, NotImplementedError, ValueError)
                ) and not _is_retryable_llm_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "Embed provider %s failed, trying fallback: %s",
                    provider.name,
                    str(exc)[:200],
                )
        raise last_error or RuntimeError("All embed providers failed")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed_batch с fallback по цепочке провайдеров.

        Аналогично embed() — при фейле primary пробует backup-провайдеров,
        с проверкой размерности векторов для предотвращения повреждения Qdrant.
        """
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                result = await provider.embed_batch(texts)
                if result:
                    if i == 0:
                        self._last_primary_dim = len(result[0])
                    elif (
                        self._last_primary_dim is not None
                        and len(result[0]) != self._last_primary_dim
                    ):
                        raise ValueError(
                            f"Embedding dimension mismatch: primary={self._last_primary_dim}, "
                            f"fallback {provider.name}={len(result[0])}. "
                            "Vectors would corrupt Qdrant index."
                        )
                return result
            except Exception as exc:
                if not isinstance(
                    exc, (ExhaustedError, NotImplementedError, ValueError)
                ) and not _is_retryable_llm_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "Embed_batch provider %s failed, trying fallback: %s",
                    provider.name,
                    str(exc)[:200],
                )
        raise last_error or RuntimeError("All embed_batch providers failed")

    async def validate_key(self) -> bool:
        for provider in self.providers:
            if await provider.validate_key():
                return True
        return False

    async def close(self) -> None:
        """Close all child provider instances."""
        for p in self.providers:
            if hasattr(p, "close"):
                await p.close()

    async def list_models(self) -> list[str]:
        """Delegate to primary provider's list_models."""
        return await self.primary.list_models()


# ─── Хелперы ──────────────────────────────────────────────────────────


def _provider_class_for(name: str) -> type | None:
    """Маппинг имени провайдера → класс."""
    return {
        "deepseek": DeepSeekProvider,
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "mistral": MistralProvider,
        "cloudflare": CloudflareProvider,
    }.get(name)


def _provider_order(primary: str) -> list[str]:
    return [primary] + [name for name in PROVIDER_ORDER if name != primary]


# ─── Task-type model resolution ──────────────────────────────────────

# Mapping TaskType → Settings attribute name for agent-specific overrides.
_TASK_TYPE_TO_SETTINGS_ATTR: dict[str, str] = {
    TaskType.MAESTRO: "maestro_model",
    TaskType.DRAFT: "draft_model",
    TaskType.MEMORY: "memory_model",
    TaskType.SEARCH: "search_model",
    TaskType.STT: "stt_model",
    TaskType.HUMANIZE: "humanize_model",
    TaskType.CLASSIFY: "classify_model",
    TaskType.SUMMARIZE: "summarize_model",
    TaskType.SKILLS: "skills_model",
    TaskType.BACKGROUND: "background_model",
    TaskType.VISION: "vision_model",
}


def _parse_user_model_overrides(user: User) -> dict[str, str] | None:
    """Parse user.settings.model_overrides JSON safely."""
    if not user.settings or not user.settings.model_overrides:
        return None
    try:
        parsed = json.loads(user.settings.model_overrides)
        if isinstance(parsed, dict):
            return parsed
        return None
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Failed to parse model_overrides for user %s",
            user.telegram_id,
        )
        return None


def _resolve_model_for_task(
    task_type: str,
    user_overrides: dict[str, str] | None,
) -> str | None:
    """Resolve model name for a given task type.

    Priority: user.model_overrides[task_type] > Settings agent override > None
    (use provider default).
    """
    from src.config import settings

    # 1. User overrides (highest priority)
    if user_overrides:
        model = user_overrides.get(task_type, "")
        if model:
            return model

    # 2. Settings agent overrides
    settings_attr = _TASK_TYPE_TO_SETTINGS_ATTR.get(task_type)
    if settings_attr:
        model = getattr(settings, settings_attr, "")
        if model:
            return model

    # 3. None = use provider default
    return None


async def build_provider(
    session: AsyncSession,
    user: User,
    purpose: str = "main",
    task_type: str = TaskType.DEFAULT,
) -> LLMProvider | None:
    """Строит провайдер с авто-ротацией ключей из LlmKeySlot.

    Сначала пробует получить активные слоты (LlmKeySlot) для нужного провайдера
    и назначения. Если слотов нет — падает на старый ApiKey.
    Для chat() строит цепочку fallback-провайдеров.
    Для embed() остаётся на первичном провайдере.
    """
    # Проверка кэша
    from src.core.context_cache import get as cache_get

    cache_key = f"provider:{user.telegram_id}:{purpose}:{task_type}"
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    provider_name = user.settings.llm_provider if user.settings else "openai"
    use_heavy = user.settings.use_heavy_model if user.settings else False

    # Попытка через новую систему LlmKeySlot
    try:
        providers: list[MultiKeyProvider] = []
        all_slot_ids: list[int] = []
        for name in _provider_order(provider_name):
            slots = await get_active_keys(session, user, name, purpose)
            if not slots:
                continue
            keys = [decrypt(s.key_enc) for s in slots]
            slot_ids = [s.id for s in slots]
            endpoints = [s.endpoint for s in slots]
            models = [s.model for s in slots]
            all_slot_ids.extend(slot_ids)
            provider_class = _provider_class_for(name)
            if provider_class is None:
                logger.warning("Unknown provider class for %s, skipping", name)
                continue
            providers.append(
                MultiKeyProvider(
                    name,
                    provider_class,
                    keys,
                    slot_ids=slot_ids,
                    endpoints=endpoints,
                    models=models,
                    # сессия для DB-трекинга открывается внутри _try_with_retry
                    # (lambda захватывает user для совместимости, session не используется)
                    session_provider=lambda: (None, user),
                    purpose=purpose,
                )
            )
        # Восстанавливаем cooldown за один проход по всем провайдерам
        await _restore_cooldowns(all_slot_ids)
        if providers:
            if len(providers) > 1:
                logger.info(
                    "LLM fallback chain (slots): %s",
                    " -> ".join(p.name for p in providers),
                )
            from src.core.context_cache import put as cache_put

            result = ProviderFallback(providers)
            result._default_heavy = use_heavy
            # Resolve model for task_type (user overrides > settings > provider default)
            if task_type != TaskType.DEFAULT:
                user_overrides = _parse_user_model_overrides(user)
                resolved_model = _resolve_model_for_task(task_type, user_overrides)
                if resolved_model:
                    result._model = resolved_model
                    logger.debug(
                        "build_provider: task_type=%s → model=%s (from overrides)",
                        task_type,
                        resolved_model,
                    )
            await cache_put(cache_key, result, ttl=300)
            return result
    except Exception:
        logger.exception("LlmKeySlot lookup failed, falling back to old ApiKey table")

    # Fallback: старый ApiKey
    providers = []
    for name in _provider_order(provider_name):
        keys = await get_api_keys(session, user, name)
        if not keys:
            continue
        provider_class = _provider_class_for(name)
        if provider_class is None:
            logger.warning("Unknown provider class for %s, skipping", name)
            continue
        providers.append(MultiKeyProvider(name, provider_class, keys, purpose=purpose))
    if not providers:
        # Проверяем: есть слоты но все в кулдауне?
        try:
            from src.db.repo import list_key_slots

            all_slots = await list_key_slots(
                session,
                user,
                provider=user.settings.llm_provider if user.settings else "openai",
            )
            in_cooldown = [
                s
                for s in all_slots
                if (cooldown := _ensure_utc(s.cooldown_until))
                and cooldown > datetime.now(timezone.utc)
            ]
            if in_cooldown:
                min_cooldown = min(
                    (
                        _ensure_utc(s.cooldown_until)
                        for s in in_cooldown
                        if s.cooldown_until
                    ),
                    default=None,
                )
                if min_cooldown is not None:
                    wait_sec = max(
                        1,
                        int(
                            (min_cooldown - datetime.now(timezone.utc)).total_seconds()
                        ),
                    )
                else:
                    wait_sec = 60
                logger.warning(
                    "build_provider: все ключи в кулдауне (wait %d сек).",
                    wait_sec,
                )
                return None
            elif all_slots:
                logger.warning("build_provider: все ключи отключены (enabled=False).")
                return None
            else:
                logger.warning("build_provider: нет ключей для провайдера.")
                return None
        except Exception:
            pass
        return None
    if len(providers) > 1:
        logger.info(
            "LLM fallback chain (legacy): %s",
            " -> ".join(p.name for p in providers),
        )
    from src.core.context_cache import put as cache_put

    result = ProviderFallback(providers)
    result._default_heavy = use_heavy
    # Resolve model for task_type (user overrides > settings > provider default)
    if task_type != TaskType.DEFAULT:
        user_overrides = _parse_user_model_overrides(user)
        resolved_model = _resolve_model_for_task(task_type, user_overrides)
        if resolved_model:
            result._model = resolved_model
            logger.debug(
                "build_provider: task_type=%s → model=%s (from overrides, legacy)",
                task_type,
                resolved_model,
            )
    await cache_put(cache_key, result, ttl=300)
    return result


class ExhaustedProvider:
    """Заглушка — все ключи в кулдауне или отсутствуют."""

    name: str = "exhausted"

    def __init__(self, reason: str = "no keys available") -> None:
        self._reason = reason

    async def validate_key(self) -> bool:
        return False

    async def chat(
        self,
        messages: object,
        *,
        heavy: bool = False,
        task_type: str = TaskType.DEFAULT,
    ) -> str:
        raise ExhaustedError(self._reason)

    async def chat_stream(
        self,
        messages: object,
        *,
        heavy: bool = False,
        task_type: str = TaskType.DEFAULT,
    ) -> AsyncGenerator[str, None]:
        raise ExhaustedError(self._reason)

    async def embed(self, text: str) -> list[float]:
        raise ExhaustedError("Cannot embed: all keys exhausted")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise ExhaustedError("Cannot embed batch: all keys exhausted")

    async def close(self) -> None:
        pass
