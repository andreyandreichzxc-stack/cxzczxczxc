import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.crypto import decrypt
from src.db.models import User
from src.db.repo import get_active_keys, get_api_keys, mark_key_failure, mark_key_used
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.mistral_provider import MistralProvider
from src.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class ExhaustedError(Exception):
    """Все API-ключи провайдера исчерпаны (колдаун/отключены)."""

    pass


PROVIDER_ORDER = ("openai", "gemini", "mistral")
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
)
KEY_COOLDOWN_SECONDS = 90.0
_FAILED_KEYS: dict[tuple[str, str], float] = {}

# Per-purpose лимиты параллельных запросов
_PURPOSE_SEMAPHORES: dict[str, asyncio.Semaphore] = {
    "main": asyncio.Semaphore(2),
    "draft": asyncio.Semaphore(1),
    "memory": asyncio.Semaphore(1),
    "background": asyncio.Semaphore(1),
    "analysis": asyncio.Semaphore(1),
    "urgent": asyncio.Semaphore(2),
    "fallback": asyncio.Semaphore(2),
}


async def acquire_purpose_slot(purpose: str) -> asyncio.Semaphore:
    """Захватывает слот для purpose. Возвращает семафор."""
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
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in {429, 500, 503}:
        return True
    code = str(getattr(exc, "code", "") or "").lower()
    if code in {"429", "500", "503", "3505", "service_tier_capacity_exceeded"}:
        return True
    body = getattr(exc, "body", None) or getattr(exc, "response", None)
    text = f"{type(exc).__name__} {exc} {body}".lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}…{key[-4:]}"


# ─── MultiKey: обёртка для ротации ключей ─────────────────────────────


class MultiKeyProvider:
    """Обёртка: ротирует ключи провайдера при ошибке 429/503/500.

    Позволяет указать несколько API-ключей для одного LLM-провайдера.
    При получении ошибки пропускной способности (rate limit, capacity exceeded)
    автоматически переключается на следующий ключ.
    """

    def __init__(
        self,
        provider_name: str,
        provider_class: type,
        keys: list[str],
        slot_ids: list[int] | None = None,
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
        self._session_provider = session_provider
        self._kwargs = kwargs
        self._idx = 0
        self._lock = asyncio.Lock()
        self._current_purpose = purpose
        self.name = f"{provider_name}(×{len(self._keys)})"

    async def _try_with_retry(self, operation, *args: object, **kwargs: object):
        """Пробует операцию со всеми ключами по очереди.

        Пропускает ключи, которые фейлились менее 60 секунд назад.
        При успехе обновляет активный индекс и отмечает слот (DB).
        При временной ошибке помечает слот как упавший (DB cooldown).
        """
        async with self._lock:
            last_error: Exception | None = None
            now = asyncio.get_running_loop().time()
            skipped = 0
            for attempt in range(len(self._keys)):
                idx = (self._idx + attempt) % len(self._keys)
                key = self._keys[idx]
                failed_at = _FAILED_KEYS.get((self.provider_name, key))
                if failed_at is not None and now - failed_at < KEY_COOLDOWN_SECONDS:
                    skipped += 1
                    continue
                try:
                    provider = self._provider_class(key, **self._kwargs)
                    result = await operation(provider, *args, **kwargs)
                    self._idx = idx
                    _FAILED_KEYS.pop((self.provider_name, key), None)
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
                    return result
                except Exception as exc:
                    if _is_retryable_llm_error(exc):
                        _FAILED_KEYS[(self.provider_name, key)] = (
                            asyncio.get_running_loop().time()
                        )
                        last_error = exc
                        logger.warning(
                            "LLM %s key %s temporarily failed, rotating: %s",
                            self.provider_name,
                            _mask_key(key),
                            exc,
                        )
                        # DB: отметить падение слота (fresh session)
                        if self._slot_ids:
                            try:
                                async with get_session() as fresh_s:
                                    await mark_key_failure(
                                        fresh_s, self._slot_ids[idx], str(exc)[:256]
                                    )
                            except Exception:
                                logger.exception(
                                    "Failed to mark key slot %d as failed",
                                    self._slot_ids[idx],
                                )
                        continue
                    raise
            if last_error:
                raise ExhaustedError(
                    f"Все {len(self._keys)} ключей {self.provider_name} недоступны "
                    f"(последняя ошибка: {last_error})"
                )
            raise ExhaustedError(
                f"Все {len(self._keys)} ключей {self.provider_name} в кулдауне"
            )

    async def chat(self, messages, *, heavy: bool = False) -> str:
        sem = await acquire_purpose_slot(self._current_purpose)
        try:
            return await self._chat_with_retry(messages, heavy=heavy)
        finally:
            release_purpose_slot(sem)

    async def _chat_with_retry(self, messages, *, heavy: bool = False) -> str:
        # Early exit: все ли ключи в кулдауне?
        now = asyncio.get_running_loop().time()
        all_dead = all(
            (self.provider_name, key) in _FAILED_KEYS
            and now - _FAILED_KEYS[(self.provider_name, key)] < KEY_COOLDOWN_SECONDS
            for key in self._keys
        )
        if all_dead:
            self._idx = 0
            raise ExhaustedError(
                f"Все {len(self._keys)} ключей {self.provider_name} в кулдауне. Попробуй позже."
            )
        return await self._try_with_retry(lambda p: p.chat(messages, heavy=heavy))

    async def embed(self, text: str) -> list[float]:
        """Embed с защитой backpressure (background семафор)."""
        sem = await acquire_purpose_slot("background")
        try:
            return await self._try_with_retry(lambda p: p.embed(text))
        finally:
            release_purpose_slot(sem)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed_batch с защитой backpressure (background семафор)."""
        sem = await acquire_purpose_slot("background")
        try:
            return await self._try_with_retry(lambda p: p.embed_batch(texts))
        finally:
            release_purpose_slot(sem)

    async def validate_key(self) -> bool:
        try:
            return await self._try_with_retry(lambda p: p.validate_key())
        except Exception:
            return False


@dataclass
class ProviderFallback:
    """Primary provider with chat fallback to other configured providers.

    Embeddings intentionally stay on the primary provider to avoid mixing vector
    dimensions in Qdrant.
    """

    providers: list[MultiKeyProvider]

    @property
    def name(self) -> str:
        return " → ".join(p.name for p in self.providers)

    @property
    def primary(self) -> MultiKeyProvider:
        return self.providers[0]

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return await provider.chat(messages, heavy=heavy)
            except Exception as exc:
                if not _is_retryable_llm_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "LLM provider %s failed, trying fallback: %s", provider.name, exc
                )
        raise last_error or RuntimeError("All LLM providers failed")

    async def embed(self, text: str) -> list[float]:
        return await self.primary.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.primary.embed_batch(texts)

    async def validate_key(self) -> bool:
        for provider in self.providers:
            if await provider.validate_key():
                return True
        return False


# ─── Хелперы ──────────────────────────────────────────────────────────


def _provider_class_for(name: str) -> type:
    """Маппинг имени провайдера → класс."""
    return {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "mistral": MistralProvider,
    }[name]


def _provider_order(primary: str) -> list[str]:
    return [primary] + [name for name in PROVIDER_ORDER if name != primary]


async def build_provider(
    session: AsyncSession,
    user: User,
    purpose: str = "main",
) -> LLMProvider | None:
    """Строит провайдер с авто-ротацией ключей из LlmKeySlot.

    Сначала пробует получить активные слоты (LlmKeySlot) для нужного провайдера
    и назначения. Если слотов нет — падает на старый ApiKey.
    Для chat() строит цепочку fallback-провайдеров.
    Для embed() остаётся на первичном провайдере.
    """
    provider_name = user.settings.llm_provider if user.settings else "openai"

    # Попытка через новую систему LlmKeySlot
    try:
        providers: list[MultiKeyProvider] = []
        for name in _provider_order(provider_name):
            slots = await get_active_keys(session, user, name, purpose)
            if not slots:
                continue
            keys = [decrypt(s.key_enc) for s in slots]
            slot_ids = [s.id for s in slots]
            providers.append(
                MultiKeyProvider(
                    name,
                    _provider_class_for(name),
                    keys,
                    slot_ids=slot_ids,
                    # сессия для DB-трекинга открывается внутри _try_with_retry
                    # (lambda захватывает user для совместимости, session не используется)
                    session_provider=lambda: (None, user),
                    purpose=purpose,
                )
            )
        if providers:
            if len(providers) > 1:
                logger.info(
                    "LLM fallback chain (slots): %s",
                    " -> ".join(p.name for p in providers),
                )
            return ProviderFallback(providers)
    except Exception:
        logger.exception("LlmKeySlot lookup failed, falling back to old ApiKey table")

    # Fallback: старый ApiKey
    providers = []
    for name in _provider_order(provider_name):
        keys = await get_api_keys(session, user, name)
        if not keys:
            continue
        providers.append(
            MultiKeyProvider(name, _provider_class_for(name), keys, purpose=purpose)
        )
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
                if s.cooldown_until and s.cooldown_until > datetime.now(timezone.utc)
            ]
            if in_cooldown:
                min_cooldown = min(
                    (s.cooldown_until for s in in_cooldown if s.cooldown_until),
                    default=None,
                )
                wait_sec = (
                    int((min_cooldown - datetime.now(timezone.utc)).total_seconds())
                    if min_cooldown
                    else 60
                )
                return ExhaustedProvider(
                    f"Все ключи в кулдауне. Попробуй через {wait_sec} сек."
                )
            elif all_slots:
                return ExhaustedProvider(
                    "Все ключи отключены (enabled=False). Проверь /keys."
                )
            else:
                return ExhaustedProvider(
                    "Нет ключей. Добавь через /keys add или /settings."
                )
        except Exception:
            pass
        return None
    if len(providers) > 1:
        logger.info(
            "LLM fallback chain (legacy): %s",
            " -> ".join(p.name for p in providers),
        )
    return ProviderFallback(providers)


class ExhaustedProvider:
    """Заглушка — все ключи в кулдауне или отсутствуют."""

    name: str = "exhausted"

    def __init__(self, reason: str = "no keys available") -> None:
        self._reason = reason

    async def validate_key(self) -> bool:
        return False

    async def chat(  # type: ignore[return]
        self, messages: object, *, heavy: bool = False
    ) -> str:
        return f"❌ {self._reason}"

    async def embed(self, text: str) -> list[float]:
        raise ExhaustedError("Cannot embed: all keys exhausted")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise ExhaustedError("Cannot embed batch: all keys exhausted")
