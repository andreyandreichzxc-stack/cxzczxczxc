import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.crypto import decrypt
from src.db.models import User
from src.db.repo import get_active_keys, get_api_keys, mark_key_failure, mark_key_used
from src.llm.base import ChatMessage, LLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.mistral_provider import MistralProvider
from src.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

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
                    # DB: отметить успешное использование
                    if self._slot_ids and self._session_provider:
                        try:
                            s, _ = self._session_provider()
                            await mark_key_used(s, self._slot_ids[idx])
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
                        # DB: отметить падение слота
                        if self._slot_ids and self._session_provider:
                            try:
                                s, _ = self._session_provider()
                                await mark_key_failure(
                                    s, self._slot_ids[idx], str(exc)[:256]
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to mark key slot %d as failed",
                                    self._slot_ids[idx],
                                )
                        continue
                    raise
            if skipped and last_error is None:
                raise RuntimeError(
                    f"All {self.provider_name} API keys are cooling down after capacity errors"
                )
            raise last_error or RuntimeError(
                f"All {self.provider_name} API keys failed"
            )

    async def chat(self, messages, *, heavy: bool = False) -> str:
        return await self._try_with_retry(lambda p: p.chat(messages, heavy=heavy))

    async def embed(self, text: str) -> list[float]:
        return await self._try_with_retry(lambda p: p.embed(text))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._try_with_retry(lambda p: p.embed_batch(texts))

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
                    session_provider=lambda: (session, user),
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
        providers.append(MultiKeyProvider(name, _provider_class_for(name), keys))
    if not providers:
        return None
    if len(providers) > 1:
        logger.info(
            "LLM fallback chain (legacy): %s",
            " -> ".join(p.name for p in providers),
        )
    return ProviderFallback(providers)
