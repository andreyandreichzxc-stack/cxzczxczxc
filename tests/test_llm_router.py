"""Tests for LLM key/provider fallback behavior."""

from __future__ import annotations

import pytest

from src.llm.base import ChatMessage
from src.llm.router import ExhaustedError, MultiKeyProvider, ProviderFallback


class CapacityError(Exception):
    status_code = 429
    code = "service_tier_capacity_exceeded"


class FakeProvider:
    calls: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(self, messages, *, heavy: bool = False) -> str:
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


@pytest.mark.asyncio
async def test_multikey_rotates_on_capacity_error():
    FakeProvider.calls = []
    provider = MultiKeyProvider("fake", FakeProvider, ["bad-1", "good-2"])

    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result == "ok:good-2"
    assert FakeProvider.calls == ["bad-1", "good-2"]


@pytest.mark.asyncio
async def test_provider_fallback_tries_next_provider_for_chat():
    FakeProvider.calls = []
    primary = MultiKeyProvider("primary", FakeProvider, ["bad-primary"])
    secondary = MultiKeyProvider("secondary", FakeProvider, ["good-secondary"])
    provider = ProviderFallback([primary, secondary])

    result = await provider.chat([ChatMessage(role="user", content="hi")])

    assert result == "ok:good-secondary"
    assert FakeProvider.calls == ["bad-primary", "good-secondary"]


@pytest.mark.asyncio
async def test_provider_fallback_keeps_embeddings_on_primary_provider():
    FakeProvider.calls = []
    primary = MultiKeyProvider("primary-embed", FakeProvider, ["good-primary"])
    secondary = MultiKeyProvider("secondary-embed", FakeProvider, ["good-secondary"])
    provider = ProviderFallback([primary, secondary])

    result = await provider.embed("hello")

    assert result == [1.0]
    assert FakeProvider.calls == ["embed:good-primary"]


# ---------------------------------------------------------------------------
# ExhaustedError fallback — primary exhausted → secondary used
# ---------------------------------------------------------------------------


class ExhaustingProvider:
    """Provider that always raises ExhaustedError."""

    calls: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(self, messages, *, heavy: bool = False) -> str:
        self.calls.append(self.api_key)
        raise ExhaustedError(f"All keys for {self.api_key} exhausted")


class GoodProvider:
    """Provider that always succeeds."""

    calls: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(self, messages, *, heavy: bool = False) -> str:
        self.calls.append(self.api_key)
        return f"ok:{self.api_key}"


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
