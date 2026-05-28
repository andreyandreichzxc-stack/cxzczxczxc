"""Shared mixins for OpenAI-compatible providers.

Hierarchy:
    OpenAICompatBaseMixin  — validate_key, list_models, close
        └── OpenAICompatEmbedMixin  — embed, embed_batch (requires _embed_model)

OpenRouter uses only the base (no embeddings).
OpenAI, DeepSeek, Mistral, Cloudflare use the full embed mixin.

Requires subclasses to set:
    self._client  — AsyncOpenAI-compatible client
    self._embed_model — str, embedding model name (embed mixin only)
"""

from __future__ import annotations

from typing import Any

from openai import APIConnectionError, AuthenticationError, PermissionDeniedError

from src.core.actions.embedding_cache import get as cache_get, set as cache_set


class OpenAICompatBaseMixin:
    """Common OpenAI-compatible methods: validate_key, list_models, close.

    Used by providers that don't support embeddings (e.g., OpenRouter).
    """

    _client: Any  # AsyncOpenAI

    async def validate_key(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except AuthenticationError:
            return False
        except PermissionDeniedError:
            return False
        except APIConnectionError:
            raise
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        models = await self._client.models.list()
        return [m.id for m in models.data]

    async def close(self) -> None:
        await self._client.close()


class OpenAICompatEmbedMixin(OpenAICompatBaseMixin):
    """Embedding + common OpenAI-compatible methods shared across providers."""

    _embed_model: str

    async def embed(self, text: str) -> list[float]:
        cached = cache_get(text, self._embed_model)
        if cached is not None:
            return cached
        resp = await self._client.embeddings.create(model=self._embed_model, input=text)
        result = resp.data[0].embedding
        cache_set(text, result, self._embed_model)
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for i, t in enumerate(texts):
            cached = cache_get(t, self._embed_model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if uncached_texts:
            resp = await self._client.embeddings.create(
                model=self._embed_model, input=uncached_texts
            )
            api_results = [d.embedding for d in resp.data]
            for idx, emb in zip(uncached_indices, api_results):
                cache_set(texts[idx], emb, self._embed_model)
                results[idx] = emb

        return results  # type: ignore[return-value]
