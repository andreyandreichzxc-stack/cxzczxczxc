import httpx
from openai import AsyncOpenAI

from src.config import LLMDefaults
from src.llm.base import ChatMessage


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider:
    name = "mistral"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=MISTRAL_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def validate_key(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = (
            LLMDefaults.MISTRAL_CHAT_HEAVY if heavy else LLMDefaults.MISTRAL_CHAT_LIGHT
        )
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        from src.core.embedding_cache import get as cache_get, set as cache_set

        cached = cache_get(text)
        if cached is not None:
            return cached
        resp = await self._client.embeddings.create(
            model=LLMDefaults.MISTRAL_EMBED, input=text
        )
        result = resp.data[0].embedding
        cache_set(text, result)
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        from src.core.embedding_cache import get as cache_get, set as cache_set

        if not texts:
            return []

        # Проверяем кэш — собираем только некэшированные тексты
        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for i, t in enumerate(texts):
            cached = cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if uncached_texts:
            resp = await self._client.embeddings.create(
                model=LLMDefaults.MISTRAL_EMBED, input=uncached_texts
            )
            api_results = [d.embedding for d in resp.data]
            for idx, emb in zip(uncached_indices, api_results):
                cache_set(texts[idx], emb)
                results[idx] = emb

        return results  # type: ignore[return-value]
