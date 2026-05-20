import asyncio

from google import genai

from src.config import LLMDefaults
from src.llm.base import ChatMessage


def _to_gemini_contents(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    """Возвращает (system_instruction, contents) для google-genai."""
    system_chunks: list[str] = []
    contents: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_chunks.append(m.content)
        else:
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
    system = "\n\n".join(system_chunks) if system_chunks else None
    return system, contents


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key, http_options={"timeout": 60000})

    async def validate_key(self) -> bool:
        def _check() -> bool:
            try:
                # пагинированный итератор; первый элемент достаточен
                next(iter(self._client.models.list()))
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_check)

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = (
            LLMDefaults.GEMINI_CHAT_HEAVY if heavy else LLMDefaults.GEMINI_CHAT_LIGHT
        )
        system, contents = _to_gemini_contents(messages)

        def _call() -> str:
            config = {"system_instruction": system} if system else None
            resp = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return resp.text or ""

        return await asyncio.to_thread(_call)

    async def embed(self, text: str) -> list[float]:
        from src.core.actions.embedding_cache import get as cache_get, set as cache_set

        cached = cache_get(text, LLMDefaults.GEMINI_EMBED)
        if cached is not None:
            return cached

        def _call() -> list[float]:
            resp = self._client.models.embed_content(
                model=LLMDefaults.GEMINI_EMBED,
                contents=text,
            )
            return list(resp.embeddings[0].values)

        result = await asyncio.to_thread(_call)
        cache_set(text, result, LLMDefaults.GEMINI_EMBED)
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        from src.core.actions.embedding_cache import get as cache_get, set as cache_set

        if not texts:
            return []

        # Проверяем кэш — собираем только некэшированные тексты
        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for i, t in enumerate(texts):
            cached = cache_get(t, LLMDefaults.GEMINI_EMBED)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if uncached_texts:
            # Gemini поддерживает до 100 текстов за вызов — разбиваем на чанки
            api_results: list[list[float]] = []
            chunk_size = 100
            for start in range(0, len(uncached_texts), chunk_size):
                chunk = uncached_texts[start : start + chunk_size]

                def _call(chunk: list[str] = chunk) -> list[list[float]]:
                    resp = self._client.models.embed_content(
                        model=LLMDefaults.GEMINI_EMBED,
                        contents=chunk,
                    )
                    return [list(e.values) for e in resp.embeddings]

                api_results.extend(await asyncio.to_thread(_call))

            for idx, emb in zip(uncached_indices, api_results):
                cache_set(texts[idx], emb, LLMDefaults.GEMINI_EMBED)
                results[idx] = emb

        return results  # type: ignore[return-value]
