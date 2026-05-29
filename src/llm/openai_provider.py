import httpx
from collections.abc import AsyncGenerator
from openai import AsyncOpenAI

from src.llm._openai_compat_mixin import OpenAICompatEmbedMixin
from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage

OPENAI_CHAT_LIGHT = "gpt-5-mini"
OPENAI_CHAT_HEAVY = "gpt-5.5"


class OpenAIProvider(OpenAICompatEmbedMixin):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        base_url = _validate_base_url(base_url)
        kwargs: dict = dict(api_key=api_key, timeout=httpx.Timeout(60.0, connect=10.0))
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._embed_model = embed_model

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (OPENAI_CHAT_HEAVY if heavy else OPENAI_CHAT_LIGHT)

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = self._resolve_model(heavy)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(
        self, messages: list[ChatMessage], *, heavy: bool = False
    ) -> AsyncGenerator[str, None]:
        model = self._resolve_model(heavy)
        fmt = [{"role": m.role, "content": m.content} for m in messages]
        stream = await self._client.chat.completions.create(
            model=model, messages=fmt, stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
