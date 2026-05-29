import httpx
from openai import AsyncOpenAI

from src.llm._openai_compat_mixin import OpenAICompatEmbedMixin
from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_CHAT_LIGHT = "mistral-small-latest"
MISTRAL_CHAT_HEAVY = "mistral-medium-latest"


class MistralProvider(OpenAICompatEmbedMixin):
    name = "mistral"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        base_url = _validate_base_url(base_url)
        kwargs: dict = dict(
            api_key=api_key,
            base_url=base_url or MISTRAL_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._embed_model = embed_model

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (MISTRAL_CHAT_HEAVY if heavy else MISTRAL_CHAT_LIGHT)

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = self._resolve_model(heavy)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""
