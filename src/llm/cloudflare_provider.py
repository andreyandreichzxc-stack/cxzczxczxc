import httpx
from openai import AsyncOpenAI

from src.config import LLMDefaults, settings
from src.llm._openai_compat_mixin import OpenAICompatEmbedMixin
from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage


class CloudflareProvider(OpenAICompatEmbedMixin):
    """Cloudflare Workers AI провайдер (OpenAI-совместимый API).

    Использует AsyncOpenAI с кастомным base_url на Cloudflare Accounts AI Gateway.
    Поддерживает chat (Kimi K2.6, Qwen3) и embeddings (BGE-M3).
    """

    name = "cloudflare"

    def __init__(
        self, api_key: str, *, base_url: str | None = None, model: str | None = None
    ) -> None:
        base_url = _validate_base_url(base_url)
        if not base_url:
            account_id = settings.cloudflare_account_id
            if not account_id:
                raise ValueError(
                    "CLOUDFLARE_ACCOUNT_ID не задан в .env. "
                    "Добавь CLOUDFLARE_ACCOUNT_ID=<твой account_id> в .env"
                )
            base_url = (
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            )
        kwargs: dict = dict(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._embed_model = LLMDefaults.CLOUDFLARE_EMBED

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (
            LLMDefaults.CLOUDFLARE_CHAT_HEAVY
            if heavy
            else LLMDefaults.CLOUDFLARE_CHAT_LIGHT
        )

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = self._resolve_model(heavy)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""
