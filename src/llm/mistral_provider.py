from openai import AsyncOpenAI

from src.config import LLMDefaults
from src.llm.base import ChatMessage


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider:
    name = "mistral"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=MISTRAL_BASE_URL)

    async def validate_key(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        model = LLMDefaults.MISTRAL_CHAT_HEAVY if heavy else LLMDefaults.MISTRAL_CHAT_LIGHT
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=LLMDefaults.MISTRAL_EMBED, input=text)
        return resp.data[0].embedding
