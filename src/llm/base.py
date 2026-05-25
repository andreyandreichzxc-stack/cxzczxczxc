from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class LLMProvider(Protocol):
    name: str

    async def validate_key(self) -> bool:
        """Лёгкий запрос: подходит ли ключ. Используется в /settings."""

    async def chat(
        self, messages: list[ChatMessage], *, heavy: bool = False
    ) -> str: ...

    async def chat_stream(
        self, messages: list[ChatMessage], *, heavy: bool = False
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from chat completion. Raises NotImplementedError if unsupported."""
        raise NotImplementedError("chat_stream not supported by this provider")

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None:
        """Close underlying HTTP client and release connections."""
