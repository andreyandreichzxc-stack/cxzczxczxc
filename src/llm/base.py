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

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        ...

    async def embed(self, text: str) -> list[float]:
        ...
