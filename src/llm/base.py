from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol


Role = Literal["system", "user", "assistant"]


class TaskType:
    """Типы LLM-задач — определяют выбор модели и параметры генерации."""

    MAESTRO = "maestro"  # Основной orchestration (планирование, синтез)
    DRAFT = "draft"  # Черновики ответов
    MEMORY = "memory"  # Извлечение/тегирование памяти
    SEARCH = "search"  # Семантический поиск
    STT = "stt"  # Распознавание речи
    HUMANIZE = "humanize"  # Очеловечивание текста
    CLASSIFY = "classify"  # Классификация намерений
    SUMMARIZE = "summarize"  # Саммари/дайджест
    SKILLS = "skills"  # Навыки и инструменты
    BACKGROUND = "background"  # Фоновые задачи
    VISION = "vision"  # Мультимодальный анализ изображений
    DEFAULT = "default"  # Обычный чат


@dataclass
class ChatMessage:
    role: Role
    content: str


class LLMProvider(Protocol):
    name: str

    async def validate_key(self) -> bool:
        """Лёгкий запрос: подходит ли ключ. Используется в /settings."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        heavy: bool = False,  # DEPRECATED: use task_type instead. Kept for backward compat.
        task_type: str = "default",
    ) -> str: ...

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        heavy: bool = False,  # DEPRECATED: use task_type instead. Kept for backward compat.
        task_type: str = "default",
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from chat completion. Raises NotImplementedError if unsupported."""
        raise NotImplementedError("chat_stream not supported by this provider")

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    async def list_models(self) -> list[str]:
        """Return available model IDs from the provider. Raises NotImplementedError if unsupported."""
        ...

    async def close(self) -> None:
        """Close underlying HTTP client and release connections."""


class VisionProvider(Protocol):
    """Protocol for providers that support multimodal (image+text) inputs."""

    async def chat_with_image(
        self,
        messages: list[ChatMessage],
        image_data: bytes,
        image_mime: str = "image/jpeg",
        *,
        task_type: str = "default",
    ) -> str: ...
