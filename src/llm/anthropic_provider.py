"""Anthropic provider — Claude via Messages API."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from src.llm._ssrf_guard import validate_base_url as _validate_base_url
from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic Claude provider via Messages API.

    Unlike OpenAI, Anthropic uses a different API structure:
    - System prompt is top-level, not a message
    - Messages are list[{"role": "user"|"assistant", "content": [...]}]
    - Supports streaming via server-sent events
    - Models: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus

    Models are hardcoded to match the Anthropic catalog from provider_catalog.py.
    """

    name = "anthropic"

    def __init__(
        self, api_key: str, *, base_url: str | None = None, model: str | None = None
    ) -> None:
        import anthropic

        base_url = _validate_base_url(base_url)
        kwargs: dict = {"api_key": api_key, "max_retries": 2}
        if base_url:
            kwargs["base_url"] = base_url
        self._client: anthropic.AsyncAnthropic = anthropic.AsyncAnthropic(**kwargs)
        self._model = model

    async def validate_key(self) -> bool:
        try:
            await self._client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False

    def _resolve_model(self, heavy: bool) -> str:
        return self._model or (
            "claude-3-5-sonnet-20241022" if heavy else "claude-3-5-haiku-20241022"
        )

    async def chat(self, messages: list[ChatMessage], *, heavy: bool = False) -> str:
        system, anthropic_messages = self._convert_messages(messages)
        model = self._resolve_model(heavy)
        kwargs: dict = {
            "model": model,
            "max_tokens": 4000,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        # Anthropic returns content as list of blocks
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def chat_stream(
        self, messages: list[ChatMessage], *, heavy: bool = False
    ) -> AsyncGenerator[str, None]:
        system, anthropic_messages = self._convert_messages(messages)
        model = self._resolve_model(heavy)
        kwargs: dict = {
            "model": model,
            "max_tokens": 4000,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    yield event.delta.text

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Anthropic does not support embeddings")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Anthropic does not support embeddings")

    async def list_models(self) -> list[str]:
        raise NotImplementedError("Anthropic does not expose model listing API")

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await self._client.close()

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict]]:
        """Convert ChatMessage list to Anthropic format.

        Returns (system_text, [{"role": "user"|"assistant", "content": str}])
        Anthropic requires: system is top-level, roles are only user/assistant.
        """
        system_parts: list[str] = []
        anthropic_msgs: list[dict] = []

        for msg in messages:
            role = msg.role
            if role == "system":
                system_parts.append(msg.content)
            elif role in ("user", "assistant"):
                anthropic_msgs.append({"role": role, "content": msg.content})
            elif role == "tool":
                # Tool results → wrap as user message
                anthropic_msgs.append(
                    {"role": "user", "content": f"[Tool result]: {msg.content}"}
                )
            else:
                # Unknown role → treat as user
                anthropic_msgs.append({"role": "user", "content": msg.content})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, anthropic_msgs
