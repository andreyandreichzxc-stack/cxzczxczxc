from __future__ import annotations

import logging

from src.core.context.spec import ContextChunk, ContextProvider

logger = logging.getLogger(__name__)


class ContextEngine:
    """Pluggable engine that gathers context from registered providers."""

    def __init__(self) -> None:
        self._providers: list[ContextProvider] = []

    def register(self, provider: ContextProvider) -> "ContextEngine":
        self._providers.append(provider)
        logger.debug("Registered context provider: %s", provider.name)
        return self

    async def gather(
        self,
        query: str,
        *,
        telegram_id: int,
        contact_id: int | None = None,
        limit: int = 8,
    ) -> list[ContextChunk]:
        """Gather context chunks from all registered providers."""
        chunks: list[ContextChunk] = []
        for provider in self._providers:
            try:
                result = await provider.get_context(
                    query,
                    telegram_id=telegram_id,
                    contact_id=contact_id,
                    limit=limit,
                )
                chunks.extend(result)
            except Exception:
                logger.debug(
                    "Context provider '%s' failed, continuing",
                    provider.name,
                    exc_info=True,
                )
        return chunks

    @property
    def providers(self) -> list[str]:
        return [p.name for p in self._providers]


# Module-level singleton
engine = ContextEngine()
