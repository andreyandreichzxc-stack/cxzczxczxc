from __future__ import annotations

import asyncio
import logging

from src.core.context.spec import ContextChunk, ContextProvider

logger = logging.getLogger(__name__)


class ContextEngine:
    """Pluggable engine that gathers context from registered providers."""

    def __init__(self) -> None:
        self._providers: list[ContextProvider] = []

    def register(self, provider: ContextProvider) -> "ContextEngine":
        if any(p is provider or p.name == provider.name for p in self._providers):
            logger.debug("Context provider already registered: %s", provider.name)
            return self
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
        """Gather context chunks from all registered providers in parallel."""
        if not self._providers:
            return []

        tasks = [
            provider.get_context(
                query,
                telegram_id=telegram_id,
                contact_id=contact_id,
                limit=limit,
            )
            for provider in self._providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunks: list[ContextChunk] = []
        for provider, result in zip(self._providers, results):
            if isinstance(result, BaseException):
                logger.debug(
                    "Context provider '%s' failed: %s",
                    provider.name,
                    result,
                )
                continue
            chunks.extend(result)
        return chunks

    @property
    def providers(self) -> list[str]:
        return [p.name for p in self._providers]


# Module-level singleton
engine = ContextEngine()
