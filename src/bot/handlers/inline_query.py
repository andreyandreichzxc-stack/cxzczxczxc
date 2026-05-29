"""Inline-режим: @botname запрос → поиск памяти и фактов."""

import asyncio
import logging

from aiogram import Router
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
)

from src.bot.filters import OwnerOnly
from src.core.infra.text_sanitizer import sanitize_html

logger = logging.getLogger(__name__)
router = Router()


@router.inline_query(OwnerOnly())
async def handle_inline_query(inline_query: InlineQuery) -> None:
    """Обрабатывает @botname запрос — ищет в памяти, контексте и переписках."""
    query = inline_query.query.strip()
    if not query or len(query) < 2:
        results = [
            InlineQueryResultArticle(
                id="empty",
                title="Введи запрос для поиска",
                description="Например: @botname встреча с Машей",
                input_message_content=InputTextMessageContent(
                    message_text="🔍 Введи запрос после @botname для поиска в памяти."
                ),
            )
        ]
        await inline_query.answer(results, cache_time=60)
        return

    # Поиск в 3 источниках параллельно

    async def _search_memory(q: str) -> list[str]:
        results: list[str] = []
        try:
            from src.core.memory.memory_recall import recall

            result = await recall(
                inline_query.from_user.id,
                query=q,
                limit=5,
                mode="light",
            )
            for f in result.facts[:5]:
                fact_text = f.fact[:200]
                results.append(f"🧠 {fact_text} — {f.reason}")
        except Exception as e:
            logger.debug("Memory search inline: %s", e)
        return results

    async def _search_context(q: str) -> list[str]:
        results: list[str] = []
        try:
            from src.core.memory.context_files import search_in_contexts

            ctx_results = await asyncio.to_thread(search_in_contexts, q, limit=3)
            for c in ctx_results[:3]:
                snippet = (c.get("snippet", "") or c.get("content", "") or str(c))[:200]
                key = c.get("key", "")
                label = f"📄 {key}: {snippet}" if key else f"📄 {snippet}"
                results.append(label)
        except Exception as e:
            logger.debug("Context search inline: %s", e)
        return results

    async def _search_messages(q: str, user_id: int) -> list[str]:
        results: list[str] = []
        try:
            from src.db.session import get_session
            from src.db.repo import get_or_create_user, cross_chat_search

            async with get_session() as session:
                owner = await get_or_create_user(session, user_id)
                conversations = await cross_chat_search(session, owner, q, limit=3)
                for conv in conversations[:3]:
                    display = conv.get("display_name", "чат")
                    snippets = conv.get("snippets", [])
                    for snip in snippets[:2]:
                        text = snip.get("text", str(snip))[:150]
                        sender = snip.get("sender_name", "")
                        prefix = f"{sender}: " if sender else ""
                        results.append(f"💬 {display} | {prefix}{text}")
        except Exception as e:
            logger.debug("Message search inline: %s", e)
        return results

    memory_results_future = asyncio.ensure_future(_search_memory(query))
    context_results_future = asyncio.ensure_future(_search_context(query))
    message_results_future = asyncio.ensure_future(
        _search_messages(query, inline_query.from_user.id)
    )

    memory_results, context_results, message_results = await asyncio.gather(
        memory_results_future,
        context_results_future,
        message_results_future,
        return_exceptions=True,
    )

    # Если gather вернул Exception вместо списка — заменяем на []
    if isinstance(memory_results, Exception):
        memory_results = []
    if isinstance(context_results, Exception):
        context_results = []
    if isinstance(message_results, Exception):
        message_results = []

    # ── Сборка inline-результатов ──────────────────────────────────
    inline_results: list[InlineQueryResultArticle] = []
    idx = 0

    for text in memory_results:
        inline_results.append(
            InlineQueryResultArticle(
                id=f"mem_{idx}",
                title=f"🧠 Память #{idx + 1}",
                description=sanitize_html(text[:100].replace("\n", " ")),
                input_message_content=InputTextMessageContent(
                    message_text=sanitize_html(text[:1000])
                ),
            )
        )
        idx += 1

    for text in context_results:
        inline_results.append(
            InlineQueryResultArticle(
                id=f"ctx_{idx}",
                title=f"📄 Контекст #{idx + 1}",
                description=sanitize_html(text[:100].replace("\n", " ")),
                input_message_content=InputTextMessageContent(
                    message_text=sanitize_html(text[:1000])
                ),
            )
        )
        idx += 1

    for text in message_results:
        inline_results.append(
            InlineQueryResultArticle(
                id=f"msg_{idx}",
                title=f"💬 Переписка #{idx + 1}",
                description=sanitize_html(text[:100].replace("\n", " ")),
                input_message_content=InputTextMessageContent(
                    message_text=sanitize_html(text[:1000])
                ),
            )
        )
        idx += 1

    if not inline_results:
        inline_results.append(
            InlineQueryResultArticle(
                id="no_results",
                title="Ничего не найдено",
                description=f"По запросу «{query[:50]}» ничего не найдено",
                input_message_content=InputTextMessageContent(
                    message_text=f"🔍 По запросу «{query}» ничего не найдено в памяти."
                ),
            )
        )

    await inline_query.answer(inline_results, cache_time=300)
