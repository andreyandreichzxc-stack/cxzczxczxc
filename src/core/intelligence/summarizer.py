"""Промпты для саммари, черновика ответа и «где мы остановились»."""

import logging

from src.core.contacts.chat_service import message_to_text


logger = logging.getLogger(__name__)
from src.core.contacts.style_profile import style_profile_as_prompt_hint
from src.core.infra.text_sanitizer import sanitize_html
from src.core.actions.vector_store import vector_store
from src.db.models import Contact, Message
from src.llm.base import ChatMessage, LLMProvider


SUMMARY_SYSTEM = (
    "Сделай КОМПАКТНОЕ (≤5 строк) саммари переписки. Без списков, без маркдауна.\n"
    "Формат: <b>Суть:</b> 1-2 фразы | <b>Ждут:</b> что от меня надо | <b>Тон:</b> агрессивный/нейтральный/тёплый\n"
    "Только <b>жирный</b> и <i>курсив</i>. Без <ul>/<ol>/<li>/<p>/<code>."
)


DRAFT_SYSTEM = (
    "Ты пишешь черновик ответа от моего имени. Только текст ответа, без префиксов и пояснений.\n"
    "Учитывай контекст последних сообщений и не повторяй уже сказанное.\n"
    "Если важная информация неоднозначна — задай короткий уточняющий вопрос вместо домысла."
)


CATCHUP_SYSTEM = (
    "Я долго не отвечал. Сделай КОМПАКТНО (≤6 строк, без списков):\n"
    "<b>Где остановились:</b> 1-3 факта коротко\n"
    "<b>Ждут:</b> что от меня хотят\n"
    "<b>Черновик:</b> 1-3 предложения ответа в моём стиле\n"
    "Только <b>жирный</b>, без <ul>/<ol>/<li>/<p>/<code>. Пиши как человек в Telegram."
)


async def summarize_chat(
    provider: LLMProvider,
    contact: Contact,
    messages: list[Message],
    *,
    owner_id: int | None = None,
    heavy: bool = False,
    global_style: str | None = None,
) -> str:
    transcript = "\n".join(message_to_text(m) for m in messages)
    user_prompt = (
        f"Собеседник: {contact.display_name}\n\n"
        f"Переписка (последние {len(messages)} сообщений):\n{transcript}"
    )
    system = SUMMARY_SYSTEM
    if global_style:
        system = system + "\n\n" + global_style

    # --- RAG: контекст про этого собеседника из всей истории ---
    if owner_id is not None:
        try:
            query_vec = await provider.embed(contact.display_name)
            hits = await vector_store.search(
                user_id=owner_id, embedding=query_vec, limit=3
            )
            if hits:
                rag_lines = []
                for h in hits:
                    prefix = f"[{h.peer_name}]" if h.peer_name else ""
                    rag_lines.append(f"{prefix} {h.text[:200]}")
                system = (
                    system
                    + "\n\nРелевантный контекст из истории переписок:\n"
                    + "\n".join(rag_lines)
                )
        except Exception:
            logger.debug("RAG search non-critical fail", exc_info=True)

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_prompt),
        ],
        heavy=heavy,
    )
    return sanitize_html(raw)


async def draft_reply(
    provider: LLMProvider,
    contact: Contact,
    messages: list[Message],
    *,
    instruction: str | None = None,
    owner_id: int | None = None,
    heavy: bool = False,
    global_style: str | None = None,
) -> str:
    transcript = "\n".join(message_to_text(m) for m in messages)
    style_hint = style_profile_as_prompt_hint(contact.style_profile, global_style)
    system = DRAFT_SYSTEM
    if style_hint:
        system = system + "\n" + style_hint

    # --- RAG: контекст про этого собеседника из всей истории ---
    if owner_id is not None:
        try:
            query_vec = await provider.embed(contact.display_name)
            hits = await vector_store.search(
                user_id=owner_id, embedding=query_vec, limit=3
            )
            if hits:
                rag_lines = []
                for h in hits:
                    prefix = f"[{h.peer_name}]" if h.peer_name else ""
                    rag_lines.append(f"{prefix} {h.text[:200]}")
                system = (
                    system
                    + "\n\nРелевантный контекст из истории переписок:\n"
                    + "\n".join(rag_lines)
                )
        except Exception:
            logger.debug("RAG search non-critical fail", exc_info=True)

    user_prompt = (
        f"Собеседник: {contact.display_name}\n\n"
        f"Контекст переписки:\n{transcript}\n\n"
        + (
            f"Инструкция: {instruction}"
            if instruction
            else "Напиши уместный ответ на последнее сообщение."
        )
    )
    raw = await provider.chat(
        [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_prompt),
        ],
        heavy=heavy,
    )
    return sanitize_html(raw)


async def catchup(
    provider: LLMProvider,
    contact: Contact,
    messages: list[Message],
    *,
    owner_id: int | None = None,
    heavy: bool = False,
    global_style: str | None = None,
) -> str:
    transcript = "\n".join(message_to_text(m) for m in messages)
    style_hint = style_profile_as_prompt_hint(contact.style_profile, global_style)
    system = CATCHUP_SYSTEM
    if style_hint:
        system = system + "\n" + style_hint

    # --- RAG: контекст про этого собеседника из всей истории ---
    if owner_id is not None:
        try:
            query_vec = await provider.embed(contact.display_name)
            hits = await vector_store.search(
                user_id=owner_id, embedding=query_vec, limit=3
            )
            if hits:
                rag_lines = []
                for h in hits:
                    prefix = f"[{h.peer_name}]" if h.peer_name else ""
                    rag_lines.append(f"{prefix} {h.text[:200]}")
                system = (
                    system
                    + "\n\nРелевантный контекст из истории переписок:\n"
                    + "\n".join(rag_lines)
                )
        except Exception:
            logger.debug("RAG search non-critical fail", exc_info=True)

    user_prompt = (
        f"Собеседник: {contact.display_name}\n\nПоследние сообщения:\n{transcript}"
    )
    raw = await provider.chat(
        [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_prompt),
        ],
        heavy=heavy,
    )
    return sanitize_html(raw)
