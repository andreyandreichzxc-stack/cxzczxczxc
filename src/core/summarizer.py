"""Промпты для саммари, черновика ответа и «где мы остановились»."""

from src.core.chat_service import message_to_text
from src.core.style_profile import style_profile_as_prompt_hint
from src.core.text_sanitizer import sanitize_html
from src.db.models import Contact, Message
from src.llm.base import ChatMessage, LLMProvider


SUMMARY_SYSTEM = (
    "Ты делаешь сжатое саммари переписки. Структура ответа:\n"
    "📝 <b>Главное</b> — 2–4 буллета.\n"
    "🎯 <b>Открытые вопросы / задачи</b> — что от меня ждут.\n"
    "📅 <b>Договорённости</b> — даты, встречи, обещания (с датой если есть).\n"
    "🌡 <b>Тон</b> — одной фразой.\n"
    "Используй HTML-разметку aiogram (<b>, <i>, <code>). Без markdown."
)


DRAFT_SYSTEM = (
    "Ты пишешь черновик ответа от моего имени. Только текст ответа, без префиксов и пояснений.\n"
    "Учитывай контекст последних сообщений и не повторяй уже сказанное.\n"
    "Если важная информация неоднозначна — задай короткий уточняющий вопрос вместо домысла."
)


CATCHUP_SYSTEM = (
    "Я давно не отвечал в этом чате. Сделай:\n"
    "1) <b>Где мы остановились</b> — 2–3 буллета о текущем состоянии.\n"
    "2) <b>Чего от меня ждут</b> — что нужно ответить или сделать.\n"
    "3) <b>Черновик ответа</b> — 1–4 предложения, в моём стиле.\n"
    "Используй HTML-разметку aiogram."
)


async def summarize_chat(
    provider: LLMProvider,
    contact: Contact,
    messages: list[Message],
    *,
    heavy: bool = False,
) -> str:
    transcript = "\n".join(message_to_text(m) for m in messages)
    user_prompt = (
        f"Собеседник: {contact.display_name}\n\n"
        f"Переписка (последние {len(messages)} сообщений):\n{transcript}"
    )
    raw = await provider.chat(
        [
            ChatMessage(role="system", content=SUMMARY_SYSTEM),
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
    heavy: bool = False,
) -> str:
    transcript = "\n".join(message_to_text(m) for m in messages)
    style_hint = style_profile_as_prompt_hint(contact.style_profile)
    system = DRAFT_SYSTEM
    if style_hint:
        system = system + "\n" + style_hint
    user_prompt = (
        f"Собеседник: {contact.display_name}\n\n"
        f"Контекст переписки:\n{transcript}\n\n"
        + (f"Инструкция: {instruction}" if instruction else "Напиши уместный ответ на последнее сообщение.")
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
    heavy: bool = False,
) -> str:
    transcript = "\n".join(message_to_text(m) for m in messages)
    style_hint = style_profile_as_prompt_hint(contact.style_profile)
    system = CATCHUP_SYSTEM
    if style_hint:
        system = system + "\n" + style_hint
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
