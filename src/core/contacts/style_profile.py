"""Профиль стиля общения с конкретным контактом. Подмешивается в промпт авто-ответа
и черновика, чтобы они звучали так же, как обычно пишет владелец этому собеседнику."""

import json
import logging
from datetime import datetime, timezone

from src.core.contacts.chat_service import message_to_text
from src.db.models import Contact, User
from src.db.repo import (
    fetch_my_messages_global,
    fetch_my_messages_in_chat,
    get_contact,
    get_or_create_user,
)
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider


logger = logging.getLogger(__name__)


STYLE_SYSTEM = (
    "Ты эксперт по анализу стиля переписки. По набору сообщений ОДНОГО автора "
    "верни компактный JSON-профиль его стиля общения с конкретным собеседником.\n\n"
    "Поля JSON:\n"
    '  "address": как обращается ("ты"/"вы"/имя/никак),\n'
    '  "register": формальный | разговорный | дружеский | официальный,\n'
    '  "length": краткие | средние | развернутые,\n'
    '  "emoji_usage": none | rare | moderate | frequent,\n'
    '  "punctuation": строгая | расслабленная (точки в конце, восклицания),\n'
    '  "typical_openings": [до 3 типичных приветствий или зачинов],\n'
    '  "typical_closings": [до 3 типичных завершений],\n'
    '  "phrases": [до 5 характерных фраз/слов-маркеров],\n'
    '  "notes": одна-две фразы — общее ощущение от стиля.\n\n'
    "Возвращай ТОЛЬКО валидный JSON, без префиксов и markdown."
)


def _parse_json_safe(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        logger.warning("Style profile JSON parse failed, using empty: %r", text[:120])
        return {}


async def build_style_profile(
    provider: LLMProvider,
    *,
    contact_label: str,
    my_messages_text: str,
) -> dict:
    if not my_messages_text.strip():
        return {}
    user_prompt = (
        f"Собеседник: {contact_label}.\n"
        "Ниже — мои (автора) сообщения этому собеседнику:\n\n"
        f"{my_messages_text}\n\n"
        "Сформируй JSON-профиль моего стиля общения с этим собеседником."
    )
    raw = await provider.chat(
        [
            ChatMessage(role="system", content=STYLE_SYSTEM),
            ChatMessage(role="user", content=user_prompt),
        ],
        heavy=False,
    )
    return _parse_json_safe(raw)


async def update_style_profile_for_contact(
    provider: LLMProvider,
    owner_telegram_id: int,
    peer_id: int,
    *,
    sample_size: int = 80,
) -> dict | None:
    async with get_session() as session:
        owner: User = await get_or_create_user(session, owner_telegram_id)
        my_msgs = await fetch_my_messages_in_chat(
            session, owner, peer_id, limit=sample_size
        )
        contact: Contact | None = await get_contact(session, owner, peer_id)

    if not my_msgs or contact is None:
        return None

    text = "\n".join(message_to_text(m) for m in my_msgs)
    profile = await build_style_profile(
        provider,
        contact_label=contact.display_name,
        my_messages_text=text,
    )
    if not profile:
        return None

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        contact = await get_contact(session, owner, peer_id)
        if contact is not None:
            contact.style_profile = json.dumps(profile, ensure_ascii=False)
            contact.style_updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    return profile


def style_profile_as_prompt_hint(
    profile_json: str | None,
    global_profile_json: str | None = None,
) -> str:
    if not profile_json:
        result = ""
    else:
        try:
            p = json.loads(profile_json)
        except Exception:
            result = ""
        else:
            parts: list[str] = []
            if p.get("address"):
                parts.append(f"обращение: {p['address']}")
            if p.get("register"):
                parts.append(f"регистр: {p['register']}")
            if p.get("length"):
                parts.append(f"длина: {p['length']}")
            if p.get("emoji_usage"):
                parts.append(f"эмодзи: {p['emoji_usage']}")
            if p.get("punctuation"):
                parts.append(f"пунктуация: {p['punctuation']}")
            if p.get("typical_openings"):
                parts.append("типичные зачины: " + ", ".join(p["typical_openings"]))
            if p.get("phrases"):
                parts.append("маркерные фразы: " + ", ".join(p["phrases"]))
            if p.get("notes"):
                parts.append("заметки: " + p["notes"])
            result = (
                "Пиши в моём стиле общения с этим собеседником: "
                + "; ".join(parts)
                + "."
            )

    # Добавляем глобальный стиль, если есть
    if global_profile_json:
        try:
            g = json.loads(global_profile_json)
        except Exception:
            logger.debug("global_style_profile JSON parse failed", exc_info=True)
            pass
        else:
            lines: list[str] = ["Твой общий стиль общения:"]

            mf = g.get("mat_frequency", 0)
            lines.append(
                f"- Мат: {'часто' if mf > 0.3 else 'редко' if mf > 0 else 'нет'}"
                + (
                    f" ({', '.join(g.get('mat_words', [])[:5])})"
                    if g.get("mat_words")
                    else ""
                )
            )

            ef = g.get("emoji_frequency", 0)
            top_emoji = g.get("top_emoji", [])
            emoji_line = (
                f"- Эмодзи: {'часто' if ef > 0.3 else 'редко' if ef > 0 else 'нет'}"
            )
            if top_emoji:
                emoji_line += f", любимые: {', '.join(top_emoji)}"
            lines.append(emoji_line)

            cf = g.get("caps_frequency", 0)
            lines.append(
                f"- CAPS: {'часто' if cf > 0.3 else 'редко' if cf > 0 else 'нет'}"
            )

            avg_len = g.get("avg_msg_length", 0)
            lines.append(f"- Средняя длина сообщения: {avg_len:.0f} символов")

            affection = g.get("affectionate_forms", [])
            if affection:
                lines.append(f"- Ласкательные обращения: {', '.join(affection)}")

            slang = g.get("slang_words", [])
            if slang:
                lines.append(f"- Сленг: {', '.join(slang[:8])}")

            result = (
                (result + "\n\n" + "\n".join(lines)) if result else "\n".join(lines)
            )

    return result


async def update_global_style_profile(user_id: int, provider=None):
    """
    Анализирует все исходящие сообщения владельца и сохраняет глобальный стиль-профиль
    в поле User.global_style_profile (JSON).
    """
    from src.core.contacts.style_heuristics import analyze_messages_heuristic

    async with get_session() as session:
        owner = await get_or_create_user(session, user_id)
        messages = await fetch_my_messages_global(session, owner, limit=200)

    texts = [m.text for m in messages if m.text]
    if not texts:
        logger.warning("global_style: нет исходящих сообщений для анализа")
        return

    profile = analyze_messages_heuristic(texts)
    json_str = json.dumps(profile, ensure_ascii=False)

    async with get_session() as session:
        owner = await get_or_create_user(session, user_id)
        owner.global_style_profile = json_str
        owner.global_style_updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.commit()

    logger.info(
        "Global style profile updated: %d msgs, mat=%.2f, emoji=%.2f, caps=%.2f",
        len(texts),
        profile["mat_frequency"],
        profile["emoji_frequency"],
        profile["caps_frequency"],
    )
