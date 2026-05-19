"""Contact Archetypes — классификация контактов по паттернам общения."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.db.repo import (
    fetch_chat_messages,
    get_contact,
    get_or_create_user,
    list_contacts,
    list_memories,
)
from src.db.session import get_session

logger = logging.getLogger(__name__)

ARCHETYPE_PROFILES = {
    "close_friend": {
        "label": "🤝 Близкий друг",
        "reply_style": "очень тёплый, на «ты», с уменьшительно-ласкательными именами и смайликами. Используй «солнышко», «дорогой/дорогая» если уместно.",
        "min_messages": 30,
        "min_positive_ratio": 0.5,
        "max_negative_ratio": 0.15,
    },
    "family": {
        "label": "👨‍👩‍👧 Семья",
        "reply_style": "заботливый, тёплый, но без излишней фамильярности коллег. На «ты», с эмодзи ❤️.",
        "min_messages": 20,
        "min_positive_ratio": 0.4,
        "max_negative_ratio": 0.25,
    },
    "colleague": {
        "label": "💼 Коллега",
        "reply_style": "вежливо, по-деловому, без фамильярности. На «вы» если из контекста. Без смайликов.",
        "min_messages": 15,
    },
    "romantic": {
        "label": "💕 Романтический интерес",
        "reply_style": "очень тёплый и интимный. Ласковые слова, много эмодзи ❤️🥰😘. Уменьшительно-ласкательные формы имени.",
        "min_messages": 40,
        "min_positive_ratio": 0.6,
        "max_negative_ratio": 0.1,
    },
    "acquaintance": {
        "label": "👤 Знакомый",
        "reply_style": "нейтрально, вежливо. Коротко. Без смайликов.",
        "min_messages": 5,
    },
    "toxic": {
        "label": "⚠️ Токсичный",
        "reply_style": "холодно и сухо. Игнорировать провокации. Отвечать односложно.",
        "min_negative_ratio": 0.4,
        "min_messages": 10,
    },
}


async def classify_contact(owner_id: int, contact_id: int) -> str | None:
    """Классифицирует контакт по памяти и сообщениям. Возвращает архетип или None."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        contact = await get_contact(session, owner, contact_id)
        if not contact:
            return None

        memories = await list_memories(session, owner, contact_id=contact_id)
        messages = await fetch_chat_messages(session, owner, contact_id, limit=100)

        total_msgs = len(messages)
        if total_msgs < 5:
            return "unknown"

        # Сентимент-анализ
        active_memories = [m for m in memories if m.is_active and m.sentiment]
        total_facts = len(active_memories)
        if total_facts == 0:
            # Только сообщения, без фактов — по количеству
            if total_msgs > 50:
                return "close_friend"
            elif total_msgs > 20:
                return "acquaintance"
            return "unknown"

        positive = sum(1 for m in active_memories if m.sentiment == "positive")
        negative = sum(
            1 for m in active_memories if m.sentiment in ("negative", "contradictory")
        )
        pos_ratio = positive / total_facts
        neg_ratio = negative / total_facts

        # Проверка токсичности первой
        if neg_ratio >= 0.4 and total_facts >= 10:
            return "toxic"
        if neg_ratio >= 0.35 and total_msgs >= 20:
            return "toxic"

        if total_msgs >= 40 and pos_ratio >= 0.6 and neg_ratio <= 0.1:
            return "romantic"
        if total_msgs >= 30 and pos_ratio >= 0.5 and neg_ratio <= 0.15:
            return "close_friend"
        if total_msgs >= 20 and pos_ratio >= 0.4:
            return "family"
        if total_msgs >= 15:
            return "colleague"

        return "acquaintance"


def archetype_reply_hint(archetype: str | None) -> str:
    """Возвращает подсказку для авто-ответа на основе архетипа."""
    if not archetype or archetype == "unknown":
        return ""
    profile = ARCHETYPE_PROFILES.get(archetype, {})
    label = profile.get("label", "")
    style = profile.get("reply_style", "")
    if not style:
        return ""
    return f"\n\nАРХЕТИП КОНТАКТА: {label}\nСтиль ответа: {style}"


async def classify_all_contacts(owner_id: int) -> dict[str, int]:
    """Классифицирует ВСЕ контакты владельца. Возвращает статистику."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        contacts = await list_contacts(
            session, owner, kinds=("user",), include_bots=False
        )
        stats: dict[str, int] = defaultdict(int)
        for contact in contacts:
            archetype = await classify_contact(owner_id, contact.peer_id)
            if archetype:
                contact.archetype = archetype
                stats[archetype] += 1
        await session.commit()
        return dict(stats)


def format_archetype_stats(stats: dict) -> str:
    """Форматирует статистику архетипов."""
    if not stats:
        return "Архетипы не определены."
    lines = ["<b>🏷 Архетипы контактов:</b>", ""]
    for archetype, count in sorted(stats.items(), key=lambda x: -x[1]):
        profile = ARCHETYPE_PROFILES.get(archetype, {})
        label = profile.get("label", archetype)
        lines.append(f"{label}: {count}")
    return "\n".join(lines)
