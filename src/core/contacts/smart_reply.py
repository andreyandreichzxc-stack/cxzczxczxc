"""Simple emoji/sticker replies for short common messages + memory correction detection."""

from __future__ import annotations

import random
import re
from typing import Any

SIMPLE_REPLIES: dict[str, list[str]] = {
    "ok": ["👍", "👌"],
    "ладно": ["👍", "👌", "ок"],
    "спасибо": ["❤️", "🙏", "😊"],
    "благодарю": ["🙏", "❤️"],
    "привет": ["👋", "✌️"],
    "здарова": ["✌️", "👋"],
    "пока": ["👋", "😘"],
    "давай": ["👋", "🤝"],
    "ага": ["👍"],
    "ясно": ["👌", "понял 🤝"],
    "понял": ["👌"],
    "хорошо": ["👍", "🙂"],
    "отлично": ["🔥", "💯"],
    "супер": ["🔥", "🎉"],
    "нет": ["👎", "🙅"],
    "да": ["✅"],
    "конечно": ["✅", "👍"],
}


def get_simple_reply(text: str) -> str | None:
    """Return a random emoji reply if *text* is a single word matching a known pattern.

    Only single-word messages (after stripping whitespace) are considered.
    Common trailing punctuation (``.,!?;:``) is ignored during matching.

    Returns ``None`` when no pattern matches.
    """
    stripped = text.strip().lower()
    # Only match single-word messages
    if " " in stripped:
        return None
    # Strip common trailing punctuation for matching
    cleaned = stripped.strip(".,!?;:")
    if cleaned in SIMPLE_REPLIES:
        return random.choice(SIMPLE_REPLIES[cleaned])
    return None


# ── Memory Correction Patterns (Feature 2) ────────────────────────────

# Negation patterns that indicate a memory correction:
# "нет, я не в Яндексе...", "я не работаю в X", "это не так", "ты ошибся",
# "я больше не...", "уже не...", "перестал..."
_CORRECTION_NEGATIONS = re.compile(
    r"(?:^|\s)"
    r"(?:(?:нет|не|неправда|ошибся|ошиблась|неверно|не так|неправильно)"
    r"|(?:я\s+(?:больше\s+)?не\b)"
    r"|(?:уже\s+не\b)"
    r"|(?:перестал[а]?\b)"
    r")",
    re.IGNORECASE,
)

# Correction verbs that suggest the user is fixing a memory
_CORRECTION_VERBS = re.compile(
    r"(?:работаю|живу|учусь|люблю|ненавижу|хожу|езжу|знаю|умею|делаю|занимаюсь"
    r"|нахожусь|являюсь|стал[а]?|был[а]?)",
    re.IGNORECASE,
)


def detect_memory_correction(text: str) -> dict[str, Any] | None:
    """Detect if the user message is a memory correction (negation of a fact).

    Examples:
      - "нет, я не в Яндексе работаю" → correction detected
      - "я больше не веган" → correction detected
      - "ты ошибся, я не живу в Москве" → correction detected
      - "уже не хожу в спортзал" → correction detected

    Returns:
        dict with 'action': 'update'|'delete', 'old_fact_keywords': [...],
        'new_fact': str|None — or None if no correction detected.
    """
    text_clean = text.strip()
    if len(text_clean) < 5:
        return None

    # Must contain negation AND a correction verb (or factual statement)
    if not _CORRECTION_NEGATIONS.search(text_clean):
        return None

    # Try to extract what's being corrected
    # Strategy: find the core statement by removing negation words
    cleaned_for_extraction = re.sub(
        r"(?i)(?:^|[,;]?\s*)(?:нет|не|неправда|ошибся|ошиблась|неверно|это\s+не\s+так|неправильно"
        r"|я\s+больше\s+не|я\s+уже\s+не|уже\s+не|перестал[а]?)[,;]?\s*",
        "",
        text_clean,
    ).strip()

    # Also strip filler words
    cleaned_for_extraction = re.sub(
        r"(?i)^(?:я|ты|вы)\s+",
        "",
        cleaned_for_extraction,
    ).strip()

    # Extract keywords for DB search (take meaningful words)
    words = re.findall(r"[а-яёa-z]{4,}", cleaned_for_extraction.lower())
    keywords = list(set(words)) if words else [cleaned_for_extraction[:40]]

    # Detect whether this is a deletion or update
    # "больше не", "уже не", "перестал" → delete old fact
    # "не X, а Y" → update
    # Simple negation → delete
    is_delete = bool(
        re.search(
            r"(?:больше\s+не|уже\s+не|перестал[а]?|не\s+надо)",
            text_clean,
            re.IGNORECASE,
        )
    )

    # Check for correction: "не X, а Y" / "не X, Y"
    correction_match = re.search(
        r"(?:не|нет)\s+.+?(?:[,;]\s*(?:а\s+)?)(.+)",
        text_clean,
        re.IGNORECASE,
    )
    new_fact = correction_match.group(1).strip() if correction_match else None

    if is_delete and new_fact:
        # "я больше не веган, я теперь мясоед" → delete old, add new
        action = "update"
    elif is_delete:
        action = "delete"
    elif new_fact:
        action = "update"
    else:
        action = "delete"

    return {
        "action": action,
        "old_fact_keywords": keywords,
        "new_fact": new_fact if action == "update" else None,
    }


async def handle_memory_correction(
    correction: dict[str, Any],
    telegram_id: int,
) -> str:
    """Handle a detected memory correction: search and delete/update.

    Returns a response string to send to the user.
    """
    from src.db.repo import (
        delete_memory,
        get_or_create_user,
        search_memories,
        add_memory,
    )
    from src.db.session import get_session

    keywords = correction["old_fact_keywords"]
    action = correction["action"]
    new_fact = correction.get("new_fact")

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        found_all: list = []
        for kw in keywords[:3]:  # search up to 3 keywords
            results = await search_memories(session, owner, kw)
            for mem in results:
                if mem not in found_all:
                    found_all.append(mem)

    if not found_all:
        if action == "update":
            # No old fact found — just add the new one
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                await add_memory(
                    session, owner, fact=new_fact or "", source="user", confidence=0.85
                )
            return "🤔 Не нашёл что удалить, но запомнил новое. Спасибо за уточнение!"
        return "🤔 Не нашёл такого в памяти. Может, я ещё не запомнил? Уточни, что именно поправить."

    # Delete matching memories
    deleted_count = 0
    deleted_facts: list[str] = []
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        for mem in found_all[:5]:  # max 5 deletions
            success = await delete_memory(session, owner, mem.id)
            if success:
                deleted_count += 1
                if mem.fact:
                    deleted_facts.append(mem.fact[:50])

    if action == "update" and new_fact:
        # Add the corrected fact
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            await add_memory(
                session, owner, fact=new_fact, source="user", confidence=0.9
            )
        return (
            f"🧠 Понял! Забыл про «{deleted_facts[0]}…» и запомнил: «{new_fact[:80]}»."
        )

    if deleted_count > 0:
        return f"🗑 Удалил из памяти: «{deleted_facts[0]}…»."

    return "🤔 Не нашёл что именно удалить. Уточни?"
