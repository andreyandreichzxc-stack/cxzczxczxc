"""Simple emoji/sticker replies for short common messages."""

import random

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
