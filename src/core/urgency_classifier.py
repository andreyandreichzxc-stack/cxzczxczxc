import re
from typing import Literal

# Списки ключевых слов
URGENT_PATTERNS = [
    r"ты где\??",
    r"ответь",
    r"срочно",
    r"позвони",
    r"алло",
    r"ауу?",
    r"deadline",
    r"дедлайн",
    r"горит",
    r"🔥",
    r"‼️",
    r"❗",
    r"пропал",
    r"куда пропал",
    r"не игнорь",
    r"ау",
]
ANGRY_PATTERNS = [
    r"обиделась",
    r"обиделся",
    r"достало",
    r"я в ярости",
    r"бесит",
    r"я злюсь",
    r"ты меня бесишь",
    r"задолбал",
    r"задолбала",
    r"достал",
    r"достала",
]
QUESTION_PATTERNS = [r"\?$", r"почему", r"зачем", r"когда", r"где", r"как"]


def classify_message(text: str) -> Literal["urgent", "important", "normal"]:
    """
    Эвристический классификатор срочности сообщения.
    Не использует LLM — только regex.
    """
    text_lower = text.lower().strip()

    # CAPS_LOCK CHECK (больше 50% букв заглавные и длина > 10)
    letters = [c for c in text if c.isalpha()]
    if letters and len(text) > 10:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.5:
            return "urgent"

    # URGENT keywords
    for pattern in URGENT_PATTERNS:
        if re.search(pattern, text_lower):
            return "urgent"

    # ANGRY keywords → important (not urgent, but needs attention)
    for pattern in ANGRY_PATTERNS:
        if re.search(pattern, text_lower):
            return "important"

    # QUESTION with urgency cues
    has_question = any(re.search(p, text_lower) for p in QUESTION_PATTERNS)
    if has_question and len(text) < 80:
        return "important"

    return "normal"
