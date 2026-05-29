# Regex-паттерны для анти-AI детекции
import re

AI_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    # Шаблон "(глагол) вам (глагол)" — "помогу вам разобраться", "предлагаю вам рассмотреть"
    (re.compile(r"\b\w+\s+(?:вам|тебе)\s+\w+"), 0.2, "шаблонная конструкция"),
    # Перечисление через запятую из 4+ однородных
    (re.compile(r"[^,]+,[^,]+,[^,]+,[^,]+"), 0.15, "длинное перечисление"),
    # "Как [сущ], ..." — AI-конструкция
    (re.compile(r"^Как \w+,"), 0.15, "AI-вступление"),
    # Многоточие в конце абзаца
    (re.compile(r"\.\.\.\s*$"), 0.1, "многоточие в конце"),
    # "Важно:" / "Примечание:" в начале предложения
    (re.compile(r"(?i)^(важно|примечание|заметка)\s*:"), 0.25, "AI-маркер"),
    # Двойной дефис/тире — AI-пунктуация
    (re.compile(r"\w\s*—\s*\w.*\w\s*—\s*\w"), 0.1, "двойное тире"),
    # Parenthetical "(или ...)" — объяснение в скобках
    (re.compile(r"\(или\s"), 0.1, "пояснение в скобках"),
    # Роботное перечисление памяти: «я помню», «по моим данным», «согласно памяти»
    (
        re.compile(
            r"(?i)\b(я помню|по моим данным|согласно памяти|в моей базе|у меня в памяти|я знаю что)\b"
        ),
        0.35,
        "роботное перечисление памяти",
    ),
    # Список с дефисами/тире: "• факт" или "- факт" или "— факт" (2+ подряд)
    (
        re.compile(r"(?:^|\n)\s*[•\-\—]\s.*(?:\n\s*[•\-\—]\s.*){1,}"),
        0.25,
        "список фактов",
    ),
    # ▼▼▼ Новые AI-паттерны для современных LLM ▼▼▼
    # Claude patterns
    (re.compile(r"(?i)\bI aim to be\b"), 0.12, "Claude-маркер"),
    (re.compile(r"(?i)\bI strive to\b"), 0.12, "Claude-маркер"),
    (re.compile(r"(?i)\bI'm happy to\b"), 0.12, "Claude-маркер"),
    (re.compile(r"(?i)\bI'd be glad to\b"), 0.12, "Claude-маркер"),
    (re.compile(r"(?i)\blet me know if\b"), 0.10, "Claude-маркер"),
    (re.compile(r"(?i)\bfeel free to\b"), 0.10, "Claude-маркер"),
    # Gemini patterns
    (re.compile(r"(?i)\bcertainly!\b"), 0.18, "Gemini-маркер"),
    (re.compile(r"(?i)\bhere's what I found\b"), 0.15, "Gemini-маркер"),
    (re.compile(r"(?i)\blet me break this down\b"), 0.15, "Gemini-маркер"),
    # DeepSeek patterns
    (re.compile(r"(?i)\bfinally,\b"), 0.10, "DeepSeek-маркер"),
    (re.compile(r"(?i)\bin conclusion,\b"), 0.12, "DeepSeek-маркер"),
    (re.compile(r"(?i)\bto summarize,\b"), 0.12, "DeepSeek-маркер"),
    # Grok patterns
    (re.compile(r"(?i)\bhope this helps\b"), 0.10, "Grok-маркер"),
    (re.compile(r"(?i)\blet me know what you think\b"), 0.10, "Grok-маркер"),
]

# Per-word штрафы за повторения (дополняют глобальный REPEAT_PENALTY из vocabulary.py)
# Используется scorer.py для тонкой настройки: если слово найдено здесь —
# используется его штраф, иначе — глобальный REPEAT_PENALTY.
REPEAT_PENALTIES: dict[str, float] = {
    "however": 0.05,
    "moreover": 0.08,
    "furthermore": 0.08,
    "additionally": 0.06,
    "in addition": 0.06,
    "it's important to note": 0.10,
    "keep in mind": 0.07,
    # Русские аналоги
    "однако": 0.05,
    "более того": 0.08,
    "кроме того": 0.06,
    "в дополнение": 0.06,
    "важно отметить": 0.10,
    "имейте в виду": 0.07,
}

# Длина сообщения: идеальный диапазон
IDEAL_LENGTH_MIN: int = 10
IDEAL_LENGTH_MAX: int = 300
