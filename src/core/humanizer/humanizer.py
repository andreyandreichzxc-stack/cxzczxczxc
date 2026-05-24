"""Humanizer — заменяет AI-маркеры на человеческие аналоги."""

import re

from .scorer import analyze_ai_score


_REPLACEMENTS: dict[str, str] = {
    "конечно": "",  # удаляем
    "разумеется": "",
    "я понимаю": "понимаю",
    "я понимаю вашу": "понимаю твою",
    "я здесь чтобы помочь": "",
    "я здесь чтобы поддержать": "",
    "это совершенно нормально": "это нормально",
    "вы не одиноки": "",
    "во-первых": "",
    "во-вторых": "",
    "в-третьих": "",
    "в заключение": "",
    "подводя итог": "короче",
    "следует отметить": "",
    "необходимо подчеркнуть": "",
    "обратите внимание": "",
    "хочу подчеркнуть": "",
    "искренне": "",
    "с радостью": "",
    "всегда рад": "",
    "надеюсь это поможет": "",
    "если у вас будут вопросы": "",
    "в данном контексте": "",
    "в рамках": "",
    "давайте": "",
    "позвольте": "",
    "приношу извинения": "сорри",
    "прошу прощения за": "сорри за",
    "благодарю за": "спасибо за",
    "стоит отметить": "",
    "важно помнить": "",
}


def humanize_text(text: str) -> str:
    """Убрать AI-маркеры из текста. Case-insensitive."""
    if not text:
        return text or ""
    result = text
    for phrase, replacement in _REPLACEMENTS.items():
        if phrase.lower() in result.lower():
            # Case-insensitive replace preserving original case where possible
            result = re.sub(re.escape(phrase), replacement, result, flags=re.IGNORECASE)
    # Убрать множественные пробелы и пустые строки
    result = re.sub(r"  +", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# Фразы-клише, которые бот часто ставит в конце ответа
_CLICHÉ_ENDINGS: list[str] = [
    "если что пиши",
    "обращайся",
    "я всегда рядом",
    "рад помочь",
    "буду рад помочь",
    "чем ещё могу помочь",
    "если будут вопросы",
    "если понадобится",
]

# Контекстные дополнения к ответу
_CONTEXT_FOLLOWUPS: dict[str, str] = {
    "recipe": "приятного аппетита! 🍲",
    "news": "буду держать в курсе 📰",
    "summary": "буду держать в курсе 📰",
    "search": "если нужно копнуть глубже — скажи 🔍",
    "analysis": "если нужно копнуть глубже — скажи 🔍",
    "memory": "запомнил, не забуду 🧠",
    "reminder": "запомнил, не забуду 🧠",
    "contact": "отправлю, как скажешь ✉️",
    "send": "отправлю, как скажешь ✉️",
    "task": "сделаем 📋",
    "commitment": "сделаем 📋",
}


def humanize_response(text: str, context_hint: str | None = None) -> str:
    """Улучшить ответ бота: убрать шаблонные концовки и добавить естественное
    завершение в зависимости от контекста.

    Args:
        text: Исходный текст ответа.
        context_hint: Категория контекста для подбора фразы-дополнения.
            Одна из: recipe, news, summary, search, analysis,
            memory, reminder, contact, send, task, commitment или None.

    Returns:
        Текст с более естественным тоном.
    """
    if not text:
        return text or ""

    # 1. Оценка AI-шности
    score, breakdown = analyze_ai_score(text)
    has_ai_patterns = (
        score > 0.15
        or bool(breakdown.get("markers"))
        or bool(breakdown.get("patterns"))
    )

    # 2. Короткий и чистый — ничего не меняем
    if not has_ai_patterns and len(text) < 30:
        return text

    # 3. Удаление шаблонных концовок
    result = text
    for ending in _CLICHÉ_ENDINGS:
        # Ищем фразу как концовку (опционально с пунктуацией перед/после)
        pattern = re.compile(
            r"[\s,.\!?;:\-–—]*" + re.escape(ending) + r"(?:[\s,.\!?;:\-–—]*(?:\n|$))",
            re.IGNORECASE,
        )
        result = pattern.sub("", result)

    # Очистка хвостовой пунктуации после удаления
    result = result.rstrip(" ,.!?;:\u2013\u2014-")
    result = result.strip()

    # 4. Добавление контекстной фразы
    if context_hint and context_hint in _CONTEXT_FOLLOWUPS:
        followup = _CONTEXT_FOLLOWUPS[context_hint]
        if result:
            # Если текст уже заканчивается точкой — убираем её для многоточия
            result = result.rstrip(".!?")
            result = f"{result}... {followup}"
        else:
            result = followup

    return result
