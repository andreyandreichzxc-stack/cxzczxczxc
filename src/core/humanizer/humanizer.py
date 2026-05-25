"""Humanizer — заменяет AI-маркеры на человеческие аналоги."""

import logging
import re
import time

from .scorer import analyze_ai_score

logger = logging.getLogger(__name__)


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
    "если что",
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

# Хранилище фидбека: user_id → list[{original, corrected, accepted, time}]
_feedback_store: dict[int, list[dict]] = {}

# Кеш последнего humanized-ответа: user_id → последний ответ бота ДО коррекции
_last_response_cache: dict[int, str] = {}

# TTL для cleanup (в секундах)
_FEEDBACK_TTL: float = 86400.0  # 24 часа
_RESPONSE_CACHE_TTL: float = 3600.0  # 1 час
_RESPONSE_CACHE_MAX: int = 500
_feedback_last_cleanup: float = 0.0
_response_cache_times: dict[int, float] = {}  # user_id → timestamp


def _maybe_cleanup_caches() -> None:
    """Periodic cleanup of in-memory caches. Called lazily on write."""
    global _feedback_last_cleanup
    now = time.time()

    # Cleanup feedback store (every 10 minutes)
    if now - _feedback_last_cleanup > 600:
        _feedback_last_cleanup = now
        cutoff = now - _FEEDBACK_TTL
        stale_users = [
            uid
            for uid, entries in list(_feedback_store.items())
            if not entries or entries[-1].get("time", 0) < cutoff
        ]
        for uid in stale_users:
            del _feedback_store[uid]

    # Cleanup response cache (evict oldest when over max, or expired)
    if len(_last_response_cache) > _RESPONSE_CACHE_MAX:
        # Evict 20% oldest
        evict_count = max(1, len(_last_response_cache) // 5)
        oldest = sorted(_response_cache_times.items(), key=lambda x: x[1])
        for uid, _ in oldest[:evict_count]:
            _last_response_cache.pop(uid, None)
            _response_cache_times.pop(uid, None)

    # TTL-based cleanup for response cache
    expired = [
        uid
        for uid, ts in list(_response_cache_times.items())
        if now - ts > _RESPONSE_CACHE_TTL
    ]
    for uid in expired:
        _last_response_cache.pop(uid, None)
        _response_cache_times.pop(uid, None)


DEEP_HUMANIZE_PROMPT = """Перепиши текст как человек, а не AI.
Убери: канцелярит, шаблонные фразы, «конечно», «безусловно», излишнюю вежливость, перечисления через «во-первых».
Сохрани ВСЕ факты и смысл.
Не добавляй эмодзи без причины.
{style_hint}
Пиши естественно, как в переписке с другом."""


def humanize_response(
    text: str,
    context_hint: str | None = None,
    style_profile: str = "",
) -> str:
    """Улучшить ответ бота: убрать шаблонные концовки и добавить естественное
    завершение в зависимости от контекста.

    Args:
        text: Исходный текст ответа.
        context_hint: Категория контекста для подбора фразы-дополнения.
            Одна из: recipe, news, summary, search, analysis,
            memory, reminder, contact, send, task, commitment или None.
        style_profile: Строка стилевого профиля пользователя
            (из get_or_update_style_profile). Если содержит указания
            «без эмодзи» — эмодзи из контекстных фраз убираются.

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

    has_cliche = any(ending.lower() in text.lower() for ending in _CLICHÉ_ENDINGS)

    # 2. Короткий и чистый — ничего не меняем
    if not context_hint and not has_cliche and not has_ai_patterns and len(text) < 30:
        return text

    # 3. Удаление шаблонных концовок
    result = text
    for _ in range(3):
        for ending in _CLICHÉ_ENDINGS:
            # Ищем фразу как концовку (опционально с пунктуацией перед/после)
            pattern = re.compile(
                r"[\s,.\!?;:\-–—]*"
                + re.escape(ending)
                + r"(?:[\s,.\!?;:\-–—]*(?:\n|$))",
                re.IGNORECASE,
            )
            result = pattern.sub("", result)

    # Очистка хвостовой пунктуации после удаления
    result = result.rstrip(" ,.!?;:\u2013\u2014-")
    result = result.strip()

    # Определяем, нужно ли убирать эмодзи из стилевого профиля
    _no_emoji = style_profile and (
        "без эмодзи" in style_profile
        or "минимально или не используй" in style_profile
        or "почти без эмодзи" in style_profile
    )

    # 4. Защита: не добавлять контекстный хвост к коротким/техническим ответам
    if context_hint and context_hint in _CONTEXT_FOLLOWUPS:
        stripped = result.rstrip(".!?,;: \n")
        short_answer = len(stripped) < 50 and not any(
            kw in stripped.lower() for kw in ["вот", "смотри", "рецепт", "список"]
        )
        code_or_json = "```" in result or result.strip().startswith("{")
        if code_or_json or (
            short_answer and context_hint in {"send", "search", "analysis"}
        ):
            pass  # не добавляем tail
        else:
            followup = _CONTEXT_FOLLOWUPS[context_hint]
            if _no_emoji:
                # Убираем эмодзи из контекстной фразы
                followup = re.sub(
                    r"[\U0001F300-\U0001F9FF\u2600-\u27BF\u2B50\u2702-\u27B0]",
                    "",
                    followup,
                ).strip()
            if result:
                # Если текст уже заканчивается точкой — убираем её для многоточия
                result = result.rstrip(".!?")
                result = f"{result}... {followup}"
            else:
                result = followup

    return result


def _preservation_check(original: str, humanized: str) -> str:
    """Проверяет, что критичные данные сохранились после humanize_deep.

    Если что-то важное пропало — возвращает оригинал как fallback.
    """
    if not humanized or len(humanized) < len(original) * 0.3:
        return original  # too much stripped

    # Извлекаем критичные паттерны из оригинала
    critical: list[str] = []

    # URLs
    urls = re.findall(r'https?://[^\s<>"]+', original)
    critical.extend(urls)

    # @mentions
    mentions = re.findall(r"@\w+", original)
    critical.extend(mentions[:3])

    # Даты в разных форматах
    dates = re.findall(
        r"\b(?:\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?|\d{1,2}\s+(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)\w*\s+\d{4}?)\b",
        original,
        re.IGNORECASE,
    )
    critical.extend(dates[:3])

    # Кодовые блоки (тройные обратные кавычки)
    code_blocks = re.findall(r"```[\s\S]*?```", original)
    critical.extend(code_blocks[:2])

    # Числа > 4 цифр (телефоны, суммы и т.п.)
    big_nums = re.findall(r"\b\d{3,}\b", original)
    critical.extend([n for n in big_nums if len(n) >= 5][:3])

    # Проверяем присутствие
    for item in critical:
        if item and item not in humanized:
            logger.warning(
                "Preservation fail: %r lost in humanize_deep, falling back to original",
                item[:50],
            )
            return original  # fallback

    return humanized


async def humanize_deep(text: str, provider, user_style: str = "") -> str:
    """LLM-based глубокое очеловечивание текста.

    Вызывается только когда ``analyze_ai_score(text) > 0.3``
    и ``len(text) > 100``. Для всего остального — ``humanize_response``.

    Args:
        text: Исходный текст ответа бота.
        provider: LLM-провайдер (должен иметь метод chat).
        user_style: Дополнительная подсказка о стиле пользователя.

    Returns:
        Переписанный текст. При ошибке возвращает оригинал.
    """
    if not text or len(text) < 50:
        return text
    style_hint = f"Стиль: {user_style}" if user_style else ""
    prompt = DEEP_HUMANIZE_PROMPT.format(style_hint=style_hint)
    try:
        from src.llm.base import ChatMessage

        result = await provider.chat(
            [
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=text),
            ]
        )
        result = result if result else text
        return _preservation_check(text, result)
    except Exception:
        return text  # fallback to original


def record_humanizer_feedback(
    user_id: int,
    original: str,
    corrected: str,
    accepted: bool,
) -> None:
    """Сохраняет фидбек о качестве очеловечивания.

    Вызывается когда пользователь явно поправляет бота
    («нет, не так», «исправь») — accepted=False,
    либо когда бот спрашивает «Так лучше?» и получает «да» — accepted=True.

    Хранит последние 50 записей на пользователя в in-memory dict.
    """
    _feedback_store.setdefault(user_id, []).append(
        {
            "original": original[:200],
            "corrected": corrected[:200],
            "accepted": accepted,
            "time": time.time(),
        }
    )
    # Keep last 50 entries
    if len(_feedback_store[user_id]) > 50:
        _feedback_store[user_id] = _feedback_store[user_id][-50:]
    _maybe_cleanup_caches()


def _cache_last_humanized(user_id: int, text: str) -> None:
    """Кеширует последний humanized-ответ бота для пользователя.

    Используется чтобы при коррекции («нет, не так») знать,
    что именно бот сказал до этого.
    """
    if text:
        _last_response_cache[user_id] = text
        _response_cache_times[user_id] = time.time()
        _maybe_cleanup_caches()


def _pop_last_humanized(user_id: int) -> str | None:
    """Возвращает и удаляет последний humanized-ответ из кеша."""
    _response_cache_times.pop(user_id, None)
    return _last_response_cache.pop(user_id, None)
