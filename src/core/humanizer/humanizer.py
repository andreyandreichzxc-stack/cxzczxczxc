"""Humanizer — заменяет AI-маркеры на человеческие аналоги."""

import logging
import re
import time

from .scorer import analyze_ai_score
from .stats import record_check

logger = logging.getLogger(__name__)

ANTI_AI_MODES = {"off", "log", "fix"}

_REPLACEMENTS: dict[str, str] = {
    "конечно": "",  # удаляем
    "разумеется": "",
    "безусловно": "",
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


def humanize_text(text: str, user_id: int = 0) -> str:
    """Убрать AI-маркеры из текста. Case-insensitive."""
    if not text:
        return text or ""
    result = text
    for phrase, replacement in _REPLACEMENTS.items():
        if phrase.lower() in result.lower():
            # Case-insensitive replace preserving original case where possible
            result = re.sub(re.escape(phrase), replacement, result, flags=re.IGNORECASE)
    # Learned replacements from feedback
    if user_id:
        learned = _get_learned_replacements(user_id)
        for phrase, replacement in learned.items():
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

# Контекстные дополнения к ответу (текст без эмодзи — эмодзи добавляет _pick_context_emoji)
_CONTEXT_FOLLOWUPS: dict[str, str] = {
    "recipe": "приятного аппетита!",
    "news": "буду держать в курсе",
    "summary": "буду держать в курсе",
    "search": "если нужно копнуть глубже — скажи",
    "analysis": "если нужно копнуть глубже — скажи",
    "memory": "запомнил, не забуду",
    "reminder": "запомнил, не забуду",
    "contact": "отправлю, как скажешь",
    "send": "отправлю, как скажешь",
    "task": "сделаем",
    "commitment": "сделаем",
}

# Контекстно-зависимые эмодзи для разных стилей
_EMOJI_BY_CONTEXT: dict[str, dict[str, str]] = {
    "recipe": {"short": "🍲", "warm": "🍲👨‍🍳", "default": "🍳"},
    "news": {"default": "📰"},
    "summary": {"default": "📰"},
    "search": {"default": "🔍"},
    "analysis": {"default": "🔍"},
    "memory": {"default": "🧠"},
    "reminder": {"default": "🧠"},
    "contact": {"default": "✉️"},
    "send": {"default": "✉️"},
    "task": {"default": "📋"},
    "commitment": {"default": "📋"},
}


def _pick_context_emoji(context_hint: str, style_profile: str) -> str:
    """Выбирает эмодзи на основе контекста и стилевого профиля.

    Args:
        context_hint: Категория контекста (recipe, news, search, ...).
        style_profile: Строка стилевого профиля пользователя.

    Returns:
        Строка с эмодзи (может быть пустой, если стиль запрещает эмодзи).
    """
    if not context_hint:
        return ""
    entry = _EMOJI_BY_CONTEXT.get(context_hint, {})
    style_lower = (style_profile or "").lower()
    if "сухой" in style_lower or "без эмодзи" in style_lower:
        return ""
    if "тёплый" in style_lower or "warm" in style_lower:
        return entry.get("warm", entry.get("default", ""))
    if "коротко" in style_lower or "brief" in style_lower:
        return entry.get("short", entry.get("default", ""))
    return entry.get("default", "")


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


def _get_humanize_threshold(text_len: int) -> float:
    """Adaptive AI-score threshold based on text length.

    Short texts need stronger evidence of AI patterns to be flagged (higher threshold).
    Long texts are more suspicious and get flagged at lower threshold.
    """
    if text_len < 50:
        return 0.4
    if text_len < 200:
        return 0.3
    return 0.2


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
    user_id: int = 0,
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
        user_id: ID пользователя для персонализированных замен (0=нет).

    Returns:
        Текст с более естественным тоном.
    """
    if not text:
        return text or ""

    # 1. Оценка AI-шности
    score, breakdown = analyze_ai_score(text)
    ai_threshold = _get_humanize_threshold(len(text))
    has_ai_patterns = (
        score > ai_threshold
        or bool(breakdown.get("markers"))
        or bool(breakdown.get("patterns"))
    )

    has_cliche = any(ending.lower() in text.lower() for ending in _CLICHÉ_ENDINGS)

    # 2. Короткий и чистый — ничего не меняем
    if not context_hint and not has_cliche and not has_ai_patterns and len(text) < 30:
        return text

    # 3. Light pass: реально применяем marker replacements, если скорер их нашёл.
    result = text
    if has_ai_patterns:
        result = _preservation_check(text, humanize_text(result, user_id=user_id))

    # 4. Удаление шаблонных концовок
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

    # 5. Защита: не добавлять контекстный хвост к коротким/техническим ответам
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
            # Контекстно-зависимый эмодзи вместо хардкода
            context_emoji = _pick_context_emoji(context_hint, style_profile)
            if context_emoji:
                followup = f"{followup} {context_emoji}"
            if result:
                # Если текст уже заканчивается точкой — убираем её для многоточия
                result = result.rstrip(".!?")
                result = f"{result}... {followup}"
            else:
                result = followup

    # 6. Filler words — casual style, low AI score, moderate length
    result = _maybe_add_fillers(result, score, style_profile)

    # 7. Learned replacements from feedback
    if user_id:
        learned = _get_learned_replacements(user_id)
        for phrase, replacement in learned.items():
            result = re.sub(re.escape(phrase), replacement, result, flags=re.IGNORECASE)

    return result


def normalize_anti_ai_mode(mode: str | None, *, enabled: bool | None = None) -> str:
    """Normalize Anti-AI runtime mode.

    off: do nothing.
    log: only measure/log AI markers.
    fix: apply light humanizer to assistant responses.
    """
    normalized = (mode or "").strip().lower()
    if normalized in ANTI_AI_MODES:
        return normalized
    if enabled:
        return "fix"
    return "off"


def apply_anti_ai_mode(
    text: str,
    *,
    mode: str | None,
    context_hint: str | None = None,
    style_profile: str = "",
    user_id: int = 0,
    source: str = "assistant_response",
) -> str:
    """Apply Anti-AI runtime semantics to assistant responses.

    This helper is intentionally not used for exact user-authored send text.
    """
    if not text:
        return text or ""

    normalized_mode = normalize_anti_ai_mode(mode)
    score_before, breakdown = analyze_ai_score(text)

    if normalized_mode == "off":
        record_check(score_before, score_before, False)
        return text

    if normalized_mode == "log":
        logger.info(
            "Anti-AI log source=%s score=%.3f markers=%s patterns=%s",
            source,
            score_before,
            breakdown.get("markers", []),
            breakdown.get("patterns", []),
        )
        record_check(score_before, score_before, False)
        return text

    fixed = humanize_response(
        text,
        context_hint=context_hint,
        style_profile=style_profile,
        user_id=user_id,
    )
    score_after, _ = analyze_ai_score(fixed)
    changed = fixed != text
    record_check(score_before, score_after, changed)
    if changed:
        logger.debug(
            "Anti-AI fix source=%s score %.3f -> %.3f",
            source,
            score_before,
            score_after,
        )
    return fixed


# Filler words for casual/friendly style
_FILLERS_START = ["ну", "короче", "слушай", "так"]
_FILLERS_MID = ["кстати", ", блин", ", как бы"]
_FILLERS_END = [", короче", ", ну такое"]


def _maybe_add_fillers(text: str, ai_score: float, style_profile: str) -> str:
    """Add natural filler words for casual/friendly style. Only when:
    - AI score is low (already natural)
    - Style is casual (not formal/business)
    - Text is moderate length (30-300 chars)
    - Random 30% chance — not every message
    """
    if not text or len(text) < 30 or len(text) > 300:
        return text
    if ai_score > 0.2:  # only enhance already-natural text
        return text
    casual_keywords = (
        "коротко",
        "кратко",
        "разговорный",
        "дружеский",
        "casual",
        "warm",
    )
    if style_profile and not any(kw in style_profile.lower() for kw in casual_keywords):
        return text
    import random

    if random.random() > 0.3:
        return text

    # Choose position: start, mid, or end
    r = random.random()
    if r < 0.35 and text[0].isupper():
        filler = random.choice(_FILLERS_START)
        return f"{filler}, {text[0].lower()}{text[1:]}"
    elif r < 0.70:
        filler = random.choice(_FILLERS_MID)
        # Insert after first sentence
        dot = text.find(". ")
        if dot > 10 and dot < len(text) - 5:
            return (
                text[: dot + 1] + f" {filler}, {text[dot + 2].lower()}{text[dot + 3 :]}"
            )
    else:
        filler = random.choice(_FILLERS_END)
        return text.rstrip(".!?") + filler
    return text


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

    # Names (capitalized Russian words not at sentence start). Do not preserve
    # words that are known AI markers, otherwise light cleanup of "Конечно"
    # would be rolled back as if a name was lost.
    removable_words = {phrase.split()[0].lower() for phrase in _REPLACEMENTS}
    removable_words.update(phrase.split()[0].lower() for phrase in _CLICHÉ_ENDINGS)
    names = [
        word
        for word in re.findall(r"(?<![.!?] )(?<![.!?] )[А-ЯЁ][а-яё]+", original)
        if word.lower() not in removable_words
    ]
    critical.extend(names[:5])

    # Quoted text
    quoted = re.findall(r'["«][^"»]+["»]', original)
    critical.extend(quoted[:3])

    # List structure
    orig_list_items = len(re.findall(r"^[\d•\-]\s", original, re.MULTILINE))
    human_list_items = len(re.findall(r"^[\d•\-]\s", humanized, re.MULTILINE))
    if orig_list_items > 2 and human_list_items < orig_list_items * 0.5:
        return original

    # Проверяем присутствие
    for item in critical:
        if item and item not in humanized:
            logger.warning(
                "Preservation fail: %r lost in humanize_deep, falling back to original",
                item[:50],
            )
            return original  # fallback

    return humanized


async def humanize_deep(
    text: str, provider, user_style: str = "", user_id: int = 0
) -> str:
    """LLM-based глубокое очеловечивание текста.

    Вызывается только когда ``analyze_ai_score(text) > threshold``
    (адаптивный threshold зависит от длины текста).
    Для всего остального — ``humanize_response``.

    Args:
        text: Исходный текст ответа бота.
        provider: LLM-провайдер (должен иметь метод chat).
        user_style: Дополнительная подсказка о стиле пользователя.
        user_id: ID пользователя для персонализированных замен (0=нет).

    Returns:
        Переписанный текст. При ошибке возвращает оригинал.
    """
    if not text or len(text) < 50:
        return text
    # Self-contained adaptive threshold check
    score, _ = analyze_ai_score(text)
    if score <= _get_humanize_threshold(len(text)):
        # Not AI-like enough to warrant deep humanization
        return text
    style_hint = f"Стиль: {user_style}" if user_style else ""
    prompt = DEEP_HUMANIZE_PROMPT.format(style_hint=style_hint)
    try:
        from src.llm.base import ChatMessage, TaskType

        result = await provider.chat(
            [
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=text),
            ],
            task_type=TaskType.HUMANIZE,
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


def _get_learned_replacements(user_id: int) -> dict[str, str]:
    """Извлекает замены из rejected feedback-записей пользователя.

    Для каждого rejected фидбека находит слова из original,
    которых нет в corrected — и добавляет их в словарь замен
    (с пустой строкой в качестве удаления).
    Кешируется на 5 минут через обычный dict / cache.
    """
    replacements: dict[str, str] = {}
    entries = _feedback_store.get(user_id, [])
    for entry in entries:
        if entry.get("accepted", True):
            continue  # только rejected
        original_words = set(entry.get("original", "").lower().split())
        corrected_words = set(entry.get("corrected", "").lower().split())
        # Слова, которые были в original, но убраны/изменены в corrected
        removed = original_words - corrected_words
        for word in removed:
            if len(word) > 2:  # не трогаем короткие слова/союзы
                replacements[word] = ""
    return replacements


def get_few_shot_examples(user_id: int) -> str:
    """Get recent rejection examples for ANTI_AI_BLOCK injection.

    Args:
        user_id: Telegram ID пользователя.

    Returns:
        Отформатированный блок примеров (до 3) или пустая строка.
    """
    feedbacks = _feedback_store.get(user_id, [])[-10:]
    rejected = [
        f
        for f in feedbacks
        if not f.get("accepted", True) and f.get("original") and f.get("corrected")
    ]
    if not rejected:
        return ""
    examples = []
    for fb in rejected[-3:]:
        examples.append(
            f"- Было: «{fb['original'][:100]}»\n  Стало: «{fb['corrected'][:100]}»"
        )
    return "Примеры того как НЕ надо отвечать:\n" + "\n".join(examples)
