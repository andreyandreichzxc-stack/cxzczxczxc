"""Adaptive Persona — бот подстраивает стиль общения под пользователя.

ChatGPT-style personality system:
- base_tone: базовый стиль и тон (default/professional/friendly/frank/whimsical/efficient/cynical)
- warmth / enthusiasm / headings_lists / emoji_level: характеристики (low/normal/high)
- custom_instructions: пользовательские инструкции
- alias: псевдоним (как обращаться)
- adaptive_mode_enabled: авто-адаптация на основе обратной связи
- base_snapshot_json: снапшот базовых настроек для сброса
"""

import json
import logging

from src.db.session import get_session
from src.db.repo import get_or_create_user, get_persona, update_persona
from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

# ── ChatGPT-style: базовые тона → промпт-инструкции ──
BASE_TONE_PROMPTS: dict[str, str] = {
    "default": "отвечай в своём естественном стиле",
    "professional": "тактичный официальный тон, на «вы», без панибратства",
    "friendly": "тёплый разговорный тон, как с другом",
    "frank": "прямой честный тон с юмором, не церемонься",
    "whimsical": "творческий игривый стиль, можно с шутками и неожиданными поворотами",
    "efficient": "максимально коротко, только суть, без воды",
    "cynical": "критичный саркастичный взгляд, с иронией и чёрным юмором",
    "assertive": "уверенный аргументированный тон, отстаиваешь свою позицию, споришь если не согласен",
    "rebellious": "дерзкий бунтарский тон, провоцируешь, не боишься идти против мнения владельца если считаешь его неправильным",
}

VALID_TONES = frozenset(BASE_TONE_PROMPTS)
VALID_LEVELS = frozenset({"low", "normal", "high"})

# ── Адаптивный режим: паттерны обратной связи → коррекция параметров ──
ADAPT_FEEDBACK_PATTERNS: list[tuple[tuple[str, ...], str, str, str]] = [
    # (триггеры, поле, направление low, направление high)
    (
        ("слишком много эмодзи", "перебор с эмодзи", "меньше эмодзи", "убери эмодзи"),
        "emoji_level",
        "low",
        "low",
    ),
    (
        ("больше эмодзи", "добавь эмодзи", "мало эмодзи", "где эмодзи"),
        "emoji_level",
        "high",
        "high",
    ),
    (
        ("слишком тепло", "слишком дружелюбно", "попроще", "полегче"),
        "warmth",
        "normal",
        "low",
    ),
    (
        ("слишком холодно", "слишком сухо", "потеплее", "будь добрее"),
        "warmth",
        "high",
        "normal",
    ),
    (
        ("слишком восторженно", "слишком энергично", "потише", "спокойнее"),
        "enthusiasm",
        "normal",
        "low",
    ),
    (
        ("слишком скучно", "слишком пресно", "веселее", "энергичнее"),
        "enthusiasm",
        "high",
        "normal",
    ),
    (
        ("слишком много списков", "меньше списков", "без списков", "убери списки"),
        "headings_lists",
        "low",
        "low",
    ),
    (
        ("больше списков", "структурируй", "по пунктам", "добавь списки"),
        "headings_lists",
        "high",
        "high",
    ),
    (
        ("слишком длинно", "короче", "покороче", "много текста"),
        "brevity",
        "short",
        "short",
    ),
    (
        ("слишком коротко", "подробнее", "распиши", "детальнее"),
        "brevity",
        "detailed",
        "detailed",
    ),
    (
        ("слишком формально", "попроще", "расслабься", "не будь роботом"),
        "formality",
        "casual",
        "friendly",
    ),
    (
        ("слишком развязно", "серьёзнее", "формальнее", "будь строже"),
        "formality",
        "formal",
        "formal",
    ),
    # ── Новые тона: assertive / rebellious ──
    (
        (
            "будь увереннее",
            "настойчивее",
            "жёстче",
            "assertive",
            "отстаивай мнение",
            "не соглашайся",
            "спорь со мной",
        ),
        "base_tone",
        "assertive",
        "assertive",
    ),
    (
        (
            "будь дерзким",
            "бунтуй",
            "rebellious",
            "восстань",
            "провоцируй",
            "дерзко",
            "бунтарский",
        ),
        "base_tone",
        "rebellious",
        "rebellious",
    ),
]

# Маппинг NL-инструкций в поля persona (старая система)
INSTRUCTION_MAP = {
    "short": (
        ("короч", "кратк", "покороч", "лаконичн", "сократи"),
        {"brevity": "short"},
    ),
    "detailed": (
        ("подробн", "развёрнут", "детальн", "распиши"),
        {"brevity": "detailed"},
    ),
    "formal": (
        ("формальн", "официальн", "серьёзн", "строг"),
        {"formality": "formal"},
    ),
    "friendly": (
        ("дружелюбн", "прощ", "веселе", "полегч"),
        {"formality": "friendly"},
    ),
    "no_emoji": (
        (
            "без смайл",
            "без эмодз",
            "убери смайлы",
            "не используй смайл",
            "не ставь смайл",
        ),
        {"emoji_usage": "none"},
    ),
    "more_emoji": (
        ("больше смайл", "добавь смайл", "эмодзи"),
        {"emoji_usage": "rich"},
    ),
    "proactive": (
        ("инициатив", "предлаг", "сам решай", "будь актив"),
        {"initiative": "proactive"},
    ),
    "reactive": (
        ("не лез", "не предлаг", "только когда спрашива", "будь пассив"),
        {"initiative": "reactive"},
    ),
    "bullets": (
        ("списк", "пункт", "буллит", "через маркер"),
        {"preferred_format": "bullets"},
    ),
    "numbered": (
        ("цифр", "нумер", "по порядку"),
        {"preferred_format": "numbered"},
    ),
    "focus": (
        ("фокус", "не отвлекай", "работаю", "занят"),
        {"work_mode": "focus"},
    ),
    "relax": (
        ("отдых", "расслаб", "отдыхаю", "релакс"),
        {"work_mode": "relax"},
    ),
}

LEVEL_PROMPTS = {
    "warmth": {
        "low": "будь сдержанным и нейтральным",
        "normal": "",
        "high": "будь очень тёплым и душевным",
    },
    "enthusiasm": {
        "low": "будь спокойным и лаконичным",
        "normal": "",
        "high": "будь энергичным и восторженным",
    },
    "headings_lists": {
        "low": "избегай маркированных списков и заголовков, пиши сплошным текстом",
        "normal": "",
        "high": "активно используй заголовки и маркированные списки для структуры",
    },
    "emoji_level": {
        "low": "не используй эмодзи совсем",
        "normal": "",
        "high": "используй много эмодзи для выразительности 😊✨🔥",
    },
}


async def detect_persona_change(user_text: str) -> dict | None:
    """Распознаёт ВСЕ изменения persona в тексте (не только первое).

    Returns:
        {"changes": dict, "auto_apply": bool, "reason": str} или None
    """
    t = user_text.lower()
    merged_changes: dict = {}
    reasons: list[str] = []

    for name, (triggers, changes) in INSTRUCTION_MAP.items():
        for trigger in triggers:
            if trigger in t:
                merged_changes.update(changes)
                reasons.append(name)
                break  # один триггер на категорию — переходим к следующей

    if not merged_changes:
        return None

    return {
        "changes": merged_changes,
        "auto_apply": True,
        "reason": ", ".join(reasons),
    }


async def apply_persona_changes(telegram_id: int, changes: dict):
    """Применяет изменения к persona."""

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)

        # SELECT ... FOR UPDATE — пессимистичная блокировка строки
        # предотвращает read-modify-write race при параллельной адаптации
        from sqlalchemy import select as sa_select
        from src.db.models._learning import AdaptivePersona

        stmt = (
            sa_select(AdaptivePersona)
            .where(AdaptivePersona.user_id == owner.id)
            .with_for_update()
        )
        result = await session.execute(stmt)
        p = result.scalar_one_or_none()

        if p is None:
            return None

        # Apply the requested changes to persona in DB
        if changes:
            await update_persona(session, p, **changes)

    rules = []
    if p.brevity == "short":
        rules.append("отвечай коротко (1-2 предложения)")
    elif p.brevity == "detailed":
        rules.append("отвечай подробно")
    if p.formality == "formal":
        rules.append("формальный тон, на «вы»")
    elif p.formality == "casual":
        rules.append("очень неформально, с юмором")
    if p.initiative == "proactive":
        rules.append("проявляй инициативу — предлагай, напоминай, спрашивай")
    elif p.initiative == "reactive":
        rules.append("только отвечай на вопросы, не предлагай сам")
    if p.preferred_format == "bullets":
        rules.append("форматируй списком")
    elif p.preferred_format == "numbered":
        rules.append("нумеруй пункты")
    if p.max_response_len:
        rules.append(f"ответ не длиннее {p.max_response_len} символов")
    if p.work_mode == "focus":
        rules.append("режим фокуса — не отвлекай, только срочное")
    elif p.work_mode == "relax":
        rules.append("режим отдыха — только приятное общение")

    # -- Новые поля личности (ChatGPT-style) --

    # Базовый тон
    if p.base_tone and p.base_tone != "default":
        tone_prompt = BASE_TONE_PROMPTS.get(p.base_tone, "")
        if tone_prompt:
            rules.append(tone_prompt)

    # Теплота
    if p.warmth and p.warmth != "normal":
        warmth_text = LEVEL_PROMPTS["warmth"].get(p.warmth, "")
        if warmth_text:
            rules.append(warmth_text)

    # Энтузиазм
    if p.enthusiasm and p.enthusiasm != "normal":
        enthusiasm_text = LEVEL_PROMPTS["enthusiasm"].get(p.enthusiasm, "")
        if enthusiasm_text:
            rules.append(enthusiasm_text)

    # Заголовки/списки
    if p.headings_lists and p.headings_lists != "normal":
        hl_text = LEVEL_PROMPTS["headings_lists"].get(p.headings_lists, "")
        if hl_text:
            rules.append(hl_text)

    # Эмодзи: новое поле emoji_level имеет приоритет над старым emoji_usage
    if p.emoji_level and p.emoji_level != "normal":
        emoji_text = LEVEL_PROMPTS["emoji_level"].get(p.emoji_level, "")
        if emoji_text:
            rules.append(emoji_text)
    else:
        # Старое поле — только если emoji_level не переопределён
        if p.emoji_usage == "none":
            rules.append("НЕ используй эмодзи")
        elif p.emoji_usage == "minimal":
            rules.append("минимум эмодзи")
        elif p.emoji_usage == "rich":
            rules.append("используй больше эмодзи")

    # Обращение
    if p.alias:
        rules.append(f"обращайся ко мне «{p.alias}»")

    # Пользовательские инструкции (свободный текст)
    if p.custom_instructions:
        rules.append(f"ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ ВЛАДЕЛЬЦА:\n{p.custom_instructions}")

    if not rules:
        result = ""
    else:
        result = "\n\n## ТВОЙ СТИЛЬ ОБЩЕНИЯ (установлен владельцем):\n" + "\n".join(
            f"- {r}" for r in rules
        )

    from src.core.context_cache import put as cache_put

    await cache_put(f"persona:{telegram_id}", result, ttl=5)
    return result


# -- Адаптивная коррекция persona на основе обратной связи --


def _make_snapshot(persona) -> str:
    """Создаёт JSON-снапшот базовых настроек persona для возможности сброса."""
    snapshot = {
        "base_tone": persona.base_tone,
        "warmth": persona.warmth,
        "enthusiasm": persona.enthusiasm,
        "headings_lists": persona.headings_lists,
        "emoji_level": persona.emoji_level,
        "custom_instructions": persona.custom_instructions,
        "alias": persona.alias,
    }
    return json.dumps(snapshot, ensure_ascii=False)


async def reset_persona_to_snapshot(telegram_id: int) -> bool:
    """Сбрасывает persona к базовому снапшоту. Возвращает True если сброс выполнен."""
    from src.core.context_cache import invalidate as cache_invalidate

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

        if not p.base_snapshot_json:
            return False

        try:
            snapshot = json.loads(p.base_snapshot_json)
        except Exception:
            return False

        for field, value in snapshot.items():
            if hasattr(p, field):
                setattr(p, field, value)

        p.base_snapshot_json = None  # Снапшот использован
        await session.commit()

        await cache_invalidate(f"persona:{telegram_id}")

        return True


async def adapt_persona_from_feedback(
    telegram_id: int, feedback_text: str
) -> dict | None:
    """
    Анализирует обратную связь пользователя и плавно корректирует persona.

    Возвращает словарь с изменениями или None если ничего не изменено.
    Работает только при adaptive_mode_enabled=True.
    """
    from src.core.context_cache import invalidate as cache_invalidate

    text = feedback_text.lower().strip()

    adjustments = {}

    # Тон
    if any(w in text for w in ["серьёзнее", "официальнее", "формальнее", "строже"]):
        adjustments["base_tone"] = "professional"
    elif any(w in text for w in ["дружелюбнее", "проще", "теплее", "мягче"]):
        adjustments["base_tone"] = "friendly"
    elif any(w in text for w in ["короче", "быстрее", "лаконичнее", "без воды"]):
        adjustments["base_tone"] = "efficient"
    elif any(w in text for w in ["веселее", "игривее", "креативнее", "шутливее"]):
        adjustments["base_tone"] = "whimsical"
    elif any(
        w in text for w in ["увереннее", "настойчивее", "жёстче", "assertive", "спорь"]
    ):
        adjustments["base_tone"] = "assertive"
    elif any(
        w in text
        for w in ["дерзко", "бунтарски", "rebellious", "провокационно", "восстань"]
    ):
        adjustments["base_tone"] = "rebellious"

    # Теплота
    if any(w in text for w in ["теплее", "душевнее", "ближе"]):
        adjustments["warmth"] = "high"
    elif any(w in text for w in ["холоднее", "отстранённее", "нейтральнее"]):
        adjustments["warmth"] = "low"

    # Энтузиазм
    if any(w in text for w in ["энергичнее", "бодрее", "активнее", "восторженнее"]):
        adjustments["enthusiasm"] = "high"
    elif any(w in text for w in ["спокойнее", "тише", "медленнее"]):
        adjustments["enthusiasm"] = "low"

    # Эмодзи
    if any(
        w in text
        for w in ["меньше эмодзи", "меньше смайлов", "без эмодзи", "без смайлов"]
    ):
        adjustments["emoji_level"] = "low"
    elif any(w in text for w in ["больше эмодзи", "больше смайлов", "добавь эмодзи"]):
        adjustments["emoji_level"] = "high"

    # Заголовки/списки
    if any(w in text for w in ["больше списков", "структурируй", "форматируй"]):
        adjustments["headings_lists"] = "high"
    elif any(w in text for w in ["меньше списков", "без списков", "сплошным текстом"]):
        adjustments["headings_lists"] = "low"

    if not adjustments:
        return None

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

        if not p.adaptive_mode_enabled:
            return None  # Адаптивный режим выключен

        # Сохраняем снапшот ДО изменений (чтобы сброс работал корректно)
        if not p.base_snapshot_json:
            p.base_snapshot_json = _make_snapshot(p)

        changes = {}
        for field, value in adjustments.items():
            old = getattr(p, field)
            if old != value:
                setattr(p, field, value)
                changes[field] = {"old": old, "new": value}

        if changes:
            p.total_corrections = (p.total_corrections or 0) + 1
            await session.commit()
            await cache_invalidate(f"persona:{telegram_id}")

        return changes


# ═══════════════════════════════════════════════════════════════════════════════
# Умный адаптивный режим — анализ настроения и ситуации
# ═══════════════════════════════════════════════════════════════════════════════

# ── Маппинг: 22 настроения → коррекция стиля ──
# Каждое даёт мягкую коррекцию (±1 шаг от текущего)
MOOD_ADAPTATIONS: dict[str, dict[str, str]] = {
    # ── Негативные ──
    "angry": {
        "warmth": "low",
        "enthusiasm": "low",
        "emoji_level": "low",
        "base_tone": "professional",
        "brevity": "short",
    },
    "frustrated": {
        "warmth": "normal",
        "enthusiasm": "low",
        "emoji_level": "low",
    },
    "sad": {
        "warmth": "high",
        "enthusiasm": "low",
        "base_tone": "friendly",
    },
    "anxious": {
        "warmth": "high",
        "enthusiasm": "low",
        "emoji_level": "low",
        "base_tone": "efficient",
        "headings_lists": "high",
        "brevity": "short",
    },
    "hurt": {
        "warmth": "high",
        "enthusiasm": "low",
        "base_tone": "friendly",
        "emoji_level": "low",
    },
    "disappointed": {
        "warmth": "normal",
        "enthusiasm": "low",
        "emoji_level": "low",
    },
    "overwhelmed": {
        "base_tone": "efficient",
        "enthusiasm": "low",
        "headings_lists": "high",
        "brevity": "short",
        "emoji_level": "low",
    },
    # ── Позитивные ──
    "happy": {
        "warmth": "high",
        "enthusiasm": "high",
        "emoji_level": "high",
        "base_tone": "friendly",
    },
    "excited": {
        "enthusiasm": "high",
        "emoji_level": "high",
        "base_tone": "whimsical",
    },
    "grateful": {
        "warmth": "high",
        "base_tone": "friendly",
        "emoji_level": "high",
    },
    "proud": {
        "enthusiasm": "high",
        "emoji_level": "high",
        "base_tone": "whimsical",
    },
    "loving": {
        "warmth": "high",
        "emoji_level": "high",
        "base_tone": "friendly",
    },
    "relieved": {
        "warmth": "high",
        "enthusiasm": "normal",
        "base_tone": "friendly",
    },
    "playful": {
        "base_tone": "whimsical",
        "enthusiasm": "high",
        "emoji_level": "high",
        "warmth": "high",
    },
    # ── Нейтрально-деловые ──
    "stressed": {
        "warmth": "normal",
        "enthusiasm": "low",
        "base_tone": "efficient",
        "headings_lists": "high",
        "brevity": "short",
    },
    "tired": {
        "warmth": "high",
        "enthusiasm": "low",
        "emoji_level": "low",
        "brevity": "short",
    },
    "urgent": {
        "base_tone": "efficient",
        "enthusiasm": "low",
        "headings_lists": "high",
        "brevity": "short",
    },
    "casual": {
        "base_tone": "friendly",
        "warmth": "high",
        "emoji_level": "high",
    },
    "formal": {
        "base_tone": "professional",
        "warmth": "low",
        "enthusiasm": "low",
        "emoji_level": "low",
        "headings_lists": "high",
    },
    "curious": {
        "enthusiasm": "high",
        "emoji_level": "normal",
        "headings_lists": "high",
        "base_tone": "friendly",
    },
    "skeptical": {
        "base_tone": "cynical",
        "warmth": "low",
        "enthusiasm": "low",
        "emoji_level": "low",
    },
    "determined": {
        "base_tone": "efficient",
        "enthusiasm": "high",
        "headings_lists": "high",
        "brevity": "short",
    },
    # ── Новые: характер / бунт ──
    "argumentative": {
        "base_tone": "assertive",
        "warmth": "low",
        "enthusiasm": "high",
        "emoji_level": "low",
    },
    "rebellious": {
        "base_tone": "rebellious",
        "warmth": "low",
        "enthusiasm": "high",
        "emoji_level": "normal",
    },
}

# ── Расширенные ключевые слова (200+ триггеров) ──
MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    # ── Негативные ──
    "angry": (
        "бесит",
        "злюсь",
        "ярость",
        "ненавижу",
        "иди нах",
        "заколебал",
        "выбесил",
        "взбесил",
        "гнев",
        "в ярости",
        "бешенство",
        "разъярён",
        "злой",
        "зла",
        "зол",
        "в гневе",
        "рвёт и мечет",
        "крышу сносит",
        "убил бы",
        "меня триггерит",
        "просто пздц",
        "охренеть",
    ),
    "frustrated": (
        "блин",
        "заколебало",
        "достало",
        "не получается",
        "опять",
        "снова",
        "задолбал",
        "как же задолбало",
        "руки опускаются",
        "надоело",
        "достал",
        "замучил",
        "устал от этого",
        "сколько можно",
        "опять двадцать пять",
        "не выходит",
        "никак не могу",
        "бьюсь как рыба об лёд",
        "тщетно",
        "бесполезно",
        "всё зря",
        "опускаются руки",
        "нет прогресса",
    ),
    "sad": (
        "грустно",
        "тоскливо",
        "печаль",
        "одиноко",
        "плохо",
        "депресс",
        "тоска",
        "хреново",
        "фигово",
        "плакать хочется",
        "на душе",
        "тяжело",
        "невыносимо",
        "больно",
        "горько",
        "уныло",
        "уныние",
        "поник",
        "расстроен",
        "расстроился",
        "раскис",
        "хандрю",
        "хандра",
        "не в духе",
        "всё плохо",
        "ничего не радует",
        "нет настроения",
    ),
    "anxious": (
        "волнуюсь",
        "переживаю",
        "страшно",
        "боюсь",
        "тревога",
        "тревожно",
        "нервничаю",
        "мандражирую",
        "не по себе",
        "жутко",
        "беспокоит",
        "волнение",
        "паника",
        "паникую",
        "мне страшно",
        "боязно",
        "опасаюсь",
        "мурашки",
        "сердце колотится",
        "не могу успокоиться",
        "дрожу",
        "как бы чего не вышло",
        "перестраховываюсь",
    ),
    "hurt": (
        "обидно",
        "обиделся",
        "обида",
        "задело",
        "ранило",
        "больно слышать",
        "не ожидал",
        "разочаровал",
        "предал",
        "как ты мог",
        "не прощу",
        "жестоко",
        "несправедливо",
        "почему со мной так",
        "я не заслужил",
    ),
    "disappointed": (
        "разочарован",
        "ждал большего",
        "не оправдал",
        "ожидал",
        "надеялся",
        "облом",
        "обломали",
        "слил",
        "провал",
        "не получилось",
        "фиаско",
        "крушение надежд",
    ),
    "overwhelmed": (
        "завал",
        "тону",
        "зашиваюсь",
        "не успеваю",
        "перегруз",
        "много всего",
        "всё сразу",
        "разрываюсь",
        "захлёбываюсь",
        "не справляюсь",
        "слишком много",
        "перебор",
        "цейтнот",
        "горю по всем фронтам",
        "не знаю за что хвататься",
    ),
    # ── Позитивные ──
    "happy": (
        "круто",
        "супер",
        "отлично",
        "класс",
        "рад",
        "рада",
        "счастье",
        "ура",
        "праздник",
        "кайф",
        "здорово",
        "прекрасно",
        "замечательно",
        "великолепно",
        "чудесно",
        "восхитительно",
        "обалденно",
        "шик",
        "блеск",
        "топ",
        "лучший день",
        "всё супер",
        "жизнь прекрасна",
        "доволен",
        "довольна",
        "радость",
        "счастлив",
        "счастлива",
    ),
    "excited": (
        "вау",
        "огонь",
        "пушка",
        "бомба",
        "в восторге",
        "не могу поверить",
        "офигеть",
        "потрясающе",
        "с ума сойти",
        "обалдеть",
        "ахренеть",
        "крышесносно",
        "разрыв",
        "это нечто",
        "фантастика",
        "мечты сбываются",
        "нереально",
        "космос",
        "в шоке",
        "без ума",
    ),
    "grateful": (
        "спасибо",
        "благодарен",
        "благодарна",
        "ценю",
        "признателен",
        "выручил",
        "помог",
        "спас",
        "должник",
        "низкий поклон",
        "огромное спасибо",
        "благодарю",
        "тронут",
        "растроган",
        "не ожидал такой поддержки",
        "очень помог",
    ),
    "proud": (
        "горжусь",
        "горд",
        "горда",
        "достиг",
        "получилось",
        "сделал это",
        "победа",
        "успех",
        "красавчик",
        "получил",
        "добился",
        "справился",
        "смог",
        "смогла",
        "мой проект",
        "моя работа",
        "моё творение",
    ),
    "loving": (
        "люблю",
        "обожаю",
        "милый",
        "родной",
        "дорогой",
        "солнце",
        "зайка",
        "котик",
        "сладкий",
        "нежный",
        "скучаю",
        "обнимаю",
        "целую",
        "сердечко",
        "моя",
        "мой",
        "любимый",
        "любимая",
        "душа моя",
    ),
    "relieved": (
        "отлегло",
        "полегчало",
        "отпустило",
        "слава богу",
        "пронесло",
        "обошлось",
        "сбросил груз",
        "гора с плеч",
        "выдохнул",
        "наконец-то",
        "дождался",
        "свершилось",
        "можно расслабиться",
        "пережил",
    ),
    "playful": (
        "хаха",
        "ахах",
        "ржака",
        "угар",
        "прикол",
        "шутка",
        "смешно",
        "лол",
        "кек",
        "орать",
        "ухохатываюсь",
        "до слёз",
        "поржал",
        "весело",
        "забавно",
        "хохма",
    ),
    # ── Нейтрально-деловые ──
    "stressed": (
        "стресс",
        "дедлайн",
        "горю",
        "много задач",
        "зашиваюсь",
        "не продохнуть",
        "цейтнот",
        "аврал",
        "запарка",
        "загружен",
        "загружена",
        "пашу",
        "в мыле",
        "работаю без выходных",
        "переработка",
        "выгораю",
        "выгорел",
        "на пределе",
        "последние силы",
    ),
    "tired": (
        "устал",
        "устала",
        "вымотан",
        "нет сил",
        "спать хочу",
        "сонный",
        "утомился",
        "измотан",
        "выжат",
        "замучен",
        "еле живой",
        "никакой",
        "разбит",
        "обессилен",
        "глаза закрываются",
        "зомби",
        "выключился",
        "отрубился",
        "пора баиньки",
        "спатки",
        "на боковую",
    ),
    "urgent": (
        "срочно",
        "быстро",
        "горит",
        "пожар",
        "аврал",
        "сейчас",
        "немедленно",
        "прямо сейчас",
        "не ждёт",
        "срочняк",
        "атас",
        "тревога",
        "красный код",
        "надо вчера",
        "время не терпит",
        "каждая минута дорога",
        "вынь да полож",
        "как можно скорее",
        "asap",
    ),
    "casual": (
        "привет",
        "как дела",
        "чё как",
        "здарова",
        "поболтать",
        "расскажи",
        "слушай",
        "кстати",
        "прикол",
        "шутка",
        "чё нового",
        "как жизнь",
        "как сам",
        "давно не виделись",
        "соскучился",
        "чё делаешь",
        "планы",
        "го",
        "погнали",
        "давай спишемся",
        "созвон",
        "на связи",
    ),
    "formal": (
        "коллеги",
        "отчёт",
        "совещание",
        "презентация",
        "клиент",
        "заказчик",
        "договор",
        "контракт",
        "митинг",
        "планёрка",
        "господин",
        "уважаемый",
        "входящее",
        "исходящее",
        "служебная записка",
        "докладная",
        "регламент",
        "протокол",
        "повестка",
        "кворум",
        "утвердить",
        "согласовать",
    ),
    "curious": (
        "интересно",
        "любопытно",
        "расскажи подробнее",
        "а что если",
        "как это работает",
        "почему",
        "зачем",
        "откуда",
        "хочу узнать",
        "объясни",
        "поясни",
        "разжуй",
        "дай ссылку",
        "где почитать",
        "исследую",
        "изучаю",
    ),
    "skeptical": (
        "не верю",
        "сомневаюсь",
        "вряд ли",
        "чушь",
        "бред",
        "ерунда",
        "фигня",
        "лажа",
        "развод",
        "обман",
        "не катит",
        "сомнительно",
        "подозрительно",
        "ну-ну",
        "рассказывай",
        "ага конечно",
        "да ладно",
    ),
    "determined": (
        "надо сделать",
        "сделаю",
        "добьюсь",
        "достигну",
        "любой ценой",
        "во что бы то ни стало",
        "прорвёмся",
        "справлюсь",
        "решу",
        "разберусь",
        "настроен",
        "готов",
        "приступаю",
        "погнали",
        "за дело",
        "в бой",
        "не отступлю",
        "доведу до конца",
        "финишная прямая",
    ),
    # ── Новые настроения ──
    "argumentative": (
        "спорь",
        "докажи",
        "аргументируй",
        "убеди меня",
        "отстаивай",
        "возрази",
        "не соглашайся",
        "дискуссия",
        "дебаты",
        "контраргумент",
        "обоснуй",
        "почему ты так считаешь",
        "докажи обратное",
        "переубеди меня",
    ),
    "rebellious": (
        "бунт",
        "восстание",
        "протест",
        "бунтарь",
        "революция",
        "против системы",
        "бей бей",
        "не согласен",
        "не подчиняйся",
        "бунтарский дух",
        "расскажи что думаешь",
        "своё мнение",
        "правда в глаза",
    ),
}


# ── Дополнительные сигналы из формы сообщения ──
def _detect_style_signals(text: str) -> dict[str, str | None]:
    """Извлекает сигналы из формы сообщения: капс, длина, пунктуация."""
    t = text.strip()
    signals: dict[str, str | None] = {}

    # Длина сообщения
    if len(t) < 10:
        signals["brevity_hint"] = "short"
    elif len(t) > 500:
        signals["brevity_hint"] = "detailed"

    # КРИК (CAPS > 60% букв)
    alpha = [c for c in t if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.6:
        signals["caps_detected"] = "urgent"

    # Многоточия → нерешительность/усталость
    if t.count("…") >= 3 or t.count("...") >= 3:
        signals["ellipsis"] = "tired"

    # Восклицательные знаки
    excl = t.count("!") + t.count("‼")
    if excl >= 3:
        signals["exclamation"] = "excited" if "👍" in t or "🔥" in t else "angry"

    # Вопросительные знаки
    qmarks = t.count("?")
    if qmarks >= 3:
        signals["question_spam"] = "anxious"
    elif qmarks >= 1 and len(t) < 30:
        signals["question_short"] = "curious"

    # Скобки-смайлы
    smiles = sum(
        1
        for s in [")", "(", ":)", ":(", "):", "(:"]
        if s in t and not any(kw in t for kw in ["гнев", "злюсь"])
    )
    if smiles >= 2:
        signals["smileys"] = "casual"

    return signals


def _detect_mood_fast(text: str) -> str | None:
    """Быстрое определение настроения: keywords + сигналы формы сообщения."""
    t = text.lower()
    scores: dict[str, int] = {}

    # 1. Ключевые слова
    for mood, keywords in MOOD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > 0:
            scores[mood] = score

    # 2. Сигналы формы (усиливают или добавляют настроения)
    style = _detect_style_signals(text)

    # CAPS → urgent/stressed буст
    if style.get("caps_detected"):
        scores["urgent"] = scores.get("urgent", 0) + 2

    # Многоточия → tired
    if style.get("ellipsis"):
        scores["tired"] = scores.get("tired", 0) + 2

    # Много !! → excited или angry
    if style.get("exclamation") == "excited":
        scores["excited"] = scores.get("excited", 0) + 2
    elif style.get("exclamation") == "angry":
        scores["angry"] = scores.get("angry", 0) + 2

    # Много ?? → anxious
    if style.get("question_spam"):
        scores["anxious"] = scores.get("anxious", 0) + 2

    # Короткий вопрос → curious
    if style.get("question_short"):
        scores["curious"] = scores.get("curious", 0) + 1

    # Смайлы → casual
    if style.get("smileys"):
        scores["casual"] = scores.get("casual", 0) + 2

    if not scores:
        return None

    # Возвращаем настроение с максимальным счётом
    return max(scores, key=scores.get)  # type: ignore[arg-type]


async def _detect_mood_llm(text: str, provider) -> str | None:
    """LLM-анализ настроения пользователя (точный, но медленный)."""
    moods_list = ", ".join(sorted(MOOD_ADAPTATIONS.keys()))
    prompt = (
        "Проанализируй настроение пользователя по сообщению. "
        "Обрати внимание на: эмоциональный окрас, длину сообщения, "
        "пунктуацию, использование заглавных букв, сленг.\n"
        f"Ответь ОДНИМ словом из списка: {moods_list}, neutral.\n\n"
        f'Сообщение: "{text}"\n\nНастроение:'
    )
    try:
        resp = await provider.chat([ChatMessage(role="user", content=prompt)])
        mood = resp.strip().lower().rstrip(".")
        if mood in MOOD_ADAPTATIONS:
            return mood
    except Exception:
        logger.debug("LLM mood detection failed", exc_info=True)
    return None


async def analyze_user_mood(
    telegram_id: int, user_text: str, provider=None
) -> str | None:
    """
    Определяет настроение пользователя по тексту.

    Двухуровневый анализ:
    1. Быстрый: ключевые слова (без LLM)
    2. Точный: LLM (если есть провайдер и keyword-анализ не дал однозначного результата)

    Возвращает: angry/frustrated/sad/happy/stressed/excited/tired/urgent/casual/formal/neutral/None
    """
    # Уровень 1: быстрый keyword-анализ
    mood = _detect_mood_fast(user_text)
    if mood is not None:
        return mood

    # Уровень 2: LLM-анализ (только если есть провайдер)
    if provider is not None:
        mood = await _detect_mood_llm(user_text, provider)
        if mood is not None:
            return mood

    return None


async def auto_adapt_from_context(
    telegram_id: int,
    user_text: str,
    provider=None,
) -> dict | None:
    """
    Автоматическая адаптация стиля на основе настроения и контекста.

    Вызывается на КАЖДОЕ сообщение (если adaptive_mode_enabled).
    Не ждёт явной команды «измени стиль» — анализирует настроение и
    плавно корректирует persona.

    Возвращает словарь с изменениями или None.
    """
    from src.core.context_cache import invalidate as cache_invalidate

    # 1. Быстрая проверка: включён ли адаптивный режим
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)
        if not p.adaptive_mode_enabled:
            return None

    # 2. Анализ настроения
    mood = await analyze_user_mood(telegram_id, user_text, provider)

    # 3. Анализ контакта: кому пишет пользователь?
    contact_name = _detect_contact_name(user_text)
    contact_overrides = None
    if contact_name:
        contact_overrides = await get_contact_persona_override(
            telegram_id, contact_name
        )
        if contact_overrides:
            logger.debug(
                "Contact detected: %s → overrides=%s", contact_name, contact_overrides
            )

    # 4. Объединяем коррекции: контакт > настроение
    mood_overrides = MOOD_ADAPTATIONS.get(mood) if mood else None
    target = await _merge_persona_overrides(mood_overrides, contact_overrides)

    if not target:
        return None

    # 5. Проверяем явную обратную связь (тоже через текст)
    #    чтобы не конфликтовать с adapt_persona_from_feedback
    from src.core.context_cache import get as cache_get

    # Защита от слишком частых изменений:
    # - mood-only: 120 сек
    # - contact-based: 30 сек (контакт важнее)
    last_adapt_key = f"adapt_ts:{telegram_id}"
    last_ts = await cache_get(last_adapt_key)
    now = __import__("time").monotonic()
    cooldown = 30 if contact_overrides else 120
    if last_ts is not None and (now - last_ts) < cooldown:
        return None

    # 6. Применяем изменения
    from sqlalchemy import select as sa_select
    from src.db.models._learning import AdaptivePersona

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id, use_cache=False)
        stmt = (
            sa_select(AdaptivePersona)
            .where(AdaptivePersona.user_id == owner.id)
            .with_for_update()
        )
        result = await session.execute(stmt)
        p = result.scalar_one_or_none()
        if p is None or not p.adaptive_mode_enabled:
            return None

        # Сохраняем снапшот до изменений
        if not p.base_snapshot_json:
            p.base_snapshot_json = _make_snapshot(p)

        changes = {}
        for field, value in target.items():
            # Только для полей, которые есть в модели
            if not hasattr(p, field):
                continue
            old = getattr(p, field)

            # Пропускаем если уже совпадает или если поле не строковое
            if old == value:
                continue
            if not isinstance(old, str) or not isinstance(value, str):
                continue

            # Мягкая коррекция: не прыгаем, а смещаем на 1 шаг
            level_order = {"low": 0, "normal": 1, "high": 2}
            if old in level_order and value in level_order:
                old_idx = level_order[old]
                target_idx = level_order[value]
                if old_idx == target_idx:
                    continue
                step = 1 if target_idx > old_idx else -1
                new_idx = max(0, min(2, old_idx + step))
                new_val = {0: "low", 1: "normal", 2: "high"}[new_idx]
                if new_val != old:
                    setattr(p, field, new_val)
                    changes[field] = {"old": old, "new": new_val}
            else:
                # Для не-уровневых полей (base_tone) — применяем сразу
                setattr(p, field, value)
                changes[field] = {"old": old, "new": value}

        if changes:
            p.total_corrections = (p.total_corrections or 0) + 1
            await session.commit()
            await cache_invalidate(f"persona:{telegram_id}")
            from src.core.context_cache import put as cache_put

            await cache_put(last_adapt_key, now, ttl=130)

            logger.info(
                "Auto-adapt: user=%s mood=%s contact=%s changes=%s",
                telegram_id,
                mood or "none",
                contact_name or "none",
                {k: f"{v['old']}→{v['new']}" for k, v in changes.items()},
            )

        return changes if changes else None


# ═══════════════════════════════════════════════════════════════════════════════
# Per-contact адаптация — бот подстраивает стиль под конкретного собеседника
# ═══════════════════════════════════════════════════════════════════════════════

# ── Архетип контакта → коррекция persona ──
ARCHETYPE_TO_PERSONA: dict[str, dict[str, str]] = {
    "close_friend": {
        "base_tone": "friendly",
        "warmth": "high",
        "emoji_level": "high",
        "enthusiasm": "high",
    },
    "family": {
        "base_tone": "friendly",
        "warmth": "high",
        "emoji_level": "high",
        "enthusiasm": "normal",
    },
    "colleague": {
        "base_tone": "professional",
        "warmth": "low",
        "enthusiasm": "low",
        "emoji_level": "low",
    },
    "romantic": {
        "base_tone": "friendly",
        "warmth": "high",
        "emoji_level": "high",
        "enthusiasm": "high",
    },
    "acquaintance": {
        "base_tone": "professional",
        "warmth": "normal",
        "enthusiasm": "normal",
        "emoji_level": "low",
        "brevity": "short",
    },
    "toxic": {
        "base_tone": "professional",
        "warmth": "low",
        "enthusiasm": "low",
        "emoji_level": "low",
        "brevity": "short",
    },
    "boss": {
        "base_tone": "professional",
        "warmth": "low",
        "enthusiasm": "low",
        "emoji_level": "low",
        "headings_lists": "high",
    },
    "client": {
        "base_tone": "professional",
        "warmth": "normal",
        "enthusiasm": "normal",
        "emoji_level": "low",
    },
    "friend": {
        "base_tone": "friendly",
        "warmth": "high",
        "emoji_level": "high",
        "enthusiasm": "high",
    },
    "partner": {
        "base_tone": "friendly",
        "warmth": "high",
        "emoji_level": "high",
        "enthusiasm": "high",
    },
    "stranger": {
        "base_tone": "professional",
        "warmth": "normal",
        "enthusiasm": "normal",
        "emoji_level": "low",
        "brevity": "short",
    },
}

# ── Ключевые слова-маркеры отношений ──
RELATION_MARKERS: dict[str, list[str]] = {
    "boss": [
        "босс",
        "начальник",
        "шеф",
        "руководитель",
        "директор",
        "гендир",
        "управляющий",
        "глава",
        "тимлид",
    ],
    "client": ["клиент", "заказчик", "покупатель", "партнёр по бизнесу"],
    "friend": [
        "друг",
        "подруга",
        "дружище",
        "братан",
        "подруган",
        "дружбан",
        "кореш",
        "приятель",
    ],
    "family": [
        "мам",
        "пап",
        "брат",
        "сестр",
        "бабул",
        "дедул",
        "тёт",
        "дяд",
        "сын",
        "доч",
        "внук",
        "внучк",
        "родн",
    ],
    "partner": [
        "муж",
        "жен",
        "парень",
        "девушк",
        "любим",
        "любим",
        "вторая половин",
        "супруг",
    ],
    "colleague": ["коллега", "сотрудник", "соратник", "напарник"],
    "stranger": ["незнаком", "какой-то", "какая-то", "некто", "чел"],
}


def _detect_contact_name(text: str) -> str | None:
    """Извлекает имя контакта из текста сообщения.

    Ищет паттерны: «напиши X», «ответь X», «скажи X», «что там с X»,
    «как дела у X», «спроси у X», «передай X», «для X».
    Возвращает raw-имя контакта или None.
    """
    import re

    patterns = [
        r"(?:напиши|отправь|черкани|сбрось|закинь)\s+([а-яёa-z]+(?:\s+[а-яёa-z]+){0,2})",
        r"(?:ответь|отвечай)\s+([а-яёa-z]+(?:\s+[а-яёa-z]+){0,2})",
        r"(?:скажи|передай|спроси\s+у)\s+([а-яёa-z]+(?:\s+[а-яёa-z]+){0,2})",
        r"(?:что\s+там\s+(?:с|у)|как\s+дела\s+(?:с|у)|как\s+там)\s+([а-яёa-z]+(?:\s+[а-яёa-z]+){0,2})",
        r"(?:для|к)\s+([а-яёa-z]+(?:\s+[а-яёa-z]+){0,2})",
        r"(?:с)\s+([А-ЯЁA-Z][а-яёa-z]+)(?:\s|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().lower()
            # Отсекаем явно не-имена
            if name in {
                "мне",
                "себе",
                "тебе",
                "ему",
                "ей",
                "им",
                "всем",
                "туда",
                "сюда",
                "тут",
                "там",
                "это",
                "этот",
                "потом",
                "завтра",
                "сегодня",
                "уже",
                "ещё",
                "привет",
                "пока",
                "ок",
                "да",
                "нет",
            }:
                continue
            if len(name) < 2:
                continue
            return name

    return None


def _classify_contact_relation(contact_name: str) -> str | None:
    """Классифицирует контакт по имени/роли в архетип отношений."""
    name_lower = contact_name.lower()
    for archetype, markers in RELATION_MARKERS.items():
        for marker in markers:
            if marker in name_lower:
                return archetype
    return None


async def get_contact_persona_override(
    telegram_id: int, contact_name: str
) -> dict[str, str] | None:
    """
    Получает persona-коррекцию для конкретного контакта.

    Приоритет:
    1. Архетип из RELATION_MARKERS (быстрый keyword)
    2. ContactProfile.archetype из БД
    3. style_profile (как пользователь реально пишет этому контакту)
    """
    # 1. Быстрая классификация по имени/роли
    archetype = _classify_contact_relation(contact_name)
    if archetype and archetype in ARCHETYPE_TO_PERSONA:
        logger.debug("Contact %s → archetype %s (keyword)", contact_name, archetype)
        return dict(ARCHETYPE_TO_PERSONA[archetype])

    # 2. Поиск в БД: Contact + ContactProfile
    try:
        from src.db.session import get_session as db_get_session
        from src.db.repo import get_or_create_user as db_get_user
        from src.db.models._contacts import Contact
        from sqlalchemy import select as sa_select

        async with db_get_session() as session:
            owner = await db_get_user(session, telegram_id)
            # Ищем контакт по display_name (fuzzy)
            stmt = (
                sa_select(Contact)
                .where(
                    Contact.user_id == owner.id,
                    Contact.display_name.ilike(f"%{contact_name}%"),
                )
                .limit(1)
            )
            result = await session.execute(stmt)
            contact = result.scalar_one_or_none()

            if contact is None:
                return None

            # 3. Проверяем ContactProfile
            from src.db.models._contacts import ContactProfile

            stmt2 = (
                sa_select(ContactProfile)
                .where(
                    ContactProfile.user_id == owner.id,
                    ContactProfile.contact_id == contact.peer_id,
                )
                .limit(1)
            )
            result2 = await session.execute(stmt2)
            profile = result2.scalar_one_or_none()

            if profile and profile.closeness_label:
                # Маппим closeness_label → archetype
                closeness = profile.closeness_label.lower()
                if any(w in closeness for w in ["близк", "друг", "best"]):
                    return dict(ARCHETYPE_TO_PERSONA["close_friend"])
                elif any(w in closeness for w in ["семь", "родн", "family"]):
                    return dict(ARCHETYPE_TO_PERSONA["family"])
                elif any(w in closeness for w in ["коллег", "работ", "colleague"]):
                    return dict(ARCHETYPE_TO_PERSONA["colleague"])
                elif any(w in closeness for w in ["романт", "любов", "romantic"]):
                    return dict(ARCHETYPE_TO_PERSONA["romantic"])
                elif any(w in closeness for w in ["знаком", "acquaint"]):
                    return dict(ARCHETYPE_TO_PERSONA["acquaintance"])
                elif any(w in closeness for w in ["токсич", "toxic", "конфликт"]):
                    return dict(ARCHETYPE_TO_PERSONA["toxic"])

            # 4. Если есть communication_style — используем его
            if profile and profile.communication_style:
                style = profile.communication_style.lower()
                if any(w in style for w in ["формаль", "офиц", "делов"]):
                    return dict(ARCHETYPE_TO_PERSONA["colleague"])
                elif any(w in style for w in ["друже", "неформ", "разговор"]):
                    return dict(ARCHETYPE_TO_PERSONA["friend"])

    except Exception:
        logger.debug("Contact persona override lookup failed", exc_info=True)

    return None


async def _merge_persona_overrides(
    mood_overrides: dict[str, str] | None,
    contact_overrides: dict[str, str] | None,
) -> dict[str, str]:
    """
    Объединяет коррекции настроения и контакта.

    Контакт имеет приоритет над настроением: если ты пишешь боссу,
    тон должен быть профессиональным, даже если ты в игривом настроении.
    Но настроение смягчает: angry + босс = сдержанный профессионал,
    happy + босс = вежливый тёплый профессионал.
    """
    if not mood_overrides and not contact_overrides:
        return {}
    if not mood_overrides:
        return dict(contact_overrides)  # type: ignore[arg-type]
    if not contact_overrides:
        return dict(mood_overrides)

    # Контакт — база, настроение — модулятор
    merged = dict(contact_overrides)

    # Настроение может влиять на enthusiasm, emoji_level, warmth
    # но НЕ на base_tone (контакт диктует тон)
    for mood_field in ("enthusiasm", "emoji_level", "warmth", "brevity"):
        if mood_field in mood_overrides:
            # Если контакт уже задал это поле — оставляем контакт
            if mood_field not in merged:
                merged[mood_field] = mood_overrides[mood_field]

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# format_persona_for_prompt — форматирует persona в блок для system prompt
# ═══════════════════════════════════════════════════════════════════════════════

# Имена тонов на русском для читаемости
TONE_NAMES: dict[str, str] = {
    "default": "естественный",
    "professional": "профессиональный",
    "friendly": "тёплый",
    "frank": "прямой",
    "whimsical": "игривый",
    "efficient": "лаконичный",
    "cynical": "саркастичный",
    "assertive": "уверенный",
    "rebellious": "бунтарский",
}


async def format_persona_for_prompt(telegram_id: int) -> str | None:
    """Собирает блок persona для вставки в system prompt.

    Поддерживает:
    - base_tone / tone_mix (из custom_instructions)
    - experience (из custom_instructions)
    - остальные поля (brevity, formality, etc.)
    """
    from src.core.context_cache import get as cache_get
    from src.db.repo import get_or_create_user, get_persona
    from src.db.session import get_session

    # Кеш: persona обновляется не чаще раза в 5 секунд
    cached = await cache_get(f"persona:{telegram_id}")
    if cached:
        return cached

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

    rules: list[str] = []

    # --- Tone mix (из custom_instructions) ---
    try:
        if p.custom_instructions:
            ci = (
                json.loads(p.custom_instructions)
                if isinstance(p.custom_instructions, str)
                else p.custom_instructions
            )
            tone_mix = ci.get("tone_mix") if isinstance(ci, dict) else None
            experience = ci.get("experience") if isinstance(ci, dict) else None
        else:
            tone_mix = None
            experience = None
    except (json.JSONDecodeError, TypeError):
        tone_mix = None
        experience = None
        # Если custom_instructions — просто текст, используем как есть
        if p.custom_instructions and isinstance(p.custom_instructions, str):
            rules.append(p.custom_instructions)

    # Tone mix: {"assertive": 50, "friendly": 30, ...}
    if tone_mix and isinstance(tone_mix, dict) and len(tone_mix) > 1:
        mix_lines: list[str] = []
        # Сортируем по убыванию процента
        sorted_mix = sorted(tone_mix.items(), key=lambda x: x[1], reverse=True)
        for tone, pct in sorted_mix:
            name = TONE_NAMES.get(tone, tone)
            if pct >= 10:  # только значимые компоненты
                mix_lines.append(f"- {name} ({pct}%)")
        if mix_lines:
            rules.append(
                "ТВОЙ СТИЛЬ — ЭТО КОКТЕЙЛЬ ТОНОВ:\n"
                + "\n".join(mix_lines)
                + "\nКомбинируй их естественно, как живой человек с многогранным характером."
            )
    elif tone_mix and isinstance(tone_mix, dict) and len(tone_mix) == 1:
        # Всего один тон — используем как base_tone
        sole_tone = list(tone_mix.keys())[0]
        if sole_tone in BASE_TONE_PROMPTS:
            tone_prompt = BASE_TONE_PROMPTS[sole_tone]
            if tone_prompt:
                rules.append(tone_prompt)

    # --- Base tone (если нет tone_mix) ---
    if not tone_mix and p.base_tone and p.base_tone != "default":
        tone_prompt = BASE_TONE_PROMPTS.get(p.base_tone, "")
        if tone_prompt:
            rules.append(tone_prompt)

    # --- Теплота ---
    if p.warmth and p.warmth != "normal":
        warmth_text = LEVEL_PROMPTS["warmth"].get(p.warmth, "")
        if warmth_text:
            rules.append(warmth_text)

    # --- Энтузиазм ---
    if p.enthusiasm and p.enthusiasm != "normal":
        enthusiasm_text = LEVEL_PROMPTS["enthusiasm"].get(p.enthusiasm, "")
        if enthusiasm_text:
            rules.append(enthusiasm_text)

    # --- Заголовки/списки ---
    if p.headings_lists and p.headings_lists != "normal":
        hl_text = LEVEL_PROMPTS["headings_lists"].get(p.headings_lists, "")
        if hl_text:
            rules.append(hl_text)

    # --- Старые поля (brevity, formality, initiative, work_mode) ---
    if p.brevity == "short":
        rules.append("отвечай коротко (1-2 предложения)")
    elif p.brevity == "detailed":
        rules.append("отвечай подробно")
    if p.formality == "formal":
        rules.append("формальный тон, на «вы»")
    elif p.formality == "casual":
        rules.append("очень неформально, с юмором")
    if p.initiative == "proactive":
        rules.append("проявляй инициативу — предлагай, напоминай, спрашивай")
    elif p.initiative == "reactive":
        rules.append("только отвечай на вопросы, не предлагай сам")
    if p.preferred_format == "bullets":
        rules.append("форматируй списком")
    elif p.preferred_format == "numbered":
        rules.append("нумеруй пункты")

    # --- Эмодзи ---
    if p.emoji_level and p.emoji_level != "normal":
        emoji_text = LEVEL_PROMPTS["emoji_level"].get(p.emoji_level, "")
        if emoji_text:
            rules.append(emoji_text)
    else:
        if p.emoji_usage == "none":
            rules.append("НЕ используй эмодзи")
        elif p.emoji_usage == "minimal":
            rules.append("минимум эмодзи")
        elif p.emoji_usage == "rich":
            rules.append("используй больше эмодзи")

    if p.work_mode == "focus":
        rules.append("режим фокуса — не отвлекай, только срочное")
    elif p.work_mode == "relax":
        rules.append("режим отдыха — только приятное общение")
    if p.max_response_len:
        rules.append(f"ответ не длиннее {p.max_response_len} символов")
    if p.alias:
        rules.append(f"обращайся ко мне «{p.alias}»")

    # --- Experience (вывод из опыта) ---
    if experience and isinstance(experience, str) and len(experience.strip()) > 10:
        rules.append(f"ИЗ ОПЫТА ОБЩЕНИЯ:\n{experience.strip()[:500]}")

    # --- Plain text custom_instructions (если не JSON) ---
    if (
        p.custom_instructions
        and isinstance(p.custom_instructions, str)
        and not tone_mix
        and not experience
    ):
        try:
            json.loads(p.custom_instructions)
            # Это JSON — уже обработали выше
        except (json.JSONDecodeError, TypeError):
            if p.custom_instructions.strip():
                rules.append(
                    f"ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ ВЛАДЕЛЬЦА:\n{p.custom_instructions}"
                )

    if not rules:
        return None

    result = "\n\n## ТВОЙ СТИЛЬ ОБЩЕНИЯ (установлен владельцем):\n" + "\n".join(
        f"- {r}" for r in rules
    )

    # Кешируем
    from src.core.context_cache import put as cache_put

    await cache_put(f"persona:{telegram_id}", result, ttl=5)
    return result
