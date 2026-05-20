import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

URGENCY_SYSTEM = """Ты — классификатор срочности входящих сообщений в Telegram.
Твоя задача: прочитай сообщение и определи его срочность.

## Категории
- **urgent** — срочное. Человек зовёт, волнуется, требует немедленного ответа, ситуация критическая. Слова-маркеры (ПРИМЕРЫ, не жёсткие правила): срочно, тревога, ты где, позвони, алло, help, SOS, ЧП, помоги, выручай, пропал, трубку не берёшь, не отвечаешь, дозвониться не могу. Много восклицательных знаков или CAPS LOCK тоже сигнал.
- **important** — важное. Обида, злость, вопрос требующий внимания, эмоциональное сообщение. Слова-маркеры (ПРИМЕРЫ): обиделась, бесит, достал, твою мать, сука, заколебал, почему, что случилось, серьёзно?
- **normal** — обычное. Болтовня, мемы, новости, приветствия, простая информация без эмоциональной окраски.

## Правила
1. Слова выше — ПРИМЕРЫ. Ты должен понимать СМЫСЛ сообщения, даже если таких слов там нет.
2. Если человек явно ждёт реакции/ответа прямо сейчас — это urgent.
3. Если человек эмоционален (зол, обижен, расстроен) но не требует немедленного ответа — important.
4. Если сообщение спокойное, бытовое, информационное — normal.
5. Короткие сообщения типа «ок», «ага», «понял», «+», мемы, ссылки — normal.

## Формат ответа
Верни ТОЛЬКО JSON: {"urgency": "urgent|important|normal"}
Без пояснений, без markdown."""


# Списки ключевых слов (для эвристического fallback-а)
URGENT_PATTERNS = [
    r"ты где[\?\!]*$",
    r"ты где[^a-zа-я]",
    r"ты куда пропал",
    r"ответь",
    r"отзовись",
    r"откликнись",
    r"срочно",
    r"позвони",
    r"набери",
    r"алло",
    r"алё",
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
    r"игноришь",
    r"трубку",
    r"трубка",
    r"не берёшь",
    r"не берешь",
    r"не беру",
    r"не слышу",
    r"не отвечаешь",
    r"дозвониться",
    r"ау",
    r"жив\??$",
    r"ты жив",
    r"ты там\??",
    r"приём",
    r"прием",
    r"слушаешь",
    r"слышишь",
    r"отбой",
    r"тревог",
    r"чп",
    r"ЧП",
    r"экстрен",
    r"помоги",
    r"выручай",
    r"нужна помощь",
    r"перезвони",
]
ANGRY_PATTERNS = [
    r"обиделась",
    r"обиделся",
    r"обидно",
    r"обижаешь",
    r"достало",
    r"я в ярости",
    r"бесит",
    r"бесишь",
    r"я злюсь",
    r"ты меня бесишь",
    r"задолбал",
    r"задолбала",
    r"достал",
    r"достала",
    r"твою мать",
    r"твою ж",
    r"ёб твою",
    r"еб твою",
    r"иди на",
    r"пошёл ты",
    r"пошел ты",
    r"козёл",
    r"козел",
    r"сволочь",
    r"урод",
    r"тварь",
    r"сука",
    r"сучара",
    r"заколебал",
    r"заколебала",
    r"достал уже",
    r"хватит",
    r"прекрати",
    r"отвали",
    r"отъебись",
    r"отъеб",
    r"выебыва",
    r"нахер",
    r"нахуй",
    r"похуй",
    r"пофиг",
    r"всё равно",
    r"все равно",
]
QUESTION_PATTERNS = [
    r"\?$",
    r"\?\!+$",
    r"\!\!+$",
    r"почему",
    r"зачем",
    r"когда",
    r"где",
    r"как",
    r"сколько",
    r"кто",
    r"чей",
    r"чья",
    r"чьё",
    r"чье",
    r"куда",
    r"откуда",
    r"зачем",
    r"чего",
    r"че",
    r"что случилось",
    r"что произошло",
    r"что за",
    r"в смысле",
    r"серьёзно",
    r"серьезно",
    r"правда",
    r"точно",
]


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


async def classify_message_llm(
    provider, text: str, sender_name: str | None = None
) -> Literal["urgent", "important", "normal"]:
    """Классифицирует срочность через LLM. Понимает смысл, а не только ключевые слова."""
    from src.llm.base import ChatMessage

    sender_info = f"Отправитель: {sender_name}.\n" if sender_name else ""
    user_msg = f"{sender_info}Сообщение: {text}"

    try:
        raw = await provider.chat(
            [
                ChatMessage(role="system", content=URGENCY_SYSTEM),
                ChatMessage(role="user", content=user_msg),
            ],
            heavy=False,
        )
        # Очистка от markdown-обёрток
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("\n", 1)[0]
        if raw.startswith("```"):
            raw = raw.strip("`")
        data = json.loads(raw)
        urgency = data.get("urgency", "normal")
        if urgency in ("urgent", "important", "normal"):
            return urgency
        return "normal"
    except Exception:
        logger.debug("LLM urgency classification failed, using fallback")
        return classify_message(text)


async def classify_urgency(
    text: str,
    provider=None,
    sender_name: str | None = None,
) -> Literal["urgent", "important", "normal"]:
    """Единая точка входа: LLM если есть провайдер, иначе эвристика."""
    if provider is not None:
        return await classify_message_llm(provider, text, sender_name)
    return classify_message(text)
