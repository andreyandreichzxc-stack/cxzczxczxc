"""LLM-извлечение фактов-воспоминаний о контакте из переписки.

После извлечения факты ставятся в очередь на фоновое сохранение (memory_queue),
чтобы не блокировать основной поток обработки сообщений.
"""

import json
import logging

from src.core.chat_service import message_to_text
from src.db.models import Contact, Message
from src.llm.base import ChatMessage, LLMProvider


logger = logging.getLogger(__name__)


MEMORIES_SYSTEM = (
    "Ты извлекаешь факты-воспоминания о собеседнике из переписки. "
    "Факт — конкретная информация: предпочтения, события, биография, "
    "договорённости, проблемы, интересы, планы.\n\n"
    "Возвращай JSON-массив (только массив, без обёрток):\n"
    "[\n"
    '  {"fact": "краткий факт одной фразой на русском",\n'
    '   "sentiment": "positive" | "negative" | "neutral",\n'
    '   "importance": 7,\n'
    '   "decay_rate": 0.05,\n'
    '   "relation_type": "cause" | null,\n'
    '   "relation_to_index": 0 | null}\n'
    "]\n"
    "importance (1-10):\n"
    "  1-3 — мелкая деталь, быстро забывается\n"
    "  4-7 — значимый факт, живёт недели\n"
    "  8-10 — критично (аллергии, адреса, отношения, контакты)\n"
    "decay_rate:\n"
    "  0.01 — почти не забывается (критичные факты)\n"
    "  0.07 — норма (неделя-две)\n"
    "  0.15 — быстро устаревает (настроения, планы на день)\n"
    "  0.30 — моментально (погода, «я поел»)\n"
    "Для каждого факта укажи связь с ПРЕДЫДУЩИМИ фактами из того же диалога, если она есть:\n"
    '- "relation_type": "cause" (причина), "effect" (следствие), "contradicts" (противоречие), '
    '"supports" (подтверждение), "continues" (продолжение темы), "example_of" (пример)\n'
    '- "relation_to_index": индекс предыдущего факта (0-based) в этом же ответе, с которым связан\n'
    "Если значимых фактов нет — пустой массив [].\n"
    "Не выдумывай то, чего нет в переписке. Пиши на русском."
)


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        logger.warning("Memories JSON parse failed: %r", text[:120])
        return []


async def extract_and_save_memories(
    provider: LLMProvider,
    user_id: int,
    contact: Contact | None,
    messages: list[Message] | None = None,
    transcript: str | None = None,
) -> int:
    """Извлекает факты о контакте из переписки и ставит в очередь на сохранение.

    Аргументы:
        provider — LLM-провайдер для извлечения фактов.
        user_id — User.id (внутренний ID БД).
        contact — объект Contact (нужен для display_name в промпте).
        messages — список сообщений для построения транскрипта.
        transcript — готовая текстовая расшифровка переписки (альтернатива messages).

    Возвращает количество найденных фактов.
    """
    if contact is None:
        return 0

    # Строим транскрипт из сообщений, если не передан готовый
    if transcript is None and messages:
        transcript = "\n".join(message_to_text(m) for m in messages)
    if not transcript:
        return 0

    user_prompt = (
        f"Собеседник: {contact.display_name}.\n"
        "Извлеки важные факты о собеседнике из этой переписки:\n\n"
        f"{transcript}"
    )

    try:
        raw = await provider.chat(
            [
                ChatMessage(role="system", content=MEMORIES_SYSTEM),
                ChatMessage(role="user", content=user_prompt),
            ],
            heavy=False,
        )
    except Exception:
        logger.exception("Memory extraction LLM call failed")
        return 0

    items = _parse_json_array(raw)
    if not items:
        return 0

    # --- Собираем валидные факты (пропускаем не-словари и пустые факты) ---
    valid_facts: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fact = (item.get("fact") or "").strip()
        if not fact:
            continue

        sentiment = item.get("sentiment")
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = None

        # importance 1-10 → 0.0-1.0
        raw_importance = item.get("importance")
        if isinstance(raw_importance, (int, float)):
            importance = max(0.0, min(1.0, raw_importance / 10.0))
        else:
            importance = None

        # decay_rate из LLM (0.01-0.30)
        decay_rate = item.get("decay_rate")
        if not isinstance(decay_rate, (int, float)):
            decay_rate = None

        valid_facts.append(
            {
                "fact": fact,
                "sentiment": sentiment,
                "source": "chat",
                "importance": importance,
                "decay_rate": decay_rate,
            }
        )

    if not valid_facts:
        return 0

    # --- Batch embedding — один API-вызов вместо N ---
    texts = [vf["fact"] for vf in valid_facts]
    try:
        embeddings = await provider.embed_batch(texts)
    except Exception:
        logger.warning("Failed to embed batch of %d facts", len(texts))
        embeddings = [None] * len(texts)

    # Прикрепляем эмбеддинги к фактам
    for idx, vf in enumerate(valid_facts):
        if idx < len(embeddings) and embeddings[idx] is not None:
            vf["embedding"] = embeddings[idx]

    # --- Ставим в очередь на фоновое сохранение ---
    # (lazy import — избегаем циклической зависимости на уровне модулей)
    from src.core.memory_queue import enqueue, MemoryJob

    await enqueue(
        MemoryJob(
            owner_id=user_id,
            contact_id=contact.peer_id if contact else None,
            facts=valid_facts,
            job_type="save",
        )
    )

    logger.info(
        "Extracted %d facts for user %d, contact %s (enqueued for save)",
        len(valid_facts),
        user_id,
        contact.display_name,
    )
    return len(valid_facts)
