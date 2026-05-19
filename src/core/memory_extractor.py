"""LLM-извлечение фактов-воспоминаний о контакте из переписки."""
import json
import logging

from sqlalchemy import select

from src.core.chat_service import message_to_text
from src.db.models import Contact, Message, User
from src.db.repo import add_memory
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider


logger = logging.getLogger(__name__)


MEMORIES_SYSTEM = (
    "Ты извлекаешь факты-воспоминания о собеседнике из переписки. "
    "Факт — конкретная информация: предпочтения, события, биография, "
    "договорённости, проблемы, интересы, планы.\n\n"
    "Возвращай JSON-массив (только массив, без обёрток):\n"
    "[\n"
    '  {"fact": "краткий факт одной фразой на русском",\n'
    '   "sentiment": "positive" | "negative" | "neutral"}\n'
    "]\n"
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
    messages: list[Message],
) -> list[dict]:
    """Извлекает факты о контакте из переписки и сохраняет в БД (fire-and-forget)."""
    if not messages or contact is None:
        return []

    transcript = "\n".join(message_to_text(m) for m in messages)
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
        return []

    items = _parse_json_array(raw)
    if not items:
        return []

    saved: list[dict] = []
    async with get_session() as session:
        # Подтягиваем User по telegram_id
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user is None:
            logger.warning("Memory extraction: user %s not found", user_id)
            return []

        for item in items:
            fact = (item.get("fact") or "").strip()
            if not fact:
                continue
            sentiment = item.get("sentiment")
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = None
            await add_memory(
                session,
                user,
                fact=fact,
                contact_id=contact.peer_id if contact else None,
                sentiment=sentiment,
                source="chat",
            )
            saved.append(item)

    if saved:
        logger.info("Saved %d memories for user %d, contact %s", len(saved), user_id, contact.display_name)
    return saved
