"""LLM-извлечение обещаний из переписки в Commitment-список."""
import json
import logging
from datetime import datetime

from src.core.chat_service import message_to_text
from src.db.models import Contact, Message
from src.db.repo import add_commitment
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider


logger = logging.getLogger(__name__)


COMMITMENTS_SYSTEM = (
    "Ты выделяешь явные обязательства из переписки. Обязательство — конкретное обещание "
    "что-то сделать, прислать, ответить, прийти. Игнорируй риторические фразы.\n\n"
    "Возвращай JSON-массив (только массив, без обёрток):\n"
    "[\n"
    '  {"direction": "mine" | "theirs",\n'
    '   "message_id": <int или null>,\n'
    '   "text": "обещание одной фразой",\n'
    '   "deadline": "ISO-8601 datetime UTC или null"}\n'
    "]\n"
    "Если обязательств нет — пустой массив [].\n"
    "Не выдумывай дедлайны, если их нет в тексте."
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
        logger.warning("Commitments JSON parse failed: %r", text[:120])
        return []


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


async def extract_and_save_commitments(
    provider: LLMProvider,
    *,
    user_id: int,
    contact: Contact,
    messages: list[Message],
) -> list[dict]:
    if not messages:
        return []

    transcript = "\n".join(message_to_text(m) for m in messages)
    user_prompt = (
        f"Собеседник: {contact.display_name}.\n"
        "Переписка:\n\n"
        f"{transcript}\n\n"
        "Выдели обязательства."
    )

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=COMMITMENTS_SYSTEM),
            ChatMessage(role="user", content=user_prompt),
        ],
        heavy=False,
    )
    items = _parse_json_array(raw)
    if not items:
        return []

    saved: list[dict] = []
    async with get_session() as session:
        for item in items:
            direction = item.get("direction")
            text = (item.get("text") or "").strip()
            if direction not in {"mine", "theirs"} or not text:
                continue
            await add_commitment(
                session,
                user_id=user_id,
                peer_id=contact.peer_id,
                peer_name=contact.display_name,
                message_id=item.get("message_id"),
                direction=direction,
                text=text,
                deadline_at=_parse_iso(item.get("deadline")),
            )
            saved.append(item)
    return saved
