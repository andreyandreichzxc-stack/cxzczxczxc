"""Индексация сообщений в Qdrant — запускается командой /index."""
import logging
from datetime import datetime

from sqlalchemy import select, update

from src.core.actions.vector_store import vector_store
from src.db.models import Contact, Message, User
from src.db.session import get_session
from src.llm.base import LLMProvider


logger = logging.getLogger(__name__)


def _msg_text_for_embed(m: Message) -> str:
    body = m.transcript or m.text or m.extracted_text or ""
    return body.strip()


async def index_chat(
    provider: LLMProvider,
    user: User,
    contact: Contact,
    *,
    batch_limit: int = 200,
) -> int:
    indexed = 0
    while True:
        async with get_session() as session:
            result = await session.execute(
                select(Message)
                .where(
                    Message.user_id == user.id,
                    Message.peer_id == contact.peer_id,
                    Message.indexed_in_vector.is_(False),
                )
                .order_by(Message.date.asc())
                .limit(batch_limit)
            )
            batch = list(result.scalars().all())

        if not batch:
            break

        ids_done: list[int] = []
        for m in batch:
            text = _msg_text_for_embed(m)
            if not text:
                ids_done.append(m.id)
                continue
            try:
                vec = await provider.embed(text)
            except Exception:
                logger.exception("embed failed for message %s", m.id)
                continue

            await vector_store.upsert(
                user_id=user.id,
                peer_id=contact.peer_id,
                peer_name=contact.display_name,
                message_id=m.message_id,
                text=text[:2000],
                date_iso=m.date.isoformat() if m.date else None,
                embedding=vec,
            )
            ids_done.append(m.id)
            indexed += 1

        if ids_done:
            async with get_session() as session:
                await session.execute(
                    update(Message)
                    .where(Message.id.in_(ids_done))
                    .values(indexed_in_vector=True)
                )

        if len(batch) < batch_limit:
            break

    return indexed
