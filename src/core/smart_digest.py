"""Smart Incoming Digest: собирает входящие сообщения за интервал
и отправляет единую нотификацию, сгруппированную по срочности."""

import asyncio
import json
import logging
from datetime import datetime, timedelta

from sqlalchemy import select

from src.config import settings as app_settings
from src.core.notifier import notifier
from src.core.urgency_classifier import classify_message
from src.db.models import Message, User
from src.db.repo import get_contact, get_or_create_user
from src.db.session import get_session


logger = logging.getLogger(__name__)


async def collect_recent_messages(
    session,
    user: User,
    since_minutes: int = 30,
) -> dict[int, dict]:
    """SELECT входящих сообщений за последние since_minutes минут,
    группировка по peer_id.

    Возвращает {peer_id: {"sender_name": str, "last_text": str,
                           "count": int, "urgency": str}}
    """
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    result = await session.execute(
        select(Message)
        .where(
            Message.user_id == user.id,
            Message.is_outgoing.is_(False),
            Message.date >= since,
        )
        .order_by(Message.date.desc())
    )
    rows = list(result.scalars().all())

    by_peer: dict[int, dict] = {}
    for m in rows:
        if m.peer_id not in by_peer:
            # Folder filter: если monitor_only_selected_folders и контакт не в выбранных папках — пропустить
            if (
                user.settings.monitor_only_selected_folders
                and user.settings.monitored_folders
            ):
                monitored = json.loads(user.settings.monitored_folders)
                if monitored:
                    contact = await get_contact(session, user, m.peer_id)
                    contact_folders = (
                        (contact.folder_names or "").split(",") if contact else []
                    )
                    contact_folders = [f.strip() for f in contact_folders if f.strip()]
                    if not any(f in monitored for f in contact_folders):
                        continue

            text_content = m.transcript or m.text or m.extracted_text or ""
            urgency = classify_message(text_content) if text_content else "normal"
            by_peer[m.peer_id] = {
                "sender_name": m.sender_name or str(m.peer_id),
                "last_text": text_content,
                "count": 0,
                "urgency": urgency,
            }
        by_peer[m.peer_id]["count"] += 1
        # сохраняем самое свежее (первое в порядке desc)
        if by_peer[m.peer_id]["count"] == 1:
            text_content = m.transcript or m.text or m.extracted_text or ""
            by_peer[m.peer_id]["last_text"] = text_content
            by_peer[m.peer_id]["urgency"] = (
                classify_message(text_content) if text_content else "normal"
            )

    return by_peer


def build_smart_digest(messages_by_peer: dict[int, dict], interval: int) -> str:
    """Собирает HTML-текст дайджеста с группировкой по срочности."""
    if not messages_by_peer:
        return "✅ Нет новых сообщений"

    urgent: list[dict] = []
    important: list[dict] = []
    normal: list[dict] = []

    for peer_id, data in messages_by_peer.items():
        entry = {**data, "peer_id": peer_id}
        if data["urgency"] == "urgent":
            urgent.append(entry)
        elif data["urgency"] == "important":
            important.append(entry)
        else:
            normal.append(entry)

    parts = [f"📊 <b>Дайджест за {interval} минут</b>\n"]

    if urgent:
        lines = []
        for e in urgent:
            snippet = (e["last_text"][:80] or "").replace("\n", " ")
            lines.append(f"• {e['sender_name']}: «{snippet}»")
        parts.append("🔴 <b>Срочное:</b>\n" + "\n".join(lines))

    if important:
        lines = []
        for e in important:
            snippet = (e["last_text"][:80] or "").replace("\n", " ")
            count = f" ({e['count']} сообщений)" if e["count"] > 1 else ""
            lines.append(f"• {e['sender_name']}: «{snippet}»{count}")
        parts.append("🟡 <b>Важное:</b>\n" + "\n".join(lines))

    if normal:
        lines = []
        for e in normal:
            snippet = (e["last_text"][:80] or "").replace("\n", " ")
            lines.append(f"• {e['sender_name']}: «{snippet}»")
        parts.append("🟢 <b>Обычное:</b>\n" + "\n".join(lines))

    return "\n\n".join(parts)


async def smart_digest_loop(owner_telegram_id: int) -> None:
    """Фоновый цикл: каждые 60 секунд проверяет, пора ли отправить дайджест."""
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_telegram_id)
                settings = owner.settings

                if not settings.smart_digest_enabled:
                    await asyncio.sleep(60)
                    continue

                now = datetime.utcnow()
                last_sent = settings.smart_digest_last_sent
                interval = settings.smart_digest_interval_min

                if last_sent is not None:
                    elapsed = (now - last_sent).total_seconds() / 60
                    if elapsed < interval:
                        await asyncio.sleep(60)
                        continue

                messages = await collect_recent_messages(
                    session, owner, since_minutes=interval
                )
                text = build_smart_digest(messages, interval)

                await notifier.notify(text)

                settings.smart_digest_last_sent = now
        except Exception:
            logger.exception("smart_digest_loop tick failed")
        await asyncio.sleep(60)
