"""Утренний дайджест: входящие без ответа, горящие обещания и авто-ответы за ночь."""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src.config import settings as app_settings
from src.core.notification_queue import notification_queue
from src.core.text_sanitizer import sanitize_html
from src.db.models import Notification
from src.core.timeutil import fmt_local, now_in_tz
from src.db.models import AutoReplyLog, Commitment, Message, User
from src.db.repo import get_or_create_user, list_open_commitments
from src.db.session import get_session
from src.llm.base import ChatMessage
from src.llm.router import build_provider


logger = logging.getLogger(__name__)


DIGEST_SYSTEM = (
    "Ты делаешь короткий утренний дайджест по моей Telegram-активности.\n"
    "Структура (HTML aiogram):\n"
    "☀ <b>Доброе утро!</b>\n\n"
    "📨 <b>Ждут ответа</b> (если есть): кто и про что (1 строка на собеседника).\n"
    "🔥 <b>Мои горящие обещания</b>: те, что просрочены или ближайшие 24ч.\n"
    "💼 <b>Обещания мне</b>: что просрочено или скоро.\n"
    "🤖 <b>Авто-ответы</b>: сколько и кому, без подробностей.\n"
    "Если в каком-то блоке пусто — пропускай блок целиком."
)


async def _gather_payload(owner: User) -> dict:
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=14)

    async with get_session() as session:
        # входящие за период
        incoming_result = await session.execute(
            select(Message)
            .where(
                Message.user_id == owner.id,
                Message.is_outgoing.is_(False),
                Message.date >= since,
            )
            .order_by(Message.date.desc())
            .limit(200)
        )
        incoming = list(incoming_result.scalars().all())

        # сгруппировать по peer и взять только тех, где после последнего входящего нет моего ответа
        by_peer: dict[int, list[Message]] = {}
        for m in incoming:
            by_peer.setdefault(m.peer_id, []).append(m)

        waiting: list[tuple[int, str | None, str]] = []
        for peer_id, msgs in by_peer.items():
            last_in = max(msgs, key=lambda x: x.date)
            my_after = await session.execute(
                select(Message)
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id == peer_id,
                    Message.is_outgoing.is_(True),
                    Message.date > last_in.date,
                )
                .limit(1)
            )
            if my_after.scalar_one_or_none() is None:
                snippet = (
                    last_in.transcript or last_in.text or last_in.extracted_text or ""
                )[:200]
                waiting.append((peer_id, last_in.sender_name, snippet))

        mine = await list_open_commitments(session, owner, direction="mine")
        theirs = await list_open_commitments(session, owner, direction="theirs")

        autoreplies_result = await session.execute(
            select(AutoReplyLog).where(
                AutoReplyLog.user_id == owner.id, AutoReplyLog.created_at >= since
            )
        )
        autoreplies = list(autoreplies_result.scalars().all())

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    soon = now + timedelta(hours=24)

    def _hot(items: list[Commitment]) -> list[Commitment]:
        out = []
        for c in items:
            if c.deadline_at and (c.deadline_at < now or c.deadline_at <= soon):
                out.append(c)
            elif c.deadline_at is None and (now - c.created_at) > timedelta(days=2):
                out.append(c)
        return out

    return {
        "waiting": waiting,
        "mine_hot": _hot(mine),
        "theirs_hot": _hot(theirs),
        "autoreplies": autoreplies,
    }


def _payload_to_text(payload: dict, tz_name: str) -> str:
    parts: list[str] = []
    if payload["waiting"]:
        lines = [
            f"- {name or peer_id}: {snippet}"
            for peer_id, name, snippet in payload["waiting"][:20]
        ]
        parts.append("Ждут ответа:\n" + "\n".join(lines))
    if payload["mine_hot"]:
        lines = []
        for c in payload["mine_hot"][:20]:
            d = fmt_local(c.deadline_at, tz_name) if c.deadline_at else "без срока"
            lines.append(f"- {c.peer_name or c.peer_id}: {c.text} (до {d})")
        parts.append("Мои горящие обещания:\n" + "\n".join(lines))
    if payload["theirs_hot"]:
        lines = []
        for c in payload["theirs_hot"][:20]:
            d = fmt_local(c.deadline_at, tz_name) if c.deadline_at else "без срока"
            lines.append(f"- {c.peer_name or c.peer_id}: {c.text} (до {d})")
        parts.append("Обещания мне (горящие):\n" + "\n".join(lines))
    if payload["autoreplies"]:
        peers = {a.peer_name or a.peer_id for a in payload["autoreplies"]}
        parts.append(
            f"Авто-ответов: {len(payload['autoreplies'])} (кому: {', '.join(map(str, peers))})"
        )
    return "\n\n".join(parts) or "Активности не было."


async def build_digest(owner_telegram_id: int) -> str:
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        provider = await build_provider(session, owner)
        heavy = owner.settings.use_heavy_model
        tz_name = owner.settings.timezone

    if provider is None:
        return "Не задан LLM-ключ — не могу собрать дайджест. Открой /settings."

    payload = await _gather_payload(owner)
    raw_text = _payload_to_text(payload, tz_name)
    if raw_text == "Активности не было.":
        return "☀ Доброе утро! За ночь — тишина."

    response = await provider.chat(
        [
            ChatMessage(role="system", content=DIGEST_SYSTEM),
            ChatMessage(role="user", content=raw_text),
        ],
        heavy=heavy,
    )
    return sanitize_html(response)


async def send_digest(owner_telegram_id: int) -> None:
    text = await build_digest(owner_telegram_id)
    await notification_queue.enqueue(
        topic="digest",
        text=text,
        priority=Notification.PRIORITY_MEDIUM,
        category="morning_report",
    )


async def digest_scheduler_loop() -> None:
    """Каждую минуту проверяет, пора ли отправлять дайджест.
    Сравнение времени — в TZ владельца (UserSettings.timezone)."""
    import asyncio

    last_sent: dict[int, str] = {}  # telegram_id -> "YYYY-MM-DD"
    while True:
        try:
            owner_id = app_settings.owner_telegram_id
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = owner.settings.timezone
                enabled = owner.settings.digest_enabled
                target_hm = owner.settings.digest_time
            local_now = now_in_tz(tz_name)
            current_hm = local_now.strftime("%H:%M")
            current_day = local_now.strftime("%Y-%m-%d")
            if (
                enabled
                and target_hm == current_hm
                and last_sent.get(owner_id) != current_day
            ):
                await send_digest(owner_id)
                last_sent[owner_id] = current_day
        except Exception:
            logger.exception("digest scheduler tick failed")
        await asyncio.sleep(60)
