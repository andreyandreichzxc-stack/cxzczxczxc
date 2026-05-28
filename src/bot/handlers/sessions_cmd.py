"""View conversation session history — /sessions command."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.bot.handlers.free_text_common import safe_answer
from src.core.memory.session_recorder import get_session_history
from src.db.session import get_session

logger = logging.getLogger(__name__)

router = Router(name="sessions_cmd")
router.message.filter(OwnerOnly())


@router.message(Command("sessions"))
async def cmd_sessions(message: Message) -> None:
    """Show recent conversation sessions with message previews."""
    uid = message.from_user.id
    async with get_session() as session:
        history = await get_session_history(session, uid, limit=5)

    if not history:
        await safe_answer(message, "📭 Нет записанных сессий.")
        return

    lines = ["📋 <b>Последние сессии:</b>\n"]
    for s in history:
        start = s["started_at"][:16] if s["started_at"] else "?"
        status = "🔴" if s["ended_at"] is None else "🟢"
        lines.append(
            f"{status} Сессия #{s['session_id']} — {start} ({s['turn_count']} сообщ.)"
        )
        if s["summary"]:
            lines.append(f"   📝 {s['summary'][:100]}")
        # Show last 2 messages as preview
        for m in s["messages"][-2:]:
            role_icon = "👤" if m["role"] == "user" else "🤖"
            content = m["content"][:80].replace("\n", " ")
            lines.append(f"   {role_icon} {content}")
        lines.append("")

    await safe_answer(message, "\n".join(lines))
