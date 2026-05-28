"""Команда /timeline — хронология обсуждения темы по чатам.

Ищет по FTS5, группирует по контакту, показывает даты и сниппеты.
"""

import logging
from collections import defaultdict

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.infra.formatting import bold
from src.core.infra.text_sanitizer import sanitize_html
from src.db.repo import (
    cross_chat_search,
    fts_search,
    get_or_create_user,
)
from src.db.session import get_session


logger = logging.getLogger(__name__)
router = Router(name="timeline")
router.message.filter(OwnerOnly())


@router.message(Command("timeline"))
async def cmd_timeline(
    message: Message,
    command: CommandObject,
) -> None:
    query = (command.args or "").strip()
    if not query:
        await message.answer("Использование: /timeline <тема>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

    # ── Primary: cross-chat FTS5 search ───────────────────────────────
    async with get_session() as session:
        results = await cross_chat_search(session, owner, query, limit=5)

    # ── Fallback: flat FTS search + group manually ────────────────────
    if not results:
        async with get_session() as session:
            hits = await fts_search(session, owner.id, query, limit=20)

        if not hits:
            await message.answer(f"По теме «{sanitize_html(query)}» ничего не найдено.")
            return

        # Группировка по peer_id
        groups: dict[int, list] = defaultdict(list)
        for h in hits:
            groups[h.peer_id].append(h)

        lines = [f"📅 Хронология «{sanitize_html(query)}»:", ""]
        shown = 0
        for peer_id, peer_hits in groups.items():
            if shown >= 5:
                lines.append("")
                lines.append("⚠ Показаны первые 5 чатов.")
                break
            name = peer_hits[0].peer_name or peer_hits[0].sender_name or str(peer_id)
            lines.append(f"🧑 {bold(sanitize_html(name))} ({len(peer_hits)} совпад.)")
            for h in peer_hits[:3]:
                date_str = h.date.strftime("%d.%m") if h.date else ""
                snippet = (h.snippet or "")[:80]
                if date_str:
                    lines.append(f"  {date_str} «{sanitize_html(snippet)}»")
                else:
                    lines.append(f"  «{sanitize_html(snippet)}»")
            shown += 1

        await message.answer("\n".join(lines))
        return

    # ── Format cross_chat_search results ──────────────────────────────
    lines = [f"📅 Хронология «{sanitize_html(query)}»:", ""]
    shown = 0
    for r in results:
        if shown >= 5:
            lines.append("")
            lines.append("⚠ Показаны первые 5 чатов.")
            break
        name = r["display_name"] or str(r["peer_id"])
        lines.append(f"🧑 {bold(sanitize_html(name))} ({r['total_matches']} совпад.)")
        for s in r["snippets"]:
            date_str = s["date"].strftime("%d.%m") if s.get("date") else ""
            text = (s["text"] or "")[:80]
            if date_str:
                lines.append(f"  {date_str} «{sanitize_html(text)}»")
            else:
                lines.append(f"  «{sanitize_html(text)}»")
        shown += 1

    await message.answer("\n".join(lines))
