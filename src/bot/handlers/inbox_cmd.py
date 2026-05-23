"""/inbox — priority radar showing "what needs my attention".

Четыре секции:
  1. Ждут ответа  — диалоги, где последнее сообщение от собеседника >1ч назад
  2. Обязательства — открытые commitment из /todos
  3. Конфликты     — триггеры от conflict_predictor
  4. Низкое здоровье — контакты со score <50 по health_score
"""

import logging
from datetime import datetime, timezone

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.contacts.health_score import get_contact_health
from src.core.contacts.reply_radar import collect_reply_radar
from src.core.actions.conflict_predictor import detect_silence_triggers
from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.timeutil import fmt_local
from src.db.repo import get_or_create_user, list_open_commitments, list_contacts
from src.db.session import get_session

logger = logging.getLogger(__name__)
router = Router(name="inbox_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


@router.message(Command("inbox"))
async def cmd_inbox(message: Message) -> None:
    telegram_id = message.from_user.id
    lines: list[str] = ["<b>📥 Входящие</b>", ""]

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        tz_name = owner.settings.timezone if owner.settings else "UTC"

        # ── 1. Ждут ответа ─────────────────────────────────────────
        radar = await collect_reply_radar(telegram_id, limit=20)
        waiting = [r for r in radar if r.waiting_hours >= 1]
        if waiting:
            lines.append(f"🟡 <b>Ждут ответа ({len(waiting)}):</b>")
            for item in waiting[:5]:
                hours_str = _fmt_hours_ago(item.waiting_hours)
                lines.append(f"  • {sanitize_html(item.contact_name)} — {hours_str}")
            if len(waiting) > 5:
                lines.append(f"  … и ещё {len(waiting) - 5}")
            lines.append("")
        else:
            lines.append("✅ <b>Нет сообщений, ждущих ответа</b>")
            lines.append("")

        # ── 2. Обязательства ───────────────────────────────────────
        commits = await list_open_commitments(session, owner)
        if commits:
            now = datetime.now(timezone.utc)
            overdue = [c for c in commits if c.deadline_at and c.deadline_at < now]
            upcoming = [c for c in commits if c not in overdue]
            lines.append(f"🔴 <b>Обязательства ({len(commits)}):</b>")
            for c in (overdue + upcoming)[:5]:
                who = c.peer_name or "Я"
                if c in overdue:
                    lines.append(
                        f"  ⚠️ {sanitize_html(who)}: {sanitize_html(c.text[:60])} "
                        f"(<b>просрочено</b>)"
                    )
                elif c.deadline_at:
                    dl = fmt_local(c.deadline_at, tz_name, fmt="%d %b %H:%M")
                    lines.append(
                        f"  • {sanitize_html(who)}: {sanitize_html(c.text[:60])} "
                        f"(до {dl})"
                    )
                else:
                    lines.append(
                        f"  • {sanitize_html(who)}: {sanitize_html(c.text[:60])}"
                    )
            if len(commits) > 5:
                lines.append(f"  … и ещё {len(commits) - 5}")
            lines.append("")

        # ── 3. Конфликты ───────────────────────────────────────────
        try:
            conflicts = await detect_silence_triggers(telegram_id)
        except Exception:
            logger.warning("inbox: conflict_predictor unavailable", exc_info=True)
            conflicts = []

        if conflicts:
            lines.append(f"⚡ <b>Конфликты ({len(conflicts)}):</b>")
            for c in conflicts[:3]:
                lines.append(
                    f"  • {sanitize_html(c['contact_name'])} — "
                    f"молчание {c['current_hours']:.0f}ч"
                )
            if len(conflicts) > 3:
                lines.append(f"  … и ещё {len(conflicts) - 3}")
            lines.append("")

        # ── 4. Низкое здоровье контактов ───────────────────────────
        try:
            contacts = await list_contacts(
                session, owner, kinds=("user",), include_bots=False
            )
        except Exception:
            contacts = []

        low_health: list[tuple] = []
        for contact in contacts[:20]:  # проверяем первые 20 контактов
            try:
                health = await get_contact_health(telegram_id, contact.peer_id)
                if health["score"] < 50:
                    low_health.append((contact, health))
            except Exception:
                continue

        low_health.sort(key=lambda x: x[1]["score"])

        if low_health:
            lines.append(f"💚 <b>Низкое здоровье ({len(low_health)}):</b>")
            for contact, health in low_health[:5]:
                emoji = "🔴" if health["score"] < 30 else "🟡"
                lines.append(
                    f"  • {sanitize_html(contact.display_name)} — "
                    f"{health['score']}/100 {emoji}"
                )
            if len(low_health) > 5:
                lines.append(f"  … и ещё {len(low_health) - 5}")
            lines.append("")

    text = "\n".join(lines).strip()
    await message.answer(text or "📥 Входящие пусты — всё в порядке ✅")


def _fmt_hours_ago(hours: float) -> str:
    """Форматирует часы ожидания в человекочитаемый вид."""
    if hours < 24:
        return f"{hours:.0f}ч назад"
    days = int(hours / 24)
    if days == 1:
        return "вчера"
    return f"{days}д назад"
