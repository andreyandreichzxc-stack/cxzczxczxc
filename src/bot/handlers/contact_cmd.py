"""Команда /contact — показать всё, что бот знает о контакте."""

import json
import logging
from datetime import datetime, timezone

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.contacts.contact_resolver import resolve
from src.core.contacts.health_score import get_contact_health
from src.core.memory.context_files import get_contact_context
from src.core.memory.memory_recall import recall
from src.core.infra.text_sanitizer import sanitize_html
from src.db.repo import (
    fetch_chat_messages,
    get_contact,
    get_contact_profile,
    get_conversation_state,
    get_or_create_user,
    list_open_commitments,
)
from src.db.session import get_session
from src.userbot.manager import UserbotManager

logger = logging.getLogger(__name__)
router = Router(name="contact_cmd")
router.message.filter(OwnerOnly())


def _format_style_profile(style_json: str | None) -> list[str]:
    """Parse Contact.style_profile JSON and return readable lines."""
    if not style_json:
        return []
    try:
        p = json.loads(style_json)
    except (json.JSONDecodeError, TypeError):
        return []

    parts: list[str] = []
    if p.get("address"):
        parts.append(f"обращение: {p['address']}")
    if p.get("register"):
        parts.append(f"регистр: {p['register']}")
    if p.get("length"):
        parts.append(f"длина: {p['length']}")
    if p.get("emoji_usage"):
        parts.append(f"эмодзи: {p['emoji_usage']}")
    if p.get("punctuation"):
        parts.append(f"пунктуация: {p['punctuation']}")
    if p.get("notes"):
        parts.append(f"заметки: {p['notes']}")
    if p.get("typical_openings"):
        parts.append("зачины: " + ", ".join(p["typical_openings"][:3]))
    if p.get("phrases"):
        parts.append("маркеры: " + ", ".join(p["phrases"][:5]))

    if not parts:
        return []

    lines = ["💬 Стиль общения:"]
    for part in parts:
        lines.append(f"  • {sanitize_html(part)}")
    return lines


def _format_time_ago(dt: datetime | None) -> str:
    """Format datetime as a short relative time string."""
    if dt is None:
        return "?"
    # Make naive datetime offset-aware for subtraction
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 0:
        return "только что"
    if seconds < 60:
        return f"{seconds}с"
    if seconds < 3600:
        return f"{seconds // 60}мин"
    if seconds < 86400:
        return f"{seconds // 3600}ч"
    days = seconds // 86400
    if days < 30:
        return f"{days}д"
    return dt.strftime("%d.%m")


@router.message(Command("contact"))
async def cmd_contact(
    message: Message,
    command: CommandObject,
    userbot_manager: UserbotManager,
) -> None:
    name = (command.args or "").strip()
    if not name:
        await message.answer("Использование: /contact <имя>")
        return

    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        # 1. Resolve contact via fuzzy matching (inside session — resolve may access owner.settings etc.)
        candidates = await resolve(client, owner, name)
    if not candidates:
        await message.answer(
            f"❌ Не нашёл контакт «{sanitize_html(name)}». Сделай /sync."
        )
        return

    target = candidates[0]
    peer_id = target.peer_id

    async with get_session() as session:
        contact = await get_contact(session, owner, peer_id)
        if contact is None:
            await message.answer("❌ Контакт не найден в БД. Сделай /sync.")
            return

        profile = await get_contact_profile(session, owner, peer_id)
        conv_state = await get_conversation_state(session, owner, peer_id)
        recent_msgs = await fetch_chat_messages(session, owner, peer_id, limit=3)
        commitments = await list_open_commitments(session, owner, peer_id=peer_id)

        # ── Extract ORM values to local vars (session-safe) ──────────
        contact_username = contact.username
        contact_phone = contact.phone
        contact_is_bot = contact.is_bot
        contact_folder_names = contact.folder_names
        contact_archetype = contact.archetype
        contact_style_profile = contact.style_profile
        profile_closeness = profile.closeness if profile is not None else None
        conv_state = (
            {
                "status": conv_state.status,
                "unread_count": conv_state.unread_count,
                "last_incoming_at": conv_state.last_incoming_at,
            }
            if conv_state
            else None
        )
        recent_msgs = [
            {
                "is_outgoing": m.is_outgoing,
                "sender_name": m.sender_name,
                "text": m.text,
                "transcript": m.transcript,
                "date": m.date,
                "kind": m.kind,
            }
            for m in recent_msgs
        ]
        commitments = [
            {
                "direction": c.direction,
                "deadline_at": c.deadline_at,
                "text": c.text,
            }
            for c in commitments
        ]

    # 2. Memory facts (outside session to avoid holding it during LLM calls)
    memory_result = await recall(
        message.from_user.id,
        contact_id=peer_id,
        limit=5,
        mode="normal",
    )

    # 3. Context file
    context_text = get_contact_context(target.display_name)

    lines: list[str] = []

    # ── Contact profile ──────────────────────────────────────────────
    header = f"🧑 {sanitize_html(target.display_name)}"
    if contact_username:
        header += f" · @{sanitize_html(contact_username)}"
    if contact_phone:
        header += f" · 📞 {sanitize_html(contact_phone)}"
    if contact_is_bot:
        header += " · 🤖 Бот"
    lines.append(header)

    if contact_folder_names:
        folders = [f.strip() for f in contact_folder_names.split(",") if f.strip()]
        if folders:
            lines.append(f"📂 Папки: {sanitize_html(', '.join(folders))}")

    if contact_archetype:
        lines.append(f"🎭 Архетип: {sanitize_html(contact_archetype)}")

    if profile_closeness is not None:
        closeness_pct = round(profile_closeness * 10)
        lines.append(f"❤️ Closeness: {closeness_pct}/10")

    # ── Contact health ───────────────────────────────────────────────
    health = await get_contact_health(message.from_user.id, peer_id)
    lines.append(f"💚 <b>Здоровье:</b> {health['score']}/100 {health['status']}")
    if health["days_since_last"] > 0:
        lines.append(f"  • Последнее сообщение: {health['days_since_last']} дн. назад")
    if health["open_commitments"] > 0:
        lines.append(f"  • Открытых обещаний: {health['open_commitments']}")

    # ── Style profile ────────────────────────────────────────────────
    style_lines = _format_style_profile(contact_style_profile)
    if style_lines:
        lines.append("")
        lines.extend(style_lines)

    # ── Memory facts ─────────────────────────────────────────────────
    if memory_result.facts:
        lines.append("")
        lines.append("🧠 Что я знаю:")
        for f in memory_result.facts[:5]:
            conf_str = f"{f.confidence:.2f}" if f.confidence else "?"
            lines.append(f"  • {sanitize_html(f.fact)} (conf {conf_str})")

    # ── Commitments ──────────────────────────────────────────────────
    if commitments:
        lines.append("")
        lines.append("📋 Обещания:")
        for i, c in enumerate(commitments[:5], 1):
            who = "Я" if c["direction"] == "mine" else "Они"
            deadline = ""
            if c["deadline_at"]:
                deadline = f" (до {c['deadline_at'].strftime('%d.%m')})"
            lines.append(f"  {i}. <b>{who}</b>: {sanitize_html(c['text'])}{deadline}")

    # ── Context file ─────────────────────────────────────────────────
    if context_text:
        lines.append("")
        lines.append("📝 Контекст:")
        ctx_preview = context_text.strip()[:300]
        for cl in ctx_preview.split("\n")[:4]:
            lines.append(f"  {sanitize_html(cl[:120])}")

    # ── Recent messages ──────────────────────────────────────────────
    if recent_msgs:
        lines.append("")
        lines.append("💌 Последние сообщения:")
        for m in recent_msgs:
            sender = "Я" if m["is_outgoing"] else (m["sender_name"] or "Они")
            text = (m["text"] or m["transcript"] or "")[:80]
            time_str = _format_time_ago(m["date"])
            if text:
                lines.append(
                    f"  {sanitize_html(sender)}: {sanitize_html(text)} ({time_str})"
                )
            else:
                lines.append(f"  {sanitize_html(sender)}: [{m['kind']}] ({time_str})")

    # ── Conversation state ───────────────────────────────────────────
    if conv_state:
        lines.append("")
        status_emoji = {
            "active": "🟢",
            "waiting_reply": "🟡",
            "archived": "⚪",
        }
        emoji = status_emoji.get(conv_state["status"], "⚪")
        status_labels = {
            "active": "активен",
            "waiting_reply": "жду ответа",
            "archived": "в архиве",
        }
        status_text = status_labels.get(conv_state["status"], conv_state["status"])
        state_parts = [f"{emoji} Статус: {status_text}"]
        if conv_state["unread_count"]:
            state_parts.append(f"· 📨 {conv_state['unread_count']} непрочитанных")
        if conv_state["last_incoming_at"]:
            state_parts.append(
                f"· вх. {_format_time_ago(conv_state['last_incoming_at'])}"
            )
        lines.append(" ".join(state_parts))

    await message.answer("\n".join(lines))
