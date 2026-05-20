"""Команды /me (SelfProfile) и /profile (ContactProfile)."""

import json
import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.filters import OwnerOnly
from src.core.contacts.contact_resolver import resolve
from src.core.contacts.profile_builder import build_profile
from src.db.repo import (
    get_contact,
    get_contact_profile,
    get_or_create_user,
    get_self_profile,
)
from src.db.session import get_session
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager

logger = logging.getLogger(__name__)
router = Router(name="profile_cmd")
router.message.filter(OwnerOnly())


def _load_json_field(val: str | None) -> list | str | None:
    """Распарсить JSON-поле SelfProfile, если это список."""
    if val is None:
        return None
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else val
    except (json.JSONDecodeError, TypeError):
        return val


@router.message(Command("me"))
async def cmd_me(message: Message, command: CommandObject) -> None:
    """Показать SelfProfile или перестроить через --rebuild."""
    args = (command.args or "").strip()

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        if "--rebuild" in args:
            await message.answer("\U0001f9e0 Строю SelfProfile из памяти...")
            provider = await build_provider(session, owner)
            if provider is None:
                await message.answer("\u274c Нет LLM-провайдера. Настрой /settings.")
                return

            from src.core.contacts.self_profile_builder import build_self_profile

            profile = await build_self_profile(message.from_user.id, provider)
            if profile:
                await message.answer("\u2705 SelfProfile обновлён!")
            else:
                await message.answer(
                    "\u274c Недостаточно личных фактов (нужно минимум 5)."
                )
            return

        # Показ профиля
        sp = await get_self_profile(session, owner)
        if not sp:
            await message.answer(
                "\U0001f4cb SelfProfile ещё не построен.\n"
                "Используй <code>/me --rebuild</code> для создания."
            )
            return

    lines = ["<b>\U0001f4cb SelfProfile</b>", ""]

    prefs = _load_json_field(sp.preferences)
    if prefs:
        lines.append("<b>\U0001f3af Предпочтения:</b>")
        lines.extend(
            f"  \u2022 {p}"
            for p in (prefs if isinstance(prefs, list) else [prefs])[:10]
        )
        lines.append("")

    goals = _load_json_field(sp.goals)
    if goals:
        lines.append("<b>\U0001f680 Цели:</b>")
        lines.extend(
            f"  \u2022 {g}" for g in (goals if isinstance(goals, list) else [goals])[:5]
        )
        lines.append("")

    projects = _load_json_field(sp.current_projects)
    if projects:
        lines.append("<b>\U0001f4bc Проекты:</b>")
        lines.extend(
            f"  \u2022 {p}"
            for p in (projects if isinstance(projects, list) else [projects])[:5]
        )
        lines.append("")

    style = sp.decision_style
    if style:
        lines.append(f"<b>\U0001f9e0 Стиль решений:</b> {style}")

    comm = _load_json_field(sp.communication_preferences)
    if comm:
        items = comm if isinstance(comm, list) else [comm]
        lines.append(f"<b>\U0001f4ac Коммуникация:</b> {'; '.join(items[:5])}")

    sleep = sp.sleep_pattern
    if sleep:
        lines.append(f"<b>\U0001f319 Сон:</b> {sleep}")

    work = sp.work_hours
    if work:
        lines.append(f"<b>\U0001f4bb Рабочие часы:</b> {work}")

    lines.append("")
    lines.append("\U0001f504 <code>/me --rebuild</code> — перестроить")

    await message.answer("\n".join(lines))


def _load_profile_json(val: str | None) -> list | str | None:
    """Распарсить JSON-поле ContactProfile, если это список."""
    if val is None:
        return None
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else val
    except (json.JSONDecodeError, TypeError):
        return val


def _format_contact_profile(
    profile,
    display_name: str,
    show_rebuild_hint: bool = False,
) -> list[str]:
    """Форматирует ContactProfile в список строк для сообщения."""
    lines = [f"<b>👤 ContactProfile — {display_name}</b>", ""]

    if profile is None:
        lines.append("Профиль ещё не построен.")
        if show_rebuild_hint:
            lines.append("Нажми «🔄 Перестроить» для создания.")
        return lines

    if profile.closeness_label:
        closeness_str = (
            f" ({profile.closeness:.2f})"
            if hasattr(profile, "closeness") and profile.closeness
            else ""
        )
        lines.append(f"<b>🤝 Близость:</b> {profile.closeness_label}{closeness_str}")

    if profile.communication_style:
        lines.append(f"<b>📝 Стиль:</b> {profile.communication_style}")

    if profile.key_topics:
        topics = _load_profile_json(profile.key_topics)
        if topics:
            items = topics if isinstance(topics, list) else [topics]
            lines.append(f"<b>🔑 Темы:</b> {'; '.join(items[:6])}")

    if profile.sensitivity:
        lines.append(f"<b>🎯 Чувствительность:</b> {profile.sensitivity:.1f}")

    if profile.communication_dos:
        dos = _load_profile_json(profile.communication_dos)
        if dos:
            items = dos if isinstance(dos, list) else [dos]
            lines.append(f"<b>✅ Можно:</b> {'; '.join(items[:3])}")

    if profile.communication_donts:
        donts = _load_profile_json(profile.communication_donts)
        if donts:
            items = donts if isinstance(donts, list) else [donts]
            lines.append(f"<b>❌ Нельзя:</b> {'; '.join(items[:3])}")

    if profile.current_status:
        status_map = {
            "active": "🟢",
            "tension": "🟡",
            "resolved": "🔵",
            "distant": "⚪",
        }
        emoji = status_map.get(profile.current_status, "⚪")
        lines.append(f"<b>{emoji} Статус:</b> {profile.current_status}")

    if profile.relationship_phase:
        phase_map = {"warming": "📈", "cooling": "📉", "stable": "📊"}
        emoji = phase_map.get(profile.relationship_phase, "📊")
        lines.append(f"<b>{emoji} Фаза:</b> {profile.relationship_phase}")

    if profile.open_questions:
        questions = _load_profile_json(profile.open_questions)
        if questions:
            items = questions if isinstance(questions, list) else [questions]
            lines.append(f"<b>❓ Открытые вопросы:</b>")
            for q in items[:3]:
                lines.append(f"  • {q}")

    return lines


@router.message(Command("profile"))
async def cmd_profile(
    message: Message,
    command: CommandObject,
    userbot_manager: UserbotManager,
) -> None:
    """Показать ContactProfile контакта."""
    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "Использование: /profile <имя контакта>\nПример: /profile Иван Иванов"
        )
        return

    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        peer_id = None
        try:
            peer_id = int(args)
        except ValueError:
            candidates = await resolve(client, owner, args)
            if not candidates:
                await message.answer(f"🙅 Не нашёл контакт «{args}». Попробуй /sync.")
                return
            if len(candidates) > 1 and candidates[0].score < 90:
                names = "\n".join(f"• {c.label()} · {c.score}%" for c in candidates[:5])
                await message.answer(f"Нашёл несколько кандидатов. Уточни:\n{names}")
                return
            peer_id = candidates[0].peer_id

        contact = await get_contact(session, owner, peer_id)
        if contact is None:
            await message.answer(f"🙅 Контакт не найден в БД. Попробуй /sync.")
            return

        profile = await get_contact_profile(session, owner, peer_id)
        display_name = contact.display_name or str(peer_id)

    lines = _format_contact_profile(
        profile, display_name, show_rebuild_hint=profile is None
    )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Перестроить",
                    callback_data=f"profile:rebuild:{peer_id}",
                )
            ]
        ]
    )

    await message.answer("\n".join(lines), reply_markup=kb)


@router.callback_query(F.data.startswith("profile:rebuild:"))
async def cb_profile_rebuild(
    callback: CallbackQuery,
    userbot_manager: UserbotManager,
) -> None:
    """Перестроить ContactProfile через LLM."""
    peer_id = int(callback.data.split(":")[2])

    await callback.answer()
    await callback.message.edit_text("🧠 Строю профиль из памяти...")

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        provider = await build_provider(session, owner)

    if provider is None:
        await callback.message.edit_text("❌ Нет LLM-провайдера. Настрой /settings.")
        return
    if contact is None:
        await callback.message.edit_text("🙅 Контакт не найден.")
        return

    profile_data = await build_profile(
        owner_id=owner.telegram_id,
        contact_id=peer_id,
        provider=provider,
    )

    if not profile_data:
        await callback.message.edit_text(
            "❌ Не удалось построить профиль: нет фактов в памяти "
            "или ошибка LLM.\n\n"
            "Добавь факты через «🧠 Запомни» или в /memory."
        )
        return

    display_name = contact.display_name or str(peer_id)
    lines = [f"<b>👤 ContactProfile — {display_name}</b>", ""]

    closeness_label = profile_data.get("closeness_label")
    if closeness_label:
        lines.append(f"<b>🤝 Близость:</b> {closeness_label}")

    comm_style = profile_data.get("communication_style")
    if comm_style:
        lines.append(f"<b>📝 Стиль:</b> {comm_style}")

    topics = profile_data.get("key_topics")
    if topics:
        items = topics if isinstance(topics, list) else [topics]
        lines.append(f"<b>🔑 Темы:</b> {'; '.join(items[:6])}")

    sensitivity = profile_data.get("sensitivity")
    if sensitivity is not None:
        lines.append(f"<b>🎯 Чувствительность:</b> {float(sensitivity):.1f}")

    dos = profile_data.get("communication_dos")
    if dos:
        items = dos if isinstance(dos, list) else [dos]
        lines.append(f"<b>✅ Можно:</b> {'; '.join(items[:3])}")

    donts = profile_data.get("communication_donts")
    if donts:
        items = donts if isinstance(donts, list) else [donts]
        lines.append(f"<b>❌ Нельзя:</b> {'; '.join(items[:3])}")

    status = profile_data.get("current_status")
    if status:
        status_map = {
            "active": "🟢",
            "tension": "🟡",
            "resolved": "🔵",
            "distant": "⚪",
        }
        emoji = status_map.get(status, "⚪")
        lines.append(f"<b>{emoji} Статус:</b> {status}")

    phase = profile_data.get("relationship_phase")
    if phase:
        phase_map = {"warming": "📈", "cooling": "📉", "stable": "📊"}
        emoji = phase_map.get(phase, "📊")
        lines.append(f"<b>{emoji} Фаза:</b> {phase}")

    questions = profile_data.get("open_questions")
    if questions:
        items = questions if isinstance(questions, list) else [questions]
        lines.append(f"<b>❓ Открытые вопросы:</b>")
        for q in items[:3]:
            lines.append(f"  • {q}")

    lines.append("")
    lines.append("✅ Профиль обновлён!")

    rebuild_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Перестроить",
                    callback_data=f"profile:rebuild:{peer_id}",
                )
            ]
        ]
    )

    await callback.message.edit_text("\n".join(lines), reply_markup=rebuild_kb)
