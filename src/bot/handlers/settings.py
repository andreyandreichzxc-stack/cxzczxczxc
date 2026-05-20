"""/settings — главное меню и разделы. callback_data: set:sec / set:tog / set:choose / set:input."""

import json
import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.bot.states import SettingsStates
from src.config import LLMDefaults
from src.core.timeutil import HM_RE, TZ_PRESETS, is_valid_tz, tz_short
from src.db.repo import get_api_key, get_or_create_user, list_folders, upsert_api_key
from src.db.session import get_session
from src.userbot.dialogs import sync_dialogs
from src.userbot import get_active_telethon_client, get_userbot_manager
from src.llm.gemini_provider import GeminiProvider
from src.llm.mistral_provider import MistralProvider
from src.llm.openai_provider import OpenAIProvider


logger = logging.getLogger(__name__)
router = Router(name="settings")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


def _check(value: bool) -> str:
    return "✅" if value else "❌"


# ---------- Главное меню ----------


async def _render_menu(telegram_id: int) -> tuple[str, InlineKeyboardMarkup]:
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        s = owner.settings
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")

    text = (
        "⚙ <b>Настройки</b>\n\n"
        f"🌍 Часовой пояс: <b>{tz_short(s.timezone)}</b>\n"
        f"🔄 Авто-ответ: {_check(s.auto_reply_enabled)} (кулдаун {s.auto_reply_cooldown_min}м)\n"
        f"🔄 Авто-синк: {_check(getattr(s, 'auto_sync_enabled', True))} (каждые {getattr(s, 'auto_sync_interval_sec', 7200)}с)\n"
        f"🧠 Авто-память: {_check(getattr(s, 'auto_extract_memories', False))}\n"
        f"⭐ Избранное: {_check(getattr(s, 'include_saved_messages', False))}\n"
        f"☀ Дайджест: {_check(s.digest_enabled)} ({s.digest_time})\n"
        f"⏰ Напоминания: {_check(s.reminders_enabled)} (за {s.reminder_lead_hours}ч; просрочки {_check(s.reminder_overdue_enabled)})\n"
        f"📰 Новости: {_check(s.news_enabled)} (окно {s.news_window_hours}ч)\n"
        f"🛡 Игнорировать архив: {_check(s.ignore_archived)}\n"
        f"📊 Smart дайджест: {_check(getattr(s, 'smart_digest_enabled', False))} (каждые {getattr(s, 'smart_digest_interval_min', 30)}м)\n"
        f"🤖 LLM: <b>{s.llm_provider}</b> · {'тяжёлая' if s.use_heavy_model else 'лёгкая'}\n"
        f"🎤 Транскрипция: <b>{s.transcription_mode}</b> ({getattr(s, 'transcription_api_provider', 'openai')})\n"
        f"🔑 Ключи: OpenAI {_check(bool(openai_key))} · Gemini {_check(bool(gemini_key))} · Mistral {_check(bool(mistral_key))}\n\n"
        "<i>Тапни раздел, чтобы открыть его настройки и описание.</i>"
    )
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(text="🌍 Часовой пояс", callback_data="set:sec:tz"),
        InlineKeyboardButton(text="🔄 Авто-ответ", callback_data="set:sec:auto_reply"),
    )
    kb.row(
        InlineKeyboardButton(text="🤖 Авто-режим", callback_data="set:sec:auto_mode"),
    )
    kb.row(
        InlineKeyboardButton(text="🌅 Дайджест", callback_data="set:sec:digest"),
        InlineKeyboardButton(text="⏰ Напоминания", callback_data="set:sec:reminders"),
    )
    kb.row(
        InlineKeyboardButton(
            text="📊 Smart-дайджест", callback_data="set:sec:smart_digest"
        ),
        InlineKeyboardButton(text="📰 Новости", callback_data="set:sec:news"),
    )
    kb.row(
        InlineKeyboardButton(text="🤖 LLM", callback_data="set:sec:llm"),
        InlineKeyboardButton(
            text="🎤 Транскрипция", callback_data="set:sec:transcription"
        ),
    )
    kb.row(
        InlineKeyboardButton(text="✍️ Черновики", callback_data="set:sec:drafts"),
        InlineKeyboardButton(text="🔒 Приватность", callback_data="set:sec:privacy"),
    )
    kb.row(
        InlineKeyboardButton(text="🔄 Синхронизация", callback_data="set:sec:sync"),
        InlineKeyboardButton(text="🔑 API-ключи", callback_data="set:sec:keys"),
    )
    kb.row(InlineKeyboardButton(text="📁 Папки", callback_data="set:sec:folders"))
    kb.row(InlineKeyboardButton(text="📬 Треды", callback_data="thread:refresh"))
    kb.row(InlineKeyboardButton(text="🧠 Полный анализ", callback_data="set:analyze"))
    kb.row(InlineKeyboardButton(text="❌ Закрыть", callback_data="set:close"))
    # Быстрые тогглы (авто-память, избранное, дайджест, авто-ответ)
    text += "\n⚡ <b>Быстрые тогглы:</b>"
    kb.row(
        InlineKeyboardButton(
            text=f"🧠 Авто-память {_check(getattr(s, 'auto_extract_memories', False))}",
            callback_data="set:tog:auto_extract_memories",
        ),
        InlineKeyboardButton(
            text=f"⭐ Избранное {_check(getattr(s, 'include_saved_messages', False))}",
            callback_data="set:tog:include_saved_messages",
        ),
    )
    kb.row(
        InlineKeyboardButton(
            text=f"☀ Дайджест {_check(s.digest_enabled)}",
            callback_data="set:tog:digest_enabled",
        ),
        InlineKeyboardButton(
            text=f"🔄 Авто-ответ {_check(s.auto_reply_enabled)}",
            callback_data="set:tog:auto_reply_enabled",
        ),
    )
    return text, kb.as_markup()


@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    text, kb = await _render_menu(message.from_user.id)
    await message.answer(text, reply_markup=kb)


async def _safe_edit(message, text: str, kb) -> None:
    # глушит безобидное "message is not modified" при повторном тапе той же опции
    if message is None:
        return
    try:
        await message.edit_text(text, reply_markup=kb)
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            raise


@router.callback_query(F.data == "set:menu")
async def cb_menu(callback: CallbackQuery) -> None:
    text, kb = await _render_menu(callback.from_user.id)
    await _safe_edit(callback.message, text, kb)
    await callback.answer()


@router.callback_query(F.data == "set:close")
async def cb_close(callback: CallbackQuery) -> None:
    if callback.message:
        await callback.message.delete()
    await callback.answer()


@router.callback_query(F.data == "set:analyze")
async def cb_settings_analyze(callback: CallbackQuery) -> None:
    await callback.answer("Запускаю анализ...")
    await callback.message.answer(
        "🧠 <b>Полный анализ переписок</b>\n\n"
        "Используй команду /analyze для полного анализа.\n\n"
        "<b>Примеры:</b>\n"
        "<code>/analyze</code> — все контакты из выбранных папок\n"
        "<code>/analyze Работа</code> — только папка «Работа»\n"
        "<code>/analyze Работа Семья</code> — папки «Работа» и «Семья»"
    )


# ---------- Универсальные ручки тогглов и выбора ----------

BOOL_KEYS = {
    "auto_reply_enabled",
    "ignore_archived",
    "digest_enabled",
    "reminders_enabled",
    "reminder_overdue_enabled",
    "news_enabled",
    "use_heavy_model",
    "auto_sync_enabled",
    "auto_extract_memories",
    "include_saved_messages",
    "draft_suggestions_enabled",
    "draft_only_important",
    "smart_digest_enabled",
    "urgent_notify_enabled",
    "monitor_only_selected_folders",
    "auto_reply_close_contacts",
    "notify_on_auto_reply",
}

CHOICE_KEYS = {
    "llm_provider": {"openai", "gemini", "mistral"},
    "transcription_mode": {"local", "api", "hybrid"},
    "transcription_api_provider": {"openai", "gemini", "mistral"},
    "auto_reply_mode": {"static", "smart"},
    "auto_mode": {"offline_only", "always", "smart"},
}

NUMERIC_KEYS = {
    "auto_reply_cooldown_min",
    "reminder_lead_hours",
    "news_window_hours",
    "auto_sync_interval_sec",
    "draft_max_per_hour",
    "smart_digest_interval_min",
}


@router.callback_query(F.data.startswith("set:tog:"))
async def cb_toggle(callback: CallbackQuery) -> None:
    key = callback.data.split(":", 2)[2]
    if key not in BOOL_KEYS:
        await callback.answer("Неизвестный переключатель", show_alert=True)
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        current = getattr(owner.settings, key)
        setattr(owner.settings, key, not current)
    await callback.answer("Готово")
    await _refresh_section(callback, _section_for_key(key))


@router.callback_query(F.data.startswith("set:choose:"))
async def cb_choose(callback: CallbackQuery) -> None:
    _, _, key, value = callback.data.split(":", 3)
    if key in CHOICE_KEYS:
        if value not in CHOICE_KEYS[key]:
            await callback.answer("Невалидное значение", show_alert=True)
            return
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            setattr(owner.settings, key, value)
    elif key in NUMERIC_KEYS:
        try:
            ivalue = max(0, int(value))
        except ValueError:
            await callback.answer("Невалидное число", show_alert=True)
            return
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            setattr(owner.settings, key, ivalue)
    else:
        await callback.answer("Неизвестное поле", show_alert=True)
        return
    await callback.answer("Готово")
    await _refresh_section(callback, _section_for_key(key))


def _section_for_key(key: str) -> str:
    return {
        "auto_reply_enabled": "auto_reply",
        "auto_reply_cooldown_min": "auto_reply",
        "auto_reply_mode": "auto_reply",
        "auto_reply_text": "auto_reply",
        "ignore_archived": "privacy",
        "digest_enabled": "digest",
        "reminders_enabled": "reminders",
        "reminder_lead_hours": "reminders",
        "reminder_overdue_enabled": "reminders",
        "news_enabled": "news",
        "news_window_hours": "news",
        "llm_provider": "llm",
        "use_heavy_model": "llm",
        "transcription_mode": "transcription",
        "transcription_api_provider": "transcription",
        "draft_suggestions_enabled": "drafts",
        "draft_only_important": "drafts",
        "draft_max_per_hour": "drafts",
        "auto_mode": "auto_mode",
        "auto_reply_close_contacts": "auto_mode",
        "notify_on_auto_reply": "auto_mode",
    }.get(key, "menu")


async def _refresh_section(callback: CallbackQuery, section: str) -> None:
    if section == "menu":
        text, kb = await _render_menu(callback.from_user.id)
    else:
        text, kb = await _render_section(callback.from_user.id, section)
    await _safe_edit(callback.message, text, kb)


# ---------- Разделы ----------


@router.callback_query(F.data.startswith("set:sec:"))
async def cb_open_section(callback: CallbackQuery) -> None:
    section = callback.data.split(":", 2)[2]
    text, kb = await _render_section(callback.from_user.id, section)
    await _safe_edit(callback.message, text, kb)
    await callback.answer()


def _back_row():
    return [InlineKeyboardButton(text="← Меню настроек", callback_data="set:menu")]


async def _render_section(
    telegram_id: int, section: str
) -> tuple[str, InlineKeyboardMarkup]:
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        s = owner.settings
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")

    kb = InlineKeyboardBuilder()

    if section == "auto_reply":
        mode_label = (
            "🤖 умный (LLM в твоём стиле)"
            if s.auto_reply_mode == "smart"
            else "📝 заготовленный текст"
        )
        snippet = (s.auto_reply_text or "").strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "…"
        text = (
            "🔄 <b>Авто-ответ</b>\n\n"
            "Когда я <b>оффлайн</b> и приходит личное сообщение — бот отправляет ответ.\n"
            "Только ЛС, не группы и не боты. Один ответ на контакт раз в кулдаун.\n\n"
            "<b>Режимы</b>:\n"
            "• <b>заготовленный</b> — отправляется один и тот же текст (ниже).\n"
            "• <b>умный</b> — LLM пишет короткий ответ в твоём стиле, опираясь на контекст переписки.\n\n"
            f"Статус: <b>{'ВКЛ' if s.auto_reply_enabled else 'ВЫКЛ'}</b>\n"
            f"Режим: <b>{mode_label}</b>\n"
            f"Кулдаун: <b>{s.auto_reply_cooldown_min} мин</b>\n"
            f"Текст заготовки:\n<i>«{snippet}»</i>"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.auto_reply_enabled)} Включить авто-ответ",
                callback_data="set:tog:auto_reply_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=("• " if s.auto_reply_mode == "static" else "") + "📝 Заготовка",
                callback_data="set:choose:auto_reply_mode:static",
            ),
            InlineKeyboardButton(
                text=("• " if s.auto_reply_mode == "smart" else "") + "🤖 Умный",
                callback_data="set:choose:auto_reply_mode:smart",
            ),
        )
        kb.row(
            InlineKeyboardButton(
                text="✏ Изменить текст заготовки",
                callback_data="set:input:auto_reply_text",
            )
        )
        kb.row(
            *[
                InlineKeyboardButton(
                    text=("• " if s.auto_reply_cooldown_min == m else "") + f"{m}м",
                    callback_data=f"set:choose:auto_reply_cooldown_min:{m}",
                )
                for m in (5, 15, 30, 60)
            ]
        )
        kb.row(*_back_row())

    elif section == "digest":
        text = (
            "☀ <b>Утренний дайджест</b>\n\n"
            "Раз в сутки в указанное время получаю сводку: что произошло за ночь, кто ждёт ответа, "
            "горящие обещания и сколько было авто-ответов.\n\n"
            f"Статус: <b>{'ВКЛ' if s.digest_enabled else 'ВЫКЛ'}</b>\n"
            f"Время: <b>{s.digest_time}</b> · {tz_short(s.timezone)}\n\n"
            "Часовой пояс — отдельный раздел в /settings.\n"
            "Для разовой сводки — команда /digest"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.digest_enabled)} Включить дайджест",
                callback_data="set:tog:digest_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"⏰ Время: {s.digest_time}", callback_data="set:input:digest_time"
            )
        )
        kb.row(*_back_row())

    elif section == "reminders":
        text = (
            "⏰ <b>Напоминания о дедлайнах</b>\n\n"
            "Бот подгружает обещания из переписок (см. /todos и кнопку «Задачи» в /chat) и пинает, "
            "когда дедлайн близок или просрочен.\n\n"
            f"Статус: <b>{'ВКЛ' if s.reminders_enabled else 'ВЫКЛ'}</b>\n"
            f"Заранее за: <b>{s.reminder_lead_hours} ч</b>\n"
            f"Алерт о просрочках: <b>{'ВКЛ' if s.reminder_overdue_enabled else 'ВЫКЛ'}</b>"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.reminders_enabled)} Включить напоминания",
                callback_data="set:tog:reminders_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.reminder_overdue_enabled)} Алерт при просрочке",
                callback_data="set:tog:reminder_overdue_enabled",
            )
        )
        kb.row(
            *[
                InlineKeyboardButton(
                    text=("• " if s.reminder_lead_hours == h else "") + f"{h}ч",
                    callback_data=f"set:choose:reminder_lead_hours:{h}",
                )
                for h in (1, 2, 4, 12, 24)
            ]
        )
        kb.row(*_back_row())

    elif section == "smart_digest":
        text = (
            "📊 <b>Smart дайджест</b>\n\n"
            "Входящие сообщения за последние N минут собираются в один дайджест "
            "с группировкой по срочности (🔴 срочное → 🟡 важное → 🟢 обычное).\n\n"
            f"Smart дайджест: <b>{'ВКЛ' if s.smart_digest_enabled else 'ВЫКЛ'}</b>\n"
            f"Интервал: <b>{s.smart_digest_interval_min} мин</b>\n"
            f"Мгновенные 🔴 уведомления: <b>{'ВКЛ' if s.urgent_notify_enabled else 'ВЫКЛ'}</b>\n\n"
            "Мгновенные уведомления приходят сразу при получении срочного сообщения.\n"
            "Дайджест собирает все сообщения за интервал и присылает единый отчёт.\n"
            "Ручной запуск: /smart_digest"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.smart_digest_enabled)} Включить smart дайджест",
                callback_data="set:tog:smart_digest_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.urgent_notify_enabled)} Мгновенные 🔴 уведомления",
                callback_data="set:tog:urgent_notify_enabled",
            )
        )
        kb.row(
            *[
                InlineKeyboardButton(
                    text=("• " if s.smart_digest_interval_min == m else "") + f"{m}мин",
                    callback_data=f"set:choose:smart_digest_interval_min:{m}",
                )
                for m in (15, 30, 60, 120)
            ]
        )
        kb.row(*_back_row())

    elif section == "news":
        text = (
            "📰 <b>Новости</b>\n\n"
            "Команда <code>/news тема</code> ищет посты в твоих подписанных каналах за последние N часов и "
            "собирает структурированный обзор.\n\n"
            "<b>Авто-новости</b> (этот тогглер): если включено, каждое утро в указанное время бот шлёт "
            "дайджест по каждой теме из <b>/news_topics</b>.\n\n"
            "Чтобы ограничить выборку конкретными каналами — /news_channels.\n\n"
            f"Авто-новости: <b>{'ВКЛ' if s.news_enabled else 'ВЫКЛ'}</b>\n"
            f"Время отправки: <b>{s.news_digest_time}</b> · {tz_short(s.timezone)}\n"
            f"Окно по умолчанию: <b>{s.news_window_hours} ч</b>"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.news_enabled)} Включить авто-новости",
                callback_data="set:tog:news_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"⏰ Время: {s.news_digest_time}",
                callback_data="set:input:news_time",
            )
        )
        kb.row(
            *[
                InlineKeyboardButton(
                    text=("• " if s.news_window_hours == h else "") + f"{h}ч",
                    callback_data=f"set:choose:news_window_hours:{h}",
                )
                for h in (6, 12, 24, 48, 72)
            ]
        )
        kb.row(
            InlineKeyboardButton(
                text="📋 Темы → /news_topics", callback_data="set:noop:news_topics"
            )
        )
        kb.row(*_back_row())

    elif section == "llm":
        active = (
            LLMDefaults.OPENAI_CHAT_HEAVY
            if s.use_heavy_model and s.llm_provider == "openai"
            else LLMDefaults.OPENAI_CHAT_LIGHT
            if s.llm_provider == "openai"
            else LLMDefaults.GEMINI_CHAT_HEAVY
            if s.use_heavy_model and s.llm_provider == "gemini"
            else LLMDefaults.GEMINI_CHAT_LIGHT
            if s.llm_provider == "gemini"
            else LLMDefaults.MISTRAL_CHAT_HEAVY
            if s.use_heavy_model
            else LLMDefaults.MISTRAL_CHAT_LIGHT
        )
        text = (
            "🤖 <b>LLM-провайдер</b>\n\n"
            "Кто отвечает на запросы и пишет черновики/саммари. Лёгкая модель — для рутины, "
            "тяжёлая — для длинных переписок и сложного анализа.\n\n"
            f"Провайдер: <b>{s.llm_provider}</b>\n"
            f"Режим: <b>{'тяжёлая' if s.use_heavy_model else 'лёгкая'}</b>\n"
            f"Активная модель: <code>{active}</code>"
        )
        kb.row(
            InlineKeyboardButton(
                text=("• " if s.llm_provider == "openai" else "") + "OpenAI",
                callback_data="set:choose:llm_provider:openai",
            ),
            InlineKeyboardButton(
                text=("• " if s.llm_provider == "gemini" else "") + "Gemini",
                callback_data="set:choose:llm_provider:gemini",
            ),
        )
        kb.row(
            InlineKeyboardButton(
                text=("• " if s.llm_provider == "mistral" else "")
                + "Mistral (бесплатно)",
                callback_data="set:choose:llm_provider:mistral",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.use_heavy_model)} Тяжёлая модель",
                callback_data="set:tog:use_heavy_model",
            )
        )
        kb.row(*_back_row())

    elif section == "transcription":
        api_provider = getattr(s, "transcription_api_provider", "openai")
        labels = {
            "openai": "OpenAI Whisper",
            "gemini": "Gemini (бесплатно)",
            "mistral": "Mistral (бесплатно)",
        }
        api_label = labels.get(api_provider, "OpenAI Whisper")
        text = (
            "🎤 <b>Транскрипция голосовых и аудио</b>\n\n"
            "<b>local</b> — faster-whisper на твоей машине (бесплатно, приватно, нужны ресурсы).\n"
            "<b>api</b> — облачная транскрипция через выбранный сервис.\n"
            "<b>hybrid</b> — local, fallback в API при ошибке.\n\n"
            f"Режим: <b>{s.transcription_mode}</b>\n"
            f"API-провайдер: <b>{api_label}</b>"
        )
        for mode in ("local", "api", "hybrid"):
            kb.row(
                InlineKeyboardButton(
                    text=("• " if s.transcription_mode == mode else "") + mode,
                    callback_data=f"set:choose:transcription_mode:{mode}",
                )
            )
        for prov in ("openai", "gemini", "mistral"):
            prov_label = labels.get(prov, prov)
            kb.row(
                InlineKeyboardButton(
                    text=("• " if api_provider == prov else "") + prov_label,
                    callback_data=f"set:choose:transcription_api_provider:{prov}",
                )
            )
        kb.row(*_back_row())

    elif section == "tz":
        text = (
            "🌍 <b>Часовой пояс</b>\n\n"
            "От него отталкиваются:\n"
            "• время утреннего дайджеста и авто-новостей\n"
            "• отображение дедлайнов в /todos и напоминаниях\n"
            "• временные метки в дайджестах\n\n"
            f"Сейчас: <b>{tz_short(s.timezone)}</b>\n\n"
            "Тапни пресет ниже или введи свой IANA-таймзону кнопкой «Другой…»."
        )
        # пресеты по 2 в ряд
        for i in range(0, len(TZ_PRESETS), 2):
            buttons = []
            for tz in TZ_PRESETS[i : i + 2]:
                mark = "• " if s.timezone == tz else ""
                buttons.append(
                    InlineKeyboardButton(text=mark + tz, callback_data=f"set:tz:{tz}")
                )
            kb.row(*buttons)
        kb.row(
            InlineKeyboardButton(text="✏ Другой…", callback_data="set:input:timezone")
        )
        kb.row(*_back_row())

    elif section == "privacy":
        text = (
            "🛡 <b>Приватность и видимость</b>\n\n"
            "Что бот <b>смотрит и обрабатывает</b> по умолчанию.\n\n"
            "<b>Игнорировать архив</b> — чаты в архиве Telegram не подгружаются ни в /chat, "
            "ни в /search, ни в /news, ни в авто-ответ. Включено по умолчанию.\n\n"
            f"Игнорировать архив: <b>{'ВКЛ' if s.ignore_archived else 'ВЫКЛ'}</b>\n\n"
            "<i>Изменения вступают в силу для следующих запросов. Архивный статус подтягивается "
            "при /sync.</i>"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.ignore_archived)} Игнорировать архив",
                callback_data="set:tog:ignore_archived",
            )
        )
        kb.row(*_back_row())

    elif section == "sync":
        sync_enabled = getattr(s, "auto_sync_enabled", True)
        sync_sec = getattr(s, "auto_sync_interval_sec", 7200)
        auto_mem = getattr(s, "auto_extract_memories", False)
        saved_msgs = getattr(s, "include_saved_messages", False)
        if sync_sec >= 3600:
            intv = f"{sync_sec // 3600}ч"
        elif sync_sec >= 60:
            intv = f"{sync_sec // 60}м"
        else:
            intv = f"{sync_sec}с"
        text = (
            "🔄 <b>Синхронизация и разведка</b>\n\n"
            "Раз в указанный интервал бот обновляет список контактов и архивный статус.\n\n"
            f"Авто-синк: <b>{'ВКЛ' if sync_enabled else 'ВЫКЛ'}</b> · {intv}\n"
            f"Авто-память: <b>{'ВКЛ' if auto_mem else 'ВЫКЛ'}</b> (после синка извлекает факты без вопроса)\n"
            f"Избранное: <b>{'ВКЛ' if saved_msgs else 'ВЫКЛ'}</b> (индексировать и искать в Избранном)"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(sync_enabled)} Включить авто-синк",
                callback_data="set:tog:auto_sync_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(auto_mem)} Авто-извлечение памяти",
                callback_data="set:tog:auto_extract_memories",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(saved_msgs)} Индексировать Избранное",
                callback_data="set:tog:include_saved_messages",
            )
        )
        for v, label in [
            (60, "1м"),
            (300, "5м"),
            (1800, "30м"),
            (3600, "1ч"),
            (7200, "2ч"),
            (14400, "4ч"),
            (86400, "24ч"),
        ]:
            kb.row(
                InlineKeyboardButton(
                    text=("• " if sync_sec == v else "") + label,
                    callback_data=f"set:choose:auto_sync_interval_sec:{v}",
                )
            )
        kb.row(
            InlineKeyboardButton(
                text="✏ Свой интервал…", callback_data="set:input:auto_sync_interval"
            )
        )
        kb.row(*_back_row())

    elif section == "drafts":
        text = (
            "✍️ <b>Авто-черновики</b>\n\n"
            "Когда приходит новое сообщение — бот может автоматически предложить черновик ответа "
            "с кнопками «Отправить / Редактировать / Игнорировать».\n\n"
            "• <b>Только важные</b> — черновик предлагается только для срочных/важных сообщений "
            "(классификация по тексту).\n"
            "• <b>Лимит</b> — макс. черновиков в час, чтобы не спамить.\n\n"
            f"Статус: <b>{'ВКЛ' if s.draft_suggestions_enabled else 'ВЫКЛ'}</b>\n"
            f"Только важные: <b>{'ВКЛ' if s.draft_only_important else 'ВЫКЛ'}</b>\n"
            f"Лимит: <b>{s.draft_max_per_hour} в час</b>"
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.draft_suggestions_enabled)} Включить авто-черновики",
                callback_data="set:tog:draft_suggestions_enabled",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.draft_only_important)} Только важные",
                callback_data="set:tog:draft_only_important",
            )
        )
        kb.row(
            *[
                InlineKeyboardButton(
                    text=("• " if s.draft_max_per_hour == m else "") + f"{m}/ч",
                    callback_data=f"set:choose:draft_max_per_hour:{m}",
                )
                for m in (3, 5, 10)
            ]
        )
        kb.row(*_back_row())

    elif section == "keys":
        text = (
            "🔑 <b>API-ключи</b>\n\n"
            "Хранятся зашифрованными (Fernet). Можно перезаписать в любой момент.\n\n"
            f"OpenAI: {_check(bool(openai_key))}\n"
            f"Gemini: {_check(bool(gemini_key))}\n"
            f"Mistral: {_check(bool(mistral_key))}"
        )
        kb.row(
            InlineKeyboardButton(
                text="🔑 OpenAI key", callback_data="set:input:openai_key"
            ),
            InlineKeyboardButton(
                text="🔑 Gemini key", callback_data="set:input:gemini_key"
            ),
        )
        kb.row(
            InlineKeyboardButton(
                text="🔑 Mistral key", callback_data="set:input:mistral_key"
            )
        )
        kb.row(*_back_row())

    elif section == "auto_mode":
        mode_labels = {
            "offline_only": "🌙 Только когда оффлайн",
            "always": "🔄 Всегда отвечать",
            "smart": "🧠 Умный режим (по срочности)",
        }
        qh_start = s.quiet_hours_start or "не задано"
        qh_end = s.quiet_hours_end or "не задано"
        close_contacts = _check(s.auto_reply_close_contacts)
        notify = _check(s.notify_on_auto_reply)

        text = (
            "🤖 <b>Авто-режим</b>\n\n"
            "Определяет, когда и как бот отвечает на сообщения.\n\n"
            f"Режим: <b>{mode_labels.get(s.auto_mode, s.auto_mode)}</b>\n"
            f"🔕 Тихие часы: <b>{qh_start} – {qh_end}</b>\n"
            f"{close_contacts} Авто-ответ близким контактам\n"
            f"{notify} Уведомлять об авто-ответах"
        )

        for mode in ("offline_only", "always", "smart"):
            prefix = "• " if s.auto_mode == mode else ""
            kb.button(
                text=f"{prefix}{mode_labels[mode]}",
                callback_data=f"set:choose:auto_mode:{mode}",
            )
        kb.adjust(1)

        kb.row(
            InlineKeyboardButton(
                text="🔕 Начало тихих часов",
                callback_data="set:input:quiet_hours_start",
            ),
            InlineKeyboardButton(
                text="🔕 Конец тихих часов", callback_data="set:input:quiet_hours_end"
            ),
        )
        kb.row(
            InlineKeyboardButton(
                text=f"{_check(s.auto_reply_close_contacts)} Авто-ответ близким",
                callback_data="set:tog:auto_reply_close_contacts",
            ),
            InlineKeyboardButton(
                text=f"{_check(s.notify_on_auto_reply)} Уведомлять об авто-ответах",
                callback_data="set:tog:notify_on_auto_reply",
            ),
        )
        kb.row(*_back_row())

    elif section == "folders":
        async with get_session() as session:
            folders_data = await list_folders(session, owner)

        monitored = json.loads(s.monitored_folders) if s.monitored_folders else []

        lines = ["📁 <b>Мониторинг папок</b>", ""]

        if not folders_data:
            lines.append("⚠️ Папки не найдены. Сделай /sync.")
        else:
            for f in folders_data:
                icon = "✅" if f.title in monitored else "⬜"
                lines.append(f"{icon} {f.emoji or '📂'} {f.title}")
            lines.append("")
            lines.append("Нажимай на папку чтобы включить/выключить мониторинг.")

        monitor_only = "✅" if s.monitor_only_selected_folders else "⬜"
        lines.append(f"{monitor_only} Мониторить ТОЛЬКО выбранные папки")

        text = "\n".join(lines)

        for f in folders_data:
            icon = "✅" if f.title in monitored else "⬜"
            kb.button(
                text=f"{icon} {f.emoji or '📂'} {f.title}",
                callback_data=f"set:folder:tog:{f.title}",
            )

        kb.row(
            InlineKeyboardButton(
                text=f"{'✅' if s.monitor_only_selected_folders else '⬜'} Только выбранные",
                callback_data="set:tog:monitor_only_selected_folders",
            )
        )
        kb.row(
            InlineKeyboardButton(
                text="🔄 Обновить папки", callback_data="set:folder:refresh"
            )
        )
        kb.row(*_back_row())

    else:
        text = "Раздел не найден."
        kb.row(*_back_row())

    return text, kb.as_markup()


# ---------- FSM-вводы ----------


@router.callback_query(F.data == "set:input:openai_key")
async def cb_input_openai(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_openai_key)
    await callback.message.answer(
        "Пришли OpenAI API key (начинается с <code>sk-</code>). Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Поддерживается несколько ключей через запятую: <code>key1, key2, key3</code>\n"
        "При ошибке 429 (превышение лимита) бот автоматически переключится на следующий ключ."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:gemini_key")
async def cb_input_gemini(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_gemini_key)
    await callback.message.answer(
        "Пришли Gemini API key с <code>aistudio.google.com</code>. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Поддерживается несколько ключей через запятую: <code>key1, key2, key3</code>\n"
        "При ошибке 429 (превышение лимита) бот автоматически переключится на следующий ключ."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:mistral_key")
async def cb_input_mistral(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_mistral_key)
    await callback.message.answer(
        "Пришли Mistral API key с <code>console.mistral.ai</code>. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Поддерживается несколько ключей через запятую: <code>key1, key2, key3</code>\n"
        "При ошибке 429 (превышение лимита) бот автоматически переключится на следующий ключ."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:digest_time")
async def cb_input_digest(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_digest_time)
    await callback.message.answer(
        "Введи время в формате <code>HH:MM</code> (UTC). /cancel — отмена."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:auto_reply_text")
async def cb_input_auto_reply(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_auto_reply_text)
    await callback.message.answer(
        "Пришли новый текст автоответа. Будет отправляться, когда ты оффлайн "
        "(в режиме «заготовка»). /cancel — отмена."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:auto_sync_interval")
async def cb_input_sync_interval(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_sync_interval)
    await callback.message.answer(
        "Введи интервал в секундах (минимум 30). Например: 3600 = 1 час, 7200 = 2 часа. /cancel — отмена."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:news_time")
async def cb_input_news_time(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_news_time)
    await callback.message.answer(
        "Введи время утренних авто-новостей в <code>HH:MM</code> (UTC). /cancel — отмена."
    )
    await callback.answer()


@router.callback_query(F.data == "set:noop:news_topics")
async def cb_noop_news_topics(callback: CallbackQuery) -> None:
    await callback.answer("Открой /news_topics в меню команд", show_alert=True)


@router.callback_query(F.data.startswith("set:folder:tog:"))
async def cb_folder_toggle(callback: CallbackQuery) -> None:
    folder_name = callback.data.split(":", 3)[3]

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        s = owner.settings

        monitored = json.loads(s.monitored_folders) if s.monitored_folders else []

        if folder_name in monitored:
            monitored.remove(folder_name)
        else:
            monitored.append(folder_name)

        s.monitored_folders = json.dumps(monitored, ensure_ascii=False)
        await session.flush()

    await _refresh_section(callback, "folders")
    await callback.answer()


@router.callback_query(F.data == "set:folder:refresh")
async def cb_folder_refresh(callback: CallbackQuery) -> None:
    client = get_active_telethon_client(callback.from_user.id)
    if client:
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
        await sync_dialogs(client, owner, limit=500)
        await callback.answer("✅ Папки обновлены!")
    else:
        mgr = get_userbot_manager()
        if mgr is None:
            await callback.answer("❌ Userbot не запущен", show_alert=True)
        else:
            await callback.answer("❌ Сначала /login", show_alert=True)

    await _refresh_section(callback, "folders")


@router.callback_query(F.data.startswith("set:tz:"))
async def cb_pick_tz(callback: CallbackQuery) -> None:
    tz_value = callback.data[len("set:tz:") :]
    if not is_valid_tz(tz_value):
        await callback.answer("Неизвестный TZ", show_alert=True)
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        owner.settings.timezone = tz_value
    await callback.answer(f"TZ: {tz_value}")
    await _refresh_section(callback, "tz")


@router.callback_query(F.data == "set:input:timezone")
async def cb_input_tz(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_timezone)
    await callback.message.answer(
        "Введи название часового пояса в формате IANA, например <code>Europe/Moscow</code> или "
        "<code>Asia/Tashkent</code>. /cancel — отмена."
    )
    await callback.answer()


@router.message(SettingsStates.waiting_openai_key)
async def step_openai_key(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    if not raw:
        await message.answer("Пустой ключ. Повтори или /cancel.")
        return
    parts = [k.strip() for k in raw.split(",") if k.strip()]
    if not parts:
        await message.answer("Нет ни одного непустого ключа. Повтори или /cancel.")
        return
    try:
        await message.delete()
    except Exception:
        logger.exception("failed to delete message with openai key")
    # Валидируем первый ключ как индикатор; остальные считаем рабочими
    if not await OpenAIProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "openai", ",".join(parts))
    await state.clear()
    count = len(parts)
    await message.answer(f"✅ Сохранено OpenAI ключей: {count}.")


@router.message(SettingsStates.waiting_gemini_key)
async def step_gemini_key(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    if not raw:
        await message.answer("Пустой ключ. Повтори или /cancel.")
        return
    parts = [k.strip() for k in raw.split(",") if k.strip()]
    if not parts:
        await message.answer("Нет ни одного непустого ключа. Повтори или /cancel.")
        return
    try:
        await message.delete()
    except Exception:
        logger.exception("failed to delete message with gemini key")
    # Валидируем первый ключ как индикатор; остальные считаем рабочими
    if not await GeminiProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "gemini", ",".join(parts))
    await state.clear()
    count = len(parts)
    await message.answer(f"✅ Сохранено Gemini ключей: {count}.")


@router.message(SettingsStates.waiting_mistral_key)
async def step_mistral_key(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    if not raw:
        await message.answer("Пустой ключ. Повтори или /cancel.")
        return
    parts = [k.strip() for k in raw.split(",") if k.strip()]
    if not parts:
        await message.answer("Нет ни одного непустого ключа. Повтори или /cancel.")
        return
    try:
        await message.delete()
    except Exception:
        logger.exception("failed to delete message with mistral key")
    # Валидируем первый ключ как индикатор; остальные считаем рабочими
    if not await MistralProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "mistral", ",".join(parts))
    await state.clear()
    count = len(parts)
    await message.answer(f"✅ Сохранено Mistral ключей: {count}.")


@router.message(SettingsStates.waiting_digest_time)
async def step_digest_time(message: Message, state: FSMContext) -> None:
    hm = (message.text or "").strip()
    if not HM_RE.match(hm):
        await message.answer(
            "Формат HH:MM, например <code>06:30</code>. Повтори или /cancel."
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.digest_time = hm
    await state.clear()
    await message.answer(f"✅ Время дайджеста: <b>{hm} UTC</b>.")


@router.message(SettingsStates.waiting_news_time)
async def step_news_time(message: Message, state: FSMContext) -> None:
    hm = (message.text or "").strip()
    if not HM_RE.match(hm):
        await message.answer(
            "Формат HH:MM, например <code>07:30</code>. Повтори или /cancel."
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.news_digest_time = hm
        tz = owner.settings.timezone
    await state.clear()
    await message.answer(f"✅ Время авто-новостей: <b>{hm}</b> · {tz_short(tz)}.")


@router.message(SettingsStates.waiting_auto_reply_text)
async def step_auto_reply_text(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустой текст. Повтори или /cancel.")
        return
    if len(text) > 1000:
        await message.answer(
            "Слишком длинно (макс. 1000 символов). Повтори или /cancel."
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.auto_reply_text = text
    await state.clear()
    await message.answer(f"✅ Текст автоответа сохранён:\n<i>«{text}»</i>")


@router.message(SettingsStates.waiting_timezone)
async def step_timezone(message: Message, state: FSMContext) -> None:
    tz_value = (message.text or "").strip()
    if not is_valid_tz(tz_value):
        await message.answer(
            "Не нашёл такой TZ. Используй IANA-формат, например <code>Europe/Moscow</code>. "
            "Список: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones — "
            "колонка «TZ identifier». /cancel — отмена."
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.timezone = tz_value
    await state.clear()
    await message.answer(f"✅ Часовой пояс: <b>{tz_short(tz_value)}</b>")


@router.message(SettingsStates.waiting_sync_interval)
async def step_sync_interval(message: Message, state: FSMContext) -> None:
    val = (message.text or "").strip()
    if not val.isdigit():
        await message.answer("Ожидаю число (секунд). Повтори или /cancel.")
        return
    secs = max(30, int(val))
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.auto_sync_interval_sec = secs
    await state.clear()
    await message.answer(f"✅ Интервал авто-синка: <b>{secs} сек</b>")


# ---------- FSM: тихие часы ----------


@router.callback_query(F.data == "set:input:quiet_hours_start")
async def cb_input_quiet_hours_start(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await callback.message.answer(
        "Введи время начала тихих часов (HH:MM, например 23:00):"
    )
    await state.set_state(SettingsStates.waiting_quiet_hours_start)
    await callback.answer()


@router.callback_query(F.data == "set:input:quiet_hours_end")
async def cb_input_quiet_hours_end(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.message.answer(
        "Введи время конца тихих часов (HH:MM, например 07:00):"
    )
    await state.set_state(SettingsStates.waiting_quiet_hours_end)
    await callback.answer()


@router.message(SettingsStates.waiting_quiet_hours_start)
async def step_quiet_hours_start(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if not HM_RE.match(text):
        await message.answer("❌ Неверный формат. Введи HH:MM (например 23:00):")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.quiet_hours_start = text
        await session.flush()
    await state.clear()
    await message.answer(f"✅ Тихие часы начало: <b>{text}</b>")


@router.message(SettingsStates.waiting_quiet_hours_end)
async def step_quiet_hours_end(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if not HM_RE.match(text):
        await message.answer("❌ Неверный формат. Введи HH:MM (например 07:00):")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.quiet_hours_end = text
        await session.flush()
    await state.clear()
    await message.answer(f"✅ Тихие часы конец: <b>{text}</b>")
