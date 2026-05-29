"""/settings — главное меню и разделы. callback_data: set:sec / set:tog / set:choose / set:input."""

import io
import json
import logging
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.bot.states import SettingsStates
from src.config import settings

from src.core.infra.key_guard import safe_str
from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.timeutil import HM_RE, TZ_PRESETS, is_valid_tz, tz_short
from src.core.intelligence.adaptive_persona import (
    reset_persona_to_snapshot,
)
from src.db.repo import (
    add_key_slot,
    get_api_key,
    get_or_create_user,
    get_persona,
    list_folders,
    list_key_slots,
    upsert_api_key,
)
from src.db.session import get_session
from src.userbot.dialogs import sync_dialogs
from src.userbot import get_active_telethon_client, get_userbot_manager
from src.llm.cloudflare_provider import CloudflareProvider
from src.llm.deepseek_provider import DeepSeekProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.grok_provider import GrokProvider
from src.llm.groq_provider import GroqProvider
from src.llm.mimo_provider import MIMO_REGIONS, MiMoProvider
from src.llm.mistral_provider import MistralProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.custom_provider import CustomProvider


logger = logging.getLogger(__name__)
router = Router(name="settings")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


def _check(value: bool) -> str:
    return "✅" if value else "❌"


SEARCHABLE_SETTINGS: dict[str, str] = {
    # Раздел: Часовой пояс
    "timezone": "Часовой пояс (IANA, напр. Europe/Moscow)",
    # Раздел: Авто-ответ
    "auto_reply_enabled": "Включить авто-ответ при оффлайн",
    "auto_reply_cooldown_min": "Кулдаун между авто-ответами (мин)",
    "auto_reply_mode": "Режим авто-ответа (заготовка/умный)",
    "auto_reply_text": "Текст заготовки для авто-ответа",
    "auto_reply_close_contacts": "Авто-ответ только близким контактам",
    "notify_on_auto_reply": "Уведомлять об отправленных авто-ответах",
    # Раздел: Авто-режим
    "auto_mode": "Режим работы (оффлайн/всегда/умный)",
    "quiet_hours_start": "Начало тихих часов",
    "quiet_hours_end": "Конец тихих часов",
    # Раздел: Дайджест
    "digest_enabled": "Включить утренний дайджест",
    "digest_time": "Время отправки дайджеста (UTC)",
    # Раздел: Напоминания
    "reminders_enabled": "Включить напоминания о дедлайнах",
    "reminder_lead_hours": "За сколько часов напоминать о дедлайне",
    "reminder_overdue_enabled": "Алерт при просрочке дедлайна",
    # Раздел: Smart-дайджест
    "smart_digest_enabled": "Включить smart дайджест",
    "smart_digest_interval_min": "Интервал smart дайджеста (мин)",
    "urgent_notify_enabled": "Мгновенные уведомления о срочных сообщениях",
    # Раздел: Новости
    "news_enabled": "Включить авто-новости",
    "news_digest_time": "Время отправки авто-новостей",
    "news_window_hours": "Окно поиска новостей (ч)",
    # Раздел: LLM
    "llm_provider": "LLM-провайдер (openai/gemini/mistral/cloudflare/openrouter)",
    "use_heavy_model": "Использовать тяжёлую модель LLM",
    # Раздел: Транскрипция
    "transcription_mode": "Режим транскрипции (local/api/hybrid)",
    "transcription_api_provider": "API-провайдер транскрипции",
    # Раздел: Черновики
    "draft_suggestions_enabled": "Включить авто-черновики ответов",
    "draft_only_important": "Черновики только для важных сообщений",
    "draft_max_per_hour": "Максимум черновиков в час",
    # Раздел: Приватность
    "ignore_archived": "Игнорировать архивные чаты",
    # Раздел: Синхронизация
    "auto_sync_enabled": "Включить авто-синхронизацию",
    "auto_sync_interval_sec": "Интервал авто-синхронизации (сек)",
    "auto_extract_memories": "Авто-извлечение памяти из переписок",
    "include_saved_messages": "Индексировать Избранное (Saved Messages)",
    # Раздел: API-ключи
    "openai_key": "API ключ OpenAI",
    "gemini_key": "API ключ Gemini",
    "mistral_key": "API ключ Mistral",
    "cloudflare_key": "API ключ Cloudflare",
    "deepseek_key": "API ключ DeepSeek",
    # Раздел: Модели
    "model_overrides": "Переопределения моделей по типу задач",
    # Раздел: Папки
    "monitored_folders": "Отслеживаемые папки Telegram",
    "monitor_only_selected_folders": "Мониторить только выбранные папки",
    # Раздел: Личность
    "alias": "Псевдоним (обращение к владельцу)",
    "custom_instructions": "Пользовательские инструкции для личности",
    "base_tone": "Базовый тон личности",
    "warmth": "Теплота общения (low/normal/high)",
    "enthusiasm": "Восторженность (low/normal/high)",
    "headings_lists": "Заголовки и списки (low/normal/high)",
    "emoji_level": "Уровень использования эмодзи (low/normal/high)",
    "adaptive_mode_enabled": "Адаптивный режим личности",
    # Anti-AI
    "anti_ai_enabled": "Включить Anti-AI защиту",
    "anti_ai_mode": "Режим Anti-AI (off/log/fix)",
}


# ---------- Главное меню ----------


async def _render_menu(telegram_id: int) -> tuple[str, InlineKeyboardMarkup]:
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        s = owner.settings
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")
        cloudflare_key = await get_api_key(session, owner, "cloudflare")
        deepseek_key = await get_api_key(session, owner, "deepseek")
        grok_key = await get_api_key(session, owner, "grok")
        mimo_key = await get_api_key(session, owner, "mimo")
        groq_key = await get_api_key(session, owner, "groq")
        custom_key = await get_api_key(session, owner, "custom")

        # Also check LlmKeySlot for custom providers
        try:
            all_slots = await list_key_slots(session, owner)
            has_custom_slots = any(
                s.provider
                not in {
                    "openai",
                    "gemini",
                    "mistral",
                    "deepseek",
                    "cloudflare",
                    "grok",
                    "mimo",
                    "groq",
                }
                and s.enabled
                for s in all_slots
            )
        except Exception:
            has_custom_slots = False

        custom_ok = bool(custom_key) or has_custom_slots

        # ── Extract ORM values to local vars (session-safe) ──────────
        _tz = s.timezone
        _auto_reply_enabled = s.auto_reply_enabled
        _auto_reply_cooldown_min = s.auto_reply_cooldown_min
        _auto_sync_enabled = getattr(s, "auto_sync_enabled", True)
        _auto_sync_interval_sec = getattr(s, "auto_sync_interval_sec", 7200)
        _auto_extract_memories = getattr(s, "auto_extract_memories", False)
        _include_saved_messages = getattr(s, "include_saved_messages", False)
        _digest_enabled = s.digest_enabled
        _digest_time = s.digest_time
        _reminders_enabled = s.reminders_enabled
        _reminder_lead_hours = s.reminder_lead_hours
        _reminder_overdue_enabled = s.reminder_overdue_enabled
        _news_enabled = s.news_enabled
        _news_window_hours = s.news_window_hours
        _ignore_archived = s.ignore_archived
        _smart_digest_enabled = getattr(s, "smart_digest_enabled", False)
        _smart_digest_interval_min = getattr(s, "smart_digest_interval_min", 30)
        _llm_provider = s.llm_provider
        _use_heavy_model = s.use_heavy_model
        _transcription_mode = s.transcription_mode
        _transcription_api_provider = getattr(s, "transcription_api_provider", "openai")

    text = (
        "⚙ <b>Настройки</b>\n\n"
        f"🌍 Часовой пояс: <b>{tz_short(_tz)}</b>\n"
        f"🔄 Авто: ответ {_check(_auto_reply_enabled)} ({_auto_reply_cooldown_min}м) · синк {_check(_auto_sync_enabled)} ({_auto_sync_interval_sec}с)\n"
        f"🧠 Авто-память: {_check(_auto_extract_memories)}\n"
        f"⭐ Избранное: {_check(_include_saved_messages)}\n"
        f"☀ Дайджест: {_check(_digest_enabled)} ({_digest_time}) · smart: {_check(_smart_digest_enabled)} ({_smart_digest_interval_min}м)\n"
        f"⏰ Напоминания: {_check(_reminders_enabled)} (за {_reminder_lead_hours}ч; просрочки {_check(_reminder_overdue_enabled)})\n"
        f"📰 Новости: {_check(_news_enabled)} (окно {_news_window_hours}ч)\n"
        f"🛡 Игнорировать архив: {_check(_ignore_archived)}\n"
        f"🧠 LLM: <b>{_llm_provider}</b> · {'тяжёлая' if _use_heavy_model else 'лёгкая'} · tr: {_transcription_mode}\n"
        f"🔑 Ключи: OpenAI {_check(bool(openai_key))} · Gemini {_check(bool(gemini_key))} · Mistral {_check(bool(mistral_key))} · DeepSeek {_check(bool(deepseek_key))} · Cloudflare {_check(bool(cloudflare_key))} · Grok {_check(bool(grok_key))} · MiMo {_check(bool(mimo_key))} · Groq {_check(bool(groq_key))} · Свой {_check(bool(custom_key))}\n\n"
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
        InlineKeyboardButton(text="🧠 LLM и модели", callback_data="set:sec:brain"),
    )
    kb.row(
        InlineKeyboardButton(text="✍️ Черновики", callback_data="set:sec:drafts"),
        InlineKeyboardButton(text="🔒 Приватность", callback_data="set:sec:privacy"),
    )
    kb.row(
        InlineKeyboardButton(text="🔄 Синхронизация", callback_data="set:sec:sync"),
        InlineKeyboardButton(text="🔑 API-ключи", callback_data="set:sec:keys"),
    )
    kb.row(InlineKeyboardButton(text="📬 Треды", callback_data="thread:refresh"))
    kb.row(InlineKeyboardButton(text="🧠 Полный анализ", callback_data="set:analyze"))
    kb.row(
        InlineKeyboardButton(text="🎭 Личность", callback_data="set:sec:personality")
    )
    kb.row(
        InlineKeyboardButton(
            text="📤 Экспорт конфига", callback_data="set:export_config"
        ),
        InlineKeyboardButton(
            text="📥 Импорт конфига", callback_data="set:import_config"
        ),
    )
    kb.row(InlineKeyboardButton(text="❌ Закрыть", callback_data="set:close"))
    # Быстрые тогглы (авто-память, избранное, дайджест, авто-ответ)
    text += "\n⚡ <b>Быстрые тогглы:</b>"
    kb.row(
        InlineKeyboardButton(
            text=f"🧠 Авто-память {_check(_auto_extract_memories)}",
            callback_data="set:tog:auto_extract_memories",
        ),
        InlineKeyboardButton(
            text=f"⭐ Избранное {_check(_include_saved_messages)}",
            callback_data="set:tog:include_saved_messages",
        ),
    )
    kb.row(
        InlineKeyboardButton(
            text=f"☀ Дайджест {_check(_digest_enabled)}",
            callback_data="set:tog:digest_enabled",
        ),
        InlineKeyboardButton(
            text=f"🔄 Авто-ответ {_check(_auto_reply_enabled)}",
            callback_data="set:tog:auto_reply_enabled",
        ),
    )
    return text, kb.as_markup()


@router.message(Command("settings"))
async def cmd_settings(message: Message, command: CommandObject) -> None:
    """/settings поиск <keyword> — быстрый поиск по настройкам."""
    args = (command.args or "").strip()
    if args.startswith("поиск "):
        query = args[6:].strip().lower()
        if not query:
            await message.answer("Использование: /settings поиск <ключевое слово>")
            return

        results = []
        for key, desc in SEARCHABLE_SETTINGS.items():
            if query in key.lower() or query in desc.lower():
                results.append(f"• <b>{key}</b> — {desc}")

        if results:
            await message.answer(
                f"🔍 Результаты по «{query}»:\n\n" + "\n".join(results[:15])
            )
        else:
            await message.answer(f"❌ Ничего не найдено по «{sanitize_html(query)}».")
        return

    text, kb = await _render_menu(message.from_user.id)
    await message.answer(text, reply_markup=kb)


async def _safe_edit(message, text: str, kb) -> None:
    # глушит безобидное "message is not modified" при повторном тапе той же опции
    if message is None:
        return
    try:
        await message.edit_text(text, reply_markup=kb)
    except TelegramBadRequest as e:
        if "not modified" not in safe_str(e).lower():
            raise


@router.callback_query(F.data == "set:menu")
async def cb_menu(callback: CallbackQuery) -> None:
    text, kb = await _render_menu(callback.from_user.id)
    await _safe_edit(callback.message, text, kb)
    await callback.answer()


@router.callback_query(F.data.startswith("settings:back"))
async def cb_settings_back(callback: CallbackQuery) -> None:
    parts = callback.data.split(":", 2)
    parent = parts[2] if len(parts) > 2 else "menu"
    if parent == "menu":
        await _show_main_menu(callback)
    else:
        text, kb = await _render_section(callback.from_user.id, parent)
        await _safe_edit(callback.message, text, kb)
    await callback.answer()


async def _show_main_menu(callback: CallbackQuery) -> None:
    text, kb = await _render_menu(callback.from_user.id)
    await _safe_edit(callback.message, text, kb)


@router.callback_query(F.data == "set:close")
async def cb_close(callback: CallbackQuery) -> None:
    if callback.message:
        await callback.message.delete()
    await callback.answer()


# ---------- Экспорт / Импорт конфига ----------


@router.callback_query(F.data == "set:export_config")
async def cb_export_config(callback: CallbackQuery) -> None:
    """Экспорт всех настроек бота в JSON-файл."""
    await callback.answer("📤 Готовлю экспорт...")

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        s = owner.settings

        # model_overrides в БД — JSON-строка, парсим для экспорта
        try:
            overrides = json.loads(s.model_overrides) if s.model_overrides else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}

        # Собираем настройки
        config = {
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "settings": {
                "llm_provider": s.llm_provider,
                "use_heavy_model": s.use_heavy_model,
                "transcription_mode": s.transcription_mode,
                "transcription_api_provider": getattr(
                    s, "transcription_api_provider", "openai"
                ),
                "anti_ai_enabled": getattr(s, "anti_ai_enabled", False),
                "anti_ai_mode": getattr(s, "anti_ai_mode", "off"),
                "adaptive_mode_enabled": getattr(s, "adaptive_mode_enabled", False),
                "auto_sync_enabled": getattr(s, "auto_sync_enabled", True),
                "auto_extract_memories": getattr(s, "auto_extract_memories", False),
                "include_saved_messages": getattr(s, "include_saved_messages", False),
                "monitor_only_selected_folders": getattr(
                    s, "monitor_only_selected_folders", False
                ),
                "monitored_folders": s.monitored_folders,
                "timezone": s.timezone,
                "auto_reply_close_contacts": getattr(
                    s, "auto_reply_close_contacts", False
                ),
                "smart_digest_enabled": getattr(s, "smart_digest_enabled", False),
                "urgent_notify_enabled": getattr(s, "urgent_notify_enabled", False),
                "digest_time": s.digest_time,
                "auto_sync_interval_sec": getattr(s, "auto_sync_interval_sec", 7200),
            },
            "model_overrides": overrides,
            "keys": [],
        }

        # Собираем ключи
        slots = await list_key_slots(session, owner)
        for slot in slots:
            if slot.enabled:
                config["keys"].append(
                    {
                        "provider": slot.provider,
                        "purpose": slot.purpose,
                        "model": slot.model,
                        "endpoint": slot.endpoint,
                        "category": slot.category,
                        "label": slot.label,
                        "priority": slot.priority,
                        "key_enc": slot.key_enc,  # ключ уже зашифрован — экспортируем как есть
                    }
                )

        json_str = json.dumps(config, ensure_ascii=False, indent=2)
        bio = io.BytesIO(json_str.encode("utf-8"))
        bio.seek(0)

        await callback.message.answer_document(
            FSInputFile(bio, "telegram_helper_config.json"),
            caption="📤 Твой конфиг бота. Сохрани этот файл.\n\n"
            "Для восстановления используй 📥 Импорт конфига в настройках.",
        )


@router.callback_query(F.data == "set:import_config")
async def cb_import_config(callback: CallbackQuery, state: FSMContext) -> None:
    """Запуск импорта — просим прислать файл."""
    await state.set_state(SettingsStates.waiting_config_import)
    await callback.message.answer(
        "📥 Пришли JSON-файл конфига (telegram_helper_config.json).\n/cancel — отмена."
    )
    await callback.answer()


@router.message(SettingsStates.waiting_config_import, F.document)
async def step_import_config(message: Message, state: FSMContext) -> None:
    """Обрабатываем загруженный конфиг-файл."""
    if (
        not message.document
        or not message.document.file_name
        or not message.document.file_name.endswith(".json")
    ):
        await message.answer("❌ Нужен .json файл. Попробуй ещё раз или /cancel.")
        return

    await message.answer("📥 Импортирую конфиг...")

    try:
        # Скачиваем файл
        file = await message.bot.get_file(message.document.file_id)
        bio = io.BytesIO()
        await message.bot.download_file(file.file_path, bio)
        config = json.loads(bio.getvalue().decode("utf-8"))

        if "version" not in config:
            await message.answer("❌ Невалидный файл конфига (нет version).")
            await state.clear()
            return

        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            s = owner.settings

            # Восстанавливаем настройки
            settings_data = config.get("settings", {})
            for key, value in settings_data.items():
                if hasattr(s, key) and value is not None:
                    setattr(s, key, value)

            # Восстанавливаем model_overrides (сериализуем в JSON-строку)
            overrides = config.get("model_overrides", {})
            if overrides:
                s.model_overrides = json.dumps(overrides, ensure_ascii=False)

            # Восстанавливаем ключи
            from src.db.models._auth import LlmKeySlot

            imported_keys = config.get("keys", [])
            existing = await list_key_slots(session, owner)

            for key_data in imported_keys:
                # Проверяем — нет ли уже такого ключа
                duplicate = False
                for existing_slot in existing:
                    if existing_slot.provider == key_data[
                        "provider"
                    ] and existing_slot.purpose == key_data.get("purpose", "main"):
                        duplicate = True
                        break

                if duplicate:
                    continue  # пропускаем дубликаты

                # Создаём слот напрямую — key_enc уже зашифрован, не пропускаем через encrypt()
                slot = LlmKeySlot(
                    user_id=owner.id,
                    provider=key_data["provider"],
                    purpose=key_data.get("purpose", "main"),
                    model=key_data.get("model"),
                    endpoint=key_data.get("endpoint"),
                    category=key_data.get("category", "llm"),
                    label=key_data.get("label"),
                    priority=key_data.get("priority", 0),
                    key_enc=key_data["key_enc"],  # уже зашифрован тем же encryption_key
                )
                session.add(slot)

            await session.commit()

        count = len(imported_keys)
        await message.answer(
            f"✅ Конфиг импортирован!\n"
            f"⚙️ Настроек: {len(settings_data)}\n"
            f"🔑 Ключей: {count}\n"
            f"📋 Переопределений моделей: {len(overrides)}\n\n"
            f"Проверь настройки в /settings.",
        )

    except json.JSONDecodeError:
        await message.answer("❌ Файл повреждён — невалидный JSON.")
    except Exception as e:
        await message.answer(f"❌ Ошибка импорта: {e}")
    finally:
        await state.clear()


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
    "adaptive_mode_enabled",
    "anti_ai_enabled",
}

CHOICE_KEYS = {
    "llm_provider": {
        "openrouter",
        "openai",
        "gemini",
        "mistral",
        "cloudflare",
        "deepseek",
        "grok",
        "mimo",
        "groq",
        "custom",
    },
    "transcription_mode": {"local", "api", "hybrid"},
    "transcription_api_provider": {"openai", "gemini", "mistral"},
    "auto_reply_mode": {"static", "smart"},
    "auto_mode": {"offline_only", "always", "smart"},
    # Личность (ChatGPT-style)
    "base_tone": {
        "default",
        "professional",
        "friendly",
        "frank",
        "whimsical",
        "efficient",
        "cynical",
    },
    "warmth": {"low", "normal", "high"},
    "enthusiasm": {"low", "normal", "high"},
    "headings_lists": {"low", "normal", "high"},
    "emoji_level": {"low", "normal", "high"},
    "anti_ai_mode": {"off", "log", "fix"},
}

NUMERIC_KEYS = {
    "auto_reply_cooldown_min",
    "reminder_lead_hours",
    "news_window_hours",
    "auto_sync_interval_sec",
    "draft_max_per_hour",
    "smart_digest_interval_min",
}

# Ключи, которые относятся к AdaptivePersona (не к owner.settings)
PERSONA_KEYS = frozenset(
    {
        "base_tone",
        "warmth",
        "enthusiasm",
        "headings_lists",
        "emoji_level",
        "adaptive_mode_enabled",
    }
)


@router.callback_query(F.data.startswith("set:tog:"))
async def cb_toggle(callback: CallbackQuery) -> None:
    key = callback.data.split(":", 2)[2]
    if key not in BOOL_KEYS:
        await callback.answer("Неизвестный переключатель", show_alert=True)
        return

    # adaptive_mode_enabled — специальный случай (на AdaptivePersona)
    if key == "adaptive_mode_enabled":
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            p = await get_persona(session, owner)
            p.adaptive_mode_enabled = not p.adaptive_mode_enabled
            await session.flush()
        # Инвалидируем кэш ПОСЛЕ коммита
        from src.core.context_cache import invalidate

        await invalidate(f"persona:{callback.from_user.id}")
        await callback.answer(
            f"Адаптивный режим {'✅ ВКЛ' if p.adaptive_mode_enabled else '❌ ВЫКЛ'}"
        )
        await _refresh_section(callback, "personality")
        return

    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            current = getattr(owner.settings, key)
            setattr(owner.settings, key, not current)
    except AttributeError:
        await callback.answer("Ошибка: настройка не найдена", show_alert=True)
        return
    await callback.answer("Готово")
    await _refresh_section(callback, _section_for_key(key))


@router.callback_query(F.data.startswith("set:choose:"))
async def cb_choose(callback: CallbackQuery) -> None:
    parts = callback.data.split(":", 3)
    if len(parts) < 4:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    _, _, key, value = parts

    # Поля личности (на AdaptivePersona, не на UserSettings)
    PERSONALITY_FIELDS = {
        "base_tone",
        "warmth",
        "enthusiasm",
        "headings_lists",
        "emoji_level",
    }
    if key in PERSONALITY_FIELDS:
        valid_values = CHOICE_KEYS.get(key, set())
        if value not in valid_values:
            await callback.answer("Невалидное значение", show_alert=True)
            return
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            p = await get_persona(session, owner)
            setattr(p, key, value)
            await session.flush()
        # Инвалидируем кэш ПОСЛЕ коммита
        from src.core.context_cache import invalidate

        await invalidate(f"persona:{callback.from_user.id}")
        await callback.answer("Готово")
        await _refresh_section(callback, "personality")
        return

    if key in CHOICE_KEYS:
        if value not in CHOICE_KEYS[key]:
            await callback.answer("Невалидное значение", show_alert=True)
            return
        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            if key in PERSONA_KEYS:
                persona = await get_persona(session, owner)
                setattr(persona, key, value)
            else:
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
        "llm_provider": "brain",
        "use_heavy_model": "brain",
        "transcription_mode": "brain",
        "transcription_api_provider": "brain",
        "draft_suggestions_enabled": "drafts",
        "draft_only_important": "drafts",
        "draft_max_per_hour": "drafts",
        "auto_mode": "auto_mode",
        "auto_reply_close_contacts": "auto_mode",
        "notify_on_auto_reply": "auto_mode",
        # ChatGPT-style personality
        "base_tone": "personality",
        "warmth": "personality",
        "enthusiasm": "personality",
        "headings_lists": "personality",
        "emoji_level": "personality",
        "adaptive_mode_enabled": "personality",
        "anti_ai_enabled": "personality",
        "anti_ai_mode": "personality",
        "monitor_only_selected_folders": "privacy",
        "auto_sync_enabled": "sync",
        "auto_extract_memories": "sync",
        "include_saved_messages": "sync",
        "smart_digest_enabled": "smart_digest",
        "urgent_notify_enabled": "smart_digest",
        "auto_sync_interval_sec": "sync",
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


def _back_row(parent: str = "menu"):
    return [
        InlineKeyboardButton(text="🔙 Назад", callback_data=f"settings:back:{parent}")
    ]


async def _render_section(
    telegram_id: int, section: str
) -> tuple[str, InlineKeyboardMarkup]:
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        s = owner.settings
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")
        cloudflare_key = await get_api_key(session, owner, "cloudflare")
        deepseek_key = await get_api_key(session, owner, "deepseek")
        grok_key = await get_api_key(session, owner, "grok")
        mimo_key = await get_api_key(session, owner, "mimo")
        groq_key = await get_api_key(session, owner, "groq")
        custom_key = await get_api_key(session, owner, "custom")

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
                    text=("• " if s.auto_reply_mode == "static" else "")
                    + "📝 Заготовка",
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
                    text=f"⏰ Время: {s.digest_time}",
                    callback_data="set:input:digest_time",
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
                        text=("• " if s.smart_digest_interval_min == m else "")
                        + f"{m}мин",
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

        elif section == "brain":
            # ── LLM provider ──
            # Display-friendly model names per provider
            _provider_model_names = {
                "openai": ("gpt-5-mini", "gpt-5.5"),
                "gemini": ("gemini-3-flash", "gemini-3.1-pro"),
                "mistral": ("mistral-small-latest", "mistral-medium-latest"),
                "cloudflare": (
                    "@cf/qwen/qwen3-30b-a3b-fp8",
                    "@cf/moonshotai/kimi-k2.6",
                ),
                "deepseek": ("deepseek-chat", "deepseek-reasoner"),
                "grok": ("grok-4.3", "grok-4.20-0309-reasoning"),
                "mimo": ("mimo-v2-flash", "mimo-v2.5-pro"),
                "groq": ("llama-3.3-70b-versatile", "mixtral-8x7b-32768"),
            }
            _names = _provider_model_names.get(s.llm_provider, ("?", "?"))
            active = (
                "DeepSeek V4 Flash (бесплатно)"
                if s.llm_provider == "openrouter"
                else _names[1]
                if s.use_heavy_model
                else _names[0]
            )

            # ── Transcription ──
            api_provider = getattr(s, "transcription_api_provider", "openai")
            tts_labels = {
                "openai": "OpenAI Whisper",
                "gemini": "Gemini (бесплатно)",
                "mistral": "Mistral (бесплатно)",
            }
            api_label = tts_labels.get(api_provider, "OpenAI Whisper")

            text = (
                "🧠 <b>LLM и модели</b>\n\n"
                "━━━ 🤖 Провайдер ━━━\n"
                f"Провайдер: <b>{s.llm_provider}</b>\n"
                f"Режим: <b>{'тяжёлая' if s.use_heavy_model else 'лёгкая'}</b>\n"
                f"Модель: <code>{active}</code>\n\n"
                "━━━ 🎤 Транскрипция ━━━\n"
                f"Режим: <b>{s.transcription_mode}</b> · {api_label}\n\n"
                "━━━ 🧠 Модели задач ━━━\n"
                "<i>Настрой модель под каждую задачу</i>"
            )

            # ── LLM provider buttons ──
            # Показываем если есть кастомные оверрайды
            try:
                overrides = json.loads(s.model_overrides) if s.model_overrides else {}
            except (json.JSONDecodeError, TypeError):
                overrides = {}
            if overrides:
                ov_count = len(overrides)
                text += (
                    f"\n⚠️ <b>Активны переопределения ({ov_count} задач)</b> — "
                    "нажми «🧠 Модели задач» чтобы посмотреть"
                )

            # ── LLM provider buttons ──
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
                    text=("• " if s.llm_provider == "openrouter" else "")
                    + "🔥 DeepSeek (free)",
                    callback_data="set:choose:llm_provider:openrouter",
                ),
                InlineKeyboardButton(
                    text=("• " if s.llm_provider == "mistral" else "")
                    + "Mistral (free)",
                    callback_data="set:choose:llm_provider:mistral",
                ),
            )
            kb.row(
                InlineKeyboardButton(
                    text=("• " if s.llm_provider == "cloudflare" else "")
                    + "Cloudflare",
                    callback_data="set:choose:llm_provider:cloudflare",
                ),
            )
            # DeepSeek, Grok, MiMo, Groq — дополнительные провайдеры
            kb.row(
                InlineKeyboardButton(
                    text=("• " if s.llm_provider == "deepseek" else "") + "DeepSeek",
                    callback_data="set:choose:llm_provider:deepseek",
                ),
                InlineKeyboardButton(
                    text=("• " if s.llm_provider == "grok" else "") + "Grok (xAI)",
                    callback_data="set:choose:llm_provider:grok",
                ),
            )
            kb.row(
                InlineKeyboardButton(
                    text=("• " if s.llm_provider == "mimo" else "") + "MiMo (Xiaomi)",
                    callback_data="set:choose:llm_provider:mimo",
                ),
                InlineKeyboardButton(
                    text=("• " if s.llm_provider == "groq" else "") + "Groq",
                    callback_data="set:choose:llm_provider:groq",
                ),
            )
            # Свой провайдер — показываем все кастомные
            try:
                custom_slots = await list_key_slots(session, owner)
                custom_names = sorted(
                    {
                        s.provider
                        for s in custom_slots
                        if s.provider
                        not in {
                            "openai",
                            "gemini",
                            "mistral",
                            "deepseek",
                            "cloudflare",
                            "grok",
                            "mimo",
                            "groq",
                            "openrouter",
                        }
                        and s.enabled
                    }
                )
            except Exception:
                custom_names = []
            if custom_names:
                for cn in custom_names:
                    kb.row(
                        InlineKeyboardButton(
                            text=("• " if s.llm_provider == cn else "") + cn,
                            callback_data=f"set:choose:llm_provider:{cn}",
                        )
                    )
            kb.row(
                InlineKeyboardButton(
                    text=f"{_check(s.use_heavy_model)} Тяжёлая модель",
                    callback_data="set:tog:use_heavy_model",
                )
            )

            # ── Transcription buttons ──
            for mode in ("local", "api", "hybrid"):
                kb.row(
                    InlineKeyboardButton(
                        text=("• " if s.transcription_mode == mode else "") + mode,
                        callback_data=f"set:choose:transcription_mode:{mode}",
                    )
                )
            for prov in ("openai", "gemini", "mistral"):
                prov_label = tts_labels.get(prov, prov)
                kb.row(
                    InlineKeyboardButton(
                        text=("• " if api_provider == prov else "") + prov_label,
                        callback_data=f"set:choose:transcription_api_provider:{prov}",
                    )
                )

            # ── Возможности AI (глобальные, из .env) ──
            emb_on = settings.embedding_enabled
            vis_on = settings.vision_enabled
            aud_on = settings.audio_enabled
            tts_on = settings.tts_enabled
            auto_on = settings.auto_select_model

            text += (
                "\n\n⚙️ <b>Возможности AI (глобальные):</b>\n"
                f"🔤 Embedding: {'✅' if emb_on else '❌'}  👁️ Vision: {'✅' if vis_on else '❌'}\n"
                f"🎤 STT/Audio: {'✅' if aud_on else '❌'}  🔊 TTS: {'✅' if tts_on else '❌'}\n"
                f"🤖 Авто-выбор: {'✅' if auto_on else '❌'}\n"
                f"<i>Настрой через .env / переменные окружения</i>"
            )

            # ── Контекст Maestro ──
            try:
                from src.core.context.engine import ContextEngine

                stats = (
                    ContextEngine.get_load_stats()
                    if hasattr(ContextEngine, "get_load_stats")
                    else None
                )
            except Exception:
                stats = None
            if stats:
                pct = min(100, int(stats.get("used_pct", 0)))
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                text += (
                    f"\n\n📊 <b>Контекст Maestro:</b> [{bar}] {pct}%\n"
                    f"   Память: {stats.get('memory_tokens', 0)} · Вектор: {stats.get('vector_tokens', 0)} · Wiki: {stats.get('wiki_tokens', 0)}"
                )
            else:
                text += "\n\n📊 <b>Контекст Maestro:</b> будет доступен после запуска"

            # ── Model overrides ──
            kb.row(
                InlineKeyboardButton(
                    text="🧠 Настроить модели задач →",
                    callback_data="set:sec:models_brain",
                )
            )
            kb.row(*_back_row())

        elif section == "models_brain":
            try:
                overrides = json.loads(s.model_overrides) if s.model_overrides else {}
            except (json.JSONDecodeError, TypeError):
                overrides = {}

            task_labels = {
                "maestro": "🎭 Maestro (оркестрация)",
                "draft": "✍️ Черновики",
                "memory": "🧠 Память",
                "search": "🔍 Поиск",
                "stt": "🎤 Распознавание речи",
                "humanize": "✨ Хуманайзер",
                "classify": "🏷 Классификация",
                "summarize": "📝 Саммари",
                "skills": "🛠 Навыки",
                "background": "🌙 Фоновые задачи",
                "default": "💬 Обычный чат",
            }

            lines = ["🧠 <b>Модели для задач</b>", ""]
            for task_type, label in task_labels.items():
                override = overrides.get(task_type)
                model_str = (
                    f"<code>{override}</code>" if override else "<i>по умолчанию</i>"
                )
                lines.append(f"{label}: {model_str}")
            lines.append("")
            lines.append(
                "<i>Нажми на задачу, чтобы выбрать модель. "
                "Переопределения имеют приоритет над LLM-провайдером.</i>"
            )
            text = "\n".join(lines)

            for task_type, label in task_labels.items():
                kb.row(
                    InlineKeyboardButton(
                        text=label, callback_data=f"set:model:{task_type}"
                    )
                )
            kb.row(
                InlineKeyboardButton(
                    text="🗑 Сбросить все", callback_data="set:model:reset_all"
                )
            )
            kb.row(*_back_row("brain"))

        elif section.startswith("model_sel:"):
            # Подменю выбора модели для конкретного task_type
            task_type = section.split(":", 1)[1]

            task_labels = {
                "maestro": "🎭 Maestro (оркестрация)",
                "draft": "✍️ Черновики",
                "memory": "🧠 Память",
                "search": "🔍 Поиск",
                "stt": "🎤 Распознавание речи",
                "humanize": "✨ Хуманайзер",
                "classify": "🏷 Классификация",
                "summarize": "📝 Саммари",
                "skills": "🛠 Навыки",
                "background": "🌙 Фоновые задачи",
                "default": "💬 Обычный чат",
            }
            task_label = task_labels.get(task_type, task_type)

            try:
                overrides = json.loads(s.model_overrides) if s.model_overrides else {}
            except (json.JSONDecodeError, TypeError):
                overrides = {}

            current = overrides.get(task_type)

            # Собираем доступные модели из ВСЕХ ключей пользователя
            slots = await list_key_slots(session, owner)
            provider_models: dict[str, set[str]] = {}
            for slot in slots:
                if not slot.enabled:
                    continue
                if slot.provider not in provider_models:
                    provider_models[slot.provider] = set()
                if slot.model:
                    provider_models[slot.provider].add(slot.model)

            from src.llm.provider_catalog import get_provider

            available_models: list[str] = []
            seen: set[str] = set()
            for provider, models in provider_models.items():
                if models:
                    for model in sorted(models):
                        key = f"{provider}/{model}"
                        if key not in seen:
                            seen.add(key)
                            available_models.append(key)
                else:
                    pi = get_provider(provider)
                    if pi:
                        for model in pi.models:
                            key = f"{provider}/{model}"
                            if key not in seen:
                                seen.add(key)
                                available_models.append(key)

            available_models.sort()

            # Если ничего нет — показываем каталог основного провайдера
            if not available_models:
                pi = get_provider(s.llm_provider)
                if pi and pi.models:
                    available_models = [f"{s.llm_provider}/{m}" for m in pi.models]

            lines = [
                f"🧠 <b>Модель для: {task_label}</b>",
                "",
                f"Текущая: <code>{current}</code>"
                if current
                else "Текущая: <i>по умолчанию</i>",
                "",
            ]
            text = "\n".join(lines)

            # Кнопка «По умолчанию» (удаляет override)
            kb.row(
                InlineKeyboardButton(
                    text=("• " if not current else "") + "🔄 По умолчанию",
                    callback_data=f"set:model:set:{task_type}:__default__",
                )
            )
            # Кнопки моделей из каталога
            for model in available_models:
                # current may be old (bare model) or new (provider/model) format
                is_selected = current and (
                    current == model or model.endswith(f"/{current}")
                )
                mark = "• " if is_selected else ""
                kb.row(
                    InlineKeyboardButton(
                        text=f"{mark}{model}",
                        callback_data=f"set:model:set:{task_type}:{model}",
                    )
                )
            # Кнопка ручного ввода
            kb.row(
                InlineKeyboardButton(
                    text="✏ Ввести вручную…",
                    callback_data=f"set:model:custom:{task_type}",
                )
            )
            # Удалить override (если есть)
            if current:
                kb.row(
                    InlineKeyboardButton(
                        text="🗑 Удалить переопределение",
                        callback_data=f"set:model:del:{task_type}",
                    )
                )
            kb.row(*_back_row("models_brain"))

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
                        InlineKeyboardButton(
                            text=mark + tz, callback_data=f"set:tz:{tz}"
                        )
                    )
                kb.row(*buttons)
            kb.row(
                InlineKeyboardButton(
                    text="✏ Другой…", callback_data="set:input:timezone"
                )
            )
            kb.row(*_back_row())

        elif section == "privacy":
            folders_data = await list_folders(session, owner)

            try:
                monitored = (
                    json.loads(s.monitored_folders) if s.monitored_folders else []
                )
            except json.JSONDecodeError:
                monitored = []

            text = (
                "🛡 <b>Приватность и видимость</b>\n\n"
                "Что бот <b>смотрит и обрабатывает</b> по умолчанию.\n\n"
                "<b>Игнорировать архив</b> — чаты в архиве Telegram не подгружаются ни в /chat, "
                "ни в /search, ни в /news, ни в авто-ответ. Включено по умолчанию.\n\n"
                f"Игнорировать архив: <b>{'ВКЛ' if s.ignore_archived else 'ВЫКЛ'}</b>\n\n"
                "<i>Изменения вступают в силу для следующих запросов. Архивный статус подтягивается "
                "при /sync.</i>\n\n"
                "━━━ 📁 <b>Мониторинг папок</b> ━━━"
            )
            kb.row(
                InlineKeyboardButton(
                    text=f"{_check(s.ignore_archived)} Игнорировать архив",
                    callback_data="set:tog:ignore_archived",
                )
            )

            if not folders_data:
                text += "\n\n⚠️ Папки не найдены. Сделай /sync."
            else:
                for f in folders_data:
                    icon = "✅" if f.title in monitored else "⬜"
                    kb.button(
                        text=f"{icon} {f.emoji or '📂'} {f.title}",
                        callback_data=f"set:folder:tog:{f.title}",
                    )
                text += (
                    "\n\n<i>Нажимай на папку чтобы включить/выключить мониторинг.</i>"
                )

            monitor_only = "✅" if s.monitor_only_selected_folders else "⬜"
            kb.row(
                InlineKeyboardButton(
                    text=f"{monitor_only} Только выбранные",
                    callback_data="set:tog:monitor_only_selected_folders",
                )
            )
            kb.row(
                InlineKeyboardButton(
                    text="🔄 Обновить папки", callback_data="set:folder:refresh"
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
                    text="✏ Свой интервал…",
                    callback_data="set:input:auto_sync_interval",
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
            # Load custom key slots from LlmKeySlot
            custom_slots = await list_key_slots(session, owner)
            custom_providers = {
                s.provider
                for s in custom_slots
                if s.provider
                not in {
                    "openai",
                    "gemini",
                    "mistral",
                    "deepseek",
                    "cloudflare",
                    "grok",
                    "mimo",
                    "groq",
                }
                and s.enabled
            }

            text = (
                "🔑 <b>API-ключи</b>\n\n"
                "Хранятся зашифрованными (Fernet). Можно перезаписать в любой момент.\n\n"
                f"OpenAI: {_check(bool(openai_key))}\n"
                f"Gemini: {_check(bool(gemini_key))}\n"
                f"Mistral: {_check(bool(mistral_key))}\n"
                f"DeepSeek: {_check(bool(deepseek_key))}\n"
                f"Cloudflare: {_check(bool(cloudflare_key))}\n"
                f"Grok: {_check(bool(grok_key))}\n"
                f"MiMo: {_check(bool(mimo_key))}\n"
                f"Groq: {_check(bool(groq_key))}\n"
                f"Свой: {_check(bool(custom_key))}"
            )
            if custom_providers:
                text += "\n\n🛠 <b>Кастомные провайдеры:</b>"
                for cp in sorted(custom_providers):
                    text += f"\n{cp}: ✅"
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
                ),
                InlineKeyboardButton(
                    text="🔑 DeepSeek key", callback_data="set:input:deepseek_key"
                ),
            )
            kb.row(
                InlineKeyboardButton(
                    text="🔑 Cloudflare key", callback_data="set:input:cloudflare_key"
                ),
            )
            kb.row(
                InlineKeyboardButton(
                    text="🔑 Grok key", callback_data="set:input:grok_key"
                ),
                InlineKeyboardButton(
                    text="🔑 MiMo key", callback_data="set:input:mimo_key"
                ),
            )
            kb.row(
                InlineKeyboardButton(
                    text="🔑 Groq key", callback_data="set:input:groq_key"
                ),
            )
            kb.row(
                InlineKeyboardButton(
                    text="➕ Свой провайдер", callback_data="set:input:custom_name"
                ),
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
                    text="🔕 Конец тихих часов",
                    callback_data="set:input:quiet_hours_end",
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

        elif section == "personality":
            from src.db.repo import get_persona

            p = await get_persona(session, owner)

            tone_labels = {
                "default": "По умолчанию",
                "professional": "Профессиональный",
                "friendly": "Дружелюбный",
                "frank": "Откровенный",
                "whimsical": "Причудливый",
                "efficient": "Эффективный",
                "cynical": "Циничный",
            }
            level_labels = {"low": "Менее", "normal": "По умолчанию", "high": "Более"}
            anti_ai_mode_labels = {"off": "Выкл", "log": "Лог", "fix": "Исправлять"}
            current_tone = tone_labels.get(p.base_tone, "По умолчанию")

            text = (
                "🎭 <b>Личность</b>\n\n"
                "<b>Базовый стиль и тон</b>\n"
                f"Сейчас: <b>{current_tone}</b>\n\n"
                "<b>Характеристики</b>\n"
                f"🔥 Теплый: <b>{level_labels.get(p.warmth, '—')}</b>\n"
                f"⚡ Восторженный: <b>{level_labels.get(p.enthusiasm, '—')}</b>\n"
                f"📋 Заголовки и списки: <b>{level_labels.get(p.headings_lists, '—')}</b>\n"
                f"😊 Эмодзи: <b>{level_labels.get(p.emoji_level, '—')}</b>\n\n"
                f"📝 Инструкции: {'есть' if p.custom_instructions else 'нет'}\n"
                f"👤 Псевдоним: {p.alias or 'не задан'}\n"
                f"🧠 Адаптивный режим: <b>{'ВКЛ' if p.adaptive_mode_enabled else 'ВЫКЛ'}</b>\n"
                f"🛡️ Anti-AI: <b>{'ВКЛ' if s.anti_ai_enabled else 'ВЫКЛ'}</b>"
                f" ({anti_ai_mode_labels.get(s.anti_ai_mode, '—')})"
            )

            # -- Базовый тон (выпадающий список) --
            for tone_key, tone_label in tone_labels.items():
                prefix = "• " if p.base_tone == tone_key else ""
                kb.button(
                    text=f"{prefix}{tone_label}",
                    callback_data=f"set:choose:base_tone:{tone_key}",
                )
            kb.adjust(2)

            # -- Характеристики: Теплый --
            kb.row(
                InlineKeyboardButton(
                    text="🔥 Теплый",
                    callback_data="set:noop:warmth",
                )
            )
            kb.row(
                *[
                    InlineKeyboardButton(
                        text=("• " if p.warmth == lvl else "") + label,
                        callback_data=f"set:choose:warmth:{lvl}",
                    )
                    for lvl, label in [
                        ("low", "Менее"),
                        ("normal", "По умолч."),
                        ("high", "Более"),
                    ]
                ]
            )

            # -- Характеристики: Восторженный --
            kb.row(
                InlineKeyboardButton(
                    text="⚡ Восторженный",
                    callback_data="set:noop:enthusiasm",
                )
            )
            kb.row(
                *[
                    InlineKeyboardButton(
                        text=("• " if p.enthusiasm == lvl else "") + label,
                        callback_data=f"set:choose:enthusiasm:{lvl}",
                    )
                    for lvl, label in [
                        ("low", "Менее"),
                        ("normal", "По умолч."),
                        ("high", "Более"),
                    ]
                ]
            )

            # -- Характеристики: Заголовки и списки --
            kb.row(
                InlineKeyboardButton(
                    text="📋 Заголовки и списки",
                    callback_data="set:noop:headings_lists",
                )
            )
            kb.row(
                *[
                    InlineKeyboardButton(
                        text=("• " if p.headings_lists == lvl else "") + label,
                        callback_data=f"set:choose:headings_lists:{lvl}",
                    )
                    for lvl, label in [
                        ("low", "Менее"),
                        ("normal", "По умолч."),
                        ("high", "Более"),
                    ]
                ]
            )

            # -- Характеристики: Эмодзи --
            kb.row(
                InlineKeyboardButton(
                    text="😊 Эмодзи",
                    callback_data="set:noop:emoji_level",
                )
            )
            kb.row(
                *[
                    InlineKeyboardButton(
                        text=("• " if p.emoji_level == lvl else "") + label,
                        callback_data=f"set:choose:emoji_level:{lvl}",
                    )
                    for lvl, label in [
                        ("low", "Менее"),
                        ("normal", "По умолч."),
                        ("high", "Более"),
                    ]
                ]
            )

            # -- Пользовательские инструкции --
            kb.row(
                InlineKeyboardButton(
                    text="📝 Изменить инструкции",
                    callback_data="set:input:custom_instructions",
                )
            )

            # -- Псевдоним --
            kb.row(
                InlineKeyboardButton(
                    text="👤 Изменить псевдоним",
                    callback_data="set:input:alias",
                )
            )

            # -- Адаптивный режим --
            kb.row(
                InlineKeyboardButton(
                    text=f"{'✅' if p.adaptive_mode_enabled else '❌'} Адаптивный режим",
                    callback_data="set:tog:adaptive_mode_enabled",
                )
            )

            # -- Anti-AI защита --
            kb.row(
                InlineKeyboardButton(
                    text=f"{'✅' if s.anti_ai_enabled else '❌'} Anti-AI защита",
                    callback_data="set:tog:anti_ai_enabled",
                )
            )

            # -- Anti-AI режим --
            kb.row(
                InlineKeyboardButton(
                    text="⚙️ Режим Anti-AI",
                    callback_data="set:noop:anti_ai_mode",
                )
            )
            kb.row(
                *[
                    InlineKeyboardButton(
                        text=("\u2022 " if s.anti_ai_mode == mode else "") + label,
                        callback_data=f"set:choose:anti_ai_mode:{mode}",
                    )
                    for mode, label in [
                        ("off", "Выкл"),
                        ("log", "Лог"),
                        ("fix", "Исправлять"),
                    ]
                ]
            )

            # -- Сброс --
            kb.row(
                InlineKeyboardButton(
                    text="↩ Сбросить к базовым",
                    callback_data="set:persona:reset",
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
        "Пришли OpenAI API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:gemini_key")
async def cb_input_gemini(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_gemini_key)
    await callback.message.answer(
        "Пришли Gemini API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:mistral_key")
async def cb_input_mistral(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_mistral_key)
    await callback.message.answer(
        "Пришли Mistral API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:cloudflare_key")
async def cb_input_cloudflare(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_cloudflare_key)
    await callback.message.answer(
        "Пришли Cloudflare API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:deepseek_key")
async def cb_input_deepseek(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_deepseek_key)
    await callback.message.answer(
        "Пришли DeepSeek API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:grok_key")
async def cb_input_grok(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_grok_key)
    await callback.message.answer(
        "Пришли Grok API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:mimo_key")
async def cb_input_mimo(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_mimo_key)
    await callback.message.answer(
        "Пришли MiMo API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:groq_key")
async def cb_input_groq(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_groq_key)
    await callback.message.answer(
        "Пришли Groq API-ключ. Проверю и сохраню. /cancel — отмена.\n\n"
        "💡 Можно несколько ключей через запятую."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:custom_name")
async def cb_input_custom_name(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_custom_name)
    await callback.message.answer(
        "➕ <b>Свой провайдер</b>\n\n"
        "Шаг 1/4: Пришли название провайдера (например: <code>Local LLM</code>).\n"
        "/cancel — отмена."
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

        try:
            monitored = json.loads(s.monitored_folders) if s.monitored_folders else []
        except json.JSONDecodeError:
            monitored = []

        if folder_name in monitored:
            monitored.remove(folder_name)
        else:
            monitored.append(folder_name)

        s.monitored_folders = json.dumps(monitored, ensure_ascii=False)
        await session.flush()

    await _refresh_section(callback, "privacy")
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

    await _refresh_section(callback, "privacy")


# ---------- Модели: callback'и ----------


@router.callback_query(F.data == "set:model:reset_all")
async def cb_model_reset_all(callback: CallbackQuery) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        owner.settings.model_overrides = None
        await session.flush()
    await callback.answer("🗑 Все переопределения моделей сброшены")
    await _refresh_section(callback, "models_brain")


@router.callback_query(F.data.startswith("set:model:set:"))
async def cb_model_set(callback: CallbackQuery) -> None:
    """set:model:set:<task_type>:<model_name>"""
    parts = callback.data.split(":")
    # parts: ["set", "model", "set", task_type, ...model_parts]
    if len(parts) < 5:
        await callback.answer("Ошибка данных", show_alert=True)
        return
    task_type = parts[3]
    model_name = ":".join(parts[4:])  # модели могут содержать ":" (напр. @cf/...)

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        s = owner.settings
        try:
            overrides = json.loads(s.model_overrides) if s.model_overrides else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}

        if model_name == "__default__":
            overrides.pop(task_type, None)
        else:
            overrides[task_type] = model_name

        s.model_overrides = (
            json.dumps(overrides, ensure_ascii=False) if overrides else None
        )
        await session.flush()

    display = model_name if model_name != "__default__" else "по умолчанию"
    await callback.answer(f"✅ {task_type} → {display}")
    await _refresh_section(callback, f"model_sel:{task_type}")


@router.callback_query(F.data.startswith("set:model:del:"))
async def cb_model_del(callback: CallbackQuery) -> None:
    """set:model:del:<task_type>"""
    task_type = callback.data.split(":")[3]
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        s = owner.settings
        try:
            overrides = json.loads(s.model_overrides) if s.model_overrides else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}
        overrides.pop(task_type, None)
        s.model_overrides = (
            json.dumps(overrides, ensure_ascii=False) if overrides else None
        )
        await session.flush()
    await callback.answer(f"🗑 Переопределение для {task_type} удалено")
    await _refresh_section(callback, "models_brain")


@router.callback_query(F.data.startswith("set:model:custom:"))
async def cb_model_custom(callback: CallbackQuery, state: FSMContext) -> None:
    """set:model:custom:<task_type> — ввод имени модели вручную."""
    task_type = callback.data.split(":")[3]
    await state.set_state(SettingsStates.waiting_custom_model_name)
    await state.update_data(custom_model_task_type=task_type)
    await callback.message.answer(
        "✏ Введи название модели (например: <code>deepseek-reasoner</code>, "
        "<code>gpt-4o-mini</code>). /cancel — отмена."
    )
    await callback.answer()


@router.message(SettingsStates.waiting_custom_model_name)
async def step_custom_model_name(message: Message, state: FSMContext) -> None:
    model_name = (message.text or "").strip()
    if not model_name:
        await message.answer("Пустое название. Повтори или /cancel.")
        return
    if len(model_name) > 128:
        await message.answer(
            "Слишком длинное название (макс. 128). Повтори или /cancel."
        )
        return

    # Hard validation: regex
    from src.bot.handlers.free_text_common import _MODEL_NAME_RE

    if not _MODEL_NAME_RE.match(model_name):
        await message.answer(
            "❌ Недопустимые символы в имени модели. "
            "Допустимы: буквы, цифры, <code>@ / _ . : -</code>\n"
            "Повтори или /cancel."
        )
        return

    data = await state.get_data()
    task_type = data.get("custom_model_task_type", "default")

    # Soft validation: catalog check across ALL user's providers (warn but still save)
    catalog_warning = ""
    try:
        from src.llm.provider_catalog import get_provider, LLM_PROVIDERS

        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slots = await list_key_slots(session, owner)
            user_providers = {s.provider for s in slots if s.enabled}
        # Ищем модель во всех каталогах, для которых есть ключи
        found_in = None
        for pi in LLM_PROVIDERS:
            if pi.name in user_providers and pi.models and model_name in pi.models:
                found_in = pi
                break
        if found_in is None:
            current_provider = owner.settings.llm_provider
            provider_info = get_provider(current_provider)
            if provider_info and provider_info.models:
                catalog_warning = (
                    f"\n\n⚠️ Модель <code>{model_name}</code> не найдена в каталогах "
                    f"твоих провайдеров.\n"
                    f"Доступные у <b>{current_provider}</b>: "
                    f"{', '.join(f'<code>{m}</code>' for m in provider_info.models[:8])}\n"
                    f"Сохраняю, но проверь имя на опечатки."
                )
    except Exception:
        logger.debug("catalog soft-validation skipped", exc_info=True)

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        s = owner.settings
        try:
            overrides = json.loads(s.model_overrides) if s.model_overrides else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}
        overrides[task_type] = model_name
        s.model_overrides = json.dumps(overrides, ensure_ascii=False)
        await session.flush()
    await state.clear()
    await message.answer(
        f"✅ Модель для <b>{task_type}</b>: <code>{model_name}</code>{catalog_warning}"
    )


@router.callback_query(F.data.startswith("set:model:"))
async def cb_model_open(callback: CallbackQuery) -> None:
    """set:model:<task_type> — открыть подменю выбора модели."""
    task_type = callback.data.split(":")[2]
    text, kb = await _render_section(callback.from_user.id, f"model_sel:{task_type}")
    await _safe_edit(callback.message, text, kb)
    await callback.answer()


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


async def _count_slots_for_provider(session, owner, provider: str) -> int:
    """Сколько ключей у пользователя для данного провайдера в LlmKeySlot."""
    slots = await list_key_slots(session, owner, provider=provider)
    return len(slots)


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
        total = await _count_slots_for_provider(session, owner, "openai")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:openai_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено OpenAI ключей: {count}.\n🔑 В базе OpenAI ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


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
        total = await _count_slots_for_provider(session, owner, "gemini")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:gemini_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено Gemini ключей: {count}.\n🔑 В базе Gemini ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


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
        total = await _count_slots_for_provider(session, owner, "mistral")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:mistral_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено Mistral ключей: {count}.\n🔑 В базе Mistral ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


@router.message(SettingsStates.waiting_cloudflare_key)
async def step_cloudflare_key(message: Message, state: FSMContext) -> None:
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
        logger.exception("failed to delete message with cloudflare key")
    # Валидируем первый ключ как индикатор; остальные считаем рабочими
    if not await CloudflareProvider(parts[0]).validate_key():
        await message.answer(
            "❌ Ключ не работает. Проверь API Token и CLOUDFLARE_ACCOUNT_ID в .env. /cancel."
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "cloudflare", ",".join(parts))
        total = await _count_slots_for_provider(session, owner, "cloudflare")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:cloudflare_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено Cloudflare ключей: {count}.\n🔑 В базе Cloudflare ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


@router.message(SettingsStates.waiting_deepseek_key)
async def step_deepseek_key(message: Message, state: FSMContext) -> None:
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
        logger.exception("failed to delete message with deepseek key")
    # Валидируем первый ключ как индикатор; остальные считаем рабочими
    if not await DeepSeekProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "deepseek", ",".join(parts))
        total = await _count_slots_for_provider(session, owner, "deepseek")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:deepseek_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено DeepSeek ключей: {count}.\n🔑 В базе DeepSeek ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


@router.message(SettingsStates.waiting_grok_key)
async def step_grok_key(message: Message, state: FSMContext) -> None:
    """Сохраняет Grok API ключ."""
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
        logger.exception("failed to delete message with grok key")
    if not await GrokProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "grok", ",".join(parts))
        total = await _count_slots_for_provider(session, owner, "grok")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:grok_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено Grok ключей: {count}.\n🔑 В базе Grok ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


@router.message(SettingsStates.waiting_mimo_key)
async def step_mimo_key(message: Message, state: FSMContext) -> None:
    """Сохраняет MiMo API ключ, затем спрашивает регион."""
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
        logger.exception("failed to delete message with mimo key")
    if not await MiMoProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    # Сохраняем ключ в state для последующего использования с регионом
    await state.update_data(mimo_key=",".join(parts))
    await state.set_state(SettingsStates.waiting_mimo_region)
    kb = InlineKeyboardBuilder()
    for region_key, region_url in MIMO_REGIONS.items():
        label = {"eu": "🇪🇺 EU", "us": "🇺🇸 US", "asia": "🌏 Asia"}.get(
            region_key, region_key.upper()
        )
        kb.button(text=label, callback_data=f"set:mimo_region:{region_key}")
    kb.button(text="⏭ Пропустить (Asia)", callback_data="set:mimo_region:skip")
    kb.adjust(2)
    await message.answer(
        "🌍 <b>Выбери регион MiMo API:</b>\n\n"
        "MiMo имеет региональные endpoint'ы. Выбери ближайший к тебе регион "
        "для минимальной задержки.\n\n"
        "• 🇪🇺 EU — Европа\n"
        "• 🇺🇸 US — США\n"
        "• 🌏 Asia — Азия (по умолчанию)\n\n"
        "/cancel — отмена.",
        reply_markup=kb.as_markup(),
    )


@router.callback_query(F.data.startswith("set:mimo_region:"))
async def cb_mimo_region(callback: CallbackQuery, state: FSMContext) -> None:
    """Обрабатывает выбор региона MiMo — сохраняет ключ с endpoint."""
    region_raw = callback.data.split(":", 2)[2]
    await callback.answer()

    if region_raw == "skip":
        endpoint = MIMO_REGIONS["asia"]
        region_label = "Asia (по умолчанию)"
    else:
        endpoint = MIMO_REGIONS.get(region_raw, MIMO_REGIONS["asia"])
        region_label = {"eu": "EU", "us": "US", "asia": "Asia"}.get(
            region_raw, region_raw
        )

    data = await state.get_data()
    mimo_key = data.get("mimo_key", "")
    if not mimo_key:
        await callback.message.answer(
            "❌ Ключ не найден. Начни заново: /settings → API-ключи → MiMo key."
        )
        await state.clear()
        return

    parts = [k.strip() for k in mimo_key.split(",") if k.strip()]
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        # Сохраняем в ApiKey (старое хранилище)
        await upsert_api_key(session, owner, "mimo", mimo_key)
        # Сохраняем в LlmKeySlot с endpoint (новое хранилище)
        for i, single_key in enumerate(parts):
            slot, _is_new = await add_key_slot(
                session,
                owner,
                "mimo",
                single_key,
                purpose="main",
                priority=i,
                endpoint=endpoint,
            )
            # upsert_api_key мог создать слот без endpoint — обновляем
            if not slot.endpoint:
                slot.endpoint = endpoint
        await session.flush()
        total = await _count_slots_for_provider(session, owner, "mimo")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:mimo_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    if callback.message:
        await callback.message.edit_text(
            f"✅ Сохранено MiMo ключей: {count} (регион: {region_label}).\n"
            f"🔑 В базе MiMo ключей: {total}.\n\n"
            "Добавить ещё?",
            reply_markup=kb.as_markup(),
        )


@router.message(SettingsStates.waiting_groq_key)
async def step_groq_key(message: Message, state: FSMContext) -> None:
    """Сохраняет Groq API ключ."""
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
        logger.exception("failed to delete message with groq key")
    if not await GroqProvider(parts[0]).validate_key():
        await message.answer("❌ Ключ не работает. Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await upsert_api_key(session, owner, "groq", ",".join(parts))
        total = await _count_slots_for_provider(session, owner, "groq")
    await state.clear()
    count = len(parts)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё ключ", callback_data="set:input:groq_key")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Сохранено Groq ключей: {count}.\n🔑 В базе Groq ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


# ── Custom provider FSM (4 шага) ──


@router.message(SettingsStates.waiting_custom_name)
async def step_custom_name(message: Message, state: FSMContext) -> None:
    """Шаг 1/4: название провайдера."""
    name = (message.text or "").strip()
    if name == "/cancel":
        await state.clear()
        text, kb = await _render_menu(message.from_user.id)
        await message.answer("🚫 Отменено.")
        await message.answer(text, reply_markup=kb)
        return
    if not name:
        await message.answer("Введи название. /cancel — отмена.")
        return
    await state.update_data(custom_name=name)
    await state.set_state(SettingsStates.waiting_custom_endpoint)
    await message.answer(
        f"✅ Название: <b>{sanitize_html(name)}</b>\n\n"
        "Шаг 2/4: Пришли endpoint (базовый URL API).\n"
        "Например: <code>https://api.openai.com/v1</code>\n"
        "/cancel — отмена."
    )


@router.message(SettingsStates.waiting_custom_endpoint)
async def step_custom_endpoint(message: Message, state: FSMContext) -> None:
    """Шаг 2/4: endpoint."""
    endpoint = (message.text or "").strip()
    if not endpoint:
        await message.answer("Введи URL. /cancel — отмена.")
        return
    if not endpoint.startswith("https://") and not endpoint.startswith("http://"):
        await message.answer("❌ URL должен начинаться с https:// или http://")
        return
    await state.update_data(custom_endpoint=endpoint)
    await state.set_state(SettingsStates.waiting_custom_key)
    await message.answer(
        f"✅ Endpoint: <code>{sanitize_html(endpoint)}</code>\n\n"
        "Шаг 3/4: Пришли API-ключ.\n"
        "💡 Можно несколько ключей через запятую.\n"
        "/cancel — отмена."
    )


@router.message(SettingsStates.waiting_custom_key)
async def step_custom_key(message: Message, state: FSMContext) -> None:
    """Шаг 3/4: API-ключ + валидация."""
    raw = (message.text or "").strip()
    if raw == "/cancel":
        await state.clear()
        text, kb = await _render_menu(message.from_user.id)
        await message.answer("🚫 Отменено.")
        await message.answer(text, reply_markup=kb)
        return
    if not raw:
        await message.answer("Пустой ключ. Повтори или /cancel.")
        return
    parts = [k.strip() for k in raw.split(",") if k.strip()]
    if not parts:
        await message.answer("Нет ни одного непустого ключа. Повтори или /cancel.")
        return
    data = await state.get_data()
    endpoint = data.get("custom_endpoint", "")
    try:
        await message.delete()
    except Exception:
        logger.exception("failed to delete message with custom key")
    try:
        valid = await CustomProvider(parts[0], endpoint=endpoint).validate_key()
    except Exception:
        valid = False
    if not valid:
        await message.answer(
            "❌ Ключ не работает или endpoint недоступен. Повтори или /cancel."
        )
        return
    await state.update_data(custom_key=",".join(parts))
    await state.set_state(SettingsStates.waiting_custom_models)
    await message.answer(
        "✅ Ключ работает!\n\n"
        "Шаг 4/4: Пришли модели через запятую.\n"
        "Например: <code>gpt-4, gpt-3.5-turbo, my-model</code>\n"
        "💡 Каждая модель будет доступна для всех задач.\n"
        "/cancel — отмена."
    )


@router.message(SettingsStates.waiting_custom_models)
async def step_custom_models(message: Message, state: FSMContext) -> None:
    """Шаг 4/4: модели — создаёт слоты в БД."""
    raw_models = (message.text or "").strip()
    if raw_models == "/cancel":
        await state.clear()
        text, kb = await _render_menu(message.from_user.id)
        await message.answer("🚫 Отменено.")
        await message.answer(text, reply_markup=kb)
        return
    if not raw_models:
        await message.answer("Введи хотя бы одну модель. /cancel — отмена.")
        return
    models = [m.strip() for m in raw_models.split(",") if m.strip()]
    data = await state.get_data()
    name = data.get("custom_name", "custom")
    endpoint = data.get("custom_endpoint", "")
    key = data.get("custom_key", "")
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        for model in models:
            await add_key_slot(
                session,
                owner,
                provider="custom",
                purpose="main",
                model=model,
                label=f"{name}:{model}",
                endpoint=endpoint,
                key=key,
            )
        total = await _count_slots_for_provider(session, owner, "custom")
    await state.clear()
    count = len(models)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Ещё провайдер", callback_data="set:input:custom_name")
    kb.button(text="✅ Назад", callback_data="set:done:key")
    kb.adjust(2)
    await message.answer(
        f"✅ Провайдер <b>{sanitize_html(name)}</b> добавлен!\n"
        f"Моделей: {count} · Всего custom ключей: {total}.\n\n"
        "Добавить ещё?",
        reply_markup=kb.as_markup(),
    )


@router.callback_query(F.data == "set:done:key")
async def cb_done_adding_key(callback: CallbackQuery, state: FSMContext) -> None:
    """Закрывает ввод ключей, возвращается в настройки."""
    await state.clear()
    text, kb = await _render_section(callback.from_user.id, "keys")
    await _safe_edit(callback.message, text, kb)
    await callback.answer()


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
    await message.answer(
        sanitize_html(f"✅ Текст автоответа сохранён:\n<i>«{text}»</i>")
    )


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


# ---------- Личность: callback'и ввода ----------


@router.callback_query(F.data == "set:input:alias")
async def cb_input_alias(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(SettingsStates.waiting_alias)
    await callback.message.answer(
        "👤 Как к тебе обращаться?\n\n"
        "Напиши имя или прозвище (например: <i>Миша, Александр Петрович, шеф</i>). "
        "Бот будет использовать это обращение в общении.\n"
        "/cancel — отмена."
    )
    await callback.answer()


@router.callback_query(F.data == "set:input:custom_instructions")
async def cb_input_custom_instructions(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await state.set_state(SettingsStates.waiting_custom_instructions)
    await callback.message.answer(
        "📝 <b>Пользовательские инструкции</b>\n\n"
        "Напиши свободный текст — как бот должен себя вести, что знать, "
        "какие темы избегать, и т.д.\n\n"
        "Например: <i>«Не используй англицизмы. Всегда проверяй факты. "
        "Перед ответом на сложный вопрос предупреждай что думаешь.»</i>\n\n"
        "/cancel — отмена."
    )
    await callback.answer()


# ---------- Личность: FSM-обработчики ----------


@router.message(SettingsStates.waiting_alias)
async def step_alias(message: Message, state: FSMContext) -> None:
    alias = (message.text or "").strip()
    if not alias:
        await message.answer("Пустое обращение. Повтори или /cancel.")
        return
    if len(alias) > 64:
        await message.answer("Слишком длинное (макс. 64 символа). Повтори или /cancel.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        p = await get_persona(session, owner)
        p.alias = alias
        await session.flush()
    from src.core.context_cache import invalidate as cache_invalidate

    await cache_invalidate(f"persona:{message.from_user.id}")
    await state.clear()
    await message.answer(sanitize_html(f"✅ Обращение сохранено: <b>{alias}</b>"))


@router.message(SettingsStates.waiting_custom_instructions)
async def step_custom_instructions(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустой текст. Повтори или /cancel.")
        return
    if len(text) > 2000:
        await message.answer(
            "Слишком длинный текст (макс. 2000 символов). Повтори или /cancel."
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        p = await get_persona(session, owner)
        p.custom_instructions = text
        await session.flush()
    from src.core.context_cache import invalidate as cache_invalidate

    await cache_invalidate(f"persona:{message.from_user.id}")
    await state.clear()
    await message.answer(
        sanitize_html(
            "✅ Инструкции сохранены!\n\n"
            f"<i>«{text[:300]}{'…' if len(text) > 300 else ''}»</i>"
        )
    )


# ---------- Личность: сброс к базовым ----------


@router.callback_query(F.data == "set:persona:reset")
async def cb_persona_reset(callback: CallbackQuery) -> None:
    ok = await reset_persona_to_snapshot(callback.from_user.id)
    if ok:
        await callback.answer("♻️ Настройки сброшены к базовым", show_alert=True)
    else:
        await callback.answer("Нет сохранённого снапшота для сброса", show_alert=True)
    await _refresh_section(callback, "personality")


# ---------- /cancel для состояний настроек ----------


@router.message(Command("cancel"), F.state.in_(SettingsStates))
async def cancel_settings_state(message: Message, state: FSMContext) -> None:
    await state.clear()
    text, kb = await _render_menu(message.from_user.id)
    await message.answer("🚫 Отменено.", reply_markup=kb)
    await message.answer(text, reply_markup=kb)
