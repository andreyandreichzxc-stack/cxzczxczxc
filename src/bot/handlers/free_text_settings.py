"""Обработчики настроек: set_setting, news topics, auto_mode, quiet_hours."""

import logging

from aiogram import Router

from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.timeutil import HM_RE, tz_short
from src.db.repo import (
    add_news_topic,
    delete_news_topic,
    get_or_create_user,
    list_news_topics,
)
from src.db.session import get_session

from .free_text_common import _coerce_setting_value, invalidate_settings_cache

logger = logging.getLogger(__name__)
router = Router(name="free_text_settings")


# Поля UserSettings, которые агент может менять через set_setting (имя → тип значения)
# Решения auto-reply принимаются через src.core.contacts.auto_reply_decision.decide().
# При добавлении новых настроек, связанных с auto-reply (стиль, спам-порог и т.п.),
# обновляй SETTING_FIELDS здесь и соответствующую логику в auto_reply_decision.py.
SETTING_FIELDS: dict[str, str] = {
    "auto_reply_enabled": "bool",
    "auto_reply_mode": "choice:static,smart",
    "auto_reply_text": "str",
    "auto_reply_cooldown_min": "int",
    "digest_enabled": "bool",
    "digest_time": "hm",
    "news_enabled": "bool",
    "news_digest_time": "hm",
    "news_window_hours": "int",
    "reminders_enabled": "bool",
    "reminder_lead_hours": "int",
    "reminder_overdue_enabled": "bool",
    "ignore_archived": "bool",
    "use_heavy_model": "bool",
    "llm_provider": "choice:openrouter,openai,gemini,mistral,cloudflare",
    "transcription_mode": "choice:local,api,hybrid",
    "transcription_api_provider": "choice:openai,gemini,mistral",
    "auto_sync_enabled": "bool",
    "auto_sync_interval_sec": "int",
    "auto_extract_memories": "bool",
    "include_saved_messages": "bool",
    "smart_digest_enabled": "bool",
    "smart_digest_interval_min": "int",
    "urgent_notify_enabled": "bool",
    "monitor_only_selected_folders": "bool",
    "pattern_caching_enabled": "bool",
    "monitored_folders": "str",
    "timezone": "tz",
}


async def _exec_set_setting(intent, message) -> None:
    key = (intent.get("key") or "").strip()
    value = intent.get("value")
    spec = SETTING_FIELDS.get(key)
    if spec is None:
        await message.answer(sanitize_html(f"Не умею менять «{key}»."))
        return
    validated, err = _coerce_setting_value(spec, value)
    if err:
        await message.answer(
            sanitize_html(f"Не понял значение для <b>{key}</b>: {err}.")
        )
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        setattr(owner.settings, key, validated)
        await session.commit()
        await invalidate_settings_cache(message.from_user.id)
        new_tz = owner.settings.timezone
    if key == "timezone":
        await message.answer(f"✅ Часовой пояс: <b>{tz_short(new_tz)}</b>")
    elif isinstance(validated, bool):
        await message.answer(f"✅ <b>{key}</b>: {'ВКЛ' if validated else 'ВЫКЛ'}")
    else:
        shown = str(validated)
        if len(shown) > 100:
            shown = shown[:97] + "…"
        await message.answer(sanitize_html(f"✅ <b>{key}</b> = <code>{shown}</code>"))


async def _exec_add_news_topic(intent, message) -> None:
    topic = (intent.get("topic") or "").strip()
    if not topic:
        await message.answer("Не понял какую тему добавить.")
        return
    try:
        hours = int(intent.get("hours") or 24)
    except (TypeError, ValueError):
        hours = 24
    hours = max(1, min(168, hours))
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await add_news_topic(session, owner, topic, hours=hours)
    await message.answer(
        sanitize_html(f"✅ Добавил тему: <b>{topic}</b> (окно {hours}ч)")
    )


async def _exec_remove_news_topic(intent, message) -> None:
    needle = (intent.get("topic") or "").strip().lower()
    if not needle:
        await message.answer("Какую тему удалить?")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        topics = await list_news_topics(session, owner)
        matched = [t for t in topics if needle in t.topic.lower()]
        if not matched:
            await message.answer(sanitize_html(f"Тем по «{needle}» не нашёл."))
            return
        for t in matched:
            await delete_news_topic(session, owner, t.id)
    names = ", ".join(f"«{t.topic}»" for t in matched)
    await message.answer(sanitize_html(f"🗑 Удалил: {names}"))


async def _exec_change_auto_mode(intent, message) -> None:
    mode = (intent.get("mode") or "").strip()
    if mode not in ("offline_only", "always", "smart"):
        await message.answer("❌ Укажи режим: offline_only, always или smart")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.auto_mode = mode
        await session.commit()
        await invalidate_settings_cache()
    labels = {"offline_only": "только оффлайн", "always": "всегда", "smart": "умный"}
    await message.answer(f"✅ Режим авто-ответа: <b>{labels[mode]}</b>")


async def _exec_set_quiet_hours(intent, message) -> None:
    start = (intent.get("start") or "").strip()
    end = (intent.get("end") or "").strip()
    if not HM_RE.match(start) or not HM_RE.match(end):
        await message.answer("❌ Укажи время в формате HH:MM (например 23:00 и 07:00)")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.quiet_hours_start = start
        owner.settings.quiet_hours_end = end
        await session.commit()
    await message.answer(f"✅ Тихие часы: <b>{start} – {end}</b>")
