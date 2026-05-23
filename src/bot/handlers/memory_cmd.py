import logging
from datetime import datetime, timezone

from src.llm.router import _ensure_utc

from aiogram import F, Router
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.filters import OwnerOnly
from src.core.contacts.contact_resolver import resolve
from src.core.infra.text_sanitizer import sanitize_html
from src.core.memory.memory_fuel import (
    format_depleted_contacts,
    format_fuel_line,
    get_fuel_stats,
)
from src.core.memory.memory_neighbors import format_neighbors, get_neighbors
from src.db.models import Commitment, LlmKeySlot, Memory, MemoryCandidate
from src.db.repo import (
    add_commitment,
    add_key_slot,
    add_memory,
    delete_memory,
    get_commitment_by_source_memory,
    get_memory_stats,
    get_or_create_user,
    get_persona,
    list_key_slots,
    list_memories,
    list_memory_candidates,
    search_memories,
    update_persona,
)
from src.db.session import get_session
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)
router = Router(name="memory_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())

# ─── /keys import: очередь ожидающих импорта (без FSM) ────────────────

_PENDING_IMPORTS: dict[int, str] = {}  # user_id → purpose

# ─── /keys import helpers ─────────────────────────────────────────────

_KEY_PREFIX_MAP: list[tuple[str, str]] = [
    ("sk-or", "openrouter"),  # OpenRouter: sk-or-v1-...
    ("sk-", "openai"),  # OpenAI: sk-proj-..., sk-...
    ("sk-ant", "openai"),  # Anthropic через OpenAI-совместимость
    ("AIza", "gemini"),  # Google Gemini
    ("cfat_", "cloudflare"),  # Cloudflare API token
    ("CF-", "cloudflare"),  # Cloudflare API key
    ("ms-", "mistral"),  # Mistral API (не гарантировано)
    ("mistral-", "mistral"),  # Mistral API key
]

_PROVIDER_ORDER = ("openrouter", "openai", "gemini", "mistral", "cloudflare")


def _guess_provider(key: str) -> str | None:
    """Угадывает провайдера по префиксу API-ключа."""
    for prefix, provider in _KEY_PREFIX_MAP:
        if key.startswith(prefix):
            return provider
    return None


async def _detect_provider(key: str) -> str | None:
    """Определяет провайдера: префикс → валидация → перебор."""
    # 1. Префиксная эвристика
    by_prefix = _guess_provider(key)
    if by_prefix:
        from src.llm.router import _provider_class_for

        prov_class = _provider_class_for(by_prefix)
        try:
            prov = prov_class(key)
            if await prov.validate_key():
                return by_prefix
        except Exception:
            pass

    # 2. Перебор всех провайдеров
    from src.llm.router import _provider_class_for

    for provider_name in _PROVIDER_ORDER:
        if provider_name == by_prefix:
            continue  # уже пробовали
        try:
            prov_class = _provider_class_for(provider_name)
            prov = prov_class(key)
            if await prov.validate_key():
                return provider_name
        except Exception:
            continue

    return None


async def _do_import_keys(
    message: Message, keys_text: str, purpose: str = "main"
) -> None:
    """Парсит и импортирует ключи с автоопределением провайдера."""
    from src.llm.router import _provider_class_for
    from src.crypto import decrypt

    lines = keys_text.strip().split("\n")
    raw_keys = []
    for line in lines:
        line = line.strip()
        # Пропускаем комментарии и пустые строки
        if not line or line.startswith("#"):
            continue
        raw_keys.append(line)

    if not raw_keys:
        await message.answer("❌ Не найдено ни одного ключа в сообщении.")
        return

    # Удаляем исходное сообщение (безопасность)
    try:
        await message.delete()
    except Exception:
        logger.exception("failed to delete message with keys")

    results: list[str] = []
    by_provider: dict[str, int] = {}
    total_found = 0

    for i, api_key in enumerate(raw_keys):
        detected = await _detect_provider(api_key)
        if not detected:
            results.append(f"  ❓ ключ #{i + 1} — провайдер не определён")
            continue

        by_provider[detected] = by_provider.get(detected, 0) + 1

        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slot, is_new = await add_key_slot(
                session,
                owner,
                detected,
                api_key,
                purpose=purpose,
                label=f"{detected}/{purpose}",
                priority=i,
            )

        if not is_new:
            results.append(
                f"  #{slot.id} {detected}/{purpose} — Этот ключ уже добавлен (слот #{slot.id})"
            )
            total_found += 1
            continue

        # Валидируем (только для новых слотов)
        try:
            key = decrypt(slot.key_enc)
            prov_class = _provider_class_for(detected)
            prov = prov_class(key)
            valid = await prov.validate_key()
            if not valid:
                async with get_session() as session:
                    owner = await get_or_create_user(session, message.from_user.id)
                    bad_slot = await session.get(LlmKeySlot, slot.id)
                    if bad_slot:
                        await session.delete(bad_slot)
                        await session.flush()
                results.append(
                    f"  #{slot.id} {detected}/{purpose} ❌ (не прошёл проверку)"
                )
            else:
                results.append(f"  #{slot.id} {detected}/{purpose} ✅")
                total_found += 1
        except Exception:
            results.append(f"  #{slot.id} {detected}/{purpose} ✅ (ошибка проверки)")
            total_found += 1

    # Итоговое сообщение
    header = f"📥 <b>Импорт ключей</b> (purpose: {purpose})\n\n"
    if by_provider:
        header += "Найдено: " + ", ".join(
            f"{p} ×{c}" for p, c in sorted(by_provider.items())
        )
        header += f"\nУспешно: {total_found}/{len(raw_keys)}\n\n"
    else:
        header += f"Успешно: 0/{len(raw_keys)}\n\n"

    await message.answer(header + "\n".join(results))


@router.message(Command("keys"))
async def cmd_keys(message: Message) -> None:
    """Управление ключами LLM."""
    args = (message.text or "").split()
    if len(args) >= 4 and args[1] == "add":
        provider = args[2].lower()
        purpose_raw = args[3].lower()
        api_keys_raw = " ".join(args[4:])

        if provider not in ("openrouter", "openai", "gemini", "mistral", "cloudflare"):
            await message.answer(
                "❌ Провайдер: openrouter, openai, gemini, mistral или cloudflare"
            )
            return

        # Auto-increment priority when purpose ends with "+"
        auto_inc = purpose_raw.endswith("+")
        purpose = purpose_raw.rstrip("+") if auto_inc else purpose_raw

        # Split by comma for bulk add
        keys = [k.strip() for k in api_keys_raw.split(",") if k.strip()]
        if not keys:
            await message.answer("❌ Не указан ключ(и).")
            return

        # Удаляем сообщение с ключами из чата
        try:
            await message.delete()
        except Exception:
            logger.exception("failed to delete message with key")

        from src.llm.router import _provider_class_for
        from src.crypto import decrypt

        success = 0
        failed = 0
        results = []

        for i, api_key in enumerate(keys):
            priority = i if auto_inc else 0
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
                slot, is_new = await add_key_slot(
                    session,
                    owner,
                    provider,
                    api_key,
                    purpose=purpose,
                    label=f"{provider}/{purpose}",
                    priority=priority,
                )
            if not is_new:
                results.append(
                    f"  #{slot.id} {provider}/{purpose} — Этот ключ уже добавлен (слот #{slot.id})"
                )
                success += 1
                continue
            # Валидируем ключ (только для новых слотов)
            try:
                key = decrypt(slot.key_enc)
                prov_class = _provider_class_for(provider)
                prov = prov_class(key)
                valid = await prov.validate_key()
                if not valid:
                    async with get_session() as session:
                        owner = await get_or_create_user(session, message.from_user.id)
                        bad_slot = await session.get(LlmKeySlot, slot.id)
                        if bad_slot:
                            await session.delete(bad_slot)
                            await session.flush()
                    results.append(f"  #{slot.id} {provider}/{purpose} ❌")
                    failed += 1
                else:
                    results.append(f"  #{slot.id} {provider}/{purpose} ✅")
                    success += 1
            except Exception:
                results.append(
                    f"  #{slot.id} {provider}/{purpose} ✅ (ошибка проверки)"
                )
                success += 1

        if len(keys) == 1 and success == 1:
            if not is_new:
                await message.answer(
                    f"ℹ️ Ключ {provider}/{purpose} уже был добавлен ранее (слот #{slot.id})."
                )
            else:
                await message.answer(
                    f"✅ Ключ {provider}/{purpose} добавлен и проверен! (слот #{slot.id})"
                )
        elif len(keys) == 1 and failed == 1:
            await message.answer(
                f"❌ Ключ {provider}/{purpose} не прошёл валидацию. Проверь ключ."
            )
        else:
            lines = [f"<b>Добавлено {len(keys)} ключей {provider}/{purpose}:</b>", ""]
            lines.extend(results)
            await message.answer("\n".join(lines))
        return

    if len(args) >= 3 and args[1] == "remove":
        try:
            slot_id = int(args[2])
        except ValueError:
            await message.answer(
                "❌ Неверный номер слота. Использование: <code>/keys remove &lt;номер&gt;</code>\n"
                "Пример: <code>/keys remove 5</code>"
            )
            return
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slot = await session.get(LlmKeySlot, slot_id)
            if slot and slot.user_id == owner.id:
                provider = slot.provider
                purpose = slot.purpose
                await session.delete(slot)
                await session.commit()
                await message.answer(
                    f"✅ Слот #{slot_id} ({provider}/{purpose}) удалён."
                )
            else:
                await message.answer("❌ Слот не найден или не твой.")
        return

    if len(args) >= 2 and args[1] == "remove":
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slots = await list_key_slots(session, owner)
            if not slots:
                await message.answer("❌ Нет ключевых слотов для удаления.")
                return
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=f"{'✅' if s.enabled else '🚫'} #{s.id} {s.provider}/{s.purpose}",
                            callback_data=f"keys:remove:{s.id}",
                        )
                    ]
                    for s in slots
                ]
            )
            await message.answer("🗑 Выбери слот для удаления:", reply_markup=kb)
        return

    if len(args) >= 3 and args[1] == "toggle":
        slot_id = int(args[2])
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slot = await session.get(LlmKeySlot, slot_id)
            if slot and slot.user_id == owner.id:
                slot.enabled = not slot.enabled
                await session.commit()
                status = "включён" if slot.enabled else "выключен"
                await message.answer(
                    f"✅ Слот #{slot_id} ({slot.provider}/{slot.purpose}) {status}."
                )
            else:
                await message.answer("❌ Слот не найден или не твой.")
        return

    if len(args) >= 2 and args[1] == "--stats":
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slots = await list_key_slots(session, owner)
            if not slots:
                await message.answer("Нет ключевых слотов.")
                return
            lines = ["<b>📊 Статистика ключей:</b>", ""]
            total_used = sum(s.usage_count for s in slots)
            total_fail = sum(s.failure_count for s in slots)
            fail_rate = (total_fail / max(total_used, 1)) * 100
            lines.append(f"Всего вызовов: {total_used}")
            lines.append(f"Всего фейлов: {total_fail} ({fail_rate:.1f}%)")
            lines.append(f"Активных: {sum(1 for s in slots if s.enabled)}")
            lines.append(
                f"В кулдауне: {sum(1 for s in slots if (c := _ensure_utc(s.cooldown_until)) and c > datetime.now(timezone.utc))}"
            )
            lines.append("")
            for s in sorted(
                slots,
                key=lambda s: s.failure_count / max(s.usage_count, 1),
                reverse=True,
            )[:5]:
                fail_pct = (s.failure_count / max(s.usage_count, 1)) * 100
                lines.append(
                    f"<b>{s.provider}/{s.purpose}</b>: {s.usage_count}× вызовов, {s.failure_count}× фейлов ({fail_pct:.1f}%)"
                )
            await message.answer("\n".join(lines))
            return

    if len(args) >= 2 and args[1] == "import":
        # /keys import [purpose] [keys] — автоимпорт ключей с определением провайдера
        purpose = "main"
        keys_text = ""
        if len(args) >= 3:
            if args[2].lower() in (
                "main",
                "draft",
                "memory",
                "background",
                "search",
                "analysis",
                "urgent",
                "fallback",
            ):
                purpose = args[2].lower()
                keys_text = " ".join(args[3:]) if len(args) >= 4 else ""
            else:
                keys_text = " ".join(args[2:])

        # Если ключи переданы в команде — парсим сразу
        if keys_text.strip():
            keys_text = keys_text.replace(",", "\n")
            await _do_import_keys(message, keys_text, purpose)
            return

        # Иначе — ждём следующего сообщения
        _PENDING_IMPORTS[message.from_user.id] = purpose
        await message.answer(
            "📥 <b>Отправь ключи:</b>\n\n"
            "Один ключ на строку. Можно несколько.\n"
            "Провайдер определится автоматически.\n\n"
            "<i>Пример:</i>\n"
            "<code>sk-proj-abcd1234...\n"
            "cfat_xyz9876...\n"
            "AIzaSyABC...</code>\n\n"
            "Для отмены — /cancel"
        )
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        slots = await list_key_slots(session, owner)
        if not slots:
            await message.answer(
                "🔑 <b>Нет ключевых слотов.</b>\n\n"
                "Добавь ключ через /keys add openai main sk-...\n"
                "Где:\n"
                "• провайдер: openrouter/openai/gemini/mistral/cloudflare\n"
                "• purpose: main/draft/memory/background/search/analysis/urgent/fallback\n"
                "• ключ: сам API ключ"
            )
            return
        lines = ["<b>🔑 Ключевые слоты:</b>", ""]
        for s in slots[:10]:
            status = "✅" if s.enabled else "🚫"
            cool = (
                " 🔒"
                if (c := _ensure_utc(s.cooldown_until))
                and c > datetime.now(timezone.utc)
                else ""
            )
            lines.append(
                f"{status} <b>{s.provider}</b> / {s.purpose} "
                f"(приоритет {s.priority}, исп. {s.usage_count}×{cool})"
            )
            if s.last_error:
                lines.append(f"   ⚠️ {s.last_error[:80]}")
            if s.label:
                lines.append(f"   🏷 {s.label}")
        lines.append("")
        lines.append(
            "<i>/keys add &lt;provider&gt; &lt;purpose&gt; &lt;key1,key2,...&gt; — добавить ключи</i>"
        )
        lines.append(
            "<i>/keys import [purpose] [keys...] — автоимпорт с определением провайдера</i>"
        )
        lines.append("<i>/keys remove &lt;slot_id&gt; — удалить слот</i>")
        lines.append("<i>/keys toggle &lt;slot_id&gt; — вкл/выкл слот</i>")
        await message.answer("\n".join(lines))


@router.callback_query(F.data.startswith("keys:remove:"))
async def cb_keys_remove(callback: CallbackQuery) -> None:
    """Удалить слот ключа по inline-кнопке."""
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    slot_id = int(parts[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        slot = await session.get(LlmKeySlot, slot_id)
        if slot and slot.user_id == owner.id:
            provider = slot.provider
            purpose = slot.purpose
            await session.delete(slot)
            await session.commit()
            text = f"✅ Слот #{slot_id} ({provider}/{purpose}) удалён."
        else:
            text = "❌ Слот не найден или не твой."
    if callback.message:
        await callback.message.edit_text(text)
    await callback.answer()


class _PendingImportFilter(BaseFilter):
    """Фильтр: сообщение от юзера, ожидающего импорта ключей.
    Пропускает только если у юзера нет активного FSM-состояния.
    """

    async def __call__(
        self, message: Message, state: FSMContext | None = None
    ) -> bool | dict:
        if message.from_user is None:
            return False
        if message.from_user.id not in _PENDING_IMPORTS:
            return False
        if state is not None and await state.get_state() is not None:
            return False
        return True


@router.message(_PendingImportFilter())
async def _pending_import_handler(message: Message) -> None:
    """Принимает ключи после /keys import (без FSM)."""
    uid = message.from_user.id  # type: ignore[union-attr]
    purpose = _PENDING_IMPORTS.pop(uid)
    keys_text = message.text or ""

    if not keys_text.strip() or keys_text.strip().lower() in ("/cancel", "отмена"):
        await message.answer("❌ Импорт отменён.")
        return

    await _do_import_keys(message, keys_text, purpose)


@router.message(Command("llm_status"))
async def cmd_llm_status(message: Message) -> None:
    """Показать статус LLM: семафоры, слоты, использование."""
    from src.llm.router import _PURPOSE_SEMAPHORES

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        slots = await list_key_slots(session, owner)

    lines = ["<b>📊 LLM Status</b>", ""]
    total_used = sum(s.usage_count or 0 for s in slots)
    total_fail = sum(s.failure_count or 0 for s in slots)
    lines.append(f"Всего вызовов: {total_used} | фейлов: {total_fail}")
    lines.append("")

    for purpose, sem in _PURPOSE_SEMAPHORES.items():
        active = sem._value
        limit = sem._bound_value if hasattr(sem, "_bound_value") else "?"
        lines.append(f"🔹 {purpose}: {active}/{limit} слотов свободно")
    lines.append("")
    for s in slots[:10]:
        cooldown_active = (c := _ensure_utc(s.cooldown_until)) and c > datetime.now(
            timezone.utc
        )
        status = "❌" if not s.enabled else "⏳" if cooldown_active else "✅"
        lines.append(
            f"{status} <b>{s.provider}</b> / {s.purpose} — {s.usage_count}× ({s.failure_count}× фейлов)"
        )

    await message.answer("\n".join(lines))


@router.message(Command("memory"))
async def cmd_memory(message: Message, userbot_manager: UserbotManager) -> None:
    """Показать память — всё или про конкретный контакт, или --inbox."""
    args = (message.text or "").replace("/memory", "").strip()

    inbox_mode = "--inbox" in args
    if inbox_mode:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            candidates = await list_memory_candidates(session, owner)
        if not candidates:
            await message.answer("📭 Входящих фактов на подтверждение нет.")
            return
        lines = ["📬 <b>Входящие факты (Memory Inbox):</b>", ""]
        for i, c in enumerate(candidates, 1):
            sent_emoji = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "⚪",
            }.get(c.sentiment or "", "⚪")
            mem_type = f" ({c.memory_type})" if c.memory_type else ""
            lines.append(
                f"{i}. {sent_emoji} <i>{sanitize_html(c.fact)}</i>{mem_type}\n"
                f"   важность={c.importance}, затухание={c.decay_rate}, источник={c.source}"
            )
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="✅ Запомнить",
                            callback_data=f"memb:confirm:{c.id}",
                        ),
                        InlineKeyboardButton(
                            text="✏️ Исправить",
                            callback_data=f"memb:edit:{c.id}",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="⏳ На неделю",
                            callback_data=f"memb:temporary:{c.id}",
                        ),
                        InlineKeyboardButton(
                            text="♾ Навсегда",
                            callback_data=f"memb:permanent:{c.id}",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="❌ Удалить",
                            callback_data=f"memb:discard:{c.id}",
                        ),
                    ],
                ]
            )
            await message.answer(lines[-1], reply_markup=kb)
        return

    tag_mode = "--tag" in args
    if tag_mode:
        parts = args.split("--tag", 1)
        tag = parts[1].strip().split()[0] if len(parts) > 1 and parts[1].strip() else ""
        from src.core.memory.memory_tagger import format_tagged, search_by_tag

        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            facts = await search_by_tag(session, owner, tag)
        text = format_tagged(facts, tag)
        await message.answer(text)
        return

    timeline_mode = "--timeline" in args
    if timeline_mode:
        args = args.replace("--timeline", "").strip()

    story_mode = "--story" in args
    if story_mode:
        args = args.replace("--story", "").strip()

    contact_id = None
    label = ""
    if args:
        contact_name = args.strip()
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            client = (
                userbot_manager.get_client(message.from_user.id)
                if userbot_manager
                else None
            )
            if client is not None:
                candidates = await resolve(client, owner, contact_name)
                if candidates:
                    contact_id = candidates[0].peer_id
                    label = f" — {candidates[0].label()}"

    if story_mode:
        if contact_id:
            from src.core.memory.memory_chain import build_chain_narrative

            narrative = await build_chain_narrative(contact_id, message.from_user.id)
            if narrative:
                await message.answer(sanitize_html(narrative))
            else:
                await message.answer(
                    "Недостаточно данных для истории (нужно минимум 3 факта)."
                )
        else:
            await message.answer("Укажи контакт: <code>/memory --story имя</code>")
        return

    # ── Timeline mode ──────────────────────────────────────────────────
    if timeline_mode:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            items = await list_memories(session, owner, contact_id=contact_id)

        if not items:
            await message.answer("Память пуста.")
            return

        text = _format_timeline(items, contact_id, message.from_user.id)
        await message.answer(text)
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        all_items = await list_memories(session, owner, contact_id=contact_id)
        stats = await get_memory_stats(session, owner)

        # Отделяем task-факты для показа с кнопками и статусом Commitment
        task_memories = [m for m in all_items if m.memory_type == "task"]
        task_commitments: dict[int, Commitment | None] = {}
        for m in task_memories:
            task_commitments[m.id] = await get_commitment_by_source_memory(
                session, owner.id, m.id
            )

    items = [m for m in all_items if m.memory_type != "task"]

    if not items and not task_memories:
        await message.answer("Память пуста.")
        return

    # Статистика
    pos = stats["by_sentiment"].get("positive", 0)
    neg = stats["by_sentiment"].get("negative", 0)
    neu = stats["by_sentiment"].get("neutral", 0)
    stat_line = f"🧠 <b>Память{label}</b>: {stats['total']} фактов ({pos} позитивных, {neg} негативных, {neu} нейтральных)\n"

    # Индикатор здоровья памяти
    from src.core.memory.memory_health import (
        calculate_health_score,
        format_health_compact,
    )

    health = await calculate_health_score(message.from_user.id)
    health_line = format_health_compact(health)

    # Индикатор топлива памяти
    fuel = await get_fuel_stats(message.from_user.id)
    fuel_line = format_fuel_line(fuel)
    fuel_depleted = format_depleted_contacts(fuel)

    # Отправляем task-факты отдельными сообщениями с кнопками
    for m in task_memories:
        date_str = m.created_at.strftime("%d.%m.%Y") if m.created_at else "?"
        sent = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            m.sentiment or "", "⚪"
        )
        c = task_commitments.get(m.id)
        if c:
            status_emoji = {
                "done": "✅",
                "cancelled": "❌",
                "open": "📋",
                "reminded": "⏰",
            }.get(c.status, "📋")
            line = (
                f"• {sent} [{date_str}] {sanitize_html(m.fact)}\n"
                f"   {status_emoji} Задача: <b>{c.status}</b>"
            )
            await message.answer(line)
        else:
            line = f"• {sent} [{date_str}] {sanitize_html(m.fact)}"
            # Кнопка создания задачи из факта памяти
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="📋 Создать задачу",
                            callback_data=f"mem:totask:{m.id}",
                        ),
                    ]
                ]
            )
            await message.answer(line, reply_markup=kb)

    # Если есть только task-факты — завершаем
    if not items:
        return

    # Группировка по sentiment для остальных фактов
    positive_lines: list[str] = []
    negative_lines: list[str] = []
    neutral_lines: list[str] = []

    for m in items:
        date_str = m.created_at.strftime("%d.%m.%Y") if m.created_at else "?"
        sent = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            m.sentiment or "", "⚪"
        )
        rel_icon = {
            "cause": "🎯",
            "effect": "⚡",
            "contradicts": "⚠️",
            "supports": "✅",
            "continues": "➡️",
            "example_of": "📌",
        }.get(m.relation_type or "", "")
        rel_prefix = f"{rel_icon} " if rel_icon else ""
        # Distillation факты — с маркером 💡 и жирным шрифтом
        if m.source == "distillation":
            display_fact = m.fact
            if display_fact.startswith("💡 "):
                display_fact = display_fact[2:]
            line = f"• 💡 <b>{sanitize_html(display_fact)}</b>"
        else:
            line = f"• {sent} [{date_str}]{rel_prefix} {sanitize_html(m.fact)}"
        if m.sentiment == "positive":
            positive_lines.append(line)
        elif m.sentiment == "negative":
            negative_lines.append(line)
        else:
            neutral_lines.append(line)

    body_parts = [stat_line, health_line, fuel_line]
    if fuel_depleted:
        body_parts.append(fuel_depleted)
    if positive_lines:
        body_parts.append(f"\n<b>🟢 Позитивные ({len(positive_lines)}):</b>")
        body_parts.extend(positive_lines[:10])
    if negative_lines:
        body_parts.append(f"\n<b>🔴 Негативные ({len(negative_lines)}):</b>")
        body_parts.extend(negative_lines[:10])
    if neutral_lines:
        body_parts.append(f"\n<b>⚪ Нейтральные ({len(neutral_lines)}):</b>")
        body_parts.extend(neutral_lines[:10])

    body = "\n".join(body_parts)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🗑 Очистить негативные",
                    callback_data="memory:clear_negative",
                ),
                InlineKeyboardButton(
                    text="📊 Статистика", callback_data="memory:stats"
                ),
            ]
        ]
    )
    await message.answer(body, reply_markup=kb)


@router.callback_query(F.data == "memory:clear_negative")
async def cb_memory_clear_negative(callback: CallbackQuery) -> None:
    """Удалить все негативные факты."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        items = await list_memories(session, owner)
        removed = 0
        for m in items:
            if m.sentiment == "negative":
                await delete_memory(session, owner, m.id)
                removed += 1
    if callback.message:
        await callback.message.edit_text(f"🧹 Удалено {removed} негативных фактов.")
    await callback.answer(f"Удалено {removed}")


@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    """Показать здоровье памяти — единый скоринг 0-100."""
    from src.core.memory.memory_health import calculate_health_score, format_health

    health = await calculate_health_score(message.from_user.id)
    text = format_health(health)
    await message.answer(text)


@router.callback_query(F.data == "memory:stats")
async def cb_memory_stats(callback: CallbackQuery) -> None:
    """Показать детальную статистику памяти."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        stats = await get_memory_stats(session, owner)

    lines = [
        "📊 <b>Статистика памяти</b>",
        "",
        f"🧠 Всего фактов: {stats['total']}",
        "",
        "<b>По тональности:</b>",
    ]
    for sentiment, count in stats["by_sentiment"].items():
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            sentiment, "⚪"
        )
        lines.append(f"  {emoji} {sentiment}: {count}")
    lines.extend(
        [
            "",
            "<b>По источникам:</b>",
        ]
    )
    for source, count in stats["by_source"].items():
        lines.append(f"  📄 {source}: {count}")
    lines.extend(
        [
            "",
            f"🎯 Высокая уверенность (≥0.8): {stats['high_confidence']}",
            f"👤 Связано с контактами: {stats['with_contact']}",
        ]
    )

    # Индикатор топлива памяти
    fuel = await get_fuel_stats(callback.from_user.id)
    lines.append("")
    lines.append(format_fuel_line(fuel))
    depleted_text = format_depleted_contacts(fuel)
    if depleted_text:
        lines.append(depleted_text)

    if callback.message:
        await callback.message.answer("\n".join(lines))
    await callback.answer()


@router.message(Command("remember"))
async def cmd_remember(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    """Вручную сохранить факт. /remember Настя злится из-за дедлайна"""
    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "Использование: <code>/remember [контакт] факт</code>\nПример: <code>/remember Настя злится</code>"
        )
        return

    # пробуем отделить имя контакта от факта
    contact_name = None
    fact = args
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    client = (
        userbot_manager.get_client(message.from_user.id) if userbot_manager else None
    )
    if client is not None:
        candidates = await resolve(client, owner, args)
        if candidates and candidates[0].score >= 70:
            contact_name = candidates[0].label()
            # пытаемся отделить: берём первое слово как имя
            words = args.split(None, 1)
            if len(words) > 1:
                fact = words[1]

    contact_id = None
    if contact_name:
        candidates = await resolve(client, owner, contact_name)
        if candidates:
            contact_id = candidates[0].peer_id

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await add_memory(
            session, owner, fact=fact, contact_id=contact_id, source="user"
        )

    await message.answer(sanitize_html(f"🧠 Запомнил: <i>{fact}</i>"))


@router.message(Command("habits"))
async def cmd_habits(message: Message) -> None:
    """Показать обнаруженные привычки на основе повторяющихся фактов."""
    from src.core.scheduling.habit_tracker import find_habit_candidates, format_habits

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        memories = await list_memories(session, owner)
        active = [m for m in memories if m.is_active and m.created_at]
    habits = find_habit_candidates(active)
    text = format_habits(habits)
    await message.answer(text)


@router.message(Command("insights"))
async def cmd_insights(message: Message) -> None:
    from src.core.memory.memory_patterns import detect_patterns, format_insights

    insights = await detect_patterns(message.from_user.id)
    text, keyboards = format_insights(insights)
    # Если инсайтов нет — шлём один текст
    if not insights:
        await message.answer(text)
        return
    # Если есть — каждый инсайт отдельным сообщением с клавиатурой
    for ins, kb in zip(insights[:5], keyboards):
        detail = (
            f"<b>{sanitize_html(ins['title'])}</b>\n"
            f"{sanitize_html(ins['detail'])}\n"
            f"💡 {sanitize_html(ins['action'])}"
        )
        await message.answer(detail, reply_markup=kb)


@router.message(Command("forget"))
async def cmd_forget(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    """Удалить факты по подстроке. /forget злится"""
    args = (command.args or "").strip()
    if not args:
        await message.answer("Использование: <code>/forget часть текста</code>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        found = await search_memories(session, owner, args)

    if not found:
        await message.answer("Ничего не нашёл.")
        return

    async with get_session() as session:
        owner = await session.merge(owner)
        for m in found:
            await delete_memory(session, owner, m.id)

    names = ", ".join(
        f"«{m.fact[:50]}…»" if len(m.fact) > 50 else f"«{m.fact}»" for m in found
    )
    await message.answer(sanitize_html(f"🗑 Забыл: {names}"))


@router.message(Command("archetypes"))
async def cmd_archetypes(message: Message) -> None:
    """Показать архетипы всех контактов."""
    from src.core.contacts.contact_archetypes import (
        classify_all_contacts,
        format_archetype_stats,
    )

    await message.answer("🏷 Анализирую контакты...")
    stats = await classify_all_contacts(message.from_user.id)
    text = format_archetype_stats(stats)
    await message.answer(text)


@router.message(Command("distill"))
async def cmd_distill(message: Message, userbot_manager: UserbotManager) -> None:
    """Запустить дистилляцию фактов (10+ → 1 summary)."""
    from src.core.memory.knowledge_distiller import run_distillation

    args = (message.text or "").split()
    contact_name = args[1] if len(args) > 1 else None
    contact_id = None
    if contact_name:
        client = (
            userbot_manager.get_client(message.from_user.id)
            if userbot_manager
            else None
        )
        if client is not None:
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id

    await message.answer("🧠 Запускаю дистилляцию...")
    result = await run_distillation(message.from_user.id, contact_id)
    if result["success"]:
        await message.answer(
            f"✅ <b>Дистилляция завершена:</b>\n"
            f"Сжато {result['deactivated']} фактов →\n"
            f"<i>«{sanitize_html(result['fact'][:200])}»</i>"
        )
    else:
        await message.answer("❌ Недостаточно фактов для дистилляции (нужно 10+).")


@router.callback_query(F.data.startswith("pattern:"))
async def cb_pattern_action(callback: CallbackQuery) -> None:
    """Обрабатывает нажатия на inline-кнопки паттернов."""
    data = callback.data.split(":")
    action = data[1]  # remind, dismiss, history, write
    contact_id = int(data[2]) if len(data) > 2 else 0

    if action == "dismiss":
        if callback.message:
            await callback.message.edit_text(
                callback.message.text + "\n\n🔕 Ок, не сейчас."
            )
        await callback.answer()
        return

    if action == "remind":
        from src.db.repo import get_contact, get_or_create_user

        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            contact = await get_contact(session, owner, contact_id)
            name = contact.display_name if contact else str(contact_id)
            # Сохраняем факт в память
            await add_memory(
                session,
                owner,
                fact=f"Пользователь хочет напоминание о созвоне с {name}",
                source="user",
                sentiment="neutral",
            )
        if callback.message:
            await callback.message.edit_text(
                sanitize_html(
                    f"📅 Напоминание для <b>{name}</b>\n"
                    f"Напиши: <code>/remind за час до созвона с {name}</code>"
                )
            )
        await callback.answer(f"Напоминание для {sanitize_html(name)}")
        return

    if action == "history":
        await callback.answer(
            f"История контакта {contact_id} — открой /chat {contact_id} или /memory"
        )
        return

    if action == "write":
        await callback.answer("Напиши: /send контакт текст")
        return

    await callback.answer()


@router.message(Command("instructions"))
async def cmd_instructions(message: Message) -> None:
    from src.core.intelligence.adaptive_instructions import get_active_rules

    async with get_session() as session:
        await get_or_create_user(session, message.from_user.id)
    rules = await get_active_rules(message.from_user.id)
    if not rules:
        await message.answer(
            "У тебя нет активных правил. Скажи что-то вроде «отвечай короче» или «не используй смайлики»."
        )
        return
    lines = ["<b>📋 Активные правила:</b>", ""]
    for i, r in enumerate(rules, 1):
        lines.append(f"{i}. {r}")
    await message.answer("\n".join(lines))


@router.message(Command("tag"))
async def cmd_tag(message: Message) -> None:
    """Проставить теги всем нетэгированным фактам."""
    from src.core.memory.memory_tagger import tag_all_untagged

    await message.answer("🏷 Тегирую факты...")
    count = await tag_all_untagged(message.from_user.id)
    if count > 0:
        await message.answer(f"✅ Протегировано {count} фактов.")
    else:
        await message.answer("✅ Все факты уже протегированы, или нет активных фактов.")


@router.callback_query(F.data.startswith("mem:neighbors:"))
async def cb_mem_neighbors(callback: CallbackQuery) -> None:
    """Показать семантических соседей для факта памяти."""
    mid = int(callback.data.split(":")[2])
    neighbors = await get_neighbors(callback.from_user.id, mid)
    text = format_neighbors(neighbors)
    if text:
        await callback.message.answer(text)  # type: ignore[union-attr]
    else:
        await callback.answer("Соседей не найдено")


@router.message(Command("conflicts"))
async def cmd_conflicts(message: Message) -> None:
    """Показать и разрешить конфликты в памяти."""
    from src.core.actions.conflict_resolver import find_conflicts, format_conflicts

    conflicts = await find_conflicts(message.from_user.id)
    text = format_conflicts(conflicts)
    await message.answer(text)


@router.message(Command("warnings"))
async def cmd_warnings(message: Message) -> None:
    """Показать активные предупреждения о риске конфликтов."""
    from src.core.actions.conflict_predictor import (
        detect_silence_triggers,
        format_conflict_warnings,
    )

    triggers = await detect_silence_triggers(message.from_user.id)
    text = format_conflict_warnings(triggers) or "✅ Нет рисков конфликтов."
    await message.answer(text)


# ── Memory Inbox (memb:*) handlers ──────────────────────────────────


@router.callback_query(F.data.startswith("memb:"))
async def cb_memory_inbox(callback: CallbackQuery) -> None:
    """Обрабатывает кнопки Inbox для MemoryCandidate."""

    parts = callback.data.split(":")
    action = parts[1]
    candidate_id = int(parts[2])

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        candidate = await session.get(MemoryCandidate, candidate_id)

        if candidate is None or candidate.user_id != owner.id:
            await callback.answer("Факт не найден", show_alert=True)
            return

        if action == "confirm":
            # Перенести в Memory как есть
            await add_memory(
                session,
                owner,
                fact=candidate.fact,
                contact_id=candidate.contact_id,
                sentiment=candidate.sentiment or None,
                source=candidate.source,
                importance=candidate.importance,
                decay_rate=candidate.decay_rate,
            )
            await session.delete(candidate)
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"✅ Запомнил: <i>{sanitize_html(candidate.fact)}</i>"
            )
            await callback.answer("Факт сохранён")

        elif action == "discard":
            await session.delete(candidate)
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"🗑 Удалил: <i>{sanitize_html(candidate.fact)}</i>"
            )
            await callback.answer("Факт удалён")

        elif action == "temporary":
            # Перенести с memory_type="temporary", decay_rate=0.3 (быстро протухнет)
            await add_memory(
                session,
                owner,
                fact=candidate.fact,
                contact_id=candidate.contact_id,
                sentiment=candidate.sentiment or None,
                source=candidate.source,
                memory_type="temporary",
                importance=candidate.importance,
                decay_rate=0.3,
            )
            await session.delete(candidate)
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"⏳ Сохранено на неделю: <i>{sanitize_html(candidate.fact)}</i>"
            )
            await callback.answer("Факт сохранён временно")

        elif action == "permanent":
            # Перенести с decay_rate=0.01 (почти не протухнет)
            await add_memory(
                session,
                owner,
                fact=candidate.fact,
                contact_id=candidate.contact_id,
                sentiment=candidate.sentiment or None,
                source=candidate.source,
                importance=min(1.0, candidate.importance + 0.2),
                decay_rate=0.01,
            )
            await session.delete(candidate)
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"♾ Сохранено навсегда: <i>{sanitize_html(candidate.fact)}</i>"
            )
            await callback.answer("Факт сохранён навсегда")

        elif action == "edit":
            await session.delete(candidate)
            await callback.message.edit_text(  # type: ignore[union-attr]
                f"✏️ Напиши исправленный текст для факта:\n\n"
                f"<i>{sanitize_html(candidate.fact)}</i>\n\n"
                f"<code>/remember исправленный текст</code>"
            )
            await callback.answer("Напиши /remember с исправленным текстом")

        else:
            await callback.answer("Неизвестное действие")


@router.callback_query(F.data.startswith("mem:totask:"))
async def cb_mem_to_task(callback: CallbackQuery) -> None:
    """Создать задачу (Commitment) из факта памяти."""
    memory_id = int(callback.data.split(":")[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        mem = await session.get(Memory, memory_id)
        if mem is None or mem.user_id != owner.id:
            await callback.answer("Факт не найден", show_alert=True)
            return

        # Проверяем, нет ли уже задачи для этого факта
        existing = await get_commitment_by_source_memory(session, owner.id, mem.id)
        if existing:
            await callback.answer("Задача уже существует", show_alert=True)
            return

        # Создаём обязательство со ссылкой на факт памяти
        await add_commitment(
            session,
            user_id=owner.id,
            peer_id=mem.contact_id or 0,
            peer_name=None,
            message_id=None,
            direction="mine",
            text=mem.fact,
            deadline_at=None,
            source_memory_id=mem.id,
        )

    if callback.message:
        await callback.message.edit_text(
            sanitize_html(f"📋 Задача создана:\n<i>{mem.fact}</i>")
        )
    await callback.answer("✅ Задача создана")


@router.callback_query(F.data.startswith("conflict:resolve:"))
async def cb_conflict_resolve(callback: CallbackQuery) -> None:
    """Обработать разрешение конфликта памяти."""
    parts = callback.data.split(":")
    positive_id = int(parts[2])
    negative_id = int(parts[3])
    resolution = parts[4]
    from src.core.actions.conflict_resolver import resolve_conflict

    success = await resolve_conflict(
        callback.from_user.id, positive_id, negative_id, resolution
    )
    if success:
        await callback.message.edit_text(  # type: ignore[union-attr]
            callback.message.text + "\n\n✅ Конфликт разрешён."
        )
    else:
        await callback.answer("Ошибка при разрешении конфликта")


@router.message(Command("clusters"))
async def cmd_clusters(message: Message) -> None:
    """Показать кластеры памяти."""
    from src.db.repo import list_clusters_for_contact

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        clusters = await list_clusters_for_contact(session, owner)

    if not clusters:
        await message.answer(
            "Нет кластеров памяти. Они создадутся автоматически ночью."
        )
        return

    lines = ["<b>🧩 Кластеры памяти:</b>", ""]
    for cluster, fact_count in clusters[:8]:
        lines.append(f"📦 <b>{cluster.topic}</b> — {fact_count} фактов")
        if cluster.summary:
            lines.append(f"   <i>{cluster.summary[:80]}</i>")
    await message.answer("\n".join(lines))


@router.message(Command("persona"))
async def cmd_persona(message: Message) -> None:
    """Показать/сбросить адаптивный профиль личности."""

    args = (message.text or "").split()

    # /persona reset — сброс
    if len(args) > 1 and args[1] == "reset":
        async with get_session() as session:
            owner_db = await get_or_create_user(session, message.from_user.id)
            p = await get_persona(session, owner_db)
            await update_persona(
                session,
                p,
                brevity="normal",
                formality="friendly",
                emoji_usage="normal",
                initiative="reactive",
                preferred_format="text",
                work_mode="normal",
                forbidden_patterns=None,
                max_response_len=500,
            )
        await message.answer("✅ Персона сброшена к стандартным настройкам.")
        return

    # /persona — показать
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        p = await get_persona(session, owner)

    brevity_labels = {
        "short": "📝 Коротко",
        "normal": "📝 Обычно",
        "detailed": "📝 Подробно",
    }
    formality_labels = {
        "formal": "👔 Формально",
        "friendly": "🤝 Дружелюбно",
        "casual": "😎 Панибратски",
    }
    emoji_labels = {
        "none": "🚫 Без эмодзи",
        "minimal": "😊 Минимум",
        "normal": "😊 Обычно",
        "rich": "🎉 Много",
    }
    initiative_labels = {
        "reactive": "🔇 По запросу",
        "proactive": "📢 Инициативный",
        "balanced": "⚖️ Умеренно",
    }
    work_labels = {
        "normal": "💼 Обычный",
        "focus": "🎯 Фокус",
        "relax": "🏖 Отдых",
    }
    format_labels = {
        "text": "📄 Текст",
        "bullets": "📋 Список",
        "numbered": "🔢 Нумерация",
    }

    lines = ["<b>🧑‍🎤 Твой стиль общения:</b>", ""]
    lines.append(brevity_labels.get(p.brevity, p.brevity))
    lines.append(formality_labels.get(p.formality, p.formality))
    lines.append(emoji_labels.get(p.emoji_usage, p.emoji_usage))
    lines.append(initiative_labels.get(p.initiative, p.initiative))
    lines.append(format_labels.get(p.preferred_format, p.preferred_format))
    lines.append(work_labels.get(p.work_mode, p.work_mode))
    lines.append(f"📏 Макс. длина: {p.max_response_len} символов")
    lines.append(f"🔄 Коррекций: {p.total_corrections}")
    lines.append("")
    lines.append(
        "<i>Скажи «отвечай короче», «будь формальнее», «без смайликов» — я подстроюсь.\n"
        "/persona reset — сбросить всё.</i>"
    )
    await message.answer("\n".join(lines))


# ── Timeline format ───────────────────────────────────────────────────


def _format_timeline(
    items: list,
    contact_id: int | None,
    owner_id: int,
) -> str:
    """Форматирует факты как хронологию по неделям."""
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict

    _now = datetime.now(timezone.utc)

    # Разделяем на долгосрочные (tier 3 или distillation) и обычные
    longterm = [m for m in items if m.memory_tier == 3 or m.source == "distillation"]
    regular = [m for m in items if m not in longterm and m.is_active]

    # Группируем regular факты по ISO неделям
    weekly: dict[str, list] = defaultdict(list)
    for m in regular:
        if not m.created_at:
            continue
        # Начало недели (понедельник)
        iso = m.created_at.isocalendar()
        week_start = datetime.strptime(
            f"{iso[0]}-W{iso[1]:02d}-1", "%G-W%V-%u"
        ).replace(tzinfo=timezone.utc)
        week_end = week_start + timedelta(days=6)
        label = f"{week_start.strftime('%-d')}-{week_end.strftime('%-d %b')}"
        weekly[label].append(m)

    # Сортируем недели по дате (от новых к старым)
    sorted_weeks = sorted(weekly.items(), key=lambda x: x[0], reverse=True)

    lines: list[str] = []
    if contact_id:
        # contact name is already resolved, but we don't have it here directly
        lines.append("📅 <b>История отношений:</b>\n")
    else:
        lines.append("📅 <b>Хронология памяти:</b>\n")

    for week_label, fact_list in sorted_weeks:
        # Считаем тренд недели
        pos = sum(1 for m in fact_list if m.sentiment == "positive")
        neg = sum(1 for m in fact_list if m.sentiment == "negative")
        if pos > neg:
            trend = "улучшение ⬆️"
        elif neg > pos:
            trend = "напряжение ⬇️"
        else:
            trend = "стабильно ➖"

        lines.append(f"🗓 <b>Неделя {week_label}:</b>")
        for m in fact_list[:10]:
            sent_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                m.sentiment or "", "⚪"
            )
            lines.append(f"  • {sent_emoji} «{sanitize_html(m.fact)}»")
        lines.append(f"  📊 Тренд: {trend}")
        lines.append("")

    # Долгосрочные факты
    if longterm:
        lines.append("🏛️ <b>Долгосрочные факты:</b>")
        for m in longterm:
            fact_text = m.fact
            if fact_text.startswith("💡 "):
                fact_text = fact_text[2:]
            lines.append(f"  • 💡 {sanitize_html(fact_text)}")
        lines.append("")

    return "\n".join(lines) if lines else "Нет данных для хронологии."
