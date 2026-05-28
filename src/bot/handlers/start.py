"""Handler for /start and the onboarding wizard for first-time users."""

import logging

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from sqlalchemy import func, select

from src.bot.filters import OwnerOnly, is_onboarded
from src.bot.states import OnboardingStates
from src.db.models._contacts import Contact
from src.db.models._learning import AdaptivePersona
from src.db.models._memory import Memory
from src.db.repo import get_or_create_user, upsert_api_key
from src.db.session import get_session
from src.core.infra.timeutil import TZ_PRESETS, is_valid_tz, tz_short
from src.llm.gemini_provider import GeminiProvider
from src.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

router = Router(name="start")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


# ─── helpers ──────────────────────────────────────────────────────────


def _pretty_provider(name: str | None) -> str:
    """Человеческое имя провайдера для отображения."""
    names = {
        "openrouter": "OpenRouter (DeepSeek V4)",
        "openai": "OpenAI",
        "gemini": "Gemini",
        "mistral": "Mistral",
        "cloudflare": "Cloudflare",
    }
    return names.get(name or "", "—")


WELCOME = (
    "👋 <b>Привет! Я твой AI-ассистент для Telegram</b>\n\n"
    "<b>Аккаунт</b>\n"
    "🔑 /login — подключить Telegram-аккаунт (api_id, api_hash, телефон, код, 2FA)\n"
    "🚪 /logout — удалить сохранённую сессию\n"
    "🔄 /sync — обновить список контактов из диалогов\n\n"
    "<b>Настройки</b>\n"
    "⚙️ /settings — авто-ответ, выбор LLM, API-ключи\n\n"
    "<b>Работа с чатами</b>\n"
    "💬 /chat &lt;имя&gt; — саммари, задачи, черновик ответа, «где остановились»\n"
    "⏪ /catchup &lt;имя&gt; — где мы остановились + черновик ответа\n"
    "🔍 /search &lt;текст&gt; — поиск по проиндексированным сообщениям\n"
    "📇 /index &lt;имя&gt; — проиндексировать чат для семантического поиска\n"
    "📤 /send &lt;инструкция&gt; — «скажи Оле, что созвон в 8» (с подтверждением)\n\n"
    "<b>Новости</b>\n"
    "📰 /news &lt;тема&gt; [--hours=24] — дайджест из подписанных каналов\n"
    "📡 /news_channels — отметить каналы-источники\n"
    "🏷 /news_topics — темы для утренних авто-новостей\n\n"
    "<b>Память и фичи</b>\n"
    "📋 /todos — открытые обещания (мои и мне)\n"
    "☀️ /digest [now|on|off|at HH:MM] — утренний дайджест\n"
    "🎭 /style &lt;имя&gt; — пересчитать профиль моего стиля общения с этим контактом\n"
    "🧠 /memory — показать память (факты о контактах)\n"
    "📬 /threads — активные переписки\n\n"
    "📖 /help — эта подсказка\n\n"
    "<b>Можно писать своими словами:</b>\n"
    "<i>• «Напиши Ивану что задержусь на 10 минут»</i> → отправка сообщения\n"
    "<i>• «Что нового в чате с Петей?»</i> → саммари переписки\n"
    "<i>• «Напомни завтра в 10 про отчёт»</i> → напоминание\n"
    "<i>• «Где мы остановились с Машей?»</i> → catchup\n"
    "<i>• «Запомни: у Насти ДР 15 июня»</i> → память\n"
    "<i>• «Сделай краткую выжимку новостей про AI»</i> → дайджест\n"
    "<i>• «Ответь Игорю: давай в среду»</i> → черновик ответа\n"
    "<i>• «Какие у меня задачи?»</i> → список обещаний\n"
)


# ─── existing greeting (returning users) ──────────────────────────────


def _greeting_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📋 /help", callback_data="nav:help"),
                InlineKeyboardButton(text="⚙️ /settings", callback_data="nav:settings"),
                InlineKeyboardButton(text="💬 /chat", callback_data="nav:chat"),
            ],
            [
                InlineKeyboardButton(text="📬 Треды", callback_data="thread:refresh"),
                InlineKeyboardButton(text="📋 Задачи", callback_data="nav:todos"),
                InlineKeyboardButton(text="🧠 Память", callback_data="nav:memory"),
            ],
            [
                InlineKeyboardButton(
                    text="🎭 Личность", callback_data="set:sec:personality"
                ),
            ],
        ]
    )


# ─── /start ────────────────────────────────────────────────────────────


@router.message(Command("start"), StateFilter(default_state))
async def cmd_start(message: Message) -> None:
    """Точка входа. Если пользователь уже прошёл онбординг — обычное приветствие."""
    tg_id = message.from_user.id

    if await is_onboarded(tg_id):
        await _show_regular_greeting(message)
        return

    # Начинаем онбординг — шаг 1: Welcome
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🚀 Начать", callback_data="onboarding:start"
                ),
            ],
        ]
    )
    await message.answer(
        "👋 <b>Привет! Я твой AI-ассистент для Telegram</b>\n\n"
        "Давай настроим всё за 5 шагов, чтобы я мог полноценно работать.\n\n"
        "<b>Что настроим:</b>\n"
        "1️⃣ 🔑 Подключим твой Telegram-аккаунт\n"
        "2️⃣ 🤖 Добавим API-ключ для ИИ\n"
        "3️⃣ 🕐 Выберем часовой пояс\n"
        "4️⃣ 📱 Настроим синхронизацию чатов\n\n"
        "Готов? 👇",
        reply_markup=kb,
    )


@router.message(Command("start"), StateFilter(*list(OnboardingStates)))
async def cmd_start_during_onboarding(message: Message) -> None:
    """Если пользователь нажал /start во время онбординга — показываем текущий шаг."""
    await message.answer(
        "🔄 Ты уже проходишь настройку. Напиши /cancel чтобы выйти, "
        "или продолжай — я жду твой ответ на текущий шаг 😊"
    )


async def _show_regular_greeting(message: Message) -> None:
    """Полное приветствие для вернувшегося пользователя."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        has_session = owner.session is not None
        llm = _pretty_provider(owner.settings.llm_provider)
        tz = tz_short(owner.settings.timezone) if owner.settings.timezone else "UTC"

        # Проверяем, новый ли пользователь (нет persona или 0 взаимодействий)
        from src.db.models._learning import AdaptivePersona
        from sqlalchemy import select

        stmt = select(AdaptivePersona).where(AdaptivePersona.user_id == owner.id)
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

    is_new = (persona is None) or (persona.total_interactions == 0)

    auth_status = "Ты авторизован ✅" if has_session else "Не авторизован ❌"

    header = (
        f"👋 <b>Привет! Я твой AI-ассистент для Telegram</b>\n\n"
        f"<b>📊 Текущий статус</b>\n"
        f"{auth_status}\n"
        f"🤖 LLM: {llm}\n"
        f"🕐 Часовой пояс: {tz}\n\n"
    )

    onboarding_text = ""
    if is_new:
        onboarding_text = (
            "\n\n🎭 <b>Хочешь настроить личность бота под себя?</b>\n"
            "Я могу общаться в разных стилях: профессионально, дружелюбно, "
            "игриво, лаконично и даже с сарказмом!\n\n"
            "Нажми кнопку ниже чтобы настроить."
        )

    await message.answer(
        header + WELCOME + onboarding_text, reply_markup=_greeting_kb()
    )


# ─── /help ─────────────────────────────────────────────────────────────


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        auth_status = "✅" if owner.session else "❌"
        llm = _pretty_provider(owner.settings.llm_provider)
    header = (
        f"📖 <b>Помощь по командам</b>\n"
        f"{'Ты авторизован' if owner.session else 'Не авторизован'} {auth_status} · "
        f"LLM: {llm}\n\n"
    )
    await message.answer(header + WELCOME)


# ─── navigation callbacks (existing) ───────────────────────────────────


@router.callback_query(F.data.startswith("nav:"))
async def cb_nav(callback: CallbackQuery) -> None:
    """Обработка навигационных кнопок."""
    target = callback.data.split(":", 1)[1]
    mapping = {
        "help": "/help",
        "settings": "/settings",
        "chat": "/chat",
        "todos": "/todos",
        "memory": "/memory",
        "threads": "/threads",
    }
    cmd = mapping.get(target, f"/{target}")
    await callback.answer(f"Выполняю {cmd}")
    if callback.message:
        await callback.message.edit_text(
            f"🔄 Нажми в поле ввода: <code>{cmd}</code> и отправь."
        )


@router.callback_query(F.data == "persona:skip_onboarding")
async def cb_skip_onboarding(callback: CallbackQuery) -> None:
    """Пользователь пропустил onboarding личности."""
    await callback.answer(
        "Ок, настройки можно изменить в любой момент в /settings → 🎭 Личность"
    )
    if callback.message:
        try:
            await callback.message.delete()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# ONBOARDING WIZARD
# ═══════════════════════════════════════════════════════════════════════

# ─── Step 1: Welcome → "🚀 Начать" callback ───────────────────────────


@router.callback_query(F.data == "onboarding:start")
async def cb_onboarding_start(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь нажал «Начать» — переходим к шагу авторизации."""
    await state.set_state(OnboardingStates.waiting_login)
    await callback.answer()

    if callback.message:
        # Убираем кнопку "Начать"
        try:
            await callback.message.edit_text(
                "👋 <b>Привет! Я твой AI-ассистент для Telegram</b>\n\n"
                "Давай настроим всё за 5 шагов 🚀"
            )
        except Exception:
            pass

    if callback.message is None:
        await callback.answer("Сообщение недоступно.")
        return
    await _send_login_step(callback.message.chat.id, callback.bot)


async def _send_login_step(chat_id: int, bot, state: FSMContext | None = None) -> None:
    """Отправляет сообщение шага «Авторизация»."""
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔑 /login", callback_data="onboarding:hint_login"
                ),
            ],
        ]
    )
    await bot.send_message(
        chat_id,
        "🚀 <b>Шаг 1/4 — Авторизация</b>\n\n"
        "Подключи свой Telegram-аккаунт командой /login",
        reply_markup=kb,
    )


@router.callback_query(F.data == "onboarding:hint_login")
async def cb_onboarding_hint_login(callback: CallbackQuery) -> None:
    """Подсказка как отправить /login."""
    await callback.answer()
    if callback.message:
        await callback.message.edit_text(
            "🔑 Просто отправь в чат команду:\n\n"
            "<code>/login</code>\n\n"
            "И следуй инструкциям бота. После успешного входа "
            "я продолжу настройку автоматически."
        )


@router.message(OnboardingStates.waiting_login)
async def step_onboarding_login(message: Message, state: FSMContext) -> None:
    """Пользователь что-то отправил на шаге login (не /login)."""
    # Проверяем, может уже есть сессия
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        has_session = owner.session is not None

    if has_session:
        # Уже авторизован — переходим к следующему шагу
        await state.set_state(OnboardingStates.waiting_llm_key)
        await _send_llm_key_step(message.chat.id, message.bot)
        return

    await message.answer(
        "🔑 Нажми /login или нажми на кнопку выше, чтобы авторизоваться.\n"
        "/cancel — отменить настройку."
    )


# ─── Step 2: LLM Key ─────────────────────────────────────────────────


async def _send_llm_key_step(chat_id: int, bot) -> None:
    """Отправляет сообщение шага «Ключ для ИИ»."""
    text = (
        "🔑 <b>Шаг 2/4 — подключи мозг</b>\n\n"
        "Форматы ключей:\n"
        "• OpenAI:      sk-proj-...\n"
        "• Anthropic:   sk-ant-api03-...\n"
        "• Gemini:      AIzaSy...\n"
        "• Mistral:     Nb...\n"
        "• OpenRouter:  sk-or-...\n"
        "• Cloudflare:  (длинный токен)\n"
        "• Groq:        gsk_..."
    )
    await bot.send_message(chat_id, text)


@router.message(OnboardingStates.waiting_llm_key)
async def step_onboarding_llm_key(message: Message, state: FSMContext) -> None:
    """Обрабатывает отправленный LLM-ключ."""
    raw = (message.text or "").strip()
    if not raw:
        await message.answer("Пустой ключ. Пришли API-ключ или /cancel.")
        return

    tg_id = message.from_user.id

    # Пробуем определить провайдера и валидировать
    provider = _detect_provider(raw)
    if provider is None:
        await message.answer(
            "❌ Не удалось определить провайдера по формату ключа.\n\n"
            "Поддерживаются:\n"
            "• <b>OpenAI</b> — начинается на <code>sk-</code>\n"
            "• <b>Gemini</b> — AIzaSy...\n\n"
            "Попробуй ещё раз или /cancel."
        )
        return

    validated, error_hint = await _validate_key(provider, raw)
    if not validated:
        hint = (
            error_hint
            or f"Ключ {provider} не прошёл проверку. Убедись что ключ правильный."
        )
        await message.answer(f"❌ {hint}\n/cancel — отмена.")
        return

    # Сохраняем ключ
    try:
        await message.delete()
    except Exception:
        pass

    async with get_session() as session:
        owner = await get_or_create_user(session, tg_id)
        await upsert_api_key(session, owner, provider, raw)

    await state.set_state(OnboardingStates.waiting_timezone)
    await message.answer(f"✅ Ключ <b>{provider}</b> сохранён и проверен!")
    await _send_timezone_step(message.chat.id, message.bot)


def _detect_provider(key: str) -> str | None:
    """Пытается определить провайдера по формату ключа."""
    key = key.strip()
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    if key.startswith("AIzaSy"):
        return "gemini"
    if key.startswith("Nb"):
        return "mistral"
    # Cloudflare — Workers AI tokens (long base64, no standard prefix)
    if len(key) > 64 and not key.startswith("sk-") and not key.startswith("AIzaSy"):
        return "cloudflare"
    return None


async def _validate_key(provider: str, key: str) -> tuple[bool, str | None]:
    """Валидирует ключ через провайдера. Возвращает (valid, error_hint)."""
    try:
        if provider == "openai":
            return (await OpenAIProvider(key).validate_key(), None)
        if provider == "gemini":
            return (await GeminiProvider(key).validate_key(), None)
        if provider == "mistral":
            from src.llm.mistral_provider import MistralProvider

            return (await MistralProvider(key).validate_key(), None)
        if provider == "cloudflare":
            from src.llm.cloudflare_provider import CloudflareProvider

            return (await CloudflareProvider(key).validate_key(), None)
        if provider == "openrouter":
            from src.llm.openrouter_provider import OpenRouterProvider

            return (await OpenRouterProvider(key).validate_key(), None)
        if provider == "anthropic":
            from src.llm.anthropic_provider import AnthropicProvider

            return (await AnthropicProvider(key).validate_key(), None)
    except Exception as e:
        err_str = str(e).lower()
        if any(
            w in err_str
            for w in ("timeout", "connect", "resolve", "network", "refused", "reset")
        ):
            return (False, "Сетевая ошибка. Проверь подключение и попробуй снова.")
        logger.exception("Key validation failed for %s", provider)
        return (False, "Не удалось проверить ключ. Попробуй позже.")
    return (False, f"Неизвестный провайдер: {provider}")


# ─── Step 3: Timezone ─────────────────────────────────────────────────


async def _send_timezone_step(chat_id: int, bot) -> None:
    """Отправляет сообщение шага «Часовой пояс»."""
    # Строим клавиатуру с популярными TZ
    rows = []
    for tz_name in TZ_PRESETS:
        label = _tz_button_label(tz_name)
        rows.append(
            [InlineKeyboardButton(text=label, callback_data=f"onboarding:tz:{tz_name}")]
        )

    kb = InlineKeyboardMarkup(inline_keyboard=rows)

    await bot.send_message(
        chat_id,
        "🕐 <b>Шаг 3/4 — часовой пояс</b>\n\n"
        "Выбери свой город или введи вручную (например, Europe/Moscow)",
        reply_markup=kb,
    )


def _tz_button_label(tz_name: str) -> str:
    """Короткая метка кнопки TZ."""
    try:
        short = tz_short(tz_name)
        return short
    except Exception:
        return tz_name


@router.callback_query(F.data.startswith("onboarding:tz:"))
async def cb_onboarding_tz(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь выбрал часовой пояс из списка."""
    tz_value = callback.data[len("onboarding:tz:") :]
    if not is_valid_tz(tz_value):
        await callback.answer("Неизвестный часовой пояс", show_alert=True)
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        owner.settings.timezone = tz_value

    await callback.answer(f"✅ Часовой пояс: {tz_short(tz_value)}")
    await state.set_state(OnboardingStates.waiting_sync_choice)
    if callback.message is None:
        await callback.answer("Сообщение недоступно.")
        return
    await _send_sync_step(callback.message.chat.id, callback.bot)

    # Убираем клавиатуру
    if callback.message:
        try:
            await callback.message.edit_text(
                f"✅ Часовой пояс: <b>{tz_short(tz_value)}</b>"
            )
        except Exception:
            pass


@router.message(OnboardingStates.waiting_timezone)
async def step_onboarding_timezone(message: Message, state: FSMContext) -> None:
    """Пользователь ввёл часовой пояс текстом."""
    tz_value = (message.text or "").strip()
    if not is_valid_tz(tz_value):
        await message.answer(
            "Не нашёл такой TZ. Используй IANA-формат, например "
            "<code>Europe/Moscow</code>.\n"
            "Или выбери из списка выше.\n"
            "/cancel — отмена."
        )
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.timezone = tz_value

    await state.set_state(OnboardingStates.waiting_sync_choice)
    await message.answer(f"✅ Часовой пояс: <b>{tz_short(tz_value)}</b>")
    await _send_sync_step(message.chat.id, message.bot)


# ─── Step 4: Sync choice ──────────────────────────────────────────────


async def _send_sync_step(chat_id: int, bot) -> None:
    """Отправляет сообщение шага «Синхронизация чатов»."""
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📱 Все личные чаты",
                    callback_data="onboarding:sync:all",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="📂 Выбрать папки",
                    callback_data="onboarding:sync:folders",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="⏭ Пропустить",
                    callback_data="onboarding:sync:skip",
                ),
            ],
        ]
    )

    await bot.send_message(
        chat_id,
        "📱 <b>Шаг 4/4 — синхронизация контактов</b>\n\n"
        "Я прочитаю твои диалоги и запомню важное. Это займёт 2-5 минут.",
        reply_markup=kb,
    )


@router.callback_query(F.data == "onboarding:sync:all")
async def cb_onboarding_sync_all(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь выбрал синхронизацию всех личных чатов."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        owner.settings.monitor_only_selected_folders = False

    await callback.answer("📱 Начинаю синхронизацию...")

    # Запускаем синхронизацию в фоне
    from src.userbot.dialogs import sync_dialogs

    try:
        await sync_dialogs(callback.from_user.id)
        status = "✅ Список чатов обновлён!"
    except Exception as exc:
        logger.exception("sync_dialogs during onboarding failed")
        status = f"⚠️ Синхронизация не удалась: {exc}"

    await state.clear()
    await _finish_onboarding(
        callback.message.chat.id,
        callback.bot,
        tg_id=callback.from_user.id,
        extra=status,
    )

    if callback.message:
        try:
            await callback.message.edit_text("📱 Синхронизация запущена ✅")
        except Exception:
            pass


@router.callback_query(F.data == "onboarding:sync:folders")
async def cb_onboarding_sync_folders(
    callback: CallbackQuery, state: FSMContext
) -> None:
    """Пользователь выбрал синхронизацию по папкам."""
    # Запрашиваем имена папок
    await state.set_state(OnboardingStates.waiting_sync_choice)
    await callback.answer()
    if callback.message:
        await callback.message.edit_text(
            "📂 Напиши названия папок через запятую, которые нужно отслеживать.\n\n"
            "Например: <code>Работа, Семья, Друзья</code>\n\n"
            "Или нажми /cancel чтобы пропустить."
        )


@router.message(OnboardingStates.waiting_sync_choice)
async def step_onboarding_sync_folders_text(
    message: Message, state: FSMContext
) -> None:
    """Пользователь ввёл названия папок для синхронизации."""
    folders_text = (message.text or "").strip()
    if not folders_text:
        await message.answer(
            "Пустой список. Напиши названия папок через запятую или /cancel."
        )
        return

    folder_names = [f.strip() for f in folders_text.split(",") if f.strip()]
    if not folder_names:
        await message.answer(
            "Нужно хотя бы одно название папки. Попробуй ещё раз или /cancel."
        )
        return

    import json

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.monitored_folders = json.dumps(folder_names)
        owner.settings.monitor_only_selected_folders = True

    # Запускаем синхронизацию
    from src.userbot.dialogs import sync_dialogs

    try:
        await sync_dialogs(message.from_user.id)
        status = "✅ Чаты из выбранных папок синхронизированы!"
    except Exception as exc:
        logger.exception("sync_dialogs during onboarding (folders) failed")
        status = f"⚠️ Синхронизация не удалась: {exc}"

    await state.clear()
    await _finish_onboarding(
        message.chat.id,
        message.bot,
        tg_id=message.from_user.id,
        extra=f"📂 Папки: {', '.join(folder_names)}\n{status}",
    )


@router.callback_query(F.data == "onboarding:sync:skip")
async def cb_onboarding_sync_skip(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь пропустил синхронизацию."""
    await callback.answer("Ок, можно настроить позже в /settings → Синхронизация")
    await state.clear()

    if callback.message:
        try:
            await callback.message.edit_text("⏭ Синхронизация пропущена")
        except Exception:
            pass

    if callback.message is None:
        await callback.answer("Сообщение недоступно.")
        return
    await _finish_onboarding(
        callback.message.chat.id, callback.bot, tg_id=callback.from_user.id
    )


# ─── Finish ────────────────────────────────────────────────────────────


async def _finish_onboarding(chat_id: int, bot, tg_id: int, extra: str = "") -> None:
    """Финальное сообщение с детальным саммари после завершения онбординга."""

    tone_labels = {
        "professional": "Деловой",
        "friendly": "Тёплый",
        "efficient": "Эффективный",
        "default": "Стандартный",
        "cynical": "Циничный",
        "warm": "Тёплый",
    }

    async with get_session() as session:
        owner = await get_or_create_user(session, tg_id)

        # Считаем контакты
        contact_count: int = (
            await session.scalar(
                select(func.count())
                .select_from(Contact)
                .where(Contact.user_id == owner.id)
            )
        ) or 0

        # Считаем активные факты в памяти
        fact_count: int = (
            await session.scalar(
                select(func.count())
                .select_from(Memory)
                .where(Memory.user_id == owner.id, Memory.is_active.is_(True))
            )
        ) or 0

        # Данные сессии
        session_label = "—"
        if owner.session:
            session_label = owner.session.account_label or owner.session.phone or "—"

        # LLM ключи
        providers = sorted(
            {k.provider for k in owner.key_slots if getattr(k, "enabled", True)}
        )
        key_names = ", ".join(_pretty_provider(p) for p in providers)
        if not key_names:
            key_names = "—"

        # Часовой пояс
        tz_name = owner.settings.timezone or "UTC"

        # Режим личности
        persona = await session.scalar(
            select(AdaptivePersona).where(AdaptivePersona.user_id == owner.id)
        )
        tone_key = persona.base_tone if persona else "default"
        tone_label = tone_labels.get(tone_key, tone_key)

    msg = (
        "<b>Итог настройки</b>\n"
        f"• Сессия: {session_label}\n"
        f"• Контакты: {contact_count}\n"
        f"• Факты в памяти: {fact_count}\n"
        f"• LLM-ключи: {len(providers)} ({key_names})\n"
        f"• Часовой пояс: {tz_name}\n"
        f"• Тон: {tone_label}\n\n"
        "🎉 <b>Я полностью настроен и готов к работе!</b>\n\n"
        "Что я теперь умею:\n"
        "🧠 Помню факты о тебе и контактах\n"
        "💬 Авто-отвечаю в ЛС пока ты занят\n"
        "📋 Веду список дел и напоминаю\n"
        "📰 Собираю дайджест новостей\n"
        "🔍 Ищу по истории переписок\n"
        "🌤️ Погода, крипта, whois, таймеры\n\n"
        "Просто напиши мне — я пойму.\n"
        "Подробнее: /help"
    )
    if extra:
        msg = extra + "\n\n" + msg

    await bot.send_message(chat_id, msg)


# ─── advance_onboarding_after_login ────────────────────────────────────


async def advance_onboarding_after_login(message: Message, state: FSMContext) -> bool:
    """Вызывается из login.py после успешного входа.

    Если пользователь ещё не прошёл онбординг — переводит на следующий шаг
    и возвращает True. Если онбординг не нужен — возвращает False.
    """
    tg_id = message.from_user.id
    async with get_session() as session:
        owner = await get_or_create_user(session, tg_id)
        has_session = owner.session is not None
        has_llm_key = len(owner.key_slots) > 0
        has_tz = owner.settings.timezone not in (None, "", "UTC", "Etc/UTC")

    # Если после логина пользователь уже полностью готов — не вмешиваемся
    if has_session and has_llm_key and has_tz:
        return False

    # Переходим к шагу LLM ключа
    await state.set_state(OnboardingStates.waiting_llm_key)
    await message.answer(
        "✅ Готово! <b>Шаг 2/4 — API-ключ</b>\n\nТеперь нужен ключ для доступа к LLM. Выбери провайдера:"
    )
    await _send_llm_key_step(message.chat.id, message.bot)
    return True
