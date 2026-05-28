import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from telethon.errors import (
    ApiIdInvalidError,
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    SessionPasswordNeededError,
)

from src.config import settings
from src.bot.filters import OwnerOnly
from src.bot.states import CustomProviderStates, LoginStates, OnboardingStates
from src.db.repo import (
    delete_telegram_session,
    get_or_create_user,
    load_telegram_session,
    save_telegram_session,
)
from src.db.session import get_session
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)
router = Router(name="login")
router.message.filter(OwnerOnly())


CANCEL_HINT = "В любой момент можно отменить командой /cancel."


# Global /cancel handler — сбрасывает ЛЮБОЕ FSM-состояние, не только login.
@router.message(Command("cancel"))
async def cmd_cancel(
    message: Message, state: FSMContext, userbot_manager: UserbotManager
) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Нечего отменять.")
        return

    # Если пользователь в процессе кастомного провайдера — возвращаем к выбору
    if current and current.startswith("CustomProviderStates"):
        await state.set_state(OnboardingStates.waiting_provider_choice)
        await message.answer("⏪ Отменено.")
        from src.bot.handlers.start import _send_llm_key_step

        await _send_llm_key_step(message.chat.id, message.bot)
        return

    await userbot_manager.cancel_pending(message.from_user.id)
    await state.clear()
    await message.answer("Отменено.")


@router.message(Command("logout"))
async def cmd_logout(message: Message, userbot_manager: UserbotManager) -> None:
    tg_id = message.from_user.id
    await userbot_manager.remove_client(tg_id)
    async with get_session() as session:
        user = await get_or_create_user(session, tg_id)
        await delete_telegram_session(session, user)
    await message.answer("✅ Сессия удалена. Чтобы подключиться заново — /login.")


@router.message(Command("login"))
async def cmd_login(
    message: Message, state: FSMContext, userbot_manager: UserbotManager
) -> None:
    tg_id = message.from_user.id

    async with get_session() as session:
        user = await get_or_create_user(session, tg_id)
        existing = await load_telegram_session(session, user)

    if existing is not None and userbot_manager.get_client(tg_id) is not None:
        await message.answer(
            "Аккаунт уже подключён. Сначала выполни /logout, если хочешь подключить другой."
        )
        return

    # Предзаполняем дефолтные значения из конфига или хардкод
    _default_api_id = settings.api_id if settings.api_id is not None else 2040
    _default_api_hash = (
        settings.api_hash
        if settings.api_hash is not None
        else "b18441a1ff607e10a989891a5462e627"
    )
    await state.update_data(api_id=_default_api_id, api_hash=_default_api_hash)

    await state.set_state(LoginStates.api_id)
    await message.answer(
        "🔐 <b>Подключение Telegram-аккаунта</b>\n\n"
        "Получи <code>api_id</code> и <code>api_hash</code> на https://my.telegram.org → API development tools.\n"
        "Никому их не отправляй, кроме этого бота. Я храню их в зашифрованном виде.\n\n"
        f"Введи <b>api_id</b> или напиши <b>1</b> — поставлю дефолтный (<code>{_default_api_id}</code>).\n"
        f"{CANCEL_HINT}"
    )


@router.message(LoginStates.api_id)
async def step_api_id(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if text == "1":
        # Хардкод-дефолт или из .env
        text = str(settings.api_id if settings.api_id is not None else 2040)
    if not text.isdigit():
        await message.answer("api_id — это число. Попробуй ещё раз или /cancel.")
        return
    await state.update_data(api_id=int(text))
    await state.set_state(LoginStates.api_hash)
    _default_hash = (
        settings.api_hash
        if settings.api_hash is not None
        else "b18441a1ff607e10a989891a5462e627"
    )
    await message.answer(
        "Отлично. Теперь введи <b>api_hash</b> (32 hex-символа).\n"
        f"Напиши <b>1</b> — поставлю дефолтный (<code>{_default_hash[:4]}••••{_default_hash[-4:]}</code>).\n"
        "Или введи свой."
    )


@router.message(LoginStates.api_hash)
async def step_api_hash(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    # Auto-accept default api_hash if user types keyword or presses Enter
    _default_hash = (
        settings.api_hash
        if settings.api_hash is not None
        else "b18441a1ff607e10a989891a5462e627"
    )
    if not text or text.lower() in (
        "дефолтный",
        "дефолт",
        "1",
        "default",
        "стандартный",
        "да",
        "ок",
        "yes",
        "ok",
    ):
        text = _default_hash
    if len(text) != 32 or not all(c in "0123456789abcdefABCDEF" for c in text):
        await message.answer(
            "api_hash должен быть строкой из 32 hex-символов. Попробуй ещё раз или /cancel."
        )
        return
    await state.update_data(api_hash=text)
    await state.set_state(LoginStates.phone)
    await message.answer(
        "Введи номер телефона в международном формате, например <code>+79991234567</code>."
    )


@router.message(LoginStates.phone)
async def step_phone(
    message: Message, state: FSMContext, userbot_manager: UserbotManager
) -> None:
    phone = (message.text or "").strip().replace(" ", "")
    if not phone.startswith("+") or not phone[1:].isdigit() or len(phone) < 8:
        await message.answer(
            "Не похоже на телефон. Должно быть как <code>+79991234567</code>. /cancel — выйти."
        )
        return

    data = await state.get_data()
    api_id: int = data["api_id"]
    api_hash: str = data["api_hash"]

    pending = userbot_manager.start_pending(message.from_user.id, api_id, api_hash)
    pending.phone = phone

    try:
        await pending.client.connect()
        sent = await pending.client.send_code_request(phone)
        pending.phone_code_hash = sent.phone_code_hash
    except PhoneNumberInvalidError:
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer(
            "❌ Telegram сказал: неверный номер. Запусти /login заново."
        )
        return
    except ApiIdInvalidError:
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer("❌ api_id/api_hash неверны. Запусти /login заново.")
        return
    except FloodWaitError as e:
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer(
            f"❌ FloodWait: подожди {e.seconds} секунд и попробуй /login снова."
        )
        return
    except Exception:
        logger.exception("send_code_request failed")
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer("❌ Не удалось отправить код. Запусти /login заново.")
        return

    await state.set_state(LoginStates.code)
    await message.answer(
        "📨 Код отправлен. Введи его, но <b>с пробелами между цифрами</b>, например: "
        "<code>1 2 3 4 5</code> — иначе Telegram автоматически инвалидирует код, "
        "увидев его открыто в чате."
    )


@router.message(LoginStates.code)
async def step_code(
    message: Message, state: FSMContext, userbot_manager: UserbotManager
) -> None:
    raw = (message.text or "").strip()
    code = "".join(ch for ch in raw if ch.isdigit())
    if not code:
        await message.answer("Не вижу цифр. Попробуй ещё раз или /cancel.")
        return

    pending = userbot_manager.get_pending(message.from_user.id)
    if pending is None:
        await state.clear()
        await message.answer("Сессия логина потерялась. Начни заново через /login.")
        return

    try:
        await pending.client.sign_in(
            phone=pending.phone,
            code=code,
            phone_code_hash=pending.phone_code_hash,
        )
    except SessionPasswordNeededError:
        await state.set_state(LoginStates.password_2fa)
        await message.answer(
            "🔒 У аккаунта включена двухфакторная аутентификация. Введи пароль 2FA.\n"
            "Сообщение с паролем удалю сразу после успешного входа."
        )
        return
    except PhoneCodeInvalidError:
        await message.answer("❌ Неверный код. Попробуй ещё раз или /cancel.")
        return
    except PhoneCodeExpiredError:
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer("❌ Код истёк. Запусти /login заново.")
        return
    except Exception:
        logger.exception("sign_in failed")
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer("❌ Не удалось войти. Запусти /login заново.")
        return

    await _finalize_login(message, state, userbot_manager)


@router.message(LoginStates.password_2fa)
async def step_2fa(
    message: Message, state: FSMContext, userbot_manager: UserbotManager
) -> None:
    password = (message.text or "").strip()
    if not password:
        await message.answer("Пустой пароль. Введи 2FA-пароль или /cancel.")
        return

    pending = userbot_manager.get_pending(message.from_user.id)
    if pending is None:
        await state.clear()
        await message.answer("Сессия логина потерялась. Начни заново через /login.")
        return

    try:
        await pending.client.sign_in(password=password)
    except PasswordHashInvalidError:
        await message.answer("❌ Неверный пароль 2FA. Попробуй ещё раз.")
        return
    except Exception:
        logger.exception("2FA sign_in failed")
        await userbot_manager.cancel_pending(message.from_user.id)
        await state.clear()
        await message.answer("❌ Не удалось войти. Запусти /login заново.")
        return
    else:
        # Удалим сообщение с паролем — гигиена.
        try:
            await message.delete()
        except Exception:
            logger.debug("login: could not delete password message", exc_info=True)

        await _finalize_login(message, state, userbot_manager)
    finally:
        # Очищаем пароль из памяти — гарантированно выполняется всегда
        password = None  # allow GC
        del password  # remove reference


async def _finalize_login(
    message: Message, state: FSMContext, userbot_manager: UserbotManager
) -> None:
    tg_id = message.from_user.id
    pending = userbot_manager.clear_pending(tg_id)
    if pending is None:
        await state.clear()
        await message.answer("Что-то пошло не так. Запусти /login заново.")
        return

    me = await pending.client.get_me()
    label_parts = [
        p
        for p in [getattr(me, "first_name", None), getattr(me, "last_name", None)]
        if p
    ]
    label = " ".join(label_parts) or (me.username or str(me.id))
    session_string = pending.client.session.save()

    async with get_session() as session:
        user = await get_or_create_user(session, tg_id)
        await save_telegram_session(
            session,
            user,
            api_id=pending.api_id,
            api_hash=pending.api_hash,
            session_string=session_string,
            phone=pending.phone or "",
            account_label=label,
        )

    userbot_manager.register_client(tg_id, pending.client)
    await state.clear()

    # Если пользователь в процессе онбординга — переводим на следующий шаг
    from src.bot.handlers.start import advance_onboarding_after_login

    if await advance_onboarding_after_login(message, state):
        return  # онбординг продолжится сам

    await message.answer(
        f"✅ Аккаунт <b>{label}</b> подключён. Сессия сохранена в зашифрованном виде.\n\n"
        "Дальше — /settings, чтобы выбрать LLM и настроить авто-ответ."
    )
    await message.answer(
        "🎉 <b>Готово!</b>\n\n"
        "Вот что я умею:\n"
        "👤 /contact Имя — что я знаю о человеке\n"
        "📝 /send Имя текст — написать кому-то\n"
        "🔍 /search запрос — найти в чатах\n"
        "📋 /todos — твои обещания\n"
        "📰 /news тема — дайджест каналов\n"
        "⚙️ /settings — настроить всё\n"
        "📖 /help — все команды\n\n"
        "Или просто напиши мне — я пойму на обычном языке 🗣"
    )
