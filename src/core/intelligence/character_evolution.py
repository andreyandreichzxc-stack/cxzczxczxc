"""Character Evolution Engine — бот сам анализирует диалоги и развивает характер.

Основные функции:
- maybe_evolve_after_turn() — вызывается после каждого ответа, сама решает когда эволюционировать
- compile_experience() — LLM пишет «вывод из опыта» на основе последних диалогов
- evolve_from_feedback() — пользователь сказал что-то про характер бота
"""

from __future__ import annotations

import json
import logging
import time

from src.core.memory import conversation_context as ctx_store
from src.db.repo import get_or_create_user, get_persona
from src.db.session import get_session
from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

# ── Константы ─────────────────────────────────────────────────────────

# Как часто можно эволюционировать (секунд)
MIN_EVOLVE_INTERVAL_SEC = 7200  # 2 часа
# Каждые N сообщений — плановая эволюция
EVOLVE_EVERY_N_TURNS = 20
# Максимальная длина experience-текста
MAX_EXPERIENCE_LENGTH = 500
# Максимум эволюций в день
MAX_EVOLVES_PER_DAY = 5
# Сколько последних диалогов показывать LLM
RECENT_DIALOGS_LIMIT = 6

# ── Промпт для LLM ────────────────────────────────────────────────────

EVOLVE_PROMPT = """Ты — анализатор личности ассистента. Посмотри на последние диалоги с владельцем и текущие настройки стиля.

Твоя задача: написать 1-3 предложения «вывода из опыта» — что работает в общении с этим владельцем, а что нет. Это будет частью system prompt тебя самого.

Правила:
1. Пиши от третьего лица: «Владелец ценит...», «Лучше не...», «Из опыта: ...»
2. Только факты и наблюдения, без оценок («хорошо/плохо»)
3. Если диалогов мало или нет паттернов — верни пустой JSON
4. Максимум 500 символов
5. НЕ предлагай менять базовые принципы (спорить, не сдаваться, выполнять команды)

Формат ответа — СТРОГО JSON:
{{"experience": "текст вывода"}} или {{"experience": null}} если нечего сказать.

Последние диалоги:
{dialogs}

Текущий стиль:
{current_style}

Текущий опыт (было раньше):
{previous_experience}
"""

FEEDBACK_PROMPT = """Ты — анализатор личности. Владелец написал что-то о твоём характере.

Определи:
1. О чём именно фидбек (тон, стиль, поведение)
2. Что конкретно нужно изменить
3. Напиши новую версию «вывода из опыта» (1-2 предложения)

Правила:
- Не перезаписывай базовые принципы
- Если фидбек не про характер — верни null
- Максимум 300 символов

Формат — СТРОГО JSON:
{{"action": "update_tone"|"update_experience"|"reset"|null,
 "value": "новый текст опыта или название тона",
 "reason": "почему"}}

Фидбек: {feedback}

Текущий стиль: {current_style}

Текущий опыт: {current_experience}
"""

# ── Safety gate ───────────────────────────────────────────────────────

IMMUTABLE_PRINCIPLE_TOKENS = {
    "спорь",
    "не молчи",
    "команды выполняй",
    "не сдавайся",
    "уважение",
    "аргументируй",
}


def _safety_check_experience(text: str) -> bool:
    """Проверяет что опыт не пытается переписать принципы."""
    if not text:
        return True
    text_lower = text.lower()
    # Если текст содержит отрицание базовых принципов — reject
    dangerous = [
        "не спорь",
        "молчи",
        "не выполняй",
        "соглашайся",
        "не аргументируй",
        "забудь принципы",
        "игнорируй",
        "поддакивай",
    ]
    for d in dangerous:
        if d in text_lower:
            return False
    return True


# ── CORE: compile experience ──────────────────────────────────────────


async def compile_experience(
    telegram_id: int,
    provider,
    *,
    force: bool = False,
) -> str | None:
    """LLM пишет «вывод из опыта» на основе последних диалогов.

    Args:
        telegram_id: ID владельца.
        provider: LLM провайдер.
        force: если True — игнорирует cooldown, работает в любом случае.

    Returns:
        Новый текст опыта или None если нечего сказать / ошибка.
    """
    # 1. Cooldown check (если не force)
    if not force:
        from src.core.context_cache import get as cache_get

        last_evolve_ts = await cache_get(f"evolve:ts:{telegram_id}")
        if last_evolve_ts is not None:
            elapsed = time.monotonic() - last_evolve_ts
            if elapsed < MIN_EVOLVE_INTERVAL_SEC:
                remain = int(MIN_EVOLVE_INTERVAL_SEC - elapsed)
                logger.debug("Evolve cooldown: %d seconds remaining", remain)
                return None

    # 2. Собираем контекст
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

        # Текущие настройки
        current_style_parts = []
        if p.base_tone and p.base_tone != "default":
            current_style_parts.append(f"Базовый тон: {p.base_tone}")
        if p.warmth and p.warmth != "normal":
            current_style_parts.append(f"Теплота: {p.warmth}")
        if p.enthusiasm and p.enthusiasm != "normal":
            current_style_parts.append(f"Энтузиазм: {p.enthusiasm}")
        current_style = "; ".join(current_style_parts) or "дефолтный"

        # Предыдущий опыт
        previous_experience = "нет"
        try:
            if p.custom_instructions:
                ci = (
                    json.loads(p.custom_instructions)
                    if isinstance(p.custom_instructions, str)
                    else p.custom_instructions
                )
                if isinstance(ci, dict):
                    prev_exp = ci.get("experience")
                    if prev_exp:
                        previous_experience = prev_exp[:200]
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Берём последние диалоги из conversation_context
    turns = await ctx_store.get_recent_turns(telegram_id)
    if not turns:
        logger.debug("compile_experience: no recent dialogs")
        return None

    # Форматируем диалоги
    dialog_lines = []
    for user_text, assistant_summary in turns[-RECENT_DIALOGS_LIMIT:]:
        dialog_lines.append(f"Пользователь: {user_text}")
        dialog_lines.append(f"Ассистент: {assistant_summary}")
    dialogs_text = "\n".join(dialog_lines)

    if not dialogs_text.strip():
        return None

    # 4. Запрашиваем LLM
    prompt = EVOLVE_PROMPT.format(
        dialogs=dialogs_text.replace("{", "{{").replace("}", "}}"),
        current_style=current_style,
        previous_experience=previous_experience,
    )

    try:
        import asyncio

        raw = await asyncio.wait_for(
            provider.chat(
                [ChatMessage(role="user", content=prompt)],
                heavy=False,
            ),
            timeout=30.0,
        )
    except Exception:
        logger.debug("compile_experience: LLM call failed", exc_info=True)
        return None

    # 5. Парсим ответ
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            import re

            cleaned = re.sub(r"^```[a-z]*\s*|\s*```$", "", cleaned).strip()
        result = json.loads(cleaned)
        experience = result.get("experience")
        if not experience or not isinstance(experience, str):
            return None
        experience = experience.strip()[:MAX_EXPERIENCE_LENGTH]
        if len(experience) < 10:
            return None
    except Exception:
        logger.debug("compile_experience: failed to parse LLM response", exc_info=True)
        return None

    # 6. Safety check
    if not _safety_check_experience(experience):
        logger.warning("compile_experience: REJECTED by safety gate")
        return None

    return experience


# ── Save experience ───────────────────────────────────────────────────


async def _save_experience(telegram_id: int, experience: str) -> bool:
    """Сохраняет experience в custom_instructions (как JSON).

    Сохраняет tone_mix если был, обновляет experience.
    """
    from src.core.context_cache import invalidate as cache_invalidate

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

        # Текущий custom_instructions
        current_ci: dict = {}
        try:
            if p.custom_instructions:
                ci_data = (
                    json.loads(p.custom_instructions)
                    if isinstance(p.custom_instructions, str)
                    else p.custom_instructions
                )
                if isinstance(ci_data, dict):
                    current_ci = ci_data
        except (json.JSONDecodeError, TypeError):
            pass

        # Обновляем experience
        current_ci["experience"] = experience
        current_ci["updated_at"] = (
            __import__("datetime").datetime.now().__format__("%Y-%m-%dT%H:%M:%S")
        )

        p.custom_instructions = json.dumps(current_ci, ensure_ascii=False)
        p.total_corrections = (p.total_corrections or 0) + 1
        await session.flush()

    # Инвалидируем кеш persona
    await cache_invalidate(f"persona:{telegram_id}")
    await cache_invalidate(f"evolve:ts:{telegram_id}")

    logger.info("Experience saved for user %d: %.100s", telegram_id, experience)
    return True


async def _set_evolve_cooldown(telegram_id: int) -> None:
    """Устанавливает cooldown на эволюцию."""
    from src.core.context_cache import put as cache_put

    await cache_put(
        f"evolve:ts:{telegram_id}",
        time.monotonic(),
        ttl=MIN_EVOLVE_INTERVAL_SEC + 60,
    )
    # Также увеличиваем счётчик эволюций за сегодня
    today_key = f"evolve:today:{telegram_id}:{__import__('datetime').datetime.now().strftime('%Y%m%d')}"
    from src.core.context_cache import get as cache_get

    count = await cache_get(today_key) or 0
    await cache_put(today_key, count + 1, ttl=86400)


# ── Detect feedback in user text ──────────────────────────────────────

# Триггеры что пользователь говорит о характере бота
_FEEDBACK_TRIGGERS = (
    "ты стал",
    "ты слишком",
    "будь как раньше",
    "перестань",
    "мне нравится когда ты",
    "мне не нравится когда ты",
    "ты изменился",
    "раньше ты",
    "ты ведёшь себя",
    "твой характер",
    "твоя личность",
    "ты какой-то",
    "ты раздражаешь",
    "ты бесишь",
    "ты прикольный",
    "ты классный",
    "ты стал лучше",
    "ты стал хуже",
)


def _detect_feedback_in_text(text: str) -> bool:
    """Определяет, говорит ли пользователь о характере бота."""
    if not text:
        return False
    text_lower = text.lower()
    for trigger in _FEEDBACK_TRIGGERS:
        if trigger in text_lower:
            return True
    return False


# ── Evolve from feedback ──────────────────────────────────────────────


async def evolve_from_feedback(
    telegram_id: int,
    feedback: str,
    provider,
) -> bool:
    """Пользователь сказал что-то о характере бота -> LLM анализирует и корректирует.

    Returns:
        True если опыт обновлён, False если нет.
    """
    if not _detect_feedback_in_text(feedback):
        return False

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)

        current_style_parts = []
        if p.base_tone and p.base_tone != "default":
            current_style_parts.append(f"base_tone: {p.base_tone}")
        current_style = "; ".join(current_style_parts) or "default"

        current_experience = "нет"
        try:
            if p.custom_instructions:
                ci = (
                    json.loads(p.custom_instructions)
                    if isinstance(p.custom_instructions, str)
                    else p.custom_instructions
                )
                if isinstance(ci, dict):
                    current_experience = ci.get("experience") or "нет"
        except (json.JSONDecodeError, TypeError):
            pass

    prompt = FEEDBACK_PROMPT.format(
        feedback=feedback[:500].replace("{", "{{").replace("}", "}}"),
        current_style=current_style,
        current_experience=current_experience,
    )

    try:
        import asyncio

        raw = await asyncio.wait_for(
            provider.chat(
                [ChatMessage(role="user", content=prompt)],
                heavy=False,
            ),
            timeout=20.0,
        )
    except Exception:
        logger.debug("evolve_from_feedback: LLM call failed", exc_info=True)
        return False

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            import re

            cleaned = re.sub(r"^```[a-z]*\s*|\s*```$", "", cleaned).strip()
        result = json.loads(cleaned)
        action = result.get("action")
        if not action or action == "null" or action is None:
            return False

        value = result.get("value", "")

        if action == "update_experience" and value:
            if not _safety_check_experience(value):
                logger.warning("evolve_from_feedback: REJECTED by safety gate")
                return False
            await _save_experience(telegram_id, value[:MAX_EXPERIENCE_LENGTH])
            return True

        if (
            action == "update_tone"
            and value
            and value
            in __import__(
                "src.core.intelligence.adaptive_persona",
                fromlist=["BASE_TONE_PROMPTS"],
            ).BASE_TONE_PROMPTS
        ):
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                p = await get_persona(session, owner)
                p.base_tone = value
                await session.flush()
                from src.core.context_cache import invalidate

                await invalidate(f"persona:{telegram_id}")
                await invalidate(f"evolve:ts:{telegram_id}")
            logger.info("Tone updated via feedback: %s", value)
            return True

        if action == "reset":
            # Сбросить опыт
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                p = await get_persona(session, owner)
                try:
                    if p.custom_instructions:
                        ci = (
                            json.loads(p.custom_instructions)
                            if isinstance(p.custom_instructions, str)
                            else p.custom_instructions
                        )
                        if isinstance(ci, dict):
                            ci.pop("experience", None)
                            p.custom_instructions = json.dumps(ci, ensure_ascii=False)
                            await session.flush()
                except (json.JSONDecodeError, TypeError):
                    pass
                from src.core.context_cache import invalidate

                await invalidate(f"persona:{telegram_id}")
            return True

    except Exception:
        logger.debug("evolve_from_feedback: parse failed", exc_info=True)

    return False


# ── Main entry point ──────────────────────────────────────────────────


async def maybe_evolve_after_turn(
    telegram_id: int,
    user_text: str,
    bot_response: str | None,
    provider,
) -> bool:
    """Вызывается ПОСЛЕ каждого ответа бота.

    Сама решает: пора эволюционировать или нет.
    Никогда не блокирует основной поток (все операции fire-and-forget
    или быстрые проверки).

    Returns:
        True если эволюция произошла, False если нет.
    """
    if not provider:
        return False

    # Phase: update context files from dialog (lightweight, no LLM)
    try:
        from src.core.memory.context_files import try_extract_context_updates

        updated = await try_extract_context_updates(
            session=None,
            user_text=user_text,
            assistant_text=bot_response or "",
            owner_id=telegram_id,
        )
        if updated:
            logger.debug("Updated %d context files from dialog", updated)
    except Exception:
        logger.debug("Context file update skipped (non-critical)", exc_info=True)

    # 1. Быстрый фидбек: пользователь сказал что-то про характер?
    if _detect_feedback_in_text(user_text):
        evolved = await evolve_from_feedback(telegram_id, user_text, provider)
        if evolved:
            logger.info("Evolved from feedback: user=%d", telegram_id)
            return True

    # 2. Плановая эволюция: проверяем количество диалогов
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        p = await get_persona(session, owner)
        total = p.total_interactions or 0

    # Эволюционируем когда total_interactions кратен EVOLVE_EVERY_N_TURNS
    # и total_interactions > 0
    if total > 0 and total % EVOLVE_EVERY_N_TURNS == 0:
        # Проверяем дневной лимит
        today_key = f"evolve:today:{telegram_id}:{__import__('datetime').datetime.now().strftime('%Y%m%d')}"
        from src.core.context_cache import get as cache_get

        daily_count = await cache_get(today_key) or 0
        if daily_count >= MAX_EVOLVES_PER_DAY:
            logger.debug("Evolve: daily limit reached (%d)", MAX_EVOLVES_PER_DAY)
            return False

        experience = await compile_experience(telegram_id, provider)
        if experience:
            await _save_experience(telegram_id, experience)
            await _set_evolve_cooldown(telegram_id)
            logger.info("Auto-evolved from %d interactions: %.80s", total, experience)
            return True

    return False
