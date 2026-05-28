"""Свободный текст (и голос) → агент → действие. Регистрируется последним в bot/app.py,
чтобы команды и FSM перехватывали свои события раньше."""

import asyncio
import logging
import random
import sys
import time
from pathlib import Path

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.config import settings
from src.core.actions.trajectory import actions_from_intent
from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.task_manager import track_ff
from src.core.intelligence.agent import route_intent
from src.core.intelligence.smart_autorouter import make_plan
from src.core.memory import conversation_context as ctx_store
from src.core.infra.timeutil import now_in_tz
from src.core.infra.transcription import transcription_service
from src.db.repo import (
    get_api_key,
    get_or_create_user,
)
from src.db.session import get_session
from src.llm.base import TaskType
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager

from .free_text_common import (
    _fire_record_trajectory,
    _get_owner_context,
    _summarize_intent_for_memory,
)
from src.core.intelligence.character_evolution import maybe_evolve_after_turn

from .free_text_pipeline import (
    _dispatch,
    _save_intent_context,
    check_contact_rules,
    check_followup,
    check_instructions,
    check_persona,
    execute_fast_route,
    execute_instant,
    execute_maestro,
)
from src.core.humanizer import record_humanizer_feedback, _pop_last_humanized
from .rate_limiter import check_rate_limit
from src.core.security.prompt_injection_scanner import scan_content


logger = logging.getLogger(__name__)
router = Router(name="free_text")
router.message.filter(OwnerOnly())

# Singalong search cache: user_id → list of search result dicts
_singalong_search_cache: dict = {}


# Voice transcription queue (non-blocking background processing)
_voice_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.max_voice_queue_size)
_voice_worker_task: asyncio.Task | None = None

# Per-user active tasks for priority preemption
# Light tasks (instant, fast_route, send, draft) preempt heavy tasks (maestro, analysis)
_active_tasks: dict[int, asyncio.Task] = {}
_active_tasks_lock = asyncio.Lock()


_HEAVY_MODES = frozenset({"maestro", "analysis"})

_MODE_NOTICE = "Обрабатываю предыдущий запрос. Если это срочно — просто напиши заново, я прерву тяжёлую задачу."


_WAITING_MESSAGES = [
    "⏳ Дай подумать…",
    "🤔 Сейчас соображу…",
    "💭 Уже думаю…",
    "🔍 Смотрю в переписке…",
    "📝 Анализирую…",
    "⏳ Секунду…",
    "🤖 Обрабатываю…",
    "💡 Генерирую ответ…",
]


def _get_waiting_message() -> str:
    return random.choice(_WAITING_MESSAGES)


def start_voice_worker() -> asyncio.Task:
    """Запустить фонового worker'а для транскрипции голоса (если ещё не запущен).

    Вызывается при старте приложения (main.py).
    """
    global _voice_worker_task
    if _voice_worker_task is None or _voice_worker_task.done():
        _voice_worker_task = asyncio.create_task(
            _voice_worker(), name="voice-transcription-worker"
        )
    return _voice_worker_task


async def stop_voice_worker() -> None:
    """Остановить voice worker (graceful shutdown)."""
    global _voice_worker_task
    if _voice_worker_task and not _voice_worker_task.done():
        _voice_worker_task.cancel()
        try:
            await _voice_worker_task
        except asyncio.CancelledError:
            pass
        _voice_worker_task = None
        logger.info("Voice transcription worker stopped")


async def _voice_worker() -> None:
    """Фоновый обработчик очереди голосовой транскрипции.

    Бесконечный цикл: забирает задание из очереди, транскрибирует,
    чистит файл, отвечает пользователю и передаёт текст в _process_text.
    При крахе одной задачи не падает — логирует и идёт дальше.

    Если внутренний цикл неожиданно прерывается (неперехваченная ошибка),
    автоматически перезапускает worker после паузы в 5 секунд.
    """
    while True:
        try:
            while True:
                got_job = False
                try:
                    job = await _voice_queue.get()
                    got_job = True
                    (
                        voice_path,
                        message,
                        _state_str,  # string | None — FSMContext value, NOT the FSMContext object
                        userbot_manager,
                        file_unique_id,
                        mode,
                        api_provider,
                        openai_key,
                        gemini_key,
                        mistral_key,
                    ) = job

                    try:
                        text = await transcription_service.transcribe(
                            voice_path,
                            file_id=file_unique_id,
                            mode=mode,
                            openai_key=openai_key,
                            gemini_key=gemini_key,
                            mistral_key=mistral_key,
                            api_provider=api_provider,
                        )
                    except Exception:
                        logger.exception("voice transcription failed in worker")
                        try:
                            await message.answer("❌ Не удалось распознать голосовое.")
                        except Exception:
                            logger.exception("failed to send error message from worker")
                        finally:
                            _cleanup_voice_file(voice_path)
                        continue

                    text = (text or "").strip()
                    if not text:
                        try:
                            await message.answer(
                                "🎙 Не услышал текста в этом сообщении."
                            )
                        except Exception:
                            logger.exception(
                                "failed to send empty transcription message"
                            )
                        _cleanup_voice_file(voice_path)
                        continue

                    try:
                        await message.answer(sanitize_html(f"🎙 <i>Услышал:</i> {text}"))
                    except Exception:
                        logger.exception("failed to send transcription result")

                    try:
                        # State is stale in background worker — pass None.
                        # Any code needing FSMContext methods will log a warning and skip.
                        await _process_text(text, message, None, userbot_manager)
                    except Exception:
                        logger.exception("Failed to process transcribed text in worker")
                    finally:
                        _cleanup_voice_file(voice_path)

                except asyncio.CancelledError:
                    raise  # propagate to outer handler for clean shutdown
                except Exception:
                    logger.exception("Voice worker error")
                    try:
                        from src.core.infra.hooks import hooks

                        await hooks.emit(
                            "on_error",
                            error="Voice worker error",
                            context="free_text._voice_worker",
                        )
                    except Exception:
                        pass  # hooks are optional, never break core flow
                finally:
                    if got_job:
                        _voice_queue.task_done()

        except asyncio.CancelledError:
            break  # intentional shutdown
        except Exception:
            logger.critical("Voice worker crashed, restarting in 5s", exc_info=True)
            try:
                from src.core.infra.hooks import hooks

                await hooks.emit(
                    "on_error",
                    error="Voice worker crashed",
                    context="free_text._voice_worker",
                )
            except Exception:
                pass  # hooks are optional, never break core flow
            await asyncio.sleep(5.0)


def _cleanup_voice_file(voice_path: Path) -> None:
    """Безопасно удалить временный файл голосового сообщения."""
    try:
        voice_path.unlink(missing_ok=True)
    except Exception:
        logger.debug("cleanup voice file failed: %s", voice_path, exc_info=True)


async def _process_text_fallback(
    raw: str,
    provider,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
    tz_name: str,
    owner_telegram_id: int,
    history_block: str,
    plan,
    turn_started: float,
    now_local_str: str,
) -> None:
    """Stage 9: Fallback — route_intent → _dispatch (extracted for reuse from background tasks)."""
    try:
        intent = await route_intent(
            provider,
            raw,
            heavy=False,
            now_local=now_local_str,
            tz_name=tz_name,
            history_block=history_block,
            memory_context=getattr(plan, "memory_context", "") or None,
            user_id=owner_telegram_id,
        )
    except Exception as e:
        logger.exception("agent route_intent failed")
        err_msg = str(e)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="intent",
            success=False,
            error=err_msg[:4000],
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        if len(err_msg) > 300:
            err_msg = err_msg[:300] + "…"
        await message.answer(
            sanitize_html(
                f"❌ Ошибка при обработке запроса.\n\n"
                f"<code>{err_msg}</code>\n\n"
                "<i>Если ошибка повторяется — проверь ключ в /settings → 🔑 API-ключи "
                "и модель в /settings → 🤖 LLM.</i>"
            )
        )
        return

    if intent.get("intent") == "multi":
        actions = intent.get("actions") or []
        if not isinstance(actions, list) or not actions:
            await message.answer("Не понял, что сделать.")
            return
        for sub in actions:
            await _dispatch(sub, message, state, userbot_manager, tz_name=tz_name)
    elif "intents" in intent:
        for sub in intent["intents"]:
            await _dispatch(sub, message, state, userbot_manager, tz_name=tz_name)
    else:
        await _dispatch(intent, message, state, userbot_manager, tz_name=tz_name)

    _save_intent_context(owner_telegram_id, intent)

    _fire_record_trajectory(
        owner_telegram_id,
        request_text=raw,
        route_mode="intent",
        intent_json=intent,
        actions_json=actions_from_intent(intent),
        response_text=_summarize_intent_for_memory(intent),
        success=True,
        latency_ms=int((time.monotonic() - turn_started) * 1000),
    )

    summary = _summarize_intent_for_memory(intent)
    await ctx_store.add_turn(message.from_user.id, raw, summary)
    try:
        if plan and plan.tasks:
            await ctx_store.set_last_purpose(
                message.from_user.id, plan.tasks[0].purpose.value
            )
    except Exception:
        logger.exception("failed to set last purpose")


async def _process_text(
    raw: str,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    turn_started = time.monotonic()

    # Rate-limit: не чаще 1 запроса в 3 секунды на пользователя
    if not await check_rate_limit(message.from_user.id):
        await message.answer("⏳ Подожди пару секунд, обрабатываю предыдущий запрос…")
        return

    ctx = await _get_owner_context(message.from_user.id)
    tz_name = str(ctx["tz_name"])
    owner_telegram_id = int(ctx["owner_telegram_id"])  # type: ignore[arg-type]
    use_heavy = bool(ctx["use_heavy"])

    now_local_str = now_in_tz(tz_name).strftime("%Y-%m-%d %H:%M")
    history_block = await ctx_store.render_history_block(message.from_user.id)

    # ── Stage 0: Smart emoji/sticker replies ─────────────────────────
    from src.core.contacts.smart_reply import get_simple_reply

    emoji_reply = get_simple_reply(raw)
    if emoji_reply:
        await message.answer(emoji_reply)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="smart_reply",
            intent_json={"intent": "smart_reply"},
            response_text=emoji_reply,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        return

    # ── Stage 0b: Memory correction detection (Feature 2) ────────────
    from src.core.contacts.smart_reply import (
        detect_memory_correction,
        handle_memory_correction,
    )

    correction = detect_memory_correction(raw)
    if correction:
        response = await handle_memory_correction(correction, owner_telegram_id)

        # ── Humanizer feedback loop ───────────────────────────────
        # Если пользователь поправляет бота — последний humanized-ответ
        # был отвергнут. Записываем фидбек.
        last_humanized = _pop_last_humanized(owner_telegram_id)
        if last_humanized:
            record_humanizer_feedback(
                user_id=owner_telegram_id,
                original=last_humanized,
                corrected=raw,
                accepted=False,
            )
        # ── End feedback loop ─────────────────────────────────────

        await message.answer(response)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="memory_correction",
            intent_json={"intent": "memory_correction", "action": correction["action"]},
            response_text=response,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        return

    # ── Stage 0c: Contradiction detection ────────────────────────────
    from src.core.memory.contradiction_detector import (
        check_contradiction_response,
        detect_contradiction,
        store_pending_contradiction,
    )

    # Check if this message is a response to a pending contradiction question
    cr_response = await check_contradiction_response(owner_telegram_id, raw)
    if cr_response:
        await message.answer(cr_response)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="contradiction_response",
            intent_json={"intent": "contradiction_response"},
            response_text=cr_response,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        return

    # Check for new contradictions against stored facts
    contradiction = await detect_contradiction(owner_telegram_id, raw)
    if contradiction:
        await store_pending_contradiction(owner_telegram_id, contradiction)
        await message.answer(
            f"🤔 {contradiction['suggestion']}\n"
            f"(уверенность: {contradiction['confidence']:.0%})"
        )
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="contradiction",
            intent_json={"intent": "contradiction"},
            response_text=contradiction["suggestion"],
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        return

    # ── Stage 0d: Smart correction / cancellation detection ──────────
    from src.bot.handlers.smart_correction import (
        apply_correction,
        detect_correction,
    )

    correction = await detect_correction(owner_telegram_id, raw)
    if correction:
        reply = await apply_correction(owner_telegram_id, correction)
        await message.answer(reply)

        # ── Learn from correction (Feature: Learning from Corrections) ──
        try:
            from src.core.intelligence.correction_learner import learn_correction

            if correction["action"] == "cancel":
                await learn_correction(
                    owner_telegram_id,
                    original_text="[cancelled]",
                    corrected_text="",
                    feedback_type="cancel",
                )
            elif correction["action"] == "replace":
                new_text = correction.get("new_text", "")
                is_fact = any(
                    w in (new_text or "").lower() for w in ("факт", "помню", "знаю")
                )
                await learn_correction(
                    owner_telegram_id,
                    original_text=raw,
                    corrected_text=new_text,
                    feedback_type="fact" if is_fact else "style",
                )
        except Exception:
            pass  # never break core flow

        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="smart_correction",
            intent_json={"intent": "smart_correction", "action": correction["action"]},
            response_text=reply,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        return

    # ── Stage 0e: Singalong — подпевание строчками из песен ───────────
    # Flow: LLM определяет песню → спрашивает подтверждение → подпевает
    # Если отказано → ищет через DuckDuckGo → снова спрашивает
    # ⚠️ Ответ отправляется НАПРЯМУЮ через message.answer(), минуя humanizer.
    # Все LLM output проходит через sanitize_html() для защиты от HTML injection.
    from src.core.intelligence.singalong import (
        _is_confirmation,
        _looks_like_lyrics,
        identify_and_get_next_line,
        get_singalong_reply,
        consume_pending_singalong,
        peek_pending_singalong,
        store_pending_singalong,
    )

    # Импорт _search_lyrics для denial flow
    from src.core.intelligence.singalong import _search_lyrics

    async def _singalong_identify(
        text: str,
        telegram_id: int,
        use_heavy: bool,
        *,
        search_hint: str | None = None,
    ) -> dict | None:
        """Определить песню через LLM. Возвращает {song, artist, next_line} или None."""
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            provider = await build_provider(
                session, owner, purpose="main", task_type=TaskType.DEFAULT
            )
        if not provider:
            return None
        return await identify_and_get_next_line(
            text, provider, heavy=use_heavy, search_hint=search_hint
        )

    async def _singalong_get_reply(
        text: str,
        telegram_id: int,
        use_heavy: bool,
    ) -> str | None:
        """Получить следующую строчку через LLM."""
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            provider = await build_provider(
                session, owner, purpose="main", task_type=TaskType.DEFAULT
            )
        if not provider:
            return None
        return await get_singalong_reply(text, provider, heavy=use_heavy)

    async def _ask_singalong_confirmation(
        lyrics: str,
        identified: dict | None,
    ) -> None:
        """Общий хелпер: сохранить pending и отправить подтверждение."""
        if identified and identified.get("song"):
            title = identified["song"]
            artist = identified.get("artist", "")
            display = f"{title}" + (f" — {artist}" if artist else "")
            await store_pending_singalong(
                owner_telegram_id,
                lyrics,
                song_title=display,
                next_line=identified.get("next_line"),
            )
            await message.answer(sanitize_html(f"Это {display}? Подпевать? 🎵"))
        else:
            await store_pending_singalong(owner_telegram_id, lyrics)
            await message.answer("Подпевать тебе? 🎵")

    async def _try_singalong() -> bool:
        """Stage 0e: попробовать обработать как текст песни. Возвращает True если сообщение consumed."""
        try:
            pending = await peek_pending_singalong(owner_telegram_id)

            # ── Pending exists ──────────────────────────────────────
            if pending:
                # Новая песня при существующем pending — заменяем
                if _looks_like_lyrics(raw):
                    await consume_pending_singalong(owner_telegram_id)
                    _singalong_search_cache.pop(owner_telegram_id, None)
                    identified = await _singalong_identify(
                        raw, owner_telegram_id, use_heavy
                    )
                    await _ask_singalong_confirmation(raw, identified)
                    _fire_record_trajectory(
                        owner_telegram_id,
                        request_text=raw,
                        route_mode="singalong_ask",
                        intent_json={"intent": "singalong", "phase": "ask"},
                        response_text="Подпевать тебе? 🎵",
                        success=True,
                        latency_ms=int((time.monotonic() - turn_started) * 1000),
                    )
                    return True

                # Проверить confirmation/denial
                decision = _is_confirmation(raw)

                # ── Подтверждение ───────────────────────────────────
                if decision is True:
                    data = await consume_pending_singalong(owner_telegram_id)
                    if not data:
                        return False  # pending истёк

                    if data.get("next_line"):
                        await message.answer(sanitize_html(data["next_line"]))
                        _fire_record_trajectory(
                            owner_telegram_id,
                            request_text=raw,
                            route_mode="singalong",
                            intent_json={"intent": "singalong"},
                            response_text=data["next_line"],
                            success=True,
                            latency_ms=int((time.monotonic() - turn_started) * 1000),
                        )
                        return True

                    # Нет next_line — вызываем LLM
                    reply = await _singalong_get_reply(
                        data["lyrics"], owner_telegram_id, use_heavy
                    )
                    if reply:
                        await message.answer(sanitize_html(reply))
                        _fire_record_trajectory(
                            owner_telegram_id,
                            request_text=raw,
                            route_mode="singalong",
                            intent_json={"intent": "singalong"},
                            response_text=reply,
                            success=True,
                            latency_ms=int((time.monotonic() - turn_started) * 1000),
                        )
                        return True
                    await message.answer("Не узнал эту песню \U0001f914")
                    return True

                # ── Отклонение ──────────────────────────────────────
                if decision is False:
                    data = await consume_pending_singalong(owner_telegram_id)
                    if not data:
                        return False  # pending истёк

                    # Ищем через DuckDuckGo (общий хелпер из singalong)
                    search_items = await _search_lyrics(data["lyrics"])

                    if search_items:
                        variants = []
                        for i, item in enumerate(search_items[:3], 1):
                            t = sanitize_html(item.get("title", "?"))
                            variants.append(f"{i}. {t}")
                        text = "Какая из этих?\n" + "\n".join(variants)
                        await message.answer(text)

                        # Сохраняем pending для numeric selection
                        await store_pending_singalong(
                            owner_telegram_id,
                            data["lyrics"],
                            song_title=search_items[0].get("title", ""),
                            next_line=None,
                        )
                        _singalong_search_cache[owner_telegram_id] = search_items[:3]
                        _fire_record_trajectory(
                            owner_telegram_id,
                            request_text=raw,
                            route_mode="singalong_search",
                            intent_json={"intent": "singalong", "phase": "search"},
                            response_text=text,
                            success=True,
                            latency_ms=int((time.monotonic() - turn_started) * 1000),
                        )
                        return True

                    await message.answer("Не нашёл такую песню в интернете \U0001f914")
                    _fire_record_trajectory(
                        owner_telegram_id,
                        request_text=raw,
                        route_mode="singalong_fail",
                        intent_json={"intent": "singalong", "result": "not_found"},
                        response_text="Не нашёл такую песню",
                        success=True,
                        latency_ms=int((time.monotonic() - turn_started) * 1000),
                    )
                    return True

                # ── Numeric selection (1/2/3) после поиска ──────────
                stripped_num = raw.strip()
                if (
                    stripped_num in ("1", "2", "3")
                    and owner_telegram_id in _singalong_search_cache
                ):
                    cache = _singalong_search_cache.pop(owner_telegram_id)
                    idx = int(stripped_num) - 1
                    if 0 <= idx < len(cache):
                        chosen = cache[idx]
                        title = chosen.get("title", "")
                        # Re-peek чтобы не использовать stale pending
                        current = await peek_pending_singalong(owner_telegram_id)
                        if not current:
                            return False
                        # Пытаемся определить next_line через LLM с контекстом
                        identified = await _singalong_identify(
                            current.get("lyrics", raw),
                            owner_telegram_id,
                            use_heavy,
                            search_hint=title,
                        )
                        if identified and identified.get("next_line"):
                            await message.answer(sanitize_html(identified["next_line"]))
                            await consume_pending_singalong(owner_telegram_id)
                            _fire_record_trajectory(
                                owner_telegram_id,
                                request_text=raw,
                                route_mode="singalong",
                                intent_json={"intent": "singalong", "selected": title},
                                response_text=identified["next_line"],
                                success=True,
                                latency_ms=int(
                                    (time.monotonic() - turn_started) * 1000
                                ),
                            )
                            return True
                        # LLM не смог — спрашиваем уточнение
                        await consume_pending_singalong(owner_telegram_id)
                        _singalong_search_cache.pop(owner_telegram_id, None)
                        await message.answer(
                            f"Не могу найти текст «{sanitize_html(title)}». Напиши название песни?"
                        )
                        return True
                    # Неверный номер — очищаем кеш
                    _singalong_search_cache.pop(owner_telegram_id, None)

                # decision is None + не numeric — другое сообщение, идём дальше
                return False

            # ── Нет pending — новое сообщение ───────────────────────
            # Clean stale search cache from previous sessions
            _singalong_search_cache.pop(owner_telegram_id, None)

            if _looks_like_lyrics(raw):
                identified = await _singalong_identify(
                    raw, owner_telegram_id, use_heavy
                )
                await _ask_singalong_confirmation(raw, identified)
                _fire_record_trajectory(
                    owner_telegram_id,
                    request_text=raw,
                    route_mode="singalong_ask",
                    intent_json={"intent": "singalong", "phase": "ask"},
                    response_text="Подпевать тебе? 🎵",
                    success=True,
                    latency_ms=int((time.monotonic() - turn_started) * 1000),
                )
                return True

            return False

        except Exception:
            logger.warning("singalong check failed", exc_info=True)
            return False

    if await _try_singalong():
        return

    # Stage 1: Adaptive instructions
    if await check_instructions(raw, owner_telegram_id, message):
        return

    # Stage 1b: Contact-specific rules (e.g. "с Олей будь вежливее")
    if await check_contact_rules(raw, owner_telegram_id, message, userbot_manager):
        return

    # Stage 2: Adaptive persona
    if await check_persona(raw, owner_telegram_id, message):
        return

    # Stage 3: Follow-up context
    if await check_followup(
        raw, owner_telegram_id, message, state, userbot_manager, tz_name, turn_started
    ):
        return

    # Stage 4: Smart AutoRouter
    _last_purpose = None
    try:
        _last_purpose = await ctx_store.get_last_purpose(message.from_user.id)
    except Exception:
        logger.exception("failed to get last purpose")
    plan = await make_plan(
        raw,
        owner_telegram_id,
        heavy_available=use_heavy,
        last_purpose=_last_purpose,
    )
    if plan is None:
        return
    if plan.tasks:
        t0 = plan.tasks[0]
        logger.debug(
            "AutoRouter plan: risk=%s purpose=%s heavy=%s cache_ttl=%d agents=%s",
            t0.risk.value,
            t0.purpose.value,
            t0.heavy,
            t0.cache_ttl,
            t0.need_agents or "—",
        )

    # Stage 5: INSTANT mode
    if plan.response_mode == "instant" and plan.final_response:
        await execute_instant(
            plan, message, raw, owner_telegram_id, turn_started, tz_name=tz_name
        )
        return

    # Stage 6: Build provider
    purpose = plan.tasks[0].purpose.value if plan.tasks else "main"
    async with get_session() as session:
        owner_db = await get_or_create_user(session, owner_telegram_id)
        provider = await build_provider(
            session, owner_db, purpose=purpose, task_type=TaskType.DEFAULT
        )
        if provider is None and purpose != "main":
            logger.debug("No key for purpose '%s', falling back to main", purpose)
            provider = await build_provider(
                session, owner_db, purpose="main", task_type=TaskType.DEFAULT
            )

    if provider is None:
        await message.answer(
            "Чтобы я мог понимать свободный текст — добавь LLM-ключ в /settings → 🔑 API-ключи."
        )
        return

    # Stage 7: FAST_ROUTE
    if plan.response_mode == "fast_route":
        await execute_fast_route(
            raw,
            plan,
            provider,
            message,
            state,
            userbot_manager,
            tz_name,
            owner_telegram_id,
            history_block,
            turn_started,
            now_local_str,
        )
        # Character evolution: fire-and-forget (никогда не блокирует)
        track_ff(
            asyncio.create_task(
                maybe_evolve_after_turn(owner_telegram_id, raw, None, provider)
            )
        )
        return

    # Stage 8: MAESTRO — heavy tasks run as background tasks for preemption
    if plan.response_mode == "maestro":
        injected_style: str | None = ctx.get("global_style_profile") or None  # type: ignore[assignment]

        async def _run_maestro_background():
            _my_task = asyncio.current_task()
            try:
                ok = await execute_maestro(
                    raw,
                    plan,
                    provider,
                    message,
                    state,
                    userbot_manager,
                    tz_name,
                    owner_telegram_id,
                    history_block,
                    turn_started,
                    injected_style,
                )
                # Character evolution: fire-and-forget после ответа
                track_ff(
                    asyncio.create_task(
                        maybe_evolve_after_turn(owner_telegram_id, raw, None, provider)
                    )
                )
                if not ok:
                    await _process_text_fallback(
                        raw,
                        provider,
                        message,
                        state,
                        userbot_manager,
                        tz_name,
                        owner_telegram_id,
                        history_block,
                        plan,
                        turn_started,
                        now_local_str,
                    )
            except asyncio.CancelledError:
                logger.debug("Maestro task cancelled for user %s", owner_telegram_id)
            except Exception as e:
                logger.exception(
                    "Maestro background task failed for user %s", owner_telegram_id
                )
                err_msg = str(e)
                if len(err_msg) > 300:
                    err_msg = err_msg[:300] + "…"
                await message.answer(
                    sanitize_html(
                        f"❌ Ошибка при обработке запроса.\n\n"
                        f"<code>{err_msg}</code>\n\n"
                        "<i>Если ошибка повторяется — проверь ключ в /settings → 🔑 API-ключи "
                        "и модель в /settings → 🤖 LLM.</i>"
                    )
                )
            finally:
                async with _active_tasks_lock:
                    if _active_tasks.get(owner_telegram_id) is _my_task:
                        _active_tasks.pop(owner_telegram_id, None)

        task = asyncio.create_task(_run_maestro_background())
        async with _active_tasks_lock:
            _active_tasks[owner_telegram_id] = task
        await message.answer(_get_waiting_message())
        return

    # Stage 9: Fallback — route_intent → _dispatch
    await _process_text_fallback(
        raw,
        provider,
        message,
        state,
        userbot_manager,
        tz_name,
        owner_telegram_id,
        history_block,
        plan,
        turn_started,
        now_local_str,
    )
    # Character evolution: fire-and-forget
    track_ff(
        asyncio.create_task(
            maybe_evolve_after_turn(owner_telegram_id, raw, None, provider)
        )
    )


@router.message(F.text & ~F.text.startswith("/"))
async def free_text(
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    if await state.get_state() is not None:
        return
    raw = (message.text or "").strip()
    if not raw:
        return

    uid = message.from_user.id

    # Scan for prompt injection in user input
    scan_result = scan_content(raw, f"user:{uid}")
    if scan_result.blocked:
        logger.warning(
            "Prompt injection blocked from user %d: %s (%s)",
            uid,
            scan_result.category,
            scan_result.match,
        )
        await message.answer(
            "⚠️ Сообщение содержит потенциально опасные конструкции и было заблокировано.\n"
            "Если это ошибка — переформулируйте запрос."
        )
        return

    # 🎭 Onboarding: первый контакт → предложить настроить личность
    from src.db.repo import get_persona

    is_new = False
    async with get_session() as session:
        owner = await get_or_create_user(session, uid)

        # Atomically: UPDATE ... SET total_interactions=1 WHERE total_interactions=0
        # If rowcount==1, this is a new user (no race condition possible)
        from src.db.models._learning import AdaptivePersona
        from sqlalchemy import update as sa_update

        result = await session.execute(
            sa_update(AdaptivePersona)
            .where(AdaptivePersona.user_id == owner.id)
            .where(AdaptivePersona.total_interactions == 0)
            .values(total_interactions=1)
        )
        is_new = result.rowcount > 0

    if is_new:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🎭 Настроить личность",
                        callback_data="set:sec:personality",
                    ),
                    InlineKeyboardButton(
                        text="⏭ Пропустить", callback_data="persona:skip_onboarding"
                    ),
                ]
            ]
        )
        await message.answer(
            "🎭 <b>Привет! Давай настроим мой характер под тебя?</b>\n\n"
            "Я умею общаться в разных стилях: профессионально, дружелюбно, "
            "игриво, лаконично и даже с сарказмом!\n\n"
            "Это займёт меньше минуты и улучшит наше общение. "
            "В любой момент можно изменить в /settings → 🎭 Личность.",
            reply_markup=kb,
        )
        return

    if len(raw) > 2000:
        raw = raw[:1997] + "...(truncated)"
    # Priority preemption: if a heavy task is running, cancel it for the new request
    uid = message.from_user.id
    async with _active_tasks_lock:
        existing = _active_tasks.get(uid)
        if existing and not existing.done():
            logger.info(
                "Preempting running task for user %s with new request: %s",
                uid,
                raw[:80],
            )
            existing.cancel()
            _active_tasks.pop(uid, None)
            should_send_preempt = True
        else:
            should_send_preempt = False

    if should_send_preempt:
        await message.answer("⏯ Прервал предыдущую задачу. Обрабатываю новый запрос…")

    try:
        await _process_text(raw, message, state, userbot_manager)
    except Exception:
        logger.exception("_process_text failed for user %s", uid)
        try:
            from src.core.infra.hooks import hooks

            await hooks.emit(
                "on_error",
                error=str(sys.exc_info()[1])
                if sys.exc_info()[1]
                else "_process_text failed",
                context="free_text.free_text",
            )
        except Exception:
            pass  # hooks are optional, never break core flow
        raise

    # ── Session recording (non-blocking, best-effort) ─────────────────
    try:
        from src.core.memory.session_recorder import record_turn

        async with get_session() as rec_session:
            await record_turn(rec_session, uid, "user", raw[:4000])
            await record_turn(rec_session, uid, "assistant", "(ответ отправлен)")
    except Exception:
        logger.warning(
            "Failed to record conversation turn for user %s", uid, exc_info=True
        )


@router.message(F.voice | F.audio)
async def free_voice(
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    if await state.get_state() is not None:
        return

    media = message.voice or message.audio
    if media is None:
        return

    # 1. Быстрая загрузка настроек пользователя из БД
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        mode = owner.settings.transcription_mode
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")
        api_provider = getattr(owner.settings, "transcription_api_provider", "openai")

    # 2. Скачивание .ogg файла (быстрая сетевая операция)
    media_dir = settings.data_dir / "media" / "control_bot"
    media_dir.mkdir(parents=True, exist_ok=True)
    target = media_dir / f"{message.message_id}_{media.file_unique_id}.ogg"

    try:
        await message.bot.download(media.file_id, destination=str(target))
    except Exception:
        logger.exception("voice download failed")
        await message.answer("❌ Не удалось скачать голосовое.")
        return

    # 3. Извлекаем текущее состояние FSM (как строку) до того, как хендлер завершится.
    #    Сам FSMContext в фоне станет stale, поэтому передаём только значение.
    current_state = await state.get_state()

    # Ставим в очередь фоновой обработки (транскрипция + process_text)
    # Таймаут 10с — если очередь переполнена, не блокируем event loop
    try:
        await asyncio.wait_for(
            _voice_queue.put(
                (
                    target,
                    message,
                    current_state,
                    userbot_manager,
                    media.file_unique_id,
                    mode,
                    api_provider,
                    openai_key,
                    gemini_key,
                    mistral_key,
                )
            ),
            timeout=settings.voice_queue_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Voice queue full for user %d, dropping voice message",
            message.from_user.id,
        )
        _cleanup_voice_file(target)
        await message.answer("⏳ Слишком много голосовых в очереди. Попробуй позже.")
        return

    # 4. Мгновенный ответ — пользователь не ждёт транскрипцию
    await message.answer("🎙 Принял, расшифровываю…")
