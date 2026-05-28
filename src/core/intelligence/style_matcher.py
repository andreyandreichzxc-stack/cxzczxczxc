"""Style Matcher — анализирует стиль сообщений пользователя и генерирует
инструкцию для LLM (style_match_block), которая вставляется в system‑prompt.

Не использует статичные шаблоны — все метрики считаются динамически
по реальным сообщениям пользователя из таблицы Message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

from sqlalchemy import select, desc

from src.db.models import Message
from src.db.repo import get_or_create_user, get_persona, update_persona
from src.db.session import get_session
from src.llm.base import ChatMessage, TaskType

logger = logging.getLogger(__name__)

# ── TTL‑кеш стилевого профиля (один час) ──────────────────────────
_style_cache: dict[int, tuple[float, str]] = {}  # owner_id → (monotonic, block_str)
_STYLE_CACHE_TTL: float = 3600.0
_style_lock = asyncio.Lock()

_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251]"
)


# ---------------------------------------------------------------------------
# analyze_user_style
# ---------------------------------------------------------------------------


async def analyze_user_style(owner_id: int) -> dict:
    """Анализирует последние исходящие сообщения пользователя и возвращает
    словарь с эвристиками и (опционально) LLM‑классификацией тона.

    Returns:
        Пустой ``dict``, если подходящих сообщений < 3.
        Иначе — словарь с ключами:
          - avg_len (int)
          - emoji_rate (float)
          - caps_rate (float)
          - tone (str)
          - directness (str)
          - water_tolerance (str)  — «низкая» / «средняя» / «высокая»
          - examples (list[str])   — до 3 примеров сообщений
          - updated_at (str)       — ISO‑8601 UTC
    """
    async with get_session() as session:
        result = await session.execute(
            select(Message.text)
            .where(
                Message.user_id == owner_id,
                Message.is_outgoing.is_(True),
                Message.text.isnot(None),
            )
            .order_by(desc(Message.date))
            .limit(50)
        )
        rows = result.all()
        texts = [row[0] for row in rows if row[0] and len(row[0].strip()) >= 2]

        if len(texts) < 3:
            logger.debug(
                "Style analysis skipped: only %d messages for owner %d",
                len(texts),
                owner_id,
            )
            return {}

        # -- эвристики -------------------------------------------------
        lengths = [len(t) for t in texts]
        avg_len = int(sum(lengths) / len(lengths))

        emoji_count = sum(1 for t in texts if _EMOJI_RE.search(t))
        emoji_rate = round(emoji_count / len(texts), 2)

        def _is_caps(s: str) -> bool:
            letters = [c for c in s if c.isalpha()]
            if not letters:
                return False
            return sum(1 for c in letters if c.isupper()) / len(letters) > 0.5

        caps_count = sum(1 for t in texts if _is_caps(t))
        caps_rate = round(caps_count / len(texts), 2)

        # water_tolerance
        if avg_len < 20:
            water_tolerance = "низкая"
        elif avg_len < 60:
            water_tolerance = "средняя"
        else:
            water_tolerance = "высокая"

        examples = texts[:3]

        # -- классификация тона ----------------------------------------
        tone = "нейтральный"
        directness = "средняя"

        user = await get_or_create_user(session, owner_id)

        try:
            from src.llm.router import build_provider

            provider = await build_provider(
                session, user, purpose="style", task_type=TaskType.CLASSIFY
            )
            if provider:
                sample = "\n---\n".join(texts[:20])
                prompt = (
                    "Проанализируй стиль сообщений пользователя. "
                    "Определи тональность и прямоту на основе этих сообщений. "
                    "Верни ТОЛЬКО JSON (без markdown‑обёртки): "
                    '{"tone": "<ироничный|деловой|дружеский|резкий|нейтральный>", '
                    '"directness": "<высокая|средняя|низкая>", '
                    '"wordiness": "<коротко|нейтрально|развёрнуто>"}\n\n'
                    f"Сообщения:\n{sample}"
                )
                raw = await provider.chat(
                    [ChatMessage(role="user", content=prompt)],
                    task_type=TaskType.CLASSIFY,
                )
                raw = raw.strip()
                # убираем markdown‑обёртку если есть
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    parsed = json.loads(m.group(0))
                    tone = parsed.get("tone", tone)
                    directness = parsed.get("directness", directness)
                logger.debug(
                    "LLM style classification for owner %d: tone=%s directness=%s",
                    owner_id,
                    tone,
                    directness,
                )
        except Exception:
            logger.debug(
                "LLM style classification failed for owner %d, using heuristic",
                owner_id,
                exc_info=True,
            )
            # эвристический тон (fallback)
            if avg_len < 20:
                tone = "коротко"
            elif avg_len < 60:
                tone = "нейтрально"
            else:
                tone = "развёрнуто"

        return {
            "avg_len": avg_len,
            "emoji_rate": emoji_rate,
            "caps_rate": caps_rate,
            "tone": tone,
            "directness": directness,
            "water_tolerance": water_tolerance,
            "examples": examples,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# format_style_block
# ---------------------------------------------------------------------------


async def format_style_block(style: dict) -> str:
    """Генерирует лаконичную инструкцию на русском для вставки в system‑prompt.

    Возвращает пустую строку, если ``style`` — пустой словарь.
    Не генерирует шаблон личности — только компактную директиву.
    """
    if not style:
        return ""

    tone = style.get("tone", "нейтральный")
    directness = style.get("directness", "средняя")
    water = style.get("water_tolerance", "средняя")
    avg_len = style.get("avg_len", 40)
    emoji_rate = style.get("emoji_rate", 0.0)
    caps_rate = style.get("caps_rate", 0.0)

    # ── Первая строка: наблюдение ──
    len_desc = (
        "коротко" if avg_len < 20 else ("развёрнуто" if avg_len > 60 else "умеренно")
    )
    emoji_desc = (
        "часто использует эмодзи"
        if emoji_rate > 0.5
        else ("иногда использует эмодзи" if emoji_rate > 0.1 else "почти без эмодзи")
    )
    caps_note = "часто пишет КАПСОМ" if caps_rate > 0.3 else ""
    caps_segment = f", {caps_note}" if caps_note else ""

    observation = (
        f"[СТИЛЬ ПОЛЬЗОВАТЕЛЯ] Анализ последних сообщений: "
        f"пользователь пишет {len_desc} (средняя длина {avg_len} символов), "
        f"{emoji_desc}{caps_segment}."
    )

    # ── Вторая строка: директива ──
    directives: list[str] = []

    # длина ответа
    if water == "низкая":
        directives.append("Отвечай 1–2 предложениями. Без воды.")
    elif water == "высокая":
        directives.append("Можно развёрнуто, детально.")
    else:
        directives.append("Отвечай умеренно, по делу.")

    # тон
    tone_map = {
        "ироничный": "Будь ироничен. Не используй шаблонные фразы вроде «Я здесь чтобы помочь».",
        "деловой": "Деловой тон. По существу, без лишних любезностей.",
        "дружеский": "Тёплый, дружеский тон. Можно на «ты», но без подхалимажа.",
        "резкий": "Прямой, резкий стиль. Не смягчай — пользователь ценит честность.",
        "нейтральный": "Нейтральный, спокойный тон.",
        "коротко": "Пиши коротко. Пользователь явно не любит длинные сообщения.",
        "нейтрально": "Нейтральный тон, по делу.",
        "развёрнуто": "Можно развёрнуто, пользователь ценит детали.",
    }
    tone_directive = tone_map.get(tone, tone_map["нейтральный"])
    directives.append(tone_directive)

    # прямота
    directness_map = {
        "высокая": "Будь прямолинеен. Без экивоков.",
        "низкая": "Выражайся мягко, дипломатично.",
        "средняя": "",
    }
    direct_d = directness_map.get(directness, "")
    if direct_d:
        directives.append(direct_d)

    # эмодзи
    if emoji_rate > 0.5:
        directives.append("Используй эмодзи активно, как пользователь.")
    elif emoji_rate > 0.1:
        directives.append("Эмодзи — умеренно.")
    else:
        directives.append("Эмодзи — минимально или не используй.")

    # caps
    if caps_rate > 0.3:
        directives.append("Можно использовать КАПС для эмфазы, как пользователь.")

    directive_block = "[ТВОЙ СТИЛЬ] " + " ".join(directives)

    return f"{observation}\n{directive_block}"


# ---------------------------------------------------------------------------
# get_or_update_style_profile
# ---------------------------------------------------------------------------


async def get_or_update_style_profile(owner_id: int) -> str | None:
    """Возвращает актуальный style‑блок для пользователя.

    Использует модульный TTL‑кеш + БД (AdaptivePersona.style_profile).
    При необходимости пересчитывает профиль заново.

    Returns:
        Строка‑инструкция для system‑prompt, или ``None`` если недостаточно данных.
    """
    async with _style_lock:
        now = time.monotonic()
        cached = _style_cache.get(owner_id)
        if cached is not None:
            ts, block = cached
            if now - ts < _STYLE_CACHE_TTL:
                return block if block else None
            # просрочен — удаляем
            del _style_cache[owner_id]

    async with get_session() as session:
        user = await get_or_create_user(session, owner_id)
        persona = await get_persona(session, user)

        # Проверяем свежесть БД‑профиля
        if persona.style_profile and persona.style_profile_updated_at:
            age_sec = (
                datetime.now(timezone.utc) - persona.style_profile_updated_at
            ).total_seconds()
            if age_sec < _STYLE_CACHE_TTL:
                # кешируем и возвращаем
                style_data = json.loads(persona.style_profile)
                block = await format_style_block(style_data)
                async with _style_lock:
                    _style_cache[owner_id] = (now, block)
                return block if block else None

        # Пересчитываем
        style_data = await analyze_user_style(owner_id)
        if not style_data:
            # нет данных — сохраняем None в кеш чтобы не дёргать БД
            async with _style_lock:
                _style_cache[owner_id] = (now, "")
            return None

        # Сохраняем в БД
        await update_persona(
            session,
            persona,
            style_profile=json.dumps(style_data, ensure_ascii=False),
            style_profile_updated_at=datetime.now(timezone.utc),
        )
        await session.commit()

        # Кешируем
        block = await format_style_block(style_data)
        async with _style_lock:
            _style_cache[owner_id] = (now, block)
        return block if block else None
