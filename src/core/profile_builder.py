"""Авто-построение ContactProfile из фактов памяти."""

import json
import logging
from datetime import datetime, timezone

from src.db.repo import (
    get_contact_profile,
    list_memories,
    upsert_contact_profile,
)
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

BUILD_PROFILE_PROMPT = """Ты — аналитик отношений. На основе фактов памяти о контакте собери JSON-профиль.
Поля профиля:
- closeness_label: str — метка близости ("друг", "работа", "семья", "клиент", "знакомый", "романтический интерес" или другое уместное)
- communication_style: str — описание стиля общения (1-2 предложения, напр. "мягко и коротко", "деловито")
- key_topics: list[str] — ключевые темы общения (3-7 штук)
- sensitivity: float — чувствительность к тону (0.0 = толстокожий, 1.0 = очень чувствительный)
- communication_dos: list[str] — что стоит делать (напр. "писать утром", "без голосовых")
- communication_donts: list[str] — чего избегать (напр. "не критиковать", "не слать ночью")
- current_status: str — статус отношений ("active", "tension", "resolved", "distant")
- relationship_phase: str — фаза ("warming", "cooling", "stable")
- open_questions: list[str] — открытые вопросы/темы, которые нужно обсудить

Верни ТОЛЬКО JSON без markdown-разметки, без пояснений."""


async def build_profile(
    owner_id: int,
    contact_id: int,
    provider: LLMProvider,
) -> dict:
    """Собирает профиль контакта из фактов памяти через LLM.

    1. Загружает все факты о контакте.
    2. Отправляет в LLM с промптом "собери профиль".
    3. Сохраняет результат через upsert_contact_profile.

    Возвращает dict с полями профиля (тот же JSON, что от LLM).
    """
    async with get_session() as session:
        from src.db.repo import get_or_create_user

        owner = await get_or_create_user(session, owner_id)
        if owner is None:
            logger.warning("build_profile: owner %s not found", owner_id)
            return {}

        memories = await list_memories(session, owner, contact_id=contact_id)
        if not memories:
            logger.info("build_profile: no memories for contact %s, skip", contact_id)
            return {}

        # Собираем факты в текст
        facts_text = "\n".join(
            f"- [{m.sentiment or 'neutral'}] {m.fact}" for m in memories if m.is_active
        )
        if not facts_text:
            return {}

        messages = [
            ChatMessage(role="system", content=BUILD_PROFILE_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Вот факты о контакте (contact_id={contact_id}):\n"
                    f"{facts_text}\n\n"
                    "Собери JSON-профиль."
                ),
            ),
        ]

        try:
            raw = await provider.chat(messages)
        except Exception:
            logger.exception(
                "build_profile: LLM call failed for contact %s", contact_id
            )
            return {}

        # Парсим JSON из ответа
        profile_data = _extract_json(raw)
        if not profile_data:
            logger.warning(
                "build_profile: failed to parse JSON from LLM for contact %s",
                contact_id,
            )
            return {}

        # Нормализуем поля — отдаём только те, что есть в модели
        allowed = {
            "closeness_label",
            "communication_style",
            "key_topics",
            "sensitivity",
            "communication_dos",
            "communication_donts",
            "current_status",
            "relationship_phase",
            "open_questions",
        }

        kwargs: dict = {}
        for key in allowed:
            val = profile_data.get(key)
            if val is not None:
                # Списки → JSON-строка для хранения в Text-колонках
                if isinstance(val, list):
                    kwargs[key] = json.dumps(val, ensure_ascii=False)
                else:
                    kwargs[key] = val

        # float-поля с дефолтами
        if "sensitivity" in kwargs:
            try:
                kwargs["sensitivity"] = float(kwargs["sensitivity"])
            except (ValueError, TypeError):
                kwargs["sensitivity"] = 0.5

        await upsert_contact_profile(session, owner, contact_id, **kwargs)
        await session.commit()
        logger.info("build_profile: saved profile for contact %s", contact_id)

        # Возвращаем оригинальные данные (списками, не строками)
        return profile_data


def _extract_json(raw: str) -> dict | None:
    """Извлекает JSON из ответа LLM (сбрасывает ```json … ``` и лишнее)."""
    text = raw.strip()
    # Убираем markdown-обёртку
    if text.startswith("```"):
        # Ищем первую и последнюю ```
        start = text.index("\n") + 1 if "\n" in text else len(text)
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
        else:
            text = text[3:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Попробуем найти { … } в тексте
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass
        return None
