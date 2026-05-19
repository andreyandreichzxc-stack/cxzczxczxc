"""Авто-построение SelfProfile из персональных фактов памяти."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

SELF_PROFILE_SYSTEM = """Ты анализируешь факты памяти владельца Telegram-аккаунта. На основе фактов составь профиль в JSON.

Верни ТОЛЬКО JSON:
{"preferences": ["список предпочтений"], "goals": ["список целей"], "current_projects": ["проекты"], "decision_style": "быстрый|аналитический|советуется", "communication_preferences": ["список"], "sleep_pattern": "сова|жаворонок|00:00-08:00", "work_hours": "09:00-18:00"}
Если не знаешь — null."""


async def build_self_profile(
    telegram_id: int, provider: "LLMProvider"
) -> object | None:
    """Строит SelfProfile из персональных фактов через LLM."""
    from src.db.repo import (
        get_or_create_user,
        list_memories,
        upsert_self_profile,
    )
    from src.db.session import get_session
    from src.llm.base import ChatMessage

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        memories = await list_memories(session, owner)
        personal = [
            m
            for m in memories
            if m.is_active and (m.memory_type == "personal" or m.contact_id is None)
        ]
        if len(personal) < 5:
            logger.info("Not enough personal facts for self-profile: %d", len(personal))
            return None

        facts_text = "\n".join(f"- {m.fact}" for m in personal[:50])
        raw = await provider.chat(
            [
                ChatMessage(role="system", content=SELF_PROFILE_SYSTEM),
                ChatMessage(role="user", content=f"Факты о владельце:\n{facts_text}"),
            ],
            heavy=False,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw[3:-3].strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group(0))
            return await upsert_self_profile(session, owner, **data)
    return None
