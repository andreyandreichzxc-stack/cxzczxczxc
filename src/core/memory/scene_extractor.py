"""L2 Scene extraction — generates narrative scenes from clustered memory facts.

Groups facts by (contact_id, topic), calls LLM to produce a coherent
narrative scene description. Stores result in MemoryCluster.summary.

Pattern: TencentDB-Agent-Memory L2 scene blocks, simplified for TelegramHelper.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage, TaskType
from src.llm.router import build_provider
from src.db.session import get_session
from src.db.repo import get_or_create_user

logger = logging.getLogger(__name__)

SCENE_SYSTEM_PROMPT = """Ты — Memory Consolidation Architect. Твоя задача — превратить
разрозненные факты памяти в связный нарратив сцены.

Дано: список фактов, извлечённых из переписки с контактом.
Создай ОДНУ сцену — связный эпизод, описывающий что происходило.

Правила:
- Нарратив: 2-4 предложения, настоящее время, связный текст
- Опиши: контекст, ключевые события, итог/договорённости
- Сохрани важные детали: имена, даты, цифры
- Не выдумывай фактов, которых нет в списке
- Не используй маркдаун-форматирование в тексте сцены

Верни ТОЛЬКО валидный JSON (без markdown-обёрток):
{
  "scene_title": "краткое название (3-7 слов)",
  "narrative": "связный текст сцены (2-4 предложения)",
  "sentiment": "positive|negative|neutral|contradictory"
}"""


def _parse_scene_json(raw: str) -> dict[str, str] | None:
    """Parse LLM response, handling markdown code fences."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try first { ... } block
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse scene JSON: %.200s", raw)
    return None


async def generate_scene_narrative(
    telegram_id: int,
    contact_id: int | None,
    topic: str,
    facts: list[str],
) -> dict[str, Any] | None:
    """Generate a narrative scene from facts.

    Args:
        telegram_id: Owner's Telegram ID.
        contact_id: Contact this scene is about (None for self-facts).
        topic: Cluster topic (e.g. "работа", "проект X").
        facts: List of fact strings to synthesize into a scene.

    Returns:
        dict with keys: scene_title, narrative, sentiment, or None on failure.
    """
    if len(facts) < 3:
        logger.debug("Too few facts (%d) for scene generation", len(facts))
        return None

    # Build user prompt
    facts_text = "\n".join(f"- {f}" for f in facts[:20])
    contact_label = f"контактом {contact_id}" if contact_id else "собой (self-profile)"
    user_prompt = (
        f"Контакт: {contact_label}\n"
        f"Тема: {topic}\n\n"
        f"Факты ({len(facts)}):\n{facts_text}"
    )

    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            provider = await build_provider(session, owner, task_type=TaskType.MEMORY)
    except Exception:
        logger.exception("Failed to get provider for scene extraction")
        return None

    if provider is None:
        logger.warning("No LLM provider available for scene extraction")
        return None

    try:
        response = await provider.chat(
            [
                ChatMessage(role="system", content=SCENE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            task_type=TaskType.MEMORY,
        )
    except Exception:
        logger.exception("LLM call failed for scene extraction")
        return None

    if not response:
        return None

    parsed = _parse_scene_json(response)
    if parsed is None:
        return None

    # Validate required fields
    narrative = parsed.get("narrative", "").strip()
    if not narrative or len(narrative) < 20:
        logger.debug("Scene narrative too short: %d chars", len(narrative))
        return None

    return {
        "scene_title": parsed.get("scene_title", topic)[:64],
        "narrative": narrative[:600],
        "sentiment": parsed.get("sentiment", "neutral"),
    }


async def extract_scenes_for_user(telegram_id: int) -> int:
    """Extract L2 scenes for all clusters belonging to a user.

    Runs after cluster rebuild. For each cluster with >= 3 facts,
    generates a narrative scene and updates MemoryCluster.summary.

    Args:
        telegram_id: Owner's Telegram ID.

    Returns:
        Number of scenes generated.
    """
    from collections import defaultdict
    from sqlalchemy import select
    from src.db.models import Memory, MemoryCluster, MemoryClusterMember

    generated = 0

    try:
        async with get_session() as session:
            # Resolve telegram_id → DB user_id (MemoryCluster.user_id is FK to users.id)
            owner = await get_or_create_user(session, telegram_id)
            # Get all clusters for user
            result = await session.execute(
                select(MemoryCluster).where(
                    MemoryCluster.user_id == owner.id,
                    MemoryCluster.fact_count >= 3,
                )
            )
            clusters = result.scalars().all()

            if not clusters:
                return 0

            # Batch 1: load all cluster members in one query
            cluster_ids = [c.id for c in clusters]
            members_result = await session.execute(
                select(MemoryClusterMember).where(
                    MemoryClusterMember.cluster_id.in_(cluster_ids),
                )
            )
            all_members = members_result.scalars().all()

            # Group members by cluster_id and collect all memory_ids
            members_by_cluster: dict[int, list[MemoryClusterMember]] = defaultdict(list)
            memory_ids: set[int] = set()
            for mbr in all_members:
                members_by_cluster[mbr.cluster_id].append(mbr)
                memory_ids.add(mbr.memory_id)

            # Batch 2: load all memories in one query
            if memory_ids:
                memories_result = await session.execute(
                    select(Memory).where(Memory.id.in_(list(memory_ids)))
                )
                all_memories = memories_result.scalars().all()
                memories_by_id = {m.id: m for m in all_memories}
            else:
                memories_by_id = {}

            for cluster in clusters:
                members = members_by_cluster.get(cluster.id, [])

                fact_texts: list[str] = []
                contact_id: int | None = None
                for mbr in members:
                    mem = memories_by_id.get(mbr.memory_id)
                    if mem:
                        fact_texts.append(mem.fact)
                        if contact_id is None:
                            contact_id = mem.contact_id

                if len(fact_texts) < 3:
                    continue

                scene = await generate_scene_narrative(
                    telegram_id=telegram_id,
                    contact_id=contact_id,
                    topic=cluster.topic,
                    facts=fact_texts,
                )

                if scene:
                    cluster.summary = scene["narrative"]
                    cluster.fact_count = len(fact_texts)
                    generated += 1
                    logger.info(
                        "Scene generated: cluster=%d topic=%s title=%s",
                        cluster.id,
                        cluster.topic,
                        scene["scene_title"],
                    )

    except Exception:
        logger.exception("Scene extraction failed for user %d", telegram_id)

    return generated


__all__ = [
    "generate_scene_narrative",
    "extract_scenes_for_user",
    "SCENE_SYSTEM_PROMPT",
]
