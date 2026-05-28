"""Cold-start skill seeder: reads skills/*/SKILL.md and inserts them into DB."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.intelligence.skill_docs import list_skill_docs
from src.db.models import Skill

_DOC_SECTIONS = re.compile(r"\n## (?:Files|Tools|Output|Dependencies)\b")
_ACTIVATION_SECTION = re.compile(r"## Activation\s*\n(.*?)(?:\n##\s|\Z)", re.DOTALL)
_PURPOSE_LINE = re.compile(r"## Purpose\s*\n(.+?)(?:\n##\s|\n\n|\Z)", re.DOTALL)


def _parse_body(content: str) -> str:
    idx = _DOC_SECTIONS.search(content)
    if idx is not None:
        return content[: idx.start()].strip()
    return content.strip()


def _parse_description(content: str) -> str | None:
    m = _PURPOSE_LINE.search(content)
    if m:
        line = m.group(1).strip()
        return line[:512] if line else None
    return None


def _parse_trigger_patterns(content: str) -> list[str]:
    m = _ACTIVATION_SECTION.search(content)
    if not m:
        return []
    section = m.group(1)
    patterns: list[str] = []
    for line in section.split("\n"):
        stripped = line.strip().lower()
        if stripped.startswith(("- **automatic", "**automatic")):
            patterns.append("automatic")
        elif stripped.startswith(("- **manual", "**manual")):
            patterns.append("manual")
        elif stripped.startswith(("- **agent", "**agent")):
            patterns.append("agent")
    return list(dict.fromkeys(patterns))


async def seed_skills_from_docs(
    session: AsyncSession, user_id: int = 0, force: bool = False
) -> int:
    docs = list_skill_docs()
    created = 0

    for doc in docs:
        name: str = doc["name"]
        content: str = doc["content"]

        if not force:
            r = await session.execute(
                select(Skill.id).where(
                    Skill.user_id == user_id,
                    func.lower(Skill.name) == name.lower().strip(),
                )
            )
            if r.scalar_one_or_none() is not None:
                continue

        description = _parse_description(content)
        body = _parse_body(content)
        trigger_patterns = _parse_trigger_patterns(content)

        now = datetime.now(timezone.utc)
        skill = Skill(
            user_id=user_id,
            name=name.strip(),
            description=description,
            trigger_patterns_json=trigger_patterns if trigger_patterns else None,
            body=body,
            enabled=True,
            review_status="approved",
            version="1.0.0",
            created_at=now,
            updated_at=now,
        )
        session.add(skill)
        created += 1

    if created:
        await session.flush()
    return created
