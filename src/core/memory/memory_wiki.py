"""Memory Wiki Generator — human-readable Markdown from bot memory."""

from __future__ import annotations
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.db.session import get_session
from src.db.repo import get_or_create_user

logger = logging.getLogger(__name__)

WIKI_DIR: Path = settings.data_dir / "memory-wiki"


async def generate_memory_wiki(owner_telegram_id: int) -> dict[str, int]:
    """Generate Markdown wiki pages from memory. Returns {category: count}."""
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)

        # 1. Load all active memories
        from sqlalchemy import select
        from src.db.models._memory import Memory

        result = await session.execute(
            select(Memory)
            .where(
                Memory.user_id == owner.id,
                Memory.is_active.is_(True),
            )
            .order_by(Memory.created_at.desc())
        )
        memories = result.scalars().all()

        # 2. Group by type
        by_type: dict[str, list] = defaultdict(list)
        for m in memories:
            cat = m.memory_type or "general"
            by_type[cat].append(m)

        # 3. Generate pages
        for category, items in by_type.items():
            page = _render_category(category, items)
            path = WIKI_DIR / f"{_safe_filename(category)}.md"
            path.write_text(page, encoding="utf-8")
            stats[category] = len(items)

        # 4. Generate index
        index = _render_index(by_type, memories)
        (WIKI_DIR / "index.md").write_text(index, encoding="utf-8")

        # 5. Generate bootstrap MEMORY.md (always in context)
        bootstrap = _render_bootstrap(memories)
        (WIKI_DIR / "MEMORY.md").write_text(bootstrap, encoding="utf-8")

        # 6. Generate changelog
        changelog = _render_changelog(memories)
        (WIKI_DIR / "changelog.md").write_text(changelog, encoding="utf-8")

    logger.info(
        "Memory wiki generated: %d categories, %d total facts",
        len(stats),
        len(memories),
    )
    return stats


def _render_category(category: str, items: list) -> str:
    lines = [f"# {_category_label(category)}\n"]
    for item in items[:200]:  # cap at 200 per category
        ts = ""
        if item.created_at:
            ts = item.created_at.strftime("%Y-%m-%d")
        confidence = f" ({item.confidence:.0%})" if item.confidence else ""
        lines.append(f"## [{ts}] {item.fact[:200]}{confidence}")
        if item.tags:
            lines.append(f"*Теги: {item.tags}*")
        lines.append("")
    return "\n".join(lines)


def _render_index(by_type: dict, memories: list) -> str:
    lines = ["# 📚 Memory Wiki — Оглавление\n"]
    lines.append(
        f"*Сгенерировано: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
    )
    lines.append(f"*Всего фактов: {len(memories)}*\n")

    total_confidence = sum(m.confidence or 0 for m in memories) / max(len(memories), 1)
    lines.append(f"*Средняя уверенность: {total_confidence:.0%}*\n")

    for category, items in sorted(by_type.items()):
        filename = _safe_filename(category)
        lines.append(
            f"- [{_category_label(category)}]({filename}.md) — {len(items)} фактов"
        )

    lines.append("\n- [📋 Changelog](changelog.md)")
    lines.append("- [🧠 Bootstrap](MEMORY.md)")
    return "\n".join(lines)


def _render_bootstrap(memories: list) -> str:
    """Minimal bootstrap layer — always loaded into context."""
    lines = ["# MEMORY.md (bootstrap)\n"]
    lines.append("*Авто-генерируемый файл. Всегда в контексте агента.*\n")

    # Top-10 highest confidence facts
    top = sorted(memories, key=lambda m: m.confidence or 0, reverse=True)[:10]
    for m in top:
        lines.append(f"- {m.fact[:300]}")

    return "\n".join(lines)


def _render_changelog(memories: list) -> str:
    lines = ["# 📋 Changelog памяти\n"]
    lines.append("*Хронология изменений памяти*\n")

    # Group by date
    by_date: dict[str, list] = defaultdict(list)
    for m in sorted(memories, key=lambda x: x.created_at or datetime.min, reverse=True):
        if m.created_at:
            date_key = m.created_at.strftime("%Y-%m-%d")
            by_date[date_key].append(m)

    for date_key, items in list(by_date.items())[:90]:  # last 90 days
        lines.append(f"## {date_key} (+{len(items)} фактов)")
        for m in items[:5]:
            lines.append(f"- {m.fact[:150]}")
        if len(items) > 5:
            lines.append(f"  *...и ещё {len(items) - 5}*")
        lines.append("")

    return "\n".join(lines)


def _category_label(cat: str) -> str:
    return {
        "personal": "👤 Личное",
        "preference": "⭐ Предпочтения",
        "contact_fact": "👥 О контактах",
        "temporary": "⏳ Временное",
        "cached_knowledge": "📦 Кеш-знания",
        "general": "📝 Общее",
    }.get(cat, cat)


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")[:64]


async def wiki_changelog_recent(days: int = 7) -> str:
    """Return recent changes as text for notifications."""
    changelog_path = WIKI_DIR / "changelog.md"
    if not changelog_path.exists():
        return "Wiki не сгенерирована. Используй /wiki."
    return changelog_path.read_text(encoding="utf-8")[:2000]
