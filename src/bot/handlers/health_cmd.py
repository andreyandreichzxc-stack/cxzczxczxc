"""Command: /health — system diagnostics."""

from __future__ import annotations

from pathlib import Path

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly

router = Router()
router.message.filter(OwnerOnly())


@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    lines = ["🏥 <b>Состояние бота</b>\n"]
    from src.config import settings

    # 1. DB size
    db = settings.data_dir / "app.db"
    if db.exists():
        lines.append(f"🗄 БД: {db.stat().st_size / 1024 / 1024:.1f} MB")

    # 2. Qdrant
    qdrant = settings.data_dir / "qdrant"
    if qdrant.exists():
        total = sum(f.stat().st_size for f in qdrant.rglob("*") if f.is_file())
        lines.append(f"🔍 Qdrant: {total / 1024 / 1024:.1f} MB")

    # 3. Gates
    from src.core.infra.gating import gates

    status = gates.status
    passed_count = len(status["passed"])
    total_count = status["total"]
    lines.append(f"\n🔧 Зависимости: {passed_count}/{total_count}")
    for name in sorted(status["failed"]):
        lines.append(f"  ❌ {name}")

    # 4. Context files
    from src.core.memory.context_files import list_context_files

    ctx = list_context_files()
    lines.append(f"\n📝 Контекстных файлов: {len(ctx)}")

    # 5. Skills
    try:
        from src.core.intelligence.skill_docs import list_skill_docs

        docs = list_skill_docs()
        lines.append(f"📋 Документированных навыков: {len(docs)}")
    except Exception:
        pass

    await message.answer("\n".join(lines))
