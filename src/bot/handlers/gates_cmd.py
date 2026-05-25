"""Command: /gates — show dependency check status."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.infra.gating import gates

router = Router(name="gates_cmd")
router.message.filter(OwnerOnly())


@router.message(Command("gates"))
async def cmd_gates(message: Message) -> None:
    status = gates.status
    lines = ["🔍 **Состояние зависимостей**\n"]
    for name in sorted(status["passed"]):
        lines.append(f"✅ {name}")
    for name, reason in status["failed"].items():
        fallback = gates.get_fallback(name)
        if fallback:
            lines.append(f"⏭️ {name} → fallback: {fallback}")
        else:
            lines.append(f"❌ {name} (отключено)")
    lines.append(f"\nВсего: {status['total']}, пройдено: {len(status['passed'])}")
    await message.answer("\n".join(lines))
