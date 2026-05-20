from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.skills import suggest_skills_from_trajectories
from src.db.repo import get_or_create_user, list_skills, list_trajectories
from src.db.session import get_session

router = Router(name="trajectory_cmd")
router.message.filter(OwnerOnly())


@router.message(Command("trajectory"))
async def cmd_trajectory(message: Message, command: CommandObject) -> None:
    args = (command.args or "recent").strip()
    only_errors = args.startswith("errors")
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        rows = await list_trajectories(session, owner, only_errors=only_errors, limit=10)

    if not rows:
        await message.answer("Trajectory пока пустой.")
        return

    title = "Ошибки trajectory" if only_errors else "Последние trajectory"
    lines = [f"<b>{title}</b>"]
    for row in rows:
        status = "ok" if row.success else "err"
        req = (row.request_text or "").replace("\n", " ")[:80]
        tail = f" · {row.error[:80]}" if row.error else ""
        lines.append(f"• #{row.id} · {status} · {row.route_mode or '-'} · {row.latency_ms or 0}ms{tail}\n  <i>{req}</i>")
    await message.answer("\n".join(lines))


@router.message(Command("evolve"))
async def cmd_evolve(message: Message) -> None:
    created = await suggest_skills_from_trajectories(message.from_user.id)
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        pending = await list_skills(
            session, owner, enabled=False, review_status="pending", limit=20
        )

    if not pending and not created:
        await message.answer("Новых skill-кандидатов нет. Нужно больше успешных trajectory.")
        return

    lines = [f"<b>Self-evolution</b>: новых кандидатов {created}"]
    for skill in pending[:10]:
        lines.append(f"• <b>{skill.name}</b> — {skill.description or ''}")
    lines.append("\nОдобрить: <code>/skills enable name</code>")
    await message.answer("\n".join(lines))

