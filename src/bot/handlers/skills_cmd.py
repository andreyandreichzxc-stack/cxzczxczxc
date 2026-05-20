from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.db.repo import (
    get_or_create_user,
    list_skills,
    set_skill_enabled,
    get_skill_by_name,
)
from src.db.session import get_session

router = Router(name="skills_cmd")
router.message.filter(OwnerOnly())


@router.message(Command("skills"))
async def cmd_skills(message: Message, command: CommandObject) -> None:
    args = (command.args or "").strip()
    parts = args.split(maxsplit=1)

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        if parts and parts[0] == "show" and len(parts) > 1:
            skill = await get_skill_by_name(session, owner, parts[1])
            if not skill:
                await message.answer("Skill не найден.")
                return
            await message.answer(
                f"<b>{skill.name}</b>\n"
                f"enabled={skill.enabled} · status={skill.review_status}\n\n"
                f"{skill.description or ''}\n\n"
                f"<pre>{skill.body[:3000]}</pre>"
            )
            return

        if parts and parts[0] in {"disable", "off"} and len(parts) > 1:
            skill = await set_skill_enabled(session, owner, parts[1], False)
            await message.answer("Skill отключен." if skill else "Skill не найден.")
            return

        if parts and parts[0] in {"enable", "on"} and len(parts) > 1:
            skill = await set_skill_enabled(
                session, owner, parts[1], True, review_status="approved"
            )
            await message.answer("Skill включен." if skill else "Skill не найден.")
            return

        skills = await list_skills(session, owner, limit=30)

    if not skills:
        await message.answer("Skills пока пусты. Они появятся после /evolve.")
        return

    lines = ["<b>Skills</b>"]
    for skill in skills:
        state = "on" if skill.enabled else "off"
        lines.append(
            f"• <b>{skill.name}</b> · {state} · {skill.review_status} · "
            f"ok:{skill.success_count} err:{skill.failure_count}"
        )
    lines.append("\n<code>/skills show name</code> · <code>/skills enable name</code> · <code>/skills disable name</code>")
    await message.answer("\n".join(lines))

