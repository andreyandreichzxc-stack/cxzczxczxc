from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.db.repo import (
    get_or_create_user,
    list_skills,
    set_skill_enabled,
    get_skill_by_name,
    upsert_skill,
)
from src.db.session import get_session
from src.core.intelligence.skill_editor import bump_version
from src.core.context_cache import invalidate as cache_invalidate

router = Router(name="skills_cmd")
router.message.filter(OwnerOnly())

logger = logging.getLogger(__name__)


@router.message(Command("skills"))
async def cmd_skills(message: Message, command: CommandObject) -> None:
    args = (command.args or "").strip()
    parts = args.split(maxsplit=1)

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        # Feature 3: /skills yaml add <name> ---\nkey: value\n---
        if parts and parts[0] == "yaml" and len(parts) > 1:
            yaml_args = parts[1].split(maxsplit=1)
            if len(yaml_args) >= 2 and yaml_args[0] == "add":
                # name — первый токен после "add", всё остальное — description с YAML
                name_and_yaml = yaml_args[1].split(maxsplit=1)
                skill_name = name_and_yaml[0].strip()
                yaml_description = (
                    name_and_yaml[1].strip() if len(name_and_yaml) > 1 else ""
                )

                if not yaml_description:
                    await message.answer(
                        "⚠️ Использование: /skills yaml add &lt;name&gt; ---\\n"
                        "tags: [tag1, tag2]\\ncategory: search\\n---\\n"
                        "Описание навыка..."
                    )
                    return

                # Парсим YAML frontmatter для валидации
                try:
                    from src.core.intelligence.skill_yaml import (
                        extract_frontmatter_metadata,
                    )

                    yaml_meta, clean_desc = extract_frontmatter_metadata(
                        yaml_description
                    )
                except Exception as e:
                    await message.answer(f"⚠️ Ошибка парсинга YAML: {e}")
                    return

                if not yaml_meta:
                    await message.answer(
                        "⚠️ Не найден YAML frontmatter (---...---).\n"
                        "Добавьте в начале описания:\n"
                        "<code>---\ntags: [tag1]\ncategory: mycat\n---</code>"
                    )
                    return

                try:
                    skill = await upsert_skill(
                        session,
                        owner,
                        name=skill_name[:128],
                        description=yaml_description,
                        trigger_patterns_json=None,
                        body=clean_desc or yaml_description,
                        enabled=False,
                        review_status="proposed",
                    )
                    meta_str = ", ".join(f"{k}={v}" for k, v in yaml_meta.items())
                    await message.answer(
                        f"✅ Skill <b>{skill.name}</b> создан с YAML метаданными.\n"
                        f"Метаданные: {meta_str}\n"
                        f"Статус: proposed (включите через /skills enable {skill.name})"
                    )
                except Exception as e:
                    await message.answer(f"⚠️ Ошибка создания навыка: {e}")
                return

        if parts and parts[0] == "show" and len(parts) > 1:
            skill = await get_skill_by_name(session, owner, parts[1])
            if not skill:
                await message.answer("Skill не найден.")
                return
            # Показываем YAML метаданные если есть
            yaml_info = ""
            import json

            patterns = skill.trigger_patterns_json or []
            for p in patterns:
                if isinstance(p, dict) and "__yaml__" in p:
                    yaml_info = (
                        "\n\n<b>YAML метаданные:</b>\n<pre>"
                        + json.dumps(p["__yaml__"], ensure_ascii=False, indent=2)
                        + "</pre>"
                    )
                    break
            await message.answer(
                f"<b>{skill.name}</b> v{skill.version or '1.0.0'}\n"
                f"enabled={skill.enabled} · status={skill.review_status}\n"
                f"validation: {'%.0f%%' % (skill.validation_score * 100) if skill.validation_score is not None else '—'}\n"
                f"edits: {len(skill.edit_history_json or [])} · "
                f"rejected: {len(skill.rejected_edits_json or [])}\n\n"
                f"{skill.description or ''}"
                f"{yaml_info}\n\n"
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

        if parts and parts[0] == "rollback" and len(parts) > 1:
            try:
                skill = await get_skill_by_name(session, owner, parts[1])
                if not skill:
                    await message.answer("Skill не найден.")
                    return
                if skill.best_body is None:
                    await message.answer(
                        "Нет сохранённой стабильной версии для отката."
                    )
                    return

                from datetime import datetime, timezone

                old_version = skill.version or "1.0.0"
                skill.body = skill.best_body
                skill.validation_score = None
                new_version = bump_version(old_version, "minor")
                skill.version = new_version

                history = list(skill.edit_history_json or [])
                history.append(
                    {
                        "op": "rollback",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "reason": "Manual rollback to best_body",
                    }
                )
                skill.edit_history_json = history

                await session.flush()
                await cache_invalidate(f"skills:{owner.telegram_id}:")
                await message.answer(
                    f"✅ Skill <b>{skill.name}</b> откачен к стабильной версии v{new_version}."
                )
            except Exception as e:
                logger.exception("Rollback failed for skill %r", parts[1])
                await message.answer(f"⚠️ Ошибка отката: {e}")
            return

        skills = await list_skills(session, owner, limit=30)

    if not skills:
        await message.answer("Skills пока пусты. Они появятся после /evolve.")
        return

    lines = ["<b>Skills</b>"]
    for skill in skills:
        state = "on" if skill.enabled else "off"
        ver = skill.version or "1.0.0"
        score = (
            "%.0f%%" % (skill.validation_score * 100)
            if skill.validation_score is not None
            else "—"
        )
        edits = len(skill.edit_history_json or [])
        lines.append(
            f"• <b>{skill.name}</b> v{ver} · {state} · {skill.review_status} · "
            f"ok:{skill.success_count} err:{skill.failure_count} · "
            f"score:{score} edits:{edits}"
        )
    lines.append(
        "\n<code>/skills show name</code> · "
        "<code>/skills enable name</code> · "
        "<code>/skills disable name</code>\n"
        "<code>/skills yaml add name ---\\nkey: value\\n---</code>"
    )
    await message.answer("\n".join(lines))
